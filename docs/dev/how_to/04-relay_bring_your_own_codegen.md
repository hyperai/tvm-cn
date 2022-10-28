---
title: 向 TVM 中添加 Codegen
---

# 向 TVM 中添加 Codegen

随着深度学习工作负载所针对的硬件设备数量不断增加，用户在各种设备上实现高性能所需的知识也在不断增加。为了让数据科学家在开发新模型时不必担心性能问题，硬件厂商或是基于一些常见的深度学习算子，提供 MKLDNN 或 cuDNN 等库，或是提供 TensorRT 等框架，让用户按照某种方式描述模型，从而提高模型性能。

然而，用户在尝试使用新的库或设备时，必须学习新的编程接口。因此，一个统一的编程接口变得越来越重要：1）让所有用户及硬件厂商信息同步，2）提供一个可行的解决方案，让特定硬件或库只支持具有极高性能的、广泛使用的算子，不受支持的算子则回退到 CPU/GPU 等通用设备。

本开发手册演示了硬件厂商如何轻松实现自己的 Codegen，并将其注册为 Relay 后端编译器，从而支持自己的硬件设备/库。本手册涵盖了两种基于不同计算图的 codegen：

**1. 希望生成 C 代码。**

如果你的硬件已经具备了一个高度优化的 C/C++ 库，如对于 CPU 而言的 Intel CBLAS/MKL 库，或针对 GPU 而言的 NVIDIA CUBLAS 库，那么本节内容非常适合你。幸运的是，C 源代码模块与 TVM runtime 模块完全兼容，这意味着生成的代码可以由任何具有适当编译标志的 C/C++ 编译器编译，因此用户只需实现一个能为子图生成 C 代码的 codegen，并将 C 源代码模块集成到 TVM runtime 模块中。下一节内容讲详细演示如何为硬件实现 C codegen。

**2. 希望生成任意计算图。**

有时候，硬件可能需要其他形式的计算图如 JSON。这种情况下，用户不仅要实现一个 codegen，还要实现一个自定义 TVM runtime 模块，从而使得 TVM runtime 知道如何执行这个计算图。如果你的硬件已经拥有完整的计算图执行引擎 (graph execution engine)，如适用于 GPU 的 TensorRT，那么该解决方案对你而言非常具有参考价值。

完成 codegen 和 runtime 后，可以让客户借助你的自定义标签，对模型进行注释并加以利用。终端用户如何注释和启动特定 codegen 的教程，将在后续进行补充。

# 实现 C Codegen

在这一部分中，我们将演示如何借助预实现的算子函数，生成 C 代码的 codegen。简单起见，本示例 codegen 不依赖于第三方库。相反，我们在 C 中手动实现了两个宏：

``` c
#define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)         \
    extern "C" void p_ID_(float* a, float* b, float* out) { \
        for (int64_t i = 0; i < p_DIM1_; ++i) {             \
            out[i] = a[i] p_OP_ b[i];                       \
        }                                                   \
    }

#define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
    extern "C" void p_ID_(float* a, float* b, float* out) {   \
        for (int64_t i = 0; i < p_DIM1_; ++i) {               \
            for (int64_t j = 0; j < p_DIM2_; ++j) {           \
                int64_t k = i * p_DIM2_ + j;                  \
                out[k] = a[k] p_OP_ b[k];                     \
            }                                                 \
        }                                                     \
    }
```

使用这两个宏，可以为一维和二维张量生成二元算子 (binary operator)。例如，给定如下所示的子图，假设所有输入都是 shape 为 (10, 10) 的二维张量：

``` text
c_compiler_input0
       |
      add <-- c_compiler_input1
       |
    subtract <-- c_compiler_input2
       |
    multiply <-- c_compiler_input3
       |
      out
```

我们的目标是生成以下可编译代码来执行子图：

``` c++
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <cstdint>
#include <cstring>
#include <iostream>

#define GCC_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)           \
  extern "C" void p_ID_(float* a, float* b, float* out) { \
    for (int64_t i = 0; i < p_DIM1_; ++i) {               \
      out[i] = a[i] p_OP_ b[i];                           \
    }                                                     \
  }

#define GCC_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
  extern "C" void p_ID_(float* a, float* b, float* out) { \
    for (int64_t i = 0; i < p_DIM1_; ++i) {               \
      for (int64_t j = 0; j < p_DIM2_; ++j) {             \
        int64_t k = i * p_DIM2_ + j;                      \
        out[k] = a[k] p_OP_ b[k];                         \
      }                                                   \
    }                                                     \
  }

// 注 1
GCC_BINARY_OP_2D(gcc_0_0, *, 10, 10);
GCC_BINARY_OP_2D(gcc_0_1, -, 10, 10);
GCC_BINARY_OP_2D(gcc_0_2, +, 10, 10);

// 注 2
extern "C" void gcc_0_(float* gcc_input0, float* gcc_input1,
                       float* gcc_input2, float* gcc_input3, float* out) {
  float* buf_0 = (float*)malloc(4 * 100);
  float* buf_1 = (float*)malloc(4 * 100);
  gcc_0_2(gcc_input0, gcc_input1, buf_0);
  gcc_0_1(buf_0, gcc_input2, buf_1);
  gcc_0_0(buf_1, gcc_input3, out);
  free(buf_0);
  free(buf_1);
}

// 注 3
extern "C" int gcc_0_wrapper(DLTensor* arg0, DLTensor* arg1, DLTensor* arg2,
                             DLTensor* arg3, DLTensor* out) {
  gcc_0_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data),
         static_cast<float*>(arg2->data), static_cast<float*>(arg3->data),
         static_cast<float*>(out->data));
  return 0;
}
TVM_DLL_EXPORT_TYPED_FUNC(gcc_0, gcc_0_wrapper);
```

这里详细介绍一下上面代码里的注释：

* **注1**：子图中三个节点的函数实现。
* **注2**：通过分配中间数组 (intermediate buffer) 并调用相应函数来执行子图的函数。
* **注3**：TVM runtime 兼容的包装函数。它接收一个输入张量列表和一个输出张量（最后一个参数），并将其转换为正确的数据类型，调用注2 中描述的子图函数。此外，`TVM_DLL_EXPORT_TYPED_FUNC` 是一个 TVM 宏，它通过将所有张量打包到 `TVMArgs` 来生成另一个函数 `gcc_0`，该函数具有统一的函数参数。因此，TVM runtime 可以直接调用 `gcc_0` 来执行子图，无需其他操作。生成上述代码后，TVM 能够将其与计算图的其余部分一起编译并导出单个库以进行部署。
  
在本节的其余部分，我们将逐步创建一个 codegen，来实现上述代码。你的 codegen 必须位于 `src/relay/backend/contrib/<your-codegen-name>/`。在这个例子中，我们将 codegen 命名为 "codegen_c"，并将其放在 [/src/relay/backend/contrib/codegen_c/](https://github.com/apache/tvm/blob/main/src/relay/backend/contrib/codegen_c/codegen.cc) 目录下。你可以随时查看这个文件，了解完整的实现过程。

具体来说，我们将在这个文件中实现两个类，两个类的关系如下：

``` text
            subgraph                                subgraph
TVM backend -----------------------------> CSourceCodegen -------------> CodegenC
       ^                                       |    ^                       |
       |                                       |    |                       |
       ----------------------------------------      ------------------------
          generated C source runtime module              generated C code
```

当 TVM 后端发现 Relay 计算图中的函数（子图），用注册的编译器标记（本例中为 `ccompiler`）进行了注释时，TVM 后端就会调用 `CSourceCodegen` 并传递子图。 `CSourceCodegen` 的成员函数 `CreateCSourceModule` 将：

1）为子图生成 C 代码；

2）将生成的 C 代码包装到 C source runtime 模块中，以便 TVM 后端进行编译和部署。

特别是，C codegen 对 `CodegenC` 类是透明的，因为它提供了许多有用的实用程序来简化 codegen 实现。下面的章节将自下而上实现这两个类。

## 实现 CodegenC

在 `src/relay/backend/contrib/codegen_c/codegen.cc` 中，首先在 `tvm.relay.contrib` 的命名空间下创建一个 codegen 类骨架：

``` c++
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <sstream>

#include "codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

class CodegenC : public ExprVisitor, public CodegenCBase {
  public:
    explicit CodegenC(const std::string& id) { this->ext_func_id_ = id; }

    void VisitExpr_(const VarNode* node) { ; }
    void VisitExpr_(const CallNode* call) final { ; }
    std::string JIT() { ; }

  private:
    /*! \brief The function id that represents a C source function. */
    std::string ext_func_id_ = "";
    /*! \brief The index of a wrapped C function. */
    int func_idx = 0;
    /*! \brief The index of allocated buffers. */
    int buf_idx_ = 0;
    /*! \brief The arguments of a C compiler compatible function. */
    std::vector<std::string> ext_func_args_;
    /*! \brief The statements of a C compiler compatible function. */
    std::vector<std::string> ext_func_body;
    /*! \brief The declaration statements of a C compiler compatible function. */
    std::vector<std::string> func_decl_;
    /*! \brief The declaration statements of buffers. */
    std::vector<std::string> buf_decl_;
    /*! \brief The name and index pairs for output. */
    std::vector<std::pair<std::string, int>> out_;
}
```

`CodegenC` 类继承了两个类： `ExprVisitor` 提供遍历子图的能力，然后收集所需的信息并生成子图函数，例如 `gcc_0_`。

`CodegenCBase` 提供了生成包装函数的能力和实用程序，例如上例中的 `gcc_0`。可以看出，我们只需要在这个 codegen 类中实现三个函数就可以了。

### 算子的代码生成

首先实现 `VisitExpr_(const CallNode* call)`。该函数在遍历子图时会访问所有调用节点。每个调用节点都包含一个我们想要卸载 (offload) 到硬件中的算子。因此，我们需要按照拓扑顺序生成具有正确算子的相应 C 代码。完整实现过程如下：

#### 1. 生成函数声明

示例结果：`GCC_BINARY_OP_2D(gcc_0_0, *, 10, 10);`

要生成函数声明，如上所示，我们需要：

1）函数名（例如 `gcc_0_0`）

2）算子的类型（例如 `*` ）

3）输入张量 shape（例如 `(10, 10)` ）

这些信息可以从 `CallNode` 轻松获取：

``` c++
std::ostringstream macro_stream;
std::ostringstream decl_stream;
std::ostringstream buf_stream;

// Generate a unique function name you like.
std::string func_name = ext_func_id_ + "_" + std::to_string(func_idx++);

// Make function declaration string.
macro_stream << "CSOURCE_BINARY_OP_" << call->args.size() << "D(" << func_name << ", ";

// Check the operator type.
if (IsOp(call, "add")) {
  macro_stream << "+";
} else if (IsOp(call, "subtract")) {
  macro_stream << "-";
} else if (IsOp(call, "multiply")) {
  macro_stream << "*";
} else {
  LOG(FATAL) << "Unrecognized op";
}

// Extract the input tensor shape.
auto in_shape = GetShape(call->args[0]->checked_type());
for (size_t i = 0; i < in_shape.size(); ++i) {
  macro_stream << ", " << in_shape[i];
}
macro_stream << ");";
func_decl_.push_back(macro_stream.str());
```

可以看出，我们将生成的代码推送到类成员变量 `func_decl_` 中。这意味着在我们完成遍历整个子图之后，我们已经收集了所有必需的函数声明，我们唯一需要做的就是用 GCC 编译它们。 `VisitExpr_(const CallNode* call)` 的其余实现也遵循这个概念。

#### 2. 生成函数调用

示例结果：`gcc_0_0(buf_1, gcc_input3, out);`

生成函数声明后，我们需要生成一个具有正确输入和输出的函数调用。要想知道调用这个函数时应该放置哪些输入或数组，必须访问它的参数：

``` c++
bool first = true;
decl_stream << func_name << "(";
for (size_t i = 0; i < call->args.size(); ++i) {
  VisitExpr(call->args[i]); // 注 1
  for (auto out : out_) {
    if (!first) {
      decl_stream << ", ";
    }
    first = false;
    decl_stream << out.first;
  }
}
// 注 2
```

同样，重点介绍一下上述代码中的注释：

**注1**：`VisitExpr(call->args[i])` 是访问当前函数参数的递归调用。参数可以是另一个节点的输出或输入张量。在该示例中，需要确保每个节点在离开访问器之前，都更新一个类变量 `out_`。图解如下：

``` text
 arg_node                 arg_node <- Visit arg (Note 1)       arg_node
     |                        |                                    |
 curr_node <- Process      curr_node                            curr_node <- Put "buf_0" as an input buffer

(a) out_ = {}            (b) out_ = {}                   (c) out_ = {("buf_0", 20)}
```

从上图中可以看出，类变量 `out_` 在访问参数节点前是空的，它被填充了 `arg_node` 输出数组的名称和大小。因此在完成对参数节点的访问时，可以通过查看 `out_` 得知应该放置的正确输入数组。本节末尾以及下一节中，我们将介绍如何更新 `out_`。

**注2**：你可能注意到，我们在这一步没有关闭函数调用字符串。当前函数调用字符串看起来像：`gcc_0_0(buf_1, gcc_input3`。这是因为我们没有将最后一个参数（如 output）放入此调用中。函数调用的输出可以是分配的临时数组或子图输出张量。简单起见，在本例中我们为每个调用节点都分配老一个输出数组（下一步），并将最后一个数组中的结果复制到了输出张量。

#### 3. 生成输出数组 (output buffer)

示例结果：`float buf_0 = (float)malloc(4 * 100);`

如上一步所述，除了子图输入和输出张量外，还需要数组来保存中间结果。为了生成数组，我们提取 shape 信息，以确定数组的类型和大小：

``` c++
// 这个例子仅支持单个输出。
auto type_node = call->checked_type().as<TensorTypeNode>();
ICHECK(type_node != nullptr && runtime::TypeMatch(type_node->dtype, kDLFloat, 32))
      << "Only support single output tensor with float type";

// 生成一个唯一的数组名字。
std::string out = "buf_" + std::to_string(buf_idx_++);

// 提取 shape 作为数组大小。
auto out_shape = GetShape(call->checked_type());
int out_size = 1;
for (size_t i = 0; i < out_shape.size(); ++i) {
  out_size *= out_shape[i];
}

// 分配数组并推送至数组声明
buf_stream << "float* " << out << " = (float*)std::malloc(4 * " << out_size << ");";
buf_decl_.push_back(buf_stream.str());
```

分配了输出数组之后，现在可以关闭函数调用字符串，并将生成的函数调用推送到类变量 `ext_func_body`。

```plain
decl_stream << ", " << out << ");";
ext_func_body.push_back(decl_stream.str());
```

#### 4. 更新输出数组

为了使得下一个节点（接受当前调用节点的输出，作为其输入）知道它应该使用哪个数组，我们需要在离开这个访问函数之前更新类变量 `out_`：

```plain
out_.clear();
out_.push_back({out, out_size});
```

恭喜！到这一步我们已经完成了这个类中最困难的函数。接下来的两节中，我们将进一步完善这个函数的功能。

### 输入变量的代码生成

回想一下，我们通过访问调用节点的参数（上一节中的第 2 步）收集了输入数组信息，并处理了参数是另一个调用节点的情况（第 4 步）。本节我们将以 `VarNode` 为例，演示如何处理其他节点。

`VarNode` 表示模型中的输入张量。它非常重要的一点就是名称提示（例如，`data`、`weight` 等）。访问 `VarNode` 时，只需更新类变量 `out_` 传递名称提示，后代 (descendant) 调用节点就可以生成正确的函数调用。

``` c++
void VisitExpr_(const VarNode* node) {
  ext_func_args_.push_back(node->name_hint());
  out_.clear();
  out_.push_back({node->name_hint(), 0});
}
```

注意：在这个例子中，我们假设要卸载的子图只有调用节点和变量节点。如果子图包含其他类型的节点，如 `TupleNode`，那么你也需要访问它们并绕过输出数组信息。

### Code Emitting

Codegen Class 的最后一部分是 `JIT` 函数，它为子图 emit 一个 C 函数，并将刚生成的 C 代码作为函数体。注意，除了在前几节中生成的子图函数外，还需要一个具有统一参数的 wrapper 函数，供 TVM runtime 调用和传递数据。幸运的是，我们继承的基类已经提供了一个实现，即 `JitImpl`，来生成该函数。调用 `JitImpl`的方式如下：

``` c++
JitImpl("gcc_0" /* Subgraph symbol (ID) */,
        {"gcc_input0", "gcc_input1", "gcc_input2", "gcc_input3"} /* Input arguments */,
        {"float *buf_0 = (float*)malloc(4 * 20)", ...} /* Buffer allocations */,
        {"gcc_0_2(gcc_input0, gcc_input1, buf_0);"} /* Function body */,
        {"out"} /* Output */);
```

上述调用将生成三个函数（一个来自 TVM wrapper 宏）：

1. 子图函数 `gcc_0_`（函数名末尾多了一个下划线）以及为执行子图而生成的所有 C 代码；
2. 带有 `DLTensor` 参数列表的 wrapper 函数 `gcc_0__wrapper_` ，将数据转换为正确的类型并调用 `gcc_0_`
3. TVM runtime 兼容函数 `gcc_0` 具有 TVM 统一函数参数，可解包 TVM 打包张量并调用  `gcc_0__wrapper_`

因此，在 `JIT` 实现中唯一要做的，就是将生成的所有子图函数代码传递给 `JitImpl`：

``` c++
std::string JIT() {
  // Write function macros
  for (auto decl : func_decl_) {
    code_stream_ << decl << "\n";
  }
  return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out_);
}
```

传递的所有变量（`ext_func_id` 等）都是类变量，并在遍历子图时被填充。

#### 实现 CSourceCodegen

创建一个类骨架并实现所需功能，注意：需要延续使用 `CSourceModuleCodegenBase`：

``` c++
class CSourceCodegen : public CSourceModuleCodegenBase {
 public:
  // 传递一个子图函数, 并生成 C 代码。
  void GenCFunc(const Function& func) { ; }

  // 使用 GenCFunc 来生成 C 代码并将它包装成一个 C 源模块。
  runtime::Module CreateCSourceModule(const NodeRef& ref) override { ; }

 private:
  std::ostringstream code_stream_;
};
```

#### 实现 GenCFunc

`GenCFunc` 只是简单地使用我们刚刚实现的 `CodegenC` 来遍历一个 Relay 函数（子图），得到生成的 C 代码。内置函数 `GetExtSymbol` 在 Relay 函数中检索唯一的符号名称（例如 `gcc_0`），注意：**必须**将其用作 C 函数名称，因为该符号将用于 DSO 运行查找。

```plain
void GenCFunc(const Function& func) {
  ICHECK(func.defined()) << "Input error: expect a Relay function.";

  // 记录运行查找的外部符号。
  auto sid = GetExtSymbol(func);

  CodeGenC builder(sid);
  builder.VisitExpr(func->body);
  code_stream_ << builder.JIT();
}
```

#### 实现 CreateCSourceModule

此函数为外部库创建了一个 runtime 模块。本事例中，我们创建了一个可以直接被编译并与 TVM 生成的 DSOModule 链接在一起的 CSourceModule。`CodegenC` 实现之后，再实现这个功能就比较简单了：

``` c++
runtime::Module CreateCSourceModule(const NodeRef& ref) override {
  // 创建头文件
  code_stream_ << "#include <cstdint>\n";
  code_stream_ << "#include <iostream>\n";
  code_stream_ << "#include <cstdlib>\n";
  code_stream_ << "#include <stdio.h>\n";
  code_stream_ << "#include <cstring>\n";
  code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
  code_stream_ << "#include <dlpack/dlpack.h>\n";

  // 为算子定义添加一些公共宏。
  const char* operator_macro = R"op_macro(
  #define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)       \
    extern "C" void p_ID_(float* a, float* b, float* out) { \
      for (int64_t i = 0; i < p_DIM1_; ++i) {               \
        out[i] = a[i] p_OP_ b[i];                           \
      }                                                     \
    }

  #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
    extern "C" void p_ID_(float* a, float* b, float* out) {     \
      for (int64_t i = 0; i < p_DIM1_; ++i) {                   \
        for (int64_t j = 0; j < p_DIM2_; ++j) {                 \
          int64_t k = i * p_DIM2_ + j;                          \
          out[k] = a[k] p_OP_ b[k];                             \
        }                                                       \
      }                                                         \
    }
  )op_macro";

  code_stream_ << operator_macro << "\n\n";

  // 为子图生成 C 代码。
  if (ref->IsInstance<FunctionNode>()) {
    GenCFunc(Downcast<Function>(ref));
  } else if (ref->IsInstance<relay::ModuleNode>()) {
    relay::Module mod = Downcast<relay::Module>(ref);
    for (const auto& it : mod->functions) {
      GenCFunc(Downcast<Function>(it.second));
    }
  } else {
    LOG(FATAL) << "The input ref is expected to be a Relay function or module"
               << "\n";
  }

  // 创建一个 CSourceModule
  const auto* pf = runtime::Registry::Get("module.csource_module_create");
  ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
  return (*pf)(code_stream_.str(), "cc");
}
```

### 注册 CodegenC

最后一步是将 codegen 注册到 TVM 后端。首先实现一个简单的函数，调用 codegen 并生成一个 runtime 模块：

``` c++
runtime::Module CCompiler(const NodeRef& ref) {
  CSourceCodegen csource;
  return csource.CreateCSourceModule(ref);
}
```

接下来将此函数注册到 TVM 后端：

``` c++
TVM_REGISTER_GLOBAL("relay.ext.ccompiler").set_body_typed(CCompiler);
```

其中 `ccompiler` 是一个自定义标签，它告知 TVM 这是用 `ccompiler` 注释子图时，应该用来生成和卸载子图的 codegen。

最后，设置一个 CMake 配置标志，只包含客户的编译器。首先创建一个 cmake 文件：`cmake/modules/contrib/CODEGENC.cmake`：

``` c++
if(USE_CODEGENC)
  file(GLOB CSOURCE_RELAY_CONTRIB_SRC src/relay/backend/contrib/codegen_c/codegen.cc)
  list(APPEND COMPILER_SRCS ${CSOURCE_RELAY_CONTRIB_SRC})
endif(USE_CODEGENC)
```

用户在使用 `config.cmake` 配置 TVM 时，可以自行决定是否配置编译器：

``` cmake
set(USE_CODEGENC ON)
```

## 为表征 (Representation) 实现 Codegen

尽管我们已经演示了如何实现 C codegen，但用户硬件可能还需要其他形式的计算图表征 (Graph Representation)，如 JSON。在这种情况下，用户可以通过修改 `CodegenC` 类，生成自己的计算图表征，并实现一个自定义 runtime 模块，告诉 TVM runtime 如何执行这个计算图表征。

简单起见，本指南中定义了一个名为 "ExampleJSON" 的计算图表征。 ExampleJSON 并不是 JSON，而是没有控制流的计算图的简单表示。例如，假设有以下名为 `subgraph_0` 的子图：

``` text
input0
   |
  add <-- input1
   |
subtract <-- input2
   |
multiply <-- input3
   |
  out
```

那么这个子图的 ExampleJON 看起来类似：

``` text
subgraph_0
  input 0 10 10
  input 1 10 10
  input 2 10 10
  input 3 10 10
  add 4 inputs: 0 1 shape: 10 10
  sub 5 inputs: 4 2 shape: 10 10
  mul 6 inputs: 5 3 shape: 10 10
```

`input` 关键字声明一个输入张量及其 ID 和 shape；其他语句用 `<op> <output ID> inputs: [input ID] shape: [shape]` 语法描述了其计算过程。

在本节中，我们试图实现以下自定义 TVM runtime 模块，来执行 ExampleJSON 计算图。

``` c++
runtime::Module ExampleJsonCompiler(const NodeRef& ref) {
    ExampleJsonCodeGen codegen(ref);
    std::string code = codegen.gen(); // 注 1
    const auto* pf = runtime::Registry::Get("module.examplejson_module_create"); // 注 2
    ICHECK(pf != nullptr) << "Cannot find ExampleJson module to create the external runtime module";
    return (*pf)(code);
}
TVM_REGISTER_GLOBAL("relay.ext.examplejsoncompiler").set_body_typed(ExampleJsonCompiler);
```

**注1**：稍后我们将实现一个自定义 codegen，通过取一个子图来生成一个 ExampleJSON 代码字符串。

**注2**：此行获取了一个用于创建自定义 runtime 模块的函数的指针。可以看到它采用刚刚生成的 ExampleJSON 格式的子图代码，并对一个 runtime 模块进行了初始化。

后续章节中，我们将介绍 1）如何实现 `ExampleJsonCodeGen` 和 2）如何实现和注册 `examplejson_module_create`。

### 实现 ExampleJsonCodeGen

与 C codegen 类似，从 `ExprVisitor` 派生 `ExampleJsonCodeGen` 以访问器模式进行子图遍历。另一方面，因为不会用到 TVM C++ wrapper，所以不必继承 `CodegenCBase`。 codegen 类实现如下：

``` c++
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <sstream>

namespace tvm {
namespace relay {
namespace contrib {

class ExampleJsonCodeGen : public ExprVisitor {
  public:
    explicit ExampleJsonCodeGen();

    // 注 1
    void VisitExpr_(const VarNode* node) { /* Skip in this example. */ }
    void VisitExpr_(const CallNode* call) final { /* Skip in this example. */ }

    // 注 2
    std::string gen(NodeRef& ref) {
        this->code = "";
        if (ref->IsInstance<FunctionNode>()) {
            this->visit(Downcast<Function>(ref));
        } else if (ref->IsInstance<relay::ModuleNode>()) {
            relay::Module mod = Downcast<relay::Module>(ref);
            for (const auto& it : mod->functions) {
                this->visit(Downcast<Function>(it.second));
            }
        } else {
            LOG(FATAL) << "The input ref is expected to be a Relay function or module";
        }
        return this->code;
    }

  private:
      /*! \brief The function id that represents a C source function. */
     std::string code;
}
```

**注1**：再次实现相应的 visitor 函数，以生成 ExampleJSON 代码，并将其存储到类变量 `code` 中（由于与 C codegen 基本一致，这里跳过了 visitor 函数的实现）。完成计算图访问后，在 `code` 中会生成一个 ExampleJSON 计算图。

**注2**：定义内部 API `gen` 来获取子图，并生成 ExampleJSON 代码。用户可以依据个人喜好，为这个 API 命名。

接下来，实现一个自定义 runtime，来利用 `ExampleJsonCodeGen` 的输出。

### 实现自定义 runtime

本节将逐步演示如何自定义 TVM runtime，并将其注册到 TVM runtime 模块。自定义 runtime 应位于 `src/runtime/contrib/<your-runtime-name>/`。本示例中，我们将 runtime 命名为 "example_ext_runtime"。

首先，如下所示定义一个自定义 runtime 类。注意：这个类必须由 TVM `ModuleNode` 派生，以保证与其他 TVM runtime 模块兼容。

``` c++
#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
class ExampleJsonModule : public ModuleNode {
 public:
  explicit ExampleJsonModule(std::string graph_json);

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final;

  const char* type_key() const { return "examplejson"; }

  void SaveToBinary(dmlc::Stream* stream) final;

  static Module LoadFromBinary(void* strm);

  static Module Create(const std::string& path);

  std::string GetSource(const std::string& format = "");

  void Run(int id, const std::vector<int>& inputs, int output);

  void ParseJson(const std::string& json);

 private:
  /* \brief 代表计算图的 json 字符串。 */
  std::string graph_json_;
  /* \brief 正在被处理的子图。 */
  std::string curr_subgraph_;
  /*! \brief 由子图 id 到节点条目的简单图。 */
  std::map<std::string, std::vector<NodeEntry> > graph_;
  /* \brief 包含图中每一个节点的张量的简单池。 */
  std::vector<NDArray> data_entry_;
  /* \brief 从节点 id 到算子名字的映射。 */
  std::vector<std::string> op_id_;
};
```

以下这些从 `ModuleNode` 派生的函数，必须在 `ExampleJsonModule` 中实现：

* 构造函数：这个类的构造函数，应该接收一个表征中的子图，用户可以自行决定处理和存储的格式。保存的子图可以被以下两个函数使用。
* `GetFunction`：这是这个类中最重要的函数。当 TVM runtime 要使用编译器标记 (compiler tag) 执行子图时，它会从自定义 runtime 模块中调用此函数。它提供函数名及 runtime 参数，`GetFunction` 会返回一个打包的函数实现，以供 TVM runtime 执行。
* `SaveToBinary` 和 `LoadFromBinary`：`SaveToBinary` 将 runtime 模块序列化为二进制格式以供后续部署。用户使用 `export_library` API 时，TVM 会调用这个函数。另一方面，由于用户这时使用的是自己的计算图表征，因此必须确保 `LoadFromBinary` 能够采用`SaveToBinary` 生成的序列化二进制文件，来构造相同的 runtime 模块。
* `GetSource`（可选）：如果想查看生成的 ExampleJSON 代码，可以实现这个函数来转存；否则则可以跳过实现。

#### 实现构造函数

``` c++
explicit ExampleJsonModule(std::string graph_json) {
  this->graph_json_ = graph_json;
  ParseJson(this->graph_json_);
}
```

接下来，实现 `ParseJson` 来解析 ExampleJSON 格式的子图，并在内存中构造一个计算图供后续使用。由于本示例不支持带有分支的子图，因此只需用一个数组，按顺序存储子图中的每个节点。

``` c++
void ParseJson(const std::string& json) {
  std::string line;
  std::string curr_subgraph;
  std::stringstream ss(json);

  while (std::getline(ss, line, '\n')) {
    std::stringstream ss2(line);
    std::string token;
    int id = 0;

    ss2 >> token;
    if (token.find("subgraph_") != std::string::npos) {
      curr_subgraph = token;
      continue;
    }

    ss2 >> id;
    if (op_id_.size() <= static_cast<size_t>(id)) {
      op_id_.resize(id + 1);
      data_entry_.resize(id + 1);
    }

    int64_t total_elements = 1;
    std::vector<int64_t> shape;
    if (token == "input") {
      int64_t size = 0;
      while (ss2 >> size) {
        total_elements *= size;
        shape.push_back(size);
      }
    } else {
      op_id_[id] = token; // 注 1
      bool shape_data = false;
      NodeEntry entry;
      while (ss2 >> token) {
        if (token == "shape:") {
          shape_data = true;
        } else if (shape_data) {
          total_elements *= std::stoll(token);
          shape.push_back(std::stoll(token));
        } else if (token != "inputs:") {
          entry.inputs.push_back(std::stoi(token));
        }
      }
      entry.id = id;
      entry.output = id;
      graph_[curr_subgraph].push_back(entry); // 注 2
    }
    DLDevice dev;
    dev.device_type = static_cast<DLDeviceType>(1);
    dev.device_id = 0;
    data_entry_[id] = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, dev); // 注 3
  }
}
```

**注1**：使用类变量 `op_id_` 将子图节点 ID 映射到算子名称（例如 `add`），以便在 runtime 中调用相应的算子函数。

**注2**：使用类变量 `graph_` 从子图名称映射到节点数组。`GetFunction` 将在 runtime 通过子图 ID 查询计算图节点。

**注3**：使用类变量 *data_entry_* 将子图节点 ID 映射到张量数据占位符。将输入和输出放入 runtime 中对应的数据条目中。

#### 实现 GetFunction

构造函数实现后，以上类变量准备就绪。接下来实现 `GetFunction` 为 TVM runtime 提供可执行的子图函数：

``` c++
PackedFunc GetFunction(const std::string& name,
                       const ObjectPtr<Object>& sptr_to_self) final {
  if (this->graph_.find(name) != this->graph_.end()) {
    this->curr_subgraph_ = name;
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {

      // Copy input tensors to corresponding data entries.
      for (auto i = 0; i < args.size(); ++i) {
        ICHECK(args[i].type_code() == kNDArrayContainer || args[i].type_code() == kArrayHandle)
            << "Expect NDArray or DLTensor as inputs\n";
        if (args[i].type_code() == kArrayHandle) {
          DLTensor* arg = args[i];
          this->data_entry_[i].CopyFrom(arg);
        } else {
          NDArray arg = args[i];
          this->data_entry_[i].CopyFrom(arg);
        }
      }

      // Execute the subgraph.
      for (const auto& it : this->graph_[this->curr_subgraph_]) {
        this->Run(it.id, it.inputs, it.output);
      }
      ICHECK_GT(graph_.count(this->curr_subgraph_), 0U);

      // Copy the output from a data entry back to TVM runtime argument.
      auto out_idx = graph_[this->curr_subgraph_].back().output;
      if (args[args.size() - 1].type_code() == kArrayHandle) {
        DLTensor* arg = args[args.size() - 1];
        this->data_entry_[out_idx].CopyTo(arg);
      } else {
        NDArray arg = args[args.size() - 1];
        this->data_entry_[out_idx].CopyTo(arg);
      }
      *rv = data_entry_.back();
    });
  } else {
    LOG(FATAL) << "Unknown subgraph: " << name << "\n";
    return PackedFunc();
  }
}
```

可以看出，`GetFunction` 由三个主要部分组成。第一部分将数据从 TVM runtime 参数，复制到构造函数中指定的对应数据条目。第二部分使用 `Run` 函数执行子图（并稍后实现），并将结果保存到另一个数据条目。第三部分将输出数据条目中的结果，复制回对应的 TVM runtime 参数进行输出。

#### 实现 Run

`Run` 函数接收 1）子图 ID，2）输入数据条目索引列表和 3）输出数据条目索引。

``` c++
void Run(int id, const std::vector<int>& inputs, int output) {
  // Make a list data entry indexs.
  std::vector<int> args(inputs.begin(), inputs.end());
  args.push_back(output);

  // Initialize data holders.
  std::vector<TVMValue> values(args.size());
  std::vector<int> type_codes(args.size());

  // Initialize a TVM arg setter with TVMValue and its type code.
  TVMArgsSetter setter(values.data(), type_codes.data());

  // Set each argument to its corresponding data entry.
  if (op_id_[id] == "add" || op_id_[id] == "sub" || op_id_[id] == "mul") {
    for (size_t i = 0; i < args.size(); i++) {
      setter(i, data_entry_[args[i]]);
    }
  }

  // Invoke the corresponding operator function.
  if (op_id_[id] == "add") {
    Add(values.data(), type_codes.data(), args.size());
  } else if (op_id_[id] == "sub") {
    Sub(values.data(), type_codes.data(), args.size());
  } else if (op_id_[id] == "mul") {
    Mul(values.data(), type_codes.data(), args.size());
  } else {
    LOG(FATAL) << "Unknown op: " << op_id_[id] << "\n";
  }
}
```

`Run` 函数主要包括两部分。第一部分负责分配 `TVMValue` 列表，并映射相应的数据输入块。这也会成为算子函数的参数。第二部分调用算子函数。尽管使用的 C 函数与上一个示例相同，但用户可以将 `Add`、`Sub` 和 `Mul` 替换为自己的引擎。注意，这里需要确保引擎将结果存储到最后一个参数，从而使得它们可以传输回 TVM runtime。

实现上述功能后，用户自定义的 codegen 和 runtime 就可以执行子图了。最后一步是注册一个 API（`examplejson_module_create`）来创建这个模块：

``` c++
TVM_REGISTER_GLOBAL("module.examplejson_module_create")
.set_body_typed([](std::string code){
    auto n = make_object<ExampleJsonModule>(code);
    return runtime::Module(n);
});
```

#### 实现 SaveToBinary 和 LoadFromBinary

到目前为止，我们已经实现了与其他 TVM runtime 用法一致的自定义 runtime 的主要功能。但是，当用户想要将构建的 runtime 保存到磁盘以进行部署时，TVM 不知道如何保存。这就是实现 `SaveToBinary` 和 `LoadFromBinary` 的原因，它们会告诉 TVM 这个自定义 runtime 如何持久化和复原。

首先实现 `SaveToBinary` 函数，允许用户将此模块保存在磁盘中。

``` c++
void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(this->graph_json_);
}
```

这个函数非常简单。在构造函数中，我们采取的唯一参数是一个子图表征 (subgraph representation)。也就是说只需一个子图表征来构造/恢复这个自定义 runtime 模块。`SaveToBinary` 只是将子图写到一个输出的 DMLC 流中，当用户使用 `export_library` API 输出模块时，自定义模块将是一个子图的 ExampleJSON 流。

`LoadFromBinary` 读取子图流并重新构建自定义 runtime 模块的流程与此类似：

``` c++
static Module LoadFromBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string graph_json;
  stream->Read(&graph_json);
  auto n = tvm::runtime::make_object<ExampleJsonModule>(graph_json);
  return Module(n);
}
```

此外，还需要注册以下函数，启用相应的 Python API：

``` c++
TVM_REGISTER_GLOBAL("module.loadbinary_examplejson")
.set_body_typed(ExampleJsonModule::LoadFromBinary);
```

上述注册意味着当用户调用 `tvm.runtime.load_module(lib_path)` API，并且导出库有一个 ExampleJSON 流时，`LoadFromBinary` 将被调用以创建相同的自定义 runtime 模块。

另外，如果想支持直接从 ExampleJSON 文件创建模块，还可以实现一个非常简单的函数，并注册一个 Python API，如下所示：

``` c++
static Module Create(const std::string& path) {
    std::ifstream filep;
    filep.open(path, std::ios::in);
    std::string graph_json;
    std::string line;
    while (std::getline(filep, line)) {
        graph_json += line;
        graph_json += "\n";
    }
    filep.close();
    auto n = tvm::runtime::make_object<ExampleJsonModule>(graph_json);
    return Module(n);
}

TVM_REGISTER_GLOBAL("module.loadfile_examplejson")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = ExampleJsonModule::Create(args[0]);
});
```

这意味着用户可以手动编写/修改 ExampleJSON 文件，并使用 Python API `tvm.runtime.load_module("mysubgraph.examplejson", "examplejson")` 构建自定义模块。

## 总结

汇总前文重点：

* 从 `ExprVisitor` 和 `CodegenCBase`（仅适用于 C codegen）派生的 codegen 类，具有以下功能：
  * `VisitExpr_(const CallNode* call)` 收集调用节点信息。
  * 收集子图信息所需的其他 visitor 函数。
  * `JIT` 生成子图代码。
  * 注册 codegen。
* 创建 `CSourceModule` 的函数（用于 C codegen）。
* 从 `ModuleNode` 派生的 runtime 模块类，具有以下功能（用于计算图表征）。
  * 构造函数。
  * `GetFunction` 生成与 TVM runtime 兼容的 `PackedFunc`。
  * `Run` 执行子图。
  * 注册 runtime creation API。
  * `SaveToBinary` 和 `LoadFromBinary` 序列化/反序列化自定义 runtime 模块。
  * 注册 `LoadFromBinary` API 为`tvm.runtime.load_module(your_module_lib_path)`提供支持。
  * （可选）`Create` 支持从表征的子图文件，构建自定义 runtime 模块。
* 一个注释器，用于注释用户 Relay 程序，利用编译器和 runtime（待定）。
