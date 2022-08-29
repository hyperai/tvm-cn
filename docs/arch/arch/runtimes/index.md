---
title: TVM Runtime 系统
---

# TVM Runtime 系统

TVM 支持多种编程语言进行编译器堆栈开发和部署。本文档将介绍 TVM runtime 的关键元素。

![https://tvm.apache.org/images/release/tvm_flexible.png](https://tvm.apache.org/images/release/tvm_flexible.png)

需要满足以下要求：

* 部署：从 Python/JavaScript/C++ 语言调用编译好的函数。
* 调试：在 Python 中定义一个函数，并从编译好的函数中调用它。
* 链接：编写驱动程序代码以调用特定设备代码（CUDA），并从编译的主机函数中调用它。
* 原型：从 Python 中定义一个 IR pass，并从 C++ 后端调用它。
* 公开：将 C++ 开发的编译器堆栈用到前端（即 Python）。
* 实验：将编译好的函数发送到嵌入式设备上直接运行。
  我们期望能够用任何语言定义一个函数，然后用另一种语言调用。还期望将 runtime 内核最小化，部署到嵌入式设备。

## PackedFunc

[PackedFunc](https://github.com/apache/tvm/blob/main/include/tvm/runtime/packed_func.h) 是一个简单而优雅的解决方案，它可以解决以上问题。单个 `PackedFunc` 对象代表一个函数调用，其调用者和被调用者可能使用不同的语言。

以下代码块提供了一个 C++ 示例：

``` c++
#include <tvm/runtime/packed_func.h>

void MyAdd(TVMArgs args, TVMRetValue* rv) {
  // 自动将参数转换为所需的类型。
  int a = args[0];
  int b = args[1];
  // 自动赋值返回给 rv
  *rv = a + b;
}

void CallPacked() {
  PackedFunc myadd = PackedFunc(MyAdd);
  // 返回 3
  int c = myadd(1, 2);
}
```

以上代码块中定义了一个 PackedFunc MyAdd。它有两个参数：`args` 代表输入参数，`rv` 代表返回值。该函数是类型擦除的，这意味着函数签名不限制传入或返回的输入类型。在后台调用 PackedFunc 时，它会将输入参数打包到堆栈上的 TVMArgs，并通过 TVMRetValue 获取结果。

由于 C++ 中的模板技巧，我们可以像调用普通函数一样来调用 PackedFunc。其类型擦除的性质，使得可以从动态语言（如 Python）中调用 PackedFunc，而无需为每个创建的新类型函数添加额外的胶水代码。以下示例在 C++ 中注册 PackedFunc，并在 Python 中调用。

``` c++
// 在 C++ 中注册一个全局打包函数
TVM_REGISTER_GLOBAL("myadd")
.set_body(MyAdd);
```

``` python
import tvm

myadd = tvm.get_global_func("myadd")
# 打印 3
print(myadd(1, 2))
```

PackedFunc 的关键在于 `TVMArgs` 和 `TVMRetValue` 结构。我们限制了可传递的可能类型列表。以下是常见的类型：

* 整数、浮点数和字符串
* PackedFunc 本身
* 编译模块的模块
* DLTensor* 用于张量对象交换
* TVM 对象表示 IR 中的任何对象

该限制使实现简单，且无需序列化。尽管是最小的，但 PackedFunc 对于深度学习部署的用例来说已经足够了，因为大多数函数只需要 DLTensor 或数字。

由于一个 PackedFunc 可以将另一个 PackedFunc 作为参数，因此可以将函数从 Python（作为 PackedFunc）传递给 C++。

``` c++
TVM_REGISTER_GLOBAL("callhello")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  PackedFunc f = args[0];
  f("hello world");
});
```

``` python
import tvm

def callback(msg):
    print(msg)

# 转换成 PackedFunc
f = tvm.convert(callback)
callhello = tvm.get_global_func("callhello")
# 打印 hello world
callhello(f)
```

TVM 提供了一个 [最小的 C API](https://github.com/apache/tvm/blob/main/include/tvm/runtime/c_runtime_api.h)，因此可将 PackedFunc 嵌入到任何语言中。除了 Python，目前还支持 [java](https://github.com/apache/tvm/tree/main/jvm) 和 [javascript](https://github.com/apache/tvm/tree/main/web)。这种嵌入式 API 的原理很像 Lua，除了它用的是 C++ 语言而非新的语言。

PackedFunc 用于编译器和部署堆栈：

* TVM 的所有编译器 pass 函数都以 PackedFunc 的形式暴露给前端
* 编译好的模块还将编译好的函数返回为 PackedFunc

为了将 runtime 保持为最小，我们将 IR 对象支持与部署 runtime 隔离开来。生成的 runtime 大约需要 200K - 600K，具体取决于包含多少 runtime 驱动程序模块（例如，CUDA）。

与普通函数相比，调用 PackedFunc 的开销很小，因为它只在堆栈上保存了几个值，所以只要不包装小的函数即可。总之，PackedFunc 是 TVM 中的通用粘合剂，可以广泛使用它来支持编译器和部署。

## 模块

由于 TVM 支持多种类型的设备，因此需要支持不同类型的驱动程序。必须用驱动程序 API 来加载内核，以打包格式设置参数，并执行内核启动。

还需要为驱动程序 API 打补丁，以便公开的函数是线程安全的。因此，经常要在 C++ 中实现这些驱动粘合，并将它们提供给用户。但不能对每种类型的函数都这样做，PackedFunc 又可用来辅助实现。

TVM 将编译好的对象定义为 [Module](https://github.com/apache/tvm/blob/main/include/tvm/runtime/module.h)。用户可以从 Module 中获取编译好的函数为 PackedFunc。生成的编译代码可以从 runtime 中的 Module 中动态获取函数。它在第一次调用中缓存函数句柄，并在后续调用中再次使用。用它来将设备代码和回调函数链接到生成的代码中的任何 PackedFunc（例如 Python）。

ModuleNode 是一个抽象类，可由每种类型的设备实现。目前支持 CUDA、Metal、OpenCL 和加载动态共享库的模块。这种抽象使得新设备的引入变得容易，不需要为每种类型的设备重新生成主机代码。

## 远程部署

PackedFunc 和模块系统还可以轻松地将函数直接发送到远程设备。在底层有一个 RPCModule，用于序列化参数，从而进行数据移动并在远程启动计算。

![https://tvm.apache.org/images/release/tvm_rpc.png](https://tvm.apache.org/images/release/tvm_rpc.png)

RPC 服务器本身是最小的，可以捆绑到 runtime 中。可以在 iPhone/android/raspberry pi 甚至浏览器上启动一个最小的 TVM RPC 服务器。服务器上的交叉编译和测试模块的交付可以在同一个脚本中完成。查看 [交叉编译和 RPC](../../../user_guide/user_tutorial/rpc) 以获取更多详细信息。

这种即时反馈带来了很多优势，例如，在 iPhone 上测试生成代码的正确性，不再需要从头开始在 swift/objective-c 中编写测试用例——可以使用 RPC 在 iPhone 上执行，将结果复制回来并在主机上通过 numpy 进行验证，也可以使用相同的脚本进行分析。

## TVM 对象和编译器堆栈

如前所述，在 PackedFunc runtime 系统之上构建编译器堆栈 API。因研究需要，编译器 API 在不断变化。要测试新的原语时，都需要一个新的语言对象或 IR 节点。但我们又不想经常更改 API。除此之外，还想

* 能够序列化任何语言对象和 IR
* 能够以前端语言探索、打印和操作 IR 对象以快速进行原型设计。

引入一个名为 [Object](https://github.com/apache/tvm/blob/main/include/tvm/runtime/object.h) 的基类来解决这个问题。编译器堆栈中的所有语言对象都是 `Object` 的子类，每个对象都包含一个字符串 type_key，用于唯一标识对象的类型。

之所以选择 string 而不是 int 作为类型键，是因为可以去中心化的方式来添加新的 `Object` 类，无需将代码添加回中心仓库。为了加快调度速度，在运行时为每个 type_key 分配一个整数 type_index。

一个 `Object` 通常可以在语言中的多个位置引用，因此使用 shared_ptr 来跟踪引用。用 `ObjectRef` 类表示对 `Object` 的引用。可以粗略地将 `ObjectRef` 类视为 `Object` 容器的 shared_ptr。还可以定义子类 `ObjectRef` 来保存 `Object` 的每个子类型。 `Object` 的每个子类都需要定义 VisitAttr 函数。

``` c++
class AttrVisitor {
public:
  virtual void Visit(const char* key, double* value) = 0;
  virtual void Visit(const char* key, int64_t* value) = 0;
  virtual void Visit(const char* key, uint64_t* value) = 0;
  virtual void Visit(const char* key, int* value) = 0;
  virtual void Visit(const char* key, bool* value) = 0;
  virtual void Visit(const char* key, std::string* value) = 0;
  virtual void Visit(const char* key, void** value) = 0;
  virtual void Visit(const char* key, Type* value) = 0;
  virtual void Visit(const char* key, ObjectRef* value) = 0;
  // ...
};

class BaseAttrsNode : public Object {
public:
  virtual void VisitAttrs(AttrVisitor* v) {}
  // ...
};
```

每个 `Object` 子类将覆盖它以访问其成员。以下是 TensorNode 的实现示例：

``` c++
class TensorNode : public Object {
public:
  // 张量的形状
  Array<Expr> shape;
  // 张量内容中的简要数据类型
  Type dtype;
  // 简述源码操作，可以是None
  Operation op;
  // 简述源操作的输出索引
  int value_index{0};
  // 简要构造函数
  TensorNode() {}

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
    v->Visit("op", &op);
    v->Visit("value_index", &value_index);
  }
};
```

以上例子中，`Operation` 和 `Array<Expr>` 都是 ObjectRef。 VisitAttrs 提供了一个反射 API 来访问对象的每个成员。可以用这个函数来访问节点，并递归地序列化任何语言对象。还可以用它在前端语言中轻松获取对象的成员。例如，以下代码中，访问了 TensorNode 的 op 字段：

``` python
import tvm
from tvm import te

x = te.placeholder((3,4), name="x")
# 访问 TensorNode 的 op 字段
print(x.op.name)
```

可在不更改前端 runtime 的情况下，将新 `Object` 添加到 C++，从而轻松扩展编译器堆栈。

注意，这不是将成员提供给前端语言的最快方法，但可能是最简单的方法之一。并且它符合需求，因为我们主要用 Python 进行测试和原型设计，并仍用 C++ 来完成繁重的工作。

## 实现细节

PackedFunc 中的每个参数都包含一个关联值 [TVMValue](https://github.com/apache/tvm/blob/main/include/tvm/runtime/c_runtime_api.h#L135) 和一个类型代码。这种设计使得动态类型语言可直接转换为相应的类型，而静态类型语言在转换过程中，会进行 runtime 类型检查。

相关文件：

* [packed_func.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/packed_func.h)，用于 C++ API
* [c_runtime_api.cc](https://github.com/apache/tvm/blob/main/src/runtime/c_runtime_api.cc#L262)，用于 C API 以及如何提供回调。
  为了支持扩展类型，使用了注册表系统来注册类型相关信息，如 C++ 中对所有类型的支持，参阅 [扩展类型](https://github.com/apache/tvm/tree/main/apps/extension) 了解更多详细信息。

# Runtime 特定信息 {#runtime-specific-information}

* [Vulkan Runtime](https://tvm.apache.org/docs/arch/runtimes/vulkan.html)