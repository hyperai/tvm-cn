---
title: TVM 代码库实例讲解
---

了解新代码库是一个挑战，对于 TVM
这样组件众多、交互方式复杂的代码库来说更是如此。本指南将通过简单示例，介绍构成编译管道的关键部分，以及所有重要步骤在代码库中的实现位置，从而帮助开发者更快速地上手
TVM。

## 代码库结构概述

TVM 仓库的根目录，包括以下几个子目录：

-   `src` - 用于算子编译和部署 runtime 的 C++ 代码。
-   `src/relay` - Relay 的实现，一种用于深度学习框架的新功能 IR。
-   `python` - Python 前端，用于包装 `src` 中实现的 C++ 函数和对象。
-   `src/topi` - 标准神经网络算子的计算定义和后端调度。

用标准的深度学习术语来解释，`src/relay`
是管理计算图的组件，图结构中的节点使用 `src`
其余部分实现的基础架构进行编译和执行。`python` 为 C++ API
和执行编译的驱动代码，提供 Python 绑定。与节点对应的算子注册在
`src/relay/op` 中。算子的实现在 `topi` 中，所用编程语言为 C++ 或
Python。

用户通过 `relay.build(...)`
调用图结构编译时，图结构中的所有节点的序列会发生以下变化：

-   通过查询算子注册表来查找算子的实现
-   为算子生成计算表达式和调度
-   将算子编译成目标代码

TVM 代码库有趣的地方在于 C++ 和 Python
之间的互操作性不是单向的。通常情况下，所有执行繁重任务的代码都是用 C++
实现的，而 Python 绑定用于用户界面。TVM 中也是如此，只不过在 TVM
代码库中，C++ 代码也可以调用 Python 模块中定义的函数。例如，卷积算子是在
Python 中实现的，它的实现是由 Relay 中的 C++ 代码调用的。

## 向量加法示例

本文档将借助简单示例 \-- 向量加法，介绍如何直接调用底层 TVM
API。关于向量加法的详细介绍，请查看：`tutorial-tensor-expr-get-started`{.interpreted-text
role="ref"}

    n = 1024
    A = tvm.te.placeholder((n,), name='A')
    B = tvm.te.placeholder((n,), name='B')
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name="C")

这里，定义在 `python/tvm/te/tensor.py` 中的 `A`、`B`和 `C`，类型都是
`tvm.tensor.Tensor`。Python `Tensor` 由 C++ `Tensor` 支持，在
`include/tvm/te/tensor.h` 和 `src/te/tensor.cc` 中实现。TVM 中的所有
Python 类型都可以视为具有相同名称的底层 C++ 类型的句柄。查看以下 Python
`Tensor` 类型的定义，可以发现它是 `Object` 的一个子类：

    @register_object
    class Tensor(Object, _expr.ExprOp):
        """Tensor object, to construct, see function.Tensor"""

        def __call__(self, *indices):
           ...

对象协议是将 C++ 类型暴露给前端语言（包括 Python）的基础。TVM 实现
Python 封装的方式并不直接。在 `tvm-runtime-system`{.interpreted-text
role="ref"} 中简单介绍了这一点，感兴趣的朋友可以在 `python/tvm/_ffi/`
中查看细节。

使用 `TVM_REGISTER_*` 宏将 C++ 函数以
`tvm-runtime-system-packed-func`{.interpreted-text role="ref"}
的形式暴露给前端语言。`PackedFunc` 是 TVM 实现 C++ 和 Python
之间互操作性的另一种机制。这使得从 C++ 代码库中调用 Python
函数变得非常容易。Python 和 C++ 的语言交互接口 (FFI)
的调用之间导航，请查看 [FFI
Navigator](https://github.com/tqchen/ffi-navigator)。

每个 `Tensor` 对象有一个与之相关的 `Operation` 对象，定义在
`python/tvm/te/tensor.py`、`include/tvm/te/operation.h` 和
`src/tvm/te/operation` 子目录下。`Tensor` 是其 `Operation`
对象的输出。每个 `Operation` 对象都有 `input_tensors()`
方法，该方法返回一个输入 `Tensor` 列表。这样我们就可以跟踪 `Operation`
之间的依赖关系。

将输出张量 `C` 对应的 op 传递给 `python/tvm/te/schedule.py` 中的
`tvm.te.create_schedule()` 函数。

    s = tvm.te.create_schedule(C.op)

这个函数被映射到 `include/tvm/schedule.h` 中的 C++ 函数。

    inline Schedule create_schedule(Array<Operation> ops) {
      return Schedule(ops);
    }

`Schedule` 由 `Stage` 和输出 `Operation` 的集合组成。

`Stage` 对应一个 `Operation`。在上述 Vector Add 示例中，有两个占位符 op
和一个计算 op，所以调度 s 包含三个阶段。每个 Stage
都有关于循环嵌套结构的信息，每个循环的类型（`Parallel`、`Vectorized`、`Unrolled`），以及在下一个
`Stage` 的循环嵌套中（如果有的话）执行其计算的位置。

`Schedule` 和 `Stage` 在
`tvm/python/te/schedule.py`、`include/tvm/te/schedule.h` 和
`src/te/schedule/schedule_ops.cc` 中定义。

简单来说，上述 `create_schedule()` 函数创建的默认 schedule 调用
`tvm.build(...)`。

    target = "cuda"
    fadd = tvm.build(s, [A, B, C], target)

`tvm.build()`, defined in `python/tvm/driver/build_module.py`, takes a
schedule, input and output `Tensor`, and a target, and returns a
:py`tvm.runtime.Module`{.interpreted-text role="class"} object. A
:py`tvm.runtime.Module`{.interpreted-text role="class"} object contains
a compiled function which can be invoked with function call syntax.
定义在 `python/tvm/driver/build_module.py` 中的 `tvm.build()`，接收一个
schedule，输入和输出 `Tensor` 以及一个 target，然后返回一个
:py`tvm.runtime.Module`{.interpreted-text role="class"} 对象。一个
:py`tvm.runtime.Module`{.interpreted-text role="class"}
对象包含一个可以用函数调用语法来调用的已编译函数。

`tvm.build()` 的过程可以分为两个步骤：

-   降级，高级的、初始的循环嵌套结构被转化为最终的、底层的 IR
-   代码生成，由底层的 IR 来生成目标机器代码

降级是由 `tvm.lower()` 函数完成的，定义在 `python/tvm/build_module.py`
中。首先进行边界推断，然后创建一个初始循环嵌套结构。

    def lower(sch,
              args,
              name="default_function",
              binds=None,
              simple_mode=False):
       ...
       bounds = schedule.InferBound(sch)
       stmt = schedule.ScheduleOps(sch, bounds)
       ...

边界推断 (Bound inference)
是推断出所有循环边界和中间缓冲区大小的过程。如果你的目标是 CUDA
后端，并且使用了共享内存，那么它所需的最小尺寸就会在这里自动确定。边界推断在
`src/te/schedule/bound.cc`、`src/te/schedule/graph.cc` 和
`src/te/schedule/message_passing.cc`
中实现。更多关于边界推断的信息，请参阅
`dev-InferBound-Pass`{.interpreted-text role="ref"}。

`stmt` 是 `ScheduleOps()` 的输出，代表一个初始的循环嵌套结构。如果
schedule 已经应用了 `reorder` 或 `split`
原语，则初始循环嵌套已经反映了这些变化。`ScheduleOps()` 在
`src/te/schedule/schedule_ops.cc` 中定义。

接下来，对 `stmt` 在 `src/tir/pass` 子目录下进行降级处理。例如，如果
`vectorize` 或 `unroll` 原语已经应用于 schedule
了，那么它们将被应用于以下步骤：

    ...
    stmt = ir_pass.VectorizeLoop(stmt)
    ...
    stmt = ir_pass.UnrollLoop(
        stmt,
        cfg.auto_unroll_max_step,
        cfg.auto_unroll_max_depth,
        cfg.auto_unroll_max_extent,
        cfg.unroll_explicit)
    ...

降级完成后，`build()` 函数从降级的函数中生成目标机器代码。如果目标是
x86，这段代码会包含 SSE 或 AVX 指令；如果目标是 CUDA，则包含 PTX
指令。除了目标专用机器代码外，TVM
还会生成负责内存管理、内核启动等的宿主机代码。

代码生成是由 `build_module()` 函数完成的，定义在
`python/tvm/target/codegen.py`。在 C++ 端，代码生成是在
`src/target/codegen` 子目录下实现的。`build_module()` 这个 Python
函数将进入下面 `src/target/codegen/codegen.cc` 中的 `Build()` 函数。

`Build()` 函数在 `PackedFunc`
注册表中查找给定目标的代码生成器，并调用找到的函数。例如，`codegen.build_cuda`
函数在 `src/codegen/build_cuda_on.cc` 中注册，如下所示：

    TVM_REGISTER_GLOBAL("codegen.build_cuda")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
        *rv = BuildCUDA(args[0]);
      });

上述 `BuildCUDA()` 使用 `src/codegen/codegen_cuda.cc` 中定义的
`CodeGenCUDA` 类从降级的 IR 中生成 CUDA 内核源代码，并使用 NVRTC
编译内核。如果目标是使用 LLVM 的后端，包括 x86、ARM、NVPTX 和
AMDGPU，代码生成主要由定义在 `src/codegen/llvm/codegen_llvm.cc` 中的
`CodeGenLLVM` 类完成。`CodeGenLLVM` 将 TVM IR 翻译成 LLVM IR，运行一些
LLVM 优化，并生成目标机器代码。

`src/codegen/codegen.cc` 中的 `Build()` 函数返回一个 `runtime::Module`
对象，该对象在 `include/tvm/runtime/module.h` 和 `src/runtime/module.cc`
中定义。`Module` 对象是底层目标特定的 `ModuleNode`
对象的容器。每个后端都实现了一个 `ModuleNode` 的子类，以添加目标特定
runtime API 调用。例如，CUDA 后端在 `src/runtime/cuda/cuda_module.cc`
中实现了 `CUDAModuleNode` 类，它管理着 CUDA 驱动 API。上述 `BuildCUDA()`
函数用 `runtime::Module` 包装了 `CUDAModuleNode` 并将其返回到 Python
端。LLVM 后端在 `src/codegen/llvm/llvm_module.cc` 中实现了
`LLVMModuleNode`，它负责处理编译代码的 JIT 执行。`ModuleNode`
的其他子类可以在与每个后端对应的 `src/runtime` 的子目录下找到。

返回的模块可以被认为是已编译的函数和设备 API 的结合，可以在 TVM 的
NDArray 对象上被调用。

    dev = tvm.device(target, 0)
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd(a, b, c)
    output = c.numpy()

在底层，TVM 自动分配设备内存并管理内存传输。为此，每个后端都需要继承在
`include/tvm/runtime/device_api.h` 中定义的 DeviceAPI
类，并覆盖内存管理方法以使用特定于设备的 API。例如，CUDA 后端在
`src/runtime/cuda/cuda_device_api.cc` 中实现 `CUDADeviceAPI` 以使用
`cudaMalloc`、`cudaMemcpy` 等。

首次使用 `fadd(a, b, c)` 调用已编译的模块时，会调用 `ModuleNode` 的
`GetFunction()` 方法来获取可用于内核调用的 `PackedFunc`。例如，在
`src/runtime/cuda/cuda_module.cc` 中，CUDA 后端实现了
`CUDAModuleNode::GetFunction()`，如下所示：

    PackedFunc CUDAModuleNode::GetFunction(
          const std::string& name,
          const std::shared_ptr<ModuleNode>& sptr_to_self) {
      auto it = fmap_.find(name);
      const FunctionInfo& info = it->second;
      CUDAWrappedFunc f;
      f.Init(this, sptr_to_self, name, info.arg_types.size(), info.launch_param_tags);
      return PackFuncVoidAddr(f, info.arg_types);
    }

`PackedFunc` 的重载 `operator()` 将被调用，进而调用
`src/runtime/cuda/cuda_module.cc` 中 `CUDAWrappedFunc` 的 `operator()`
，最后实现 `cuLaunchKernel` 驱动程序的调用：

    class CUDAWrappedFunc {
     public:
      void Init(...)
      ...
      void operator()(TVMArgs args,
                      TVMRetValue* rv,
                      void** void_args) const {
        int device_id;
        CUDA_CALL(cudaGetDevice(&device_id));
        if (fcache_[device_id] == nullptr) {
          fcache_[device_id] = m_->GetFunc(device_id, func_name_);
        }
        CUstream strm = static_cast<CUstream>(CUDAThreadEntry::ThreadLocal()->stream);
        ThreadWorkLoad wl = launch_param_config_.Extract(args);
        CUresult result = cuLaunchKernel(
            fcache_[device_id],
            wl.grid_dim(0),
            wl.grid_dim(1),
            wl.grid_dim(2),
            wl.block_dim(0),
            wl.block_dim(1),
            wl.block_dim(2),
            0, strm, void_args, 0);
      }
    };

以上就是 TVM 编译和执行函数的相关简介。虽然没有涉及到 TOPI 或 Relay
的详细介绍，但所有神经网络算子的编译过程都和上述过程类似。欢迎各位开发者深入研究代码库其他部分的细节。
