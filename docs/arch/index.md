
# 设计与架构（Design and Architecture）

本文档面向希望理解 Apache TVM 架构和/或希望参与项目开发的开发者。内容结构如下：

- **整体流程（Overall Flow）**：概述 TVM 如何将模型的高层描述转换为可部署模块。建议首先阅读本节。
- **TVM 栈中关键组件简介**：可以同时参考 *TensorIR 深度解析* 与 *Relax 深度解析*，以进一步理解 TVM 中两个主要 IR 系统。

本指南提供了架构的两种互补视角：
1. 从端到端编译流程的角度，介绍关键数据结构和编译阶段的转换。
2. 从代码结构角度审视模块关系，给出整体静态架构视图。

## 整体流程（Overall Flow）

以下流程图展示了典型的编译过程。主要包含以下步骤：

- **模型创建**：构建需要优化和编译的 IRModule，其中包含表示模型的一组函数。可通过 NNModule、TVMScript 或从 Relax 前端加载模型来创建。
- **转换（Transformation）**：将 IRModule 转换为功能等价或近似等价（如量化场景） 的 IRModule。大多数转换与目标设备无关。
- **目标转换（Target Translation）**：将 IRModule 转换为对应目标可执行格式，并封装为 `runtime.Module`，可保存、加载并在目标设备运行。
- **运行时执行（Runtime Execution）**：用户加载生成的 `runtime.Module` 并在支持的运行环境中执行。

![TVM Overall Flow](https://raw.githubusercontent.com/tlc-pack/web-data/main/images/design/tvm_overall_flow.svg)

### 关键数据结构

理解复杂系统的最佳方式之一是识别其关键数据结构及其转换方式。TVM 中最核心的数据结构是：

 **IRModule**：中间表示模块，包含多个函数。主要函数形式：
 
  - **relax::Function**：高层图级 IR，支持计算图结构、控制流和复杂数据结构。
  - **tir::PrimFunc**：底层程序 IR，包含循环结构、多维 load/store、线程绑定、向量化等，用于表达算子计算。

在优化过程中：

- Relax 运算会被降低到 `tir::PrimFunc` 或 `PackedFunc`。
- Relax 调用会被降低为底层函数调用指令（如 `R.call_tir`）。

### 转换（Transformations）

转换的主要目的包括：

- **优化（Optimization）**：得到效果等价但性能更优的程序。
- **降级（Lowering）**：将程序转换为更接近硬件执行的表示。

#### Relax 转换

Relax 转换包含图级优化，例如：
- 常量折叠
- 死代码消除
- 库调度与后端特化优化

#### TIR 转换

TIR 转换包含：

- **TensorIR 调度**：通过调度生成适合目标硬件的高性能代码。GPU 上调度尤为关键。
- **降级 Pass**：例如将多维访问展平为指针访问、处理设备特定 intrinsic、适配运行时 ABI。

大部分底层编译优化（如寄存器分配）由下游编译器（如 LLVM）处理。

#### 跨层转换

由于 IRModule 同时包含 Relax 和 TIR 函数，跨层转换可对两者共同进行整体优化，例如：

- `relax.LegalizeOps`：将 Relax 运算符降低到 TIR
- `relax.FuseOps`/`relax.FuseTIR`：跨算子融合，自动分析最优融合策略

### 目标转换（Target Translation）

目标翻译阶段将 IRModule 转换为对应目标的可执行格式。对于 x86 和 ARM 等后端，我们使用 LLVM IRBuilder 构建内存中的 LLVM IR。我们也可以生成诸如 CUDA C 和 OpenCL 等源代码级语言。
最后，我们支持通过外部代码生成器，将 Relax 函数（子图）直接翻译到特定目标。需要强调的是，最终的代码生成阶段应尽可能轻量化。绝大多数的变换与 Lowering 应该在进入目标翻译阶段之前完成。

我们同时提供了 Target 结构来指定编译目标。目标翻译阶段之前的变换也可能受到该目标的影响 ——例如，目标的向量长度会影响向量化策略。

### 运行时执行（Runtime Execution）

TVM 运行时提供统一 API，可用于 Python/C++/Rust/Go/Java/JS。示例：

```python
mod = tvm.runtime.load_module("compiled_artifact.so")
arr = tvm.runtime.tensor([1, 2, 3], device=tvm.cuda(0))
fun = mod["addone"]
fun(arr)
print(arr.numpy())
```

:py:class:tvm.runtime.Module 封装了编译的结果。一个 runtime.Module 包含一个 GetFunction 方法，用于通过名称获取对应的 PackedFunc。

:py:class:tvm.runtime.PackedFunc 是一个**类型擦除（type-erased）**的函数接口，用于调用已生成的函数。一个 runtime.PackedFunc 可以接收参数并返回以下类型的值：基本类型（POD 类型，如 int、float）、string、runtime.PackedFunc、runtime.Module、runtime.Tensor，以及 runtime.Object 的其他子类。

:py:class:tvm.runtime.Module和 :py:class:tvm.runtime.PackedFunc 是将运行时进行模块化的强大机制。例如，为了在 CUDA 上获得上述 addone 函数，我们可以使用 LLVM 生成主机端代码来计算内核启动参数（例如线程组大小），然后从由 CUDA 驱动 API 支持的 CUDAModule 中调用另一个 PackedFunc。
相同的机制也适用于 OpenCL 内核。

上面的示例只涉及一个简单的 addone 函数。下面的代码片段给出了使用相同接口执行端到端模型的示例：

```python
import tvm
# Example runtime execution program in python, with types annotated
factory: tvm.runtime.Module = tvm.runtime.load_module("resnet18.so")
# Create a stateful graph execution module for resnet18 on cuda(0)
gmod: tvm.runtime.Module = factory["resnet18"](tvm.cuda(0))
data: tvm.runtime.Tensor = get_input_data()
# set input
gmod["set_input"](0, data)
# execute the model
gmod["run"]()
# get the output
result = gmod["get_output"](0).numpy()
```
主要的优点是运行时间。模块和运行时。packkedfunc足以封装运算符级程序（如addone）以及端到端的模型。

### 总结与讨论（Summary and Discussions）

总的来说，编译流程中的关键数据结构是：

- **IRModule**：包含 `relax.Function` 和 `tir.PrimFunc`
- **runtime.Module**：包含 `runtime.PackedFunc`

编译的大部分过程是这些关键数据结构之间的转换。

- `relax/transform` 和 `tir/transform` 是**基于规则的确定性**转换
- `meta-schedule` 包含**基于搜索**的转换

最后，前面所展示的编译流程示例只是 TVM 栈的一个典型使用场景。我们将这些关键数据结构和转换暴露给 Python 和 C++ API。因此，你可以像使用 numpy 一样使用 TVM，只是把关注点的数据结构从 `numpy.ndarray` 换成了 `tvm.IRModule`。  
下面是一些示例使用方式：

- 使用 Python API **直接构造 IRModule**
- 组合一组自定义的转换（例如自定义量化流程）
- 使用 TVM 的 Python API **直接操作 IR**

## tvm/support

support 模块包含基础设施中最常用的工具，如通用内存池（arena allocator）、socket 和日志系统。

## tvm/runtime

runtime 是 TVM 栈的基础。它提供了加载和执行编译产物的机制，并定义了一组稳定的 C API，用于与 Python、Rust 等前端语言交互。

`runtime::Object` 是 TVM runtime 中的主要数据结构之一（另一个是 `ffi::Function`）。它是一个带引用计数和类型索引的基类，用于支持运行时类型检查和向下转型。对象系统允许开发者向 runtime 中添加新的数据结构，例如 Array、Map 和新的 IR 数据结构。

除了部署用途外，编译器本身也大量使用 TVM 的 runtime 机制。所有 IR 数据结构都是 `runtime::Object` 的子类，因此可以从 Python 前端直接访问和操作它们。我们使用 PackedFunc 机制将各种 API 暴露到前端。

对不同硬件后端的 runtime 支持定义在 runtime 目录的子目录中（如 `runtime/opencl`）。这些硬件相关 runtime 模块定义设备内存分配和函数序列化的 API。

`runtime/rpc` 提供了用于 PackedFunc 的 RPC 支持。我们可以使用 RPC 将交叉编译后的库发送到远程设备上并测试其执行性能。RPC 基础设施使得在多种硬件后端上进行数据收集成为可能，这在基于学习的优化流程中非常重要。

## tvm/node

node 模块在 `runtime::Object` 之上为 IR 数据结构添加了额外功能，主要包括：

- 反射
- 序列化
- 结构等价检查
- 哈希

得益于 node 模块，我们可以在 Python 中通过字段名称直接访问 TVM IRNode 的字段：

```python
x = tvm.tir.Var("x", "int32")
y = tvm.tir.Add(x, x)
assert y.a == x
```

我们也可以将任意 IR 节点序列化为 JSON 格式，并加载回内存。对 IR 节点进行保存/查看的能力，使编译器更加易于理解和可调试。

## tvm/ir

`tvm/ir` 文件夹包含跨所有 IR 函数变体共享的统一数据结构和接口。其中包括：

- IRModule
- Type
- PassContext 和 Pass
- Op

不同类型的函数（如 relax.Function 和 tir.PrimFunc）可以共存于一个 IRModule 中。尽管它们内部结构不同，但它们使用统一的数据结构描述类型。因此，我们使用相同的数据结构来表示这些函数的类型签名。  
统一的类型系统允许不同函数变体通过明确的调用约定相互调用，这为跨层优化提供了可能。

我们还提供了统一的 PassContext，用于配置变换行为，并且提供组合式 Pass 管线执行方式，例如：

```python
with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 10}}):
    ...
```

Op 用于表示系统中定义的所有原始运算/内置指令。开发者可以向系统注册新的 Op 及其属性（如是否为逐元素计算）。

## tvm/target

target 模块包含所有将 IRModule 转换为目标 runtime.Module 的代码生成器，同时提供用于描述目标设备的 `Target` 类。

编译管线可以根据 target 提供的属性和已注册的目标信息（如 cuda、opencl）进行调整。

## tvm/relax

Relax 是用于表示模型计算图的高层 IR。各种优化定义在 `relax.transform` 中。  
Relax 通常与 IRModule 中的 TensorIR 函数紧密协同，大部分优化会同时作用于 Relax 和 TIR 函数。

## tvm/tir

TIR 定义了底层程序表示方式。我们使用 `tir::PrimFunc` 表示可被 TIR passes 转换的函数。  
tir 模块还包括：

- `tir/schedule` 中的 schedule 原语
- `tir/tensor_intrin` 中的内置张量指令
- `tir/analysis` 中的分析 passes
- `tir/transform` 中的优化与降低 passes

## tvm/arith

负责张量索引表达式的算术分析（边界、正性、迭代空间等），并辅助优化 TIR。

## tvm/te 和 tvm/topi

TE 是张量表达式 DSL，一个 TE 本身不能直接存入 IRModule，需要通过 `te.create_prim_func` 转换为 `tir::PrimFunc`。  
[topi] 提供常用深度学习算子的预定义实现。

## tvm/meta_schedule

MetaSchedule 是自动搜索式 schedule 优化系统，用于替代 AutoTVM 和 AutoScheduler，仅支持静态形状模型。

## tvm/dlight

DLight 提供简单、少调参、高鲁棒性的 TIR 调度策略，支持动态形状，并在无法应用时自动回退。
