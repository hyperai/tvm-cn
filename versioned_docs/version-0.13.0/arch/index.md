---
title: 设计与架构
---

# 设计与架构

本文档适用于想要了解 TVM 架构和/或积极开发项目的开发者。本文档组织结构如下：

* [编译流程示例](#example-compilation-flow) 概述了 TVM 将模型的高级描述转换为可部署模块所采取的步骤。
* [逻辑架构组件](#logical-architecture-components) 部分描述了逻辑组件。后面的部分是针对每个逻辑组件的具体指南，按组件的名称编排。
* [设备/ Target 交互](https://tvm.apache.org/docs/arch/device_target_interactions.html#tvm-target-specific-overview) 文档描述了 TVM 如何与所有受支持的物理设备，以及代码生成的 target 进行交互。
* 查看 [开发者操作指南](/docs/dev/how_to) 获取实用开发技巧。

本指南提供了架构的一些补充视图。首先研究端到端的编译流程，并讨论关键的数据结构和转换。这种基于 runtime 的视图侧重于运行编译器时每个组件的交互。接下来研究代码库的逻辑模块及其关系。这部分提供了设计的静态总体视图。

## 编译流程示例

本指南研究编译器中的编译流程示例，下图显示了流程。在高层次，它包含以下步骤：

* 导入：前端组件将模型引入到 IRModule 中，它包含了内部表示模型的函数集合。
* 转换：编译器将 IRModule 转换为功能与之等效或近似等效（例如在量化的情况下）的 IRModule。许多转换与 target（后端）无关，并且允许 target 配置转换 pipeline。
* Target 转换：编译器将 IRModule 转换（codegen）为指定 target 的可执行格式。target 的转换结果被封装为 *runtime.Module*，可以在 runtime 环境中导出、加载和执行。
* Runtime 执行：用户加载 *runtime.Module*，并在支持的 runtime 环境中运行编译好的函数。

![/img/docs/tlc-pack/web-data/main/images/design/tvm_dyn_workflow.svg](/img/docs/tlc-pack/web-data/main/images/design/tvm_dyn_workflow.svg)

### 关键数据结构

设计和理解复杂系统的最佳方法之一，就是识别关键数据结构和操作（转换）这些数据结构的 API。识别了关键数据结构后，就可以将系统分解为逻辑组件，这些逻辑组件定义了关键数据结构的集合，或是数据结构之间的转换。

**IRModule** 是整个堆栈中使用的主要数据结构。一个 IRModule（intermediate representation module）包含一组函数。目前支持两种主要的功能变体（variant）：

* **relay::Function** 是一种高级功能程序表示。一个 relay.Function 通常对应一个端到端的模型。可将 relay.Function 视为额外支持控制流、递归和复杂数据结构的计算图。
* **tir::PrimFunc** 是一种底层程序表示，包含循环嵌套选择、多维加载/存储、线程和向量/张量指令的元素。通常用于表示算子程序，这个程序在模型中执行一个（可融合的）层。 在编译期间，Relay 函数可降级为多个 tir::PrimFunc 函数和一个调用这些 tir::PrimFunc 函数的顶层函数。

### 转换

前面介绍了关键数据结构，接下来讲转换。转换的目的有：

* 优化：将程序转换为等效，甚至更优的版本。
* 降级：将程序转换为更接近 target 的较低级别表示。 **relay/transform** 包含一组优化模型的 pass。优化包括常见的程序优化（例如常量折叠和死码消除），以及特定于张量计算的 pass（例如布局转换和 scale 因子折叠）。

在 Relay 优化流程的后期，运行 pass（FuseOps），将端到端函数（例如 MobileNet）分解为子功能（例如 conv2d-relu）段。这个过程帮助将原始问题分为两个子问题：

* 所有子函数的编译和优化。
* 整体执行结构：对生成的子函数进行一系列调用，执行整个模型。 使用下层 tir 阶段来编译和优化每个子函数。对于特定的 targets，也可以直接进入 target 转换阶段，使用外部代码生成器。

有几种不同的方法（在 relay/backend 目录）来处理对整体执行问题的调用。对于具有已知 shape 且没有控制流的简单模型，可以降级为图执行器，这个图执行器存储计算图中的执行结构。我们还支持用于动态执行的虚拟机后端。

最后，我们计划支持 ahead-of-time 编译，它将高级执行结构编译成可执行和生成的原始函数。所有这些执行模式都被统一的 **runtime.Module** 接口封装，指南的后半部分将进行讨论。

**tir/transform** 包含 TIR 级函数的转换过程。许多 tir passes 的目的是降级。例如，有些 pass 将多维访问展平为一维指针访问，将内联函数扩展至特定于 target 的函数，以及将函数入口修饰为满足 runtime 调用约定。当然，也有一些 pass 的目的是为了优化，例如访问索引简化和死码消除。

LLVM、CUDA C 和其他 target 编译器都可以在 target 阶段处理许多底层优化。因此，我们将底层优化（如寄存器分配）留给下游编译器，只关注它们未涵盖的优化。

### 搜索空间和基于学习的转换

到目前为止，我们介绍的转换 pass 都是确定且遵循一定规则的。 TVM 堆栈的设计目标之一是支持不同硬件平台的高性能代码优化。因此要研究尽可能多的优化选择，包括但不限于多维张量访问、循环分块行为、特殊加速器内存层次结构和线程。

定义一个要做出所有选择的启发式方法很难。因此，我们采用搜索和基于学习的方法。

首先定义一组用来转换程序的操作。示例操作包括循环转换、内联、向量化，这些操作称为**调度原语**。调度原语的集合定义了可用于程序的优化的搜索空间。

接下来，系统搜索不同的可能调度序列，选择最佳调度组合。搜索过程通常由机器学习算法指导。

搜索完成后，可以记录（可能融合的）算子的最佳调度顺序。然后编译器可以查找最佳调度序列，并将其应用于程序。注意，这个调度应用阶段与基于规则的转换**完全一样**，能够与传统 pass 共享相同的接口。

使用基于搜索的优化来处理初始 tir 函数生成问题。模块的这部分称为 AutoTVM（auto_scheduler）。随着 TVM 堆栈开发的深入，基于学习的转换将扩展到更多领域。

### Target 转换

target 转换阶段将 IRModule 转换为相应 target 的可执行格式。对于 x86 和 ARM 等后端，使用 LLVM IRBuilder 来构建内存中的 LLVM IR。还可以生成源代码级语言，例如 CUDA C 和 OpenCL。最后，支持通过外部代码生成器将 Relay 函数（子图）直接转换为特定 target 。

重要的是，最终代码生成阶段要尽可能轻量级。绝大多数的转换和降级要在 target 转换阶段之前进行。

我们还提供了一个 Target 结构来指定编译 target。target 转换阶段之前的转换也可能受到 target 的影响，例如，target 的向量长度会改变向量化行为。

### Runtime 执行

TVM runtime 的主要目标是提供一个最小的 API，从而能以选择的语言（包括 Python、C++、Rust、Go、Java 和 JavaScript）加载和执行编译好的工件。以下代码片段展示了一个 Python 示例：

``` python
import tvm
# Python 中 runtime 执行程序示例，带有类型注释
mod: tvm.runtime.Module = tvm.runtime.load_module("compiled_artifact.so")
arr: tvm.runtime.NDArray = tvm.nd.array([1, 2, 3], device=tvm.cuda(0))
fun: tvm.runtime.PackedFunc = mod["addone"]
fun(a)
print(a.numpy())
```

`tvm.runtime.Module` 封装了编译的结果。runtime.Module 包含一个 GetFunction 方法，用于按名称获取 PackedFuncs。

`tvm.runtime.PackedFunc` 是一种为各种构造函数消解类型的函数接口。runtime.PackedFunc 的参数和返回值的类型如下：POD 类型（int, float）、string、runtime.PackedFunc、runtime.Module、runtime.NDArray 和 runtime.Object 的其他子类。

`tvm.runtime.Module` 和 `tvm.runtime.PackedFunc` 是模块化 runtime 的强大机制。例如，要在 CUDA 上获取上述 *addone* 函数，可以用 LLVM 生成主机端代码来计算启动参数（例如线程组的大小），然后用 CUDA 驱动程序 API 支持的 CUDAModule 调用另一个 PackedFunc。OpenCL 内核也有相同的机制。

以上示例只处理了一个简单的 *addone* 函数。下面的代码片段给出了用相同接口执行端到端模型的示例：

``` python
import tvm
# python 中 runtime 执行程序的示例，带有类型注释
factory: tvm.runtime.Module = tvm.runtime.load_module("resnet18.so")
# 在 cuda(0) 上为 resnet18 创建一个有状态的图执行模块
gmod: tvm.runtime.Module = factory["resnet18"](tvm.cuda(0))
data: tvm.runtime.NDArray = get_input_data()
# 设置输入
gmod["set_input"](0, data)
# 执行模型
gmod["run"]()
# 得到输出
result = gmod["get_output"](0).numpy()
```

主要的结论是 runtime.Module 和 runtime.PackedFunc 可以封装算子级别的程序（例如 addone），以及端到端模型。

### 总结与讨论

综上所述，编译流程中的关键数据结构有：

* IRModule：包含 relay.Function 和 tir.PrimFunc
* runtime.Module：包含 runtime.PackedFunc

编译基本是在进行关键数据结构之间的转换。

* relay/transform 和 tir/transform 是确定性的基于规则的转换
* auto_scheduler 和 autotvm 包含基于搜索的转换

最后，编译流程示例只是 TVM 堆栈的一个典型用例。将这些关键数据结构和转换提供给 Python 和 C++ API。然后，就可以像使用 numpy 一样使用 TVM，只不过关注的数据结构从 numpy.ndarray 改为 tvm.IRModule。以下是一些用例的示例：

* 用 Python API 直接构建 IRModule。
* 编写一组自定义转换（例如自定义量化）。
* 用 TVM 的 Python API 直接操作 IR。

## 逻辑架构组件

![/img/docs/tlc-pack/web-data/main/images/design/tvm_static_overview.svg](/img/docs/tlc-pack/web-data/main/images/design/tvm_static_overview.svg)

*TVM Architecture Diagram*[¶](#logical-architecture-components)

上图展示了项目中的主要逻辑组件。阅读以下部分，获取更多有关组件及其关系的信息。

## tvm/support

support 模块包含基础架构最常用的程序，例如通用 arena 分配器（arena allocator）、套接字（socket）和日志（logging）。

## tvm/runtime

runtime 是 TVM 堆栈的基础，它提供了加载和执行已编译工件的机制。runtime 定义了一组稳定的标准 C API，与 Python 和 Rust 等前端语言交互。

在 TVM runtime 中，*runtime::Object* 是除 *runtime::PackedFunc* 之外的主要数据结构之一。它是一个具有类型索引的引用计数基类，这个类型索引用来支持 runtime 类型检查和向下转换。

对象系统允许开发者向 runtime 引入新的数据结构，例如 Array、Map 和新的 IR 数据结构。

除了部署用例，编译器本身也大量使用 TVM 的 runtime 机制。所有 IR 数据结构都是 *runtime::Object* 的子类，因此可以从 Python 前端直接访问和操作它们。我们使用 PackedFunc 机制向前端公开各种 API。

runtime 子目录（例如 runtime/opencl）定义了 runtime 对不同硬件后端的支持。这些特定于硬件的 runtime 模块，为设备内存分配和设备功能序列化定义了 API。

*runtime/rpc* 实现了对 PackedFunc 的 RPC 支持，可以用 RPC 机制将交叉编译的库发送到远程设备，并对执行性能进行 benchmark 测试。rpc 基础架构支持从各种硬件后端收集数据，进行基于学习的优化。

* [TVM Runtime 系统](arch/runtimes)
* [特定 Runtime 信息](arch/runtimes#runtime-specific-information)
* [调试器](arch/debugger)
* [向 TVM 中添加虚拟机：Relay 虚拟机](arch/virtual_machine)
* [模块序列化简介](arch/introduction_to_module_serialization)
* [设备/Target 交互](arch/device_target_interactions)

## tvm/node

节点模块在 *runtime::Object* 之上，为 IR 数据结构添加了额外的功能。主要特征包括反射、序列化、结构等效和散列。

node 模块使得可以通过 Python 中的名称，直接访问 TVM IRNode 的任何字段。

``` python
x = tvm.tir.Var("x", "int32")
y = tvm.tir.Add(x, x)
# a 和 b 是 tir.Add 节点的字段
# 可以直接使用字段名来访问 IR 结构
assert y.a == x
```

还可以将任意 IR 节点序列化为 JSON 格式，然后将它们加载回来。保存/存储和检查 IR 节点的能力，为更容易访问编译器提供了基础。

## tvm/ir

*tvm/ir* 文件夹包含了所有 IR 函数变体的统一数据结构和接口。 *tvm/ir* 中的组件由 *tvm/relay* 和 *tvm/tir* 共享，主要包括

* IRModule
* Type
* PassContext and Pass
* Op

不同的函数变体（例如，relay.Function 和 tir.PrimFunc）可以在一个 IRModule 中共存。虽然这些变体的内容表示可能不同，但它们使用相同的数据结构来表示类型。

因此，可以用相同的数据结构来表示这些变体的函数（类型）签名。一旦明确定义了调用约定，统一的类型系统就可以用一个函数变体调用另一个函数。这为未来的跨函数变体的优化奠定了基础。

我们还提供了一个统一的 PassContext，用于配置 pass 行为，以及通用的复合 pass 来执行 pass pipeline。以下代码片段给出了 PassContext 配置的示例：

``` python
# 配置 tir.UnrollLoop pass 的行为
with tvm.transform.PassContext(config={"tir.UnrollLoop": { "auto_max_step": 10 }}):
    # 受 pass 上下文影响的代码
```

Op 是表示所有系统定义的原始算子/内联函数的通用类。开发者可以向系统注册新的 Ops，以及它们的额外属性（例如 Op 是否为元素）。

* [Pass Infrastructure](arch/pass_infra)

## tvm/target

target 模块包含将 IRModule 转换为 target runtime.Module 的所有代码生成器。它还提供了一个描述 target 的通用 Target 类。

通过查询 target 中的属性信息和注册到每个 target id（cuda、opencl）的内置信息，可以根据 target 定制编译 pipeline。

* [设备/Target 交互](arch/device_target_interactions)

## tvm/tir

TIR 包含低级程序表示的定义。用 *tir::PrimFunc* 来表示可通过 TIR pass 转换的函数。除了 IR 数据结构外，tir 模块还通过通用 Op 注册表，定义了一组内置内联函数及其属性，以及 *tir/transform* 中的转换 pass。

## tvm/arith

该模块与 TIR 密切相关，低级代码生成的关键问题之一是分析索引的算术属性（arithmetic properties）——正性（positiveness）、变量界限和描述迭代器空间的整数集。arith 模块提供了一组进行（主要是整数）分析的工具。TIR pass 可以用这些分析来简化和优化代码。

## tvm/te

te（tensor expression）代表「张量表达式」，这是一个特定领域的语言模块，它允许通过编写张量表达式来快速构造 *tir::PrimFunc* 变体（variant）。

重要的是，张量表达式本身并不是一个可以存储到 IRModule 中的自包含函数（self-contained function）。相反，它是 IR 的一个片段，可以拼接起来构建一个 IRModule。

*te/schedule* 提供了一组调度原语，来控制正在生成的函数。未来，可能会将其中一些调度组件引入到 *tir::PrimFunc* 本身。

* [InferBound Pass](arch/inferbound)
* [混合前端开发者指南](arch/hybrid_script)

## tvm/topi

虽然可以通过 TIR 或张量表达式（TE）为每个用例直接构造算子，但这样做很乏味。 *topi*（张量算子清单）提供了一组 numpy 定义的预定义算子（在 TE 或 TIR 中），并可在常见的深度学习任务中找到。还提供了一组常见的调度模板，来获得跨不同 target 平台的性能实现。

## tvm/relay

Relay 是用于表示完整模型的高级功能 IR。*relay.transform* 中定义了各种优化。 Relay 编译器定义了多种方言，每种方言都旨在支持特定的优化风格。主要包括 QNN（用于导入预量化模型）、VM（用于降级到动态虚拟机）、内存（用于内存优化）。

* [Relay IR 简介](arch/relay_intro)
* [Relay 算子策略](arch/relay_op_strategy)
* [转换布局 Pass](arch/convert_layout)

## tvm/autotvm

AutoTVM 和 AutoScheduler 都是基于程序优化的自动搜索组件。它们在迅速发展中，主要包括：

* cost 模型和特征提取。
* 存储 cost 模型构建的程序 benchmark 结果的记录格式。
* 一组程序转换的搜索策略。

自动化程序优化仍是热点研究领域。因此，我们采用了模块化的设计思路，以便研究人员通过 Python binding 来快速修改组件或应用自己的算法，也可以自定义搜索，以及从 Python binding 插入自己的算法。

* [benchmark 性能日志格式](arch/benchmark)

## 前端

前端将来自不同框架的模型引入到 TVM 堆栈中。`tvm.relay.frontend` 是模型引入 API 的命名空间。

* [TensorFlow 前端](arch/tensorflow)

## 安全

* [安全指南](arch/security)

## microTVM

* [microTVM 设计文档](arch/microtvm_design)
* [microTVM 项目 API](arch/microtvm_project_api)
* [模型库格式](arch/model_library_format)
