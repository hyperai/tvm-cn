---
title: microTVM 设计文档
sidebar_position: 250
---

# microTVM 设计文档

## 背景

TVM 是一个模型部署框架，它在传统操作系统上的各种模型中性能较好。TVM 的分层编译方法是针对裸机设备的自然扩展。虽然大多数编译流程无需更改这类设备上的概念验证 (proof-of-concept, POC) 的实现，但 runtime 不能依赖于：

* **虚拟内存**，以及任何系统提供的 `malloc`。此外，裸机设备的内存通常非常有限（以 KB 为单位）。正因如此，这类平台的库在使用内存时要更加谨慎，并且在不使用时释放内存。
* 传统的操作系统抽象，例如 **文件**，**库** 和  **内核函数**。一些项目支持这些，但它们不是标准的。
* 支持除 **C** 外的编程语言。

这类更改需要不同于传统的操作系统上的 TVM C++ runtime 的方法。

## 典型用途

本节讨论对「典型」microTVM 用例的看法。所有实现此典型用例的组件都很灵活，但这种统一的看法有助于激发每个部分的设计。

![https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_workflow.svg](https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_workflow.svg)

该过程的各个部分描述如下：

1. **模型导入**。用户导入已有模型，或向 TVM 描述新模型，生成 *Relay 模块*。
2. **模型转换**。用户可以对模型应用变换，例如量化。每次转换后，用户仍保留 Relay 模块。
3. **编译**（调度和代码生成）。TVM 通过为每个 Relay 算子指定 schedule 和 schedule 配置，将每个算子实现到 Tensor IR 中。然后，为每个算子生成代码（C 源代码或编译对象）。
4. **集成**。将生成的代码与 TVM C Runtime 库一起集成到用户提供的二进制项目中。在某些情况下（例如当项目跨多个 SoC/开发单板标准化时），此过程将会自动处理。
5. **部署**。项目已构建，剩余的固件二进制文件将烧录到设备上。模型推理由 TVM 用设备上的 RPC 服务器驱动，或者用图执行器在设备上驱动。

## 设计目标

microTVM 旨在实现以下设计目标：

1. **可移植代码**。microTVM 可将所有 Relay 模型，转换为仅用 C 标准库就可以编译的 C 代码。
2. **最小开销**。microTVM 生成特定于 target 的高度优化代码。应该避免 runtime 尽可能多的开销。
3. **易懂的代码**。microTVM 将 C 源代码视为一流的输出机制，以便固件工程师更容易理解和调整。

## 概述

microTVM 要在 TVM 编译器堆栈的所有级别上进行更改。以下小节列举了高级别的变化，后续部分将更详细地讨论这些细节。

### 对 Target 平台建模

TVM 基于搜索的优化方法使其避免了对 targets 进行系统级建模，从而支持实验结果。然而，有一些建模是必要的，它们可以确保 TVM 比较的是同类搜索结果，并避免在搜索过程中，由于为 target 编译无效代码，而浪费时间。

microTVM 对 target 的这些部分进行建模：

* 使用的 CPU，通过 `-mcpu` 和 `-march` target 标志。
* 加速器的存在与否，通过 target 的设备组件（目前只能表示加速器的缺失，但这种机制有待进行更好地扩展）。

microTVM 未来要对 target 的这些部分进行建模：

* 内存，建模为一组不相交的内存空间，每个空间都有一个标签和大小，以及预取/刷新行为。一些内存会与加速器共享空间。
* Target runtime 配置（即时钟树配置、时钟速度等）。它仅用于 AutoTVM schedule 密钥，不用于任何其他用途。

目前，TVM 不建模的部分：

* 缓存的大小、类型或关系，除预取或缓存刷新外。

### microTVM 的 TVM Target

编译过程的中心数据结构是 `tvm::target::Target` 类。 TVM 用 Target 来决定启用哪些 TIR schedules，以及如何配置代码生成器。Target 类还应该唯一地标识为特定算子生成的代码，因为自动调优日志用它来对测试的性能进行排名（参阅未来工作）。

Targets 当前表示为字符串，其结构类似于命令行参数。Targets 示例如下所示：

``` plain
c -keys=arm_cpu -mcpu=cortex-m7 -model=stm32f746xx
```

microTVM 的相关部分是：

* 代码生成器（`llvm` 或 `c`）
* `-mcpu=cortex-m7`：TOPI 用来启用 Cortex-M schedules，并且，当选择 C 源代码生成器时，将其作为注释包含在输出中，从而有利于识别代码，并配置下游 C 编译器。

### microTVM 的 Runtime 和执行器配置

使用 microTVM 时，会用 C Runtime（`Runtime('crt')`）很重要，它是最适合在微型设备上运行的 Runtime，而非更动态的 C++ Runtime。此外，还有两个执行器可以与 C Runtime 结合使用：

* `Executor("aot")` - Ahead of Time (AOT) 执行器将网络预编译成一个可运行的函数，然后将其直接添加到微应用程序中
* `Executor("graph", {"link-params": True})` - 图执行器提供了网络的 JSON 表示，并且需要生成 C Runtime 的系统库，从而在函数注册表（`Runtime("crt" , {"system-lib": True})`）中查找函数。`{"link-params":True}` 允许将参数链接到生成的文件，而非从外部提供。

这些是在构建 runtime 模块时指定的：`relay.build(..., runtime=..., executor=...)`。

### 为 microTVM 编写 Schedules

对于在 CPU 上调度的操作，microTVM 最初计划利用专门的指令和外部（即手动优化）函数来提高性能。在 TVM 中，这种方法通常是通过张量实现的——TVM 将计算分解为多个部分，而 TIR 外部函数会加速每个部分。

TVM 目前用 `tir.call_extern` 来适应这两种方法。首先，将 pragma 附加到 schedule 上，这个 schedule 定义了可移植 C 代码中的外部函数。

``` plain
sched[output].pragma(n, "import_c", "void call_asm(int32_t* a, int32_t* b) { /* ... */ }")
```

接下来，用 `tensorize` 来拆分计算

``` plain
sched[output].tensorize(owi, gemm)
```

这种方法有几个注意事项，都可以通过链接生成代码与外部库来解决：

* 内联汇编是特定于编译器的。虽然 Clang 和 GCC 已经对一种语法进行了标准化，但可能无法移植到其他编译器。SDK 根据使用的编译器，有条件地包含一个头文件来解决这个问题。但是，采用这种方法意味着生成的代码需要额外的编译器标志（即 `-Isystempath/to/header`）。
* 引用生成的代码中的辅助函数会很有用（例如，手动优化汇编的内联通用序列）。
* 最后，调用的外部函数可以完全写在外部库中。若这些函数可以完全内联，则警告与前面的相同。若不是，则需要编译额外的 C 代码，并将其链接到算子。

目前，microTVM 假定所有符合条件的 schedules 都可以编译。这意味着用户提供的项目（参见下一节）必须包含生成的代码使用的所有库。

不使用自动调优时，TVM 随机选择一个回调 schedule，因此要支持所有库。使用自动调优时，TVM 会选择性能最佳的 schedule，因此只需要该库。目前没有办法强制 TVM 选择自动调优日志外的特定的 schedule，未来将考虑增加该功能。

最后，使用 `llvm` 后端时，除了 LLVM 位码包含在生成的代码中（使用 `import_llvm` pragma）之外，这个过程是相似的。LLVM 位码提供了一种调用内联汇编的可移植方式。但是，调用外部 C 函数更复杂，在 LLVM 位码中使用辅助函数也不容易。

### 执行模型

TVM 编译器通常会输出三个部分：

1. 如上所述的模型算子实现；
2. 模型执行图，编码为 JSON；
3. 简化参数。

为了能够正确执行模型，图执行器要在内存中重建计算图，加载参数，然后以正确的顺序调用算子的实现。

microTVM 支持两种方式：

1. **主机驱动**。 图执行器可以在主机上运行并通过使用带有类似 UART 传输的 RPC 链接向设备发出命令来执行。
2. **脱机执行**。C 图执行器可用于在设备上编译，但它的内存效率不是特别高。这种方式不依附于主机独立执行。

主机驱动的方法用于在设备上试验模型，类似 AutoTVM 用 RPC 服务器来驱动设备上的计算。脱机执行的方式用于部署。

#### 主机驱动执行

在主机驱动执行中，固件二进制如下：

1. 从 TVM 生成的算子实现。
2. TVM C runtime。
3. 特定于 SoC 的初始化。
4. TVM RPC 服务器。
5. （可选）简化参数。

将这个固件镜像烧录到设备上，并在主机上创建一个 GraphExecutor 实例。GraphExecutor 通过 UART 发送 RPC 命令来驱动执行：

![https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_host_driven.svg](https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_host_driven.svg)

#### 脱机执行

在脱机执行中，GraphExecutor 在设备上进行实例化：

![https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_standalone.svg](https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_standalone.svg)

### microTVM 固件

接下来讨论 microTVM 固件。两种模型执行策略都有一项重要任务，即配置 SoC，从而匹配其在生产中的执行方式。 microTVM 认为此任务依赖于项目和 SoC。无论是 AutoTVM，主机驱动的模型推理，还是脱机部署，用户都希望项目中的 main() 执行：

1. 配置 SoC，匹配部署性能。
2. 初始化 TVM C Runtime。

配置主机驱动的推理或 AutoTVM 时，其余任务：

3. 初始化传输（即 UART），用于 TVM RPC 服务器。
4. 启动 TVM RPC 服务器。

配置脱机部署时，固件需要：

1. 通过调用 `runtime.SystemLib` PackedFunc 来实例化系统库。
2. 实例化一个 GraphExecutor，来传递系统库模块。
3. 根据需要配置参数和输入。
4. 运行模型。

### microTVM 二进制文件部分

总之，microTVM 固件二进制镜像必须包含：

1. 算子实现，由 TVM 产生。
2. TVM C runtime 库，作为静态库由 TVM 提供。
3. SoC 初始化，由用户提供。

对于主机驱动的模型执行，固件还需要：

4. TVM RPC 服务器库。

对于脱机模型执行，固件还需要：

5. TVM C GraphExecutor 库，作为静态库由 TVM 提供。
6. 其余编译器输出（简化参数和图形JSON）。

### 自动化构建流程

代码生成后，`tvm.relay.build` 返回一个 `tvm.runtime.Module`，用户可以将生成的 C 源代码或二进制对象保存到 `.c` 或 `.o` 文件中。从这一点来看，TVM 理论上可以退后一步，用户可以分开编译和运行代码。

但是，对于 AutoTVM，TVM 需要一些自动化流程来处理：

1. 将算子实现、TVM C Runtime 库和 TVM RPC 服务器库集成到固件项目中，这个固件项目包含用户提供的 SoC 初始化。
2. 构建生成的项目。
3. 将内置固件烧录到（特定）连接的设备上。
4. 识别 TVM 使用的串行端口或其他传输方式来驱动远程执行。

目前，TVM 期待用户提供 `tvm.micro.Compiler`、`tvm.micro.Flasher` 和 `tvm.micro.Transport` 接口的实现。然后 TVM：

1. 将每个部分单独构建为一个库。
2. 将库构建为二进制固件镜像。
3. 将固件镜像烧录到连接的设备上。
4. 打开一个串行端口，作为 RPC 服务器传输。

选择此设计是为了减少 microTVM 的构建时间（只需为每个候选算子实现构建一次公共库）。实际上，这些项目非常小，并且编译速度相对较快。与 TVM 的构建集成更紧密，导致复杂性增加。与这种增加的复杂性相比，性能提升可能不值得。未来的设计会将构建任务整合到一个步骤中，并缩小接口，从而提供更好的集成。

### 测试算子性能

TVM C runtime 依赖用户提供的函数来测试设备上的时间。用户应实现 `TVMPlatformTimerStart` 和 `TVMPlatformTimerStop`。这些函数应该测试时钟时间，因此这些函数的实现存在一些缺陷：

1. 若 CPU 在计算期间停止或休眠（如果它正在加速器上完成），则不应该使用循环计数器，因为它们会在 CPU 休眠时停止计数。
2. 这些函数的粒度可以根据需要放宽，以扩展定时器设备的范围。然而，若粒度太粗，则可能使用次优 schedule。
3. 如果计时器溢出，则会引发错误。
4. 除非绝对必要，否则计时器不应中断计算。这样做可能会影响结果的准确性。
5. 理想的方法是根据时钟来校准输出，但可能太麻烦了。未来的 PR 可以实现平台定时器的某些特性，例如，参考外部晶体振荡器来测试内部振荡器。

## 未来工作

### Ahead-of-Time Runtime

图执行器的限制之一是解析 JSON 所需的内存开销。当前的实现对 microTVM 的动态内存使用贡献过多，这限制了它的实用性。Ahead-of-Time Runtime 无需解析任何 Graph JSON，并且可以通过生成 C 代码来提高推理速度，这个生成的 C 代码直接调用生成的算子实现，而非依赖于图执行器的数据驱动方法。

### 内存规划

当前内存规划器仅用于限定中间张量调用 `TVMBackendDeviceAlloc()` 的次数。因为暂存器各自的差异较大，并且由于规划器将内存分配合并到彼此的 16 倍以内，所以这种策略通常会导致内存使用的高峰值出现。

### 异构执行

较新的 Cortex-M SoC 可以包含多个 CPU 和板载 ML 加速器。

### 自动调优 Target

如前所述。