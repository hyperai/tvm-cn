---
title: VTA 硬件指南
---

# VTA 硬件指南

我们提供了 VTA 硬件设计的自上而下的概述。本硬件设计指南包含两个级别的 VTA 硬件：

* VTA 设计及其 ISA 硬件-软件接口的架构概述。
* VTA 硬件模块的微架构概述，以及计算 core 的微代码规范。

## VTA 概述

VTA 是一种通用深度学习加速器，专为快速高效的密集线性代数而构建。VTA 包含一个简单的类似 RISC 的处理器，可以在秩 (rank) 为 1 或 2 的张量寄存器上执行密集线性代数运算。此外，该设计采用解耦的访问执行，来隐藏内存访问延迟。

在更广的范围内，VTA 可以作为全栈优化的模板深度学习加速器设计，将通用张量计算接口提供给编译器栈。

![/img/docs/uwsampl/web-data/main/vta/blogpost/vta_overview.png](/img/docs/uwsampl/web-data/main/vta/blogpost/vta_overview.png)

上图给出了 VTA 硬件组织的高级概述。VTA 由四个模块组成，它们通过 FIFO 队列和本地内存块 (SRAM) 相互通信，实现任务级 pipeline 并行：

* fetch 模块负责从 DRAM 加载指令流。它将这些指令解码，并将它们路由到三个命令队列的任意一个。
* load 模块负责将输入和权重张量从 DRAM 加载到数据专用的芯片存储器中。
* compute 模块用其 GEMM 内核执行密集线性代数计算，并用其张量 ALU 执行一般计算。它还负责将数据从 DRAM 加载到寄存器文件中，并将微操作内核加载到微操作缓存中。
* store 模块将计算 core 产生的结果存储回 DRAM。

## HLS 硬件源代码组织

VTA 设计目前在 Vivado HLS C++ 中指定，只有 Xilinx 工具链支持。 VTA 硬件源代码包含在 `3rdparty/vta-hw/hardware/xilinx/sources` 目录下：

* `vta.cc` 包含所有 VTA 模块的定义，以及顶层 VTA 设计的顶层行为模型。
* `vta.h` 包含用 Xilinx `ap_int` 类型实现的类型定义，以及函数原型声明。

此外，预处理器宏定义在 `3rdparty/vta-hw/include/vta/hw_spec.h` 目录下。这些宏定义大部分来自 `3rdparty/vta-hw/config/vta_config.json` 文件中列出的参数。

json 文件由 `3rdparty/vta-hw/config/vta_config.py` 处理，生成一个编译标志的字符串，来定义预处理器的宏。

makefile 文件用这个字符串，在 HLS 硬件综合编译器和构建 VTA runtime 的 C++ 编译器中，设置这些高级参数。

### HLS 模块示例

以下是一个 VTA 模块在 C++ 中的定义：

``` c++
void fetch(
  uint32_t insn_count,
  volatile insn_T *insns,
  hls::stream<insn_T> &load_queue,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<insn_T> &store_queue) {
#pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port
#pragma HLS INTERFACE axis port = load_queue
#pragma HLS INTERFACE axis port = gemm_queue
#pragma HLS INTERFACE axis port = store_queue
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

  INSN_DECODE: for (int pc = 0; pc < insn_count; pc++) {
#pragma HLS PIPELINE II = 1
    // Read instruction fields
    // 读取指令字段
    insn_T insn = insns[pc];
    // Do some partial decoding
    // 做部分解码
    opcode_T opcode = insn.range(VTA_INSN_MEM_0_1, VTA_INSN_MEM_0_0);
    memop_id_T memory_type = insn.range(VTA_INSN_MEM_5_1, VTA_INSN_MEM_5_0);
    // Push to appropriate instruction queue
    // 推送到合适的指令队列
    if (opcode == VTA_OPCODE_STORE) {
      store_queue.write(insn);
    } else if (opcode == VTA_OPCODE_LOAD &&
        (memory_type == VTA_MEM_ID_INP || memory_type == VTA_MEM_ID_WGT)) {
      load_queue.write(insn);
    } else {
      gemm_queue.write(insn);
    }
  }
}
```

**关于 HLS 编码的一些观点：**

* *参数*：每个函数的参数列表和接口编译指示，定义了生成的硬件模块公开的硬件接口。
  * 按值传递的参数表示的是，主机可以写入的只读硬件内存映射寄存器。例如，这个 fetch 函数有一个 `insn_count` 参数，该参数将被合成为主机写入的内存映射寄存器，设置给定 VTA 指令序列的长度。
  * 根据所用的接口编译指示，指针参数可以是：
    * 与 `m_axi` 接口编译指示一起使用时，将生成 AXI 请求者接口，提供对 DRAM 的 DMA 访问。
    * 与 `bram` 接口编译指示一起使用时，生成 BRAM 接口，将读和/或写端口公开到 FPGA block-RAM。
  * 将推理传递的 HLS 流与 `axis` 接口编译指示结合，产生模块的 FIFO 接口。硬件 FIFO 在模块之间提供了一种有用的同步机制。
* *编译指示**(pragmas)*：要定义每个模块的硬件实现，编译器编译指示是必不可少的。下面列出了 VTA 设计中使用的几个编译指示，其作用是将实现要求传递给编译器。
  * `HLS INTERFACE`：指定合成硬件模块的接口。
  * `HLS PIPELINE`：通过设置启动间隔目标来定义硬件 pipeline 性能 target。当设置 `II == 1` target 时，它告诉编译器合成的硬件 pipeline 能在每个周期执行一次循环迭代。
  * `HLS DEPENDENCE`：指示编译器忽略给定循环中某些类型的依赖检查。一个对相同 BRAM 结构进行写和读的循环体，需要 II 为 1。HLS 编译器必须假设最坏的情况，即：向之前写操作更新循环的地址发出读操作：鉴于 BRAM 时序特性，这是无法实现的（至少需要 2 个周期才能看到更新的值）。因此，为了实现 II 为 1，必须放宽依赖检查。注意，当打开此优化时，它会进入软件堆栈，防止写入后读取相同的地址。

:::note
本 [参考指南](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2018_2/ug902-vivado-high-level-synthesis.pdf) 给出了 Xilinx 2018.2 工具链更深入、更完整的 HLS 规范。
:::

## 架构概述

### 指令集架构

VTA 的指令集架构 (instruction set architecture，简称 ISA) 由 4 条具有可变执行延迟的 CISC 指令组成，其中两条指令通过执行微编码指令序列来执行计算。

下面列出了 VTA 指令：

* `LOAD` 指令：将 2D 张量从 DRAM 加载到输入缓冲区、权重缓冲区或寄存器文件中。它还可以将微内核加载到微操作缓存中。加载输入和权重图块时支持动态填充。
* `GEMM` 指令：在输入张量和权重张量上执行矩阵-矩阵乘法的微操作序列，并将结果添加到寄存器堆张量。
* `ALU` 指令：对寄存器文件张量数据执行矩阵-矩阵 ALU 操作的微操作序列。
* `STORE` 指令：将 2D 张量从输出缓冲区存储到 DRAM。

`LOAD` 指令由 load 和 compute 模块执行，具体取决于存储内存缓冲区位置 target。`GEMM` 和 `ALU` 指令由 compute 模块的 GEMM core 和张量 ALU 执行。最后，`STORE` 指令由 store 模块独占执行。每条指令的字段如下图所示。所有字段的含义将在 [微架构概述](https://tvm.apache.org/docs/topic/vta/dev/hardware.html#vta-uarch) 章节进一步解释。

![/img/docs/uwsampl/web-data/main/vta/developer/vta_instructions.png](/img/docs/uwsampl/web-data/main/vta/developer/vta_instructions.png)

:::note
VTA ISA 会随着 VTA 的架构参数（即 GEMM core shape、数据类型、内存大小等）的修改而变化，因此 ISA 不能保证所有 VTA 变体的兼容性。但这是可以接受的，因为 VTA runtime 会适应参数变化，并生成和生成的加速器版本匹配的二进制代码。这体现了 VTA 堆栈采用的协同设计理念，它包含硬件-软件接口的流动性。
:::

### 数据流执行

VTA 依靠硬件模块之间依赖 FIFO 队列 (dependence FIFO queues)，来同步任务并发执行。下图展示了给定的硬件模块，如何用依赖 FIFO 队列和单读取器/单写入器 SRAM 缓冲区，以数据流的方式同时从其生产者和消费者模块执行。所有模块都通过写后读 (RAW) 和读后写 (WAR) 依赖队列连接到其消费者和生产者。

![/img/docs/uwsampl/web-data/main/vta/developer/dataflow.png](/img/docs/uwsampl/web-data/main/vta/developer/dataflow.png)

以上伪代码描述了，模块如何基于与其他指令的依赖关系，执行给定指令。首先，每条指令中的依赖标志在硬件中被解码。若指令具有传入的 RAW 依赖，则基于从生产者模块接收到 RAW 依赖 token 执行。

类似地，若任务具有传入的 WAR 依赖，则基于从消费者模块接收到 WAR 依赖 token 执行。最后，任务完成后，检查输出的 RAW 和 WAR 依赖关系，并分别通知消费者和生产者模块。

:::note
此场景中的依赖 token 是无信息的。这是因为每个模块执行的指令是以 FIFO 顺序到达，无法按设计重新排序。
:::

### pipeline 可扩展性

默认的 VTA 设计由四个模块组成，它们描述了3 个阶段的 `load-compute-store` 任务 pipeline。遵循数据流硬件组织原则，可以扩展 VTA pipeline，使其包含更多阶段。

例如，设想将张量 ALU 与 GEMM core 分离，从而最大化利用 GEMM core。将产生一个密切反映 TPU 设计的 `load-gemm-activate-store` 任务 pipeline。然而，添加更多阶段是有代价的：它会增加存储和逻辑开销，这就是默认选择 3 个阶段的 pipeline 的原因。

## 微架构概述

本节描述了构成 VTA 设计的模块。模块定义包含在 `3rdparty/vta-hw/hardware/xilinx/sources/vta.cc` 中。

### Fetch 模块

VTA 由线性指令流编程。fetch 模块是 VTA 到 CPU 的入口点，通过三个内存映射寄存器进行编程：

* 读写 `control` 寄存器启动 fetch 模块，然后读取它来检查其是否完成。
* 只写 `insn_count` 寄存器设置要执行的指令数。
* 只写 `insns` 寄存器设置 DRAM 中指令流的起始地址。

CPU 在 VTA runtime 分配的物理连续的 buffer 中，准备 DRAM 中的指令流。指令流就绪后，CPU 将起始物理地址写入 `insns` 寄存器，将指令流的长度写入 `insn_count` 寄存器，并在 `control` 寄存器中断言启动信号。此过程会启动 VTA（通过 DMA 从 DRAM 读取指令流）。

访问指令流时，fetch 模块会对指令部分解码，并将这些指令推送到命令队列中，这些指令队列会送到 load、compute 和 store 模块中：

* `STORE` 指令被推送到存储命令队列，供 store 模块处理。
* `GEMM` 和 `ALU` 指令被推送到计算命令队列，供 compute 模块处理。
* 描述微操作内核或寄存器文件数据的加载操作的 `LOAD` 指令被推送到计算命令队列，供 compute 模块处理。
* 描述输入或权重数据的加载操作的 `LOAD` 指令被推送到加载命令队列，供 load 模块处理。

当其中一个命令队列被填满，fetch 模块会停止，直到队列恢复未满状态。因此，命令队列的大小要足够大，允许较宽的执行窗口，还允许多个任务在 `load-compute-store` pipeline 中同时运行。

### Compute 模块

VTA 的 compute 模块充当 RISC 处理器，在张量寄存器（而非标量寄存器）上执行计算。两个功能单元改变寄存器文件：张量 ALU 和 GEMM core。

compute 模块从微操作缓存中取出 RISC 微操作执行。有两种类型的计算微操作：ALU 和 GEMM 操作。为了最小化微操作内核的占用空间，同时避免对控制流指令（如条件跳转）的需求，compute 模块在两级嵌套循环内执行微操作序列，该循环通过一个仿射函数计算每个张量寄存器的位置。这种压缩方法有助于减少微内核指令的占用空间，适用于矩阵乘法和 2D 卷积，这在神经网络算子中很常见。

![/img/docs/uwsampl/web-data/main/vta/developer/gemm_core.png](/img/docs/uwsampl/web-data/main/vta/developer/gemm_core.png)

**GEMM core** 通过在 2 级嵌套循环（如上图所示）中执行微代码序列来评估 GEMM 指令。GEMM core 每个周期可以执行一次输入权重矩阵乘法。单周期矩阵乘法的维度定义了 TVM 编译器将计算 schedule 降级后得到的硬件*张量内联函数*。

这种张量内联函数由输入、权重和累加器张量的维度定义。每种数据类型的整数精度不同：通常权重和输入类型都是低精度（8 位或更少），而累加器张量具有更宽的类型（32 位），防止溢出。为了让 GEMM core 保持高利用率，每个输入缓冲区、权重缓冲区和寄存器文件都必须提供足够的读/写带宽。

![/img/docs/uwsampl/web-data/main/vta/developer/alu_core.png](/img/docs/uwsampl/web-data/main/vta/developer/alu_core.png)

**Tensor ALU** 支持一组标准操作来实现常见的激活、归一化和池化操作。VTA 的设计遵循模块化原则，Tensor ALU 支持的算子范围可以进一步扩大，但代价是消耗更多的资源。

张量 ALU 可以执行张量-张量运算，以及对立即值执行张量-标量运算。张量 ALU 的操作码和立即数由高级 CISC 指令指定。张量 ALU 计算上下文中的微代码只负责指定数据访问模式。

:::note
在计算吞吐量方面，Tensor ALU 的执行速度不是每个周期一个操作。缺少读取端口会带来一些限制：由于每个周期可以读取一个寄存器文件张量，因此张量 ALU 的启动间隔至少为 2（即每 2 个周期最多执行 1 个操作）。
:::

此外，一次执行单个张量-张量操作可能会很耗时，尤其是寄存器文件类型很宽 (wide)，通常是 32 位整数。因此，为了平衡 Tensor ALU 与 GEMM 核心的资源利用率，默认情况下，张量-张量操作是通过多个周期的向量-向量操作来执行的。

### Load 和 Store 模块

![/img/docs/uwsampl/web-data/main/vta/developer/2d_dma.png](/img/docs/uwsampl/web-data/main/vta/developer/2d_dma.png)

load 和 store 模块使用从 DRAM 到 SRAM 的跨步访问模式执行 2D DMA 加载。此外，load 模块可以动态插入 2D 填充（在阻塞 2D 卷积时很有用）。这意味着 VTA 可以平铺 2D 卷积输入，而无需补偿在 DRAM 中重新布局数据在输入和权重块 (weight tiles) 周围插入空间填充的开销。
