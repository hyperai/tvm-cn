---
title: microTVM：裸机上的 TVM
---

# microTVM：裸机上的 TVM

microTVM 在裸机（即物联网）设备上运行 TVM 模型。microTVM 只依赖于 C 标准库，不需要操作系统来执行。 microTVM 目前正在全力开发中。

![/img/docs/tlc-pack/web-data/main/images/dev/microtvm_workflow.svg](/img/docs/tlc-pack/web-data/main/images/dev/microtvm_workflow.svg)

microTVM 是：

* TVM 编译器的扩展，允许 TVM 以微控制器为目标
* 一种在设备上运行 TVM RPC 服务器的方法，允许自动调优
* 最小 C runtime，支持裸机设备上的独立模型推理。

## 支持的硬件

microTVM 目前测试支持 Zephyr RTOS 的 Cortex-M 微控制器；不过，它灵活，且可移植到 RISC-V 等其他处理器，也无需 Zephyr。当前的 demo 针对 QEMU 和以下硬件运行：

* [STM Nucleo-F746ZG](https://www.st.com/en/evaluation-tools/nucleo-f746zg.html)
* [STM STM32F746 Discovery](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
* [nRF 5340 开发套件](https://www.nordicsemi.com/Software-and-tools/Development-Kits/nRF5340-DK)

## microTVM 入门

在使用 microTVM 之前，推荐使用支持的开发板。然后，按照这些教程开始使用 microTVM：

1. [启动 microTVM 虚拟机参考手册](/docs/how_to/microtvm/microtvm_vm)。microTVM 教程依赖于 Zephyr 和硬件的编译器工具链。通过虚拟机参考手册，可快速安装这些依赖。
2. 尝试使用 [TFLite 教程的 microTVM](/docs/how_to/microtvm/microtvm_tflite)。
3. 尝试运行更复杂的 [CIFAR10-CNN 模型](https://github.com/areusch/microtvm-blogpost-eval)。

## microTVM 的工作原理

可以在 [microTVM 设计文档](/docs/arch/arch/microtvm_design) 中阅读有关这些部件设计的更多信息。

## 帮助及讨论

推荐访问 [TVM 论坛](https://discuss.tvm.ai/)，查看历史帖子并探讨 microTVM 的相关问题。
