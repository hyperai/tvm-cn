---
title: FAQ
sidebar_position: 220
---

# FAQ
## 如何安装
参阅 [安装 TVM](../../get_start/install_idx)

## 如何添加新的硬件后端
* 如果硬件后端支持 LLVM，则可以直接通过在 `target` 中设置正确的 `target` 三元组来生成代码。
* 如果 target 硬件是 GPU，请用 cuda、opencl 或 vulkan 后端。
* 如果 target 硬件是一个特殊的加速器，请查看 [VTA：多功能张量加速器](../../topic/vta) 和 [向 TVM 中添加自定义 Codegen ](../../dev/how_to/relay_bring_your_own_codegen)。
* 对上述所有情况，若要用 AutoTVM 添加 target-specific 的优化模板，请参阅 [使用模板和 AutoTVM 进行自动调优](../../user_guide/how_to_guide/autotune)。
* 除了使用 LLVM 的向量化，还可以嵌入微内核来利用硬件内联函数，请参阅 [使用 Tensorize 来利用硬件内联函数](../../user_guide/how_to_guide/te_schedules/tensorize)。

## TVM 与其他 IR/DSL 项目的关系
深度学习系统中通常有两个层次的 IR 抽象。TensorFlow 的 XLA 和 Intel 的 ngraph 都使用计算图表示，它是高级的表示，有助于执行通用优化，例如内存重用、布局转换和自动微分。

TVM 采用低级表示，明确表示内存布局、并行化模式、局部性和硬件原语等选择。低级 IR 更类似 target 硬件——采用了现有图像处理语言，如 Halide、darkroom 和循环转化工具（如 loopy 和基于多面体的分析）的想法。重点关注如何表达深度学习工作负载（如 recurrence）、不同硬件后端的优化，以及如何嵌入框架，从而提供端到端的编译堆栈。

## TVM 与 libDNN、cuDNN 的关系
TVM 将这些库作为外部调用。TVM 的目标之一是生成高性能内核。通过学习手动内核制作技术，并将它们作为原语添加到 DSL 的方式，我们得以增量发展 TVM。有关 TVM 中算子的组成，参见顶部。

## 安全
参阅 [安全指南](../../arch/arch/security)