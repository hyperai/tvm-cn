---
title: TVM 原理介绍
---

:::note
单击 [此处](https://tvm.apache.org/docs/tutorial/introduction.html#sphx-glr-download-tutorial-introduction-py) 下载完整的示例代码
:::

# 介绍

**作者**：[Jocelyn Shiue](https://github.com/CircleSpin)，[Chris Hoge](https://github.com/hogepodge)，[Lianmin Zheng](https://github.com/merrymercy)

Apache TVM 是一个用于 CPU、GPU 和机器学习加速器的开源机器学习编译器框架，旨在让机器学习工程师能够在任何硬件后端上高效地优化和运行计算。本教程的目的是通过定义和演示关键概念，来引导用户了解 TVM 的所有主要功能。新用户完整学完本教程后应该对 TVM 架构及其工作原理有了基本的了解，从而能够用 TVM 来自动优化模型。

## 内容
1. TVM 原理简介
2. 安装 TVM
3. 使用命令行界面编译和优化模型
4. 使用 Python 接口编译和优化模型
5. 使用张量表达式操作算子
6. 使用模板和 AutoTVM 优化算子
7. 使用无模板的 AutoScheduler 优化算子
8. 交叉编译和远程过程调用（RPC）
9. 用 GPU 编译深度学习模型

# TVM 和模型优化概述
下图说明了使用 TVM 优化编译器框架转换时所采取的步骤。

![A High Level View of TVM](/img/docs/apache/tvm-site/main/images/tutorial/overview.png)

1. 从 TensorFlow、PyTorch 或 ONNX 等框架导入模型。在导入阶段中，TVM 可以从其他框架（如 TensorFlow、PyTorch 或 ONNX）中提取模型。 TVM 为前端提供的支持水平会随着我们不断改进这个开源项目而变化。如果在将模型导入 TVM 时遇到问题，可以将其转换为 ONNX。

2. 翻译成 TVM 的高级模型语言 Relay。已导入 TVM 的模型在 Relay 中表示。Relay 是神经网络的功能语言和中间表示（IR）。它支持：
   * 传统的数据流式表示
   * 函数式作用域，let-binding 使其成为一种功能齐全的可微语言
   * 允许用户混用两种编程风格的能力

   Relay 应用图级优化 pass 来优化模型。

3. 降级为张量表达式（TE）表示。降级是指将较高级的表示转换为较低级的表示。应用了高级优化之后，Relay 通过运行 FuseOps pass，把模型划分为许多小的子图，并将子图降级为 TE 表示。张量表达式（TE）是一种用于描述张量计算的领域特定语言。 TE 还提供了一些 schedule 原语来指定底层循环优化，例如循环切分、矢量化、并行化、循环展开和融合。为辅助将 Relay 表示转换为 TE 表示的过程，TVM 包含了一个张量算子清单（TOPI），其中包含常用张量算子的预定义模板（例如，conv2d、transpose）。

4. 使用 auto-tuning 模块 AutoTVM 或 AutoScheduler 搜索最佳 schedule。schedule 为 TE 中定义的算子或子图指定底层循环优化。auto-tuning 模块搜索最佳 schedule，并将其与 cost model 和设备上的测量值进行比较。 TVM 中有两个 auto-tuning 模块。
   * AutoTVM：基于模板的 auto-tuning 模块。它运行搜索算法以在用户定义的模板中找到可调 knob 的最佳值。 TOPI 中已经提供了常用算子的模板。
   * AutoScheduler（又名 Ansor）：无模板的 auto-tuning 模块。它不需要预定义的 schedule 模板，而是通过分析计算定义自动生成搜索空间，然后在生成的搜索空间中搜索最佳 schedule。

5. 为模型编译选择最佳配置。调优后，auto-tuning 模块会生成 JSON 格式的调优记录。此步骤为每个子图选择最佳 schedule。

6. 降级为张量中间表示（TIR，TVM 的底层中间表示）。基于调优步骤选择最佳配置后，所有 TE 子图降级为 TIR 并通过底层优化 pass 进行优化。接下来，优化的 TIR 降级为硬件平台的目标编译器。这是生成可部署到生产的优化模型的最终代码生成阶段。 TVM 支持多种不同的编译器后端：

   * LLVM，针对任意微处理器架构，包括标准 x86 和 ARM 处理器、AMDGPU 和 NVPTX 代码生成，以及 LLVM 支持的任何其他平台。
   * 特定编译器，例如 NVCC（NVIDIA 的编译器）。
   * 嵌入式和特定 target，通过 TVM 的 自定义代码生成（Bring Your Own Codegen, BYOC）框架实现。

7. 编译成机器码。compiler-specific 的生成代码最终可降级为机器码。
   TVM 可将模型编译为可链接对象模块，然后轻量级 TVM runtime 可以用 C 语言的 API 来动态加载模型，也可以为 Python 和 Rust 等其他语言提供入口点。或将 runtime 和模型放在同一个 package 里时，TVM 可以对其构建捆绑部署。

本教程的其余部分将更详细地介绍 TVM 的这些方面：

[下载 Python 源代码：introduction.py](https://tvm.apache.org/docs/_downloads/31d82e25454740f5ba711497485c0dd4/introduction.py)

[下载 Jupyter Notebook：introduction.ipynb](https://tvm.apache.org/docs/_downloads/9f81bc348ac4107d0670f512b8943a99/introduction.ipynb)
