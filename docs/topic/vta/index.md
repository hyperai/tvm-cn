---
title: VTA：多功能张量加速器
---

# VTA：多功能张量加速器

多功能张量加速器 (Versatile Tensor Accelerator，简称 VTA) 是一个开放、通用和可定制的深度学习加速器，具有完整的基于 TVM 的编译器堆栈。VTA 揭示了主流深度学习加速器最显著的共同特征。TVM 和 VTA 形成了一个端到端的硬件-软件深度学习系统堆栈，其中包括硬件设计、驱动程序、JIT runtime 和基于 TVM 的优化编译器堆栈。

![图片](https://raw.githubusercontent.com/uwsampl/web-data/main/vta/blogpost/vta_overview.png)

VTA 的主要功能：

* 通用、模块化、开源硬件。
* 简化了部署到 FPGA 的工作流程。
* 模拟器支持常规工作站上的原型编译传递。
* 基于 Pynq 的驱动程序和 JIT runtime，用于模拟和 FPGA 硬件后端。
* 端到端 TVM 堆栈集成。

本节包含与 VTA 相关的所有资源的链接：

* [VTA 安装指南](https://tvm.apache.org/docs/topic/vta/install.html)
* [VTA 设计和开发指南](https://tvm.apache.org/docs/topic/vta/dev/index.html)
* [VTA 教程](https://tvm.apache.org/docs/topic/vta/tutorials/index.html)

## 文献

* 阅读 VTA [博客文章](https://tvm.apache.org/2018/07/12/vta-release-announcement)。
* 阅读 VTA 技术报告：[用于深度学习的开放硬件软件堆栈](https://arxiv.org/abs/1807.04188)。