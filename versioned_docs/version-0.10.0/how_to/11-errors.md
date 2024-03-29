---
title: 处理 TVM 报错
sidebar_position: 210
---

# 处理 TVM 报错

运行 TVM 时，可能会遇到如下报错：

``` plain
---------------------------------------------------------------
An error occurred during the execution of TVM.
For more information, please see: https://tvm.apache.org/docs/errors.html
---------------------------------------------------------------
```

下面解释了这些错误消息是如何产生的，以及当错误发生时要怎么做。

## 这些错误从何而来？

这个错误是在 TVM 执行期间违反内部不变量引起的。从技术层面来看，消息是 `ICHECK` 宏（位于 `include/tvm/runtime/logging.h` 中）生成的，TVM 多处代码用到 `ICHECK` 宏来断言执行期间某个条件为真；一旦断言失败，TVM 都会退出，并显示如上错误消息。

有关 TVM 中错误是如何生成并处理的更多详细信息，参阅*错误处理指南*。

## 遇到这种错误怎么办？

最好的做法是在 [Apache TVM 论坛](https://discuss.tvm.apache.org/) 中搜索遇到的错误，看看其他人是否遇到过，以及可能的解决方案。如果错误已经在 TVM 更新版本中进行了修复，你可以更新 TVM 的版本。

若论坛上没有相关帖子，欢迎在论坛上创建一个新帖子描述问题的详细信息。*请在帖子中包含以下关键信息*：

* 当前 TVM 版本（例如，源代码树的 git commit 的哈希值）。
* TVM 运行的硬件和操作系统的版本。
* TVM 编译的硬件设备和操作系统。
* 用于重现此问题的信息，如模型、输入或其他信息。
  
如果没有这些信息，TVM 开发者很难给予你帮助。