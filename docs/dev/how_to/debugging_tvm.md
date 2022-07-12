---
title: Debuggging TVM
---

**注意**：本章节内容持续更新中，欢迎各位提交 PR 不断丰富。本文档旨在集中分享调试 TVM 的常用方法，排列顺序以使用频率为依据。

## VLOGging

TVM 提供了一个冗余信息的记录工具，允许开发者在不影响运行中 TVM 二进制大小或 runtime 的同时，提交跟踪级别的调试信息。在代码中使用 VLOG：

``` c++
void Foo(const std::string& bar) {
  VLOG(2) << "Running Foo(" << bar << ")";
  // ...
}
```

在这个例子中，传递给 `VLOG()` 的整数 `2` 表示冗余级别。级别越高，打印的日志就越多。一般来说，TVM 级别从 0 到 2，3 只用于极底层的核心运行时属性。VLOG 系统在启动时被配置为打印 `0` 到某个整数 `N` 之间的 VLOG 语句。`N` 可以按文件或全局来设置。

默认情况下（当采用适当的优化进行编译时），VLOG 不会打印或影响二进制的大小或runtime。启用 VLOGging，请执行以下操作：


1. 在 `config/cmake` 中，确保设置 `set(USE_RELAY_DEBUG ON)`。这个标志是用来启用 VLOGging 的。
2. 通过 `TVM_LOG_DEBUG=<spec>` 启动 Python，其中 `<spec>>` 是一个形式为 `<file_name>=<level>` 的以逗号分隔的级别赋值列表。尤其需要注意：
    * 特殊文件名 `DEFAULT` 为所有文件设置 VLOG 级别。
    * `<level>>` 可以被设置为 `-1` 来禁用该文件的 VLOG。
    * `<file_name>` 是在 TVM 仓库中相对于 `src/` 目录的 c++ 源文件的名字（例如 `.cc`，而不是 `.h`）。您不需要在指定文无论在指定文件路径时，是否提供 `src/`，VLOG 都可以正确解释路径。


