---
title: 模型库格式
sidebar_position: 270

---

# 模型库格式

## 关于模型库格式

以前的 TVM 将生成的库导出为动态共享对象（例如 DLL（Windows）或 .so（linux））。通过使用 `libtvm_runtime.so` 将它们加载到可执行文件中，可以使用这些库执行推理。这个过程强依赖于传统操作系统提供的服务。

为了部署到非常规平台（例如那些缺乏传统操作系统的平台），TVM 提供了另一种输出格式——模型库格式（Model Library Format）。最开始的时候，这种格式主要应用于 microTVM。如果它可以用于其他用例（特别是，如果可以以模型库格式导出 BYOC 工件），那么就可以用作通用 TVM 导出格式。模型库格式是一个 tarball，其中包含每个 TVM 编译器输出的文件。

## 可以导出什么？

编写代码时，仅能导出用 `tvm.relay.build` 构建的完整模型。

## 目录布局

模型库格式包含在 tarball 中。所有路径都是相对于 tarball 的根目录而言的：

* `/` - tarball 的根目录
  * `codegen` - 所有生成的设备代码的根目录
    * （详见 [codegen](https://tvm.apache.org/docs/arch/model_library_format.html#codegen) 部分）
* `executor-config/` - 驱动模型推理的执行器配置
  * `graph/` - 包含 GraphExecutor 配置的根目录
    * `graph.json` - GraphExecutor JSON 配置
* `metadata.json` - 此模型的机器可解析元数据
* `parameters/` - 放置简化参数的根目录
  * `<model_name>.params` - 模型 tvm.relay._save_params 格式的参数
* `src/` - TVM 使用的所有源代码的根目录
  * `relay.txt` - 生成模型的 Relay 源代码

## 子目录说明

### `codegen`

TVM 生成的所有代码都放在这个目录中。编写代码时，生成的模块树中的每个模块有 1 个文件，但此限制未来可能会更改。此目录中的文件应具有 `<target>/(lib|src)/<unique_name>.<format>` 形式的文件名。

这些组件描述如下：

* `<target>` - 标识运行代码的 TVM target。目前，仅支持 `host`。
* `<unique_name>` - 标识此文件的唯一 slug。当前是 `lib<n>`，其中 `<n>` 是一个自动递增的整数。
* `<format>` - 标识文件名格式的后缀。目前是 `c` 或 `o`。

仅 CPU 模型的示例目录树如下所示：

* `codegen/` - codegen 目录
  * `host/` - 为 `target_host` 生成的代码
    * `lib/` - 生成的二进制目标文件
      * `lib0.o` - LLVM 模块（如果使用 `llvm` target）
      * `lib1.o` - LLVM CRT 元数据模块（如果使用 `llvm` target）
    * `src/` - 生成的 C 源代码
      * `lib0.c` - C 模块（如果使用 `c` target）
      * `lib1.c` - C CRT 元数据模块（如果使用 `c` target）

### `executor-config`

包含对执行器的机器可解析配置，这个执行器可以驱动模型推理。目前，只有 GraphExecutor（在 `graph/graph.json` 中）为此目录生成配置。读入这个文件，并将生成的字符串提供给 `GraphExecutor()` 构造函数进行解析。

### `parameters`

包含机器可解析的参数。可提供多种格式，但目前只提供 `tvm.relay._save_params` 产生的格式。用 `tvm.relay.build` 构建时，`name` 参数是模型名称。在此目录 `<model_name>.json` 中创建了一个文件。

### `src`

包含 TVM 解析的源代码。目前，在 `src/relay.txt` 中只创建了 Relay 源代码。

## 元数据

机器可解析的元数据放在 tarball 根目录下的 `metadata.json` 文件中。元数据是具有以下键的字典：

* `export_datetime`：模型库格式生成时的时间戳，为 [strftime](https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior) 格式 `"%Y-%M-%d %H:%M:%SZ"`。
* `memory`：所有生成函数的内存使用总结。记录在 [内存使用总结](#memory-usage-summary) 中。
* `model_name`：模型名称（例如，提供给 `tvm.relay.build` 的 `name` 参数）。
* `executors`：模型支持的执行器列表。当前，此列表始终为 `["graph"]`。
* `target`：将 `device_type`（潜在的整数，作为字符串）映射到描述用于该 `device_type` 的 Relay 后端的子 target 的字典。
* `version`：标识模型库格式中格式的数字版本号。当元数据结构或磁盘结构更改时，这个数字会增加。本文档反映的是第 `5` 版。

### 内存使用总结

它是具有如下子键的字典：

* `"main"`：`list[MainFunctionWorkspaceUsage]`。总结内存使用情况的列表，包括 main 函数使用的工作空间，以及调用的子函数。
* `"operator_functions"`：`map[string, list[FunctionWorkspaceUsage]]`。将算子函数名称映射到一个列表，这个列表总结了函数使用的每个工作空间的内存使用情况。

`MainFunctionWorkspaceUsage` 字典具有以下键：

* `"device"`：`int`。这个工作空间关联的 `device_type`。
* `"workspace_size_bytes"`：`int`。工作空间中调用的函数和子函数所需的字节数。
* `"constants_size_bytes"`：`int`。main 函数使用的常量的大小。
* `"io_size_bytes"`：`int`。工作空间中函数和子函数使用的缓冲区大小的总和。

`FunctionWorkspaceUsage` 字典具有以下键：

* `"device"`：`int`。这个工作空间关联的 `device_type`。
* `"workspace_size_bytes"`：`int`。此函数在此工作空间中所需的字节数。
