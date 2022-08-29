---
title: 关于 microTVM 项目 API
sidebar_position: 260
---

# 关于 microTVM 项目 API

microTVM 项目 API 使得 TVM 在非常规或嵌入式平台上自动运行模型。它使得平台可以定义标准函数，从而将 TVM 编译器输出与样板平台特定代码集成，生成可运行的**项目**。然后，项目 API 进一步定义了构建项目的函数，以及在可从 TVM 机器获取的兼容设备上编程，并与运行代码进行通信，使得 TVM 可以执行主机驱动的推理和自动调优。

在许多情况下，仅需要简单地从平台的构建过程中，把 microTVM 作为工具来调用即可。事实上，对于普通的固件开发者来说，这些就足够了。但是，有几个用例需要 microTVM 用平台的构建工具来构建固件：

1. 在平台上启用 AutoTVM 和 AutoScheduling。定义项目 API 的实现，使得 TVM 将平台上的模型调到最优的性能。
2. 让没有固件专业知识的工程师能够在平台上试验模型。定义项目 API 的实现，使得这些工程师利用标准 TVM Python 工作流，在平台上执行主机驱动的推理。
3. 集成测试。定义项目 API 的实现，使得能够创建持续集成测试，以验证平台上模型的正确性和性能。

## API 定义

完整的 API 是在 [python/tvm/micro/project_api/server.py](https://github.com/apache/tvm/blob/main/python/tvm/micro/project_api/server.py) 中 `ProjectAPIHandler` 上定义的 `abstractmethod`。这里不复制文档，只是简单介绍该类。

## TVM 如何使用项目 API

本节介绍项目 API 如何与 TVM 一起使用。项目 API 围绕*项目*定义为固件的可构建单元。 TVM 提供一个包含*模板项目*的目录，该目录与 [模型库格式](https://tvm.apache.org/docs/arch/model_library_format.html#model-library-format) 文件一起，被构建为一个可运行的项目。

模板目录内（通常）是一个 Python 脚本，这个脚本实现了 API 服务器。TVM 在子进程中启动此脚本，并向服务器发送命令，执行上述操作。

一般的使用流程如下：

1. 在模板项目中启动项目 API 服务器。
2. 通过发送 `server_info_query` 命令，验证 API 服务器与 TVM 版本是否兼容，并读取实现的属性。
3. 通过发送 `generate_project` 命令生成一个新项目。这个命令的参数是一个模型库格式和一个不存在的目录（用生成的项目填充）。模板项目 API 服务器将自身复制到新生成的项目中。
4. 终止模板项目 API 服务器。
5. 在生成的项目中启动项目 API 服务器。
6. 通过发送 `server_info_query` 命令，验证 API 服务器与 TVM 版本兼容，并读取实现的属性。
7. 通过向 API 服务器发送 `build` 和 `flash` 命令，构建和刷新项目。
8. 与 target 通信。发送 `open_transport` 命令后，再发送 `write_transport` 和 `read_transport` 命令，从 target 的串行端口进行写入和读取操作。完成后，发送 `close_transport`。
9. 终止项目 API 服务器。

## 项目的磁盘布局

在项目（模板或生成的）的根目录中，以下两个文件必须存在一个：

* `microtvm_api_server.py` - （推荐方法）。将一个兼容 Python 3 的 Python 脚本放在根目录下。 TVM 会用与执行 TVM 一致的解释器，在进程中执行此脚本。
* `microtvm_api_server.sh`（在 Windows 上，`microtvm_api_server.bat`） - （替代方法）。当需要不同的 Python 解释器时，或者想用不同的语言实现服务器时，创建这个可执行文件。 TVM 将在单独的进程中启动此文件。
  除了这两个文件之外，对布局没有其他限制。

## TVM 和项目 API 服务器之间的通信

TVM 用 [JSON-RPC 2.0](https://www.jsonrpc.org/specification) 与项目 API 服务器进行通信。 TVM 用以下命令行来启动 API 服务器：

``` bash
microtvm_api_server.py --read-fd <n> --write-fd <n>
```

命令通过 `--read-fd` 给出的文件描述符从 TVM 发送到服务器，然后 TVM 通过 `--write-fd` 给出的文件描述符从服务器接收回复。

## 在 Python 中实现 API 服务器的辅助函数

TVM 提供了辅助函数使得在 Python 中实现服务器更容易。要在 Python 中实现服务器，需创建 `microtvm_api_server.py`，并添加 `from tvm.micro.project_api import server`（或者，将这个文件复制到你的模板项目中——无需依赖——并将其导入）。接下来是子类 `ProjectAPIHander`：

``` python
class Handler(server.ProjectAPIHandler):
    def server_info_query(self, tvm_version):
        # Implement server_info_query
        # 实现 server_info_query

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
        # Implement generate_project
        # 实现 generate_project

    # ...
```

最后，调用辅助函数 `main()`：

``` python
if __name__ == "__main__":
    server.main(Handler())
```

## 使用来自 `tvmc` 的项目 API

通过 `tvmc micro` 子命令，可以使用所有主要的项目 API 命令，这样简化了调试交互。调用 `tvmc micro --help` 获取更多信息。