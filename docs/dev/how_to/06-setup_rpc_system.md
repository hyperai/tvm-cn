# 设置 RPC 系统

远程过程调用（RPC）是 Apache TVM 的一个非常重要且有用的功能，它允许我们在实际硬件上运行编译后的神经网络（NN）模型，而无需触及远程设备，输出结果将通过网络自动传回。

通过消除手动工作，如将输入数据转储到文件、将导出的 NN 模型复制到远程设备、设置设备用户环境以及将输出结果复制到主机开发环境，RPC 极大地提高了开发效率。

此外，由于仅在编译后的 NN 模型的远程设备上运行执行部分，所有其他部分都在主机开发环境上运行，因此可以使用任何 Python 包来执行预处理和后处理工作。

RPC 在以下两种情况下非常有帮助

- **硬件资源有限**

  RPC 的队列和资源管理机制可以使硬件设备为许多开发人员和测试作业正确运行编译后的 NN 模型。

- **早期端到端评估**

  除编译后的 NN 模型外，所有其他部分都在主机开发环境上执行，因此可以轻松实现复杂的预处理或后处理。

## 建议的架构

Apache TVM RPC 包含 3 个工具，RPC 追踪器、RPC 代理和 PRC 服务器。RPC 服务器是必不可少的，RPC 系统可以在没有 RPC 代理和 RPC 追踪器的情况下正常工作。只有在无法直接访问 RPC 服务器时才需要 RPC 代理。强烈建议在 RPC 系统中添加 RPC 追踪器，因为它提供许多有用的功能，例如队列功能、多个 RPC 服务器的管理以及通过密钥而不是 IP 地址管理 RPC 服务器。

![建议的架构](https://raw.githubusercontent.com/tlc-pack/web-data/main/images/dev/how-to/rpc_system_suggested_arch.svg)

如上图所示，因为机器 A 和机器 D, C 之间没有物理连接通道，所以我们在机器 B 上设置了一个 RPC 代理。RPC 追踪器管理每个 RPC 密钥的请求队列，每个用户都可以随时通过 RPC 密钥从 RPC 追踪器请求一个 RPC 服务器，如果存在具有相同 RPC 密钥的空闲 RPC 服务器，则 RPC 追踪器将该 RPC 服务器分配给用户，如果当前没有空闲的 RPC 服务器，则该请求将放入该 RPC 密钥的请求队列中，并稍后进行检查。

## 设置 RPC 追踪器和 RPC 代理

通常情况下，RPC 追踪器和 RPC 代理只需要在主机机器上运行，例如，开发服务器或 PC，它们不需要依赖于设备机器的任何环境，因此在根据官方文档 [https://tvm.apache.org/docs/install/index.html](https://tvm.apache.org/docs/install/index.html) 安装 Apache TVM 后，在相应的机器上执行以下命令即可完成设置。

- RPC 追踪器

  ```shell
  $ python3 -m tvm.exec.rpc_tracker --host RPC_TRACKER_IP --port 9190 --port-end 9191
  ```

- RPC 代理

  ```shell
  $ python3 -m tvm.exec.rpc_proxy --host RPC_PROXY_IP --port 9090 --port-end 9091 --tracker RPC_TRACKER_IP:RPC_TRACKER_PORT
  ```

请根据具体环境修改上述命令中的 *RPC_TRACKER_IP*、*RPC_TRACKER_PORT*、*RPC_PROXY_IP* 和端口号，选项 ``port-end`` 可用于避免服务以意外的端口号启动，这可能导致其他服务无法正确连接，特别是对于自动测试系统而言，这一点非常重要。

## 设置 RPC 服务器

在我们的社区中，有多个 RPC 服务器实现，例如 ``apps/android_rpc``、``apps/cpp_rpc``、``apps/ios_rpc``，以下内容只关注由 ``python/tvm/exec/rpc_server.py`` 实现的 Python 版本 RPC 服务器的设置说明，请参阅其相应目录的文档以获取其他版本 RPC 服务器的设置说明。

RPC 服务器需要在设备机器上运行，通常会依赖于 xPU 驱动程序、带有 xPU 支持的增强 TVM 运行时以及其他库，因此请先设置相关组件，例如安装 KMD 驱动程序，确保从环境变量 ``LD_LIBRARY_PATH`` 中可以找到所需的动态库。

如果可以在设备机器上设置所需的编译环境，即不需要进行交叉编译，则只需按照 [https://tvm.apache.org/docs/install/from_source.html](https://tvm.apache.org/docs/install/from_source.html) 中的说明编译 TVM 运行时，并直接跳到步骤 [3. 启动RPC服务器](#3-启动rpc服务器)。

### 1. 交叉编译 TVM 运行时

我们使用 CMake 来管理编译过程，对于交叉编译，CMake 需要一个工具链文件来获取所需的信息，因此您需要根据您的设备平台准备这个文件。以下是一个针对设备机器的示例，该设备的 CPU 是64位 ARM 架构，操作系统是 Linux。

```shell
set(CMAKE_SYSTEM_NAME Linux)
set(root_dir "/XXX/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu")

set(CMAKE_C_COMPILER "${root_dir}/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${root_dir}/bin/aarch64-linux-gnu-g++")
set(CMAKE_SYSROOT "${root_dir}/aarch64-linux-gnu/libc")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

在TVM存储库的根目录下执行类似以下内容的命令后，运行时将成功进行交叉编译，请根据您的具体要求在文件 ``config.cmake`` 中启用其他所需选项。

```shell
$ mkdir cross_build
$ cd cross_build
$ cp ../cmake/config.cmake ./

# 你可能需要打开其他选项，例如 USE_OPENCL， USE_xPU。
$ sed -i "s|USE_LLVM.*)|USE_LLVM OFF)|" config.cmake
$ sed -i "s|USE_LIBBACKTRACE.*)|USE_LIBBACKTRACE OFF)|" config.cmake
$ sed -i "s|USE_MICRO.*)|USE_MICRO OFF)|" config.cmake

$ cmake -DCMAKE_TOOLCHAIN_FILE=/YYY/aarch64-linux-gnu.cmake -DCMAKE_BUILD_TYPE=Release ..
$ cmake --build . -j -- runtime
$ cd ..

```

### 2. 打包并部署到设备机器

通过以下类似的命令打包 Python 版本的 RPC 服务器。

```shell
$ git clean -dxf python
$ cp cross_build/libtvm_runtime.so python/tvm/
$ tar -czf tvm_runtime.tar.gz python
```

然后将压缩包 ``tvm_runtime.tar.gz`` 复制到您的具体设备机器，并通过以下类似的命令在您的设备机器上正确设置环境变量 ``PYTHONPATH``。

```shell
$ tar -xzf tvm_runtime.tar.gz
$ export PYTHONPATH=`pwd`/python:${PYTHONPATH}
```

.. _luanch-rpc-server:

### 3. 启动 RPC 服务器

可以通过以下类似的命令在您的设备机器上启动 RPC 服务器，请根据您的具体环境修改 *RPC_TRACKER_IP*、*RPC_TRACKER_PORT*、*RPC_PROXY_IP*、*RPC_PROXY_PORT* 和 *RPC_KEY*。

```shell
# 如果使用 RPC 代理，请使用此命令。
$ python3 -m tvm.exec.rpc_server --host RPC_PROXY_IP --port RPC_PROXY_PORT --through-proxy --key RPC_KEY
# 如果不使用 RPC 代理，请使用此命令。
$ python3 -m tvm.exec.rpc_server --tracker RPC_TRACKER_IP:RPC_TRACKER_PORT --key RPC_KEY
```

## 验证RPC系统

```shell
$ python3 -m tvm.exec.query_rpc_tracker --host RPC_TRACKER_IP --port RPC_TRACKER_PORT
```

通过上述命令，我们可以查询所有可用的 RPC 服务器和队列状态。如果您有3个通过 RPC 代理连接到 RPC 追踪器的 RPC 服务器，则输出应该类似于以下内容。

```shell
Tracker address RPC_TRACKER_IP:RPC_TRACKER_PORT

Server List
----------------------------
server-address  key
----------------------------
RPC_PROXY_IP:RPC_PROXY_PORT       server:proxy[RPC_KEY0,RPC_KEY1,RPC_KEY2]
----------------------------

Queue Status
---------------------------------------
key               total  free  pending
---------------------------------------
RPC_KEY0          0      0     3
---------------------------------------
```

## 故障排除
### 1. 在设备机器上缺少 ``numpy`` 导致无法启动RPC服务器。
``numpy`` 包在一些 RPC 服务器依赖的 Python 文件中被导入，而消除导入关系是困难的，对于一些设备来说，交叉编译 ``numpy`` 也是非常困难的。

但实际上，TVM 运行时并不真正依赖于 ``numpy``，因此一个非常简单的解决方法是创建一个虚拟的 ``numpy``，只需要将下面的内容复制到一个名为 ``numpy.py`` 的文件中，并将其放置在类似 ``/usr/local/lib/python3.8/site-packages`` 的目录中。

```python

  class bool_:
    pass
  class int8:
      pass
  class int16:
      pass
  class int32:
      pass
  class int64:
      pass
  class uint8:
      pass
  class uint16:
      pass
  class uint32:
      pass
  class uint64:
      pass
  class float16:
      pass
  class float32:
      pass
  class float64:
      pass
  class float_:
      pass

  class dtype:
      def __init__(self, *args, **kwargs):
          pass

  class ndarray:
      pass

  def sqrt(*args, **kwargs):
      pass

  def log(*args, **kwargs):
      pass

  def tanh(*args, **kwargs):
      pass

  def power(*args, **kwargs):
      pass

  def exp(*args, **kwargs):
      pass

```

### 2. 在设备机器上缺少 ``cloudpickle`` 导致无法启动RPC服务器。

因为 ``cloudpickle`` 包是一个纯 Python 包，所以只需将它从其他机器复制到设备机器的类似 ``/usr/local/lib/python3.8/site-packages`` 目录，问题就可以得到解决。