---

title: 设置 RPC 系统

---

## 概述


远程过程调用（RPC）是 Apache TVM 中非常重要和有用的功能，它允许我们在真实硬件上运行已编译好的神经网络（NN）模型，无需手动操作远程设备，输出结果会通过网络自动返回。


通过省去一些人工操作，比如将输入数据转储到文件、将导出的神经网络模型拷贝到远程设备、设置设备用户环境、将输出结果拷贝回主机开发环境，RPC 极大地提升了开发效率。


此外，由于只有已编译神经网络模型的执行部分运行在远程设备上，所有其他部分都运行在主机开发环境中，因此可以使用任何 Python 包来完成预处理和后处理工作。


RPC 在以下两种情况下特别有用：
* **硬件资源有限时**

RPC 的队列和资源管理机制能够让硬件设备为众多开发者和测试任务提供服务，确保编译后的神经网络模型正常运行。
* **端到端评估的早期阶段**

除了已编译的神经网络模型，其他所有处理都在主机开发环境中运行，因此可以轻松实现复杂的预处理和后处理逻辑。 


## 推荐的架构

Apache TVM 的 RPC 系统包含三个工具：RPC Tracker、RPC Proxy 和 RPC Server。RPC Server 是必要组件，一个基本的 RPC 系统即使没有 RPC Proxy 和 RPC Tracker 也可以正常运行。RPC Proxy 在无法直接访问 RPC Server 时使用。强烈推荐在系统中添加 RPC Tracker，它提供了许多有用的功能，例如：队列能力、多 RPC Server 管理、通过 key 而非 IP 地址管理 RPC Server。

![图片](/img/docs/v21/02-how-to_05-development-guides_02-setup_rpc_system_1.png)

如图所示，由于机器 A 与机器 C、D 之间没有物理连接通道，因此我们在机器 B 上设置了一个 RPC Proxy。RPC Tracker 会为每个 RPC key 管理一个请求队列。任何用户都可以随时通过一个 RPC key 请求一个 RPC Server，如果有空闲的匹配 key 的 RPC Server，则分配给用户，否则请求将排入队列，稍后再检查是否可用。



## 设置 RPC 追踪器和 RPC 代理

一般来说，RPC 追踪器和 RPC 代理只需在主机（例如开发服务器或 PC）上运行，它们不依赖于设备端环境。只需根据 [TVM 文档](/docs/getting-started/installing-tvm/)完成 TVM 安装后，在相应主机上执行以下命令即可：
* RPC 追踪器

```plain
$ python3 -m tvm.exec.rpc_tracker --host RPC_TRACKER_IP --port 9190 --port-end 9191
```
* RPC 代理

```plain
$ python3 -m tvm.exec.rpc_proxy --host RPC_PROXY_IP --port 9090 --port-end 9091 --tracker RPC_TRACKER_IP:RPC_TRACKER_PORT
```


请根据实际环境替换命令中的 RPC_TRACKER_IP、RPC_TRACKER_PORT、RPC_PROXY_IP 及端口号。`port-end` 参数可避免服务绑定到意外的端口号，这可能导致其他服务无法正确连接，特别是在自动化测试系统中，这一点尤为重要。




## 设置 RPC 服务器

TVM 社区提供了多个 RPC 服务器实现，例如：`apps/android_rpc`、`apps/cpp_rpc`、`apps/ios_rpc`。以下内容只聚焦于 Python 版本的 RPC Server（`python/tvm/exec/rpc_server.py`）。其他版本请参考对应目录下的文档。


RPC Server 需运行在设备端，通常依赖于 xPU 驱动、增强版支持 xPU 的 TVM 运行时及其他库，因此请先安装相关依赖组件，例如 KMD 驱动，并确保所需动态库路径已加入 `LD_LIBRARY_PATH` 环境变量。


如果可以在设备上直接设置编译环境（无需交叉编译），请参考 [/docs/getting-started/installing-tvm/install-from-source](/docs/getting-started/installing-tvm/install-from-source) 完成编译，并直接跳转至 [步骤 3：启动 RPC Server](/docs/how-to/development-guides/setup_rpc_system#3-%E5%90%AF%E5%8A%A8-rpc-server)。

### 1. 交叉编译 TVM 运行时

我们使用 CMake 来管理编译过程，对于交叉编译，CMake 需要一个工具链文件来获取所需的信息，因此您需要根据设备平台准备该文件。下面是一个示例，适用于 CPU 为 64 位 ARM 架构且操作系统为 Linux 的设备机器。

```plain
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


在 TVM 源码根目录下执行以下命令可完成交叉编译（根据需要在 `config.cmake` 中启用额外选项）：

```plain
$ mkdir cross_build
$ cd cross_build
$ cp ../cmake/config.cmake ./

# 可能需要启用如 USE_OPENCL、USE_xPU 的选项。
$ sed -i "s|USE_LLVM.*)|USE_LLVM OFF)|" config.cmake
$ sed -i "s|USE_LIBBACKTRACE.*)|USE_LIBBACKTRACE OFF)|" config.cmake

$ cmake -DCMAKE_TOOLCHAIN_FILE=/YYY/aarch64-linux-gnu.cmake -DCMAKE_BUILD_TYPE=Release ..
$ cmake --build . -j -- runtime
$ cd ..
```


### **2.打包并部署到设备端**

使用如下命令打包 Python 版本 RPC Server：

```plain
$ git clean -dxf python
$ cp cross_build/libtvm_runtime.so python/tvm/
$ tar -czf tvm_runtime.tar.gz python
```


然后将压缩包 `tvm_runtime.tar.gz` 复制到你的具体设备上，并在该设备上通过如下所示的命令正确设置环境变量 `PYTHONPATH`。

```plain
$ tar -xzf tvm_runtime.tar.gz
$ export PYTHONPATH=`pwd`/python:${PYTHONPATH}
```


### 3. 启动 RPC Server

根据环境修改以下命令中的 RPC_TRACKER_IP、RPC_TRACKER_PORT、RPC_PROXY_IP、RPC_PROXY_PORT 和 RPC_KEY，并在设备端执行启动：

```plain
# 如果使用了 RPC Proxy：
$ python3 -m tvm.exec.rpc_server --host RPC_PROXY_IP --port RPC_PROXY_PORT --through-proxy --key RPC_KEY
# 如果未使用 RPC Proxy：
$ python3 -m tvm.exec.rpc_server --tracker RPC_TRACKER_IP:RPC_TRACKER_PORT --key RPC_KEY
```


### 验证 RPC 系统

```plain
$ python3 -m tvm.exec.query_rpc_tracker --host RPC_TRACKER_IP --port RPC_TRACKER_PORT
```


通过上述命令，我们可以查询所有可用的 RPC 服务器以及队列状态。如果你有 3 个通过 RPC 代理连接到 RPC 跟踪器的 RPC 服务器，那么输出结果应如下所示。

```plain
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


## 故障排查

### 1. 由于设备上缺少 `numpy`，导致无法启动 RPC Server


在一些依赖于 RPC 服务器的 Python 文件中，导入了 `numpy` 包，而消除这种导入关系是比较困难的。对于某些设备来说，交叉编译 `numpy` 也非常困难。

TVM Runtime 实际并不依赖 `numpy`，因此一个简单的解决方法是创建一个「空」的 `numpy`。只需将以下内容保存为 `numpy.py` 文件，放入如 `/usr/local/lib/python3.9/site-packages` 等路径中即可：

```plain
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


### 2. 由于设备端缺少 `cloudpickle`，导致 RPC 服务器无法启动。

由于 `cloudpickle` 是一个纯 Python 包，因此只需将其从其他机器复制到设备机器上的目录（例如 `/usr/local/lib/python3.9/site-packages`），即可解决该问题。


