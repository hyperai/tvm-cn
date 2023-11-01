---
title: 交叉编译和 RPC
---

# 交叉编译和 RPC

:::note
单击 [此处](https://tvm.apache.org/docs/tutorial/cross_compilation_and_rpc.html#sphx-glr-download-tutorial-cross-compilation-and-rpc-py) 下载完整的示例代码
:::

**作者**：[Ziheng Jiang](https://github.com/ZihengJiang/)，[Lianmin Zheng](https://github.com/merrymercy/)

本教程介绍了如何在 TVM 中使用 RPC 进行交叉编译和远程设备执行。

利用交叉编译和 RPC，可以实现程序在本地机器编译，在远程设备运行。这个特性在远程设备资源有限时（如在树莓派和移动平台上）尤其有用。本教程将把树莓派作为 CPU 示例，把 Firefly-RK3399 作为 OpenCL 示例进行演示。

## 在设备上构建 TVM Runtime

首先在远程设备上构建 TVM runtime。

:::note 注意
本节和下一节中的所有命令都应在目标设备（例如树莓派）上执行。假设目标设备运行 Linux 系统。
:::

由于在本地机器上只做编译，而远程设备用于运行生成的代码。所以只需在远程设备上构建 TVM runtime。

``` bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
make runtime -j2
```

成功构建 runtime 后，要在 `~/.bashrc` 文件中设置环境变量。可以用 `vi ~/.bashrc`命令编辑 `~/.bashrc`，在这个文件里添加下面这行代码（假设 TVM 目录在 `~/tvm` 中）：

``` bash
export PYTHONPATH=$PYTHONPATH:~/tvm/python
```

执行 `source ~/.bashrc` 来更新环境变量。

## 在设备上设置 RPC 服务器

在远程设备（本例为树莓派）上运行以下命令来启动 RPC 服务器：

``` bash
python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
```

看到下面这行提示，则表示 RPC 服务器已成功启动。

``` bash
INFO:root:RPCServer: bind to 0.0.0.0:9090
```

## 在本地机器上声明和交叉编译内核

:::note
现在回到本地机器（已经用 LLVM 安装了完整的 TVM）。
:::

在本地机器上声明一个简单的内核：

``` python
import numpy as np

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
s = te.create_schedule(B.op)
```

然后交叉编译内核。对于树莓派 3B，target 是“llvm -mtriple=armv7l-linux-gnueabihf”，但这里用的是“llvm”，使得本教程可以在网页构建服务器上运行。请参阅下面的详细说明。

``` python
local_demo = True

if local_demo:
    target = "llvm"
else:
    target = "llvm -mtriple=armv7l-linux-gnueabihf"

func = tvm.build(s, [A, B], target=target, name="add_one")
# 将 lib 存储在本地临时文件夹
temp = utils.tempdir()
path = temp.relpath("lib.tar")
func.export_library(path)
```

:::note
要使本教程运行在真正的远程设备上，需要将 `local_demo` 改为 False，并将 `build` 中的 `target` 替换为适合设备的 target 三元组。不同设备的 target 三元组可能不同。例如，对于树莓派 3B，它是 `llvm -mtriple=armv7l-linux-gnueabihf`；对于 RK3399，它是 `llvm -mtriple=aarch64-linux-gnu`。

通常，可以在设备上运行 `gcc -v` 来查询 target，寻找以 `Target` 开头的行：（尽管它可能仍然是一个松散的配置。）

除了 `-mtriple`，还可设置其他编译选项，例如：

* **`-mcpu=<cpuname\>`**

  指定生成的代码运行的芯片架构。默认情况这是从 target 三元组推断出来的，并自动检测到当前架构。

* **`-mattr=a1,+a2,-a3,…`**

  覆盖或控制 target 的指定属性，例如是否启用 SIMD 操作。默认属性集由当前 CPU 设置。要获取可用属性列表，执行：

``` bash
  llc -mtriple=<your device target triple> -mattr=help
```

这些选项与 [llc](http://llvm.org/docs/CommandGuide/llc.html) 一致。建议设置 target 三元组和功能集，使其包含可用的特定功能，这样我们可以充分利用单板的功能。查看 [LLVM 交叉编译指南](https://clang.llvm.org/docs/CrossCompilation.html) 获取有关交叉编译属性的详细信息。
:::

## 通过 RPC 远程运行 CPU 内核

下面将演示如何在远程设备上运行生成的 CPU 内核。首先，从远程设备获取 RPC 会话：

``` python
if local_demo:
    remote = rpc.LocalSession()
else:
    # 下面是我的环境，将这个换成你目标设备的 IP 地址
    host = "10.77.1.162"
    port = 9090
    remote = rpc.connect(host, port)
```

将 lib 上传到远程设备，然后调用设备的本地编译器重新链接它们。其中 *func* 是一个远程模块对象。

``` python
remote.upload(path)
func = remote.load_module("lib.tar")

# 在远程设备上创建数组
dev = remote.cpu()
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), dev)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
# 这个函数将在远程设备上运行
func(a, b)
np.testing.assert_equal(b.numpy(), a.numpy() + 1)
```

要想评估内核在远程设备上的性能，避免网络开销很重要。`time_evaluator` 返回一个远程函数，这个远程函数多次运行 func 函数，并测试每一次在远程设备上运行的成本，然后返回测试的成本（不包括网络开销）。

``` python
time_f = func.time_evaluator(func.entry_name, dev, number=10)
cost = time_f(a, b).mean
print("%g secs/op" % cost)
```

输出结果：

``` bash
1.369e-07 secs/op
```

## 通过 RPC 远程运行 OpenCL 内核

远程 OpenCL 设备的工作流程与上述内容基本相同。可以定义内核、上传文件，然后通过 RPC 运行。

:::note
树莓派不支持 OpenCL，下面的代码是在 Firefly-RK3399 上测试的。可以按照 [教程](https://gist.github.com/mli/585aed2cec0b5178b1a510f9f236afa2) 为 RK3399 设置 OS 及 OpenCL 驱动程序。

在 rk3399 板上构建 runtime 也需启用 OpenCL。在 TVM 根目录下执行：

``` bash
cp cmake/config.cmake .
sed -i "s/USE_OPENCL OFF/USE_OPENCL ON/" config.cmake
make runtime -j4
```
:::

下面的函数展示了如何远程运行 OpenCL 内核：

``` python
def run_opencl():
    # 注意：这是 rk3399 板的设置。你需要根据你的环境进行修改
    opencl_device_host = "10.77.1.145"
    opencl_device_port = 9090
    target = tvm.target.Target("opencl", host="llvm -mtriple=aarch64-linux-gnu")

    # 为上面的计算声明 "add one" 创建 schedule
    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=32)
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    s[B].bind(xi, te.thread_axis("threadIdx.x"))
    func = tvm.build(s, [A, B], target=target)

    remote = rpc.connect(opencl_device_host, opencl_device_port)

    # 导出并上传
    path = temp.relpath("lib_cl.tar")
    func.export_library(path)
    remote.upload(path)
    func = remote.load_module("lib_cl.tar")

    # 运行
    dev = remote.cl()
    a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), dev)
    b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
    func(a, b)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)
    print("OpenCL test passed!")
```

## 总结

本教程介绍了 TVM 中的交叉编译和 RPC 功能。

* 在远程设备上设置 RPC 服务器。
* 设置目标设备配置，使得可在本地机器上交叉编译内核。
* 通过 RPC API 远程上传和运行内核。

[下载 Python 源代码：cross_compilation_and_rpc.py](https://tvm.apache.org/docs/_downloads/766206ab8f1fd80ac34d9816cb991a0d/cross_compilation_and_rpc.py)

[下载 Jupyter Notebook：cross_compilation_and_rpc.ipynb](https://tvm.apache.org/docs/_downloads/f289ca2466fcf79c024068c1f8642bd0/cross_compilation_and_rpc.ipynb)
