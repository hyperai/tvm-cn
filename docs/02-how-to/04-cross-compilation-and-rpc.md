---

title: 交叉编译与 RPC

---


:::note

本教程可通过 Google Colab 交互式运行！也可点击[此处](https://tvm.apache.org/docs/how_to/tutorials/optimize_llm.html#sphx-glr-download-how-to-tutorials-optimize-llm-py)在本地运行 Jupyter Notebook。

[在 Google Colab 中打开](https://colab.research.google.com/github/apache/tvm-site/blob/asf-site/docs/_downloads/148819f3421b8d89b1723c3e15e3f19f/cross_compilation_and_rpc.ipynb)

:::


**作者**：[Ziheng Jiang](https://github.com/ZihengJiang/)、[Lianmin Zheng](https://github.com/merrymercy/)


本教程介绍了在 TVM 中使用交叉编译与 RPC 进行远程设备执行的方法。


通过交叉编译与 RPC，**你可以在本地机器上编译程序，并在远程设备上运行。** 这对于资源有限的远程设备（如 Raspberry Pi 或移动平台）非常有用。本教程将以 Raspberry Pi 作为 CPU 示例，以 Firefly-RK3399 作为 OpenCL 示例。


## 在设备上构建 TVM Runtime

第一步是在远程设备上构建 TVM 运行时。


:::note

本节与下一节中的所有命令都应在目标设备（例如 Raspberry Pi ）上执行。我们假设目标设备运行的是 Linux 系统。

:::


由于编译工作是在本地机器上完成的，远程设备仅用于运行生成的代码，因此只需要在远程设备上构建 TVM 的运行时部分。

```plain
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
make runtime -j2
```


成功构建运行时后，需要在 `~/.bashrc` 文件中设置环境变量。可以使用 `vi ~/.bashrc` 编辑 `~/.bashrc`，并添加以下行（假设你的 TVM 路径为 `~/tvm`）：


```plain
export PYTHONPATH=$PYTHONPATH:~/tvm/python
```



然后执行`source ~/.bashrc`命令以更新环境变量：


## 在设备上启动 RPC 服务端

在远程设备上（如 Raspberry Pi ）运行以下命令以启动 RPC 服务器：


```plain
python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
```


如果你看到如下输出，说明 RPC 服务器已成功启动：


```plain
INFO:root:RPCServer: bind to 0.0.0.0:9090
```


## 在本地机器上声明并交叉编译内核


:::note

现在回到本地机器，假设本地已完整安装了 TVM（带 LLVM 支持）。

:::


在本地机器上声明一个简单的计算内核：

```plain
import numpy as np

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]).with_attr("global_symbol", "add_one"))
```


然后进行交叉编译。对于 Raspberry Pi  3B，目标应为「llvm -mtriple=armv7l-linux-gnueabihf」，但为了方便在网页构建服务器上运行示例，这里使用「llvm」作为默认目标。详细事项可参考下一块。


```plain
local_demo = True

if local_demo:
    target = "llvm"
else:
    target = "llvm -mtriple=armv7l-linux-gnueabihf"

func = tvm.compile(mod, target=target)
# 将库保存到临时目录
temp = utils.tempdir()
path = temp.relpath("lib.tar")
func.export_library(path)
```


:::note

如果要使用真实的远程设备运行此教程，请将 `local_demo` 设置为 False，并将 `target` 替换为适用于你设备的目标三元组。不同设备的目标三元组可能有所不同。例如，对于 Raspberry Pi 3B，目标三元组是：`「llvm -mtriple=armv7l-linux-gnueabihf」`；对于 RK3399，目标三元组是：`「llvm -mtriple=aarch64-linux-gnu」`。


你可以通过在目标设备上运行 `gcc -v` 来查询其目标三元组，查看输出中以 `Target:` 开头的行（但这也可能只是一个宽松的配置）。


除了 `-mtriple`，你还可以设置其他编译选项，例如：
* -mcpu=\<cpuname\>
   * 指定要生成代码的具体芯片。默认情况下会根据目标三元组推断并自动检测
* -mattr=a1,+a2,-a3,…
   * 覆盖或控制目标的具体属性，例如是否启用 SIMD 操作。默认属性由当前 CPU 决定，你可以运行以下命令查看支持的属性：

```plain
llc -mtriple=<your device target triple> -mattr=help
```


这些选项与 [llc 工具](http://llvm.org/docs/CommandGuide/llc.html) 保持一致。建议设置目标三元组和特性集以包含具体设备可用的功能，以充分发挥硬件性能。更多交叉编译属性详见 [LLVM 跨平台编译文档](https://clang.llvm.org/docs/CrossCompilation.html)。

:::


## 通过 RPC 远程运行 CPU 内核

本节展示如何在远程设备上运行生成的 CPU 内核。首先，我们需要从远程设备获取一个 RPC 会话：

```plain
if local_demo:
    remote = rpc.LocalSession()
else:
    # 以下是我的环境，请替换为你的目标设备的 IP 地址
    host = "10.77.1.162"
    port = 9090
    remote = rpc.connect(host, port)
```



接下来将生成的库上传到远程设备，然后调用设备上的编译器进行重新链接。此时 `func` 就是一个远程模块对象。


```plain
remote.upload(path)
func = remote.load_module("lib.tar")

# 在远程设备上创建数组
dev = remote.cpu()
a = tvm.runtime.tensor(np.random.uniform(size=1024).astype(A.dtype), dev)
b = tvm.runtime.tensor(np.zeros(1024, dtype=A.dtype), dev)
# 函数将在远程设备上运行
func(a, b)
np.testing.assert_equal(b.numpy(), a.numpy() + 1)
```


当你想评估内核在远程设备上的性能时，需要避免网络传输带来的开销。`time_evaluator` 返回一个远程函数，该函数会运行多次并测量每次执行的耗时（不包括网络延迟）：


```plain
time_f = func.time_evaluator(func.entry_name, dev, number=10)
cost = time_f(a, b).mean
print("%g secs/op" % cost)
```
输出：
```plain
1.452e-07 secs/op
```


## 通过 RPC 远程运行 OpenCL 内核

对于远程 OpenCL 设备，整体流程和前面几乎一致：定义内核、上传文件并通过 RPC 运行。


:::note

Raspberry Pi 不支持 OpenCL，以下代码在 Firefly-RK3399 上测试通过。你可以参考这个 [教程](https://gist.github.com/mli/585aed2cec0b5178b1a510f9f236afa2) 设置 RK3399 的操作系统和 OpenCL 驱动。


同时，需要在 RK3399 上启用 OpenCL 构建 TVM 运行时。在 TVM 根目录下执行：

:::


```plain
cp cmake/config.cmake .
sed -i "s/USE_OPENCL OFF/USE_OPENCL ON/" config.cmake
make runtime -j4
```


以下函数展示了如何远程运行一个 OpenCL 内核：


```plain
def run_opencl():
    # 注意：这是我 rk3399 的设置，请根据你的设备环境进行修改
    opencl_device_host = "10.77.1.145"
    opencl_device_port = 9090
    target = tvm.target.Target("opencl", host="llvm -mtriple=aarch64-linux-gnu")

    # 创建上述「加一」计算的调度
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]))
    sch = tvm.tir.Schedule(mod)
    (x,) = sch.get_loops(block=sch.get_block("B"))
    xo, xi = sch.split(x, [None, 32])
    sch.bind(xo, "blockIdx.x")
    sch.bind(xi, "threadIdx.x")
    func = tvm.compile(sch.mod, target=target)

    remote = rpc.connect(opencl_device_host, opencl_device_port)

    # 导出并上传
    path = temp.relpath("lib_cl.tar")
    func.export_library(path)
    remote.upload(path)
    func = remote.load_module("lib_cl.tar")

    # 运行
    dev = remote.cl()
    a = tvm.runtime.tensor(np.random.uniform(size=1024).astype(A.dtype), dev)
    b = tvm.runtime.tensor(np.zeros(1024, dtype=A.dtype), dev)
    func(a, b)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)
    print("OpenCL test passed!")
```


## 总结


本教程完整展示了 TVM 中交叉编译和 RPC 功能的使用流程：
*  在远程设备上设置 RPC 服务器；
*  在本地设置目标设备的交叉编译配置；
*  通过 RPC API 上传并远程运行内核程序。


[下载 Jupyter notebook: ](https://tvm.apache.org/docs/_downloads/148819f3421b8d89b1723c3e15e3f19f/cross_compilation_and_rpc.ipynb)`cross_compilation_and_rpc.ipynb`

[下载 Python 源码: ](https://tvm.apache.org/docs/_downloads/3cbcc56110528f886a987b8b251e7c88/cross_compilation_and_rpc.py)`cross_compilation_and_rpc.py`

[下载压缩包: ](https://tvm.apache.org/docs/_downloads/f69380821f417ef2210f45503d81bded/cross_compilation_and_rpc.zip)`cross_compilation_and_rpc.zip`
