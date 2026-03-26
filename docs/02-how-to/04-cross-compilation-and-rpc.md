---

title: 交叉编译与 RPC

---


:::note

本教程可通过 Google Colab 交互式运行！也可点击[此处](/docs/how-to/cross-compilation-and-rpc#总结)在本地运行 Jupyter Notebook。

[在 Google Colab 中打开](https://colab.research.google.com/github/apache/tvm-site/blob/asf-site/docs/_downloads/148819f3421b8d89b1723c3e15e3f19f/cross_compilation_and_rpc.ipynb)

:::


**作者**：[Ziheng Jiang](https://github.com/ZihengJiang/)、[Lianmin Zheng](https://github.com/merrymercy/)


本教程介绍了在 TVM 中使用交叉编译与 RPC 进行远程设备执行的方法。


通过交叉编译与 RPC，**你可以在本地机器上编译程序，并在远程设备上运行。** 这对于资源有限的远程设备（如 Raspberry Pi 或移动平台）非常有用。本教程将以 Raspberry Pi 作为 CPU 示例，以 Firefly-RK3399 作为 OpenCL 示例。


## 在设备上构建 TVM Runtime

第一步是在远程设备上构建 TVM 运行时。


:::note

本节与下一节中的所有命令都应在目标设备（例如 Raspberry Pi）上执行。我们假设目标设备运行的是 Linux 系统。

:::


由于编译工作是在本地机器上完成的，远程设备仅用于运行生成的代码，因此只需要在远程设备上构建 TVM 的运行时部分。

```bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir build && cd build
cp ../cmake/config.cmake .
cmake .. && cmake --build . --parallel $(nproc)
```


成功构建运行时后，需要在 `~/.bashrc` 文件中设置环境变量。可以使用 `vi ~/.bashrc` 编辑 `~/.bashrc`，并添加以下行（假设你的 TVM 路径为 `~/tvm`）：


```bash
export PYTHONPATH=$PYTHONPATH:~/tvm/python
```



然后执行 `source ~/.bashrc` 命令以更新环境变量。


## 在设备上启动 RPC 服务端

在远程设备上（如 Raspberry Pi）运行以下命令以启动 RPC 服务器：


```bash
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

```python
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


然后进行交叉编译。对于 Raspberry Pi 3B，目标应为 `{"kind": "llvm", "mtriple": "armv7l-linux-gnueabihf"}`，但为了方便在网页构建服务器上运行示例，这里使用 `"llvm"` 作为默认目标。详细事项可参考下面的说明。


```python
local_demo = True

if local_demo:
    target = "llvm"
else:
    target = {"kind": "llvm", "mtriple": "armv7l-linux-gnueabihf"}

func = tvm.compile(mod, target=target)
# 将库保存到临时目录
temp = utils.tempdir()
path = temp.relpath("lib.tar")
func.export_library(path)
```


:::note

如果要使用真实的远程设备运行此教程，请将 `local_demo` 设置为 False，并将 `target` 替换为适用于你设备的目标配置。不同设备的目标配置可能有所不同。例如，对于 Raspberry Pi 3B，目标为 `{"kind": "llvm", "mtriple": "armv7l-linux-gnueabihf"}`；对于 RK3399，目标为 `{"kind": "llvm", "mtriple": "aarch64-linux-gnu"}`。


你可以通过在目标设备上运行 `gcc -v` 来查询其目标三元组，查看输出中以 `Target:` 开头的行（但这也可能只是一个宽松的配置）。


除了 `-mtriple`，你还可以设置其他编译选项，例如：
* -mcpu=\<cpuname\>
   * 指定要生成代码的具体芯片。默认情况下会根据目标三元组推断并自动检测
* -mattr=a1,+a2,-a3,…
   * 覆盖或控制目标的具体属性，例如是否启用 SIMD 操作。默认属性由当前 CPU 决定，你可以运行以下命令查看支持的属性：

```bash
llc -mtriple=<your device target triple> -mattr=help
```


这些选项与 [llc 工具](http://llvm.org/docs/CommandGuide/llc.html) 保持一致。建议设置目标三元组和特性集以包含具体设备可用的功能，以充分发挥硬件性能。更多交叉编译属性详见 [LLVM 跨平台编译文档](https://clang.llvm.org/docs/CrossCompilation.html)。

:::


## 通过 RPC 远程运行 CPU 内核

本节展示如何在远程设备上运行生成的 CPU 内核。首先，我们需要从远程设备获取一个 RPC 会话：

```python
if local_demo:
    remote = rpc.LocalSession()
else:
    # 以下是我的环境，请替换为你的目标设备的 IP 地址
    host = "10.77.1.162"
    port = 9090
    remote = rpc.connect(host, port)
```



接下来将生成的库上传到远程设备，然后调用设备上的编译器进行重新链接。此时 `func` 就是一个远程模块对象。


```python
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


```python
time_f = func.time_evaluator(func.entry_name, dev, number=10)
cost = time_f(a, b).mean
print(f"{cost:g} secs/op")
```


## 通过 RPC 远程运行 OpenCL 内核

对于远程 OpenCL 设备，整体流程和前面几乎一致：定义内核、上传文件并通过 RPC 运行。


:::note

Raspberry Pi 不支持 OpenCL，以下代码在 Firefly-RK3399 上测试通过。你可以参考这个 [教程](https://gist.github.com/mli/585aed2cec0b5178b1a510f9f236afa2) 设置 RK3399 的操作系统和 OpenCL 驱动。

同时，需要在 RK3399 上启用 OpenCL 构建 TVM 运行时。在 TVM 根目录下执行：

```bash
mkdir -p build && cd build
cp ../cmake/config.cmake .
sed -i "s/USE_OPENCL OFF/USE_OPENCL ON/" config.cmake
cmake .. && cmake --build . --parallel $(nproc)
```

:::


以下函数展示了如何远程运行一个 OpenCL 内核：


```python
def run_opencl():
    # 注意：这是我 rk3399 的设置，请根据你的设备环境进行修改
    opencl_device_host = "10.77.1.145"
    opencl_device_port = 9090
    target = tvm.target.Target(
        "opencl", host={"kind": "llvm", "mtriple": "aarch64-linux-gnu"}
    )

    # 创建上述「加一」计算的调度
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]))
    sch = tvm.s_tir.Schedule(mod)
    (x,) = sch.get_loops(block=sch.get_sblock("B"))
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


## 通过 RPC 将 PyTorch 模型部署到远程设备

上面的示例展示了使用低层 TensorIR（通过 TE）进行交叉编译和 RPC。对于从 PyTorch 或 ONNX 等框架部署完整的神经网络模型，TVM 的 Relax 提供了更适合端到端模型编译的高层抽象。

本节展示一个将模型部署到**任意远程设备**的现代工作流：

1. 导入 PyTorch 模型并转换为 Relax
2. 为目标架构（ARM、x86、RISC-V 等）交叉编译
3. 通过 RPC 部署到远程设备
4. 远程运行推理

此工作流适用于多种部署场景：

- **ARM 设备**：Raspberry Pi、NVIDIA Jetson、手机
- **x86 服务器**：远程 Linux 服务器、云实例
- **嵌入式系统**：RISC-V 开发板、自定义硬件
- **加速器**：配备 GPU、TPU 或其他加速器的远程机器

:::note
本示例以 PyTorch 为例，但工作流对 ONNX 模型完全相同。只需将 `from_exported_program()` 替换为 `from_onnx(model, keep_params_in_input=True)`，然后按相同步骤操作。
:::

```python
try:
    import torch
    from torch.export import export
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def run_pytorch_model_via_rpc():
    """展示将 PyTorch 模型通过 RPC 部署到远程设备的完整工作流。"""
    if not HAS_TORCH:
        print("跳过 PyTorch 示例（未安装 PyTorch）")
        return

    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program

    ######################################################################
    # 第 1 步：定义并导出 PyTorch 模型
    class TorchMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(28 * 28, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10),
            )

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            return self.net(data)

    # 使用 PyTorch 2.x 导出 API 导出模型
    torch_model = TorchMLP().eval()
    example_args = (torch.randn(1, 1, 28, 28, dtype=torch.float32),)

    with torch.no_grad():
        exported_program = export(torch_model, example_args)

    ######################################################################
    # 第 2 步：转换为 Relax 并准备编译
    mod = from_exported_program(exported_program, keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)

    print("已将 PyTorch 模型转换为 Relax：")
    print(f"  - 参数数量：{len(params['main'])}")

    ######################################################################
    # 第 3 步：为目标设备交叉编译
    if local_demo:
        target = tvm.target.Target("llvm")
        print("使用本地目标进行演示")
    else:
        # 根据你的设备选择合适的目标：
        #
        # ARM 设备：
        #   - Raspberry Pi 3/4（32位）: {"kind": "llvm", "mtriple": "armv7l-linux-gnueabihf"}
        #   - Raspberry Pi 4（64位）/ Jetson: {"kind": "llvm", "mtriple": "aarch64-linux-gnu"}
        #   - Android: {"kind": "llvm", "mtriple": "aarch64-linux-android"}
        #
        # x86 服务器：
        #   - Linux x86_64: {"kind": "llvm", "mtriple": "x86_64-linux-gnu"}
        #   - 带 AVX-512: {"kind": "llvm", "mtriple": "x86_64-linux-gnu", "mcpu": "skylake-avx512"}
        #
        # RISC-V：
        #   - RV64: {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu"}
        #
        # GPU 目标：
        #   - CUDA: tvm.target.Target("cuda", host={"kind": "llvm", "mtriple": "x86_64-linux-gnu"})
        #   - OpenCL: tvm.target.Target("opencl", host={"kind": "llvm", "mtriple": "aarch64-linux-gnu"})
        target = tvm.target.Target({"kind": "llvm", "mtriple": "aarch64-linux-gnu"})
        print(f"为目标交叉编译：{target}")

    # 应用优化管线
    pipeline = relax.get_pipeline()
    with target:
        built_mod = pipeline(mod)

    # 编译为可执行文件
    executable = tvm.compile(built_mod, target=target)

    # 导出为共享库
    lib_path = temp.relpath("model_deployed.so")
    executable.export_library(lib_path)
    print(f"已导出库到：{lib_path}")

    # 单独保存参数
    params_path = temp.relpath("model_params.npz")
    param_arrays = {f"p_{i}": p.numpy() for i, p in enumerate(params["main"])}
    np.savez(params_path, **param_arrays)
    print(f"已保存参数到：{params_path}")

    ######################################################################
    # 第 4 步：通过 RPC 部署到远程设备
    if local_demo:
        print("\nRPC 工作流（适用于任何远程设备）：")
        print("=" * 50)
        print("1. 在目标设备上启动 RPC 服务器：")
        print("   python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090")
        print("\n2. 从本地机器连接：")
        print("   remote = rpc.connect('DEVICE_IP', 9090)")
        print("\n3. 上传编译后的库：")
        print("   remote.upload('model_deployed.so')")
        print("   remote.upload('model_params.npz')")
        print("\n4. 加载并远程运行：")
        print("   lib = remote.load_module('model_deployed.so')")
        print("   vm = relax.VirtualMachine(lib, remote.cpu())")
        print("   result = vm['main'](input, *params)")
        print("\n要实际运行 RPC，请设置 local_demo=False")
        return

    # 实际 RPC 部署工作流
    device_host = "192.168.1.100"  # 替换为你的设备 IP
    device_port = 9090
    remote = rpc.connect(device_host, device_port)
    print(f"已连接到远程设备 {device_host}:{device_port}")

    # 上传库和参数到远程设备
    remote.upload(lib_path)
    remote.upload(params_path)
    print("已上传文件到远程设备")

    # 在远程设备上加载库
    lib = remote.load_module("model_deployed.so")

    # 选择远程机器上的设备
    # CPU: dev = remote.cpu()
    # CUDA GPU: dev = remote.cuda(0)
    # OpenCL: dev = remote.cl(0)
    dev = remote.cpu()

    # 创建 VM 并加载参数
    vm = relax.VirtualMachine(lib, dev)

    params_npz = np.load(params_path)
    remote_params = [
        tvm.runtime.tensor(params_npz[f"p_{i}"], dev)
        for i in range(len(params_npz))
    ]

    ######################################################################
    # 第 5 步：在远程设备上运行推理
    # 注意：通过 RPC 运行 VM 时，我们使用 set_input() + invoke_stateful()
    # 而非直接函数调用 vm["main"](...)。这是因为 RPC 以 DLTensor* 传输张量，
    # 而 VM 内建函数期望 ffi.Tensor。set_input API 在内部处理此转换。

    input_data = np.random.randn(1, 1, 28, 28).astype("float32")
    remote_input = tvm.runtime.tensor(input_data, dev)

    vm.set_input("main", remote_input, *remote_params)
    vm.invoke_stateful("main")
    output = vm.get_outputs("main")

    if isinstance(output, tvm.ir.Array) and len(output) > 0:
        result = output[0]
    else:
        result = output

    result_np = result.numpy()
    print("远程设备推理完成")
    print(f"  输出形状：{result_np.shape}")
    print(f"  预测类别：{np.argmax(result_np)}")

    ######################################################################
    # 第 6 步：性能评估（可选）
    time_f = vm.time_evaluator("invoke_stateful", dev, number=10, repeat=3)
    prof_res = time_f("main")
    print(f"远程设备推理时间：{prof_res.mean * 1000:.2f} ms")
```

:::note 性能优化建议

为了在目标设备上获得最佳性能，可以考虑：

1. **使用 MetaSchedule 自动调优**：使用自动搜索为特定硬件找到最优调度：

```python
mod = relax.get_pipeline(
    "static_shape_tuning",
    target=target,
    total_trials=2000
)(mod)
```

2. **使用 DLight 快速优化**：应用预定义的高性能调度：

```python
from tvm.s_tir import dlight as dl
with target:
    mod = dl.ApplyDefaultSchedule()(mod)
```

3. **架构特定优化**：
   - ARM NEON SIMD：`-mattr=+neon`
   - x86 AVX-512：`-mcpu=skylake-avx512`
   - RISC-V Vector：`-mattr=+v`

详见[端到端优化模型](/docs/how-to/end-to-end-optimize-model)教程。
:::


## 总结

本教程完整展示了 TVM 中交叉编译和 RPC 功能的使用流程。

我们演示了两种方法：

**低层 TensorIR（TE）方法** — 用于理解基础原理：
- 使用张量表达式定义计算
- 为 ARM 目标交叉编译
- 通过 RPC 部署和运行

**高层 Relax 方法** — 用于部署完整模型：
- 从 PyTorch（或 ONNX）导入模型
- 转换为 Relax 表示
- 为 ARM Linux 设备交叉编译
- 通过 RPC 部署到远程设备
- 运行推理并评估性能

关键要点：
- 在远程设备上设置 RPC 服务器
- 在强大的本地机器上为资源受限的目标交叉编译
- 通过 RPC API 上传并远程执行编译后的模块
- 测量性能时排除网络开销

相关教程：
- [导出和加载 Relax 可执行文件](/docs/how-to/export-and-load-executable) — 导出和加载编译后的模型
- [端到端优化模型](/docs/how-to/end-to-end-optimize-model) — 使用自动调优进行端到端优化

* [下载 Jupyter notebook：cross_compilation_and_rpc.ipynb](https://tvm.apache.org/docs/_downloads/148819f3421b8d89b1723c3e15e3f19f/cross_compilation_and_rpc.ipynb)
* [下载 Python 源码：cross_compilation_and_rpc.py](https://tvm.apache.org/docs/_downloads/3cbcc56110528f886a987b8b251e7c88/cross_compilation_and_rpc.py)
* [下载压缩包：cross_compilation_and_rpc.zip](https://tvm.apache.org/docs/_downloads/f69380821f417ef2210f45503d81bded/cross_compilation_and_rpc.zip)
