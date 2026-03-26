---

title: 导出和加载 Relax 可执行文件

---

本教程演示如何将已编译的 Relax 模块导出为共享库文件，再将其加载回 TVM 运行时并执行推理。本教程展示了如何使用 `tvm.relax` API 将 Relax（或导入的 PyTorch / ONNX）程序转换为可部署的产物。

:::note

本教程以 PyTorch 作为源格式，但导出/加载的工作流程对 ONNX 模型同样适用。对于 ONNX，使用 `from_onnx(model, keep_params_in_input=True)` 替代 `from_exported_program()`，后续的构建、导出和加载步骤完全相同。

:::

## 简介

TVM 将 Relax 程序构建为 `tvm.runtime.Executable` 对象，其中包含 VM 字节码、编译后的内核和常量。通过 `export_library` 方法导出可执行文件，你将获得一个共享库文件（例如 Linux 上的 `.so`），它可以被传输到其他机器、通过 RPC 上传，或稍后使用 TVM 运行时重新加载。本教程展示端到端的完整步骤，并解释在此过程中产生的文件。

```python
import os
from pathlib import Path

try:
    import torch
    from torch.export import export
except ImportError:
    torch = None
```

## 准备 Torch MLP 并转换为 Relax

我们从一个小型 PyTorch MLP 开始，使示例保持轻量。模型先导出为 `torch.export.ExportedProgram`，然后转换为 Relax `IRModule`。

```python
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

IS_IN_CI = os.getenv("CI", "").lower() == "true"
HAS_TORCH = torch is not None
RUN_EXAMPLE = HAS_TORCH and not IS_IN_CI


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


torch_model = TorchMLP().eval()
example_args = (torch.randn(1, 1, 28, 28, dtype=torch.float32),)

with torch.no_grad():
    exported_program = export(torch_model, example_args)

mod = from_exported_program(exported_program, keep_params_as_input=True)

# 分离模型参数，以便稍后绑定（或存储到磁盘）
mod, params = relax.frontend.detach_params(mod)

print("导入的 Relax 模块：")
mod.show()
```

## 使用 `export_library` 构建和导出

我们为 `llvm` 目标构建以生成 CPU 代码，然后导出生成的可执行文件。传入 `workspace_dir` 可以保留中间打包文件，方便检查产物内容。

```python
TARGET = tvm.target.Target("llvm")
ARTIFACT_DIR = Path("relax_export_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# 在构建前应用默认的 Relax 编译管线
pipeline = relax.get_pipeline()
with TARGET:
    built_mod = pipeline(mod)

# 构建时不包含参数——在运行时传入
executable = tvm.compile(built_mod, target=TARGET)

library_path = ARTIFACT_DIR / "mlp_cpu.so"
executable.export_library(str(library_path), workspace_dir=str(ARTIFACT_DIR))

print(f"已导出运行时库到：{library_path}")

# workspace 目录现在包含共享对象和支持文件
produced_files = sorted(p.name for p in ARTIFACT_DIR.iterdir())
print("保存的产物：")
for name in produced_files:
    print(f"  - {name}")
```

生成的文件说明：

- **`mlp_cpu.so`**：主要的可部署共享库，包含 VM 字节码、编译后的内核和常量。由于参数在运行时传入，你还需要单独保存一个参数文件（见下一节）。
- 中间目标文件（`devc.o`、`lib0.o` 等）保留在 workspace 中供检查，部署时不需要。

## 加载导出的库并运行

共享对象生成后，我们可以在任何具有兼容指令集的机器上将其重新加载到 TVM 运行时中。Relax VM 直接使用运行时模块。

```python
loaded_rt_mod = tvm.runtime.load_module(str(library_path))
dev = tvm.cpu(0)
vm = relax.VirtualMachine(loaded_rt_mod, dev)

# 准备输入数据
input_tensor = torch.randn(1, 1, 28, 28, dtype=torch.float32)
vm_input = tvm.runtime.tensor(input_tensor.numpy(), dev)

# 准备参数（分配到目标设备上）
vm_params = [tvm.runtime.tensor(p, dev) for p in params["main"]]

# 运行推理：先传入输入数据，再传入所有参数
tvm_output = vm["main"](vm_input, *vm_params)

# TVM 对元组输出返回 Array 对象，通过索引访问。
# 从 PyTorch 导入的模型输出通常是元组（即使只有单个输出）。
# ONNX 模型的输出可能直接是单个 Tensor。
if isinstance(tvm_output, tvm.ir.Array) and len(tvm_output) > 0:
    result_tensor = tvm_output[0]
else:
    result_tensor = tvm_output

print("VM 输出形状：", result_tensor.shape)
print("VM 输出类型：", type(tvm_output), "->", type(result_tensor))

# 重新加载后仍可检查可执行文件信息
print("可执行文件统计信息：\n", loaded_rt_mod["stats"]())
```

## 保存参数用于部署

由于参数在运行时传入（未嵌入到 `.so` 中），我们必须单独保存参数用于部署。这是在其他机器或独立脚本中使用模型的必要步骤。

```python
import numpy as np

# 将参数保存到磁盘
params_path = ARTIFACT_DIR / "model_params.npz"
param_arrays = {f"p_{i}": p.numpy() for i, p in enumerate(params["main"])}
np.savez(str(params_path), **param_arrays)
print(f"已保存参数到：{params_path}")
```

:::note

你也可以将参数直接嵌入到 `.so` 中以创建单文件部署。在从 PyTorch 导入时使用 `keep_params_as_input=False`：

```python
mod = from_exported_program(exported_program, keep_params_as_input=False)
# 参数现在作为常量嵌入到模块中
executable = tvm.compile(built_mod, target=TARGET)
# 运行时：vm["main"](input)  # 无需传入参数！
```

这会创建单文件部署（只需要 `.so` 文件），但你将无法在不重新编译的情况下更换参数。对于大多数生产工作流，分离代码和参数（如上所示）更具灵活性。

:::

## 加载和运行导出的模型

要在另一台机器或独立脚本中使用导出的模型，需要同时加载 `.so` 库和参数文件。以下是重新加载和运行模型的完整示例，可保存为 `run_mlp.py`：

```bash
chmod +x run_mlp.py
./run_mlp.py  # 像普通程序一样运行
```

完整脚本：

```python
#!/usr/bin/env python3
import numpy as np
import tvm
from tvm import relax

# 第 1 步：加载编译后的库
lib = tvm.runtime.load_module("relax_export_artifacts/mlp_cpu.so")

# 第 2 步：创建虚拟机
device = tvm.cpu(0)
vm = relax.VirtualMachine(lib, device)

# 第 3 步：从 .npz 文件加载参数
params_npz = np.load("relax_export_artifacts/model_params.npz")
params = [tvm.runtime.tensor(params_npz[f"p_{i}"], device)
          for i in range(len(params_npz))]

# 第 4 步：准备输入数据
data = np.random.randn(1, 1, 28, 28).astype("float32")
input_tensor = tvm.runtime.tensor(data, device)

# 第 5 步：运行推理（先传入输入数据，再传入所有参数）
output = vm["main"](input_tensor, *params)

# 第 6 步：提取结果（输出可能是元组或单个 Tensor）
# PyTorch 模型通常返回元组，ONNX 模型可能返回单个 Tensor
if isinstance(output, tvm.ir.Array) and len(output) > 0:
    result = output[0]
else:
    result = output

print("预测形状：", result.shape)
print("预测类别：", np.argmax(result.numpy()))
```

### 在 GPU 上运行

要在 GPU 而非 CPU 上运行，需要做以下更改：

**1. 编译时指定 GPU 目标：**

```python
TARGET = tvm.target.Target("cuda")  # 从 "llvm" 改为 "cuda"
```

**2. 在脚本中使用 GPU 设备：**

```python
device = tvm.cuda(0)  # 使用 CUDA 设备而非 CPU
vm = relax.VirtualMachine(lib, device)

# 将参数加载到 GPU
params = [tvm.runtime.tensor(params_npz[f"p_{i}"], device)
          for i in range(len(params_npz))]

# 在 GPU 上准备输入
input_tensor = tvm.runtime.tensor(data, device)
```

其余脚本保持不变。所有张量（参数和输入）必须分配在与编译模型相同的设备（GPU）上。

### 部署清单

将模型迁移到另一台主机（通过 RPC 或 SCP）时，必须复制**两个**文件：

1. `mlp_cpu.so`（或 GPU 版本的 `mlp_cuda.so`）— 编译后的模型代码
2. `model_params.npz` — 模型参数（序列化为 NumPy 数组）

远程机器需要将两个文件放在同一目录下。上述脚本假设它们位于脚本所在位置的 `relax_export_artifacts/` 子目录中，请根据你的部署需要调整路径。对于 GPU 部署，确保目标机器具有兼容的 CUDA 驱动程序，且模型是为相同的 GPU 架构编译的。

## 部署到远程设备

要将导出的模型部署到远程 ARM Linux 设备（例如 Raspberry Pi），可以使用 TVM 的 RPC 机制进行交叉编译、上传和远程运行模型。此工作流适用于以下场景：

- 目标设备编译资源有限
- 希望在实际硬件上调优性能
- 需要部署到嵌入式设备

详细指南请参阅[交叉编译与 RPC](/docs/how-to/cross-compilation-and-rpc)。

ARM 部署工作流快速示例：

```python
import tvm.rpc as rpc
from tvm import relax

# 第 1 步：为 ARM 目标交叉编译（在本地机器上）
TARGET = tvm.target.Target({"kind": "llvm", "mtriple": "aarch64-linux-gnu"})
executable = tvm.compile(built_mod, target=TARGET)
executable.export_library("mlp_arm.so")

# 第 2 步：连接到远程设备的 RPC 服务器
remote = rpc.connect("192.168.1.100", 9090)  # 设备 IP 和 RPC 端口

# 第 3 步：上传编译后的库和参数
remote.upload("mlp_arm.so")
remote.upload("model_params.npz")

# 第 4 步：在远程设备上加载和运行
lib = remote.load_module("mlp_arm.so")
vm = relax.VirtualMachine(lib, remote.cpu())
# ... 准备输入和参数，然后运行推理
```

关键区别在于编译时使用 ARM 目标三元组，并通过 RPC 上传文件而非直接复制。

## 常见问题

**可以将 `.so` 作为独立可执行文件运行吗（如 `./mlp_cpu.so`）？**

不可以。`.so` 文件是共享库，不是独立的可执行二进制文件。你无法直接从终端运行它，必须通过 TVM 运行时程序加载（如上面"加载和运行"一节所示）。`.so` 捆绑了 VM 字节码和编译后的内核，但仍需要 TVM 运行时来执行。

**哪些设备可以运行导出的库？**

目标必须与你编译时使用的指令集架构匹配（本示例为 `llvm`）。只要目标三元组、运行时 ABI 和可用设备一致，你就可以在不同机器之间迁移产物。对于异构构建（CPU 加 GPU），还需要一并部署额外的设备库。

**`.params` 和 `metadata.json` 文件是什么？**

这些辅助文件仅在特定配置下生成。在本教程中，由于我们在运行时传入参数，它们不会生成。当它们出现时，可以与 `.so` 一起保留用于检查，但关键内容通常已嵌入到共享对象中，因此仅部署 `.so` 通常就足够了。
