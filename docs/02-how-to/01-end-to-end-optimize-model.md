---

title: 端到端优化模型

---


:::note

本教程可通过 Google Colab 交互式运行！也可点击[此处](/docs/how-to/end-to-end-optimize-model#构建与部署)在本地运行 Jupyter Notebook。

[在 Google Colab 中打开](https://colab.research.google.com/github/apache/tvm-site/blob/asf-site/docs/_downloads/317a8cc53139718b9a36a16ba052e44b/e2e_opt_model.ipynb)

:::


本教程演示了如何使用 Apache TVM 对机器学习模型进行优化。我们将使用来自 PyTorch 的预训练 ResNet-18 模型，并通过 TVM 的 Relax API 对其进行端到端优化。请注意，默认的端到端优化可能不适用于复杂模型。

## 准备工作

首先，我们准备模型和输入信息。我们使用来自 PyTorch 的预训练 ResNet-18 模型。

```python
import os
import numpy as np
import torch
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18

torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
```


## 回顾整体流程

![图片](/img/docs/v21/02-how-to_01-end-to-end-optimize-model_1.png)

整体流程包括以下步骤：
* **构建或导入模型：** 构建一个神经网络模型，或从其他框架（例如 PyTorch、ONNX）中导入一个预训练模型，并创建 TVM 的 IRModule，其中包含了编译所需的所有信息，包括用于计算图的高级 Relax 函数，以及用于张量程序的低级 TensorIR 函数。
* **执行可组合优化：** 执行一系列优化转换，例如图优化、张量程序优化和库调度。
* **构建与通用部署：** 将优化后的模型构建为可部署模块，部署到通用运行时，并在不同设备（如 CPU、GPU 或其他加速器）上执行。

### 将模型转换为 IRModule

接下来，我们使用 PyTorch 的 Relax 前端将模型转换为 IRModule，以便进行进一步优化。

```python
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# 为 torch.export 提供示例输入参数
example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

# 在 CI 环境中跳过运行
IS_IN_CI = os.getenv("CI", "") == "true"

if not IS_IN_CI:
    # 将模型转换为 IRModule
    with torch.no_grad():
        exported_program = export(torch_model, example_args)
        mod = from_exported_program(exported_program, keep_params_as_input=True)

    mod, params = relax.frontend.detach_params(mod)
    mod.show()
```


## IRModule 优化

Apache TVM 提供了一种灵活的方式来优化 IRModule。围绕 IRModule 的优化可以与现有的 pipeline 灵活组合。注意，每一个转换都可以通过 `tvm.ir.transform.Sequential` 组合成一个优化 pipeline。

本教程聚焦于通过自动调优对模型进行端到端优化。我们利用 MetaSchedule 对模型进行调优，并将调优日志保存到数据库中。随后，我们将数据库中的结果应用于模型，以获得最佳性能。

ResNet18 模型在编译过程中将被拆分为 20 个独立的调优任务。为确保每个任务在一次迭代中获得足够的调优资源并提供早期反馈：

- 为了快速观察调优进度，每个任务在每次迭代中最多分配 16 次试验（通过 `MAX_TRIALS_PER_TASK=16` 控制）。我们应将 `TOTAL_TRIALS` 设置为至少 `320（20 个任务 × 16 次试验）`，以确保每个任务至少获得一轮完整的迭代调优。我们在配置中将其设置为 512，以允许更多迭代，从而探索更广的参数空间并可能获得更好的性能。
- 如果 `MAX_TRIALS_PER_TASK == None`，系统默认为每个任务每次迭代分配 `TOTAL_TRIALS` 次试验。`TOTAL_TRIALS` 设置不足可能导致调优不充分，甚至完全跳过某些任务。显式设置两个参数可以避免此问题，并提供跨所有任务的确定性资源分配。

:::note
这些参数设置针对教程快速演示进行了优化。对于需要更高性能的生产部署，建议将 `MAX_TRIALS_PER_TASK` 和 `TOTAL_TRIALS` 调整为更大的值，以允许更广泛的搜索空间探索，通常能带来更好的性能结果。
:::

```python
TOTAL_TRIALS = 512  # 如需更高性能可改为 20000
MAX_TRIALS_PER_TASK = 16  # 增大每个任务的试验次数以获得更好性能
target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")  # 替换为你的目标设备
work_dir = "tuning_logs"

if not IS_IN_CI:
    mod = relax.get_pipeline(
        "static_shape_tuning",
        target=target,
        work_dir=work_dir,
        total_trials=TOTAL_TRIALS,
        max_trials_per_task=MAX_TRIALS_PER_TASK,
    )(mod)

    # 仅展示主函数
    mod["main"].show()
```

## 构建与部署

最后，我们构建优化后的模型，并将其部署到目标设备。在 CI 环境中会跳过此步骤。

```python
if not IS_IN_CI:
    with target:
        mod = tvm.s_tir.transform.DefaultGPUSchedule()(mod)
    ex = tvm.compile(mod, target=target)
    dev = tvm.device("cuda", 0)
    vm = relax.VirtualMachine(ex, dev)
    # 需要在 GPU 上分配数据和参数
    gpu_data = tvm.runtime.tensor(np.random.rand(1, 3, 224, 224).astype("float32"), dev)
    gpu_params = [tvm.runtime.tensor(p, dev) for p in params["main"]]
    gpu_out = vm["main"](gpu_data, *gpu_params)[0].numpy()

    print(gpu_out.shape)
```

* [下载 Jupyter notebook：e2e_opt_model.ipynb](https://tvm.apache.org/docs/_downloads/317a8cc53139718b9a36a16ba052e44b/e2e_opt_model.ipynb)
* [下载 Python 源码：e2e_opt_model.py](https://tvm.apache.org/docs/_downloads/a4f940a6740cf66055ca729bf25bfbaa/e2e_opt_model.py)
* [下载压缩包：e2e_opt_model.zip](https://tvm.apache.org/docs/_downloads/a7dd7652b2ad50f82d7b739ce3645799/e2e_opt_model.zip)
