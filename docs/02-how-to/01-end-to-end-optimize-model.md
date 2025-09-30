---

title: 端到端优化模型

---


:::note

本教程可通过 Google Colab 交互式运行！也可点击[此处](https://tvm.apache.org/docs/get_started/tutorials/quick_start.html#sphx-glr-download-get-started-tutorials-quick-start-py)在本地运行 Jupyter Notebook。

[在 Google Colab 中打开](https://colab.research.google.com/github/apache/tvm-site/blob/asf-site/docs/_downloads/317a8cc53139718b9a36a16ba052e44b/e2e_opt_model.ipynb)

:::


本教程演示了如何使用 Apache TVM 对机器学习模型进行优化。我们将使用来自 PyTorch 的预训练 ResNet-18 模型，并通过 TVM 的 Relax API 对其进行端到端优化。请注意，默认的端到端优化可能不适用于复杂模型。

## 准备工作

首先，我们准备模型和输入信息。我们使用来自 PyTorch 的预训练 ResNet-18 模型。

```plain
import os
import numpy as np
import torch
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18

torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
```


输出：

```plain
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /workspace/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

  0%|          | 0.00/44.7M [00:00<?, ?B/s]
 12%|█▏        | 5.50M/44.7M [00:00<00:00, 56.3MB/s]
 25%|██▌       | 11.2M/44.7M [00:00<00:00, 57.8MB/s]
 38%|███▊      | 17.0M/44.7M [00:00<00:00, 58.4MB/s]
 51%|█████     | 22.6M/44.7M [00:00<00:00, 54.4MB/s]
 63%|██████▎   | 28.0M/44.7M [00:00<00:00, 53.3MB/s]
 74%|███████▍  | 33.1M/44.7M [00:00<00:00, 50.8MB/s]
 85%|████████▌ | 38.0M/44.7M [00:00<00:00, 49.1MB/s]
 98%|█████████▊| 43.8M/44.7M [00:00<00:00, 52.4MB/s]
100%|██████████| 44.7M/44.7M [00:00<00:00, 53.5MB/s]
```


## 回顾整体流程

![图片](/img/docs/v21/02-how-to_01-end-to-end-optimize-model_1.png)

整体流程包括以下步骤：
* **构建或导入模型：** 构建一个神经网络模型，或从其他框架（例如 PyTorch、ONNX）中导入一个预训练模型，并创建 TVM 的 IRModule，其中包含了编译所需的所有信息，包括用于计算图的高级 Relax 函数，以及用于张量程序的低级 TensorIR 函数。 
* **执行可组合优化：** 执行一系列优化转换，例如图优化、张量程序优化和库调度。 
* **构建与通用部署：** 将优化后的模型构建为可部署模块，部署到通用运行时，并在不同设备（如 CPU、GPU 或其他加速器）上执行。

### 将模型转换为 IRModule

接下来，我们使用 PyTorch 的 Relax 前端将模型转换为 IRModule，以便进行进一步优化。

```plain
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

Apache TVM Unity 提供了一种灵活的方式来优化 IRModule。围绕 IRModule 的优化可以与现有的 pipeline 灵活组合。注意，每一个转换都可以通过 `tvm.ir.transform.Sequential` 组合成一个优化 pipeline。


本教程聚焦于通过自动调优对模型进行端到端优化。我们利用 MetaSchedule 对模型进行调优，并将调优日志保存到数据库中。随后，我们将数据库中的结果应用于模型，以获得最佳性能。


```plain
TOTAL_TRIALS = 8000  # 如需更高性能可改为 20000
target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")  # 替换为你的目标设备
work_dir = "tuning_logs"

if not IS_IN_CI:
    mod = relax.get_pipeline("static_shape_tuning", target=target, total_trials=TOTAL_TRIALS)(mod)

    # 仅展示主函数
    mod["main"].show()
```

## 构建与部署

最后，我们构建优化后的模型，并将其部署到目标设备。在 CI 环境中会跳过此步骤。


```plain
if not IS_IN_CI:
    ex = tvm.compile(mod, target="cuda")
    dev = tvm.device("cuda", 0)
    vm = relax.VirtualMachine(ex, dev)
    # 需要在 GPU 上分配数据和参数
    gpu_data = tvm.runtime.tensor(np.random.rand(1, 3, 224, 224).astype("float32"), dev)
    gpu_params = [tvm.runtime.tensor(p, dev) for p in params["main"]]
    gpu_out = vm["main"](gpu_data, *gpu_params).numpy()

    print(gpu_out.shape)
```
* [下载 Jupyter notebook：e2e_opt_model.ipynb](https://tvm.apache.org/docs/_downloads/317a8cc53139718b9a36a16ba052e44b/e2e_opt_model.ipynb)
* [下载 Python 源码：e2e_opt_model.py](https://tvm.apache.org/docs/_downloads/a4f940a6740cf66055ca729bf25bfbaa/e2e_opt_model.py)
* [下载压缩包：e2e_opt_model.zip](https://tvm.apache.org/docs/_downloads/a7dd7652b2ad50f82d7b739ce3645799/e2e_opt_model.zip)



