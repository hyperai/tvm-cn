---
title: 编译 PyTorch 目标检测模型
---

# 编译 PyTorch 目标检测模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/deploy_models/deploy_object_detection_pytorch.html#sphx-glr-download-how-to-deploy-models-deploy-object-detection-pytorch-py) 下载完整的示例代码
:::

本文介绍如何用 Relay VM 部署 PyTorch 目标检测模型。

首先应安装 PyTorch。此外，还应安装 TorchVision，并将其作为模型合集（model zoo）。

可通过 pip 快速安装：

``` bash
pip install torch
pip install torchvision
```

或参考官网：https://pytorch.org/get-started/locally/

PyTorch 版本应该和 TorchVision 版本兼容。

目前 TVM 支持 PyTorch 1.7 和 1.4，其他版本可能不稳定。

``` python
import tvm
from tvm import relay
from tvm import relay
from tvm.runtime.vm import VirtualMachine
from tvm.contrib.download import download_testdata

import numpy as np
import cv2

# PyTorch 导入
import torch
import torchvision
```

## 从 TorchVision 加载预训练的 MaskRCNN 并进行跟踪

``` python
in_size = 300
input_shape = (1, 3, in_size, in_size)

def do_trace(model, inp):
    model_trace = torch.jit.trace(model, inp)
    model_trace.eval()
    return model_trace

def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]

class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])

model_func = torchvision.models.detection.maskrcnn_resnet50_fpn
model = TraceWrapper(model_func(pretrained=True))

model.eval()
inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

with torch.no_grad():
    out = model(inp)
    script_module = do_trace(model, inp)
```

输出结果：

``` bash
Downloading: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth" to /workspace/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth

  0%|          | 0.00/170M [00:00<?, ?B/s]
  9%|9         | 15.3M/170M [00:00<00:01, 160MB/s]
 19%|#8        | 32.1M/170M [00:00<00:00, 170MB/s]
 29%|##9       | 49.7M/170M [00:00<00:00, 176MB/s]
 40%|####      | 68.8M/170M [00:00<00:00, 185MB/s]
 51%|#####     | 86.4M/170M [00:00<00:00, 175MB/s]
 61%|######1   | 104M/170M [00:00<00:00, 178MB/s]
 71%|#######1  | 121M/170M [00:00<00:00, 169MB/s]
 86%|########6 | 147M/170M [00:00<00:00, 199MB/s]
100%|##########| 170M/170M [00:00<00:00, 193MB/s]
/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:3878: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  for i in range(dim)
/usr/local/lib/python3.7/dist-packages/torchvision/models/detection/anchor_utils.py:127: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  for g in grid_sizes
/usr/local/lib/python3.7/dist-packages/torchvision/models/detection/anchor_utils.py:127: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  for g in grid_sizes
/usr/local/lib/python3.7/dist-packages/torchvision/models/detection/rpn.py:73: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  A = Ax4 // 4
/usr/local/lib/python3.7/dist-packages/torchvision/models/detection/rpn.py:74: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  C = AxC // A
/usr/local/lib/python3.7/dist-packages/torchvision/ops/boxes.py:156: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
/usr/local/lib/python3.7/dist-packages/torchvision/ops/boxes.py:158: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
/usr/local/lib/python3.7/dist-packages/torchvision/models/detection/transform.py:293: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  for s, s_orig in zip(new_size, original_size)
/usr/local/lib/python3.7/dist-packages/torchvision/models/detection/roi_heads.py:387: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)
```

## 下载测试图像并进行预处理

``` python
img_url = (
    "/img/docs/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
)
img_path = download_testdata(img_url, "test_street_small.jpg", module="data")

img = cv2.imread(img_path).astype("float32")
img = cv2.resize(img, (in_size, in_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.transpose(img / 255.0, [2, 0, 1])
img = np.expand_dims(img, axis=0)
```

## 将计算图导入 Relay

``` python
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, shape_list)
```

输出结果：

``` bash
/workspace/python/tvm/relay/build_module.py:411: DeprecationWarning: Please use input parameter mod (tvm.IRModule) instead of deprecated parameter mod (tvm.relay.function.Function)
  DeprecationWarning,
```

## 使用 Relay VM 编译

注意：目前仅支持 CPU target。对于 x86 target，因为 TorchVision RCNN 模型中存在大型密集算子，为取得最佳性能，强烈推荐使用 Intel MKL 和 Intel OpenMP 来构建 TVM。

``` python
# 在 x86 target上添加“-libs=mkl”以获得最佳性能。
# 对于支持 AVX512 的 x86 机器，完整 target 是
# "llvm -mcpu=skylake-avx512 -libs=mkl"
target = "llvm"

with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
    vm_exec = relay.vm.compile(mod, target=target, params=params)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 使用 Relay VM 进行推理

``` python
dev = tvm.cpu()
vm = VirtualMachine(vm_exec, dev)
vm.set_input("main", **{input_name: img})
tvm_res = vm.run()
```

## 获取 score 大于 0.9 的 box

``` python
score_threshold = 0.9
boxes = tvm_res[0].numpy().tolist()
valid_boxes = []
for i, score in enumerate(tvm_res[1].numpy().tolist()):
    if score > score_threshold:
        valid_boxes.append(boxes[i])
    else:
        break

print("Get {} valid boxes".format(len(valid_boxes)))
```

输出结果：

``` bash
Get 9 valid boxes
```

**脚本总运行时长：**（2 分 57.278 秒）

[下载 Python 源代码：deploy_object_detection_pytorch.py](https://tvm.apache.org/docs/_downloads/7795da4b258c8feff986668b95ef57ad/deploy_object_detection_pytorch.py)

[下载 Jupyter Notebook：deploy_object_detection_pytorch.ipynb](https://tvm.apache.org/docs/_downloads/399e1d7889ca66b69d51655784827503/deploy_object_detection_pytorch.ipynb)
