---
title: 编译 PyTorch 模型
---

# 编译 PyTorch 模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/compile_models/from_pytorch.html#sphx-glr-download-how-to-compile-models-from-pytorch-py) 下载完整的示例代码
:::

**作者**：[Alex Wong](https://github.com/alexwong/)

本文介绍了如何用 Relay 部署 PyTorch 模型。

首先应安装 PyTorch。此外，还应安装 TorchVision，并将其作为模型合集 (model zoo)。

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

import numpy as np

from tvm.contrib.download import download_testdata

# 导入 PyTorch
import torch
import torchvision
```

## 加载预训练的 PyTorch 模型

``` python
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# 通过追踪获取 TorchScripted 模型
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
```

输出结果：

``` bash
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /workspace/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

  0%|          | 0.00/44.7M [00:00<?, ?B/s]
 11%|#         | 4.87M/44.7M [00:00<00:00, 51.0MB/s]
 22%|##1       | 9.73M/44.7M [00:00<00:00, 49.2MB/s]
 74%|#######3  | 32.9M/44.7M [00:00<00:00, 136MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 129MB/s]
```

## 加载测试图像

经典的猫咪示例：

``` python
from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# 预处理图像，并将其转换为张量
from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)
```

## 将计算图导入 Relay

将 PyTorch 计算图转换为 Relay 计算图。input_name 可以是任意值。

``` python
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
```

## Relay 构建

用给定的输入规范，将计算图编译为 llvm target。

``` python
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 在 TVM 上执行可移植计算图

将编译好的模型部署到 target 上：

``` python
from tvm.contrib import graph_executor

dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# 设置输入
m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
# 执行
m.run()
# 得到输出
tvm_output = m.get_output(0)
```

## 查找分类集名称

在 1000 个类的分类集中，查找分数最高的第一个：

``` python
synset_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_synsets.txt",
    ]
)
synset_name = "imagenet_synsets.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

class_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_classes.txt",
    ]
)
class_name = "imagenet_classes.txt"
class_path = download_testdata(class_url, class_name, module="data")
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# 获得 TVM 的前 1 个结果
top1_tvm = np.argmax(tvm_output.numpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

# 将输入转换为 PyTorch 变量，并获取 PyTorch 结果进行比较
with torch.no_grad():
    torch_img = torch.from_numpy(img)
    output = model(torch_img)

    # 获得 PyTorch 的前 1 个结果
    top1_torch = np.argmax(output.numpy())
    torch_class_key = class_id_to_key[top1_torch]

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
```

输出结果：

``` bash
Relay top-1 id: 281, class name: tabby, tabby cat
Torch top-1 id: 281, class name: tabby, tabby cat
```

[下载 Python 源代码：from_pytorch.py](https://tvm.apache.org/docs/_downloads/f90d5f6bfd99e0d9812ae5b91503e148/from_pytorch.py)

[下载 Jupyter Notebook：from_pytorch.ipynb](https://tvm.apache.org/docs/_downloads/1f4943aed1aa607b2775c18b1d71db10/from_pytorch.ipynb)
