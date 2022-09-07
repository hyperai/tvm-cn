---
title: 编译 OneFlow 模型
---

# 编译 OneFlow 模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/compile_models/from_oneflow.html#sphx-glr-download-how-to-compile-models-from-oneflow-py) 下载完整的示例代码
:::

**作者**：[Xiaoyu Zhang](https://github.com/BBuf/)

本文介绍如何用 Relay 部署 OneFlow 模型。

首先安装 OneFlow 包，可通过 pip 快速安装：

``` bash
pip install flowvision==0.1.0
python3 -m pip install -f https://release.oneflow.info oneflow==0.7.0+cpu
```

或参考官网：https://github.com/Oneflow-Inc/oneflow

目前 TVM 支持 OneFlow 0.7.0，其他版本可能不稳定。

``` python
import os, math
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# OneFlow 导入
import flowvision
import oneflow as flow
import oneflow.nn as nn

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
```

输出结果：

``` bash
/usr/local/lib/python3.7/dist-packages/flowvision/transforms/functional_pil.py:193: DeprecationWarning: BILINEAR is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BILINEAR instead.
  def resize(img, size, interpolation=Image.BILINEAR):
/usr/local/lib/python3.7/dist-packages/flowvision/transforms/functional.py:65: DeprecationWarning: NEAREST is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.NEAREST or Dither.NONE instead.
  Image.NEAREST: "nearest",
/usr/local/lib/python3.7/dist-packages/flowvision/transforms/functional.py:66: DeprecationWarning: BILINEAR is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BILINEAR instead.
  Image.BILINEAR: "bilinear",
/usr/local/lib/python3.7/dist-packages/flowvision/transforms/functional.py:67: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  Image.BICUBIC: "bicubic",
/usr/local/lib/python3.7/dist-packages/flowvision/transforms/functional.py:68: DeprecationWarning: BOX is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BOX instead.
  Image.BOX: "box",
/usr/local/lib/python3.7/dist-packages/flowvision/transforms/functional.py:69: DeprecationWarning: HAMMING is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.HAMMING instead.
  Image.HAMMING: "hamming",
/usr/local/lib/python3.7/dist-packages/flowvision/transforms/functional.py:70: DeprecationWarning: LANCZOS is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.LANCZOS instead.
  Image.LANCZOS: "lanczos",
/usr/local/lib/python3.7/dist-packages/flowvision/data/auto_augment.py:28: DeprecationWarning: BILINEAR is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BILINEAR instead.
  _RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
/usr/local/lib/python3.7/dist-packages/flowvision/data/auto_augment.py:28: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  _RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
```

## 加载和保存 OneFlow 的预训练模型

``` python
model_name = "resnet18"
model = getattr(flowvision.models, model_name)(pretrained=True)
model = model.eval()

model_dir = "resnet18_model"
if not os.path.exists(model_dir):
    flow.save(model.state_dict(), model_dir)
```

输出结果：

``` bash
Downloading: "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet18.zip" to /workspace/.oneflow/flowvision_cache/resnet18.zip

  0%|          | 0.00/41.5M [00:00<?, ?B/s]
 19%|#9        | 7.99M/41.5M [00:00<00:00, 41.9MB/s]
 39%|###8      | 16.0M/41.5M [00:00<00:00, 40.1MB/s]
 54%|#####3    | 22.3M/41.5M [00:00<00:00, 45.4MB/s]
 65%|######4   | 26.9M/41.5M [00:00<00:00, 42.8MB/s]
 82%|########2 | 34.1M/41.5M [00:00<00:00, 51.3MB/s]
 95%|#########4| 39.3M/41.5M [00:00<00:00, 47.7MB/s]
100%|##########| 41.5M/41.5M [00:00<00:00, 46.0MB/s]
```

## 加载测试图像

还是用猫的图像：

``` python
from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# 预处理图像，并转换为张量
from flowvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img.numpy(), 0)
```

## 将计算图导入到 Relay 中

将 OneFlow 计算图转换为 Relay 计算图，输入任意名称。

``` python
class Graph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x):
        out = self.m(x)
        return out

graph = Graph(model)
_ = graph._compile(flow.randn(1, 3, 224, 224))

mod, params = relay.frontend.from_oneflow(graph, model_dir)
```

## 使用 Relay 构建

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

接下来在 target 上部署编译好的模型：

``` python
target = "cuda"
with tvm.transform.PassContext(opt_level=10):
    intrp = relay.build_module.create_executor("graph", mod, tvm.cuda(0), target)

print(type(img))
print(img.shape)
tvm_output = intrp.evaluate()(tvm.nd.array(img.astype("float32")), **params)
```

输出结果：

``` bash
<class 'numpy.ndarray'>
(1, 3, 224, 224)
```

## 查找同义词集名称

在 1000 个类的同义词集中，查找分数最高的第一个：

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

# 获得 TVM 分数最高的第一个结果
top1_tvm = np.argmax(tvm_output.numpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

# 将输入转换为 OneFlow 变量，并获取 OneFlow 结果进行比较
with flow.no_grad():
    torch_img = flow.from_numpy(img)
    output = model(torch_img)

    # 获取 OneFlow 分数最高的第一个结果
    top_oneflow = np.argmax(output.numpy())
    oneflow_class_key = class_id_to_key[top_oneflow]

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print(
    "OneFlow top-1 id: {}, class name: {}".format(top_oneflow, key_to_classname[oneflow_class_key])
)
```

输出结果：

``` bash
Relay top-1 id: 281, class name: tabby, tabby cat
OneFlow top-1 id: 281, class name: tabby, tabby cat
```

[下载 Python 源代码：from_oneflow.py](https://tvm.apache.org/docs/_downloads/f7ae979fbe61064749ce0fb7a621eb4c/from_oneflow.py)

[下载 Jupyter Notebook：from_oneflow.ipynb](https://tvm.apache.org/docs/_downloads/2e7b51cb39c472626dd3f046d9b89966/from_oneflow.ipynb)