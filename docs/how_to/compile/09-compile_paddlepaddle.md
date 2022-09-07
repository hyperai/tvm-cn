---
title: 编译 PaddlePaddle 模型
---

# 编译 PaddlePaddle 模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/compile_models/from_paddle.html#sphx-glr-download-how-to-compile-models-from-paddle-py) 下载完整的示例代码
:::

**作者**：[Ziyuan Ma](https://github.com/ZiyuanMa/)

本文介绍如何用 Relay 部署 PaddlePaddle 模型，首先安装 PaddlePaddle（版本>=2.1.3），可通过 pip 快速安装：

``` bash
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

或参考官方网站：https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html

``` python
import tarfile
import paddle
import numpy as np
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
```

输出结果：

``` bash
/usr/local/lib/python3.7/dist-packages/paddle/vision/transforms/functional_pil.py:36: DeprecationWarning: NEAREST is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.NEAREST or Dither.NONE instead.
  'nearest': Image.NEAREST,
/usr/local/lib/python3.7/dist-packages/paddle/vision/transforms/functional_pil.py:37: DeprecationWarning: BILINEAR is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BILINEAR instead.
  'bilinear': Image.BILINEAR,
/usr/local/lib/python3.7/dist-packages/paddle/vision/transforms/functional_pil.py:38: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  'bicubic': Image.BICUBIC,
/usr/local/lib/python3.7/dist-packages/paddle/vision/transforms/functional_pil.py:39: DeprecationWarning: BOX is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BOX instead.
  'box': Image.BOX,
/usr/local/lib/python3.7/dist-packages/paddle/vision/transforms/functional_pil.py:40: DeprecationWarning: LANCZOS is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.LANCZOS instead.
  'lanczos': Image.LANCZOS,
/usr/local/lib/python3.7/dist-packages/paddle/vision/transforms/functional_pil.py:41: DeprecationWarning: HAMMING is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.HAMMING instead.
  'hamming': Image.HAMMING
```

## 加载预训练的 ResNet50 模型

加载 PaddlePaddle 提供的 ResNet50 预训练模型：

``` python
url = "https://bj.bcebos.com/x2paddle/models/paddle_resnet50.tar"
model_path = download_testdata(url, "paddle_resnet50.tar", module="model")

with tarfile.open(model_path) as tar:
    names = tar.getnames()
    for name in names:
        tar.extract(name, "./")

model = paddle.jit.load("./paddle_resnet50/model")
```

输出结果：

``` bash
/usr/local/lib/python3.7/dist-packages/paddle/fluid/backward.py:1666: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3,and in 3.9 it will stop working
  return list(x) if isinstance(x, collections.Sequence) else [x]
```

## 加载测试图像

还是用猫的图像：

``` python
from PIL import Image
import paddle.vision.transforms as T

transforms = T.Compose(
    [
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

img = transforms(img)
img = np.expand_dims(img, axis=0)
```

## 使用 Relay 编译模型

``` python
target = "llvm"
shape_dict = {"inputs": img.shape}
mod, params = relay.frontend.from_paddle(model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    executor = relay.build_module.create_executor(
        "graph", mod, tvm.cpu(0), target, params
    ).evaluate()
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 在 TVM 上执行

``` python
dtype = "float32"
tvm_output = executor(tvm.nd.array(img.astype(dtype))).numpy()
```

## 查找同义词集名称

在 1000 个类的同义词集中，查找分数最高的第一个：

``` python
synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = f.readlines()

top1 = np.argmax(tvm_output[0])
print(f"TVM prediction top-1 id: {top1}, class name: {synset[top1]}")
```

输出结果：

``` bash
TVM prediction top-1 id: 282, class name:  282: 'tiger cat',
```

[下载 Python 源代码：from_paddle.py](https://tvm.apache.org/docs/_downloads/16269b77359771348d507395692524cf/from_paddle.py)

[下载 Jupyter Notebook：from_paddle.ipynb](https://tvm.apache.org/docs/_downloads/a608d8b69371e9bc149dd89f6db2c38e/from_paddle.ipynb)