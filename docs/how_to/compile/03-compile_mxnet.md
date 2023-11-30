---
title: 编译 MXNet 模型
---

# 编译 MXNet 模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/compile_models/from_mxnet.html#sphx-glr-download-how-to-compile-models-from-mxnet-py) 下载完整的示例代码
:::

**作者**：[Joshua Z. Zhang](https://zhreshold.github.io/)，[Kazutaka Morita](https://github.com/kazum)

本文将介绍如何用 Relay 部署 MXNet 模型。

首先安装 mxnet 模块：

``` bash
pip install mxnet
```

或参考官方安装指南：https://mxnet.apache.org/versions/master/install/index.html

``` python
import mxnet as mx
import tvm
import tvm.relay as relay
import numpy as np
```

## 从 Gluon Model Zoo 下载 Resnet18 模型

本节会下载预训练的 imagenet 模型，并对图像进行分类。

``` python
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
from matplotlib import pyplot as plt

block = get_model("resnet18_v1", pretrained=True)
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_name = "cat.png"
synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
img_path = download_testdata(img_url, "cat.png", module="data")
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())
image = Image.open(img_path).resize((224, 224))
plt.imshow(image)
plt.show()

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)
print("x", x.shape)
```

 ![from mxnet](https://tvm.apache.org/docs/_images/sphx_glr_from_mxnet_001.png)

输出结果：

``` bash
Downloading /workspace/.mxnet/models/resnet18_v1-a0666292.zip08d19deb-ddbf-4120-9643-fcfab19e7541 from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/resnet18_v1-a0666292.zip...
x (1, 3, 224, 224)
```

## 编译计算图

只需几行代码，即可将 Gluon 模型移植到可移植计算图上。mxnet.gluon 支持 MXNet 静态图（符号）和 HybridBlock。

``` python
shape_dict = {"data": x.shape}
mod, params = relay.frontend.from_mxnet(block, shape_dict)
## 添加 softmax 算子来提高概率
func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)
```

接下来编译计算图：

``` python
target = "cuda"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target, params=params)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 在 TVM 上执行可移植计算图

接下来用 TVM 重现相同的前向计算：

``` python
from tvm.contrib import graph_executor

dev = tvm.cuda(0)
dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# 设置输入
m.set_input("data", tvm.nd.array(x.astype(dtype)))
# 执行
m.run()
# 得到输出
tvm_output = m.get_output(0)
top1 = np.argmax(tvm_output.numpy()[0])
print("TVM prediction top-1:", top1, synset[top1])
```

输出结果：

``` bash
TVM prediction top-1: 282 tiger cat
```

## 使用带有预训练权重的 MXNet 符号

MXNet 常用 *arg_params* 和 *aux_params* 分别存储网络参数，下面将展示如何在现有 API 中使用这些权重：

``` python
def block2symbol(block):
    data = mx.sym.Variable("data")
    sym = block(data)
    args = {}
    auxs = {}
    for k, v in block.collect_params().items():
        args[k] = mx.nd.array(v.data().asnumpy())
    return sym, args, auxs

mx_sym, args, auxs = block2symbol(block)
# 通常将其保存/加载为检查点
mx.model.save_checkpoint("resnet18_v1", 0, mx_sym, args, auxs)
# 磁盘上有 "resnet18_v1-0000.params" 和 "resnet18_v1-symbol.json"
```

对于一般性 MXNet 模型：

``` python
mx_sym, args, auxs = mx.model.load_checkpoint("resnet18_v1", 0)
# 用相同的 API 来获取 Relay 计算图
mod, relay_params = relay.frontend.from_mxnet(mx_sym, shape_dict, arg_params=args, aux_params=auxs)
# 重复相同的步骤，用 TVM 运行这个模型
```

[下载 Python 源代码：from_mxnet.py](https://tvm.apache.org/docs/_downloads/0e2f38fcb1a1fb3e636e5953aa600dee/from_mxnet.py)

[下载 Jupyter Notebook：from_mxnet.ipynb](https://tvm.apache.org/docs/_downloads/4bbcfcce3c35b0b795a42c998ceb3770/from_mxnet.ipynb)