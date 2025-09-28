---
title: 编译 CoreML 模型
---

# 编译 CoreML 模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/compile_models/from_coreml.html#sphx-glr-download-how-to-compile-models-from-coreml-py) 下载完整的示例代码
:::

**作者**：[Joshua Z. Zhang](https://zhreshold.github.io/)，[Kazutaka Morita](https://github.com/kazum)，[Zhao Wu](https://github.com/FrozenGene)

本文介绍如何用 Relay 部署 CoreML 模型。

首先安装 coremltools 模块：

``` bash
pip install coremltools
```

或参考官网：https://github.com/apple/coremltools

``` python
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import coremltools as cm
import numpy as np
from PIL import Image
```

## 加载预训练的 CoreML 模型

这个例子使用 Apple 提供的预训练的 mobilenet 分类网络。

``` python
model_url = "https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel"
model_file = "mobilenet.mlmodel"
model_path = download_testdata(model_url, model_file, module="coreml")
# 现在磁盘上有 mobilenet.mlmodel 模型
mlmodel = cm.models.MLModel(model_path)
```

## 加载测试图像

还是用猫的图像：

``` python
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
# Mobilenet.mlmodel 的输入是 BGR 格式
img_bgr = np.array(img)[:, :, ::-1]
x = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, :]
```

## 在 Relay 上编译模型

现在应该对这个过程较为熟悉了。

``` python
target = "llvm"
shape_dict = {"image": x.shape}

# 解析 CoreML 模型，并转换为 Relay 计算图
mod, params = relay.frontend.from_coreml(mlmodel, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 在 TVM 上执行

这个过程与其他示例的相同。

``` python
from tvm.contrib import graph_executor

dev = tvm.cpu(0)
dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# 设置输入
m.set_input("image", tvm.nd.array(x.astype(dtype)))
# 执行
m.run()
# 得到输出
tvm_output = m.get_output(0)
top1 = np.argmax(tvm_output.numpy()[0])
```

## 查找分类集名称

在 1000 个类的分类集中，查找分数最高的第一个：

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
    synset = eval(f.read())
# 结果应为 Top-1 id 282 class name tiger cat
print("Top-1 id", top1, "class name", synset[top1])
```

输出结果：

``` bash
Top-1 id 282 class name tiger cat
```

[下载 Python 源代码：from_coreml.py](https://tvm.apache.org/docs/_downloads/3aeab7c9d659bf5da70126a1aff7c403/from_coreml.py)

[下载 Jupyter Notebook：from_coreml.ipynb](https://tvm.apache.org/docs/_downloads/a883b8474634054b6a79c17a288aa8ed/from_coreml.ipynb)
