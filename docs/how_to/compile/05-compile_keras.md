---
title: 编译 Keras 模型
---

# 编译 Keras 模型

注意：单击 [此处](https://tvm.apache.org/docs/how_to/compile_models/from_keras.html#sphx-glr-download-how-to-compile-models-from-keras-py) 下载完整的示例代码

**作者**：[Yuwei Hu](https://huyuwei.github.io/)

本文介绍如何用 Relay 部署 Keras 模型。

首先安装 Keras 和 TensorFlow，可通过 pip 快速安装：

``` bash
pip install -U keras --user
pip install -U tensorflow --user
```

或参考官网：https://keras.io/#installation

``` python
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import keras
import tensorflow as tf
import numpy as np
```

## 加载预训练的 Keras 模型

加载 Keras 提供的预训练 resnet-50 分类模型：

``` python
if tuple(keras.__version__.split(".")) < ("2", "4", "0"):
    weights_url = "".join(
        [
            "https://github.com/fchollet/deep-learning-models/releases/",
            "download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        ]
    )
    weights_file = "resnet50_keras_old.h5"
else:
    weights_url = "".join(
        [
            " https://storage.googleapis.com/tensorflow/keras-applications/",
            "resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        ]
    )
    weights_file = "resnet50_keras_new.h5"

weights_path = download_testdata(weights_url, weights_file, module="keras")
keras_resnet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
)
keras_resnet50.load_weights(weights_path)
```

## 加载测试图像

这里使用的还是先前猫咪的图像：

``` python
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
plt.imshow(img)
plt.show()
# 预处理输入
data = np.array(img)[np.newaxis, :].astype("float32")
data = preprocess_input(data).transpose([0, 3, 1, 2])
print("input_1", data.shape)
```

 ![图片](https://tvm.apache.org/docs/_images/sphx_glr_from_keras_001.png)

输出结果：

``` bash
input_1 (1, 3, 224, 224)
```

## 使用 Relay 编译模型

将 Keras 模型（NHWC 布局）转换为 Relay 格式（NCHW 布局）：

``` python
shape_dict = {"input_1": data.shape}
mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)
# 编译模型
target = "cuda"
dev = tvm.cuda(0)

# TODO(mbs)：opt_level=3 导致 nn.contrib_conv2d_winograd_weight_transform
# 很可能由于潜在的错误，最终出现在 cuda 上的内存验证失败的模块中。
# 注意：只能在 evaluate() 中传递 context，它不被 create_executor() 捕获。
with tvm.transform.PassContext(opt_level=0):
    model = relay.build_module.create_executor("graph", mod, dev, target, param).evaluate()
```

## 在 TVM 上执行

``` python
dtype = "float32"
tvm_out = model(tvm.nd.array(data.astype(dtype)))
top1_tvm = np.argmax(tvm_out.numpy()[0])
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
print("Relay top-1 id: {}, class name: {}".format(top1_tvm, synset[top1_tvm]))
# 验证 Keras 输出的正确性
keras_out = keras_resnet50.predict(data.transpose([0, 2, 3, 1]))
top1_keras = np.argmax(keras_out)
print("Keras top-1 id: {}, class name: {}".format(top1_keras, synset[top1_keras]))
```

输出结果：

``` bash
Relay top-1 id: 285, class name: Egyptian cat
Keras top-1 id: 285, class name: Egyptian cat
```

[下载 Python 源代码：from_keras.py](https://tvm.apache.org/docs/_downloads/c23f7654585d9b0fa2129e1765b2a8f2/from_keras.py)

[下载 Jupyter Notebook：from_keras.ipynb](https://tvm.apache.org/docs/_downloads/c82f632d47458e76d2af9821b6778e36/from_keras.ipynb)
