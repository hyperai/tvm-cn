---
title: 编译 TFLite 模型
---

# 编译 TFLite 模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/compile_models/from_tflite.html#sphx-glr-download-how-to-compile-models-from-tflite-py) 下载完整的示例代码
:::

**作者**：[Zhao Wu](https://github.com/FrozenGene)

本文介绍如何用 Relay 部署 TFLite 模型。

首先安装 TFLite 包。

``` bash
pip install tflite==2.1.0
```

或者自行生成 TFLite 包，步骤如下：

``` bash
# 获取 flatc 编译器。
# 详细可参考 https://github.com/google/flatbuffers，确保正确安装
flatc --version

# 获取 TFLite 架构
wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.13/tensorflow/lite/schema/schema.fbs

# 生成 TFLite 包
flatc --python schema.fbs

# 将当前文件夹路径（包含生成的 TFLite 模块）添加到 PYTHONPATH。
export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$(pwd)
```

用 `python -c "import tflite"` 命令，检查 TFLite 包是否安装成功。

有关如何用 TVM 编译 TFLite 模型的示例如下：

## 用于下载和提取 zip 文件的程序

``` python
import os

def extract(path):
    import tarfile

    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError("Could not decompress the file: " + path)
```

## 加载预训练的 TFLite 模型

加载 Google 提供的 mobilenet V1 TFLite 模型：

``` python
from tvm.contrib.download import download_testdata

model_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"

# 下载模型 tar 文件，解压得到 mobilenet_v1_1.0_224.tflite
model_path = download_testdata(model_url, "mobilenet_v1_1.0_224.tgz", module=["tf", "official"])
model_dir = os.path.dirname(model_path)
extract(model_path)

# 打开 mobilenet_v1_1.0_224.tflite
tflite_model_file = os.path.join(model_dir, "mobilenet_v1_1.0_224.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()

# 从缓冲区获取 TFLite 模型
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
```

## 加载测试图像

还是用猫的图像：

``` python
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

image_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
image_path = download_testdata(image_url, "cat.png", module="data")
resized_image = Image.open(image_path).resize((224, 224))
plt.imshow(resized_image)
plt.show()
image_data = np.asarray(resized_image).astype("float32")

# 给图像添加一个维度，形成 NHWC 格式布局
image_data = np.expand_dims(image_data, axis=0)

# 预处理图像:
# https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
print("input", image_data.shape)
```

 ![图片](https://tvm.apache.org/docs/_images/sphx_glr_from_tflite_001.png)

输出结果：

``` bash
input (1, 224, 224, 3)
```

## 使用 Relay 编译模型

``` python
# TFLite 输入张量名称、shape 和类型
input_tensor = "input"
input_shape = (1, 224, 224, 3)
input_dtype = "float32"

# 解析 TFLite 模型，并将其转换为 Relay 模块
from tvm import relay, transform

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

# 针对 x86 CPU 构建模块
target = "llvm"
with transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 在 TVM 上执行

``` python
import tvm
from tvm import te
from tvm.contrib import graph_executor as runtime

# 创建 runtime 执行器模块
module = runtime.GraphModule(lib["default"](tvm.cpu()))

# 输入数据
module.set_input(input_tensor, tvm.nd.array(image_data))

# 运行
module.run()

# 得到输出
tvm_output = module.get_output(0).numpy()
```

## 显示结果

``` python
# 加载标签文件
label_file_url = "".join(
    [
        "https://raw.githubusercontent.com/",
        "tensorflow/tensorflow/master/tensorflow/lite/java/demo/",
        "app/src/main/assets/",
        "labels_mobilenet_quant_v1_224.txt",
    ]
)
label_file = "labels_mobilenet_quant_v1_224.txt"
label_path = download_testdata(label_file_url, label_file, module="data")

# 1001 个类的列表
with open(label_path) as f:
    labels = f.readlines()

# 将结果转换为一维数据
predictions = np.squeeze(tvm_output)

# 获得分数最高的第一个预测值
prediction = np.argmax(predictions)

# 将 id 转换为类名，并显示结果
print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])
```

输出结果：

``` bash
The image prediction result is: id 283 name: tiger cat
```

[下载 Python 源代码：from_tflite.py](https://tvm.apache.org/docs/_downloads/a70662bf8dc171d3d17a3945bbbb02e3/from_tflite.py)

[下载 Jupyter Notebook：from_tflite.ipynb](https://tvm.apache.org/docs/_downloads/23968bb778cd9591b7ad858bf17dcc3e/from_tflite.ipynb)