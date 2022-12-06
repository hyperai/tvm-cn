---
title: 编译 ONNX 模型
---

# 编译 ONNX 模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/compile_models/from_onnx.html#sphx-glr-download-how-to-compile-models-from-onnx-py) 下载完整的示例代码
:::

**作者**：[Joshua Z. Zhang](https://zhreshold.github.io/)

本文将介绍如何用 Relay 部署 ONNX 模型。

首先安装 ONNX 包，最便捷的方法推荐安装 protobuf 编译器：

``` bash
pip install --user onnx onnxoptimizer
```

或参考官方网站：https://github.com/onnx/onnx

``` python
import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
```

## 加载预训练的 ONNX 模型

下面示例中的超分辨率模型与 [ONNX 教程](http://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html) 中的模型完全相同，跳过 PyTorch 模型的构建部分，下载保存的 ONNX 模型：

``` python
model_url = "".join(
    [
        "https://gist.github.com/zhreshold/",
        "bcda4716699ac97ea44f791c24310193/raw/",
        "93672b029103648953c4e5ad3ac3aadf346a4cdc/",
        "super_resolution_0.2.onnx",
    ]
)
model_path = download_testdata(model_url, "super_resolution.onnx", module="onnx")
# 现在磁盘上有 super_resolution.onnx 模型
onnx_model = onnx.load(model_path)
```

## 加载测试图像

该模型接收大小为 224x224 的单个图像作为输入，输出沿每个轴放大 3 倍的图像（即大小为 672x672）。为适配输入的 shape，重新缩放猫图像，并转换为 *YCbCr*。然后超分辨率模型应用于亮度（*Y*）通道。

``` python
from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
img_y, img_cb, img_cr = img_ycbcr.split()
x = np.array(img_y)[np.newaxis, np.newaxis, :, :]
```

## 使用 Relay 编译模型

通常 ONNX 模型将输入值与参数值混合在一起，输入名称为 *1*，具体要查看模型文档来确定完整的输入和参数名称空间。

将 shape 字典传给 *relay.frontend.from_onnx* 方法，以便 Relay 知道哪些 ONNX 参数是输入，哪些是参数，并提供输入尺寸的静态定义：

``` python
target = "llvm"

input_name = "1"
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=1):
    executor = relay.build_module.create_executor(
        "graph", mod, tvm.cpu(0), target, params
    ).evaluate()
```

输出结果：

``` bash
/workspace/python/tvm/relay/frontend/onnx.py:5785: UserWarning: Mismatched attribute type in ' : kernel_shape'

==> Context: Bad node spec for node. Name:  OpType: Conv
  warnings.warn(str(e))
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 在 TVM 上执行

``` python
dtype = "float32"
tvm_output = executor(tvm.nd.array(x.astype(dtype))).numpy()
```

## 查看结果

将输入和输出图像放在一起比对。亮度通道 *Y*是模型的输出。将色度通道 *Cb* 和 *Cr* 调整到匹配简单的双三次算法，然后将图像重新组合，并转换回 *RGB*。

``` python
from matplotlib import pyplot as plt

out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode="L")
out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
canvas = np.full((672, 672 * 2, 3), 255)
canvas[0:224, 0:224, :] = np.asarray(img)
canvas[:, 672:, :] = np.asarray(result)
plt.imshow(canvas.astype(np.uint8))
plt.show()
```

 ![from onnx](https://tvm.apache.org/docs/_images/sphx_glr_from_onnx_001.png)

输出结果：

``` bash
/workspace/gallery/how_to/compile_models/from_onnx.py:120: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
/workspace/gallery/how_to/compile_models/from_onnx.py:121: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
```

## 注意

ONNX 导入器在导入时默认根据动态 shape 定义模型，编译器在编译时将模型转换为静态 shape。如果失败，模型中可能仍存在动态操作。目前并非所有 TVM 内核都支持动态 shape，如果遇到动态内核错误，请在 discuss.tvm.apache.org 上提交 issue。

这个特定的模型是用旧版本的 ONNX 构建的。在导入阶段，ONNX 导入器运行 ONNX 验证程序（可能抛出属性类型不匹配的警告）。由于 TVM 支持许多不同的 ONNX 版本，所以 Relay 模型仍然有效。

[下载 Python 源代码：from_onnx.py](https://tvm.apache.org/docs/_downloads/eb551cfff8900ec35fae9f15aa728e45/from_onnx.py)

[下载 Jupyter Notebook：from_onnx.ipynb](https://tvm.apache.org/docs/_downloads/779f52a44f2b8ab22dc21eee0c27fd4d/from_onnx.ipynb)
