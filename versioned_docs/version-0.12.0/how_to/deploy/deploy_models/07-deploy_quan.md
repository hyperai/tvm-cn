---
title: 在 CUDA 上部署量化模型
---

# 在 CUDA 上部署量化模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/deploy_models/deploy_quantized.html#sphx-glr-download-how-to-deploy-models-deploy-quantized-py) 下载完整的示例代码
:::

**作者**：[Wuwei Lin](https://github.com/vinx13)

本文介绍如何用 TVM 自动量化（TVM 的一种量化方式）。有关 TVM 中量化的更多详细信息，参阅 [此处](https://discuss.tvm.apache.org/t/quantization-story/3920)。本教程将在 ImageNet 上导入一个 GluonCV 预训练模型到 Relay，量化 Relay 模型，然后执行推理。

``` python
import tvm
from tvm import te
from tvm import relay
import mxnet as mx
from tvm.contrib.download import download_testdata
from mxnet import gluon
import logging
import os

batch_size = 1
model_name = "resnet18_v1"
target = "cuda"
dev = tvm.device(target)
```

## 准备数据集

以下演示如何为量化准备校准数据集，首先下载 ImageNet 的验证集，并对数据集进行预处理。

``` python
calibration_rec = download_testdata(
    "http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/data/val_256_q90.rec",
    "val_256_q90.rec",
)

def get_val_data(num_workers=4):
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch):
        return batch.data[0].asnumpy(), batch.label[0].asnumpy()

    img_size = 299 if model_name == "inceptionv3" else 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec=calibration_rec,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,
        resize=256,
        data_shape=(3, img_size, img_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )
    return val_data, batch_fn
```

把校准数据集（可迭代对象）定义为 Python 中的生成器对象，本教程仅用几个样本进行校准。

``` python
calibration_samples = 10

def calibrate_dataset():
    val_data, batch_fn = get_val_data()
    val_data.reset()
    for i, batch in enumerate(val_data):
        if i * batch_size >= calibration_samples:
            break
        data, _ = batch_fn(batch)
        yield {"data": data}
```

## 导入模型

用 Relay MxNet 前端从 Gluon 模型集合（model zoo）中导入模型。

``` python
def get_model():
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == "inceptionv3" else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    return mod, params
```

## 量化模型

量化过程要找到每一层的每个权重和中间特征图（feature map）张量的 scale。

对于权重而言，scales 是根据权重的值直接计算出来的。支持两种模式：power2 和 max。这两种模式都是先找到权重张量内的最大值。在 power2 模式下，最大值向下舍入为 2 的幂。如果权重和中间特征图的 scale 都是 2 的幂，则可以利用移位（bit shifting）进行乘法运算，这使得计算效率更高。在 max 模式下，最大值用作 scale。如果不进行四舍五入，在某些情况下 max 模式可能具有更好的精度。当 scale 不是 2 的幂时，将使用定点乘法。

中间特征图可以通过数据感知量化来找到 scale。数据感知量化将校准数据集作为输入参数，通过最小化量化前后激活分布之间的 KL 散度来计算 scales。或者也可以用预定义的全局 scales，这样可以节省校准时间，但会影响准确性。

``` python
def quantize(mod, params, data_aware):
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max"):
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod
```

## 运行推理

创建一个 Relay VM 来构建和执行模型。

``` python
def run_inference(mod):
    model = relay.create_executor("vm", mod, dev, target).evaluate()
    val_data, batch_fn = get_val_data()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch)
        prediction = model(data)
        if i > 10:  # 本教程只对几个样本进行推理
            break

def main():
    mod, params = get_model()
    mod = quantize(mod, params, data_aware=True)
    run_inference(mod)

if __name__ == "__main__":
    main()
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
/workspace/python/tvm/relay/build_module.py:411: DeprecationWarning: Please use input parameter mod (tvm.IRModule) instead of deprecated parameter mod (tvm.relay.function.Function)
  DeprecationWarning,
```

**脚本总运行时长：**（1 分 22.338 秒）

[下载 Python 源代码：deploy_quantized.py](https://tvm.apache.org/docs/_downloads/7810ecf51bfc05f7d5e8a400ac3e815d/deploy_quantized.py)

[下载 Jupyter Notebook：deploy_quantized.ipynb](https://tvm.apache.org/docs/_downloads/a269cb38341b190be980a0bd3ea8a625/deploy_quantized.ipynb)
