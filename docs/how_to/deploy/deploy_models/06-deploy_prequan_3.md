---
title: 使用 TVM 部署框架预量化模型 - 第 3 部分 (TFLite)
---

# 使用 TVM 部署框架预量化模型 - 第 3 部分 (TFLite)

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/deploy_models/deploy_prequantized_tflite.html#sphx-glr-download-how-to-deploy-models-deploy-prequantized-tflite-py) 下载完整的示例代码
:::

**作者**：[Siju Samuel](https://github.com/siju-samuel)

此教程介绍如何量化 TFLite 计算图，并通过 TVM 编译和执行。

有关使用 TFLite 量化模型的更多详细信息，参阅 [转换量化模型](https://www.tensorflow.org/lite/convert/quantization)。

TFLite 模型下载 [链接](https://www.tensorflow.org/lite/guide/hosted_models)。

开始前，先安装 TensorFlow 和 TFLite 包。

``` bash
# 安装 tensorflow 和 tflite
pip install tensorflow==2.1.0
pip install tflite==2.1.0
```

执行 `python -c "import tflite"` 命令检查 TFLite 包是否安装成功。

## 导入必要的包

``` python
import os
import numpy as np
import tflite

import tvm
from tvm import relay
```

## 下载预训练的量化 TFLite 模型

``` python
# 下载 Google 的 mobilenet V2 TFLite 模型
from tvm.contrib.download import download_testdata

model_url = (
    "https://storage.googleapis.com/download.tensorflow.org/models/"
    "tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz"
)

# 下载模型 tar 文件，解压得到 mobilenet_v2_1.0_224.tflite
model_path = download_testdata(
    model_url, "mobilenet_v2_1.0_224_quant.tgz", module=["tf", "official"]
)
model_dir = os.path.dirname(model_path)
```

## 下载及提取 zip 文件的所有函数

``` python
def extract(path):
    import tarfile

    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError("Could not decompress the file: " + path)



extract(model_path)
```

## 加载测试图像

## 获取 e2e 测试的真实图像

``` python
def get_real_image(im_height, im_width):
    from PIL import Image

    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype("uint8")
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data

data = get_real_image(224, 224)
```

## 加载 TFLite 模型

打开 mobilenet_v2_1.0_224.tflite：

``` python
tflite_model_file = os.path.join(model_dir, "mobilenet_v2_1.0_224_quant.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
```

运行 TFLite 预量化模型推理，获得 TFLite 预测。

``` python
def run_tflite_model(tflite_model_buf, input_data):
    """执行 TFLite 的通用函数"""
    try:
        from tensorflow import lite as interpreter_wrapper
    except ImportError:
        from tensorflow.contrib import lite as interpreter_wrapper

    input_data = input_data if isinstance(input_data, list) else [input_data]

    interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 设置输入
    assert len(input_data) == len(input_details)
    for i in range(len(input_details)):
        interpreter.set_tensor(input_details[i]["index"], input_data[i])

    # 运行
    interpreter.invoke()

    # 得到输出
    tflite_output = list()
    for i in range(len(output_details)):
        tflite_output.append(interpreter.get_tensor(output_details[i]["index"]))

    return tflite_output
```

运行 TVM 编译的预量化模型推理，获得 TVM 预测。

``` python
def run_tvm(lib):
    from tvm.contrib import graph_executor

    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    rt_mod.set_input("input", data)
    rt_mod.run()
    tvm_res = rt_mod.get_output(0).numpy()
    tvm_pred = np.squeeze(tvm_res).argsort()[-5:][::-1]
    return tvm_pred, rt_mod
```

## 使用 TFLite 推理

在量化模型上运行 TFLite 推理：

``` python
tflite_res = run_tflite_model(tflite_model_buf, data)
tflite_pred = np.squeeze(tflite_res).argsort()[-5:][::-1]
```

## TVM 编译和推理

用 TFLite-Relay 解析器将 TFLite 预量化计算图，转换为 Relay IR。注意，预量化模型的前端解析器调用与 FP32 模型的前端解析器调用完全相同。推荐删除 print(mod) 中的注释，并检查 Relay 模块。可以看到许多 QNN 算子，例如 Requantize、Quantize 和 QNN Conv2D。

``` python
dtype_dict = {"input": data.dtype.name}
shape_dict = {"input": data.shape}

mod, params = relay.frontend.from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)
# print(mod)
```

使用 "llvm" target（或替换为其他平台）编译 Relay 模块。

``` python
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build_module.build(mod, target=target, params=params)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

最后在 TVM 编译模块上调用推理。

``` python
tvm_pred, rt_mod = run_tvm(lib)
```

## 精度比较

因为 TFLite 和 Relay 之间的再量化实现不同，导致最终输出数字不匹配。因此打印 MXNet 和 TVM 推理的前 5 个标签，通过标签测试准确性。

``` python
print("TVM Top-5 labels:", tvm_pred)
print("TFLite Top-5 labels:", tflite_pred)
```

输出结果：

``` bash
TVM Top-5 labels: [387 102 386 341 349]
TFLite Top-5 labels: [387 102 386 341 349]
```

## 测试性能

以下举例说明如何测试 TVM 编译模型的性能。

``` python
n_repeat = 100  # 为使测试更准确，应选取更大的数值
dev = tvm.cpu(0)
print(rt_mod.benchmark(dev, number=1, repeat=n_repeat))
```

输出结果：

``` bash
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
  119.3810     119.3190     121.4095     118.6146      0.3802
```

:::note
如果硬件对 INT8 整数的指令没有特殊支持，量化模型与 FP32 模型速度相近。如果没有 INT8 整数的指令，TVM 会以 16 位进行量化卷积，即使模型本身是 8 位。

对于 x86，可以在具有 AVX512 指令集的 CPU 上实现最佳性能。在这种情况下，TVM 对给定 targe 使用最快的可用 8 位指令。包括对 VNNI 8 位点积指令（CascadeLake 或更新版本）的支持。对于 EC2 C5.12x 大型实例，本教程的 TVM 延迟约为 2 毫秒。

与许多 TFLite 网络的 ARM NCHW conv2d 空间包调度相比，ARM 上的英特尔 conv2d NCHWc 调度提供了更好的端到端延迟。 ARM winograd 性能更高，但内存占用较高。

此外，以下有关 CPU 性能的一般技巧同样适用：

* 将环境变量 TVM_NUM_THREADS 设置为物理内核数
* 为硬件选择最佳 target，例如「llvm -mcpu=skylake-avx512」或「llvm -mcpu=cascadelake」（未来会出现更多支持 AVX512 的 CPU）
* 执行自动调优 - [为 x86 CPU 自动调优卷积网络](../../autotune/autotuning_x86)。
* 要在 ARM CPU 上获得最佳推理性能，根据设备更改 target 参数并遵循 [自动调整 ARM CPU 的卷积网络](../../autotune/autotuning_arm)。
:::

**脚本总运行时长：**（1 分 52.874 秒）

[下载 Python 源代码：deploy_prequantized_tflite.py](https://tvm.apache.org/docs/_downloads/56691c7a27d45da61d112276334640d3/deploy_prequantized_tflite.py)

[下载 Jupyter Notebook：deploy_prequantized_tflite.ipynb](https://tvm.apache.org/docs/_downloads/1a26d790f7b98309d730181290dae3ee/deploy_prequantized_tflite.ipynb)