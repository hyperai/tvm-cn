---
title: 4. microTVM PyTorch 教程
---

:::note
单击 [此处](https://tvm.apache.org/docs/v0.13.0/how_to/work_with_microtvm/micro_pytorch.html#sphx-glr-download-how-to-work-with-microtvm-micro-pytorch-py) 下载完整的示例代码
:::

# 4.microTVM PyTorch 教程
**作者：**[Mehrdad Hessar](https://github.com/mehrdadh)

该教程展示了如何使用 PyTorch 模型进行 microTVM 主机驱动的 AOT 编译。此教程可以在使用 C 运行时（CRT）的 x86 CPU 上执行。

**注意：** 此教程仅在使用 CRT 的 x86 CPU 上运行，不支持在 Zephyr 上运行，因为该模型不适用于我们当前支持的 Zephyr 单板。

## 安装 microTVM Python 依赖项
TVM 不包含用于 Python 串行通信包，因此在使用 microTVM 之前我们必须先安装一个。我们还需要TFLite来加载模型。

```bash
pip install pyserial==3.5 tflite==2.1
```

```python
import pathlib
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.relay.backend import Executor
import tvm.micro.testing

```

## 加载预训练 PyTorch 模型
首先，从 torchvision 中加载预训练的 MobileNetV2 模型。然后，下载一张猫的图像并进行预处理，以便用作模型的输入。

```python
model = torchvision.models.quantization.mobilenet_v2(weights="DEFAULT", quantize=True)
model = model.eval()

input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# 预处理图片并转换为张量
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

input_name = "input0"
shape_list = [(input_name, input_shape)]
relay_mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

```

输出：

```
/venv/apache-tvm-py3.8/lib/python3.8/site-packages/torch/ao/quantization/utils.py:310: UserWarning: must run observer before calling calculate_qparams. Returning default values.
  warnings.warn(
Downloading: "https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth" to /workspace/.cache/torch/hub/checkpoints/mobilenet_v2_qnnpack_37f702c5.pth

  0%|          | 0.00/3.42M [00:00<?, ?B/s]
 61%|######    | 2.09M/3.42M [00:00<00:00, 11.6MB/s]
100%|##########| 3.42M/3.42M [00:00<00:00, 18.5MB/s]
/venv/apache-tvm-py3.8/lib/python3.8/site-packages/torch/_utils.py:314: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  device=storage.device,
/workspace/python/tvm/relay/frontend/pytorch_utils.py:47: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  return LooseVersion(torch_ver) > ver
/venv/apache-tvm-py3.8/lib/python3.8/site-packages/setuptools/_distutils/version.py:346: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  other = LooseVersion(other)

```

## 定义目标、运行时与执行器
在本教程中，我们使用 AOT 主机驱动执行器。为了在 x86 机器上对嵌入式模拟环境编译模型，我们使用 C 运行时（CRT），并使用主机微型目标。使用该设置，TVM 为 C 运行时编译可以在 x86 CPU 机器上运行的模型，可以在物理微控制器上运行的相同流程。CRT 使用 `src/runtime/crt/host/main.cc` 中的 main()。要使用物理硬件，请将 board 替换为另一个物理微型目标，例如 `nrf5340dk_nrf5340_cpuapp` 或 `mps2_an521`，并将平台类型更改为 Zephyr。在《[为 Arduino 上的 microTVM 训练视觉模型](https://tvm.apache.org/docs/v0.13.0/how_to/work_with_microtvm/micro_train.html#tutorial-micro-train-arduino)》和《[microTVM TFLite 教程](https://tvm.apache.org/docs/v0.13.0/how_to/work_with_microtvm/micro_tflite.html#tutorial-micro-tflite)》中，可以看到 更多目标示例。

```python
target = tvm.micro.testing.get_target(platform="crt", board=None)

# 使用 C 运行时 (crt) 并通过设置 system-lib 为 True 打开静态链接
runtime = tvm.relay.backend.Runtime("crt", {"system-lib": True})

# 使用 AOT 执行器代替图或 vm 执行器。不要使用未包装的 API 或 C 风格调用
executor = Executor("aot")

```

## 编译模型
现在为目标编译模型：

```python
with tvm.transform.PassContext(
    opt_level=3,
    config={"tir.disable_vectorize": True},
):
    module = tvm.relay.build(
        relay_mod, target=target, runtime=runtime, executor=executor, params=params
    )

```

## 创建 microTVM 项目
现在，我们已经将编译好的模型作为 IRModule 准备好，我们还需要创建一个固件项目，以便在 microTVM 中使用编译好的模型。为此，我们需要使用 Project API。

```python
template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
project_options = {"verbose": False, "workspace_size_bytes": 6 * 1024 * 1024}

temp_dir = tvm.contrib.utils.tempdir() / "project"
project = tvm.micro.generate_project(
    str(template_project_path),
    module,
    temp_dir,
    project_options,
)
```

## 构建、烧录和执行模型
接下来，我们构建 microTVM项 目并进行烧录。烧录步骤特定于物理微控制器，如果通过主机的  `main.cc` 模拟微控制器，或者选择 Zephyr 模拟单板作为目标，则会跳过该步骤。

```python
project.build()
project.flash()

input_data = {input_name: tvm.nd.array(img.astype("float32"))}
with tvm.micro.Session(project.transport()) as session:
    aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
    aot_executor.set_input(**input_data)
    aot_executor.run()
    result = aot_executor.get_output(0).numpy()
```

## 查询 Synset 名称
查询在 1000 个类别 Synset 中的 top-1 的预测。

```python
synset_url = (
    "https://raw.githubusercontent.com/Cadene/"
    "pretrained-models.pytorch/master/data/"
    "imagenet_synsets.txt"
)
synset_name = "imagenet_synsets.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

class_url = (
    "https://raw.githubusercontent.com/Cadene/"
    "pretrained-models.pytorch/master/data/"
    "imagenet_classes.txt"
)
class_path = download_testdata(class_url, "imagenet_classes.txt", module="data")
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Get top-1 result for TVM
top1_tvm = np.argmax(result)
tvm_class_key = class_id_to_key[top1_tvm]

# Convert input to PyTorch variable and get PyTorch result for comparison
with torch.no_grad():
    torch_img = torch.from_numpy(img)
    output = model(torch_img)

    # Get top-1 result for PyTorch
    top1_torch = np.argmax(output.numpy())
    torch_class_key = class_id_to_key[top1_torch]

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))

```

输出结果：

```
Relay top-1 id: 282, class name: tiger cat
Torch top-1 id: 282, class name: tiger cat
```

**该脚本总运行时间：**（1分26.552秒）

[下载 Python 源代码：micro_pytorch.py](https://tvm.apache.org/docs/v0.13.0/_downloads/12b9ecc04c41abaa12022061771821d1/micro_pytorch.py)

[下载 Jupyter notebook：micro_pytorch.ipynb](https://tvm.apache.org/docs/v0.13.0/_downloads/09df7d9b9c90a2a1bdd570520693fd9f/micro_pytorch.ipynb)