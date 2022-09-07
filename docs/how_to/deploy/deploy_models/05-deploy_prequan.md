---
title: 使用 TVM 部署框架预量化模型
---

# 使用 TVM 部署框架预量化模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/deploy_models/deploy_prequantized.html#sphx-glr-download-how-to-deploy-models-deploy-prequantized-py) 下载完整的示例代码
:::

**作者**：[Masahiro Masuda](https://github.com/masahi)

本文介绍如何将深度学习框架量化的模型加载到 TVM。预量化模型的导入是 TVM 中支持的量化之一。有关 TVM 中量化的更多信息，参阅 [此处](https://discuss.tvm.apache.org/t/quantization-story/3920)。

这里演示了如何加载和运行由 PyTorch、MXNet 和 TFLite 量化的模型。加载后，可以在任何 TVM 支持的硬件上运行编译后的量化模型。

首先，导入必要的包：

``` python
from PIL import Image
import numpy as np
import torch
from torchvision.models.quantization import mobilenet as qmobilenet

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
```

定义运行 demo 的辅助函数：

``` python
def get_transform():
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

def get_real_image(im_height, im_width):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    return Image.open(img_path).resize((im_height, im_width))

def get_imagenet_input():
    im = get_real_image(224, 224)
    preprocess = get_transform()
    pt_tensor = preprocess(im)
    return np.expand_dims(pt_tensor.numpy(), 0)

def get_synset():
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
        return eval(f.read())

def run_tvm_model(mod, params, input_name, inp, target="llvm"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device(target, 0)))

    runtime.set_input(input_name, inp)
    runtime.run()
    return runtime.get_output(0).numpy(), runtime
```

从标签到类名的映射，验证模型的输出是否合理：

``` python
synset = get_synset()
```

用猫的图像进行演示：

``` python
inp = get_imagenet_input()
```

## 部署量化的 PyTorch 模型

首先演示如何用 PyTorch 前端加载由 PyTorch 量化的深度学习模型。

参考 [PyTorch 静态量化教程](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)，了解量化的工作流程。

用下面的函数来量化 PyTorch 模型。此函数采用浮点模型，并将其转换为 uint8。这个模型是按通道量化的。

``` python
def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp)
    torch.quantization.convert(model, inplace=True)
```

## 从 torchvision 加载预量化、预训练的 Mobilenet v2 模型

之所以选择 mobilenet v2，是因为该模型接受了量化感知训练，而其他模型则需要完整的训练后校准。

``` python
qmodel = qmobilenet.mobilenet_v2(pretrained=True).eval()
```

输出结果：

``` bash
Downloading: "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth" to /workspace/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth

  0%|          | 0.00/13.6M [00:00<?, ?B/s]
 44%|####4     | 6.03M/13.6M [00:00<00:00, 63.2MB/s]
 89%|########8 | 12.1M/13.6M [00:00<00:00, 61.4MB/s]
100%|##########| 13.6M/13.6M [00:00<00:00, 66.0MB/s]
```

## 量化、跟踪和运行 PyTorch Mobilenet v2 模型

量化和 jit 的详细信息可参考 PyTorch 网站上的教程。

``` python
pt_inp = torch.from_numpy(inp)
quantize_model(qmodel, pt_inp)
script_module = torch.jit.trace(qmodel, pt_inp).eval()

with torch.no_grad():
    pt_result = script_module(pt_inp).numpy()
```

输出结果：

``` bash
/usr/local/lib/python3.7/dist-packages/torch/ao/quantization/observer.py:179: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  reduce_range will be deprecated in a future release of PyTorch."
/usr/local/lib/python3.7/dist-packages/torch/ao/quantization/observer.py:1126: UserWarning: must run observer before calling calculate_qparams.                                    Returning default scale and zero point
  Returning default scale and zero point "
```

## 使用 PyTorch 前端将量化的 Mobilenet v2 转换为 Relay-QNN

PyTorch 前端支持将量化的 PyTorch 模型，转换为具有量化感知算子的等效 Relay 模块。将此表示称为 Relay QNN dialect。

若要查看量化模型是如何表示的，可以从前端打印输出。

可以看到特定于量化的算子，例如 qnn.quantize、qnn.dequantize、qnn.requantize 和 qnn.conv2d 等。

``` python
input_name = "input"  # 对于 PyTorch 前端，输入名称可以是任意的。
input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
# print(mod) # 打印查看 QNN IR 转储
```

## 编译并运行 Relay 模块

获得量化的 Relay 模块后，剩下的工作流程与运行浮点模型相同。详细信息请参阅其他教程。

在底层，量化特定的算子在编译之前，会被降级为一系列标准 Relay 算子。

``` python
target = "llvm"
tvm_result, rt_mod = run_tvm_model(mod, params, input_name, inp, target=target)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 比较输出标签

可看到打印出相同的标签。

``` python
pt_top3_labels = np.argsort(pt_result[0])[::-1][:3]
tvm_top3_labels = np.argsort(tvm_result[0])[::-1][:3]

print("PyTorch top3 labels:", [synset[label] for label in pt_top3_labels])
print("TVM top3 labels:", [synset[label] for label in tvm_top3_labels])
```

输出结果：

``` bash
PyTorch top3 labels: ['tiger cat', 'Egyptian cat', 'tabby, tabby cat']
TVM top3 labels: ['tiger cat', 'Egyptian cat', 'tabby, tabby cat']
```

但由于数字的差异，通常原始浮点输出不应该是相同的。下面打印 mobilenet v2 的 1000 个输出中，有多少个浮点输出值是相同的。

``` python
print("%d in 1000 raw floating outputs identical." % np.sum(tvm_result[0] == pt_result[0]))
```

输出结果：

``` bash
154 in 1000 raw floating outputs identical.
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
  90.3752      90.2667      94.6845      90.0629       0.6087
```

:::note
推荐这种方法的原因如下：

* 测试是在 C++ 中完成的，因此没有 Python 开销大
* 包括几个准备工作
* 可用相同的方法在远程设备（Android 等）上进行分析。
:::

:::note
如果硬件对 INT8 整数的指令没有特殊支持，量化模型与 FP32 模型速度相近。如果没有 INT8 整数的指令，TVM 会以 16 位进行量化卷积，即使模型本身是 8 位。

对于 x86，在具有 AVX512 指令集的 CPU 上可实现最佳性能。这种情况 TVM 对给定 target 使用最快的可用 8 位指令，包括对 VNNI 8 位点积指令（CascadeLake 或更新版本）的支持。

此外，以下一般技巧对 CPU 性能的提升同样适用：

* 将环境变量 TVM_NUM_THREADS 设置为物理 core 的数量
* 为硬件选择最佳 target，例如 "llvm -mcpu=skylake-avx512" 或 "llvm -mcpu=cascadelake"（未来会有更多支持 AVX512 的 CPU）
:::

## 部署量化的 MXNet 模型

待更新

## 部署量化的 TFLite 模型

待更新

**脚本总运行时长：**（1 分 7.374 秒）

[下载 Python 源代码：deploy_prequantized.py](https://tvm.apache.org/docs/_downloads/fb8217c13f4351224c6cf3aacf1a87fc/deploy_prequantized.py)

[下载 Jupyter Notebook：deploy_prequantized.ipynb](https://tvm.apache.org/docs/_downloads/c20f81a94729f461f33b52cc110fd9d6/deploy_prequantized.ipynb)