---
title: 使用 TVMC 编译和优化模型
---

# 使用 TVMC 编译和优化模型
**作者**：[Leandro Nunes](https://github.com/leandron)，[Matthew Barrett](https://github.com/mbaret)，[Chris Hoge](https://github.com/hogepodge)

本节将介绍 TVMC（TVM 的命令行驱动程序）。TVMC 通过命令行界面执行 TVM 功能（包括对模型的自动调优、编译、分析和执行）。

学完本节后，可用 TVMC 实现下面的任务：

* 为 TVM runtime 编译预训练的 ResNet-50 v2 模型。
* 用编译好的模型预测真实图像，并解释输出和模型性能。
* 使用 TVM 在 CPU上调优模型。
* 用 TVM 收集的调优数据，重新编译优化过的模型。
* 通过优化的模型预测图像，并比较输出和模型性能。

本节对 TVM 及 TVMC 的功能进行了概述，并为了解 TVM 的工作原理奠定基础。

## 使用 TVMC

TVMC 是 Python 应用程序，也是 TVM Python 软件包的一部分。用 Python 包安装 TVM 时，会得到一个叫 ```tvmc``` 的命令行应用程序。平台和安装方法不同，此命令的位置也会发生变化。

另外，如果 ```$PYTHONPATH``` 上有 TVM 这个 Python 模块，则可通过可执行 Python 模块（用 ```python -m tvm.driver.tvmc``` 命令）来访问命令行驱动功能。

本教程用 ```tvmc <options>``` 或 ```python -m tvm.driver.tvmc <options>``` 来打开 TVMC 命令行。

使用如下命令查看帮助页：

``` bash
tvmc --help
```

```tvmc``` 可用的 TVM 的主要功能来自子命令 ```compile```、```run``` 和 ```tune```。使用 ```tvmc <subcommand> --help``` 查看给定子命令的特定选项。本教程将介绍这些命令，开始前请先下载一个预训练的模型。

## 获取模型

在本教程中，我们将使用 ResNet-50 v2。ResNet-50 是一个用来对图像进行分类的 50 层深的卷积神经网络。接下来要用的模型，已经在超过100万张具有1000种不同分类的图像上，进行了预训练。该网络的输入图像的大小为224x224。推荐下载 [Netron](https://netron.app/)（免费的 ML 模型查看器）来更深入地探索 ResNet-50 模型的组织结构。

本教程使用 ONNX 格式的模型：
``` bash
wget https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx
```
:::note 支持的模型格式：
TVMC 支持用 Keras、ONNX、TensorFlow、TFLite 和 Torch 创建的模型。可用 ```--model-format``` 选项指明正在使用的模型格式。执行 ```tvmc compile --help``` 来获取更多信息。
:::

:::note 向 TVM 添加对 ONNX 的支持
TVM 依赖系统中可用的 ONNX Python 库。用命令 ```pip3 install --user onnx onnxoptimizer``` 来安装 ONNX。如果具有 root 访问权限并且希望全局安装 ONNX，则可以删除 ```--user``` 选项。```onnxoptimizer``` 依赖是可选的，仅用于 ```onnx>=1.9```。
:::

## 将 ONNX 模型编译到 TVM Runtime

下载 ResNet-50 模型后，用 ```tvmc compile``` 对其进行编译。编译的输出结果是模型（被编译为目标平台的动态库）的 TAR 包。用 TVM runtime 可在目标设备上运行该模型：

``` bash
# 大概需要几分钟，取决于设备
tvmc compile \
--target "llvm" \
--input-shapes "data:[1,3,224,224]" \
--output resnet50-v2-7-tvm.tar \
resnet50-v2-7.onnx
```

查看 ```tvmc compile``` 在模块中创建的文件：

``` bash
mkdir model
tar -xvf resnet50-v2-7-tvm.tar -C model
ls model
```

解压后有三个文件：

* ```mod.so``` 是可被 TVM runtime 加载的模型，表示为 C++ 库。
* ```mod.json```  是 TVM Relay 计算图的文本表示。
* ```mod.params``` 是包含预训练模型参数的文件。

模块可由应用程序直接加载，而模型可通过 TVM runtime API 运行。

:::note 定义正确的 target
指定正确的 target（选项 ```--target```）可大大提升编译模块的性能，因为可利用 target 上可用的硬件功能。参阅 针对 x86 CPU 自动调优卷积网络 获取更多信息。建议确定好使用的 CPU 型号以及可选功能，然后适当地设置 target。
:::

## 使用 TVMC 运行来自编译模块的模型

将模型编译到模块后，可用 TVM runtime 对其进行预测。 TVMC 具有内置的 TVM runtime，允许运行已编译的 TVM 模型。要用 TVMC 运行模型并预测，需要：

* 刚生成的编译模块。
* 用来预测的模型的有效输入。

模型的张量 shape、格式和数据类型各不相同。因此，大多数模型都需要预处理和后处理，确保输入有效，并能够解释输出。 TVMC 采用了 NumPy 的 ```.npz``` 格式的输入和输出，可很好地支持将多个数组序列化到一个文件中。

本教程中的图像输入使用的是一张猫的图像，你也可以根据喜好选择其他图像。

![Cat](https://s3.amazonaws.com/model-server/inputs/kitten.jpg)

### 输入预处理

ResNet-50 v2 模型的输入应该是 ImageNet 格式。下面是 ResNet-50 v2 预处理图像的脚本示例。

首先用 ```pip3 install --user pillow``` 下载 Python 图像库，以满足脚本运行对图像库的依赖。

``` python
#!python ./preprocess.py
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np

img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# 重设大小为 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# ONNX 需要 NCHW 输入, 因此对数组进行转换
img_data = np.transpose(img_data, (2, 0, 1))

# 根据 ImageNet 进行标准化
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_stddev = np.array([0.229, 0.224, 0.225])
norm_img_data = np.zeros(img_data.shape).astype("float32")
for i in range(img_data.shape[0]):
      norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]

# 添加 batch 维度
img_data = np.expand_dims(norm_img_data, axis=0)

# 保存为 .npz (输出 imagenet_cat.npz)
np.savez("imagenet_cat", data=img_data)
```

### 运行编译模块

有了模型和输入数据，接下来运行 TVMC 进行预测：

``` bash
tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz \
resnet50-v2-7-tvm.tar
```

```.tar```  模型文件中包括一个 C++ 库、对 Relay 模型的描述文件，以及模型的参数文件。 TVMC 包括 TVM runtime（可加载模型，并对输入进行预测）。运行以上命令，TVMC 会输出一个新文件 ```predictions.npz```，其中包含 NumPy 格式的模型输出张量。

在此示例中，用于编译模型的和运行模型的是同一台机器。某些情况下，可能会用 RPC Tracker 来远程运行它。查看 ```tvmc run --help``` 来了解有关这些选项的更多信息。

### 输出后处理

如前所述，每个模型提供输出张量的方式都不一样。

本示例中，我们需要用专为该模型提供的查找表，运行一些后处理 (post-processing)，从而使得 ResNet-50 v2 的输出形式更具有可读性。

下面的脚本是一个后处理示例，它从编译模块的输出中提取标签：

``` python
#!python ./postprocess.py
import os.path
import numpy as np

from scipy.special import softmax

from tvm.contrib.download import download_testdata

# 下载标签列表
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

output_file = "predictions.npz"

# 打开并读入输出张量
if os.path.exists(output_file):
    with np.load(output_file) as data:
        scores = softmax(data["output_0"])
        scores = np.squeeze(scores)
        ranks = np.argsort(scores)[::-1]

        for rank in ranks[0:5]:
            print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
```

这个脚本的运行输出如下：

``` bash
python postprocess.py
# class='n02123045 tabby, tabby cat' with probability=0.610553
# class='n02123159 tiger cat' with probability=0.367179
# class='n02124075 Egyptian cat' with probability=0.019365
# class='n02129604 tiger, Panthera tigris' with probability=0.001273
# class='n04040759 radiator' with probability=0.000261
```

用其他图像替换上述猫的图像，看看 ResNet 模型做了什么样的预测。

## 自动调优 ResNet 模型

以前的模型被编译到 TVM runtime 上运行，因此不包含特定于平台的优化。本节将介绍如何用 TVMC，针对工作平台构建优化模型。

用编译的模块推理，有时可能无法获得预期的性能。在这种情况下，可用自动调优器更好地配置模型，从而提高性能。 TVM 中的调优是指，在给定 target 上优化模型，使其运行得更快。与训练或微调不同，它不会影响模型的准确性，而只会影响 runtime 性能。作为调优过程的一部分，TVM 实现并运行许多不同算子的变体，以查看哪个性能最佳。这些运行的结果存储在调优记录文件（tune 命令的最终输出）中。

调优最少要包含：

* 运行此模型的目标设备的平台要求
* 存储调优记录的输出文件的路径
* 要调优的模型的路径。

下面的示例演示了其工作流程：

``` bash
# 默认搜索算法需要 xgboost，有关调优搜索算法的详细信息，参见下文
pip install xgboost

tvmc tune \
--target "llvm" \
--output resnet50-v2-7-autotuner_records.json \
resnet50-v2-7.onnx
```

此例中，为 ```--target``` 标志指定更具体的 target 时，会得到更好的结果。例如，在 Intel i7 处理器上，可用 ```--target llvm -mcpu=skylake```。这个调优示例把 LLVM 作为指定架构的编译器，在 CPU 上进行本地调优。

TVMC 针对模型的参数空间进行搜索，为算子尝试不同的配置，然后选择平台上运行最快的配置。虽然这是基于 CPU 和模型操作的引导式搜索，但仍需要几个小时才能完成搜索。搜索的输出将保存到 ```resnet50-v2-7-autotuner_records.json``` 文件中，该文件之后会用于编译优化模型。

:::note 定义调优搜索算法
这个搜索算法默认用 ```XGBoost Grid``` 算法进行引导。根据模型复杂度和可用时间，可选择不同的算法。完整列表可查看 ```tvmc tune --help```。
:::

对于消费级的 Skylake CPU，输出如下：

``` bash
tvmc tune \
--target "llvm -mcpu=broadwell" \
--output resnet50-v2-7-autotuner_records.json \
resnet50-v2-7.onnx
# [Task  1/24]  Current/Best:    9.65/  23.16 GFLOPS | Progress: (60/1000) | 130.74 s Done.
# [Task  1/24]  Current/Best:    3.56/  23.16 GFLOPS | Progress: (192/1000) | 381.32 s Done.
# [Task  2/24]  Current/Best:   13.13/  58.61 GFLOPS | Progress: (960/1000) | 1190.59 s Done.
# [Task  3/24]  Current/Best:   31.93/  59.52 GFLOPS | Progress: (800/1000) | 727.85 s Done.
# [Task  4/24]  Current/Best:   16.42/  57.80 GFLOPS | Progress: (960/1000) | 559.74 s Done.
# [Task  5/24]  Current/Best:   12.42/  57.92 GFLOPS | Progress: (800/1000) | 766.63 s Done.
# [Task  6/24]  Current/Best:   20.66/  59.25 GFLOPS | Progress: (1000/1000) | 673.61 s Done.
# [Task  7/24]  Current/Best:   15.48/  59.60 GFLOPS | Progress: (1000/1000) | 953.04 s Done.
# [Task  8/24]  Current/Best:   31.97/  59.33 GFLOPS | Progress: (972/1000) | 559.57 s Done.
# [Task  9/24]  Current/Best:   34.14/  60.09 GFLOPS | Progress: (1000/1000) | 479.32 s Done.
# [Task 10/24]  Current/Best:   12.53/  58.97 GFLOPS | Progress: (972/1000) | 642.34 s Done.
# [Task 11/24]  Current/Best:   30.94/  58.47 GFLOPS | Progress: (1000/1000) | 648.26 s Done.
# [Task 12/24]  Current/Best:   23.66/  58.63 GFLOPS | Progress: (1000/1000) | 851.59 s Done.
# [Task 13/24]  Current/Best:   25.44/  59.76 GFLOPS | Progress: (1000/1000) | 534.58 s Done.
# [Task 14/24]  Current/Best:   26.83/  58.51 GFLOPS | Progress: (1000/1000) | 491.67 s Done.
# [Task 15/24]  Current/Best:   33.64/  58.55 GFLOPS | Progress: (1000/1000) | 529.85 s Done.
# [Task 16/24]  Current/Best:   14.93/  57.94 GFLOPS | Progress: (1000/1000) | 645.55 s Done.
# [Task 17/24]  Current/Best:   28.70/  58.19 GFLOPS | Progress: (1000/1000) | 756.88 s Done.
# [Task 18/24]  Current/Best:   19.01/  60.43 GFLOPS | Progress: (980/1000) | 514.69 s Done.
# [Task 19/24]  Current/Best:   14.61/  57.30 GFLOPS | Progress: (1000/1000) | 614.44 s Done.
# [Task 20/24]  Current/Best:   10.47/  57.68 GFLOPS | Progress: (980/1000) | 479.80 s Done.
# [Task 21/24]  Current/Best:   34.37/  58.28 GFLOPS | Progress: (308/1000) | 225.37 s Done.
# [Task 22/24]  Current/Best:   15.75/  57.71 GFLOPS | Progress: (1000/1000) | 1024.05 s Done.
# [Task 23/24]  Current/Best:   23.23/  58.92 GFLOPS | Progress: (1000/1000) | 999.34 s Done.
# [Task 24/24]  Current/Best:   17.27/  55.25 GFLOPS | Progress: (1000/1000) | 1428.74 s Done.
```

调优 session 需要很长时间，因此 ```tvmc tune``` 提供了许多选项来自定义调优过程，包括重复次数（例如 ```--repeat``` 和 ```--number```）、要用的调优算法等。查看 ```tvmc tune --help``` 了解更多信息。

## 使用调优数据编译优化模型

从上述调优过程的输出文件 ```resnet50-v2-7-autotuner_records.json`` 可获取调优记录。该文件可用来：

* 作为进一步调优的输入（通过 ```tvmc tune --tuning-records```）
* 作为编译器的输入

执行 ```tvmc compile --tuning-records``` 命令让编译器利用这个结果为指定 target 上的模型生成高性能代码。查看 ```tvmc compile --help``` 来获取更多信息。

模型的调优数据收集到后，可用优化的算子重新编译模型来加快计算速度。

``` bash
tvmc compile \
--target "llvm" \
--tuning-records resnet50-v2-7-autotuner_records.json  \
--output resnet50-v2-7-tvm_autotuned.tar \
resnet50-v2-7.onnx
```

验证优化模型是否运行并产生相同结果：

``` bash
tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz \
resnet50-v2-7-tvm_autotuned.tar

python postprocess.py
```

验证预测值是否相同：

``` bash
# class='n02123045 tabby, tabby cat' with probability=0.610550
# class='n02123159 tiger cat' with probability=0.367181
# class='n02124075 Egyptian cat' with probability=0.019365
# class='n02129604 tiger, Panthera tigris' with probability=0.001273
# class='n04040759 radiator' with probability=0.000261
```

## 比较调优和未调优的模型

TVMC 提供了模型之间的基本性能评估工具。可指定重复次数，也可指定 TVMC 报告模型的运行时间（独立于 runtime 启动）。可大致了解调优对模型性能的提升程度。例如，对 Intel i7 系统进行测试时，调优后的模型比未调优的模型运行速度快 47%：

``` bash
tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz  \
--print-time \
--repeat 100 \
resnet50-v2-7-tvm_autotuned.tar

# Execution time summary:
# mean (ms)   max (ms)    min (ms)    std (ms)
#     92.19     115.73       89.85        3.15

tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz  \
--print-time \
--repeat 100 \
resnet50-v2-7-tvm.tar

# Execution time summary:
# mean (ms)   max (ms)    min (ms)    std (ms)
#    193.32     219.97      185.04        7.11
```

## 写在最后

本教程介绍了 TVMC（ TVM 的命令行驱动程序），演示了如何编译、运行和调优模型，还讨论了对输入和输出进行预处理和后处理的必要性。调优后，演示如何比较未优化和优化模型的性能。

本文档展示了一个在本地使用 ResNet-50 v2 的简单示例。然而，TVMC 支持更多功能，包括交叉编译、远程执行和分析/基准测试。

用 ```tvmc --help``` 命令查看其他可用选项。

下个教程 ```Compiling and Optimizing a Model with the Python Interface``` 将介绍用 Python 接口的相同编译和优化步骤。

[下载 Python 源代码：tvmc_command_line_driver.py](https://tvm.apache.org/docs/_downloads/233ceda3a682ae5df93b4ce0bcfbf870/tvmc_command_line_driver.py)

[下载 Jupyter Notebook：tvmc_command_line_driver.ipynb](https://tvm.apache.org/docs/_downloads/efe0b02e219b28e0bd85fbdda35ba8ac/tvmc_command_line_driver.ipynb)
