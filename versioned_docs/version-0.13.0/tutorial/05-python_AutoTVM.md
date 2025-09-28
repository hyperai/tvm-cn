---
title: 使用 Python 接口（AutoTVM）编译和优化模型
---

# 使用 Python 接口（AutoTVM）编译和优化模型

:::note
单击 [此处](https://tvm.apache.org/docs/tutorial/autotvm_relay_x86.html#sphx-glr-download-tutorial-autotvm-relay-x86-py) 下载完整的示例代码
:::

**作者**：[Chris Hoge](https://github.com/hogepodge)

[TVMC 教程](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver) 介绍了如何用 TVM 的命令行界面（TVMC）编译、运行和调优预训练的模型 ResNet-50 v2。TVM 不仅是一个命令行工具，也是一个具有多种不同语言的 API 优化框架，极大方便了机器学习模型的使用。

本节内容将介绍与使用 TVMC 相同的基础知识，不同的是这节内容是用 Python API 来实现的。完成本节后学习后，我们将用 TVM 的 Python API 实现以下任务：

* 为 TVM runtime 编译预训练的 ResNet-50 v2 模型。
* 用编译的模型预测真实图像，并解释输出和模型性能。
* 用 TVM 对 CPU 上建模的模型进行调优。
* 用 TVM 收集的调优数据重新编译优化模型。
* 用优化模型预测图像，并比较输出和模型性能。

本节目标是概述 TVM 的功能，以及如何通过 Python API 使用它们。

TVM 是一个深度学习编译器框架，有许多不同的模块可用于处理深度学习模型和算子。本教程将介绍如何用 Python API 加载、编译和优化模型。

首先导入一些依赖，包括用于加载和转换模型的 `onnx`、用于下载测试数据的辅助实用程序、用于处理图像数据的 Python 图像库、用于图像数据预处理和后处理的 `numpy`、TVM Relay 框架和 TVM 图形处理器。

``` python
import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
```

## 下载和加载 ONNX 模型

本教程中，我们会用到 ResNet-50 v2。ResNet-50 是一个深度为 50 层的卷积神经网络，适用于图像分类任务。我们即将用到的模型已经在超过 100 万张、具有 1000 种不同分类的图像上进行了预训练。该神经网络的输入图像大小为 224x224。推荐下载 [Netron](https://netron.app/)（免费的 ML 模型查看器 ）了解更多 ResNet-50 模型的结构信息。

TVM 提供帮助库来下载预训练模型。通过提供模型 URL、文件名和模型类型，TVM 可下载模型并将其保存到磁盘。可用 ONNX runtime 将 ONNX 模型实例加载到内存。

:::note 使用其他模型格式
TVM 支持许多流行的模型格式。可在 TVM 文档的 [编译深度学习模型](https://tvm.apache.org/docs/how_to/compile_models/index.html#tutorial-frontend) 部分找到支持的列表。
:::

``` python
model_url = (
    "https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/"
    "resnet50-v2-7.onnx"
)

model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)

# 为 numpy 的 RNG 设置 seed，得到一致的结果
np.random.seed(0)
```

## 下载、预处理和加载测试图像

模型的张量 shape、格式和数据类型各不相同。因此，大多数模型都需要一些预处理和后处理，以确保输入有效，并能解释输出。 TVMC 采用了 NumPy 的 `.npz` 格式的输入和输出数据。

本教程中的图像输入使用的是一张猫的图像，你也可以根据喜好选择其他图像。

 ![https://s3.amazonaws.com/model-server/inputs/kitten.jpg](https://s3.amazonaws.com/model-server/inputs/kitten.jpg)

下载图像数据，然后将其转换为 numpy 数组作为模型的输入。

``` python
img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# 重设大小为 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# 输入图像是 HWC 布局，而 ONNX 需要 CHW 输入，所以转换数组
img_data = np.transpose(img_data, (2, 0, 1))

# 根据 ImageNet 输入规范进行归一化
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# 添加 batch 维度，期望 4 维输入：NCHW。
img_data = np.expand_dims(norm_img_data, axis=0)
```

## 使用 Relay 编译模型

下一步是编译 ResNet 模型，首先用 *from_onnx* 导入器，将模型导入到 Relay 中。然后，用标准优化，将模型构建到 TVM 库中，最后从库中创建一个 TVM 计算图 runtime 模块。

``` python
target = "llvm"
```

:::note 定义正确的 target
指定正确的 target（选项 `--target`）可大大提升编译模块的性能，因为可利用 target 上可用的硬件功能。参阅 [针对 x86 CPU 自动调优卷积网络](https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_x86.html#tune-relay-x86) 获取更多信息。建议确定好使用的 CPU 型号以及可选功能，然后适当地设置 target。例如，对于某些处理器，可用 `target = "llvm -mcpu=skylake"`；对于具有 AVX-512 向量指令集的处理器，可用 `target = "llvm -mcpu=skylake-avx512"`。
:::

``` python
# 输入名称可能因模型类型而异
# 可用 Netron 工具检查输入名称
input_name = "data"
shape_dict = {input_name: img_data.shape}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
```

输出结果：

``` plain
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 在 TVM Runtime 执行

编译好模型后，就可用 TVM runtime 对其进行预测。要用 TVM 运行模型并进行预测，需要：

* 刚生成的编译模型。
* 用来预测的模型的有效输入。

``` python
dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
```

## 收集基本性能数据

收集与未优化模型相关的基本性能数据，然后将其与调优后的模型进行比较。为了解释 CPU 噪声，在多个 batch 中多次重复计算，然后收集关于均值、中值和标准差的基础统计数据。

``` python
import timeit

timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)
```

输出结果：

``` plain
{'mean': 495.13895513002353, 'median': 494.6680843500417, 'std': 1.3081147373726523}
```

## 输出后处理

如前所述，每个模型提供输出张量的方式都不一样。

本示例中，我们需要用专为该模型提供的查找表，运行一些后处理（post-processing），从而使得 ResNet-50 v2 的输出形式更具有可读性。

``` python
from scipy.special import softmax

# 下载标签列表
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

# 打开输出文件并读取输出张量
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
```

输出结果：

``` plain
class='n02123045 tabby, tabby cat' with probability=0.621103
class='n02123159 tiger cat' with probability=0.356379
class='n02124075 Egyptian cat' with probability=0.019712
class='n02129604 tiger, Panthera tigris' with probability=0.001215
class='n04040759 radiator' with probability=0.000262
```

预期输出如下：

``` plain
# class='n02123045 tabby, tabby cat' with probability=0.610553
# class='n02123159 tiger cat' with probability=0.367179
# class='n02124075 Egyptian cat' with probability=0.019365
# class='n02129604 tiger, Panthera tigris' with probability=0.001273
# class='n04040759 radiator' with probability=0.000261
```

## 调优模型

以前的模型被编译到 TVM runtime 上运行，因此不包含特定于平台的优化。本节将介绍如何用 TVMC，针对工作平台构建优化模型。

用编译的模块推理，有时可能无法获得预期的性能。在这种情况下，可用自动调优器更好地配置模型，从而提高性能。 TVM 中的调优是指，在给定 target 上优化模型，使其运行得更快。与训练或微调不同，它不会影响模型的准确性，而只会影响 runtime 性能。作为调优过程的一部分，TVM 实现并运行许多不同算子的变体，以查看哪个性能最佳。这些运行的结果存储在调优记录文件中。

最简单的形式中，调优需要：

* 运行此模型的设备的规格
* 存储调优记录的输出文件的路径
* 要调优的模型的路径。

``` python
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
```

设置部分基本参数，运行由一组特定参数生成的编译代码并测试其性能。`number` 指定将要测试的不同配置的数量，而 `repeat` 指定将对每个配置进行多少次测试。 `min_repeat_ms` 指定运行配置测试需要多长时间，如果重复次数低于此时间，则增加其值，在 GPU 上进行精确调优时此选项是必需的，在 CPU 调优则不是必需的，将此值设置为 0表示禁用，`timeout` 指明每个测试配置运行训练代码的时间上限。

``` python
number = 10
repeat = 1
min_repeat_ms = 0  # 调优 CPU 时设置为 0
timeout = 10  # 秒

# 创建 TVM 运行器
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)
```

创建简单结构来保存调优选项。使用 XGBoost 算法来指导搜索。如果要在投产的项目中应用，则需要将试验次数设置为大于此处的 20。对于 CPU 推荐 1500，对于 GPU 推荐 3000-4000。所需的试验次数可能取决于特定的模型和处理器，要找到调优时间和模型优化之间的最佳平衡，得花一些时间评估一系列值的性能。

运行调优需要大量时间，所以这里将试验次数设置为 10，但不推荐使用这么小的值。`early_stopping` 参数是使得搜索提前停止的试验最小值。measure option 决定了构建试用代码并运行的位置，本示例用的是刚创建的 `LocalRunner` 和 `LocalBuilder`。`Tuning_records` 选项指定将调优数据写入的哪个文件中。

``` python
tuning_option = {
    "tuner": "xgb",
    "trials": 20,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "resnet-50-v2-autotuning.json",
}
```

:::note 定义调优搜索算法
此搜索默认情况下使用 XGBoost Grid 算法进行引导。根据模型复杂性和可用时长可选择不同的算法。
:::

:::note 设置调优参数
为节省时间将试验次数和提前停止次数设置为 20 和 100，数值设置越大，性能越好，所需时间也越长。收敛所需的试验次数根据模型和目标平台的不同而变化。
:::

``` python
# 首先从 onnx 模型中提取任务
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# 按顺序调优提取的任务
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    
    # choose tuner
    tuner = "xgb"

    # create tuner
    if tuner == "xgb":
        tuner_obj = XGBTuner(task, loss_type="reg")
    elif tuner == "xgb_knob":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
    elif tuner == "xgb_itervar":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
    elif tuner == "xgb_curve":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
    elif tuner == "xgb_rank":
        tuner_obj = XGBTuner(task, loss_type="rank")
    elif tuner == "xgb_rank_knob":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
    elif tuner == "xgb_rank_itervar":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
    elif tuner == "xgb_rank_curve":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
    elif tuner == "xgb_rank_binary":
        tuner_obj = XGBTuner(task, loss_type="rank-binary")
    elif tuner == "xgb_rank_binary_knob":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
    elif tuner == "xgb_rank_binary_itervar":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
    elif tuner == "xgb_rank_binary_curve":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
    elif tuner == "ga":
        tuner_obj = GATuner(task, pop_size=50)
    elif tuner == "random":
        tuner_obj = RandomTuner(task)
    elif tuner == "gridsearch":
        tuner_obj = GridSearchTuner(task)
    else:
        raise ValueError("Invalid tuner: " + tuner)

    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )
```

输出结果：

``` plain
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "

[Task  1/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task  1/25]  Current/Best:   17.48/  17.48 GFLOPS | Progress: (4/20) | 6.23 s
[Task  1/25]  Current/Best:    6.16/  17.48 GFLOPS | Progress: (8/20) | 9.24 s
[Task  1/25]  Current/Best:   11.54/  22.69 GFLOPS | Progress: (12/20) | 11.75 s
[Task  1/25]  Current/Best:   16.79/  22.69 GFLOPS | Progress: (16/20) | 13.44 s
[Task  1/25]  Current/Best:   11.62/  23.90 GFLOPS | Progress: (20/20) | 15.17 s Done.

[Task  2/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task  2/25]  Current/Best:   12.28/  12.93 GFLOPS | Progress: (4/20) | 3.93 s
[Task  2/25]  Current/Best:   14.28/  17.39 GFLOPS | Progress: (8/20) | 5.27 s
[Task  2/25]  Current/Best:   20.53/  20.53 GFLOPS | Progress: (12/20) | 6.61 s
[Task  2/25]  Current/Best:   11.88/  20.53 GFLOPS | Progress: (16/20) | 7.87 s
[Task  2/25]  Current/Best:   19.41/  20.53 GFLOPS | Progress: (20/20) | 9.47 s Done.

[Task  3/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task  3/25]  Current/Best:    1.64/  10.51 GFLOPS | Progress: (4/20) | 5.83 s
[Task  3/25]  Current/Best:   15.62/  17.07 GFLOPS | Progress: (8/20) | 7.73 s
[Task  3/25]  Current/Best:   15.07/  17.07 GFLOPS | Progress: (12/20) | 9.42 s
[Task  3/25]  Current/Best:    7.27/  24.05 GFLOPS | Progress: (16/20) | 11.36 s
[Task  3/25]  Current/Best:   12.61/  24.05 GFLOPS | Progress: (20/20) | 15.89 s Done.

[Task  4/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task  4/25]  Current/Best:    9.66/  20.74 GFLOPS | Progress: (4/20) | 2.38 s
[Task  4/25]  Current/Best:    6.87/  20.74 GFLOPS | Progress: (8/20) | 7.09 s
[Task  4/25]  Current/Best:   21.79/  21.79 GFLOPS | Progress: (12/20) | 11.94 s
[Task  4/25]  Current/Best:   17.19/  21.79 GFLOPS | Progress: (16/20) | 14.31 s
[Task  4/25]  Current/Best:   13.38/  21.79 GFLOPS | Progress: (20/20) | 16.40 s Done.

[Task  5/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task  5/25]  Current/Best:    9.68/  10.34 GFLOPS | Progress: (4/20) | 2.58 s
[Task  5/25]  Current/Best:   11.96/  13.13 GFLOPS | Progress: (8/20) | 4.62 s
[Task  5/25]  Current/Best:   11.49/  18.26 GFLOPS | Progress: (12/20) | 7.64 s
[Task  5/25]  Current/Best:   11.79/  22.76 GFLOPS | Progress: (16/20) | 9.05 s
[Task  5/25]  Current/Best:   11.98/  22.76 GFLOPS | Progress: (20/20) | 10.95 s Done.

[Task  6/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task  6/25]  Current/Best:   12.28/  20.75 GFLOPS | Progress: (4/20) | 4.08 s
[Task  6/25]  Current/Best:   19.20/  20.75 GFLOPS | Progress: (8/20) | 5.83 s
[Task  6/25]  Current/Best:   13.29/  20.75 GFLOPS | Progress: (12/20) | 7.77 s
[Task  6/25]  Current/Best:   20.29/  20.75 GFLOPS | Progress: (16/20) | 10.00 s
[Task  6/25]  Current/Best:    3.74/  20.75 GFLOPS | Progress: (20/20) | 12.53 s Done.

[Task  7/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task  7/25]  Current/Best:   11.29/  12.56 GFLOPS | Progress: (4/20) | 3.61 s
[Task  7/25]  Current/Best:   20.39/  21.29 GFLOPS | Progress: (8/20) | 5.11 s
[Task  7/25]  Current/Best:   16.12/  21.29 GFLOPS | Progress: (12/20) | 7.01 s
[Task  7/25]  Current/Best:   12.36/  21.29 GFLOPS | Progress: (16/20) | 9.05 s
[Task  7/25]  Current/Best:    6.38/  21.95 GFLOPS | Progress: (20/20) | 11.49 s Done.

[Task  8/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task  8/25]  Current/Best:   10.24/  14.44 GFLOPS | Progress: (4/20) | 2.91 s
[Task  8/25]  Current/Best:    9.52/  14.44 GFLOPS | Progress: (8/20) | 8.03 s
[Task  8/25]  Current/Best:   12.94/  14.44 GFLOPS | Progress: (12/20) | 14.49 s
[Task  8/25]  Current/Best:   18.92/  18.92 GFLOPS | Progress: (16/20) | 16.58 s
[Task  8/25]  Current/Best:   20.19/  20.19 GFLOPS | Progress: (20/20) | 23.57 s Done.

[Task  9/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task  9/25]  Current/Best:   14.38/  15.82 GFLOPS | Progress: (4/20) | 11.96 s
[Task  9/25]  Current/Best:   23.36/  23.36 GFLOPS | Progress: (8/20) | 13.71 s
[Task  9/25]  Current/Best:    8.34/  23.36 GFLOPS | Progress: (12/20) | 16.27 s
[Task  9/25]  Current/Best:   18.17/  23.36 GFLOPS | Progress: (16/20) | 19.12 s
[Task  9/25]  Current/Best:    9.22/  23.36 GFLOPS | Progress: (20/20) | 27.69 s
[Task 10/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 10/25]  Current/Best:   18.38/  18.38 GFLOPS | Progress: (4/20) | 2.56 s
[Task 10/25]  Current/Best:   15.58/  18.38 GFLOPS | Progress: (8/20) | 4.18 s
[Task 10/25]  Current/Best:   13.30/  19.16 GFLOPS | Progress: (12/20) | 5.71 s
[Task 10/25]  Current/Best:   19.30/  20.58 GFLOPS | Progress: (16/20) | 6.82 s
[Task 10/25]  Current/Best:    8.86/  20.58 GFLOPS | Progress: (20/20) | 8.33 s Done.

[Task 11/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 11/25]  Current/Best:   11.06/  18.40 GFLOPS | Progress: (4/20) | 3.40 s
[Task 11/25]  Current/Best:   17.14/  18.40 GFLOPS | Progress: (8/20) | 6.19 s
[Task 11/25]  Current/Best:   16.38/  18.40 GFLOPS | Progress: (12/20) | 8.26 s
[Task 11/25]  Current/Best:   13.59/  21.40 GFLOPS | Progress: (16/20) | 11.11 s
[Task 11/25]  Current/Best:   19.64/  21.83 GFLOPS | Progress: (20/20) | 13.19 s Done.

[Task 12/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 12/25]  Current/Best:    7.87/  18.27 GFLOPS | Progress: (4/20) | 5.68 s
[Task 12/25]  Current/Best:    5.29/  18.27 GFLOPS | Progress: (8/20) | 9.61 s
[Task 12/25]  Current/Best:   18.95/  19.15 GFLOPS | Progress: (12/20) | 11.57 s
[Task 12/25]  Current/Best:   15.07/  19.15 GFLOPS | Progress: (16/20) | 14.51 s
[Task 12/25]  Current/Best:   15.33/  19.15 GFLOPS | Progress: (20/20) | 16.46 s Done.

[Task 13/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 13/25]  Current/Best:    8.75/  17.46 GFLOPS | Progress: (4/20) | 3.75 s
[Task 13/25]  Current/Best:   15.84/  20.93 GFLOPS | Progress: (8/20) | 6.37 s
[Task 13/25]  Current/Best:   19.65/  21.81 GFLOPS | Progress: (12/20) | 9.48 s
[Task 13/25]  Current/Best:   12.32/  21.81 GFLOPS | Progress: (16/20) | 12.86 s
[Task 13/25]  Current/Best:   18.88/  21.81 GFLOPS | Progress: (20/20) | 15.15 s Done.

[Task 14/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 14/25]  Current/Best:   13.62/  13.62 GFLOPS | Progress: (4/20) | 3.39 s
[Task 14/25]  Current/Best:    6.18/  13.62 GFLOPS | Progress: (8/20) | 5.59 s
[Task 14/25]  Current/Best:   20.88/  20.88 GFLOPS | Progress: (12/20) | 8.29 s
[Task 14/25]  Current/Best:   16.63/  20.88 GFLOPS | Progress: (16/20) | 9.99 s Done.

[Task 14/25]  Current/Best:   17.36/  20.88 GFLOPS | Progress: (20/20) | 11.79 s
[Task 15/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 15/25]  Current/Best:   16.39/  17.82 GFLOPS | Progress: (4/20) | 2.72 s
[Task 15/25]  Current/Best:   14.58/  18.33 GFLOPS | Progress: (8/20) | 4.05 s
[Task 15/25]  Current/Best:   10.47/  22.24 GFLOPS | Progress: (12/20) | 6.26 s
[Task 15/25]  Current/Best:   20.60/  22.24 GFLOPS | Progress: (16/20) | 9.86 s
[Task 15/25]  Current/Best:    9.80/  22.24 GFLOPS | Progress: (20/20) | 10.87 s
[Task 16/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 16/25]  Current/Best:   20.68/  20.68 GFLOPS | Progress: (4/20) | 3.01 s
[Task 16/25]  Current/Best:    3.07/  20.68 GFLOPS | Progress: (8/20) | 4.61 s
[Task 16/25]  Current/Best:   19.68/  20.68 GFLOPS | Progress: (12/20) | 5.82 s
[Task 16/25]  Current/Best:   17.95/  20.68 GFLOPS | Progress: (16/20) | 7.18 s
[Task 16/25]  Current/Best:   10.15/  22.12 GFLOPS | Progress: (20/20) | 9.33 s Done.

[Task 17/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 17/25]  Current/Best:   13.51/  18.93 GFLOPS | Progress: (4/20) | 4.79 s
[Task 17/25]  Current/Best:   14.55/  23.37 GFLOPS | Progress: (8/20) | 7.56 s
[Task 17/25]  Current/Best:   16.89/  23.37 GFLOPS | Progress: (12/20) | 9.61 s
[Task 17/25]  Current/Best:   16.77/  23.37 GFLOPS | Progress: (16/20) | 11.81 s
[Task 17/25]  Current/Best:   10.14/  23.37 GFLOPS | Progress: (20/20) | 13.94 s Done.

[Task 18/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 18/25]  Current/Best:   11.52/  18.15 GFLOPS | Progress: (4/20) | 3.78 s
[Task 18/25]  Current/Best:   10.71/  19.51 GFLOPS | Progress: (8/20) | 7.45 s
[Task 18/25]  Current/Best:   19.44/  19.51 GFLOPS | Progress: (12/20) | 9.39 s
[Task 18/25]  Current/Best:   10.08/  19.51 GFLOPS | Progress: (16/20) | 13.17 s
[Task 18/25]  Current/Best:   20.94/  20.94 GFLOPS | Progress: (20/20) | 14.66 s Done.

[Task 19/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 19/25]  Current/Best:    7.21/  20.46 GFLOPS | Progress: (4/20) | 6.09 s
[Task 19/25]  Current/Best:    2.63/  20.46 GFLOPS | Progress: (8/20) | 9.41 s
[Task 19/25]  Current/Best:   19.25/  21.00 GFLOPS | Progress: (12/20) | 12.32 s
[Task 19/25]  Current/Best:   15.32/  22.01 GFLOPS | Progress: (16/20) | 15.25 s
[Task 19/25]  Current/Best:    2.72/  23.38 GFLOPS | Progress: (20/20) | 18.04 s Done.

[Task 20/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 20/25]  Current/Best:    8.70/  15.13 GFLOPS | Progress: (4/20) | 3.34 s Done.
 Done.

[Task 20/25]  Current/Best:   10.07/  15.13 GFLOPS | Progress: (8/20) | 6.70 s
[Task 20/25]  Current/Best:    2.35/  16.65 GFLOPS | Progress: (12/20) | 10.62 s
[Task 20/25]  Current/Best:   12.51/  16.65 GFLOPS | Progress: (16/20) | 14.49 s
[Task 20/25]  Current/Best:   13.29/  22.42 GFLOPS | Progress: (20/20) | 16.56 s
[Task 21/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 21/25]  Current/Best:    6.47/  18.00 GFLOPS | Progress: (4/20) | 3.24 s
[Task 21/25]  Current/Best:   14.70/  18.00 GFLOPS | Progress: (8/20) | 4.84 s
[Task 21/25]  Current/Best:    1.63/  18.00 GFLOPS | Progress: (12/20) | 6.96 s
[Task 21/25]  Current/Best:   18.46/  18.46 GFLOPS | Progress: (16/20) | 10.46 s
[Task 21/25]  Current/Best:    4.52/  18.46 GFLOPS | Progress: (20/20) | 17.74 s
[Task 22/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 22/25]  Current/Best:    2.74/  17.31 GFLOPS | Progress: (4/20) | 2.66 s
[Task 22/25]  Current/Best:    9.18/  22.31 GFLOPS | Progress: (8/20) | 4.69 s
[Task 22/25]  Current/Best:   19.91/  22.31 GFLOPS | Progress: (12/20) | 7.04 s
[Task 22/25]  Current/Best:   15.37/  22.31 GFLOPS | Progress: (16/20) | 9.12 s
[Task 22/25]  Current/Best:   14.47/  22.31 GFLOPS | Progress: (20/20) | 10.83 s Done.

[Task 23/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 23/25]  Current/Best:   17.75/  20.70 GFLOPS | Progress: (4/20) | 3.23 s
[Task 23/25]  Current/Best:   15.91/  20.70 GFLOPS | Progress: (8/20) | 6.67 s
[Task 23/25]  Current/Best:   21.19/  21.55 GFLOPS | Progress: (12/20) | 8.49 s
[Task 23/25]  Current/Best:    6.53/  21.55 GFLOPS | Progress: (16/20) | 15.43 s
[Task 23/25]  Current/Best:    7.96/  21.55 GFLOPS | Progress: (20/20) | 19.61 s Done.

[Task 24/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 24/25]  Current/Best:    8.44/   8.44 GFLOPS | Progress: (4/20) | 11.80 s
[Task 24/25]  Current/Best:    2.01/   8.44 GFLOPS | Progress: (8/20) | 22.82 s
[Task 24/25]  Current/Best:    4.44/   8.44 GFLOPS | Progress: (12/20) | 34.34 s Done.
 Done.

[Task 24/25]  Current/Best:    7.18/   8.73 GFLOPS | Progress: (16/20) | 40.14 s
[Task 24/25]  Current/Best:    3.29/   8.99 GFLOPS | Progress: (20/20) | 46.23 s Done.

[Task 25/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
[Task 25/25]  Current/Best:    1.56/   2.93 GFLOPS | Progress: (4/20) | 11.63 s
[Task 25/25]  Current/Best:    5.65/   7.64 GFLOPS | Progress: (8/20) | 22.93 s
[Task 25/25]  Current/Best:    5.95/   7.64 GFLOPS | Progress: (12/20) | 34.36 s
[Task 25/25]  Current/Best:    5.80/   9.36 GFLOPS | Progress: (16/20) | 36.05 s
[Task 25/25]  Current/Best:    2.94/   9.36 GFLOPS | Progress: (20/20) | 46.76 s
```

调优过程的输出如下所示：

``` plain
# [Task  1/24]  Current/Best:   10.71/  21.08 GFLOPS | Progress: (60/1000) | 111.77 s Done.
# [Task  1/24]  Current/Best:    9.32/  24.18 GFLOPS | Progress: (192/1000) | 365.02 s Done.
# [Task  2/24]  Current/Best:   22.39/ 177.59 GFLOPS | Progress: (960/1000) | 976.17 s Done.
# [Task  3/24]  Current/Best:   32.03/ 153.34 GFLOPS | Progress: (800/1000) | 776.84 s Done.
# [Task  4/24]  Current/Best:   11.96/ 156.49 GFLOPS | Progress: (960/1000) | 632.26 s Done.
# [Task  5/24]  Current/Best:   23.75/ 130.78 GFLOPS | Progress: (800/1000) | 739.29 s Done.
# [Task  6/24]  Current/Best:   38.29/ 198.31 GFLOPS | Progress: (1000/1000) | 624.51 s Done.
# [Task  7/24]  Current/Best:    4.31/ 210.78 GFLOPS | Progress: (1000/1000) | 701.03 s Done.
# [Task  8/24]  Current/Best:   50.25/ 185.35 GFLOPS | Progress: (972/1000) | 538.55 s Done.
# [Task  9/24]  Current/Best:   50.19/ 194.42 GFLOPS | Progress: (1000/1000) | 487.30 s Done.
# [Task 10/24]  Current/Best:   12.90/ 172.60 GFLOPS | Progress: (972/1000) | 607.32 s Done.
# [Task 11/24]  Current/Best:   62.71/ 203.46 GFLOPS | Progress: (1000/1000) | 581.92 s Done.
# [Task 12/24]  Current/Best:   36.79/ 224.71 GFLOPS | Progress: (1000/1000) | 675.13 s Done.
# [Task 13/24]  Current/Best:    7.76/ 219.72 GFLOPS | Progress: (1000/1000) | 519.06 s Done.
# [Task 14/24]  Current/Best:   12.26/ 202.42 GFLOPS | Progress: (1000/1000) | 514.30 s Done.
# [Task 15/24]  Current/Best:   31.59/ 197.61 GFLOPS | Progress: (1000/1000) | 558.54 s Done.
# [Task 16/24]  Current/Best:   31.63/ 206.08 GFLOPS | Progress: (1000/1000) | 708.36 s Done.
# [Task 17/24]  Current/Best:   41.18/ 204.45 GFLOPS | Progress: (1000/1000) | 736.08 s Done.
# [Task 18/24]  Current/Best:   15.85/ 222.38 GFLOPS | Progress: (980/1000) | 516.73 s Done.
# [Task 19/24]  Current/Best:   15.78/ 203.41 GFLOPS | Progress: (1000/1000) | 587.13 s Done.
# [Task 20/24]  Current/Best:   30.47/ 205.92 GFLOPS | Progress: (
```

## 使用调优数据编译优化模型

获取存储在 `resnet-50-v2-autotuning.json`（上述调优过程的输出文件）中的调优记录。编译器会用这个结果，为指定 target 上的模型生成高性能代码。

收集到模型的调优数据后，可用优化的算子重新编译模型来加快计算速度。

``` python
with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
```

输出结果：

``` plain
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

验证优化模型是否运行并产生相同的结果：

``` python
dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
```

输出结果：

``` plain
class='n02123045 tabby, tabby cat' with probability=0.621104
class='n02123159 tiger cat' with probability=0.356378
class='n02124075 Egyptian cat' with probability=0.019712
class='n02129604 tiger, Panthera tigris' with probability=0.001215
class='n04040759 radiator' with probability=0.000262
```

验证预测值是否相同：

``` plain
# class='n02123045 tabby, tabby cat' with probability=0.610550
# class='n02123159 tiger cat' with probability=0.367181
# class='n02124075 Egyptian cat' with probability=0.019365
# class='n02129604 tiger, Panthera tigris' with probability=0.001273
# class='n04040759 radiator' with probability=0.000261
```

## 比较调优和未调优的模型

收集与此优化模型相关的一些基本性能数据，并将其与未优化模型进行比较。根据底层硬件、迭代次数和其他因素，将优化模型和未优化模型比较时，可以看到性能的提升。

``` python
import timeit

timing_number = 10
timing_repeat = 10
optimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}



print("optimized: %s" % (optimized))
print("unoptimized: %s" % (unoptimized))
```

输出结果：

```plain
optimized: {'mean': 407.31687583000166, 'median': 407.3377107500164, 'std': 1.692177042688564}
unoptimized: {'mean': 495.13895513002353, 'median': 494.6680843500417, 'std': 1.3081147373726523}
```

## 写在最后

本教程通过一个简短示例，说明了如何用 TVM Python API 编译、运行和调优模型。还讨论了对输入和输出进行预处理和后处理的必要性。在调优过程之后，演示了如何比较未优化和优化模型的性能。

本文档展示了一个在本地使用 ResNet-50 v2 的简单示例。TVMC 还支持更多功能，包括交叉编译、远程执行和分析/基准测试等。

**脚本总运行时间：**（10 分 27.660 秒）

`下载 Python 源代码：autotvm_relay_x86.py`

`下载 Jupyter Notebook：autotvm_relay_x86.ipynb`
