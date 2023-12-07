---
title: 5.在 Arduino 上为 microTVM 训练视觉模型
---

# 5.在 Arduino 上为 microTVM 训练视觉模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_train.html#sphx-glr-download-how-to-work-with-microtvm-micro-train-py) 下载完整的示例代码
:::

**作者**：[Gavin Uberti](https://github.com/guberti)

本教程介绍如何训练 MobileNetV1 模型以适应嵌入式设备，以及如何使用 TVM 将这些模型部署到 Arduino。

## 背景简介

构建物联网设备时，通常想让它们能够**看到并理解**它们周围的世界。可以采取多种形式，但通常设备也会想知道**某种物体**是否在其视野中。

例如，安全摄像头可能会寻找**人**，因此它可以决定是否将视频保存到内存中。红绿灯可能会寻找**汽车**，这样它就可以判断哪个信号灯应该首先改变。或者森林相机可能会寻找**一种动物**，从而估计动物种群的数量。

为使这些设备价格合理，我们希望为这些设备配置一个低成本处理器，如 [nRF52840](https://www.nordicsemi.com/Products/nRF52840)（在 Mouser 上每个售价 5 美元）或 [RP2040](https://www.raspberrypi.com/products/rp2040/)（每个只需 1.45 美元）。

这些设备的内存非常小（\~250 KB RAM），这意味着传统的边缘 AI 视觉模型（如 MobileNet 或 EfficientNet）都不能够运行。本教程将展示如何修改这些模型以解决此问题。然后，使用 TVM 为 Arduino 编译和部署。

## 安装必要软件

本教程使用 TensorFlow（Google 创建的一个广泛使用的机器学习库）来训练模型。TensorFlow 是一个非常底层的库，因此要用 Keras 接口来从 TensorFlow 获取信息。还会用 TensorFlow Lite 量化模型，因为 TensorFlow 本身不支持这一点。

生成模型后，使用 TVM 对其进行编译和测试。为避免必须从源代码构建，需要安装 `tlcpack` - TVM 的社区构建工具，还要安装 `imagemagick` 和 `curl` 来预处理数据：

``` bash
%%bash
pip install -q tensorflow tflite
pip install -q tlcpack-nightly -f https://tlcpack.ai/wheels
apt-get -qq install imagemagick curl

# 为 Nano 33 BLE 安装 Arduino CLI 和库
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
/content/bin/arduino-cli core update-index
/content/bin/arduino-cli core install arduino:mbed_nano
```

### 使用 GPU

本教程演示如何训练神经网络，训练神经网络需要大量的计算能力，使用 GPU 训练速度会更快。若在 Google Colab 上查看本教程，可通过 **Runtime->Change runtime type** 并选择“GPU”作为硬件加速器来启用 GPU。若在本地运行，则可按照 [TensorFlow 指南](https://www.tensorflow.org/guide/gpu) 进行操作。

使用以下代码测试 GPU 是否安装：

``` python
import tensorflow as tf

if not tf.test.gpu_device_name():
    print("No GPU was detected!")
    print("Model training will take much longer (~30 minutes instead of ~5)")
else:
    print("GPU detected - you're good to go.")
```

输出结果：

``` bash
No GPU was detected!
Model training will take much longer (~30 minutes instead of ~5)
```

### 选择工作目录

选择工作目录，把图像数据集、训练好的模型和最终的  Arduino sketch 都放在此目录下，如果在 Google Colab 上运行，所有内容将保存在`/root`（又名 `~`）中，若在本地运行，可将其存储在其他地方。注意，此变量仅影响 Python 脚本 - 还必须调整 Bash 命令。

``` python
import os

FOLDER = "/root"
```

## 下载数据

卷积神经网络通过大量图像以及标签来学习，为获得这些图像，需要一个公开可用的数据集，这个数据集中要包含数千张各种各样的目标的图像，以及每张图像中内容的标签，还需要一堆**不是**汽车的图像，因为我们需要区分这两个类别。

本教程将创建一个模型，来检测图像中是否包含**汽车**，也可以用于检测其他目标物体！只需将下面的源 URL 更改为包含另一种目标图像的源 URL。

下载 [斯坦福汽车数据集](https://hyper.ai/datasets/5466)，该数据集包含 16,185 个彩色汽车图像。还需要不是汽车的随机物体的图像，这里我们使用的是 [COCO 2017](https://hyper.ai/datasets/4909) 验证集（比完整的训练集小，因此下载速度快，在完整数据集上进行训练效果更佳）。注意，COCO 2017 数据集中有一些汽车图像，但是数量很少，不会对结果产生影响 - 只会稍微降低准确度。

使用 TensorFlow 数据加载器程序，并改为手动操作，以确保能够轻松更改正在使用的数据集。最终得到以下文件层次结构：

``` bash
/root
├── images
│   ├── object
│   │   ├── 000001.jpg
│   │   │ ...
│   │   └── 016185.jpg
│   ├── object.tgz
│   ├── random
│   │   ├── 000000000139.jpg
│   │   │ ...
│   │   └── 000000581781.jpg
│   └── random.zip
```

还应该注意到，斯坦福汽车有 8k 图像，而 COCO 2017 验证集是 5k 图像——并不是对半分割！若愿意可在训练期间对这些类进行不同的加权进行纠正，不纠正仍然可以进行有效训练。下载斯坦福汽车数据集大约需要 **2分钟**，而 COCO 2017 验证集需要 **1分钟**。

``` python
import os
import shutil
import urllib.request

# 下载数据集
os.makedirs(f"{FOLDER}/downloads")
os.makedirs(f"{FOLDER}/images")
urllib.request.urlretrieve(
    "https://data.deepai.org/stanfordcars.zip", f"{FOLDER}/downloads/target.zip"
)
urllib.request.urlretrieve(
    "http://images.cocodataset.org/zips/val2017.zip", f"{FOLDER}/downloads/random.zip"
)

# 解压并重命名它们的文件夹
shutil.unpack_archive(f"{FOLDER}/downloads/target.zip", f"{FOLDER}/downloads")
shutil.unpack_archive(f"{FOLDER}/downloads/random.zip", f"{FOLDER}/downloads")
shutil.move(f"{FOLDER}/downloads/cars_train/cars_train", f"{FOLDER}/images/target")
shutil.move(f"{FOLDER}/downloads/val2017", f"{FOLDER}/images/random")
```

输出结果：

``` bash
'/tmp/tmpijas024t/images/random'
```

## 加载数据

目前，数据以各种大小的 JPG 文件形式存储在磁盘上。要使用其进行训练，必须将图像加载到内存中，并调整为 64x64，然后转换为原始的、未压缩的数据。可用 Keras 的 `image_dataset_from_directory` 来解决该问题，它加载图像时，每个像素值都是从 0 到 255 的浮点数。

从子目录结构中得知 `/objects` 中的图像是一类，而 `/random` 中的图像是另一类。设置 `label_mode='categorical'` 让 Keras 将这些转换为**分类标签**（一个 2x1 向量），对目标类的对象来说是 `[1, 0]`，对于其他任何东西来说是 `[0, 1]` 向量，此外，还将设置 `shuffle=True` 以随机化示例的顺序。

将样本分组以加快训练速度，设置 `batch_size = 32`。

最后，在机器学习中，通常希望输入是小数字。因此使用 `Rescaling` 层来更改图像，使每个像素都是 `0.0` 到 `1.0` 之间的浮点数，而不是 `0` 到 `255`。注意，因为要使用 `lambda` 函数 ，所以不要重新调整分类标签。

``` python
IMAGE_SIZE = (64, 64, 3)
unscaled_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{FOLDER}/images",
    batch_size=32,
    shuffle=True,
    label_mode="categorical",
    image_size=IMAGE_SIZE[0:2],
)
rescale = tf.keras.layers.Rescaling(scale=1.0 / 255)
full_dataset = unscaled_dataset.map(lambda im, lbl: (rescale(im), lbl))
```

输出结果：

``` bash
Found 13144 files belonging to 2 classes.
```

### 数据集中有什么？

在将数据集提供给神经网络之前，应对其进行快速验证。数据是否能正确转换？标签是否合适？目标物体与其他物体的比例是多少？可以用 `matplotlib` 从数据集中显示一些示例：

``` python
import matplotlib.pyplot as plt

num_target_class = len(os.listdir(f"{FOLDER}/images/target/"))
num_random_class = len(os.listdir(f"{FOLDER}/images/random/"))
print(f"{FOLDER}/images/target contains {num_target_class} images")
print(f"{FOLDER}/images/random contains {num_random_class} images")

# 显示一些样本及其标签
SAMPLES_TO_SHOW = 10
plt.figure(figsize=(20, 10))
for i, (image, label) in enumerate(unscaled_dataset.unbatch()):
    if i >= SAMPLES_TO_SHOW:
        break
    ax = plt.subplot(1, SAMPLES_TO_SHOW, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title(list(label.numpy()))
    plt.axis("off")
```

![图片](https://tvm.apache.org/docs/_images/sphx_glr_micro_train_001.png)

输出结果：

``` bash
/tmp/tmpijas024t/images/target contains 8144 images
/tmp/tmpijas024t/images/random contains 5000 images
```

### 验证准确度

开发模型时经常要检查它的准确度（例如，看看它在训练期间是否有所改进）。如何做到这一点？可以在*所有*数据上训练模型，然后让它对相同的数据进行分类。但是，模型可以通过记住所有样本来作弊，这会使其*看起来*具有非常高的准确性，但实际上表现非常糟糕。在实践中，这种“记忆”被称为**过拟合**。

为防止这种情况，将留出一些数据（20%）作为**验证集**，用来检查模型的准确性。

``` python
num_batches = len(full_dataset)
train_dataset = full_dataset.take(int(num_batches * 0.8))
validation_dataset = full_dataset.skip(len(train_dataset))
```

## 加载数据

在过去的十年中，[卷积神经网络](https://en.wikipedia.org/wiki/Convolutional_neural_network) 已被广泛用于图像分类任务。[EfficientNet V2](https://arxiv.org/abs/2104.00298) 等最先进的模型的图像分类效果甚至优于人类。然而，这些模型有数千万个参数，它们无法运行在便宜的监控摄像头计算机上。

应用程序的准确度达到 90% 就足够了。因此可以使用更旧版本更小的 MobileNet V1 架构。但这*仍然*不够小，默认情况下，对于具有 224x224 输入和 alpha 1.0 的 MobileNet V1，仅**存储**就需要大约 50 MB。为了减小模型的大小，可调节三个 knob。

首先，可以将输入图像的大小从 224x224 减小到 96x96 或 64x64，Keras 可以轻松做到这一点。还可以将模型的 **alpha** 从 1.0 降低到 0.25，这使得网络的宽度（和过滤器的数量）缩小了四倍。如果真的空间有限，可以通过让模型拍摄灰度图像而不是 RGB 图像来减少**通道**数量。

在本教程中，使用 RGB 64x64 输入图像和 alpha 0.25。效果虽然不是很理想，但它允许完成的模型适合 192 KB 的 RAM，同时仍然允许使用官方 TensorFlow 源模型执行迁移学习（如果使用 alpha \<0.25 或灰度输入，无法做到这样）。

### 什么是迁移学习？

深度学习长期以来一直 [主导图像分类](https://paperswithcode.com/sota/image-classification-on-imagenet)，但训练神经网络需要大量时间。当一个神经网络“从头开始”训练时，起初参数会随机初始化，缓慢地学习如何区分图像。

从一个**已经**擅长特定任务的神经网络开始迁移学习，在此示例中，该任务是对 [ImageNet 数据集](https://www.image-net.org/) 中的图像进行分类。这意味着神经网络已经具有一些目标检测。

这对于像 MobileNet 这样的图像处理神经网络特别有效，在实践中，模型的卷积层（即前 90% 的层）用于识别 line 和 shape 等低级特征——只有最后几个全连接层用于确定，这些 shape 如何构成网络要检测的目标。

可以通过使用在 ImageNet 上训练的 MobileNet 模型开始训练，该模型已经知道如何识别线条和形状。从这个预训练模型中删除最后几层，并添加自己的最终层。在汽车与非汽车数据集上训练这个联合模型，以调优第一层并从头开始训练最后一层。这种训练已经部分训练过的模型称为*微调*。

源 MobileNets 模型（用于迁移学习） [已被 TensorFlow 人员预训练](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)，因此可以下载最接近于我们想要的版本（ 128x128 输入模型，深度为 0.25）。

``` python
os.makedirs(f"{FOLDER}/models")
WEIGHTS_PATH = f"{FOLDER}/models/mobilenet_2_5_128_tf.h5"
urllib.request.urlretrieve(
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_2_5_128_tf.h5",
    WEIGHTS_PATH,
)

pretrained = tf.keras.applications.MobileNet(
    input_shape=IMAGE_SIZE, weights=WEIGHTS_PATH, alpha=0.25
)
```

### 修改网络

如上所述，预训练模型旨在对 1,000 个 ImageNet 类别进行分类，但我们希望将其转换为对汽车进行分类。由于只有最后几层是特定于任务的，因此**切断原始模型的最后五层**，通过执行 respape、dropout、flatten 和 softmax 操作来构建自己的模型的最后几层。

```plain
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE))
model.add(tf.keras.Model(inputs=pretrained.inputs, outputs=pretrained.layers[-5].output))

model.add(tf.keras.layers.Reshape((-1,)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, activation="softmax"))
```

### 微调网络

在训练神经网络时，必须设置**学习率**来控制网络学习速度。太慢和太快都会导致学习效果不好。通常对于 Adam（正在使用的优化器）来说，`0.001` 是一个相当不错的学习率（并且是 [原始论文](https://arxiv.org/abs/1412.6980) 中推荐的）。在本示例中，`0.0005` 效果更好。

将之前的验证集传递给 `model.fit`，在每次训练时评估模型性能，并能够跟踪模型性能是如何提升的。训练完成后，模型的验证准确率应该在 `0.98` 左右（这意味着在验证集上训练 100 次，其中 98 次都是预测正确的）。

``` python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_dataset, validation_data=validation_dataset, epochs=3, verbose=2)
```

输出结果：

``` bash
Epoch 1/3
328/328 - 56s - loss: 0.2151 - accuracy: 0.9280 - val_loss: 0.1213 - val_accuracy: 0.9615
Epoch 2/3
328/328 - 53s - loss: 0.1029 - accuracy: 0.9623 - val_loss: 0.1111 - val_accuracy: 0.9626
Epoch 3/3
328/328 - 52s - loss: 0.0685 - accuracy: 0.9750 - val_loss: 0.1541 - val_accuracy: 0.9505

<keras.callbacks.History object at 0x7efef1110b90>
```

## 量化

通过改变输入维度，以及移除底层，将模型减少到只有 219k 个参数，每个参数都是 `float32` 类型，占用 4 个字节，因此模型将占用将近 1 MB！

此外，硬件可能没有内置对浮点数的支持，虽然大多数高内存 Arduino（如 Nano 33 BLE）确实有硬件支持，但其他一些（如 Arduino Due）则没有。在任何*没有*专用硬件支持的板上，浮点乘法都会非常慢。

为解决这两个问题可以将模型**量化**，把权重表示为八位整数，为获得最佳性能，TensorFlow 会跟踪模型中每个神经元的激活方式，因此可以得出，如何最准确地用整数运算，来模拟神经元的原始激活。

可以创建一个具有代表性的数据集来帮助 TensorFlow 实现——原始数据集的一个子集，用于跟踪这些神经元的激活方式。然后将其传递给带有 `Optimize` 标志的 `TFLiteConverter` （Keras 本身不支持量化），以告诉 TFLite 执行转换。默认情况下，TFLite 将模型的输入和输出保持为浮点数，因此必须明确告诉它避免这种行为。

``` python
def representative_dataset():
    for image_batch, label_batch in full_dataset.take(10):
        yield [image_batch]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

quantized_model = converter.convert()
```

### 下载所需模型

现已经完成了一个模型，可以在本地或其他教程中使用它（尝试自动调优此模型或在 https://netron.app/ 上查看）。但在做这些事情之前，必须将它写入一个文件（`quantized.tflite`）。如果你在 Google Colab 上运行本教程，需要取消注释最后两行才能在编写文件后下载文件。

``` python
QUANTIZED_MODEL_PATH = f"{FOLDER}/models/quantized.tflite"
with open(QUANTIZED_MODEL_PATH, "wb") as f:
    f.write(quantized_model)
# from google.colab import files
# files.download(QUANTIZED_MODEL_PATH)
```

## 使用 TVM 为 Arduino 编译

TensorFlow 有一个用于部署到微控制器的内置框架——[TFLite Micro](https://www.tensorflow.org/lite/microcontrollers)。但是，开发板不能很好地支持它，并且不支持自动调优，因此改用 Apache TVM。

TVM 可以与其命令行界面（`tvmc`）或 Python 界面一起使用。Python 接口功能齐全，且更稳定，因此这里用 Python。

TVM 是一个优化编译器，对模型的优化是通过**中间表示**分阶段执行的。其中第一个是 [Relay](https://arxiv.org/abs/1810.00952)（一种强调可移植性的高级 intermediate representation）。从 `.tflite` 到 Relay 的转换是在不知道“最终目标”的情况下完成的，我们打算在 Arduino 上运行这个模型。

### 选择 Arduino 板

接下来确定要使用哪个 Arduino 板。最终生成的 Arduino sketch 应该与任何板子兼容，但是提前知道使用的是哪个板子，可以让 TVM 调整其编译策略从而获得更好的性能。

有一点需要注意：要有足够的**内存**（闪存和 RAM）来运行我们的模型，在 Arduino Uno 上无法运行像 MobileNet 这样的复杂的视觉模型，该板只有 2 kB 的 RAM 和 32 kB 的闪存！而模型有大约 200,000 个参数，因此无法拟合。

本教程使用具有 1 MB 闪存和 256 KB RAM 的 Nano 33 BLE，其他具有更高规格的 Arduino 同样适用。

### 生成项目

接下来把模型编译为 TVM 的 MLF（模型库格式）intermediate representation，它由 C/C++ 代码组成，专为自动调优而设计。为了提高性能，我们将告诉 TVM 我们正在为 `nrf52840` 微处理器（Nano 33 BLE 使用的那个）进行编译。此外，还告诉 TVM 使用 C runtime（缩写为 `crt`），并使用 ahead-of-time 内存分配（缩写为 `aot`，有助于减少模型的内存占用）。最后，由于 C 没有原生向量化类型，所以用 `"tir.disable_vectorize": True` 禁用向量化，。

设置了这些配置参数后，调用 `tvm.relay.build` 将 Relay 模型编译为 MLF intermediate representation。此后，只需要调用 `tvm.micro.generate_project` 并传入 Arduino 模板项目即可完成编译。

``` python
import shutil
import tvm
import tvm.micro.testing

# 在 TFLite 1 和 2 中加载模型的方法不同
try:  # TFLite 2.1 and above  # TFLite 2.1 及以上
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(quantized_model, 0)
except AttributeError:  # Fall back to TFLite 1.14 method # 回退到 TFLite 1.14 方法
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(quantized_model, 0)

# 转换为 Relay 中间表示
mod, params = tvm.relay.frontend.from_tflite(tflite_model)

# 设置配置标志以提高性能
target = tvm.micro.testing.get_target("zephyr", "nrf5340dk_nrf5340_cpuapp")
runtime = tvm.relay.backend.Runtime("crt")
executor = tvm.relay.backend.Executor("aot", {"unpacked-api": True})

# 转换为 MLF 中间表示
with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    mod = tvm.relay.build(mod, target, runtime=runtime, executor=executor, params=params)

# 从 MLF 中间表示生成一个 Arduino 项目
shutil.rmtree(f"{FOLDER}/models/project", ignore_errors=True)
arduino_project = tvm.micro.generate_project(
    tvm.micro.get_microtvm_template_projects("arduino"),
    mod,
    f"{FOLDER}/models/project",
    {
        "board": "nano33ble",
        "arduino_cli_cmd": "/content/bin/arduino-cli",
        "project_type": "example_project",
    },
)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 测试 Arduino 项目

加载下面两张 224x224 图像（一张是汽车，一张不是汽车），然后执行编译模型来测试 Arduino 项目。

![图片](/img/docs/tlc-pack/web-data/main/testdata/microTVM/data/model_train_images_combined.png)

这些是可以从 Imgur 下载的 224x224 PNG 图像。在输入这些图像之前，需要调整它们的大小并转换为原始数据，这可以使用 `imagemagick` 完成。

由于只编译 C/CPP 文件（和类似文件），因此将原始数据加载到 Arduino 上具有一定的挑战性。可以通过使用内置程序 `bin2c` 将原始数据嵌入硬编码 C 数组中来解决此问题，该程序输出如下：

``` c
static const unsigned char CAR_IMAGE[] = {
  0x22,0x23,0x14,0x22,
  ...
  0x07,0x0e,0x08,0x08
};
```

可以用几行 Bash 代码来完成这两件事：

``` bash
%%bash
mkdir -p ~/tests
curl "https://i.imgur.com/JBbEhxN.png" -o ~/tests/car_224.png
convert ~/tests/car_224.png -resize 64 ~/tests/car_64.png
stream ~/tests/car_64.png ~/tests/car.raw
bin2c -c -st ~/tests/car.raw --name CAR_IMAGE ~/models/project/car.c

curl "https://i.imgur.com/wkh7Dx2.png" -o ~/tests/catan_224.png
convert ~/tests/catan_224.png -resize 64 ~/tests/catan_64.png
stream ~/tests/catan_64.png ~/tests/catan.raw
bin2c -c -st ~/tests/catan.raw --name CATAN_IMAGE ~/models/project/catan.c
```

## 编写 Arduino 脚本

现需要 Arduino 代码来读取刚刚生成的两个二进制数组，并运行模型，把输出记录到串行监视器。该文件将替换 `arduino_sketch.ino` 作为 sketch 的主文件。

``` c
%%writefile /root/models/project.ino
#include "src/model.h"
#include "car.c"
#include "catan.c"

void setup() {
  Serial.begin(9600);
  TVMInitialize();
}

void loop() {
  uint8_t result_data[2];
  Serial.println("Car results:");
  TVMExecute(const_cast<uint8_t*>(CAR_IMAGE), result_data);
  Serial.print(result_data[0]); Serial.print(", ");
  Serial.print(result_data[1]); Serial.println();

  Serial.println("Other object results:");
  TVMExecute(const_cast<uint8_t*>(CATAN_IMAGE), result_data);
  Serial.print(result_data[0]); Serial.print(", ");
  Serial.print(result_data[1]); Serial.println();

  delay(1000);
}
```

### 编译代码

项目已经生成，TVM 的工作就基本完成了！仍然可以调用 `arduino_project.build()` 和 `arduino_project.upload()`，它们只是使用了 `arduino-cli` 的编译和 flash 命令。另外一个教程中会介绍如何自动调优我们的模型。最后，验证项目没有引发编译器错误：

``` python
shutil.rmtree(f"{FOLDER}/models/project/build", ignore_errors=True)
arduino_project.build()
print("Compilation succeeded!")
```

输出结果：

``` bash
Compilation succeeded!
```

## 上传到设备

最后一步是将 sketch 上传到 Arduino 以确保代码正常工作。Google Colab 不能做到这一点，所以必须下载 sketch。只需将项目转换为 *.zip* 文件，然后调用 *files.download*。若你在 Google Colab 上运行，则需取消最后两行注释才能在编写文件后下载文件。

``` python
ZIP_FOLDER = f"{FOLDER}/models/project"
shutil.make_archive(ZIP_FOLDER, "zip", ZIP_FOLDER)
# from google.colab import files
# files.download(f"{FOLDER}/models/project.zip")
```

在 Arduino IDE 中打开 sketch，必须为你所使用的板下载 IDE 和 SDK。对于某些板卡，例如 Sony SPRESENSE，可能需要更改设置以控制板卡使用多少内存。

### 预期结果

若一切正常，应该在串行监视器上看到以下输出：

``` bash
Car results:
255, 0
Other object results:
0, 255
```

第一个数字代表模型判断物体**是**汽车的置信度，范围为 0-255。第二个数字代表模型判断物体**不是**汽车的置信度，范围也是 0-255。以上结果表示模型非常确定第一张图像是汽车，而第二张图像不是汽车。这个判断是正确的，表明模型正常运行了。

## 总结

本教程使用迁移学习来快速训练图像识别模型，并用来识别汽车。修改模型的输入尺寸和神经网络的最后几层，使模型效果更好，同时更快更小。然后量化模型并使用 TVM 进行编译创建 Arduino sketch。最后，使用两个静态图像对模型进行测试，证明它可以按预期工作。

### 下一步

修改模型从摄像头读取实时图像，[在 GitHub](https://github.com/guberti/tvm-arduino-demos/tree/master/examples/person_detection) 上有另一个 Arduino 教程说明如何操作。也可以 [使用 TVM 的自动调整功能](https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_autotune.html) 来显著提高模型的性能。

**脚本总运行时长：**（ 4 分 44.392 秒）

[下载 Python 源代码：micro_train.py](https://tvm.apache.org/docs/v0.13.0/_downloads/b52cec46baf4f78d6bcd94cbe269c8a6/micro_train.py)

[下载 Jupyter Notebook：micro_train.ipynb](https://tvm.apache.org/docs/v0.13.0/_downloads/a7c7ea4b5017ae70db1f51dd8e6dcd82/micro_train.ipynb)
