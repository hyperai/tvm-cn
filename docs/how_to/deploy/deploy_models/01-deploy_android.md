---
title: 在 Android 上部署预训练模型
---

# 在 Android 上部署预训练模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_android.html#sphx-glr-download-how-to-deploy-models-deploy-model-on-android-py) 下载完整的示例代码
:::

**作者**：[Tomohiro Kato](https://tkat0.github.io/)

下面是用 Relay 编译 Keras 模型，并将其部署到 Android 设备上的示例：

``` python
import os
import numpy as np
from PIL import Image
import keras
from keras.applications.mobilenet_v2 import MobileNetV2
import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, ndk, graph_executor as runtime
from tvm.contrib.download import download_testdata
```

## 设置环境

由于 Android 需要的包比较多，推荐使用官方的 Docker 镜像。

首先，执行下面的命令来构建和运行 Docker 镜像：

``` bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
docker build -t tvm.demo_android -f docker/Dockerfile.demo_android ./docker
docker run --pid=host -h tvm -v $PWD:/workspace \
       -w /workspace -p 9190:9190 --name tvm -it tvm.demo_android bash
```

在容器中，克隆的 TVM 目录挂载到 /workspace。此时，挂载 RPC 要用的 9190 端口将在后面讨论。

:::note
请在容器中执行以下步骤。执行 `docker exec -it tvm bash` 在容器中打开一个新的终端。
:::

接下来构建 TVM：

``` bash
mkdir build
cd build
cmake -DUSE_LLVM=llvm-config-8 \
      -DUSE_RPC=ON \
      -DUSE_SORT=ON \
      -DUSE_VULKAN=ON \
      -DUSE_GRAPH_EXECUTOR=ON \
      ..
make -j10
```

TVM 构建成功后，设置 PYTHONPATH：

``` bash
echo 'export PYTHONPATH=/workspace/python:/workspace/vta/python:${PYTHONPATH}' >> ~/.bashrc
source ~/.bashrc
```

## 启动 RPC 跟踪器

TVM 用 RPC session 与 Android 设备进行通信。

在容器中运行这个命令来启动 RPC 跟踪器。因为整个调优过程都需要跟踪器，因此需要为这个命令打开一个新终端：

``` bash
python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```

预期输出：

``` bash
INFO:RPCTracker:bind to 0.0.0.0:9190
```

## 将 Android 设备注册到 RPC 跟踪器

按照 [readme page](https://github.com/apache/tvm/tree/main/apps/android_rpc) 在 Android 设备上安装 TVM RPC APK。

下面是 config.mk 的示例（启用了 OpenCL 和 Vulkan）：

``` cmake
APP_ABI = arm64-v8a

APP_PLATFORM = android-24

# 编译时是否启动 OpenCL
USE_OPENCL = 1

# 编译时是否启用 Vulkan
USE_VULKAN = 1

ifeq ($(USE_VULKAN), 1)
  # 静态链接 vulkan 需要 API 级别 24 或更高
  APP_PLATFORM = android-24
endif

# 要添加的其他 include 头，例如 SDK_PATH/adrenosdk/Development/Inc
ADD_C_INCLUDES += /work/adrenosdk-linux-5_0/Development/Inc
# 从 https://github.com/KhronosGroup/OpenCL-Headers 下载
ADD_C_INCLUDES += /usr/local/OpenCL-Headers/

# 要添加的附加链接库，例如 ANDROID_LIB_PATH/libOpenCL.so
ADD_LDLIBS = /workspace/pull-from-android-device/libOpenCL.so
```

:::note
不要忘记 [创建独立的工具链](https://github.com/apache/tvm/tree/main/apps/android_rpc#architecture-and-android-standalone-toolchain)。例如：

``` bash
$ANDROID_NDK_HOME/build/tools/make-standalone-toolchain.sh \
   --platform=android-24 --use-llvm --arch=arm64 --install-dir=/opt/android-toolchain-arm64
export TVM_NDK_CC=/opt/android-toolchain-arm64/bin/aarch64-linux-android-g++
```
:::

接下来，启动 Android 应用程序，输入 RPC 跟踪器的 IP 地址和端口来注册你的设备。

设备注册后，可以通过查询 rpc_tracker 来确认：

``` bash
python3 -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
```

例如，如果有 1 台 Android 设备，输出为

``` bash
Queue Status
----------------------------------
key          total  free  pending
----------------------------------
android      1      1     0
----------------------------------
```

运行下面的测试脚本，确认是否可以与 Android 通信，如果使用 OpenCL 和 Vulkan，要在脚本中设置 `test_opencl` 和 `test_vulkan`。

``` bash
export TVM_TRACKER_HOST=0.0.0.0
export TVM_TRACKER_PORT=9190
```

``` bash
cd /workspace/apps/android_rpc
python3 tests/android_rpc_test.py
```

## 加载预训练的 Keras 模型

加载 Keras 提供的预训练 MobileNetV2 (alpha=0.5) 分类模型：

``` python
keras.backend.clear_session()  # 销毁当前的 TF 计算图，并创建一个新的。
weights_url = "".join(
    [
        "https://github.com/JonathanCMitchell/",
        "mobilenet_v2_keras/releases/download/v1.1/",
        "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5",
    ]
)
weights_file = "mobilenet_v2_weights.h5"
weights_path = download_testdata(weights_url, weights_file, module="keras")
keras_mobilenet_v2 = MobileNetV2(
    alpha=0.5, include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
)
keras_mobilenet_v2.load_weights(weights_path)
```

为了测试模型，下载一张猫的图片，并转换其格式：

``` python
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_name = "cat.png"
img_path = download_testdata(img_url, img_name, module="data")
image = Image.open(img_path).resize((224, 224))
dtype = "float32"

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)
```

synset 用于将 ImageNet 类的标签，转换为人类更容易理解的单词。

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
```

## 用 Relay 编译模型

如果在 x86 服务器上运行示例，可将其设置为 `llvm`。如果在树莓派上运行，需要指定它的指令集。若要在真实设备上运行，需将 `local_demo` 设置为 False。

``` python
local_demo = True

# 默认会在 CPU target 上执行
# 可选值：'cpu'，'opencl' 和 'vulkan'
test_target = "cpu"

# 改变 target 配置。
# 运行 `adb shell cat /proc/cpuinfo` 命令查看 arch 的值。
arch = "arm64"
target = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch)

if local_demo:
    target = tvm.target.Target("llvm")
elif test_target == "opencl":
    target = tvm.target.Target("opencl", host=target)
elif test_target == "vulkan":
    target = tvm.target.Target("vulkan", host=target)

input_name = "input_1"
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_keras(keras_mobilenet_v2, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# 在 `relay.build` 之后，会得到三个返回值：计算图，库和新参数，因为我们做了一些优化，它们会改变参数，但模型的结果不变。
# 将库保存在本地临时目录中。
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.so")
fcompile = ndk.create_shared if not local_demo else None
lib.export_library(lib_fname, fcompile)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 通过 RPC 远程部署模型

利用 RPC 可将模型从主机部署到远程 Android 设备。

``` python
tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
key = "android"

if local_demo:
    remote = rpc.LocalSession()
else:
    tracker = rpc.connect_tracker(tracker_host, tracker_port)
    # 运行重型模型时，要添加 `session_timeout`
    remote = tracker.request(key, priority=0, session_timeout=60)

if local_demo:
    dev = remote.cpu(0)
elif test_target == "opencl":
    dev = remote.cl(0)
elif test_target == "vulkan":
    dev = remote.vulkan(0)
else:
    dev = remote.cpu(0)

# 将库上传到远程设备，并加载
remote.upload(lib_fname)
rlib = remote.load_module("net.so")

# 创建远程 runtime 模块
module = runtime.GraphModule(rlib["default"](dev))
```

## 在 TVM 上执行

``` python
# 设置输入数据
module.set_input(input_name, tvm.nd.array(x.astype(dtype)))
# 运行
module.run()
# 得到输出结果
out = module.get_output(0)

# 得到分数最高的第一个结果
top1 = np.argmax(out.numpy())
print("TVM prediction top-1: {}".format(synset[top1]))

print("Evaluate inference time cost...")
print(module.benchmark(dev, number=1, repeat=10))
```

输出结果：

``` bash
TVM prediction top-1: tiger cat
Evaluate inference time cost...
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
  15.5571      15.5695      15.7189      15.3987       0.0868
```

## 样本输出

以下是在骁龙 820 上使用 Adreno 530 的 ‘cpu’、‘opencl’ 和 ‘vulkan’ 的结果。

在 GPU 上运行比 CPU 慢。为了加快速度，需要根据 GPU 架构编写和优化 schedule。

``` bash
# cpu
TVM prediction top-1: tiger cat
Evaluate inference time cost...
Mean inference time (std dev): 37.92 ms (19.67 ms)

# opencl
TVM prediction top-1: tiger cat
Evaluate inference time cost...
Mean inference time (std dev): 419.83 ms (7.49 ms)

# vulkan
TVM prediction top-1: tiger cat
Evaluate inference time cost...
Mean inference time (std dev): 465.80 ms (4.52 ms)
```

[下载 Python 源代码：deploy_model_on_android.py](https://tvm.apache.org/docs/_downloads/21a9dd883b196be58ca1c5cd02700274/deploy_model_on_android.py)

[下载 Jupyter Notebook：deploy_model_on_android.ipynb](https://tvm.apache.org/docs/_downloads/eed2658f15243bab719b2de7769fa45a/deploy_model_on_android.ipynb)