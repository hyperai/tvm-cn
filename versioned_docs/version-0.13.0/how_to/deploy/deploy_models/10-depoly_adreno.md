---
title: 在 Adreno™ 上部署预训练模型
---

# 在 Adreno™ 上部署预训练模型
**作者**: Daniil Barinov, Siva Rama Krishna
:::note
单击 [此处](https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno.html#sphx-glr-download-how-to-deploy-models-deploy-model-on-adreno-py) 下载完整的示例代码
:::

本文是一个逐步教程，演示如何在 Adreno 上（不同精度）部署预训练的 PyTorch ResNet-18 模型。

首先，我们需要安装 PyTorch 与 TorchVision，因为我们将使用它作为我们的模型库。

可以通过 pip 快速安装：

```bash
pip install torch
pip install torchvision
```

除此之外，您应该已经为 Android 构建了 TVM。请参阅以下说明，了解如何构建它。

[在 Adreno GPU 上部署](https://tvm.apache.org/docs/v0.13.0/how_to/deploy/adreno.html)

在构建部分之后，构建目录中应该有两个文件：“libtvm_runtime.so” 和 “tvm_rpc”。让我们将它们推送到设备上并运行 TVM RPC 服务器。

## TVM RPC 服务器

要获取设备的哈希值，请使用：

```bash
adb devices
```

设置要使用的 Android 设备，如果您的计算机连接了多个设备。

```bash
export ANDROID_SERIAL=<device-hash>
```

然后，要将这两个文件上传到设备上，应该使用：

```bash
adb push {libtvm_runtime.so,tvm_rpc} /data/local/tmp
```

此时，您的设备上的路径 /data/local/tmp 将有 “libtvm_runtime.so” 和 “tvm_rpc” 。有时 cmake 找不到 “libc++_shared.so”。使用：

```bash
find ${ANDROID_NDK_HOME} -name libc++_shared.so
```

找到它，并使用 adb 将其推送到所需的设备：

```bash
adb push libc++_shared.so /data/local/tmp
```

我们现在准备运行 TVM RPC 服务器。在第一个控制台中使用以下行启动 rpc_tracker：

```bash
python3 -m tvm.exec.rpc_tracker --port 9190
```

然后，我们需要在第二个控制台中从所需的设备下运行 tvm_rpc 服务器：

```bash
adb reverse tcp:9190 tcp:9190
adb forward tcp:5000 tcp:5000
adb forward tcp:5002 tcp:5001
adb forward tcp:5003 tcp:5002
adb forward tcp:5004 tcp:5003
adb shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/tvm_rpc server --host=0.0.0.0 --port=5000 --tracker=127.0.0.1:9190 --key=android --port-end=5100
```

在编译和推断模型之前，请指定 TVM_TRACKER_HOST 和 TVM_TRACKER_PORT：

```bash
export TVM_TRACKER_HOST=0.0.0.0
export TVM_TRACKER_PORT=9190
```

检查 tracker 是否正在运行，并且设备是否可用：

```bash
python -m tvm.exec.query_rpc_tracker --port 9190
```

例如，如果有 1 个 Android 设备，输出可能是：

```info
Queue Status
----------------------------------
key          total  free  pending
----------------------------------
android      1      1     0
----------------------------------
```

## 配置
```python
import os
import torch
import torchvision
import tvm
from tvm import te
from tvm import relay, rpc
from tvm.contrib import utils, ndk
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import clml
from tvm import autotvm

# 下面是一组配置，用于控制脚本的行为，如本地运行或设备运行、目标定义、dtype 设置和自动调优启用。
# 如有需要，请根据需要更改这些设置。

# 与 float32 相比，Adreno 设备对 float16 的效率更高
# 鉴于降低精度不会影响预期输出
# 建议使用较低的精度。
# 我们有一个辅助 API，使精度转换变得简单
# 它支持 "float16" 和 "float16_acc32" 模式的 dtype。
# 让我们选择 "float16" 进行计算和 "float32" 进行累积。

calculation_dtype = "float16"
acc_dtype = "float32"

# 在编译以生成纹理之前指定 Adreno 目标
# 利用内核并获得所有纹理的好处
# 注意：此生成的示例在我们的 x86 服务器上运行以进行演示。
# 如果在 Android 设备上运行它，我们需要
# 指定其指令集。如果要在实际设备上运行此教程，请将 :code:`local_demo` 设置为 False。
local_demo = True

# 默认情况下，在 CPU 目标上执行。
# 选择 'cpu'、'opencl' 和 'opencl -device=adreno'
test_target = "cpu"

# 更改目标配置。
# 运行 `adb shell cat /proc/cpuinfo` 以查找架构。
arch = "arm64"
target = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch)

# 自动调整是计算密集型和耗时的任务，
# 因此默认情况下禁用。如果需要，请启用它。
is_tuning = False
tune_log = "adreno-resnet18.log"

# 启用 OpenCLML 加速运算符库。
enable_clml = False

```

## 获取 PyTorch 模型
从 torchvision models 获取 resnet18 

```python
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# 通过追踪抓取 TorchScripted 模型
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

```

Out：
```info
/venv/apache-tvm-py3.8/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/venv/apache-tvm-py3.8/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)

```

## 加载测试图片
我们使用一张经典的来自 ImageNet 的猫图片作为示例

```python
from PIL import Image
from tvm.contrib.download import download_testdata
from matplotlib import pyplot as plt
import numpy as np

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
plt.imshow(img)
plt.show()

# 处理图片并转换为 tensor
from torchvision import transforms

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

```

![cat](https://tvm.apache.org/docs/_images/sphx_glr_from_keras_001.png)

## 将 PyTorch 模型转换为 Relay 模块
TVM 具有用于各种框架 的在 relay.frontend 中的前端 API。现在对于 PyTorch 模型导入，我们有 relay.frontend.from_pytorch API。输入名称可以是任意的

```python
input_name = "input0"
shape_list = [(input_name, img.shape)]

mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

```

Out:
```info
/workspace/python/tvm/relay/frontend/pytorch_utils.py:47: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  return LooseVersion(torch_ver) > ver
/venv/apache-tvm-py3.8/lib/python3.8/site-packages/setuptools/_distutils/version.py:346: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  other = LooseVersion(other)
```

## 精度
```python
# Adreno 设备在 float16 上的效率比 float32 高
# 鉴于降低精度不会影响预期输出
# 建议使用较低的精度。

# TVM 通过 ToMixedPrecision 转换过程支持混合精度。
# 我们可能需要注册精度规则，比如精度类型、累加
# 数据类型等，以覆盖默认设置。
# 下面的辅助 API 简化了模块间的精度转换。

# 在上面的配置部分，计算 dtype 设置为 "float16"，累积 dtype 设置为 "float32"。

from tvm.driver.tvmc.transform import apply_graph_transforms

mod = apply_graph_transforms(
    mod,
    {
        "mixed_precision": True,
        "mixed_precision_ops": ["nn.conv2d", "nn.dense"],
        "mixed_precision_calculation_type": calculation_dtype,
        "mixed_precision_acc_type": acc_dtype,
    },
)

```

正如您在 IR 中所看到的那样，该架构现在包含强制转换操作，这些操作是为了将精度转换为 FP16。您还可以使用 "float16" 或 "float32" 作为其他 dtype 选项。

## 准备 TVM 目标

```python
# 此生成的示例在我们的 x86 服务器上运行以进行演示。

# 要在真实目标上部署并调试，请在上面的配置部分将 :code:`local_demo` 设置为 False。
# 同样，:code:`test_target` 设置为 :code:`llvm`，以使其与 x86 演示兼容。
# 请将其更改为 :code:`opencl` 或 :code:`opencl -device=adreno`，以用于上面配置中的 RPC 目标。


if local_demo:
    target = tvm.target.Target("llvm")
elif test_target.find("opencl"):
    target = tvm.target.Target(test_target, host=target)

```

## 自动调整
下面的几个指令可以使用 XGBoost 作为调优算法对 Relay 模块进行自动调优。

```python
# 自动调优过程包括提取任务、定义调优配置和
# 为每个任务调整最佳性能的内核配置。

# 获取与 RPC 相关的设置。
rpc_tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
rpc_tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
key = "android"

# 自动调优是计算密集型和耗时的任务。
# 在上面的配置中，由于此脚本在 x86 上运行进行演示，设置为 False。
# 请将 :code:`is_tuning` 设置为 True 以启用自动调优。

if is_tuning:
    # 自动调优阶段 1：提取可调优任务
    tasks = autotvm.task.extract_from_program(
        mod, target=test_target, target_host=target, params=params
    )

    # 自动调优阶段 2：定义调优配置
    tmp_log_file = tune_log + ".tmp"
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func=ndk.create_shared, timeout=15
        ),  # 在本地构建测试内核
        runner=autotvm.RPCRunner(  # 运行程序将在远程设备上运行。
            key,  # RPC 密钥
            host=rpc_tracker_host,  # 追踪主机
            port=int(rpc_tracker_port),  # 追踪端口
            number=3,  # 平均运行次数
            timeout=600,  # RPC 超时
        ),
    )
    n_trial = 1024  # 在选择最佳内核配置之前进行训练的迭代次数
    early_stopping = False  # 可以启用以在损失不断最小化时停止调优。

    # 自动调优阶段 3：遍历任务并进行调优。
    from tvm.autotvm.tuner import XGBTuner

    for i, tsk in enumerate(reversed(tasks[:3])):
        print("Task:", tsk)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # 选择调谐器
        tuner = "xgb"

        # 创建调谐器
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        # 其他调谐器类型的判断可以在此处添加

        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )
    # 自动调优阶段 4：从整体日志中选择性能最佳的配置。
    autotvm.record.pick_best(tmp_log_file, tune_log)

```

## 启用 OpenCLML 卸载
OpenCLML 卸载将尝试通过使用 OpenCLML 专有运算符库来加速支持的运算符。

```python
# 默认情况下，在上面的配置部分，:code:enable_clml 被设置为 False。

if not local_demo and enable_clml:
    mod = clml.partition_for_clml(mod, params)

```

## 编译
如果存在调优缓存，则使用调优缓存。

```python
if os.path.exists(tune_log):
    with autotvm.apply_history_best(tune_log):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
else:
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

```

## 远程通过 RPC 部署模型
使用 RPC，您可以将模型从主机机器部署到远程 Adreno 设备。

```python
if local_demo:
    remote = rpc.LocalSession()
else:
    tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
    # 运行大模型时，应该增加 `session_timeout`
    remote = tracker.request(key, priority=0, session_timeout=60)

if local_demo:
    dev = remote.cpu(0)
elif test_target.find("opencl"):
    dev = remote.cl(0)
else:
    dev = remote.cpu(0)

temp = utils.tempdir()
dso_binary = "dev_lib_cl.so"
dso_binary_path = temp.relpath(dso_binary)
fcompile = ndk.create_shared if not local_demo else None
lib.export_library(dso_binary_path, fcompile=fcompile)
remote_path = "/data/local/tmp/" + dso_binary
remote.upload(dso_binary_path)
rlib = remote.load_module(dso_binary)
m = graph_executor.GraphModule(rlib["default"](dev))

```

## 运行推理
我们现在可以设置输入，推理我们的模型并得到输出预测。

```python
m.set_input(input_name, tvm.nd.array(img.astype("float32")))
m.run()
tvm_output = m.get_output(0)
```

## 获取预测与性能统计
这块代码展示了 top-1 和 top-5 预测，同时提供模型的性能信息。

```python
from os.path import join, isfile
from matplotlib import pyplot as plt
from tvm.contrib import download


# 下载 ImageNet 分类
categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
categ_fn = "synset.txt"
download.download(join(categ_url, categ_fn), categ_fn)
synset = eval(open(categ_fn).read())

top_categories = np.argsort(tvm_output.asnumpy()[0])
top5 = np.flip(top_categories, axis=0)[:5]

# 记录 top-1 分类结果
print("Top-1 id: {}, class name: {}".format(top5[1 - 1], synset[top5[1 - 1]]))

# 记录 top-5 分类结果
print("\nTop5 predictions: \n")
print("\t#1:", synset[top5[1 - 1]])
print("\t#2:", synset[top5[2 - 1]])
print("\t#3:", synset[top5[3 - 1]])
print("\t#4:", synset[top5[4 - 1]])
print("\t#5:", synset[top5[5 - 1]])
print("\t", top5)
ImageNetClassifier = False
for k in top_categories[-5:]:
    if "cat" in synset[k]:
        ImageNetClassifier = True
assert ImageNetClassifier, "Failed ImageNet classifier validation check"

print("Evaluate inference time cost...")
print(m.benchmark(dev, number=1, repeat=10))
```

Out:
```info
/workspace/python/tvm/runtime/ndarray.py:199: DeprecationWarning: NDArray.asnumpy() will be deprecated in TVM v0.8 release. Please use NDArray.numpy() instead.
  warnings.warn(
Top-1 id: 281, class name: tabby, tabby cat

Top5 predictions:

        #1: tabby, tabby cat
        #2: tiger cat
        #3: lynx, catamount
        #4: red fox, Vulpes vulpes
        #5: Egyptian cat
         [281 282 287 277 285]
Evaluate inference time cost...
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
 3991.4967    3991.2103    3996.6988    3988.8485      2.0989
```

**该脚本的总运行时间:** ( 1 分 18.970 秒)