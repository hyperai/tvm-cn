---
title: 为 ARM CPU 自动调度神经网络
---

# 为 ARM CPU 自动调度神经网络

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_network_arm.html#sphx-glr-download-how-to-tune-with-autoscheduler-tune-network-arm-py) 下载完整的示例代码
:::

**作者**：[Thierry Moreau](https://github.com/tmoreau89), [Lianmin Zheng](https://github.com/merrymercy), [Chengfan Jia](https://github.com/jcf94/)

针对特定设备和工作负载的自动调优对于获得最佳性能至关重要。本文介绍如何通过 RPC 使用 auto-scheduler 为 ARM CPU 调优整个神经网络。

为了自动调优神经网络，将网络划分为小的子图并独立进行调优。每个子图被视为一个搜索任务。任务调度器对时间进行切片，并动态地为这些任务分配时间资源，预测每个任务对端到端执行时间的影响，并优先考虑最能减少执行时间的任务。

对于每个子图，使用 `tvm/python/topi` 中的计算声明来获取张量表达式形式的计算 DAG。然后使用 auto-scheduler 来构建这个 DAG 的搜索空间，并搜索合适的调度（底层优化）。

与基于 template 的 [AutoTVM](/docs/how_to/autotune)（依赖手动 template 来定义搜索空间的） 不同，auto-scheduler 不需要任何调度 template。换言之，auto-scheduler 只使用 `tvm/python/topi` 中的计算声明，不使用现有的调度 template。

注意，本教程无法在 Windows 或最新版本的 macOS 上运行。如需运行，请将本教程的主体放在 `if __name__ == "__main__":` 代码块中。

``` python
import numpy as np
import os

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor
from tvm.contrib.utils import tempdir
```

## 定义网络

首先，要用 Relay 前端 API 定义网络。可以从 `tvm.relay.testing` 加载一些预定义的网络。也可以从 MXNet、ONNX、PyTorch 和 TensorFlow 加载模型（参见 [前端教程](/docs/how_to/compile))）。

对于卷积神经网络，尽管 auto-scheduler 可以在任何布局下正常运行，但通过 NHWC 布局实现的性能最佳。auto-scheduler 对 NHWC 布局进行了很多优化，因此推荐将模型转换为 NHWC 布局，从而得以使用 auto-scheduler。可用 [ConvertLayout](https://tvm.apache.org/docs/arch/convert_layout.html#convert-layout-usage) pass 在 TVM 中进行布局转换。

``` python
def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """获取网络的符号定义和随机权重"""

    # auto-scheduler 更适合 NHWC 布局
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # MXNet 模型的示例
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
        )
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, random_params=True)

    return mod, params, input_shape, output_shape
```

## 启动 RPC 跟踪器

TVM 使用 RPC session 与 ARM 板进行通信。在调优期间，调优器会将生成的代码发送到板上并测试板上代码的速度。

为了加速调优，TVM 使用 RPC 跟踪器（集中的控制器节点）来管理分布式设备。例如，若有 10 部手机，可以将它们全部注册到跟踪器，并行运行 10 次测试，从而加快调优过程。

整个调优过程都需要跟踪器。因此需要为此命令打开一个新终端，在主机上运行如下命令启动 RPC 跟踪器：

``` bash
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```

预期输出：

``` bash
INFO:RPCTracker:bind to 0.0.0.0:9190
```

## 将设备注册到 RPC 跟踪器

接下来把设备注册到跟踪器。第一步是为 ARM 设备构建 TVM runtime 。

* 对于 Linux：按照 [在设备上构建 TVM Runtime](https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_rasp.html#build-tvm-runtime-on-device) 教程操作，然后将设备注册到 Tracker

   ``` bash
     python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=rasp4b-64
   ```

   （将 `[HOST_IP]` 换为你的主机的 IP 地址）

* 对于 Android：按照此 [说明](https://github.com/apache/tvm/tree/main/apps/android_rpc) 在 Android 设备上安装 TVM RPC APK，确保可以通过 Android rpc 测试。在调优期间，打开手机开发者选项并勾选「在更改期间保持屏幕唤醒」，为手机接通电源。

注册设备后，通过查询 rpc_tracker 来确认是否注册成功

``` bash
python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
```

例如，如果有 2 个华为 mate10 pro，11 个 64 位操作系统的树莓派 4B，以及 2 个 rk3399，则输出可以是

``` bash
Queue Status
----------------------------------
key          total  free  pending
----------------------------------
mate10pro    2      2     0
rk3399       2      2     0
rasp4b-64    11     11    0
----------------------------------
```

将多个设备注册到 tracker，从而加快调优测试。

## 设置调优配置

在调优之前，进行配置。这里以 Raspberry Pi 4b 4GB 板（64 位操作系统 Ubuntu 20.04）为例。若用 Android 手机，请将 `use_ndk` 设置为 True。

``` python
#### 设备配置 ####
# 将 "aarch64-linux-gnu" 替换为你的板子的正确 target。
# 此 target 用于交叉编译。可以通过：code:`gcc -v` 来查询。
# FIXME(tmoreau89, merrymercy): 将 '-device=arm_cpu' 排除在 target 字符串之外
# 因为共享 x86 操作策略。
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")

# 替换为跟踪器中的 device_key、rpc 主机和 rpc 端口
device_key = "rasp4b-64"
rpc_host = "127.0.0.1"
rpc_port = 9190

# 如果使用 ndk 工具进行交叉编译，则设置为 True
# 并且还要设置下面的环境变量指向交叉编译器
use_ndk = False
# os.environ["TVM_NDK_CC"] = "/usr/bin/aarch64-linux-gnu-g++"

#### 调优 OPTION ####
network = "mobilenet"
use_sparse = False
batch_size = 1
layout = "NHWC"
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
```

## 提取搜索任务

接下来，从网络中提取搜索任务及其权重。任务的权重是任务的子图在整个网络中出现的次数。通过使用权重，可以将网络的端到端延迟近似为 `sum(latency[t] * weight[t])`，其中 `latency[t]` 是任务的延迟，而`weight[t]` 是任务的权重，任务调度器只会优化这个目标。

``` python
# 从网络中提取任务
print("Get model...")
mod, params, input_shape, output_shape = get_network(
    network, batch_size, layout, dtype=dtype, use_sparse=use_sparse
)
print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
```

输出结果：

``` bash
Get model...
Extract tasks...
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
========== Task 0  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 7, 7, 1024], [1, 1, 1024, 1024], [1, 1, 1, 1024], [1, 7, 7, 1024]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 1024]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 1024, 1024]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 1024]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 1  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 14, 14, 256], [1, 1, 256, 512], [1, 1, 1, 512], [1, 14, 14, 512]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 512]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 2  (workload key: ["06fce76bd84cb904eee50b905ca9449a", [1, 28, 28, 256], [3, 3, 256, 1], [1, 1, 1, 256], [1, 28, 28, 256]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 256]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 29)) && (i2 >= 1)) && (i2 < 29)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 256, 1]
DepthwiseConv2d(b, i, j, c) += (PaddedInput[b, (i + di), (j + dj), c]*placeholder[di, dj, c, 0])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (DepthwiseConv2d[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 3  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 28, 28, 128], [1, 1, 128, 256], [1, 1, 1, 256], [1, 28, 28, 256]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 128, 256]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 4  (workload key: ["d7b65649a4dd54becea0a52aabbc5af5", [1, 1000], [1, 1000]]) ==========
placeholder = PLACEHOLDER [1, 1000]
T_softmax_maxelem(i0) max= placeholder[i0, k]
T_softmax_exp(i0, i1) = tir.exp((placeholder[i0, i1] - T_softmax_maxelem[i0]))
T_softmax_expsum(i0) += T_softmax_exp[i0, k]
T_softmax_norm(i0, i1) = (T_softmax_exp[i0, i1]/T_softmax_expsum[i0])

========== Task 5  (workload key: ["69115f188984ae34ede37c3b8ca40b43", [1, 7, 7, 1024], [1, 1, 1, 1024]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 1024]
tensor(ax0, ax1, ax2, ax3) += placeholder[ax0, ((ax1*7) + rv0), ((ax2*7) + rv1), ax3]
tensor(ax0, ax1, ax2, ax3) = (tensor[ax0, ax1, ax2, ax3]/(float32((select((bool)1, ((ax1 + 1)*7), (((ax1 + 1)*7) + 1)) - (ax1*7)))*float32((select((bool)1, ((ax2 + 1)*7), (((ax2 + 1)*7) + 1)) - (ax2*7)))))

========== Task 6  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 7, 7, 512], [1, 1, 512, 1024], [1, 1, 1, 1024], [1, 7, 7, 1024]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 512]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 1024]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 1024]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 7  (workload key: ["c87ba68bc180312f5716af09a77ca15b", [1, 56, 56, 128], [3, 3, 128, 1], [1, 1, 1, 128], [1, 28, 28, 128]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 128]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 128, 1]
DepthwiseConv2d(b, i, j, c) += (PaddedInput[b, ((i*2) + di), ((j*2) + dj), c]*placeholder[di, dj, c, 0])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (DepthwiseConv2d[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 8  (workload key: ["06fce76bd84cb904eee50b905ca9449a", [1, 7, 7, 1024], [3, 3, 1024, 1], [1, 1, 1, 1024], [1, 7, 7, 1024]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 1024]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 8)) && (i2 >= 1)) && (i2 < 8)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 1024, 1]
DepthwiseConv2d(b, i, j, c) += (PaddedInput[b, (i + di), (j + dj), c]*placeholder[di, dj, c, 0])
placeholder = PLACEHOLDER [1, 1, 1, 1024]
T_add(ax0, ax1, ax2, ax3) = (DepthwiseConv2d[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 9  (workload key: ["c87ba68bc180312f5716af09a77ca15b", [1, 28, 28, 256], [3, 3, 256, 1], [1, 1, 1, 256], [1, 14, 14, 256]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 256]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 29)) && (i2 >= 1)) && (i2 < 29)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 256, 1]
DepthwiseConv2d(b, i, j, c) += (PaddedInput[b, ((i*2) + di), ((j*2) + dj), c]*placeholder[di, dj, c, 0])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (DepthwiseConv2d[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 10  (workload key: ["c87ba68bc180312f5716af09a77ca15b", [1, 14, 14, 512], [3, 3, 512, 1], [1, 1, 1, 512], [1, 7, 7, 512]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 512]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 15)) && (i2 >= 1)) && (i2 < 15)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 512, 1]
DepthwiseConv2d(b, i, j, c) += (PaddedInput[b, ((i*2) + di), ((j*2) + dj), c]*placeholder[di, dj, c, 0])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (DepthwiseConv2d[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 11  (workload key: ["c87ba68bc180312f5716af09a77ca15b", [1, 112, 112, 64], [3, 3, 64, 1], [1, 1, 1, 64], [1, 56, 56, 64]]) ==========
placeholder = PLACEHOLDER [1, 112, 112, 64]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 113)) && (i2 >= 1)) && (i2 < 113)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 64, 1]
DepthwiseConv2d(b, i, j, c) += (PaddedInput[b, ((i*2) + di), ((j*2) + dj), c]*placeholder[di, dj, c, 0])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (DepthwiseConv2d[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 12  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 28, 28, 256], [1, 1, 256, 256], [1, 1, 1, 256], [1, 28, 28, 256]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 256]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 256]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 13  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 56, 56, 128], [1, 1, 128, 128], [1, 1, 1, 128], [1, 56, 56, 128]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 128]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 128, 128]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 14  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 14, 14, 512], [1, 1, 512, 512], [1, 1, 1, 512], [1, 14, 14, 512]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 512]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 512]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 15  (workload key: ["06fce76bd84cb904eee50b905ca9449a", [1, 112, 112, 32], [3, 3, 32, 1], [1, 1, 1, 32], [1, 112, 112, 32]]) ==========
placeholder = PLACEHOLDER [1, 112, 112, 32]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 113)) && (i2 >= 1)) && (i2 < 113)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 32, 1]
DepthwiseConv2d(b, i, j, c) += (PaddedInput[b, (i + di), (j + dj), c]*placeholder[di, dj, c, 0])
placeholder = PLACEHOLDER [1, 1, 1, 32]
T_add(ax0, ax1, ax2, ax3) = (DepthwiseConv2d[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 16  (workload key: ["2ca148ecea6508ce625f85719021344f", [1, 224, 224, 3], [3, 3, 3, 32], [1, 112, 1, 1], [1, 112, 1, 1], [1, 112, 112, 32]]) ==========
placeholder = PLACEHOLDER [1, 224, 224, 3]
pad_temp(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 225)) && (i2 >= 1)) && (i2 < 225)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 3, 32]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 112, 1, 1]
T_multiply(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3]*placeholder[ax0, ax1, 0, 0])
placeholder = PLACEHOLDER [1, 112, 1, 1]
T_add(ax0, ax1, ax2, ax3) = (T_multiply[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, 0, 0])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 17  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 56, 56, 64], [1, 1, 64, 128], [1, 1, 1, 128], [1, 56, 56, 128]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 64, 128]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 18  (workload key: ["7d44c6e3c81cd80f61ff2265b2bae89a", [1, 1024], [1000, 1024], [1, 1000], [1, 1000]]) ==========
placeholder = PLACEHOLDER [1, 1024]
placeholder = PLACEHOLDER [1000, 1024]
T_matmul_NT(i, j) += (placeholder[i, k]*placeholder[j, k])
placeholder = PLACEHOLDER [1, 1000]
T_add(ax0, ax1) = (T_matmul_NT[ax0, ax1] + placeholder[ax0, ax1])

========== Task 19  (workload key: ["06fce76bd84cb904eee50b905ca9449a", [1, 14, 14, 512], [3, 3, 512, 1], [1, 1, 1, 512], [1, 14, 14, 512]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 512]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 15)) && (i2 >= 1)) && (i2 < 15)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 512, 1]
DepthwiseConv2d(b, i, j, c) += (PaddedInput[b, (i + di), (j + dj), c]*placeholder[di, dj, c, 0])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (DepthwiseConv2d[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 20  (workload key: ["06fce76bd84cb904eee50b905ca9449a", [1, 56, 56, 128], [3, 3, 128, 1], [1, 1, 1, 128], [1, 56, 56, 128]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 128]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 128, 1]
DepthwiseConv2d(b, i, j, c) += (PaddedInput[b, (i + di), (j + dj), c]*placeholder[di, dj, c, 0])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (DepthwiseConv2d[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 21  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 112, 112, 32], [1, 1, 32, 64], [1, 1, 1, 64], [1, 112, 112, 64]]) ==========
placeholder = PLACEHOLDER [1, 112, 112, 32]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 32, 64]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)
```

## 调优及评估

接下来为调优和启动搜索任务设置一些选项

* `num_measure_trials` 是调优期间可以使用的测试次数（根据自己的时间预算调整这个参数），若要进行快速演示，可将其设置为较小的数字（例如 200）。推荐将其设置为 `800 * len(tasks)` 左右，以便使搜索收敛。比如 resnet-50 有 29 个任务，所以可以设置为 20000。
* 此外，使用 `RecordToFile` 将测试记录转储到日志文件中，测试记录可用于历史最佳查询、恢复搜索以及进行后续分析。
* 更多参数参见 `auto_scheduler.TuningOptions`，`auto_scheduler.LocalRunner`。

自动调优后，可以用找到的最佳调度来编译网络。在自动调优期间，所有测试记录都被转储到日志文件中，可以读取日志文件加载最佳调度。

``` python
def tune_and_evaluate():
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # 将此更改为 20000 以达到最佳性能
        builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
        runner=auto_scheduler.RPCRunner(
            device_key,
            host=rpc_host,
            port=rpc_port,
            timeout=30,
            repeat=1,
            min_repeat_ms=200,
            enable_cpu_cache_flush=True,
        ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)

    # 用历史最佳编译
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target, params=params)

    # 导出库
    tmp = tempdir()
    if use_ndk:
        from tvm.contrib import ndk

        filename = "net.so"
        lib.export_library(tmp.relpath(filename), ndk.create_shared)
    else:
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

    # 上传模块到设备
    print("Upload...")
    remote = auto_scheduler.utils.request_remote(device_key, rpc_host, rpc_port, timeout=10000)
    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    # 创建图执行器
    dev = remote.cpu()
    module = graph_executor.GraphModule(rlib["default"](dev))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("data", data_tvm)

    # 评估
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, repeat=3, min_repeat_ms=500))

# 不在网页服务器中运行调优，因为服务器没有树莓派，
# 或正在运行的设备跟踪器。
# 取消注释运行下面行。
# tune_and_evaluate()
```

:::note
解释调优过程中打印的信息

在调优过程中，控制台上会打印很多用于调试的信息，最重要的信息是任务调度程序的输出，下表是输出示例。

``` bash
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.013 |           0.31 |     64 |
|    1 |        0.845 |           2.43 |    448 |
|    2 |        0.046 |          -0.00 |     64 |
|    3 |        4.194 |          24.53 |   2112 |
|    4 |        0.109 |           9.21 |     64 |
|    5 |        1.759 |          29.27 |    896 |
|    6 |        0.083 |           6.01 |     64 |
|    7 |        3.084 |          33.38 |   7680 |
|    8 |        0.136 |          14.78 |    384 |
|    9 |        1.349 |          38.23 |    768 |
|   10 |        0.133 |           7.55 |    128 |
|   11 |        2.747 |          37.56 |   1536 |
|   12 |        0.338 |          11.87 |    192 |
|   13 |        1.295 |          40.00 |    704 |
|   14 |        0.482 |           4.16 |    256 |
|   15 |        2.686 |          38.56 |   1344 |
|   16 |        0.884 |           9.08 |    448 |
|   17 |        1.332 |          39.18 |    704 |
|   18 |        1.045 |           3.84 |    576 |
|   19 |        1.391 |          38.09 |    704 |
|   20 |        0.777 |          10.34 |    448 |
|   21 |        0.739 |          30.97 |    448 |
-------------------------------------------------
 Estimated total latency: 38.347 ms      Trials: 19992   Used time : 19260 s     Next ID: 3
```

此表列出了所有任务的延迟和（预估）速度，还列出了所有任务的测试分配。最后一行打印了这些任务的总加权延迟，可以粗略估计网络的端到端执行时间。最后一行还打印了测试试验的总数、自动调优所花费的总时间以及下一个要调优的任务的 ID。

还有一些「dmlc::Error」错误，因为 auto-scheduler 会尝试一些无效的调度，若调优继续运行，则可以忽略这些错误，因为这些错误与主进程隔离。
:::

:::note
提前终止调优

可以通过强制终止此进程来提前终止调优，只要在日志文件中为每个任务获得至少一个有效的调度，就能够进行编译（下面的部分）。
:::

## 其他技巧

1. 在调优过程中，auto-scheduler 需要编译许多程序，并从中提取特征。这部分会占用大量 CPU 资源，所以推荐使用多核的高性能 CPU，加快搜索速度。
2. 可以用 `python3 -m tvm.auto_scheduler.measure_record --mode distill -i log.json` 提取大日志文件，并仅保存最有用的记录。
3. 可以从以前的日志文件恢复搜索，只需要在函数 `run_tuning` 中创建任务调度程序时添加一个新参数 `load_log_file`。比如，`tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)`
4. 若有多个 target CPU，则可以将所有这些 CPU 用于并行化测试。查看此 [部分](https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_cuda.html#tutorials-autotvm-scale-up-rpc-tracker) 了解如何使用 RPC 跟踪器和 RPC 服务器。要在 auto-scheduler 中使用 RPC 跟踪器，请将 `TuningOptions` 中的 runner 替换为 `auto_scheduler.RPCRunner`。

[下载 Python 源代码：tune_network_arm.py](https://tvm.apache.org/docs/_downloads/17b139d609f9480c7eeda2da1f90f28c/tune_network_arm.py)

[下载 Jupyter Notebook：tune_network_arm.ipynb](https://tvm.apache.org/docs/_downloads/4dc30a43f3a6aa3ed4bc3077ad35ff70/tune_network_arm.ipynb)
