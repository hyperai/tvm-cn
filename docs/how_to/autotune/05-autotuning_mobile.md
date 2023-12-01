---
title: 为 Mobile GPU 自动调优卷积网络
---

# 为 Mobile GPU 自动调优卷积网络

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_mobile_gpu.html#sphx-glr-download-how-to-tune-with-autotvm-tune-relay-mobile-gpu-py) 下载完整的示例代码
:::

**作者**：[Lianmin Zheng](https://github.com/merrymercy), [Eddie Yan](https://github.com/eqy)

针对特定设备的自动调优对于获得最佳性能至关重要。本文介绍如何调优整个卷积网络。

TVM 中 Mobile GPU 的算子实现是以 template 形式编写的。该 template 有许多可调参数（tile 因子，vectorization，unrolling 等）。对神经网络中的所有卷积、深度卷积和密集算子调优后，会生成一个日志文件，它存储所有必需算子的最佳参数值。当 TVM 编译器编译这些算子时，将查询此日志文件以获取最佳参数值。

我们还发布了一些 arm 设备的预调参数，可以前往 [Mobile GPU Benchmark](https://github.com/apache/tvm/wiki/Benchmark#mobile-gpu) 查看详细信息。

注意，本教程无法在 Windows 或最新版本的 macOS 上运行。若要运行，需要将本教程的主体包装在 `if __name__ == "__main__":`  块中。

## 安装依赖

要在 TVM 中使用 autotvm 包，需要安装额外的依赖（如果用的是 Python2，请将「3」更改为「2」）：

``` bash
pip3 install --user psutil xgboost tornado cloudpickle
```

为了让 TVM 在调优过程中运行更快，推荐使用 Cython 作为 TVM 的 FFI。在 TVM 的根目录下，执行如下命令：（若使用 Python2，将「3」改为「2」）：

``` bash
pip3 install --user cython
sudo make cython3
```

在 Python 代码中导入包：

``` python
import os
import numpy as np

import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_executor as runtime
```

## 定义网络

首先要在 Relay 前端 API 中定义网络，可以从 `relay.testing` 加载一些预定义的网络，也可以从 MXNet、ONNX 和 TensorFlow 加载模型。

``` python
def get_network(name, batch_size):
    """获取网络的符号定义和随机权重"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # MXNet 模型的示例
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape
```

## 启动 RPC Tracker

TVM 使用 RPC session 与 ARM 板进行通信，在调优期间，调优器会将生成的代码发送到板上并测试板上代码的速度。

为了加速调优，TVM 使用 RPC Tracker（集中的控制器节点）来管理分布式设备。例如，若有 10 部手机，可以将它们全部注册到 Tracker，并行运行 10 次测试，从而加快调优过程。

在整个调优过程中都需要 tracker，因此需要为此命令打开一个新终端，在主机上运行如下命令启动 RPC tracker：

``` bash
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```

预期输出：

``` bash
INFO:RPCTracker:bind to 0.0.0.0:9190
```

## 将设备注册到 RPC Tracker

接下来把设备注册到 Tracker。第一步是为 ARM 设备构建 TVM runtime 。

* 对于 Linux：按照 [在设备上构建 TVM Runtime](https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_rasp.html#build-tvm-runtime-on-device) 教程操作，然后将设备注册到 Tracker

   ``` python
     python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=rk3399
   ```

   （将 `[HOST_IP]` 替换为你的主机的 IP 地址）

* 对于 Android：按照此 [说明](https://github.com/apache/tvm/tree/main/apps/android_rpc) 在 Android 设备上安装 TVM RPC APK，确保可以通过 Android rpc 测试。在调优期间，打开手机开发者选项并勾选「在更改期间保持屏幕唤醒」，为手机接通电源。
  
注册设备后，通过查询 rpc_tracker 来确认是否注册成功

``` bash
python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
```

例如，如果有 2 台华为 mate10 pro、11 台树莓派 3B 和 2 台 rk3399，则输出是

``` bash
Queue Status
----------------------------------
key          total  free  pending
----------------------------------
mate10pro    2      2     0
rk3399       2      2     0
rpi3b        11     11    0
----------------------------------
```

将多个设备注册到 tracker，从而加快调优测试。

## 设置调优选项

在调优之前，需要先进行配置。这里以 RK3399 板为例。根据自己的设备修改 target 和 device_key。若用 Android 手机，请将 `use_android` 设置为 True。

``` python
#### 设备配置 ####
# 将 "aarch64-linux-gnu" 替换为你的板子的正确 target。
# 此 target 用于交叉编译。可以通过：code:`gcc -v` 来查询。
target = tvm.target.Target("opencl -device=mali", host="llvm -mtriple=aarch64-linux-gnu")

# 根据设备替换 device_key 的值
device_key = "rk3399"

# 若使用 Android 手机，设置 use_android 为 True
use_android = False

#### 调优选项 ####
network = "resnet-18"
log_file = "%s.%s.log" % (device_key, network)
dtype = "float32"

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 1000,
    "early_stopping": 450,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
        runner=autotvm.RPCRunner(
            device_key,
            host="127.0.0.1",
            port=9190,
            number=10,
            timeout=5,
        ),
    ),
}
```

:::note
#### 如何设置调优选项
通常，提供的默认值效果就不错。如果调优时间充足，可以把 `n_trial`，`early_stopping` 设置得大一些，让调优运行的时间更长。若设备运行速度非常慢，或者 conv2d 算子有很多 GFLOP，把 timeout 设置大一点。
:::
## 开始调优

下面开始从网络中提取调优任务，并开始调优。接下来我们提供一个简单的实用函数。它只是一个初始实现，按顺序对任务列表进行调优。未来会引入更复杂的调优 scheduler。

``` python
# 可跳过此函数的实现。
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # 创建 tmp 日志文件
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # 创建调优器
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # 开始调优
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

    # 选择最佳记录到缓存文件
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
```

最后启动调优任务，并评估端到端性能。

``` python
def tune_and_evaluate(tuning_opt):
    # 从 Relay 程序中提取工作负载
    print("Extract tasks...")
    mod, params, input_shape, _ = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        params=params,
        ops=(relay.op.get("nn.conv2d"),),
    )

    # 运行调优任务
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # 编译具有历史最佳记录的内核
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
        # 导出库
        tmp = tempdir()
        if use_android:
            from tvm.contrib import ndk

            filename = "net.so"
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

        # 上传模块到设备
        print("Upload...")
        remote = autotvm.measure.request_remote(device_key, "127.0.0.1", 9190, timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # 上传参数到设备
        dev = remote.device(str(target), 0)
        module = runtime.GraphModule(rlib["default"](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("data", data_tvm)

        # 评估
        print("Evaluate inference time cost...")
        print(module.benchmark(dev, number=1, repeat=30))

# 不在网页服务器中运行调优，因为它需要的时间太长。
# 取消注释运行下一行
# tune_and_evaluate(tuning_option)
```

## 样本输出

调优需要编译许多程序，并从中提取特征，所以推荐使用高性能的 CPU。下面列出了一个输出示例。在 32T AMD Ryzen Threadripper 设备上，大约需要 3 个小时。

``` bash
Extract tasks...
Tuning...
[Task  1/17]  Current/Best:   25.30/  39.12 GFLOPS | Progress: (992/1000) | 751.22 s Done.
[Task  2/17]  Current/Best:   40.70/  45.50 GFLOPS | Progress: (736/1000) | 545.46 s Done.
[Task  3/17]  Current/Best:   38.83/  42.35 GFLOPS | Progress: (992/1000) | 1549.85 s Done.
[Task  4/17]  Current/Best:   23.31/  31.02 GFLOPS | Progress: (640/1000) | 1059.31 s Done.
[Task  5/17]  Current/Best:    0.06/   2.34 GFLOPS | Progress: (544/1000) | 305.45 s Done.
[Task  6/17]  Current/Best:   10.97/  17.20 GFLOPS | Progress: (992/1000) | 1050.00 s Done.
[Task  7/17]  Current/Best:    8.98/  10.94 GFLOPS | Progress: (928/1000) | 421.36 s Done.
[Task  8/17]  Current/Best:    4.48/  14.86 GFLOPS | Progress: (704/1000) | 582.60 s Done.
[Task  9/17]  Current/Best:   10.30/  25.99 GFLOPS | Progress: (864/1000) | 899.85 s Done.
[Task 10/17]  Current/Best:   11.73/  12.52 GFLOPS | Progress: (608/1000) | 304.85 s Done.
[Task 11/17]  Current/Best:   15.26/  18.68 GFLOPS | Progress: (800/1000) | 747.52 s Done.
[Task 12/17]  Current/Best:   17.48/  26.71 GFLOPS | Progress: (1000/1000) | 1166.40 s Done.
[Task 13/17]  Current/Best:    0.96/  11.43 GFLOPS | Progress: (960/1000) | 611.65 s Done.
[Task 14/17]  Current/Best:   17.88/  20.22 GFLOPS | Progress: (672/1000) | 670.29 s Done.
[Task 15/17]  Current/Best:   11.62/  13.98 GFLOPS | Progress: (736/1000) | 449.25 s Done.
[Task 16/17]  Current/Best:   19.90/  23.83 GFLOPS | Progress: (608/1000) | 708.64 s Done.
[Task 17/17]  Current/Best:   17.98/  22.75 GFLOPS | Progress: (736/1000) | 1122.60 s Done.
Compile...
Upload...
Evaluate inference time cost...
Mean inference time (std dev): 128.05 ms (7.74 ms)
```

:::note
**遇到困难？**

自动调优模块容易出错，若总是看到「0.00/ 0.00 GFLOPS」，则意味着有问题。首先确保设置了正确的设备配置，然后，在脚本开头添加如下行来打印调试信息，打印所有测试结果，可从中找到有用的报错消息。

``` python
import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)
```

随时在 https://discuss.tvm.apache.org 上向社区寻求帮助。
:::

[下载 Python 源代码：tune_relay_mobile_gpu.py](https://tvm.apache.org/docs/_downloads/644d28fc67dfb3099fb0d275ffcf1c7c/tune_relay_mobile_gpu.py)

[下载 Jupyter Notebook：tune_relay_mobile_gpu.ipynb](https://tvm.apache.org/docs/_downloads/6b0f549107f73f2e48c894372be08bcb/tune_relay_mobile_gpu.ipynb)