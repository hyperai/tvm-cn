---
title: 为 NVIDIA GPU 自动调优卷积网络
---

# 为 NVIDIA GPU 自动调优卷积网络

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_cuda.html#sphx-glr-download-how-to-tune-with-autotvm-tune-relay-cuda-py) 下载完整的示例代码
:::

**作者**：[Lianmin Zheng](https://github.com/merrymercy)，[Eddie Yan](https://github.com/eqy/)

针对特定设备和工作负载的自动调优对于获得最佳性能至关重要，本文介绍如何为 NVIDIA GPU 调优整个卷积网络。

TVM 中 NVIDIA GPU 的算子实现是以 template 形式编写的，该 template 有许多可调参数（tile 因子，unrolling 等）。对神经网络中的所有卷积和深度卷积算子调优后，会生成一个日志文件，它存储所有必需算子的最佳参数值。当 TVM 编译器编译这些算子时，会查询这个日志文件，从而获取最佳参数值。

我们还发布了一些 NVIDIA GPU 的预调参数，可以前往 [NVIDIA GPU Benchmark](https://github.com/apache/tvm/wiki/Benchmark#nvidia-gpu) 查看详细信息。

注意，本教程无法在 Windows 或最新版本的 macOS 上运行。如需运行，请将本教程的主体放在 `if __name__ == "__main__":` 代码块中。

## 安装依赖

要在 TVM 中使用 autotvm 包，需要安装额外的依赖（如果用的是 Python2，请将「3」更改为「2」）：

``` bash
pip3 install --user psutil xgboost tornado cloudpickle
```

为了让 TVM 在调优过程中运行更快，推荐使用 Cython 作为 TVM 的 FFI。在 TVM 的根目录下，执行如下命令：

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
import tvm.contrib.graph_executor as runtime
```

## 定义网络

首先要在 Relay 前端 API 中定义网络，可以从 `tvm.relay.testing` 加载一些预定义的网络。也可以从 MXNet、ONNX 和 TensorFlow 加载模型。

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

## 设置调优选项

在调优之前，进行配置。

``` python
#### 设备配置 ####
target = tvm.target.cuda()

#### 调优 OPTION ####
network = "resnet-18"
log_file = "%s.log" % network
dtype = "float32"

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 2000,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}
```

输出结果：

``` bash
/workspace/python/tvm/target/target.py:389: UserWarning: Try specifying cuda arch by adding 'arch=sm_xx' to your target.
  warnings.warn("Try specifying cuda arch by adding 'arch=sm_xx' to your target.")
```

:::note
如何设置调优选项

通常，提供的默认值效果很好。

如果调优时间充足，可以把 `n_trial`，`early_stopping` 设置得大一些，就可以让调优运行的时间更长。

若有多个设备，可以用所有设备进行测试，以加快调优过程。（参阅下面的 `Scale up measurement` 部分）。
:::

## 开始调优

现在从网络中提取调优任务，并开始调优。接下来我们提供一个简单的实用函数。它只是一个初始实现，按顺序对任务列表进行调优。未来会引入更复杂的调优 scheduler。

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
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
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

最后启动调优作业，并评估端到端性能。

``` python
def tune_and_evaluate(tuning_opt):
    # 从 relay 程序中提取工作负载
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # 运行调优任务
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # 编译具有历史最佳记录的内核
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # 加载参数
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("data", data_tvm)

        # 评估
        print("Evaluate inference time cost...")
        print(module.benchmark(dev, number=1, repeat=600))



# 不在网页服务器中运行调优，因为它需要的时间太长。
# 取消注释运行下一行
# tune_and_evaluate(tuning_option)
```

## 样本输出

调优需要编译许多程序并从中提取特征，所以推荐使用高性能的 CPU。下面列出了一个输出示例。在 32T AMD Ryzen Threadripper 上大约耗时 4 小时才能看到以下输出，调优 target 是 NVIDIA 1080 Ti。（编译时会看到一些报错，若调优继续运行则可以忽略。）

``` bash
Extract tasks...
Tuning...
[Task  1/12]  Current/Best:  541.83/3570.66 GFLOPS | Progress: (960/2000) | 1001.31 s Done.
[Task  2/12]  Current/Best:    0.56/ 803.33 GFLOPS | Progress: (704/2000) | 608.08 s Done.
[Task  3/12]  Current/Best:  103.69/1141.25 GFLOPS | Progress: (768/2000) | 702.13 s Done.
[Task  4/12]  Current/Best: 2905.03/3925.15 GFLOPS | Progress: (864/2000) | 745.94 sterminate called without an active exception
[Task  4/12]  Current/Best: 2789.36/3925.15 GFLOPS | Progress: (1056/2000) | 929.40 s Done.
[Task  5/12]  Current/Best:   89.06/1076.24 GFLOPS | Progress: (704/2000) | 601.73 s Done.
[Task  6/12]  Current/Best:   40.39/2129.02 GFLOPS | Progress: (1088/2000) | 1125.76 s Done.
[Task  7/12]  Current/Best: 4090.53/5007.02 GFLOPS | Progress: (800/2000) | 903.90 s Done.
[Task  8/12]  Current/Best:    4.78/1272.28 GFLOPS | Progress: (768/2000) | 749.14 s Done.
[Task  9/12]  Current/Best: 1391.45/2325.08 GFLOPS | Progress: (992/2000) | 1084.87 s Done.
[Task 10/12]  Current/Best: 1995.44/2383.59 GFLOPS | Progress: (864/2000) | 862.60 s Done.
[Task 11/12]  Current/Best: 4093.94/4899.80 GFLOPS | Progress: (224/2000) | 240.92 sterminate called without an active exception
[Task 11/12]  Current/Best: 3487.98/4909.91 GFLOPS | Progress: (480/2000) | 534.96 sterminate called without an active exception
[Task 11/12]  Current/Best: 4636.84/4912.17 GFLOPS | Progress: (1184/2000) | 1381.16 sterminate called without an active exception
[Task 11/12]  Current/Best:   50.12/4912.17 GFLOPS | Progress: (1344/2000) | 1602.81 s Done.
[Task 12/12]  Current/Best: 3581.31/4286.30 GFLOPS | Progress: (736/2000) | 943.52 s Done.
Compile...
Evaluate inference time cost...
Mean inference time (std dev): 1.07 ms (0.05 ms)
```

参考基线为 MXNet + TensorRT 在 ResNet-18 上的时间成本为 1.30ms，所以我们更快一点。

:::note
**遇到困难？**

自调优模块容易出错，若总是看到「0.00/ 0.00 GFLOPS」，则表明存在问题。

首先确保设置了正确的设备配置，然后，在脚本开头添加如下行来打印调试信息，它将打印每个测试结果，可从中找到有用的错误消息。

``` python
import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)
```

随时在 https://discuss.tvm.apache.org 上向社区寻求帮助。
:::

## 使用多个设备加速测试

若有多个设备，可用所有设备进行测试。 TVM 使用 RPC Tracker（集中的控制器节点）来管理分布式设备，若有 10 个 GPU 卡，可以将它们全部注册到 tracker，并行运行 10 次测试，从而加快调优过程。

要启动 RPC tracker，在主机上运行如下命令。整个调优过程中都需要 tracker，因此需要为此命令打开一个新终端：

``` bash
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```

预期输出：

``` bash
INFO:RPCTracker:bind to 0.0.0.0:9190
```

需要为每个设备打开一个新终端，启动一个 RPC 专用服务器。使用 key 来区分设备的类型。（注意：对于 rocm 后端，编译器存在一些内部错误，需要在参数列表中添加 --no-fork。）

``` bash
python -m tvm.exec.rpc_server --tracker=127.0.0.1:9190 --key=1080ti
```

注册设备后，可以通过查询 rpc_tracker 来确认是否注册成功

``` bash
python -m tvm.exec.query_rpc_tracker --host=127.0.0.1 --port=9190
```

比如有 4 个 1080 ti，2 个 titanx，1 个 gfx900，输出如下：

``` bash
Queue Status
----------------------------------
key          total  free  pending
----------------------------------
1080ti       4      4     0
titanx       2      2     0
gfx900       1      1     0
----------------------------------
```

最后，更改调优选项来使用 RPCRunner。用下面的代码替换上面的相应部分。

``` python
tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 2000,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.RPCRunner(
            "1080ti",  # change the device key to your key
            "127.0.0.1",
            9190,
            number=20,
            repeat=3,
            timeout=4,
            min_repeat_ms=150,
        ),
    ),
}
```

[下载 Python 源代码：tune_relay_cuda.py](https://tvm.apache.org/docs/_downloads/0387f07dee851b2b8c6b73e3e88c3140/tune_relay_cuda.py)

[下载 Jupyter Notebook：tune_relay_cuda.ipynb](https://tvm.apache.org/docs/_downloads/d1434e80dd27eef6b1c9cbaa13f1197b/tune_relay_cuda.ipynb)