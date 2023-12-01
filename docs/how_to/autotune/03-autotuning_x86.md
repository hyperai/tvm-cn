---
title: 为 x86 CPU 自动调优卷积网络
---

# 为 x86 CPU 自动调优卷积网络

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_x86.html#sphx-glr-download-how-to-tune-with-autotvm-tune-relay-x86-py) 下载完整的示例代码
:::

**作者**：[Yao Wang](https://github.com/kevinthesun), [Eddie Yan](https://github.com/eqy)

本文介绍如何为 x86 CPU 调优卷积神经网络。

注意，本教程不会在 Windows 或最新版本的 macOS 上运行。如需运行，请将本教程的主体放在 `if __name__ == "__main__":` 代码块中。

``` python
import os
import numpy as np

import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime
```

## 定义网络

首先在 Relay 前端 API 中定义网络，可以从 `relay.testing` 加载一些预定义的网络，也可以使用 Relay 构建 `relay.testing.resnet`。还可以从 MXNet、ONNX 和 TensorFlow 加载模型。

本教程用 resnet-18 作为调优示例。

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
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

# 将「llvm」替换为你的 CPU 的 target。
# 例如，对于支持 Intel Xeon Platinum 8000 系列的 AWS EC2 c5 实例，
# target 是「llvm -mcpu=skylake-avx512」。
# 对于支持 Intel Xeon E5-2666 v3 的 AWS EC2 c4 实例，target 是「llvm -mcpu=core-avx2」。
target = "llvm"
batch_size = 1
dtype = "float32"
model_name = "resnet-18"
log_file = "%s.log" % model_name
graph_opt_sch_file = "%s_graph_opt.log" % model_name

# 设置图计算图的输入名称
# 对于 ONNX 模型，它通常为“0”。
input_name = "data"

# 根据 CPU 内核设置调优的线程数
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)
```

## 配置张量调优设置并创建任务

为了在 x86 CPU 上获得更好的内核执行性能，将卷积内核的数据布局从「NCHW」更改为「NCHWc」。为了处理这种情况，在 topi 中定义了 conv2d_NCHWc 算子，调优此算子而非普通的 conv2d。

使用本地模式来调优配置，RPC tracker 模式的设置类似于 [自动调优 ARM CPU 的卷积网络](autotuning_arm) 教程中的方法。

为了精准测试，应该多次重复测试，并取结果的平均值。此外，需要在重复测试时刷新权重张量的缓存，使得算子的测试延迟更接近其在端到端推理期间的实际延迟。

``` python
tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
        ),
    ),
}

# 可跳过此函数的实现。
def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # 创建调优器
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

        # 开始调优
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )

# 使用 graph Tuner 实现计算图级别最优调度
# 如果完成时间过长，则设置 use_DP=False。
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)
```

最后启动调优作业，并评估端到端性能。

``` python
def evaluate_performance(lib, data_shape):
    # 上传参数到设备
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)

    # 评估
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=100, repeat=3))

def tune_and_evaluate(tuning_opt):
    # 从 Relay 程序中提取工作负载
    print("Extract tasks...")
    mod, params, data_shape, out_shape = get_network(model_name, batch_size)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # 运行调优任务
    tune_kernels(tasks, **tuning_opt)
    tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    # 在默认模式下编译内核
    print("Evaluation of the network compiled in 'default' mode without auto tune:")
    with tvm.transform.PassContext(opt_level=3):
        print("Compile...")
        lib = relay.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)

    # 在仅内核调优模式下编译内核
    print("\nEvaluation of the network been tuned on kernel level:")
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)

    # 编译具有计算图级最佳记录的内核
    print("\nEvaluation of the network been tuned on graph level:")
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)

# 不在网页服务器中运行调优，因为它需要的时间太长。
# 取消注释运行下一行
# tune_and_evaluate(tuning_option)
```

## 样本输出

调优需要编译许多程序，并从中提取特征，推荐使用高性能的 CPU。下面列出了一个输出示例。

``` bash
Extract tasks...
Tuning...
[Task  1/12]  Current/Best:  598.05/2497.63 GFLOPS | Progress: (252/252) | 1357.95 s Done.
[Task  2/12]  Current/Best:  522.63/2279.24 GFLOPS | Progress: (784/784) | 3989.60 s Done.
[Task  3/12]  Current/Best:  447.33/1927.69 GFLOPS | Progress: (784/784) | 3869.14 s Done.
[Task  4/12]  Current/Best:  481.11/1912.34 GFLOPS | Progress: (672/672) | 3274.25 s Done.
[Task  5/12]  Current/Best:  414.09/1598.45 GFLOPS | Progress: (672/672) | 2720.78 s Done.
[Task  6/12]  Current/Best:  508.96/2273.20 GFLOPS | Progress: (768/768) | 3718.75 s Done.
[Task  7/12]  Current/Best:  469.14/1955.79 GFLOPS | Progress: (576/576) | 2665.67 s Done.
[Task  8/12]  Current/Best:  230.91/1658.97 GFLOPS | Progress: (576/576) | 2435.01 s Done.
[Task  9/12]  Current/Best:  487.75/2295.19 GFLOPS | Progress: (648/648) | 3009.95 s Done.
[Task 10/12]  Current/Best:  182.33/1734.45 GFLOPS | Progress: (360/360) | 1755.06 s Done.
[Task 11/12]  Current/Best:  372.18/1745.15 GFLOPS | Progress: (360/360) | 1684.50 s Done.
[Task 12/12]  Current/Best:  215.34/2271.11 GFLOPS | Progress: (400/400) | 2128.74 s Done.
INFO Start to benchmark layout transformation...
INFO Benchmarking layout transformation successful.
INFO Start to run dynamic programming algorithm...
INFO Start forward pass...
INFO Finished forward pass.
INFO Start backward pass...
INFO Finished backward pass...
INFO Finished DPExecutor run.
INFO Writing optimal schedules to resnet-18_graph_opt.log successfully.

Evaluation of the network compiled in 'default' mode without auto tune:
Compile...
Evaluate inference time cost...
Mean inference time (std dev): 4.5 ms (0.03 ms)

Evaluation of the network been tuned on kernel level:
Compile...
Evaluate inference time cost...
Mean inference time (std dev): 3.2 ms (0.03 ms)

Evaluation of the network been tuned on graph level:
Compile...
Config for target=llvm -keys=cpu -link-params=0, workload=('dense_nopack.x86', ('TENSOR', (1, 512), 'float32'), ('TENSOR', (1000, 512), 'float32'), None, 'float32') is missing in ApplyGraphBest context. A fallback configuration is used, which may bring great performance regression.
Config for target=llvm -keys=cpu -link-params=0, workload=('dense_pack.x86', ('TENSOR', (1, 512), 'float32'), ('TENSOR', (1000, 512), 'float32'), None, 'float32') is missing in ApplyGraphBest context. A fallback configuration is used, which may bring great performance regression.
Evaluate inference time cost...
Mean inference time (std dev): 3.16 ms (0.03 ms)
```

[下载 Python 源代码：tune_relay_x86.py](https://tvm.apache.org/docs/_downloads/6836ce26807b8d33b8f499287c1f3d04/tune_relay_x86.py)

[下载 Jupyter notebook：tune_relay_x86.ipynb](https://tvm.apache.org/docs/_downloads/910e6ecee4ecac8d8ca0baeb6d00689d/tune_relay_x86.ipynb)