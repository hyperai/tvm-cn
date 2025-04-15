---
title: 在 VTA 上自动调优卷积网络
---

# 在 VTA 上自动调优卷积网络

:::note
单击 [此处](https://tvm.apache.org/docs/topic/vta/tutorials/autotvm/tune_relay_vta.html#sphx-glr-download-topic-vta-tutorials-autotvm-tune-relay-vta-py) 下载完整的示例代码
:::

**作者**：[Lianmin Zheng](https://github.com/merrymercy), [Thierry Moreau](https://homes.cs.washington.edu/\~moreau/)

为特定加速器设计自动调优，对于获取任何给定算子的最佳性能至关重要。本教程展示如何在 VTA 上调优整个卷积网络。

TVM 中 VTA 的算子实现是用 template 形式编写的。该 template 有许多可调 knobs（平铺因子、虚拟线程等）。对神经网络中的所有卷积算子进行调优。调优后，会生成一个日志文件，存储所有调优算子的最佳 schedule 参数。TVM 编译器编译这些算子时，会查询这个日志文件，从而获得最佳的 knob 参数。

## 安装依赖

要在 TVM 中使用 autotvm 包，需要安装额外的依赖（如果用的是 Python2，请将「3」更改为「2」）：

``` bash
pip3 install --user psutil xgboost tornado mxnet requests "Pillow<7" cloudpickle
```

为了让 TVM 在调优过程中运行更快，推荐使用 Cython 作为 TVM 的 FFI。在 TVM 的根目录下，执行如下命令（若用的是 Python2，将「3」改为「2」）：

``` bash
pip3 install --user cython
sudo make cython3
```

在 Python 代码中导入包：

``` python
import os
from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

from tvm import topi
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import vta
from vta.testing import simulator
from vta.top import graph_pack
```

## 编译网络

用来自 Gluon 模型的 Relay 执行特定于 VTA 的编译：

``` python
def compile_network(env, target, model, start_pack, stop_pack):
    # 填充 shape 和数据类型字典
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # 下架 gluon 模型，并转换为 Relay
    gluon_model = vision.get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # 更新 shape 和类型字典
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # 在 Relay 中执行量化
    # 注意：将 opt_level 设置为 3，折叠 batch norm
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

    # 对 VTA target 进行图打包和常量折叠
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=start_pack,
            stop_name=stop_pack,
        )

    return relay_prog, params
```

## 启动 RPC 跟踪器

TVM 使用 RPC session 与 Pynq 板进行通信。调优期间，调优器会将生成的代码发送到板上，并测试板上代码的速度。

为了扩大调优，TVM 用 RPC 跟踪器来管理多个设备。 RPC 跟踪器是一个集中的控制器节点。可以将所有设备注册到跟踪器。例如，若有 10 个 Pynq 板，可以将它们全部注册到跟踪器，然后并行运行 10 次测试，从而加快调优过程。

在主机上运行此命令，启动 RPC 跟踪器。整个调优过程中都需要跟踪器，因此我们需要为这个命令打开一个新终端：

``` bash
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```

预期输出：

``` bash
INFO:RPCTracker:bind to 0.0.0.0:9190
```

## 将设备注册到 RPC 跟踪器

现在可以将设备注册到跟踪器。第一步是为 Pynq 设备构建 TVM runtime。

按照 [VTA：多功能张量加速器](./) 在设备上构建 TVM runtime。然后将设备注册到跟踪器：

``` bash
python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=pynq
```

（将 `[HOST_IP]` 替换为你主机的 IP 地址）

注册设备后，可以通过查询 rpc_tracker 来确认：

``` bash
python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
```

例如，若有 6 个 Pynq 板和 11 个树莓派 3B，则输出：

``` bash
Queue Status
----------------------------------
key          total  free  pending
----------------------------------
pynq         6      6     0
rpi3b        11     11    0
----------------------------------
```

可以将多个设备注册到跟踪器，加速调优。

## 设置调优选项

调优前，需要应用一些配置。这里以 Pynq-Z1 板为例：

``` python
# 跟踪器主机和端口可以由你的环境设置
tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

# 从 3rdparty/vta-hw/config/vta_config.json 文件加载 VTA 参数
env = vta.get_env()

# 此 target 用于交叉编译。可以在你的设备上通过：code:`gcc -v` 来查询它。
# 设置 ``device=arm_cpu`` 在 CPU 上运行推理
# 或者设置 ``device=vta`` 在 FPGA 上运行推理
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# 要编译的 Gluon 模型的名称
# ``start_pack`` 和 ``stop_pack`` 标签指示在哪里开始和结束图形打包 Relay pass：换言之，从哪里开始和结束转移到 VTA。
network = "resnet18_v1"
start_pack = "nn.max_pool2d"
stop_pack = "nn.global_avg_pool2d"

# 调优选项
log_file = "%s.%s.log" % (device, network)
tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "n_trial": 1000,
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(
            env.TARGET,
            host=tracker_host,
            port=tracker_port,
            number=5,
            timeout=60,
            module_loader=vta.module_loader(),
            # check_correctness=True, # TODO: 当 check_correctness 再次起作用时重新启用。
        ),
    ),
}
```

:::note
如何设置调优选项

通常，此处提供的默认值效果很好。若调优时间充分，可以将 `n_trial`、`early_stopping` 设置为更大的值，使调优运行更长时间。若设备功率不足或 conv2d 算子很大，考虑将超时时间设置大一些。
:::

## 开始调优

现在可以从网络中提取调优任务并开始调优。这里提供了一个简单的实用函数来调优任务列表。这个函数只是初始版本，它按顺序调整任务列表。未来我们会引入更复杂的调优调度器。

由于要在 Pynq FPGA 板上完成调优，请确保 `vta_config.json` 文件中的 `TARGET` 条目设置为 `pynq`。

``` python
# 本教程可以跳过此函数的实现。
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

    # 选择最佳记录放到缓存文件
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
```

注册特定 VTA 的调优任务

``` python
def register_vta_tuning_tasks():
    from tvm.autotvm.task import TaskExtractEnv

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    # 初始化 autotvm 环境并注册 VTA 算子
    TaskExtractEnv()

    @autotvm.template("conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, W = args[:2]

        with tvm.target.vta():
            res = vta.top.conv2d_packed(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.schedule_conv2d_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, W, res]
```

最后，启动调优作业，并评估端到端性能。

``` python
def tune_and_evaluate(tuning_opt):
    # 注册 VTA 调优任务
    register_vta_tuning_tasks()

    # 对 Relay 程序进行任务提取
    print("Extract tasks...")
    relay_prog, params = compile_network(env, target, network, start_pack, stop_pack)
    mod = tvm.IRModule.from_expr(relay_prog)
    tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        ops=(relay.op.get("nn.conv2d"),),
        target=target,
        target_host=env.target_host,
    )

    # 过滤掉非打包的 conv2d 任务
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4 and "conv" in t.name, tasks))

    # 我们应该已经提取了 10 个卷积任务
    assert len(tasks) == 10
    print("Extracted {} conv2d tasks:".format(len(tasks)))
    for tsk in tasks:
        inp = tsk.args[0][1]
        wgt = tsk.args[1][1]
        batch = inp[0] * inp[4]
        in_filter = inp[1] * inp[5]
        out_filter = wgt[0] * wgt[4]
        height, width = inp[2], inp[3]
        hkernel, wkernel = wgt[2], wgt[3]
        hstride, wstride = tsk.args[2][0], tsk.args[2][1]
        hpad, wpad = tsk.args[3][0], tsk.args[3][1]
        print(
            "({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
                batch,
                height,
                width,
                in_filter,
                out_filter,
                hkernel,
                wkernel,
                hpad,
                wpad,
                hstride,
                wstride,
            )
        )

    # 不在网页服务器中运行调优，因为它需要的时间太长。
    # 注释以下行以自行运行。
    return

    # 运行调优任务
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # 评估调优历史
    if env.TARGET != "sim":
        # 从队列节点获取远程
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, tracker_port, timeout=10000
        )
        # 重新配置 JIT runtime 和 FPGA。
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)
    else:
        # 在模拟模式下，本地托管 RPC 服务器。
        remote = rpc.LocalSession()

    # 编译具有历史最佳记录的内核
    with autotvm.tophub.context(target, extra_files=[log_file]):
        # 编译网络
        print("Compile...")
        if target.device_name != "vta":
            with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
                lib = relay.build(
                    relay_prog, target=target, params=params, target_host=env.target_host
                )
        else:
            with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
                lib = relay.build(
                    relay_prog, target=target, params=params, target_host=env.target_host
                )

        # 导出库
        print("Upload...")
        temp = utils.tempdir()
        lib.export_library(temp.relpath("graphlib.tar"))
        remote.upload(temp.relpath("graphlib.tar"))
        lib = remote.load_module("graphlib.tar")

        # 生成图执行器
        ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
        m = graph_executor.GraphModule(lib["default"](ctx))

        # 上传参数到设备
        image = tvm.nd.array((np.random.uniform(size=(1, 3, 224, 224))).astype("float32"))
        m.set_input("data", image)

        # 评估
        print("Evaluate inference time cost...")
        timer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
        tcost = timer()
        prof_res = np.array(tcost.results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

# 运行调优并评估结果
tune_and_evaluate(tuning_option)
```

输出结果：

``` bash
Extract tasks...
/workspace/python/tvm/driver/build_module.py:267: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
/workspace/python/tvm/target/target.py:273: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
Extracted 10 conv2d tasks:
(1, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)
(1, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)
(1, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)
(1, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)
(1, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)
(1, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)
(1, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)
(1, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)
(1, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)
(1, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)
```

## 样本输出

调优需要编译许多程序，并从中提取特征。所以推荐使用高性能的 CPU。下面给出了一个输出示例。16T CPU 和 6 块 Pynq 板大约需要 2 个小时。

``` bash
Extract tasks...
[Warning] Invalid shape during AutoTVM task creation
Extracted 10 conv2d tasks:
    Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 16, 14, 14, 1, 16), 'int8'), ('TENSOR', (32, 16, 1, 1, 16, 16), 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 16, 14, 14, 1, 16, 'int8'), (32, 16, 1, 1, 16, 16, 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'))
    Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 8, 28, 28, 1, 16), 'int8'), ('TENSOR', (16, 8, 1, 1, 16, 16), 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 8, 28, 28, 1, 16, 'int8'), (16, 8, 1, 1, 16, 16, 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'))
    Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 4, 56, 56, 1, 16), 'int8'), ('TENSOR', (8, 4, 1, 1, 16, 16), 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 4, 56, 56, 1, 16, 'int8'), (8, 4, 1, 1, 16, 16, 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'))
    Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 4, 56, 56, 1, 16), 'int8'), ('TENSOR', (4, 4, 3, 3, 16, 16), 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 4, 56, 56, 1, 16, 'int8'), (4, 4, 3, 3, 16, 16, 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
    Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 8, 28, 28, 1, 16), 'int8'), ('TENSOR', (8, 8, 3, 3, 16, 16), 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 8, 28, 28, 1, 16, 'int8'), (8, 8, 3, 3, 16, 16, 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
    Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 4, 56, 56, 1, 16), 'int8'), ('TENSOR', (8, 4, 3, 3, 16, 16), 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 4, 56, 56, 1, 16, 'int8'), (8, 4, 3, 3, 16, 16, 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
    Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 16, 14, 14, 1, 16), 'int8'), ('TENSOR', (16, 16, 3, 3, 16, 16), 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 16, 14, 14, 1, 16, 'int8'), (16, 16, 3, 3, 16, 16, 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
    Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 8, 28, 28, 1, 16), 'int8'), ('TENSOR', (16, 8, 3, 3, 16, 16), 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 8, 28, 28, 1, 16, 'int8'), (16, 8, 3, 3, 16, 16, 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
    Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 32, 7, 7, 1, 16), 'int8'), ('TENSOR', (32, 32, 3, 3, 16, 16), 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 32, 7, 7, 1, 16, 'int8'), (32, 32, 3, 3, 16, 16, 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
    Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 16, 14, 14, 1, 16), 'int8'), ('TENSOR', (32, 16, 3, 3, 16, 16), 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 16, 14, 14, 1, 16, 'int8'), (32, 16, 3, 3, 16, 16, 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
Tuning...
[Task  1/10]  Current/Best:    0.72/  23.24 GFLOPS | Progress: (480/1000) | 640.31 s Done.
[Task  2/10]  Current/Best:    0.00/  27.69 GFLOPS | Progress: (576/1000) | 810.09 s Done.
[Task  3/10]  Current/Best:    0.00/  22.97 GFLOPS | Progress: (1000/1000) | 1125.37 s Done.
[Task  4/10]  Current/Best:    0.00/  31.26 GFLOPS | Progress: (1000/1000) | 1025.52 s Done.
[Task  5/10]  Current/Best:    0.00/  15.15 GFLOPS | Progress: (1000/1000) | 1236.58 s Done.
[Task  6/10]  Current/Best:    0.00/  22.74 GFLOPS | Progress: (1000/1000) | 906.60 s Done.
[Task  7/10]  Current/Best:    0.00/  15.27 GFLOPS | Progress: (1000/1000) | 1056.25 s Done.
[Task  8/10]  Current/Best:    0.00/   2.18 GFLOPS | Progress: (1000/1000) | 2275.29 s Done.
[Task  9/10]  Current/Best:    2.23/   3.99 GFLOPS | Progress: (1000/1000) | 2527.25 s Done.
[Task 10/10]  Current/Best:    1.56/   6.32 GFLOPS | Progress: (480/1000) | 1304.84 s Done.
Compile...
Upload...
Evaluate inference time cost...
Mean inference time (std dev): 621.79 ms (0.14 ms)
```

:::note
**遇到困难？**

自动调优模块容易出错。若总是看到「0.00/ 0.00 GFLOPS」，那么肯定有问题。

首先，确保设置了正确的设备配置。然后，通过在脚本开头添加以下代码来打印调试信息。它将打印每个测试结果，可以在其中找到有用的报错消息。

``` python
import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)
```

最后，随时在 https://discuss.tvm.apache.org 上向社区寻求帮助。
:::

[下载 Python 源代码：tune_relay_vta.py](https://tvm.apache.org/docs/_downloads/d7b7e50e9f5b4ff04d55a56e52314c71/tune_relay_vta.py)

[下载 Jupyter Notebook：tune_relay_vta.ipynb](https://tvm.apache.org/docs/_downloads/b1b0cbd807166348a0eabbad6bfbbdaf/tune_relay_vta.ipynb)
