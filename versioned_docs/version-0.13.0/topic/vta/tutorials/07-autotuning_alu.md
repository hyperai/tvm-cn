---
title: 在 VTA 上自动调优 ALU 融合算子
---

# 在 VTA 上自动调优 ALU 融合算子

:::note
单击 [此处](https://tvm.apache.org/docs/topic/vta/tutorials/autotvm/tune_alu_vta.html#sphx-glr-download-topic-vta-tutorials-autotvm-tune-alu-vta-py) 下载完整的示例代码
:::

``` python
import os
from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

from tvm import topi
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import download
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm import record

import vta
from vta.testing import simulator
from vta.top import graph_pack
import copy
```

# 编译网络

使用来自 Gluon 模型的 Relay 执行特定于 VTA 的编译：

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
    # 注意：我们将 opt_level 设置为 3 以折叠批量规范
    with relay.build_config(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

    # 对 VTA 目标进行图打包和常量折叠
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

# 设置调优选项

调优前，需要应用一些配置。这里以 Pynq-Z1 板为例：

``` python
# Tracker 主机和端口可以由你的环境设置
tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

# 从 vta/config/vta_config.json 文件中加载 VTA 参数
env = vta.get_env()

# 此 target 用于交叉编译。可以在你的设备上通过：code:`gcc -v` 来查询。
# 设置 ``device=arm_cpu`` 在 CPU 上运行推理
# 或者设置 ``device=vta`` 在 FPGA 上运行推理
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# 要编译的 Gluon 模型的名称
# ``start_pack`` 和 ``stop_pack`` 标签指示在哪里开始和结束图形打包 Relay pass：换言之，从哪里开始和结束转移到 VTA。
network = "resnet50_v2"
start_pack = "nn.max_pool2d"
stop_pack = "nn.global_avg_pool2d"

# 调优选项
log_file = "%s.alu.%s.log" % (device, network)
tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "n_trial": 1000,
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(n_parallel=1),
        runner=autotvm.RPCRunner(
            env.TARGET,
            host=tracker_host,
            port=tracker_port,
            number=5,
            timeout=60,
            # check_correctness=True, # TODO: 当 check_correctness 再次起作用时重新启用。
        ),
    ),
}

def log_to_file(file_out, protocol="json"):
    """Log the tuning records into file.
    The rows of the log are stored in the format of autotvm.record.encode.
    for lhs == rhs, we add an extra rhs = [] record
    将调优日志记录到文件中。
	日志的行以 autotvm.record.encode 的格式存储。
	对于 lhs == rhs，添加一个额外的 rhs = [] 来记录

    Parameters
    参数
    ----------
    file_out : str
        The file to log to.
        记录的文件。
    protocol: str, optional
        The log protocol. Can be 'json' or 'pickle'
        日志协议。为 'json' 或 ’pickle‘。

    Returns
    返回值
    -------
    callback : callable
        Callback function to do the logging.
        实现日志记录的回调函数。
    """

    def _callback(_, inputs, results):
        with open(file_out, "a") as f:
            for inp, result in zip(inputs, results):
                f.write(record.encode(inp, result, protocol) + "\n")

                # 只考虑具有相同 lhs 和 rhs 的任务
                if inp.task.args[0] == inp.task.args[1]:
                    args = list(inp.task.args)
                    args[1] = (args[0][0], (), args[0][2])
                    inp_copy = copy.deepcopy(inp)
                    inp_copy.task.args = tuple(args)
                    f.write(record.encode(inp_copy, result, protocol) + "\n")

    return _callback

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=10,
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
                log_to_file(tmp_log_file),
            ],
        )

    # 选择最佳记录到缓存文件
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
```

注册特定于 VTA 的调优任务

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

    # 初始化 autotvm 环境以注册 VTA 算子
    TaskExtractEnv()

    @autotvm.template("add.vta")
    def _topi_add(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, B = args[:2]

        with tvm.target.vta():
            res = vta.top.op.add_packed(*args, **kwargs)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.op.schedule_add_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, B, res]

    @autotvm.template("multiply.vta")
    def _topi_multiply(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, B = args[:2]

        with tvm.target.vta():
            res = vta.top.op.multiply_packed(*args, **kwargs)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.op.schedule_multiply_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, B, res]
```

最后，启动调优作业，并评估端到端性能。

``` python
def tune_and_evaluate(tuning_opt):

    if env.TARGET != "intelfocl":
        print("ALU only op only available for intelfocl target")
        return

    # 注册 VTA 调优任务
    register_vta_tuning_tasks()

    # 对 Relay 程序进行任务提取
    print("Extract tasks...")
    relay_prog, params = compile_network(env, target, network, start_pack, stop_pack)
    mod = tvm.IRModule.from_expr(relay_prog)
    tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        ops=(
            relay.op.get("add"),
            relay.op.get("multiply"),
        ),
        target=tvm.target.Target(target, host=env.target_host),
    )

    # 过滤掉非打包的 alu 任务
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4, tasks))
    # 过滤掉 float alu 任务
    tasks = list(filter(lambda t: t.args[0][2] != "float32", tasks))

    # 我们应该已经提取了 10 个卷积任务
    tasks_set = {}
    print("Extracted {} alu tasks:".format(len(tasks)))
    for tsk in tasks:
        print("tsk = ", tsk)

        if len(tsk.args[1][1]) == 0:
            args = list(tsk.args)
            args[1] = args[0]
            tsk.args = tuple(args)

        if (tsk.name, tsk.args) in tasks_set:
            print("task {} already exists".format(tsk))
        tasks_set[(tsk.name, tsk.args)] = tsk

    tasks = list(tasks_set.values())
    print("After merged, final #tasks={}, tasks = {}".format(len(tasks), tasks))

    # 运行调优任务
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

# 运行调优并评估结果
tune_and_evaluate(tuning_option)
```

输出结果：

``` bash
ALU only op only available for intelfocl target
```

[下载 Python 源代码：tune_alu_vta.py](https://tvm.apache.org/docs/_downloads/6bbcf342fb9192416b8e1a86a1d4e981/tune_alu_vta.py)

[下载 Jupyter Notebook：tune_alu_vta.ipynb](https://tvm.apache.org/docs/_downloads/178b6f23dffc01ac92f2cf95f41a5679/tune_alu_vta.ipynb)