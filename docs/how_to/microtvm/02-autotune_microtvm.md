---
title: 6.使用 microTVM 进行模型调优
---

# 6.使用 microTVM 进行模型调优

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_autotune.html#sphx-glr-download-how-to-work-with-microtvm-micro-autotune-py) 下载完整的示例代码
:::

**作者**：[Andrew Reusch](https://github.com/areusch), [Mehrdad Hessar](https://github.com/mehrdadh)

本教程介绍如何用 C runtime 自动调优模型。

## 安装 microTVM Python 依赖项
TVM 不包含用于 Python 串行通信包，因此在使用 microTVM 之前我们必须先安装一个。我们还需要TFLite来加载模型。

```bash
pip install pyserial==3.5 tflite==2.1
```

```python
# 如果下面的标志为 False，可以跳过下一节（安装 Zephyr）
# 安装 Zephyr 约花费20分钟
import os

use_physical_hw = bool(os.getenv("TVM_MICRO_USE_HW"))

```
## 安装 Zephyr

``` bash
# 安装 west 和 ninja
python3 -m pip install west
apt-get install -y ninja-build

# 安装 ZephyrProject
ZEPHYR_PROJECT_PATH="/content/zephyrproject"
export ZEPHYR_BASE=${ZEPHYR_PROJECT_PATH}/zephyr
west init ${ZEPHYR_PROJECT_PATH}
cd ${ZEPHYR_BASE}
git checkout v3.2-branch
cd ..
west update
west zephyr-export
chmod -R o+w ${ZEPHYR_PROJECT_PATH}

# 安装 Zephyr SDK
cd /content
ZEPHYR_SDK_VERSION="0.15.2"
wget "https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v${ZEPHYR_SDK_VERSION}/zephyr-sdk-${ZEPHYR_SDK_VERSION}_linux-x86_64.tar.gz"
tar xvf "zephyr-sdk-${ZEPHYR_SDK_VERSION}_linux-x86_64.tar.gz"
mv "zephyr-sdk-${ZEPHYR_SDK_VERSION}" zephyr-sdk
rm "zephyr-sdk-${ZEPHYR_SDK_VERSION}_linux-x86_64.tar.gz"

# 安装 python 依赖
python3 -m pip install -r "${ZEPHYR_BASE}/scripts/requirements.txt"
```

## 导入 Python 依赖项

``` python
import json
import numpy as np
import pathlib

import tvm
from tvm.relay.backend import Runtime
import tvm.micro.testing
```

## 定义模型

首先在 Relay 中定义一个要在设备上执行的模型，然后从 Relay 模型中创建一个 IRModule，并用随机数填充参数。

``` python
data_shape = (1, 3, 10, 10)
weight_shape = (6, 3, 5, 5)

data = tvm.relay.var("data", tvm.relay.TensorType(data_shape, "float32"))
weight = tvm.relay.var("weight", tvm.relay.TensorType(weight_shape, "float32"))

y = tvm.relay.nn.conv2d(
    data,
    weight,
    padding=(2, 2),
    kernel_size=(5, 5),
    kernel_layout="OIHW",
    out_dtype="float32",
)
f = tvm.relay.Function([data, weight], y)

relay_mod = tvm.IRModule.from_expr(f)
relay_mod = tvm.relay.transform.InferType()(relay_mod)

weight_sample = np.random.rand(
    weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
).astype("float32")
params = {"weight": weight_sample}
```

## 定义 target

下面定义描述执行环境的 TVM target，它与其他 microTVM 教程中的 target 定义非常相似。不同之处是用 C Runtime 来生成模型。

在物理硬件上运行时，选择一个 target 和一个描述硬件的单板。在本教程的 PLATFORM 列表中可以选择多个硬件目标。运行本教程时用 –platform 参数来选择平台。

``` python
RUNTIME = Runtime("crt", {"system-lib": True})
TARGET = tvm.micro.testing.get_target("crt")

# 为物理硬件编译
# --------------------------------------------------------------------------
# 在物理硬件上运行时，选择描述硬件的 TARGET 和 BOARD。
# 下面的示例中选择 STM32L4R5ZI Nucleo。
if use_physical_hw:
    BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_l4r5zi")
    SERIAL = os.getenv("TVM_MICRO_SERIAL", default=None)
    TARGET = tvm.micro.testing.get_target("zephyr", BOARD)
```

## 提取调优任务

并非上面打印的 Relay 程序中的所有算子都可以调优，有些算子不是很重要，所以只定义了一个实现；其他的作为调优任务没有意义。用 extract_from_program，可以生成可调优任务列表。

因为任务提取涉及到运行编译器，所以首先配置编译器的转换 pass；之后在自动调优过程中应用相同的配置。

``` python
pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
with pass_context:
    tasks = tvm.autotvm.task.extract_from_program(relay_mod["main"], {}, TARGET)
assert len(tasks) > 0
```

## 配置 microTVM

自动调优前，要定义一个模块加载器，然后将它传递给 *tvm.autotvm.LocalBuilder*。创建一个 *tvm.autotvm.LocalBuilder*，并用 builder 和 runner 为自动调优器生成多个测试值。

本教程中可以选择用 x86 主机作为示例，或用来自 Zephyr RTOS 的不同 targets。若用 x86，则传递 *–platform=host*，也可以从 *PLATFORM* 列表中选择其他选项。

``` python
module_loader = tvm.micro.AutoTvmModuleLoader(
    template_project_dir=pathlib.Path(tvm.micro.get_microtvm_template_projects("crt")),
    project_options={"verbose": False},
)
builder = tvm.autotvm.LocalBuilder(
    n_parallel=1,
    build_kwargs={"build_option": {"tir.disable_vectorize": True}},
    do_fork=True,
    build_func=tvm.micro.autotvm_build_func,
    runtime=RUNTIME,
)
runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)

measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

# 为物理硬件编译
if use_physical_hw:
    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")),
        project_options={
            "board": BOARD,
            "verbose": False,
            "project_type": "host_driven",
            "serial_number": SERIAL,
        },
    )
    builder = tvm.autotvm.LocalBuilder(
        n_parallel=1,
        build_kwargs={"build_option": {"tir.disable_vectorize": True}},
        do_fork=False,
        build_func=tvm.micro.autotvm_build_func,
        runtime=RUNTIME,
    )
    runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)

    measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)
```

## 运行自动调优

下面在 microTVM 设备上对每个提取的任务单独运行自动调优。

``` python
autotune_log_file = pathlib.Path("microtvm_autotune.log.txt")
if os.path.exists(autotune_log_file):
    os.remove(autotune_log_file)

num_trials = 10
for task in tasks:
    tuner = tvm.autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=num_trials,
        measure_option=measure_option,
        callbacks=[
            tvm.autotvm.callback.log_to_file(str(autotune_log_file)),
            tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M"),
        ],
        si_prefix="M",
    )
```

## 为未调优的程序计时

为了方便比较，编译并运行不实施任何自动调优 schedule 的计算图，TVM 会为每个算子选择一个随机调优的实现，其性能不如调优的算子。

``` python
with pass_context:
    lowered = tvm.relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params)

temp_dir = tvm.contrib.utils.tempdir()
project = tvm.micro.generate_project(
    str(tvm.micro.get_microtvm_template_projects("crt")),
    lowered,
    temp_dir / "project",
    {"verbose": False},
)

# 为物理硬件编译
if use_physical_hw:
    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects("zephyr")),
        lowered,
        temp_dir / "project",
        {
            "board": BOARD,
            "verbose": False,
            "project_type": "host_driven",
            "serial_number": SERIAL,
            "config_main_stack_size": 4096,
        },
    )

project.build()
project.flash()
with tvm.micro.Session(project.transport()) as session:
    debug_module = tvm.micro.create_local_debug_executor(
        lowered.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered.get_params())
    print("########## Build without Autotuning ##########")
    debug_module.run()
    del debug_module
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
########## Build without Autotuning ##########
Node Name                                     Ops                                           Time(us)  Time(%)  Shape              Inputs  Outputs  Measurements(us)
---------                                     ---                                           --------  -------  -----              ------  -------  ----------------
tvmgen_default_fused_nn_contrib_conv2d_NCHWc  tvmgen_default_fused_nn_contrib_conv2d_NCHWc  310.0     98.73    (1, 2, 10, 10, 3)  2       1        [310.0]
tvmgen_default_fused_layout_transform_1       tvmgen_default_fused_layout_transform_1       3.031     0.965    (1, 6, 10, 10)     1       1        [3.031]
tvmgen_default_fused_layout_transform         tvmgen_default_fused_layout_transform         0.958     0.305    (1, 1, 10, 10, 3)  1       1        [0.958]
Total_time                                    -                                             313.988   -        -                  -       -        -
```

## 为调优程序计时

自动调优完成后，用 Debug Runtime 为整个程序的执行计时：

``` python
with tvm.autotvm.apply_history_best(str(autotune_log_file)):
    with pass_context:
        lowered_tuned = tvm.relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params)

temp_dir = tvm.contrib.utils.tempdir()
project = tvm.micro.generate_project(
    str(tvm.micro.get_microtvm_template_projects("crt")),
    lowered_tuned,
    temp_dir / "project",
    {"verbose": False},
)

# 为物理硬件编译
if use_physical_hw:
    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects("zephyr")),
        lowered_tuned,
        temp_dir / "project",
        {
            "board": BOARD,
            "west_cmd": "west",
            "verbose": False,
            "project_type": "host_driven",
            "serial_number": SERIAL,
            "config_main_stack_size": 4096,
        },
    )

project.build()
project.flash()
with tvm.micro.Session(project.transport()) as session:
    debug_module = tvm.micro.create_local_debug_executor(
        lowered_tuned.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered_tuned.get_params())
    print("########## Build with Autotuning ##########")
    debug_module.run()
    del debug_module
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
########## Build with Autotuning ##########
Node Name                                     Ops                                           Time(us)  Time(%)  Shape              Inputs  Outputs  Measurements(us)
---------                                     ---                                           --------  -------  -----              ------  -------  ----------------
tvmgen_default_fused_nn_contrib_conv2d_NCHWc  tvmgen_default_fused_nn_contrib_conv2d_NCHWc  193.2     98.657   (1, 6, 10, 10, 1)  2       1        [193.2]
tvmgen_default_fused_layout_transform_1       tvmgen_default_fused_layout_transform_1       1.778     0.908    (1, 6, 10, 10)     1       1        [1.778]
tvmgen_default_fused_layout_transform         tvmgen_default_fused_layout_transform         0.851     0.435    (1, 3, 10, 10, 1)  1       1        [0.851]
Total_time                                    -                                             195.83    -        -                  -       -        -
```

[下载 Python 源代码：micro_autotune.py](https://tvm.apache.org/docs/v0.13.0/_downloads/9ccca8fd489a1486ac71b55a55c320c5/micro_autotune.py)

[下载 Jupyter notebook：micro_autotune.ipynb](https://tvm.apache.org/docs/v0.13.0/_downloads/f83ba3df2d52f9b54cf141114359481a/micro_autotune.ipynb)