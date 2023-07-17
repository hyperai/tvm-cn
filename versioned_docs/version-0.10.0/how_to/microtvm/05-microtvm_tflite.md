---
title: 支持 TFLite 模型的 microTVM
---

# 支持 TFLite 模型的 microTVM

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_tflite.html#sphx-glr-download-how-to-work-with-microtvm-micro-tflite-py) 下载完整的示例代码
:::

**作者**：[Tom Gall](https://github.com/tom-gall)

本教程介绍如何用 microTVM 和支持 Relay 的 TFLite 模型。

:::note
若要在 microTVM 虚拟机参考手册上运行本教程，请根据页面底部的链接下载 Jupyter Notebook 并将其保存到 TVM 目录中。然后：

1. 用修改后的 `vagrant ssh` 命令登录到虚拟机参考手册：
   `$ vagrant ssh -- -L8888:localhost:8888`
2. 安装 Jupyter： `pip install jupyterlab`
3. `cd` 到 TVM 目录
4. 安装 TFLite：`poetry install -E importer-tflite`
5. 启动 Jupyter Notebook：`jupyter notebook`
6. 复制 localhost URL，并将其粘贴到浏览器中
7. 导航到已保存的 Jupyter Notebook（`.ipynb` 文件）
:::

## 设置

### 安装 TFLite

开始前，先安装 TFLite 包，两种安装方式如下所示：

1. 使用 `pip` 安装

   ``` bash
   pip install tflite=2.1.0 --user
   ```

2. 生成 TFLite 包，步骤如下：

   获取 flatc 编译器，参阅 https://github.com/google/flatbuffers 了解详细信息，确保已正确安装。

   ``` bash
   flatc --version
   ```

   获取 TFLite schema。

   ``` bash
   wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.13/tensorflow/lite/schema/schema.fbs
   ```

   生成 TFLite 包。

   ``` bash
   flatc --python schema.fbs
   ```

   将当前文件夹（包含生成的 TFLite 模块）添加到 PYTHONPATH。

   ``` bash
   export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$(pwd)
   ```

用 `python -c "import tflite"` 验证 TFLite 包是否安装成功。

### 安装 Zephyr（仅限物理硬件）

用主机模拟运行本教程时（默认），使用主机 `gcc` 构建模拟设备的固件镜像。若编译到物理硬件上运行，需安装一个 *toolchain* 以及一些特定于 target 的依赖。microTVM 允许用户提供任何可以启动 TVM RPC 服务器的编译器和 runtime。开始之前，请注意本教程依赖于 Zephyr RTOS 来提供这些部分。

参考 [安装说明](https://docs.zephyrproject.org/latest/getting_started/index.html) 安装 Zephyr。

**题外话：重新创建预训练 TFLite 模型**

   本教程下载了预训练的 TFLite 模型。使用微控制器时，请注意这些设备的资源高度受限，像 MobileNet 这样的标准模型和小内存并不匹配。
   
   本教程使用 TF Micro 示例模型之一。
   
   若要复制训练步骤，参阅：https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world/train
   
   :::note
   若不小心从 `wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/micro/hello_world_2020_04_13.zip` 下载了示例预训练模型，会由于未实现的操作码（114）而失败。
   :::

## 加载并准备预训练模型

把预训练 TFLite 模型从当前目录中的文件中加载到 buffer

``` python
import os
import json
import tarfile
import pathlib
import tempfile
import numpy as np

import tvm
from tvm import relay
import tvm.contrib.utils
from tvm.contrib.download import download_testdata

use_physical_hw = bool(os.getenv("TVM_MICRO_USE_HW"))
model_url = "https://people.linaro.org/~tom.gall/sine_model.tflite"
model_file = "sine_model.tflite"
model_path = download_testdata(model_url, model_file, module="data")

tflite_model_buf = open(model_path, "rb").read()
```

利用 buffer，转换为 TFLite 模型 Python 对象

``` python
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
```

打印模型的版本

``` python
version = tflite_model.Version()
print("Model Version: " + str(version))
```

输出结果：

``` bash
Model Version: 3
```

解析 Python 模型对象，并转换为 Relay 模块和权重。注意输入张量的名称必须与模型中包含的内容相匹配。

若不确定，可通过 TensorFlow 项目中的 `visualize.py`  脚本来查看。参阅 [如何检查 .tflite 文件？](https://www.tensorflow.org/lite/guide/faq)

``` python
input_tensor = "dense_4_input"
input_shape = (1,)
input_dtype = "float32"

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)
```

## 定义 target

接下来为 Relay 创建一个构建配置，关闭两个选项，然后调用 relay.build，为选定的 TARGET 生成一个 C 源文件。

当在与主机（ Python 脚本执行的位置）相同架构的模拟 target 上运行时，为 TARGET 选择下面的「host」，选择 C Runtime  作为 RUNTIME ，并选择适当的单板/虚拟机来运行它（Zephyr 将创建基于 BOARD 的正确 QEMU 虚拟机）。

下面的示例中，选择 x86 架构并相应地选择 x86 虚拟机：

``` python
RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
TARGET = tvm.target.target.micro("host")

#
# 为物理硬件编译
# 在物理硬件上运行时，选择描述硬件的 TARGET 和 BOARD。
# 下面的示例中选择 STM32F746 Nucleo target 和单板。另一种选择是
# STM32F746 Discovery 板。由于该板具有与 Nucleo 相同的 MCU
# 板，但是一些接线和配置不同，选择「stm32f746g_disco」
# 板生成正确的固件镜像。
#

if use_physical_hw:
    boards_file = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")) / "boards.json"
    with open(boards_file) as f:
        boards = json.load(f)

    BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_f746zg")
    TARGET = tvm.target.target.micro(boards[BOARD]["model"])

#
# 对于某些单板，Zephyr 默认使用 QEMU 模拟运行，例如，下面
# TARGET 和 BOARD 用于为 mps2-an521 开发板构建 microTVM 固件。自从那块板
# 在 Zephyr 上默认模拟运行，板名称添加后缀「-qemu」通知
# microTVM 必须使用 QEMU 传输器与开发板通信。如果名称
# 已经有前缀「qemu_」，比如「qemu_x86」，就不需要加上那个后缀了。
#
#  TARGET = tvm.target.target.micro("mps2_an521")
#  BOARD = "mps2_an521-qemu"
```

为 target 编译模型：

``` python
with tvm.transform.PassContext(
    opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
):
    module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=params)

# 检查编译输出
# ---------------------------------
#
# 编译过程产生了一些计算图中实现算子的 C 代码。
# 可以通过打印 CSourceModule 内容来检查它（本教程
# 只打印前 10 行）：

c_source_module = module.get_lib().imported_modules[0]
assert c_source_module.type_key == "c", "tutorial is broken"

c_source_code = c_source_module.get_source()
first_few_lines = c_source_code.split("\n")[:10]
assert any(
    l.startswith("TVM_DLL int32_t tvmgen_default_") for l in first_few_lines
), f"tutorial is broken: {first_few_lines!r}"
print("\n".join(first_few_lines))

# 编译生成的代码
# ----------------------------
#
# 下面需要将生成的 C 代码合并到一个项目中，以便在
# 设备中运行推理。最简单的方法是自己集成，使用 microTVM 的标准输出格式
# (:doc:`模型库格式` `</dev/model_library_format>`)，这是具有标准布局的 tarball：

# 获取可以存储 tarball 的临时路径（作为教程运行）。

fd, model_library_format_tar_path = tempfile.mkstemp()
os.close(fd)
os.unlink(model_library_format_tar_path)
tvm.micro.export_model_library_format(module, model_library_format_tar_path)

with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
    print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))

# 清理：
os.unlink(model_library_format_tar_path)

# TVM 还为嵌入式平台提供了一个标准的方式来自动生成一个独立的
# 项目，编译并烧录到一个 target，使用标准的 TVM RPC 与它通信。
# 模型库格式用作此过程的模型输入。
# 平台为嵌入时提供了集成，可以被 TVM 直接用于主机驱动
# 推理和自动调优。这种集成由
# `microTVM 项目 API` <https://github.com/apache/tvm-rfcs/blob/main/rfcs/0008-microtvm-project-api.md>_提供。
#
# 嵌入式平台需要提供一个包含 microTVM API Server 的模板项目（通常，
# 存在于根目录中的“microtvm_api_server.py”文件中）。本教程使用示例“主机”
# 项目（使用 POSIX 子进程和管道模拟设备）：

template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
project_options = {}  # 可以使用 TVM 提供特定于平台 options。

# 编译物理硬件（或仿真板，如 mps_an521）
# --------------------------------------------------------------------------
# 对于物理硬件，可以通过使用不同的模板项目来试用 Zephyr 平台
# 和选项：
#

if use_physical_hw:
    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
    project_options = {"project_type": "host_driven", "zephyr_board": BOARD}

# 创建临时目录
temp_dir = tvm.contrib.utils.tempdir()
generated_project_dir = temp_dir / "generated-project"
generated_project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

# 构建并刷新项目
generated_project.build()
generated_project.flash()
```

输出结果：

``` c
// tvm target: c -keys=cpu -link-params=0 -model=host
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_add(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
 - .
 - ./codegen
 - ./codegen/host
 - ./codegen/host/src
 - ./codegen/host/src/default_lib0.c
 - ./codegen/host/src/default_lib1.c
 - ./codegen/host/src/default_lib2.c
 - ./executor-config
 - ./executor-config/graph
 - ./executor-config/graph/default.graph
 - ./metadata.json
 - ./parameters
 - ./parameters/default.params
 - ./src
 - ./src/default.relay
```

接下来，与模拟设备建立 session，并运行计算。*with session* 这一行通常会刷新连接的微控制器，但在本教程中，它只启动一个子进程来代替连接的微控制器。

``` python
with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
    graph_mod = tvm.micro.create_local_graph_executor(
        module.get_graph_json(), session.get_system_lib(), session.device
    )

    # 使用「relay.build」产生的降级参数设置模型参数。
    graph_mod.set_input(**module.get_params())

    # 模型使用单个 float32 值，并返回预测的正弦值。
    # 为传递输入值，我们构造一个带有单个构造值的 tvm.nd.array 对象作为输入。
    # 这个模型可接收的输入为 0 到 2Pi。
    graph_mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="float32")))
    graph_mod.run()

    tvm_output = graph_mod.get_output(0).numpy()
    print("result is: " + str(tvm_output))
```

输出结果：

``` bash
result is: [[0.4443792]]
```

[下载 Python 源代码：micro_tflite.py](https://tvm.apache.org/docs/_downloads/2fb9ae7bf124f72614a43137cf2919cb/micro_tflite.py)

[下载 Jupyter Notebook：micro_tflite.ipynb](https://tvm.apache.org/docs/_downloads/5b279d8a8718816263fa65b0eef1a5c0/micro_tflite.ipynb)