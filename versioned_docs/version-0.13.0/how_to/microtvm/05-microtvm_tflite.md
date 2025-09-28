---
title: 2. microTVM TFLite 指南
---

# 2. microTVM TFLite 指南

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_tflite.html#sphx-glr-download-how-to-work-with-microtvm-micro-tflite-py) 下载完整的示例代码
:::

**作者**：[Tom Gall](https://github.com/tom-gall)

本教程介绍如何用 microTVM 和支持 Relay 的 TFLite 模型。

## 安装 microTVM Python 依赖项
TVM 不包含用于 Python 串行通信包，因此在使用 microTVM 之前我们必须先安装一个。我们还需要TFLite来加载模型。

```bash
pip install pyserial==3.5 tflite==2.1
```

```python
import os


# 本指南默认运行在使用 TVM 的 C 运行时的 x86 CPU 上，如果你想
# 在 Zephyr 实机硬件上运行，你必须导入 `TVM_MICRO_USE_HW` 环境
# 变量。此外如果你使用 C 运行时，你可以跳过安装 Zephyr。
# 将花费大约20分钟安装 Zephyr。
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

```python
import json
import tarfile
import pathlib
import tempfile
import numpy as np

import tvm
import tvm.micro
import tvm.micro.testing
from tvm import relay
import tvm.contrib.utils
from tvm.micro import export_model_library_format
from tvm.contrib.download import download_testdata

model_url = (
    "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/model/sine_model.tflite"
)
model_file = "sine_model.tflite"
model_path = download_testdata(model_url, model_file, module="data")

tflite_model_buf = open(model_path, "rb").read()

```

使用 buffer，转换为 tflite 模型 python 对象：
```python
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

```

打印模型版本：

```python
version = tflite_model.Version()
print("Model Version: " + str(version))

```

输出结果：
```
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

当在与主机（ Python 脚本执行的位置）相同架构的模拟 target 上运行时，为 TARGET 选择下面的「crt」，选择 C Runtime  作为 RUNTIME ，并选择适当的单板/虚拟机来运行它（Zephyr 将创建基于 BOARD 的正确 QEMU 虚拟机）。

下面的示例中，选择 x86 架构并相应地选择 x86 虚拟机：

``` python
RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
TARGET = tvm.micro.testing.get_target("crt")

# 运行于物理硬件时，选择描述对应硬件的 TARGET 和 BOARD。
# 下面的示例选择 STM32L4R5ZI Nucleo target 和 board。你可以改变测试板，
# 只需要使用不同的 Zephyr 支持单板导入`TVM_MICRO_BOARD` 变量

if use_physical_hw:
    BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_l4r5zi")
    SERIAL = os.getenv("TVM_MICRO_SERIAL", default=None)
    TARGET = tvm.micro.testing.get_target("zephyr", BOARD)

#
# 对于某些单板，Zephyr 默认使用 QEMU 模拟运行，例如，下面
# TARGET 和 BOARD 用于为 mps2-an521 开发板构建 microTVM 固件。
#
# `mps2_an521 = "mps2_an521"`
# `TARGET = tvm.micro.testing.get_target("zephyr", BOARD)`
```

为 target 编译模型。如果你不需要执行器将默认使用图执行器。

``` python
with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=params)
```

# 检查编译输出
编译过程产生了一些计算图中实现算子的 C 代码。可以通过打印 CSourceModule 内容来检查它（本教程只打印前 10 行）：

```python
c_source_module = module.get_lib().imported_modules[0]
assert c_source_module.type_key == "c", "tutorial is broken"

c_source_code = c_source_module.get_source()
first_few_lines = c_source_code.split("\n")[:10]
assert any(
    l.startswith("TVM_DLL int32_t tvmgen_default_") for l in first_few_lines
), f"tutorial is broken: {first_few_lines!r}"
print("\n".join(first_few_lines))
```

# 编译生成的代码
下面需要将生成的 C 代码合并到一个项目中，以便在设备中运行推理。最简单的方法是自己集成，使用 microTVM 的标准输出格式化模型库格式。这是标准布局的 tarball。

```python
# 获取可以存储 tarball 的临时路径（作为教程运行）。

temp_dir = tvm.contrib.utils.tempdir()
model_tar_path = temp_dir / "model.tar"
export_model_library_format(module, model_tar_path)

with tarfile.open(model_tar_path, "r:*") as tar_f:
    print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))

# TVM 还为嵌入式平台提供了一个标准的方式来自动生成一个独立的
# 项目，编译并烧录到一个 target，使用标准的 TVM RPC 与它通信。
# 模型库格式用作此过程的模型输入。
# 平台为嵌入时提供了集成，可以被 TVM 直接用于主机驱动
# 推理和自动调优。这种集成由
# `microTVM 项目 API` [https://github.com/apache/tvm-rfcs/blob/main/rfcs/0008-microtvm-project-api.md](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0008-microtvm-project-api.md)_提供。
#
# 嵌入式平台需要提供一个包含 microTVM API Server 的模板项目（通常，
# 存在于根目录中的“microtvm_api_server.py”文件中）。本教程使用示例“主机”
# 项目（使用 POSIX 子进程和管道模拟设备）：

template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
project_options = {}  # 可以使用 TVM 提供特定于平台 options。

# 对于物理硬件，可以通过使用不同的模板项目来试用 Zephyr 平台
# 和选项：

if use_physical_hw:
    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
        project_options = {
        "project_type": "host_driven",
        "board": BOARD,
        "serial_number": SERIAL,
        "config_main_stack_size": 4096,
        "zephyr_base": os.getenv("ZEPHYR_BASE", default="/content/zephyrproject/zephyr"),
    }

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

[下载 Python 源代码：micro_tflite.py](https://tvm.apache.org/docs/v0.13.0/_downloads/2fb9ae7bf124f72614a43137cf2919cb/micro_tflite.py)

[下载 Jupyter Notebook：micro_tflite.ipynb](https://tvm.apache.org/docs/v0.13.0/_downloads/5b279d8a8718816263fa65b0eef1a5c0/micro_tflite.ipynb)
