---
title: 3. microTVM AoT 编译
---

# 3.microTVM AoT 编译

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_aot.html#sphx-glr-download-how-to-work-with-microtvm-micro-aot-py) 下载完整的示例代码
:::

**作者**：[Mehrdad Hessar](https://github.com/mehrdadh)，[Alan MacDonald](https://github.com/alanmacd)

本教程展示了 microTVM（使用 TFLite 模型）主机驱动的 AoT 编译。与 GraphExecutor 相比，AoTExecutor 减少了运行时解析图的开销。此外，我们可以通过提前编译更好地进行内存管理。本教程可以使用 C 运行时（CRT）在 x86 CPU 上执行，也可以在 Zephyr 支持的微控制器/板上的 Zephyr 平台上执行。

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

``` python
import numpy as np
import pathlib
import json

import tvm
from tvm import relay
import tvm.micro.testing
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata
```

## 导入 TFLite 模型

首先，下载并导入 Keyword Spotting TFLite 模型。该模型最初来自 [MLPerf Tiny 仓库](https://github.com/mlcommons/tiny)。使用 [Google 提供的 KWS 数据集的样本](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) 来测试这个模型。

**注意：**默认情况下，本教程使用 CRT 在 x86 CPU 上运行，若要在 Zephyr 平台上运行，需要导出 *TVM_MICRO_USE_HW* 环境变量。

``` python
MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite"
MODEL_PATH = download_testdata(MODEL_URL, "kws_ref_model.tflite", module="model")
SAMPLE_URL = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy"
SAMPLE_PATH = download_testdata(SAMPLE_URL, "keyword_spotting_int8_6.pyc.npy", module="data")

tflite_model_buf = open(MODEL_PATH, "rb").read()
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

input_shape = (1, 49, 10, 1)
INPUT_NAME = "input_1"
relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={INPUT_NAME: input_shape}, dtype_dict={INPUT_NAME: "int8"}
)
```

## 定义 target

接下来定义 target、runtime 和 executor。本教程将详细介绍使用 AOT 主机驱动的执行器。这里使用的主机微 target，它使用 CRT runtime 在 x86 CPU 上运行模型，或者在 qemu_x86 模拟器单板上运行带有 Zephyr 平台的模型。对于物理微控制器，获取物理单板（例如 nucleo_l4r5zi）的 target 模型，并将其修改 `BOARD` 支持 Zephyr 单板。

``` python
# 使用 C runtime（crt），并通过将 system-lib 设置为 True 来启用静态链接
RUNTIME = Runtime("crt", {"system-lib": True})

# 在主机上模拟一个微控制器。使用来自 `src/runtime/crt/host/main.cc`_ 的 main()。
# 若要使用物理硬件，请将「host」替换为与你的硬件匹配的内容。
TARGET = tvm.micro.testing.get_target("crt")

# 使用 AOT 执行器，而非计算图或是虚拟机执行器。不要使用未打包的 API 或 C 调用风格。
EXECUTOR = Executor("aot")

if use_physical_hw:
    BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_l4r5zi")
    SERIAL = os.getenv("TVM_MICRO_SERIAL", default=None)
    TARGET = tvm.micro.testing.get_target("zephyr", BOARD)
```

## 编译模型

接下来为 target 编译模型：

``` python
with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    module = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:267: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 创建一个 microTVM 项目

将编译好的模型作为 IRModule，然后创建一个固件项目，将编译好的模型与 microTVM 配合使用。可以借助 Project API 来实现。我们定义了 CRT 和 Zephyr microTVM 模板项目，分别用于 x86 CPU 和 Zephyr 板。

``` python
template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
project_options = {}  # 可以用选项通过 TVM 提供特定于平台的选项。

if use_physical_hw:
    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
    project_options = {
        "project_type": "host_driven",
        "board": BOARD,
        "serial_number": SERIAL,
        "config_main_stack_size": 4096,
        "zephyr_base": os.getenv("ZEPHYR_BASE", default="/content/zephyrproject/zephyr"),
    }

temp_dir = tvm.contrib.utils.tempdir()
generated_project_dir = temp_dir / "project"
project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)
```

## 构建、烧录和执行模型

接下来构建 microTVM 项目，并将其烧录。Flash 步骤是特定于物理微控制器的。若它通过主机 main.cc 来模拟微控制器，或是将 Zephyr 仿真板作为 target，则会跳过该步骤。接下来，为模型输出定义标签，并使用期望值为 6 的样本（label: left）执行模型。

``` python
project.build()
project.flash()

labels = [
    "_silence_",
    "_unknown_",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]
with tvm.micro.Session(project.transport()) as session:
    aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
    sample = np.load(SAMPLE_PATH)
    aot_executor.get_input(INPUT_NAME).copyfrom(sample)
    aot_executor.run()
    result = aot_executor.get_output(0).numpy()
    print(f"Label is `{labels[np.argmax(result)]}` with index `{np.argmax(result)}`")
#
# 输出：
# label 为 `left`，其 index 为 `6`
#
```

输出结果：

``` bash
Label is `left` with index `6`
```

[下载 Python 源代码：micro_aot.py](https://tvm.apache.org/docs/v0.13.0/_downloads/f8a7209a0e66b246185bfc41bbc82f54/micro_aot.py)

[下载 Jupyter Notebook：micro_aot.ipynb](https://tvm.apache.org/docs/v0.13.0/_downloads/c00933f3fbcf90c4f584d54607b33805/micro_aot.ipynb)
