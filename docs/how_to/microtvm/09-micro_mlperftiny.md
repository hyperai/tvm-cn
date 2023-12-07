---
title: 创建使用 microTVM 的 MLPerfTiny 提交
---

:::note
单击 [此处](https://tvm.apache.org/docs/v0.13.0/how_to/work_with_microtvm/micro_mlperftiny.html#sphx-glr-download-how-to-work-with-microtvm-micro-mlperftiny-py) 下载完整的示例代码
:::


# 8.创建使用 microTVM 的 MLPerfTiny 提交

**作者：**[Mehrdad Hessar](https://github.com/mehrdadh)

本教程展示了如何使用 microTVM 构建 MLPerfTiny 提交。该教程演示了从 MLPerfTiny 基准模型中导入一个 TFLite 模型，使用 TVM 进行编译，并生成一个可以刷写到支持 Zephyr 的板上的 Zephyr 项目，以使用 EEMBC runner 对模型进行基准测试的步骤。

## 安装 microTVM Python 依赖项
TVM 不包含用于 Python 串行通信的软件包，因此在使用 microTVM 之前，我们必须安装它。我们还需要 TFLite 来加载模型。

```bash
pip install pyserial==3.5 tflite==2.1
```

```python
import os
import pathlib
import tarfile
import tempfile
import shutil
```

## 安装 Zephyr
```bash
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

# 安装 Python 依赖项
python3 -m pip install -r "${ZEPHYR_BASE}/scripts/requirements.txt"

```

**注意**：仅在您有意使用 CMSIS-NN 代码生成器生成此提交时安装 CMSIS-NN。

## 安装 Install CMSIS-NN
```bash
CMSIS_SHA="51263182d16c92649a48144ba56c0945f9fce60e"
CMSIS_URL="http://github.com/ARM-software/CMSIS_5/archive/${CMSIS_SHA}.tar.gz"
export CMSIS_PATH=/content/cmsis
DOWNLOAD_PATH="/content/${CMSIS_SHA}.tar.gz"
mkdir ${CMSIS_PATH}
wget ${CMSIS_URL} -O "${DOWNLOAD_PATH}"
tar -xf "${DOWNLOAD_PATH}" -C ${CMSIS_PATH} --strip-components=1
rm ${DOWNLOAD_PATH}

CMSIS_NN_TAG="v4.0.0"
CMSIS_NN_URL="https://github.com/ARM-software/CMSIS-NN.git"
git clone ${CMSIS_NN_URL} --branch ${CMSIS_NN_TAG} --single-branch ${CMSIS_PATH}/CMSIS-NN

```

## 导入 Python 依赖
```python
import tensorflow as tf
import numpy as np

import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata
from tvm.micro import export_model_library_format
import tvm.micro.testing
from tvm.micro.testing.utils import (
    create_header_file,
    mlf_extract_workspace_size_bytes,
)

```

## 导入 Visual Wake Word Model
首先，从 MLPerfTiny 下载并导入 Visual Wake Word (VWW) TFLite 模型。该模型最初来自 [MLPerf Tiny 仓库](https://github.com/mlcommons/tiny)。我们还捕获了来自 TFLite 模型的元数据信息，如输入/输出名称、量化参数等，这些信息将在接下来的步骤中使用。

我们使用索引来构建各种模型的提交。索引定义如下：要构建另一个模型，您需要更新模型 URL、简短名称和索引号。

关键词识别（KWS）1

视觉唤醒词（VWW）2

异常检测（AD）3

图像分类（IC）4

如果您想要使用 CMSIS-NN 构建提交，请修改 USE_CMSIS 环境变量。

```bash
export USE_CMSIS=1
```

```python
MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"
MODEL_PATH = download_testdata(MODEL_URL, "vww_96_int8.tflite", module="model")

MODEL_SHORT_NAME = "VWW"
MODEL_INDEX = 2

USE_CMSIS = os.environ.get("TVM_USE_CMSIS", False)

tflite_model_buf = open(MODEL_PATH, "rb").read()
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_name = input_details[0]["name"]
input_shape = tuple(input_details[0]["shape"])
input_dtype = np.dtype(input_details[0]["dtype"]).name
output_name = output_details[0]["name"]
output_shape = tuple(output_details[0]["shape"])
output_dtype = np.dtype(output_details[0]["dtype"]).name

# 从 TFLite 模型中提取量化信息。
# 除了异常检测模型外，所有其他模型都需要这样做，
# 因为对于其他模型，我们从主机发送量化数据到解释器，
# 然而，对于异常检测模型，我们发送浮点数据，量化信息
# 在微控制器上进行。
if MODEL_SHORT_NAME != "AD":
    quant_output_scale = output_details[0]["quantization_parameters"]["scales"][0]
    quant_output_zero_point = output_details[0]["quantization_parameters"]["zero_points"][0]

relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_name: input_shape}, dtype_dict={input_name: input_dtype}
)

```

# 定义目标、运行时和执行器
现在我们需要定义目标、运行时和执行器来编译这个模型。在本教程中，我们使用预先编译（Ahead-of-Time，AoT）进行编译，并构建一个独立的项目。这与使用主机驱动模式的 AoT 不同，其中目标会使用主机驱动的 AoT 执行器与主机通信以运行推理。

```python
# 使用 C 运行时 (crt)
RUNTIME = Runtime("crt")

# 使用带有 `unpacked-api=True` 和 `interface-api=c` 的 AoT 执行器。`interface-api=c` 强制
# 编译器生成 C 类型的函数 API，而 `unpacked-api=True` 强制编译器生成最小的未打包格式输入，
# 这减少了调用模型推理层时的堆栈内存使用。
EXECUTOR = Executor(
    "aot",
    {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 8},
)

# 选择一个 Zephyr 板
BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_l4r5zi")

# 使用 BOARD 获取完整的目标描述
TARGET = tvm.micro.testing.get_target("zephyr", BOARD)

```


# 编译模型并导出模型库格式
现在，我们为目标编译模型。然后，我们为编译后的模型生成模型库格式。我们还需要计算编译后的模型所需的工作空间大小。
```python
config = {"tir.disable_vectorize": True}
if USE_CMSIS:
    from tvm.relay.op.contrib import cmsisnn

    config["relay.ext.cmsisnn.options"] = {"mcpu": TARGET.mcpu}
    relay_mod = cmsisnn.partition_for_cmsisnn(relay_mod, params, mcpu=TARGET.mcpu)

with tvm.transform.PassContext(opt_level=3, config=config):
    module = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )

temp_dir = tvm.contrib.utils.tempdir()
model_tar_path = temp_dir / "model.tar"
export_model_library_format(module, model_tar_path)
workspace_size = mlf_extract_workspace_size_bytes(model_tar_path)

```
# 生成输入/输出头文件
为了使用 AoT 创建 microTVM 独立项目，我们需要生成输入和输出头文件。这些头文件用于将生成的代码中的输入和输出 API 与独立项目的其余部分连接起来。对于此特定提交，我们只需要生成输出头文件，因为输入 API 调用是以不同的方式处理的。

```python
extra_tar_dir = tvm.contrib.utils.tempdir()
extra_tar_file = extra_tar_dir / "extra.tar"

with tarfile.open(extra_tar_file, "w:gz") as tf:
    create_header_file(
        "output_data",
        np.zeros(
            shape=output_shape,
            dtype=output_dtype,
        ),
        "include/tvm",
        tf,
    )

```

# 创建项目、构建并准备项目 tar 文件
现在我们有了编译后的模型作为模型库格式，可以使用 Zephyr 模板项目生成完整的项目。首先，我们准备项目选项，然后构建项目。最后，我们清理临时文件并将提交项目移动到当前工作目录，可以在开发套件上下载并使用。

```python
input_total_size = 1
for i in range(len(input_shape)):
    input_total_size *= input_shape[i]

template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
project_options = {
    "extra_files_tar": str(extra_tar_file),
    "project_type": "mlperftiny",
    "board": BOARD,
    "compile_definitions": [
        f"-DWORKSPACE_SIZE={workspace_size + 512}",  # Memory workspace size, 512 is a temporary offset
        # since the memory calculation is not accurate.
        f"-DTARGET_MODEL={MODEL_INDEX}",  # Sets the model index for project compilation.
        f"-DTH_MODEL_VERSION=EE_MODEL_VERSION_{MODEL_SHORT_NAME}01",  # Sets model version. This is required by MLPerfTiny API.
        f"-DMAX_DB_INPUT_SIZE={input_total_size}",  # Max size of the input data array.
    ],
}

if MODEL_SHORT_NAME != "AD":
    project_options["compile_definitions"].append(f"-DOUT_QUANT_SCALE={quant_output_scale}")
    project_options["compile_definitions"].append(f"-DOUT_QUANT_ZERO={quant_output_zero_point}")

if USE_CMSIS:
    project_options["compile_definitions"].append(f"-DCOMPILE_WITH_CMSISNN=1")

# 注意：根据您使用的板子可能需要调整这个值。
project_options["config_main_stack_size"] = 4000

if USE_CMSIS:
    project_options["cmsis_path"] = os.environ.get("CMSIS_PATH", "/content/cmsis")

generated_project_dir = temp_dir / "project"

project = tvm.micro.project.generate_project_from_mlf(
    template_project_path, generated_project_dir, model_tar_path, project_options
)
project.build()

# 清理构建目录和额外的工件
shutil.rmtree(generated_project_dir / "build")
(generated_project_dir / "model.tar").unlink()

project_tar_path = pathlib.Path(os.getcwd()) / "project.tar"
with tarfile.open(project_tar_path, "w:tar") as tar:
    tar.add(generated_project_dir, arcname=os.path.basename("project"))

print(f"The generated project is located here: {project_tar_path}")
```

# 使用此项目与您的板子
既然我们有了生成的项目，您可以在本地使用该项目将板子刷写并准备好运行 EEMBC runner 软件。要执行此操作，请按照以下步骤操作：

```bash
tar -xf project.tar
cd project
mkdir build
cmake ..
make -j2
west flash

```

现在，您可以按照这些说明将您的板子连接到 EEMBC runner 并在您的板子上对此模型进行基准测试。