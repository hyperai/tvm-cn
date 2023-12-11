---
title: 9. 开发环境中加入 microTVM
---

:::note
单击 [此处](https://tvm.apache.org/docs/v0.13.0/how_to/work_with_microtvm/micro_custom_ide.html#sphx-glr-download-how-to-work-with-microtvm-micro-custom-ide-py) 下载完整的示例代码
:::

# 9. 开发环境中加入 microTVM
作者：[Mohamad Katanbaf](https://github.com/mkatanbaf)

本教程描述了将使用 microTVM 编译的模型集成到自定义开发环境所需的步骤。在本教程中，我们使用 [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) 作为目标集成开发环境（IDE），但我们不依赖于此 IDE 的任何特定功能，将 microTVM 集成到其他 IDE 中的步骤类似。在这里，我们还使用了 MLPerf Tiny 的 Visual Wake Word（VWW）模型和 nucleo_l4r5zi 开发板，但相同的步骤也适用于任何其他模型或目标微控制器单元（MCU）。如果您希望在 vww 模型上使用另一个目标 MCU，我们建议选择具有约 512 KB 和约 256 KB 闪存和 RAM 的 Cortex-M4 或 Cortex-M7 设备。

以下是本教程中要执行的步骤的简要概述。

1. 首先，我们导入模型，使用 TVM 进行编译，并生成包含模型生成代码以及所有所需 TVM 依赖项的 [Model Library Format](https://tvm.apache.org/docs/arch/model_library_format.html)（MLF）tar 文件。

2. 我们还将两个二进制格式的样本图像（一个人和一个非人样本）添加到 .tar 文件中，以用于评估模型。

3. 接下来，我们使用 stmCubeMX 生成在 stmCube IDE 中项目的初始化代码。

4. 然后，我们将我们的 MLF 文件和所需的 CMSIS 库包含到项目中并进行构建。

5. 最后，我们烧写设备并在我们的样本图像上评估模型性能。

让我们开始吧。


# 安装 microTVM Python 依赖项
TVM 不包含用于 Python 串行通信的包，因此在使用 microTVM 之前，我们必须安装一个。我们还需要 TFLite 以加载模型，以及 Pillow 以准备样本图像。
```bash
pip install pyserial==3.5 tflite==2.1 Pillow==9.0 typing_extensions
```

# 导入 Python 依赖项
如果要在本地运行此脚本，请查看 [TVM 在线文档](https://tvm.apache.org/docs/install/index.html)，了解安装 TVM 的说明。

```python
import os
import numpy as np
import pathlib
import json
from PIL import Image
import tarfile

import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata
from tvm.micro import export_model_library_format
from tvm.relay.op.contrib import cmsisnn
from tvm.micro.testing.utils import create_header_file
```

# 导入 TFLite 模型
首先，下载并导入 Visual Wake Word TFLite 模型。该模型接受一个 96x96x3 的 RGB 图像，并确定图像中是否存在人物。此模型最初来自[ MLPerf Tiny 仓库](https://github.com/mlcommons/tiny)。为了测试该模型，我们使用 [COCO 2014 Train images](https://cocodataset.org/) 中的两个样本。

```python
MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"
MODEL_NAME = "vww_96_int8.tflite"
MODEL_PATH = download_testdata(MODEL_URL, MODEL_NAME, module="model")

tflite_model_buf = open(MODEL_PATH, "rb").read()
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

input_shape = (1, 96, 96, 3)
INPUT_NAME = "input_1_int8"
relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={INPUT_NAME: input_shape}, dtype_dict={INPUT_NAME: "int8"}
)
```

# 生成模型库格式文件
首先，我们定义目标、运行时和执行器。然后，我们为目标设备编译模型，最后导出生成的代码和所有必需的依赖项到单个文件中。

```python
# 我们可以使用 TVM 的本地调度或依赖于 CMSIS-NN 内核，使用 TVM 的 Bring-Your-Own-Code (BYOC) 能力。
USE_CMSIS_NN = True

# USMP (Unified Static Memory Planning) 对所有张量进行综合内存规划，以实现最佳内存利用。
DISABLE_USMP = False

# 使用 C 运行时（crt）
RUNTIME = Runtime("crt")

# 我们通过将板名称传递给 `tvm.target.target.micro` 来定义目标。
# 如果您的板型未包含在支持的模型中，您可以定义目标，如下所示：
# TARGET = tvm.target.Target("c -keys=arm_cpu,cpu -mcpu=cortex-m4")
TARGET = tvm.target.target.micro("stm32l4r5zi")

# 使用 AOT 执行器而不是图形或虚拟机执行器。使用未打包的 API 和 C 调用风格。
EXECUTOR = tvm.relay.backend.Executor(
    "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 8}
)

# 现在，我们设置编译配置并为目标编译模型：
config = {"tir.disable_vectorize": True}
if USE_CMSIS_NN:
    config["relay.ext.cmsisnn.options"] = {"mcpu": TARGET.mcpu}
if DISABLE_USMP:
    config["tir.usmp.enable"] = False

with tvm.transform.PassContext(opt_level=3, config=config):
    if USE_CMSIS_NN:
        # 当我们使用 CMSIS-NN 时，TVM 在 relay 图中搜索可以转移到 CMSIS-NN 内核的模式。
        relay_mod = cmsisnn.partition_for_cmsisnn(relay_mod, params, mcpu=TARGET.mcpu)
    lowered = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )
parameter_size = len(tvm.runtime.save_param_dict(lowered.get_params()))
print(f"Model parameter size: {parameter_size}")

# 我们需要选择一个目录来保存我们的文件。
# 如果在 Google Colab 上运行，我们将保存所有内容在 ``/root/tutorial`` 中（也就是 ``~/tutorial``），
# 但是如果在本地运行，您可能希望将其存储在其他位置。

BUILD_DIR = pathlib.Path("/root/tutorial")

BUILD_DIR.mkdir(exist_ok=True)

# 现在，我们将模型导出为一个 tar 文件：
TAR_PATH = pathlib.Path(BUILD_DIR) / "model.tar"
export_model_library_format(lowered, TAR_PATH)
```

输出：
```
Model parameter size: 32

PosixPath('/workspace/gallery/how_to/work_with_microtvm/tutorial/model.tar')
```


## 将样本图像添加到 MLF 文件中
最后，我们下载两个样本图像（一个人图像和一个非人图像），将它们转换为二进制格式，并存储在两个头文件中。

```python
with tarfile.open(TAR_PATH, mode="a") as tar_file:
    SAMPLES_DIR = "samples"
    SAMPLE_PERSON_URL = (
        "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/vww_sample_person.jpg"
    )
    SAMPLE_NOT_PERSON_URL = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/vww_sample_not_person.jpg"

    SAMPLE_PERSON_PATH = download_testdata(SAMPLE_PERSON_URL, "person.jpg", module=SAMPLES_DIR)
    img = Image.open(SAMPLE_PERSON_PATH)
    create_header_file("sample_person", np.asarray(img), SAMPLES_DIR, tar_file)

    SAMPLE_NOT_PERSON_PATH = download_testdata(
        SAMPLE_NOT_PERSON_URL, "not_person.jpg", module=SAMPLES_DIR
    )
    img = Image.open(SAMPLE_NOT_PERSON_PATH)
    create_header_file("sample_not_person", np.asarray(img), SAMPLES_DIR, tar_file)
```

在这一点上，您已经具备将编译后的模型导入到您的 IDE 并进行评估所需的一切。在 MLF 文件（model.tar）中，您应该找到以下文件层次结构：

```
/root
├── codegen
├── parameters
├── runtime
├── samples
├── src
├── templates
├── metadata.json
```

* codegen 文件夹：包含了由 TVM 为您的模型生成的 C 代码。
* runtime 文件夹：包含了目标需要编译生成的 C 代码所需的所有 TVM 依赖项。
* samples 文件夹：包含了用于评估模型的两个生成的样本文件。
* src 文件夹：包含了描述模型的 relay 模块。
* templates 文件夹：包含了两个模板文件，根据您的平台可能需要进行编辑。
* metadata.json 文件：包含有关模型、其层次和内存需求的信息。

## 生成在您的 IDE 中的项目
下一步是为目标设备创建一个项目。我们使用 STM32CubeIDE，您可以在[此处](https://www.st.com/en/development-tools/stm32cubeide.html)下载。在本教程中，我们使用的是版本 1.11.0。安装 STM32CubeIDE 后，请按照以下步骤创建项目：

1. 选择 File -> New -> STM32Project。目标选择窗口将出现。
2. 转到 “Board Selector” 选项卡，在 “Commercial Part Number” 文本框中键入板名称 “nucleo-l4r5zi”。从右侧显示的板列表中选择板，并单击 “Next”。
3. 输入项目名称（例如 microtvm_vww_demo）。我们使用默认选项（目标语言：C，二进制类型：可执行文件，项目类型：STM32Cube）。单击 “Finish”。
4. 一个文本框将出现，询问是否要 “以默认模式初始化所有外设？”。点击 “Yes”。这将生成项目并打开设备配置工具，您可以使用 GUI 设置外设。默认情况下启用了 USB、USART3 和 LPUART1，以及一些 GPIO。
5. 我们将使用 LPUART1 将数据发送到主机 PC。从连接部分中选择 LPUART1，并将 “Baud Rate” 设置为 115200，将 “Word Length” 设置为 8。保存更改并点击 “Yes” 以重新生成初始化代码。这应该会重新生成代码并打开您的 main.c 文件。您还可以从左侧的 Project Explorer 面板中找到 main.c，在 microtvm_vww_demo -> Core -> Src 下。
6. 为了进行健全性检查，请复制下面的代码并将其粘贴到主函数的无线循环（即 While(1) ）部分。
   * 注意：确保您的代码写在由 USER CODE BEGIN\<...\> 和 USER CODE END\<...\> 包围的部分内。如果重新生成初始化代码，被包围之外的代码将被擦除。


 ```
HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
HAL_UART_Transmit(&hlpuart1, "Hello World.\r\n", 14, 100);
HAL_Delay(1000);
 ```

7. 从菜单栏中选择 Project -> Build（或右键单击项目名称并选择 Build）。这将构建项目并生成 .elf 文件。选择 Run -> Run 以将二进制文件下载到您的 MCU。如果打开了“Edit Configuration”窗口，请直接点击 “OK”。

8. 在主机机器上打开终端控制台。在 Mac 上，您可以简单地使用 “screen \<usb_device\> 115200” 命令，例如 “screen tty.usbmodemXXXX 115200” 。板上的 LED 应该会闪烁，终端控制台上每秒应该会打印出字符串 “Hello World.”。按 “Control-a k” 退出 screen。

## 将模型导入生成的项目
要将编译后的模型集成到生成的项目中，请按照以下步骤操作：

1. 解压 tar 文件并将其包含在项目中
   * 打开项目属性（右键单击项目名称并选择 “Properties” 或从菜单栏选择 Project -> Properties）。
   * 选择 C/C++ General -> Paths and Symbols。选择 Source Location 选项卡。
   * 如果您将模型解压缩在项目文件夹内，请点击 “Add Folder” 并选择 “model” 文件夹（在它出现之前，您可能需要右键单击项目名称并选择 “Refresh”）。
   * 如果您在其他地方解压缩了模型文件，请点击 “Link Folder” 按钮，在出现的窗口中选中 “Link to folder in the file system” 复选框，点击 “Browse” 并选择模型文件夹。


2. 如果在编译模型时使用了 CMSIS-NN，您还需要在项目中包含 CMSIS-NN 源文件。
    * 从 [CMSIS-NN 存储库](https://github.com/ARM-software/CMSIS_5)下载或克隆文件，并按照上述步骤将 CMSIS-NN 文件夹包含在项目中。


3. 打开项目属性。在 C/C++ Build -> Settings 中：通过点击 “+” 按钮，选择 “Workspace” ，并导航到以下各个文件夹。将以下文件夹添加到 MCU GCC Compiler 的 Include Paths 列表中（如果是 C++ 项目还需添加到 MCU G++ Compiler 中）：

   - model/runtime/include
   - model/codegen/host/include
   - model/samples
   - CMSIS-NN/Include

4.  从 model/templates 复制 crt_config.h.template 到 Core/Inc 文件夹，并将其重命名为 crt_config.h。
5.  从 model/templates 复制 platform.c.template 到 Core/Src 文件夹，并将其重命名为 platform.c。
    - 此文件包含您可能需要根据平台编辑的内存管理函数。
    - 在 platform.c 中定义 “TVM_WORKSPACE_SIZE_BYTES” 的值。如果使用 USMP，则只需要比较小的值（例如 1024 字节）即可。
    - 如果不使用 USMP，请查看 metadata.json 中的 “workspace_size_bytes” 字段以估算所需内存。

6.  从构建中排除以下文件夹（右键单击文件夹名称，选择 Resource Configuration → Exclude from build）。检查 Debug 和 Release 配置。

    - CMSIS_NN/Tests

7.  从 [CMSIS Version 5 存储库](https://github.com/ARM-software/CMSIS_5)下载 CMSIS 驱动程序。
    - 在项目目录中，删除 Drivers/CMSIS/Include 文件夹（这是 CMSIS 驱动程序的旧版本），并将您从下载的版本中复制的 CMSIS/Core/Include 粘贴到相同位置。


8.  编辑 main.c 文件：
 * 包含下列头文件

```c
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "tvmgen_default.h"
#include "sample_person.h"
#include "sample_not_person.h"
```

 * 在 main 函数的无限循环前复制下面这段代码。该代码设置模型的输入和输出

```c
TVMPlatformInitialize();
signed char output[2];
struct tvmgen_default_inputs inputs = {
.input_1_int8 = (void*)&sample_person,
};
struct tvmgen_default_outputs outputs = {
.Identity_int8 = (void*)&output,
};
char msg[] = "Evaluating VWW model using microTVM:\r\n";
HAL_UART_Transmit(&hlpuart1, msg, strlen(msg), 100);
uint8_t sample = 0;
uint32_t timer_val;
char buf[50];
uint16_t buf_len;
```

  * 将以下代码复制到无限循环中。该代码将在图片上运行推断并在控制台打印结果。

```c
if (sample == 0)
    inputs.input_1_int8 = (void*)&sample_person;
else
    inputs.input_1_int8 = (void*)&sample_not_person;

timer_val = HAL_GetTick();
tvmgen_default_run(&inputs, &outputs);
timer_val = HAL_GetTick() - timer_val;
if (output[0] > output[1])
    buf_len = sprintf(buf, "Person not detected, inference time = %lu ms\r\n", timer_val);
else
    buf_len = sprintf(buf, "Person detected, inference time = %lu ms\r\n", timer_val);
HAL_UART_Transmit(&hlpuart1, buf, buf_len, 100);

sample++;
if (sample == 2)
    sample = 0;

```

 * 在 main 中定义 TVMLogf 函数，接受 TVM 运行时在控制台的报错

``` c
void TVMLogf(const char* msg, ...) {
  char buffer[128];
  int size;
  va_list args;
  va_start(args, msg);
  size = TVMPlatformFormatMessage(buffer, 128, msg, args);
  va_end(args);
  HAL_UART_Transmit(&hlpuart1, buffer, size, 100);
}

```

9. 在项目属性中，找到  C/C++ Build -> Settings, MCU GCC Compiler -> Optimization，设置 Optimization 为 Optimize more (-O2)。


## 评估模型
现在，选择菜单栏中的 Run -> Run 来刷写 MCU 并运行项目。您应该看到 LED 在闪烁，并且控制台上在打印推理结果。

[下载 Python 源代码：micro_custom_ide.py](https://tvm.apache.org/docs/v0.13.0/_downloads/9322c6c215567e9975d1df6b3a218ff1/micro_custom_ide.py)

[下载 Jupyter notebook：micro_custom_ide.ipynb](https://tvm.apache.org/docs/v0.13.0/_downloads/a74627f44186b95116fe0ed6f77e3b99/micro_custom_ide.ipynb)