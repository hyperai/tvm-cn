---
title: 在支持 CMSIS-NN 的 Arm(R) Cortex(R)-M55 CPU 和 Ethos(TM)-U55 NPU 裸机上运行 TVM
---

# 在支持 CMSIS-NN 的 Arm(R) Cortex(R)-M55 CPU 和 Ethos(TM)-U55 NPU 裸机上运行 TVM

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_ethosu.html#sphx-glr-download-how-to-work-with-microtvm-micro-ethosu-py) 下载完整的示例代码
:::

**作者**：[Grant Watson](https://github.com/grant-arm)

本节使用示例说明如何使用 TVM 在带有 CMSIS-NN 的 Arm(R) Cortex(R)-M55 CPU 和 Ethos(TM)-U55 NPU 的裸机上运行模型。Cortex(R)-M55 是一款用于嵌入式设备的小型低功耗 CPU。CMSIS-NN 是针对 Arm(R) Cortex(R)-M CPU 优化的内核集合。Ethos(TM)-U55 是一种 microNPU，专门为在资源受限的嵌入式设备中，加速机器学习的推理而设计。

在没有 Cortex(R)-M55 和 Ethos(TM)-U55 开发板的情况下，若要运行 demo 程序，可在固定虚拟平台（FVP）上运行。基于 Arm(R) Corstone(TM)-300 软件的 FVP，对包含 Cortex(R)-M55 和 Ethos(TM)-U55 的硬件系统进行建模。它对于软件开发非常友好。

本教程将编译一个 MobileNet v1 模型并使用 TVM 尽可能将算子迁移到 Ethos(TM)-U55。

## 获取 TVM

为平台获取 TVM，请访问 https://tlcpack.ai/ 并按照说明进行操作。正确安装 TVM 后，可以通过命令行访问 `tvmc`。

在命令行上键入 `tvmc` 应显示以下内容：

``` bash
usage: tvmc [-h] [-v] [--version] {tune,compile,run} ...

TVM compiler driver

optional arguments:
  -h, --help          show this help message and exit
  -v, --verbose       increase verbosity
  --version           print the version and exit

commands:
  {tune,compile,run}
    tune              auto-tune a model
    compile           compile a model.
    run               run a compiled module

TVMC - TVM driver command-line interface
```

## 安装额外的 Python 依赖

运行 demo 需要一些额外的 Python 包。通过下面的 requirements.txt 文件，来安装这些额外的 Python 包：

*requirements.txt*[¶](#requirements-txt)

``` text
 attrs==21.2.0
 cloudpickle==2.0.0
 decorator==5.1.0
 ethos-u-vela==3.2.0
 flatbuffers==1.12
 lxml==4.6.3
 nose==1.3.7
 numpy==1.19.5
 Pillow==8.3.2
 psutil==5.8.0
 scipy==1.5.4
 synr==0.4
 tflite==2.4.0
 tornado==6.1
```

运行以下命令来安装这些软件包：

``` bash
pip install -r requirements.txt
```

## 获取模型

本教程使用 MobileNet v1 模型（一种卷积神经网络，旨在对图像进行分类），该模型针对边缘设备进行了优化。要使用的模型已经预训练过，它可以可以识别 1001 个类别。该网络的输入图像大小为 224x224，任何输入图像在使用之前都需要调整到这个尺寸。

本教程使用 TFLite 格式的模型。

``` bash
mkdir -p ./build
cd build
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz
gunzip mobilenet_v1_1.0_224_quant.tgz
tar xvf mobilenet_v1_1.0_224_quant.tar
```

## 为具有 CMSIS-NN 的 Arm(R) Cortex(R)-M55 CPU 和 Ethos(TM)-U55 NPU 设备编译模型

下载 MobileNet v1 模型后，下一步是用 tvmc compile 进行编译。编译过程中得到的输出是模型的 TAR 包，该模型编译为 target  平台的模型库格式（MLF），能够用 TVM runtime 在 target 设备上运行该模型。

``` bash
tvmc compile --target=ethos-u,cmsis-nn,c \
             --target-ethos-u-accelerator_config=ethos-u55-256 \
             --target-cmsis-nn-mcpu=cortex-m55 \
             --target-c-mcpu=cortex-m55 \
             --runtime=crt \
             --executor=aot \
             --executor-aot-interface-api=c \
             --executor-aot-unpacked-api=1 \
             --pass-config tir.usmp.enable=1 \
             --pass-config tir.usmp.algorithm=hill_climb \
             --pass-config tir.disable_storage_rewrite=1 \
             --pass-config tir.disable_vectorize=1 \
             ./mobilenet_v1_1.0_224_quant.tflite \
             --output-format=mlf
```

:::note
tvmc 编译参数说明：

* `--target=ethos-u,cmsis-nn,c`：在可能的情况下将算子迁移到 microNPU，回退到 CMSIS-NN 并最终生成 microNPU 不支持的算子的 C 代码。
* `--target-ethos-u-accelerator_config=ethos-u55-256`：指定 microNPU 配置。
* `--target-c-mcpu=cortex-m55`：针对 Cortex(R)-M55 进行交叉编译。
* `--runtime=crt`：生成粘合代码以支持算子使用 C runtime。
* `--executor=aot`：使用 Ahead Of Time 编译而非 Graph Executor。
* `--executor-aot-interface-api=c`：生成一个 C 风格的接口，其结构专为在边界集成到 C 应用程序而设计。
* `--executor-aot-unpacked-api=1`：在内部使用非压缩的 API。
* `--pass-config tir.usmp.enable=1`：启用统一静态内存规划
* `--pass-config tir.usmp.algorithm=hill_climb`：对 USMP 使用爬山算法
* `--pass-config tir.disable_storage_rewrite=1`： 禁用存储重写
* `--pass-config tir.disable_vectorize=1`：禁用向量化，因为 C 中没有标准的向量化类型。
* `./mobilenet_v1_1.0_224_quant.tflite`：正在编译的 TFLite 模型。
* `--output-format=mlf`：以模型库格式生成输出。
:::

:::note
**若不想用 microNPU 进行迁移**

仅适用于 CMSIS-NN 的算子：

* 使用 `--target=cmsis-nn,c` 代替 `--target=ethos-u,cmsis-nn,c`
* 删除 microNPU 配置参数 `--target-ethos-u-accelerator_config=ethos-u55-256`
:::

## 将生成的代码解压到当前目录

``` bash
tar xvf module.tar
```

## 获取 ImageNet 标签

在图像上运行 MobileNet v1 时，输出结果是 0 到 1000 的索引。为了使应用程序对用户更加友好，将显示相关标签，而非仅显示类别索引。将图像标签下载到一个文本文件中，然后用 Python 脚本将它们包含在 C 程序中。

``` bash
curl -sS  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt \
-o ./labels_mobilenet_quant_v1_224.txt
```

## 获取输入图像

本教程使用猫的图像作为输入，也可以自行替换为其他图像。

![图片](https://s3.amazonaws.com/model-server/inputs/kitten.jpg)

将图像下载到构建目录中，下一步用 Python 脚本，将图像转换为 C 头文件中的字节数组。

``` bash
curl -sS https://s3.amazonaws.com/model-server/inputs/kitten.jpg -o ./kitten.jpg
```

## 预处理图像

以下脚本将在 src 目录中创建 2 个 C 头文件：

* `inputs.h` - 作为脚本参数提供的图像将被转换为整数数组，以输入到 MobileNet v1 模型。
* `outputs.h` - 用一个全零的整数数组，为推理的输出保留 1001 个整数值。

*convert_image.py*[¶](#convert-image-py)

``` python
#!python ./convert_image.py
import os
import pathlib
import re
import sys
from PIL import Image
import numpy as np

def create_header_file(name, section, tensor_name, tensor_data, output_path):
    """
    这个函数产生一个头文件，包含 numpy 数组提供的数据。
    """
    file_path = pathlib.Path(f"{output_path}/" + name).resolve()
    # 创建带有 npy_data 的头文件作为 C 数组
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write(
            "#include <tvmgen_default.h>\n"
            + f"const size_t {tensor_name}_len = {tensor_data.size};\n"
            + f'uint8_t {tensor_name}[] __attribute__((section("{section}"), aligned(16))) = "'
        )
        data_hexstr = tensor_data.tobytes().hex()
        for i in range(0, len(data_hexstr), 2):
            header_file.write(f"\x{data_hexstr[i:i+2]}")
        header_file.write('";\n\n')

def create_headers(image_name):
    """
    此函数为运行推理所需的输入和输出数组生成 C 头文件
    """
    img_path = os.path.join("./", f"{image_name}")

    # 将图像大小调整为 224x224
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")

    # 将输入转换为 NCHW
    img_data = np.transpose(img_data, (2, 0, 1))

    # 创建输入头文件
    input_data = img_data.astype(np.uint8)
    create_header_file("inputs", "ethosu_scratch", "input", input_data, "./include")
    # 创建输出头文件
    output_data = np.zeros([1001], np.uint8)
    create_header_file(
        "outputs",
        "output_data_sec",
        "output",
        output_data,
        "./include",
    )

if __name__ == "__main__":
    create_headers(sys.argv[1])
```

用以下命令行运行脚本：

``` bash
python convert_image.py ./kitten.jpg
```

## 预处理标签

以下脚本将在 src 目录中创建一个 `labels.h` 头文件，之前下载的 labels.txt 文件会变成字符串数组。该数组将用于显示图像分类后为的标签。

*convert_labels.py*[¶](#convert-image)

``` python
#!python ./convert_labels.py
import os
import pathlib
import sys

def create_labels_header(labels_file, section, output_path):
    """
    此函数生成一个包含 ImageNet 标签的头文件作为字符串数组
    """
    labels_path = pathlib.Path(labels_file).resolve()
    file_path = pathlib.Path(f"{output_path}/labels.h").resolve()
    with open(labels_path) as f:
        labels = f.readlines()
    with open(file_path, "w") as header_file:
        header_file.write(f'char* labels[] __attribute__((section("{section}"), aligned(16))) = {{')
        for _, label in enumerate(labels):
            header_file.write(f'"{label.rstrip()}",')
        header_file.write("};\n")

if __name__ == "__main__":
    create_labels_header(sys.argv[1], "ethosu_scratch", "./include")
```

用以下命令行运行脚本：

``` bash
python convert_labels.py
```

## 编写 demo

下面的 C 程序会在之前下载并转换为整数数组的图像上，运行 MobileNet v1 模型的单个推理。由于该模型是以「ethos-u ...」为 target 编译的，因此需要迁移 Ethos(TM)-U55 NPU 支持的算子，以便进行加速。一旦应用程序构建并运行，测试图像应该被正确地归类为「tabby」，并且结果会显示在控制台上。这个文件应该放在 `./src`。

demo.c[¶](#demo-c)

``` c
 #include <stdio.h>
 #include <tvm_runtime.h>

 #include "ethosu_mod.h"
 #include "uart.h"

 // convert_image.py 和 convert_labels.py 生成的头文件
 #include "inputs.h"
 #include "labels.h"
 #include "outputs.h"

 int abs(int v) { return v * ((v > 0) - (v < 0)); }

 int main(int argc, char** argv) {
   uart_init();
   printf("Starting Demo\n");
   EthosuInit();

   printf("Allocating memory\n");
   StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);

   printf("Running inference\n");
   struct tvmgen_default_outputs outputs = {
       .output = output,
   };
   struct tvmgen_default_inputs inputs = {
       .input = input,
   };
   struct ethosu_driver* driver = ethosu_reserve_driver();
   struct tvmgen_default_devices devices = {
       .ethos_u = driver,
   };
   tvmgen_default_run(&inputs, &outputs, &devices);
   ethosu_release_driver(driver);

   // 计算最大值的索引
   uint8_t max_value = 0;
   int32_t max_index = -1;
   for (unsigned int i = 0; i < output_len; ++i) {
     if (output[i] > max_value) {
       max_value = output[i];
       max_index = i;
     }
   }
   printf("The image has been classified as '%s'\n", labels[max_index]);

   // 当 FVP 在 UART 上接收到「EXITTHESIM」时，FVP 将关闭
   printf("EXITTHESIM\n");
   while (1 == 1)
     ;
   return 0;
 }
```

此外，需要将 github 的这些头文件放入 `./include` 目录：

[包含文件](https://github.com/apache/tvm/tree/main/apps/microtvm/ethosu/include)

:::note
若要用 FreeRTOS 进行任务调度和队列，可以在 [demo_freertos.c](https://github.com/apache/tvm/blob/main/apps/microtvm/ethosu/src/demo_freertos.c) 找到示例程序。
:::

## 创建链接脚本（linker script）

创建一个链接脚本，便于在下一节中构建应用程序时使用。链接脚本告诉链接器所有文件都放在内存中。下面的 corstone300.ld 链接脚本应该放在工作目录中。

查看 FVP 示例链接脚本 [corstone300.ld](https://github.com/apache/tvm/blob/main/apps/microtvm/ethosu/corstone300.ld)

:::note
TVM 生成的代码会将模型权重和 Arm(R) Ethos(TM)-U55 命令流放在名为 `ethosu_scratch` 的部分中，对于 MobileNet v1 大小的模型，无法将权重和命令流全部放入 SRAM 的可用空间（因为 SRAM 空间十分有限）。因此，链接脚本将 `ethosu_scratch` 这部分放入 DRAM（DDR）中。
:::

:::note
在构建和运行应用程序之前，需要更新 PATH 环境变量，使其包含 cmake 3.19.5 和 FVP 的路径。例如，若已将它们安装在 `/opt/arm` 中，则执行以下操作：

``` bash
export PATH=/opt/arm/FVP_Corstone_SSE-300_Ethos-U55/models/Linux64_GCC-6.4:/opt/arm/cmake/bin:$PATH
```
:::

## 使用 make 构建 demo

接下来可以使用 make 构建 demo。在命令行运行 `make` 之前，应将 Makefile 移在工作目录中：

Makefile 示例：[Makefile](https://github.com/apache/tvm/blob/main/apps/microtvm/ethosu/Makefile)

:::note
**若使用的是 FreeRTOS，Makefile 会从指定的 FREERTOS_PATH 构建：**

`make FREERTOS_PATH=<FreeRTOS directory>`
:::

## 运行 demo

最后，使用以下命令在固定虚拟平台（Fixed Virtual Platform，简称 FVP）上运行 demo：

``` bash
FVP_Corstone_SSE-300_Ethos-U55 -C cpu0.CFGDTCMSZ=15 \
-C cpu0.CFGITCMSZ=15 -C mps3_board.uart0.out_file=\"-\" -C mps3_board.uart0.shutdown_tag=\"EXITTHESIM\" \
-C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 \
-C mps3_board.telnetterminal1.start_telnet=0 -C mps3_board.telnetterminal2.start_telnet=0 -C mps3_board.telnetterminal5.start_telnet=0 \
-C ethosu.extra_args="--fast" \
-C ethosu.num_macs=256 ./build/demo
```

应在控制台窗口中看到以下输出：

``` bash
telnetterminal0: Listening for serial connection on port 5000
telnetterminal1: Listening for serial connection on port 5001
telnetterminal2: Listening for serial connection on port 5002
telnetterminal5: Listening for serial connection on port 5003

    Ethos-U rev dedfa618 --- Jan 12 2021 23:03:55
    (C) COPYRIGHT 2019-2021 Arm Limited
    ALL RIGHTS RESERVED

Starting Demo
ethosu_init. base_address=0x48102000, fast_memory=0x0, fast_memory_size=0, secure=1, privileged=1
ethosu_register_driver: New NPU driver at address 0x20000de8 is registered.
CMD=0x00000000
Soft reset NPU
Allocating memory
Running inference
ethosu_find_and_reserve_driver - Driver 0x20000de8 reserved.
ethosu_invoke
CMD=0x00000004
QCONFIG=0x00000002
REGIONCFG0=0x00000003
REGIONCFG1=0x00000003
REGIONCFG2=0x00000013
REGIONCFG3=0x00000053
REGIONCFG4=0x00000153
REGIONCFG5=0x00000553
REGIONCFG6=0x00001553
REGIONCFG7=0x00005553
AXI_LIMIT0=0x0f1f0000
AXI_LIMIT1=0x0f1f0000
AXI_LIMIT2=0x0f1f0000
AXI_LIMIT3=0x0f1f0000
ethosu_invoke OPTIMIZER_CONFIG
handle_optimizer_config:
Optimizer release nbr: 0 patch: 1
Optimizer config cmd_stream_version: 0 macs_per_cc: 8 shram_size: 48 custom_dma: 0
Optimizer config Ethos-U version: 1.0.6
Ethos-U config cmd_stream_version: 0 macs_per_cc: 8 shram_size: 48 custom_dma: 0
Ethos-U version: 1.0.6
ethosu_invoke NOP
ethosu_invoke NOP
ethosu_invoke NOP
ethosu_invoke COMMAND_STREAM
handle_command_stream: cmd_stream=0x61025be0, cms_length 1181
QBASE=0x0000000061025be0, QSIZE=4724, base_pointer_offset=0x00000000
BASEP0=0x0000000061026e60
BASEP1=0x0000000060002f10
BASEP2=0x0000000060002f10
BASEP3=0x0000000061000fb0
BASEP4=0x0000000060000fb0
CMD=0x000Interrupt. status=0xffff0022, qread=4724
CMD=0x00000006
00006
CMD=0x0000000c
ethosu_release_driver - Driver 0x20000de8 released
The image has been classified as 'tabby'
EXITTHESIM
Info: /OSCI/SystemC: Simulation stopped by user.
```

可以看到，输出的最后部分显示图像已被正确分类为「tabby」。

[下载 Python 源代码：micro_ethosu.py](https://tvm.apache.org/docs/_downloads/ab2eef18d10188532645b1d60fc7dd68/micro_ethosu.py)

[下载 Jupyter Notebook：micro_ethosu.ipynb](https://tvm.apache.org/docs/_downloads/55a9eff88b1303e525d53269eeb16897/micro_ethosu.ipynb)
