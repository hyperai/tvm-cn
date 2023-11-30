# Vitis AI 集成

[Vitis AI](https://github.com/Xilinx/Vitis-AI) 是用在 Xilinx 平台（包括边缘设备和 Alveo 卡）上进行硬件加速 AI 推理的 Xilinx 开发堆栈。它由优化的 IP、工具、库、模型和示例设计组成。在设计时兼顾高效率和易用性，充分发挥了 Xilinx FPGA 和 ACAP 上 AI 加速的潜力。

TVM 中当前的 Vitis AI 流支持使用 [Zynq Ultrascale+ MPSoc](https://www.xilinx.com/products/silicon-devices/soc/zynq-ultrascale-mpsoc.html), [Alveo](https://www.xilinx.com/products/boards-and-kits/alveo.html) 和 [Versal](https://www.xilinx.com/products/silicon-devices/acap/versal.html) 平台在边缘和云端加速神经网络模型推理。支持的边缘和云深度学习处理器单元（DPU）的标识符是：

| **Target Board** | **DPU ID** | **TVM Target ID** |
|:---|:---|:---|
| [ZCU104](https://www.xilinx.com/products/boards-and-kits/zcu104.html) | DPUCZDX8G | DPUCZDX8G-zcu104 |
| [ZCU102](https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html) | DPUCZDX8G | DPUCZDX8G-zcu102 |
| [Kria KV260](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html) | DPUCZDX8G | DPUCZDX8G-kv260 |
| [VCK190](https://www.xilinx.com/products/boards-and-kits/vck190.html) | DPUCVDX8G | DPUCVDX8G |
| [VCK5000](https://www.xilinx.com/products/boards-and-kits/vck5000.html) | DPUCVDX8H | DPUCVDX8H |
| [U200](https://www.xilinx.com/products/boards-and-kits/alveo/u200.html) | DPUCADF8H | DPUCADF8H |
| [U250](https://www.xilinx.com/products/boards-and-kits/alveo/u250.html) | DPUCADF8H | DPUCADF8H |
| [U50](https://www.xilinx.com/products/boards-and-kits/alveo/u50.html) | DPUCAHX8H / DPUCAHX8L | DPUCAHX8H-u50 / DPUCAHX8L |
| [U280](https://www.xilinx.com/products/boards-and-kits/alveo/u280.html) | DPUCAHX8H / DPUCAHX8L | DPUCAHX8H-u280 / DPUCAHX8L |

有关 DPU 标识符的更多信息，参见下表：

| **DPU** | **Application** | **HW Platform** | **Quantization Method** | **Quantization Bitwidth** | **Design Target** |
|:---|:---|:---|:---|:---|:---|
| Deep LearningProcessing Unit | C: CNNR: RNN | AD: Alveo DDRAH: Alveo HBMVD: Versal DDR with AIE & PLZD: Zynq DDR | X: DECENTI: Integer thresholdF: Float thresholdR: RNN | 4: 4-bit8: 8-bit16: 16-bitM: Mixed Precision | G: General purposeH: High throughputL: Low latencyC: Cost optimized |

此教程介绍有关如何在不同平台（Zynq、Alveo、Versal）上使用 Vitis AI [设置](#setup) TVM 以及如何开始 [编译模型](#compile) 并在不同平台上执行：[推理](#inference)。

## 系统要求

[Vitis AI 系统要求页面](https://github.com/Xilinx/Vitis-AI/blob/master/docs/learn/system_requirements.md) 列出了运行 Docker 容器以及在 Alveo 卡上执行的系统要求。对于边缘设备（例如 Zynq），部署模型需要使用带有 Vitis AI 流程的 TVM 编译模型的主机，以及用于运行编译模型的边缘设备。主机系统要求与上面链接中指定的相同。

## 设置说明

本节介绍如何用 Vitis AI 流为云和边缘设置 TVM。支持 Vitis AI 的 TVM 是通过 Docker 容器提供的。提供的脚本和 Dockerfile 将 TVM 和 Vitis AI 编译为单个镜像。

1. 克隆 TVM 仓库

   ``` bash
      git clone --recursive https://github.com/apache/tvm.git
      cd tvm
   ```

2. 构建并启动 TVM - Vitis AI Docker 容器。

   ``` bash
      ./docker/build.sh demo_vitis_ai bash
      ./docker/bash.sh tvm.demo_vitis_ai

      # Setup inside container
      conda activate vitis-ai-tensorflow
   ```

3. 用 Vitis AI（在 TVM 目录内）在容器内构建 TVM

   ``` bash
      mkdir build
      cp cmake/config.cmake build
      cd build
      echo set(USE_LLVM ON) >> config.cmake
      echo set(USE_VITIS_AI ON) >> config.cmake
      cmake ..
      make -j$(nproc)
   ```

4. 安装 TVM

   ``` bash
      cd ../python
      pip3 install -e . --user
   ```

在这个 Docker 容器中可以为云和边缘目标编译模型。要在 docker 容器内的云 Alveo 或 Versal VCK5000 卡上运行，按照 [Alveo](#alveo-setup) 或者 [Versal VCK5000](#versal-vck5000-setup) 设置说明进行操作。分别参照 [Zynq](#zynq-setup) 和 [Versal VCK190](#versal-vck190-setup)，为推理过程设置 Zynq 或 Versal VCK190 评估单板。

### Alveo 设置

查看 [Alveo 设置](https://github.com/Xilinx/Vitis-AI/blob/v1.4/setup/alveo/README.md) 获取设置信息。

设置后，通过以下方式在 Docker 容器内选择正确的 DPU：

``` bash
cd /workspace
git clone --branch v1.4 --single-branch --recursive https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI/setup/alveo
source setup.sh [DPU-IDENTIFIER]
```

可在此页面顶部的 DPU Targets 表的第二列中找到此 DPU 标识符。

### Versal VCK5000 设置

查看 [VCK5000 Setup](https://github.com/Xilinx/Vitis-AI/blob/v1.4/setup/vck5000/README.md) 获取设置信息。

设置后，可以通过以下方式在 Docker 容器内选择正确的 DPU：

``` bash
cd /workspace
git clone --branch v1.4 --single-branch --recursive https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI/setup/vck5000
source setup.sh
```

### Zynq 设置

除了构建 TVM - Vitis AI docker 之外，对于 Zynq 目标（DPUCZDX8G），编译阶段在主机上的 docker 内运行，不需要任何特定设置。执行模型时，首先要设置 Zynq 板，更多信息如下。

1. 下载 Petalinux 镜像：
   * [ZCU104](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu104-dpu-v2021.1-v1.4.0.img.gz)
   * [ZCU102](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu102-dpu-v2021.1-v1.4.0.img.gz)
   * [Kria KV260](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-kv260-dpu-v2020.2-v1.4.0.img.gz)
2. 使用 Etcher 软件将镜像文件刻录到 SD 卡上。
3. 将带有图像的 SD 卡插入目标单板。
4. 插入电源并使用串行端口在系统上启动该单板。
5. 用串口设置单板的 IP 信息。有关步骤 1 至 5 的更多信息，参阅 [设置评估单板](https://www.xilinx.com/html_docs/vitis_ai/1_4/installation.html#ariaid-title8)。
6. 在单板上创建 4GB 的交换空间

   ``` bash
   fallocate -l 4G /swapfile
   chmod 600 /swapfile
   mkswap /swapfile
   swapon /swapfile
   echo "/swapfile swap swap defaults 0 0" >> /etc/fstab
   ```

7. 安装 hdf5 依赖（需要 30 分钟到 1 小时）

   ``` bash
   cd /tmp && \
     wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.7/src/hdf5-1.10.7.tar.gz && \
     tar -zxvf hdf5-1.10.7.tar.gz && \
     cd hdf5-1.10.7 && \
     ./configure --prefix=/usr && \
     make -j$(nproc) && \
     make install && \
     cd /tmp && rm -rf hdf5-1.10.7*
   ```

8. 安装 Python 依赖

   ``` bash
   pip3 install Cython==0.29.23 h5py==2.10.0 pillow
   ```

9. 安装 PyXIR

   ``` bash
   git clone --recursive --branch rel-v0.3.1 --single-branch https://github.com/Xilinx/pyxir.git
   cd pyxir
   sudo python3 setup.py install --use_vart_edge_dpu
   ```

10. 用 Vitis AI 构建和安装 TVM

   ``` bash
   git clone --recursive https://github.com/apache/tvm
   cd tvm
   mkdir build
   cp cmake/config.cmake build
   cd build
   echo set(USE_LLVM OFF) >> config.cmake
   echo set(USE_VITIS_AI ON) >> config.cmake
   cmake ..
   make tvm_runtime -j$(nproc)
   cd ../python
   pip3 install --no-deps  -e .
   ```

11. 在 Python shell 中检查设置是否成功：

   ``` bash
   python3 -c 'import pyxir; import tvm'
   ```

:::note
可能会看到有关未找到 "cpu-tf" runtime 的警告，可以忽略。
:::

### Versal VCK190 设置

参考 [Zynq 设置](#zynq-setup) 设置 Versal VCK190，但在步骤 1 中参考 [VCK190 镜像](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-vck190-dpu-v2020.2-v1.4.0.img.gz)。其他步骤相同。

## 编译模型

带有 Vitis AI 流的 TVM 包含编译和推理两个阶段。在编译期间，用户可以为当前支持的云或边缘目标设备选择要编译的模型。编译模型生成的文件可用于在 [推理](#inference) 阶段在指定的目标设备上运行模型。目前，采用 Vitis AI 流程的 TVM 支持选定数量的 Xilinx 数据中心和边缘设备。

本节介绍在 TVM 中用 Vitis AI 编译模型的一般流程。

### 导入

确保导入 PyXIR 和 DPU target（为 DPUCADF8H `import pyxir.contrib.target.DPUCADF8H` ）：

``` python
import pyxir
import pyxir.contrib.target.DPUCADF8H

import tvm
import tvm.relay as relay
from tvm.contrib.target import vitis_ai
from tvm.contrib import utils, graph_executor
from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai
```

### 声明 Target

``` python
tvm_target = 'llvm'
dpu_target = 'DPUCADF8H' # options: 'DPUCADF8H', 'DPUCAHX8H-u50', 'DPUCAHX8H-u280', 'DPUCAHX8L', 'DPUCVDX8H', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102', 'DPUCZDX8G-kv260'
```

带有 Vitis AI 流的 TVM 目前支持本页顶部表格中列出的 DPU targets。一旦定义了恰当的 target，就会调用 TVM 编译器来为指定的 target 构建计算图。

### 导入模型

导入 MXNet 模型的示例代码：

``` python
mod, params = relay.frontend.from_mxnet(block, input_shape)
```

### 对模型分区

导入模型后，用 Relay API 为 DPU target 注释 Relay 表达式，并对计算图进行分区。

``` python
mod = partition_for_vitis_ai(mod, params, dpu=dpu_target)
```

### 构建模型

将分区模型传给 TVM 编译器，然后生成 TVM Runtime 的 runtime 库。

``` python
export_rt_mod_file = os.path.join(os.getcwd(), 'vitis_ai.rtmod')
build_options = {
    'dpu': dpu_target,
    'export_runtime_module': export_rt_mod_file
}
with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):
    lib = relay.build(mod, tvm_target, params=params)
```

### 量化模型

为了用 Vitis AI DPU 加速器来加速神经网络模型的推理，通常要对模型预先量化。在 TVM - Vitis AI 流中，利用动态量化来替代此预处理步骤。在这个流中，可用典型的推理执行调用（module.run）使用提供的前 N 个输入动态量化模型（参见更多信息如下），而不需要预先量化模型。这将设置和校准 Vitis-AI DPU，为后面所有输入加速推理。

注意：边缘流与推理中解释的流略有不同，边缘流在前 N 个输入后模型被量化和编译，但推理不会加速，并且它可以移动到边缘设备进行部署。查看下面的 [在 Zynq 上运行](#running-on-zynq-and-vck190) 部分了解更多信息。

``` python
module = graph_executor.GraphModule(lib["default"](tvm.cpu()))

# 前 N 个（默认 = 128）输入用于量化校准，并在 CPU 上执行
# 可以通过设置 “PX_QUANT_SIZE” 来更改此配置（例如，导出 PX_QUANT_SIZE=64）
for i in range(128):
   module.set_input(input_name, inputs[i])
   module.run()
```

用于量化的图像数量默认设置为 128，可以使用 PX_QUANT_SIZE 环境变量更改动态量化的图像数量。例如，在调用编译脚本之前在终端中执行如下命令，将量化校准数据集减少到八幅图像。

``` bash
export PX_QUANT_SIZE=8
```

最后将 TVM 编译器的编译输出存储在磁盘上，方便在目标设备上运行模型。云 DPU（Alveo 和 VCK5000）的情况如下：

``` python
lib_path = "deploy_lib.so"
lib.export_library(lib_path)
```

对于边缘 target（Zynq 和 VCK190），必须为 aarch64 重建。因此首先必须正常导出模块，并同时序列化 Vitis AI runtime 模块（vitis_ai.rtmod）。之后再次加载此 runtime 模块，为 aarch64 重建和导出。

``` python
temp = utils.tempdir()
lib.export_library(temp.relpath("tvm_lib.so"))

# 为 aarch64 target 构建和导出库
tvm_target = tvm.target.arm_cpu('ultra96')
lib_kwargs = {
   'fcompile': contrib.cc.create_shared,
   'cc': "/usr/aarch64-linux-gnu/bin/ld"
}

build_options = {
    'load_runtime_module': export_rt_mod_file
}
with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):
     lib_edge = relay.build(mod, tvm_target, params=params)

lib_edge.export_library('deploy_lib_edge.so', **lib_kwargs)
```

使用 TVM 和 Vitis AI 编译模型的教程到此结束，有关如何运行已编译的模型，参阅下一节。

## 推理

带有 Vitis AI 流的 TVM 包含编译和推理两个阶段，在编译期间，用户可以选择为当前支持的任何目标设备编译模型。编译模型后生成的文件可用于在推理阶段在目标设备上运行模型。

查看 [在 Alveo 和 VCK5000 上运行](#running-on-alveo-and-vck5000) 以及 [在 Zynq 和 VCK190 上运行](#running-on-zynq-and-vck190) 部分，分别在云加速卡和边缘板上进行推理。

### 在 Alveo 和 VCK5000 上运行

按照编译模型部分中的步骤，可以在 Docker 内的新输入上执行如下命令以加速推理：

``` python
module.set_input(input_name, inputs[i])
module.run()
```

或者加载导出的 runtime 模块（在 [编译模型](#compile) 中导出的 deploy_lib.so）：

``` python
import pyxir
import tvm
from tvm.contrib import graph_executor

dev = tvm.cpu()

# input_name = ...
# input_data = ...

# 将模块加载到内存
lib = tvm.runtime.load_module("deploy_lib.so")

module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, input_data)
module.run()
```

### 在 Zynq 和 VCK190 上运行

开始前按照 [Zynq](#zynq-setup) 或 [Versal VCK190](#versal-vck190-setup) 设置说明进行设置。

在单板上运行模型之前，需要为目标评估单板编译模型，并将编译后的模型传输到板上。如何编译模型，参阅 [编译模型](#compile) 部分。

之后将编译好的模型（deploy_lib_edge.so）传输到评估单板，然后可以在板上使用典型的「load_module」和「module.run」API 来执行。确保以 root 身份运行脚本（在终端中执行 `su` 登录到 root）。

:::note
**不要**在运行脚本（`import pyxir.contrib.target.DPUCZDX8G`）中导入 PyXIR DPU targets。
:::

``` python
import pyxir
import tvm
from tvm.contrib import graph_executor

dev = tvm.cpu()

# input_name = ...
# input_data = ...

# 将模块加载到内存
lib = tvm.runtime.load_module("deploy_lib_edge.so")

module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, input_data)
module.run()
```
