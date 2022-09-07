---
title: VTA 安装指南
sidebar_position: 100
---

# VTA 安装指南

我们提供了五个安装指南，每一个都对前一个教程进行了扩展：

1. [VTA 模拟器安装](#vta-simulator-installation)
2. [Xilinx Pynq FPGA 设置](#xilinx-pynq-fpga-setup)
3. [Intel DE10 FPGA 设置](#intel-de10-fpga-setup)
4. [使用 Xilinx 工具链生成比特流](#bitstream-generation-with-xilinx-toolchains)
5. [使用 Intel 工具链生成比特流](#bitstream-generation-with-intel-toolchains)

## VTA 模拟器安装

需要在机器上 [安装 TVM](../../getting_started/install_idx)。查看 [Docker 指南](../../getting_started/install/docker) 来快速开始。

设置以下路径后才能使用 VTA：

``` bash
export TVM_PATH=<path to TVM root>
export VTA_HW_PATH=$TVM_PATH/3rdparty/vta-hw
```

构建 TVM 时要启用 VTA 功能模拟库。

``` bash
cd <tvm-root>
mkdir build
cp cmake/config.cmake build/.
echo 'set(USE_VTA_FSIM ON)' >> build/config.cmake
cd build && cmake .. && make -j4
```

将 VTA Python 库添加到 Python 路径，运行 VTA 示例。

``` bash
export PYTHONPATH=/path/to/vta/python:${PYTHONPATH}
```

### 测试 VTA 模拟设置

为确保已正确安装 VTA Python 包，运行以下 2D 卷积进行测试。

``` bash
python <tvm root>/vta/tests/python/integration/test_benchmark_topi_conv2d.py
```

诚邀你体验 [VTA 编程教程](tutorials)。

**注意**：每个卷积层的吞吐量都会在 GOPS 中报告。这些数字实际上是模拟器通过评估软件中的卷积实现的计算吞吐量。

### 高级配置（可选）

VTA 是一个通用的可配置深度学习加速器。配置由 `3rdparty/vta-hw/config` 下的 `vta_config.json` 指定。该文件提供了 VTA 加速器的体系结构规范，以参数化 TVM 编译器堆栈和 VTA 硬件堆栈。

VTA 配置文件还指定了 TVM 编译器 target。当 `TARGET` 设置为 `sim` 时，所有 TVM 工作负载都在 VTA 模拟器上执行。可以修改配置文件的内容，将 VTA 重建为不同的参数化：

``` bash
cd <tvm root>
vim 3rdparty/vta-hw/config/vta_config.json
# 编辑 vta_config.json
make
```

## Xilinx Pynq FPGA 设置

第二个指南扩展了以上 *VTA 模拟器安装*指南，从而运行完整的 TVM 和 VTA 软件-硬件堆栈的 FPGA 硬件测试。需要的硬件组件有：

* [Pynq](http://www.pynq.io/) FPGA 开发板可从 [Digilent](https://store.digilentinc.com/pynq-z1-python-productivity-for-zynq/) 以 200 美元或 150 美元的价格购买。
* 一个以太网到 USB 适配器，用于将 Pynq 板连接到你的开发机器。
* 8+ GB micro SD 卡。
* 一个 AC 转 DC 12V 3A 电源适配器。

本指南涵盖以下主题：

1. Pynq 板设置说明。
2. Pynq 端 RPC 服务器构建和部署。
3. 再次访问 *VTA 模拟器安装指南*中的测试示例，这次是在 Pynq 板上执行。

### Pynq 板设置

根据 [Pynq 开发板入门教程](http://pynq.readthedocs.io/en/latest/getting_started.html) 设置 Pynq 开发板。

按照说明进行操作，包括*打开 PYNQ-Z1* 步骤（此后无需继续学习本教程）。

* 确保已下载最新的 Pynq 镜像 [PYNQ-Z1 v2.5](http://www.pynq.io/board.html)，并已经用它为你的 SD 卡制作镜像（推荐免费的 [Etcher](https://etcher.io/) 程序）。
* 对于这个测试设置，遵循[“连接到计算机”](https://pynq.readthedocs.io/en/latest/getting_started/pynq_z1_setup.html)以太网设置说明。为成功与板子通信，确保 [为计算机分配一个静态 IP 地址](https://pynq.readthedocs.io/en/latest/appendix.html#assign-your-computer-a-static-ip)。

一旦开发板通电，并连接到你的开发机器，尝试进行连接，并确保已正确设置 Pynq 开发板：

``` bash
# 要连接到 Pynq 开发板，使用 <username, password> 组合：<xilinx, xilinx>
ssh xilinx@192.168.2.99
```

### Pynq 端 RPC 服务器构建和部署

因为板到计算机的直接连接会阻止单板直接访问互联网，所以要使用 [sshfs](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh) 将 Pynq 的文件系统挂载到你的开发机器的文件系统中。接下来，直接将 TVM 仓库克隆到开发机器上的 sshfs 挂载点。

``` bash
# 在宿主机端
mkdir <mountpoint>
sshfs xilinx@192.168.2.99:/home/xilinx <mountpoint>
cd <mountpoint>
git clone --recursive https://github.com/apache/tvm tvm
# 完成后，可以离开挂载点，并卸载目录
cd ~
sudo umount <mountpoint>
```

现在已经在 Pynq 的文件系统中克隆了 VTA 仓库，可以通过 ssh 登入，并基于 TVM 的 RPC 服务器启动构建。构建过程大约需要 5 分钟。

``` bash
ssh xilinx@192.168.2.99
# 构建 TVM runtime 库（需要 5 分钟）
cd /home/xilinx/tvm
mkdir build
cp cmake/config.cmake build/.
echo 'set(USE_VTA_FPGA ON)' >> build/config.cmake
# 复制 pynq 具体配置
cp 3rdparty/vta-hw/config/pynq_sample.json 3rdparty/vta-hw/config/vta_config.json
cd build
cmake ..
make runtime vta -j2
# FIXME (tmoreau89): 通过修复 cmake 构建，删除此步骤
make clean; make runtime vta -j2
# 构建 VTA RPC 服务器（需要 1 分钟）
cd ..
sudo ./apps/vta_rpc/start_rpc_server.sh # pw is 'xilinx'
```

启动 RPC 服务器时，可看到以下显示。为了运行下一个示例，需要让 RPC 服务器在 `ssh` session 中运行。

``` bash
INFO:root:RPCServer: bind to 0.0.0.0:9091
```

关于 Pynq RPC 服务器的提示：

* RPC 服务器应该在端口 `9091` 上监听。若没有，早期的进程可能已经意外终止。在这种情况下推荐重新启动 Pynq，然后重新运行 RPC 服务器。
* 要终止 RPC 服务器，只需发送 `Ctrl + c` 命令。可以用 `sudo ./apps/pynq_rpc/start_rpc_server.sh` 重新运行。
* 若无响应，可以通过使用物理电源开关对其重新通电，重新启动单板。

### 测试基于 Pynq 的硬件设置

在开发机器上运行示例前，按如下方式配置主机环境：

``` bash
# 在宿主机端
export VTA_RPC_HOST=192.168.2.99
export VTA_RPC_PORT=9091
```

此外，还需将主机上的 `vta_config.json` 文件中 `TARGET` 字段设置为 `"pynq"` 来指定 target 是 Pynq 平台。

注意：与模拟设置相比，主机端没有要编译的库，因为主机会将所有计算转移到 Pynq 板上。

``` bash
# 在宿主机端
cd <tvm root>
cp 3rdparty/vta-hw/config/pynq_sample.json 3rdparty/vta-hw/config/vta_config.json
```

运行 2D 卷积 testbench。在此之前，要用 VTA 比特流对 Pynq 板 FPGA 进行编程，并通过 RPC 构建 VTA runtime。以下 `test_program_rpc.py` 脚本将执行两个操作：

* FPGA 编程，通过从 [VTA 比特流仓库](https://github.com/uwsampl/vta-distro) 中下载预编译的比特流，这个仓库与主机设置的默认配置 `vta_config.json` 匹配，并通过 RPC 发送到 Pynq，从而对 Pynq 的 FPGA 进行编程。
* `vta_config.json` 配置每次修改，都要运行在 Pynq 上构建的 Runtime。这样可确保 VTA 软件 runtime（通过 just-in-time (JIT) 编译生成加速器可执行文件）与在 FPGA 上编程的 VTA 设计规范匹配。构建过程大约需要 30 秒完成，耐心等待！

``` bash
# 在宿主机端
python <tvm root>/vta/tests/python/pynq/test_program_rpc.py
```

准备在硬件中运行 2D 卷积 testbench。

``` bash
# 在宿主机端
python <tvm root>/vta/tests/python/integration/test_benchmark_topi_conv2d.py
```

每个卷积层在 Pynq 板上测试的性能指标都会生成报告。

**提示**：可以通过查看 Pynq `ssh` session 中 RPC 服务器的日志消息来跟踪 FPGA 编程和 runtime 重建步骤的进度。

更多信息请访问 [VTA 编程教程](tutorials)。

## Intel DE10 FPGA 设置

与 Pynq 端设置步骤类似，第三个指南详细介绍了如何为 Intel FPGA 板（如 DE10-Nano）设置 Linux 环境。

就硬件组件而言，需要 [DE10-Nano 开发套件](https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&No=1046)，它可以从 [Terasic](https://www.terasic.com.tw/) 以 130 美元购入（教育优惠价格为 100 美元）。该套件提供一张 microSD 卡，以及电源线和 USB 线。但需要额外的网线将板连接到 LAN。

本指南将讲解以下步骤：

* 用最新 Angstrom Linux 镜像烧录 microSD 卡
* 交叉编译设置
* 设备端 RPC 服务器设置和部署

### DE10-Nan 板设置

启动设备前，要用最新 Angstrom Linux 镜像烧录 microSD 卡镜像。

#### 烧录 SD 卡和引导 Angstrom Linux

要在 DE10-Nano 上烧录 SD 卡，并启动 Linux，推荐查看 Terasic 公司的 DE10-Nano 产品页面的 [Resource](https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=167&No=1046&PartNo=4) tab。在网页上注册并登录后，即可下载预构建的 Angstrom Linux 镜像并烧录。具体来说，要将下载的 Linux SD 卡镜像烧录到你的物理 SD 卡中：

首先，提取 gzip 压缩的存档文件。

``` bash
tar xf de10-nano-image-Angstrom-v2016.12.socfpga-sdimg.2017.03.31.tgz
```

将生成一个名为 `de10-nano-image-Angstrom-v2016.12.socfpga-sdimg`（约 2.4 GB）的 SD 卡镜像，包含启动 Angstrom Linux 的所有文件系统。

其次，在你的 PC 中插入准备烧录的 SD 卡，并用 `fdisk -l` 查询磁盘的设备 ID，若觉得使用 GUI 更好，则用 `gparted`。磁盘的典型设备 ID 可能是 `/dev/sdb`。

然后，用以下命令将磁盘镜像烧录到你的物理 SD 卡中：

``` bash
# 注意：运行以下命令通常需要 root 权限。
dd if=de10-nano-image-Angstrom-v2016.12.socfpga-sdimg of=/dev/sdb status=progress
```

将整个文件系统写入 PC 的 SD 卡需要几分钟时间。完成后，就可以取下 SD 卡，并将其插入 DE10-Nano 板。接下来连接电源线和串口来启动 Angstrom Linux。

**注意**：从 microSD 卡启动时，可能会出现与 microSD 卡中 Linux 内核 `zImage` 不兼容的情况。这种情况下需要从 [linux-socfpga](https://github.com/altera-opensource/linux-socfpga) 仓库的 [socfpga-4.9.78-ltsi](https://github.com/altera-opensource/linux-socfpga/tree/socfpga-4.9.78-ltsi) 分支构建你自己的 `zImage` 文件。还可以 [从此链接](https://raw.githubusercontent.com/liangfu/de10-nano-supplement/master/zImage) 下载 `zImage` 文件的预构建版本来快速修复。

将 USB 线连接到 DE10-Nano 开发板后，连接电源线给开发板上电。然后，可以用主机 PC 上的 `minicom` 连接到设备的串行端口：

``` bash
# 注意：运行以下命令通常需要 root 权限。
minicom -D /dev/ttyUSB0
```

设备的默认用户名为 `root`，默认密码为空。

接下来安装支持 Python3 的包（TVM 不再支持 Python2），具体来说是 `numpy`、`attrs` 和 `decorator`。

**注意**：在 DE10-Nano 设备上用 `pip3` 可能无法安装 `numpy`。这种情况可以从 [meta-de10-nano](https://github.com/intel/meta-de10-nano) 仓库为开发板构建文件系统镜像；也可以从现有的 Linux 发行版下载预构建的包，例如 Debian。我们已在 [此处](https://raw.githubusercontent.com/liangfu/de10-nano-supplement/master/rootfs_supplement.tgz) 连接了补充二进制文件，可以将文件提取到根文件系统来快速修复。

#### 安装所需的 Python 包

从串口访问 bash 终端后，要在构建和安装 TVM 和 VTA 程序之前，安装所需的 Python 包。

#### 构建附加组件以使用 VTA 比特流

要在 DE10-Nano 硬件上使用上述构建的比特流，需要为系统编译几个附加组件。具体来说，要为系统编译应用程序可执行文件，需要下载并安装 [SoCEDS](http://fpgasoftware.intel.com/soceds/18.1/?edition=standard&download_manager=dlm3&platform=linux)（推荐），或者在主机上安装 `g++-arm-linux-gnueabihf` 包。还需要一个 `cma` 内核模块，来分配连续内存，以及一个用于与 VTA 子系统通信的驱动程序。

## 使用 Xilinx 工具链生成比特流

若对自行生成 Xilinx FPGA 比特流感兴趣，而非直接使用预构建的 VTA 比特流，请按照以下说明进行操作。

### Xilinx 工具链安装

推荐使用 Vivado 2020.1，因为我们的脚本已经通过测试，可以在此版本的 Xilinx 工具链上运行。本指南是为 Linux (Ubuntu) 安装编写的。

需要安装 Xilinx 的 FPGA 编译工具链 [Vivado HL WebPACK 2020.1](https://www.xilinx.com/products/design-tools/vivado.html)，它是 Vivado HLx 工具链的免许可版本。

#### 获取并启动 Vivado GUI 安装程序

1. 访问 [下载网页](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/2020-1.html)，下载适用于 Vivado HLx 2020.1 的 Linux 自解压 Web 安装程序：WebPACK 和 Editions。
2. 必须用 Xilinx 帐户登录。需要约 2 分钟创建一个 Xilinx 帐户。
3. 单击「Next」完成名称和地址验证，然后可以下载名为 `Xilinx_Unified_2020.1_0602_1208_Lin64.bin` 的二进制文件。
4. 文件下载后，在你的 `Downloads` 目录下更改文件权限，然后就可以执行：

   ``` bash
   chmod u+x Xilinx_Unified_2020.1_0602_1208_Lin64.bin
   ```

5. 现在可以执行二进制文件：

   ``` bash
   ./Xilinx_Unified_2020.1_0602_1208_Lin64.bin
   ```

#### Xilinx Vivado GUI 安装程序步骤

此时已启动 Vivado 2020.1 安装 GUI 程序。

 1. 在「Welcome」界面，单击「Next」。
 2. 在「Select Install Type」界面的「User Authentication」框中，输入你的 Xilinx 用户凭据，然后选择「Download and Install Now」选项，单击「Next」。
 3. 在「Accept License Agreements」界面上，接受所有条款，然后单击「Next」。
 4. 在「Select Edition to Install」界面，选择「Vivado HL WebPACK」，然后单击「Next」。
 5. 在「Vivado HL WebPACK」界面，在点击「Next」之前，检查以下选项（其余选项应取消选中）： * Design Tools -> Vivado Design Suite -> Vivado * Devices -> Production Devices -> SoCs -> Zynq -7000（若 target 是 Pynq 板）* Devices  -> Production -> SoC -> UltraScale+ MPSoC（若 target 是 Ultra-96 板）
 6. 总下载大小约为 5 GB，所需的磁盘空间量为 23 GB。
 7. 在「Select Destination Directory」界面，设置安装目录，然后单击「下一步」。某些路径可能会突出显示为红色——这是因为安装程序没有写入目录的权限。在这种情况下，选择不需要特殊写入权限的路径（例如你的主目录）。
 8. 在「Installation Summary」界面，点击「Install」。
 9. 将弹出「Installation Progress」窗口，来跟踪下载和安装的进度。
10. 此过程大约需要 20-30 分钟，具体取决于网络速度。
11. 安装成功完成会弹出窗口通知。单击「OK」。
12. 最后，「Vivado License Manager」将启动。选择「Get Free ISE WebPACK, ISE/Vivado IP or PetaLinux License」，并单击「Connect Now」，完成许可证注册过程。

#### 环境设置

最后一步是用以下代码更新你的 `~/.bashrc`。这包括所有 Xilinx 二进制路径，以便可以从命令行启动编译脚本。

``` bash
# Xilinx Vivado 2020.1 环境
export XILINX_VIVADO=${XILINX_PATH}/Vivado/2020.1
export PATH=${XILINX_VIVADO}/bin:${PATH}
```

### Pynq 基于 HLS 的自定义 VTA 比特流的编译

用户可自定义 VTA 配置文件中的高级硬件参数。尝试自定义 VTA 比特流编译时，可将时钟频率调快一点。

* 将 `HW_FREQ` 字段设置为 `142`。Pynq 板支持 100、142、167 和 200MHz 时钟频率。注意，频率越高，关闭时序就越困难。增加频率会导致时序违规，从而导致硬件执行错误。
* 将 `HW_CLK_TARGET` 设置为 `6`。这个参数指的是 HLS 的目标时钟周期（以纳秒为单位）——较低的时钟周期会导致流水线操作更快，从而在较高频率下实现时序收敛。从技术上讲，142MHz 的时钟需要 7ns 的目标机，但我们刻意将时钟目标机降低到 6ns，从而更好地将设计流水线化。

比特流的生成是由 `<tvm root>/3rdparty/vta-hw/hardware/xilinx/` 顶层目录的 `Makefile` 驱动的。

若只想在软件仿真中模拟 VTA 设计，确保其正常工作，输入：

``` bash
cd <tvm root>/3rdparty/vta-hw/hardware/xilinx
make ip MODE=sim
```

若只想生成基于 HLS 的 VTA IP 内核，而不启动整个设计布局和布线，输入：

``` bash
make ip
```

然后就可以在 `<tvm root>/3rdparty/vta-hw/build/hardware/xilinx/hls/<configuration>/<block>/solution0/syn/report/<block>_csynth.rpt` 下查看 HLS 综合报告。

**注意**：`<configuration>` 的名称是一个字符串，它汇总了 `vta_config.json` 中列出的 VTA 配置参数。`<block>` 的名称指的是构成高级 VTA 管道的特定模块（或 HLS 函数）。

最后，执行 `make` 命令来运行完整的硬件编译，并生成 VTA 比特流。

这个过程很长，大约一个小时才能完成，具体时长取决于机器的规格。建议在 Makefile 中设置 `VTA_HW_COMP_THREADS` 变量，充分利用开发机器上的所有内核。

编译后，可以在 `<tvm root>/3rdparty/vta-hw/build/hardware/xilinx/vivado/<configuration>/export/vta.bit` 下找到生成的比特流。

### 使用自定义比特流

可以通过在教程示例或 `test_program_rpc.py` 脚本中设置 `vta.program_fpga()` 函数的比特流路径，来对新的 VTA FPGA 比特流进行编程。

``` python
vta.program_fpga(remote, bitstream="<tvm root>/3rdparty/vta-hw/build/hardware/xilinx/vivado/<configuration>/export/vta.bit")
```

TVM 不会从 VTA 比特流仓库中下载预构建的比特流，而是用生成的新比特流，这是一种时钟频率更高的 VTA 设计。是否有观察到 ImageNet 分类示例的性能显著提高？

## 使用 Intel 工具链生成比特流

若对自行生成 Xilinx FPGA 比特流感兴趣，而非直接使用预构建的 VTA 比特流，按照以下说明进行操作。

### Intel 工具链安装

推荐使用 `Intel Quartus Prime 18.1`，因为本文档中包含的测试脚本已经在该版本上进行了测试。

需要安装 Intel 的 FPGA 编译工具链 [Quartus Prime Lite](http://fpgasoftware.intel.com/?edition=lite)，它是 Intel Quartus Prime 软件的免许可版本。

#### 获取和启动 Quartus GUI 安装程序

1. 访问 [下载中心](http://fpgasoftware.intel.com/?edition=lite)，在「Separate file」选项卡中下载 Linux 版本的「Quartus Prime (include Nios II EDS)」和「Cyclone V device support」。这样可以避免下载未使用的设备支持文件。
2. 若有帐户，填写表单登录；或在网页右侧注册帐户。
3. 登录后，可以下载安装程序和设备支持文件。
4. 文件下载后，在你的 `Downloads` 目录下更改文件权限：

   ``` bash
   chmod u+x QuartusLiteSetup-18.1.0.625-linux.run
   ```

5. 现在确保安装程序和设备支持文件都在同一目录中，用以下命令运行安装：

   ```plain
   ./QuartusLiteSetup-18.1.0.625-linux.run
   ```

6. 按照弹出的 GUI 表单上的说明，将所有内容安装在 `/usr/local` 目录中。安装后，会创建 `/usr/local/intelFPGA_lite/18.1` 文件夹，包含 Quartus 程序和其他程序。

#### 环境设置

与 Xilinx 工具链的操作类似，将以下代码添加到 `~/.bashrc` 中。

``` bash
# Intel Quartus 18.1 环境
export QUARTUS_ROOTDIR="/usr/local/intelFPGA_lite/18.1/quartus"
export PATH=${QUARTUS_ROOTDIR}/bin:${PATH}
export PATH=${QUARTUS_ROOTDIR}/sopc_builder/bin:${PATH}
```

quartus 二进制路径会添加到 `PATH` 环境变量中，然后可以从命令行启动编译脚本。

### DE10-Nano 基于 Chisel 的自定义 VTA 比特流的编译

与基于 HLS 的设计类似，用户可以自定义 VTA 配置文件 [Configs.scala](https://github.com/apache/tvm/blob/main/3rdparty/vta-hw/hardware/chisel/src/main/scala/core/Configs.scala) 中基于 Chisel 设计的高级硬件参数。

对于 Intel FPGA，比特流的生成是由 `<tvm root>/3rdparty/vta-hw/hardware/intel` 顶级目录下的 `Makefile` 驱动的。

若只为 DE10-Nano 板生成基于 Chisel 的 VTA IP 核，而不为 FPGA 硬件编译设计，输入：

``` bash
cd <tvm root>/3rdparty/vta-hw/hardware/intel
make ip
```

然后，就可以在 `<tvm root>/3rdparty/vta-hw/build/hardware/intel/chisel/<configuration>/VTA.DefaultDe10Config.v` 中找到生成的 verilog 文件。

若要为 `de10nano` 板运行完整的硬件编译：

``` bash
make
```

这个过程可能会有点长，需要半小时才能完成，具体时长取决于 PC 的性能。Quartus Prime 软件会自动检测 PC 上可用的内核数量，并用所有内核来执行此过程。

编译后，可以在 `<tvm root>/3rdparty/vta-hw/build/hardware/intel/quartus/<configuration>/export/vta.rbf` 下找到生成的比特流。还可以打开 `<tvm root>/3rdparty/vta-hw/build/hardware/intel/quartus/<configuration>/de10_nano_top.qpf` 路径的 Quartus 项目文件 (.qpf)，查看生成的报告。
