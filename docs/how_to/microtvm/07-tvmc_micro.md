---
title: 使用 TVMC Micro 执行微模型
---

# 使用 TVMC Micro 执行微模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_tvmc.html#sphx-glr-download-how-to-work-with-microtvm-micro-tvmc-py) 下载完整的示例代码
:::

**作者**：[Mehrdad Hessar](https://github.com/mehrdadh)

本教程介绍了如何为微型设备编译一个微模型，并在 Zephyr 平台上构建一个程序，来执行这个模型，烧录程序，并用 tvmc micro 命令来执行所有模型。

:::note
本教程将介绍如何在 Zephyr 平台上使用 TVMC Mirco。学习本教程前，请安装 Zephyr 依赖项，或通过以下方式（已经安装了 Zephyr 依赖）之一运行本教程。

* 使用 [microTVM 虚拟机参考手册](https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_reference_vm.html#sphx-glr-how-to-work-with-microtvm-micro-reference-vm-py)。
* 使用 TVM 提供的 QEMU Docker 镜像。下载并登录到 Docker 镜像：

``` bash
cd tvm
./docker/bash.sh tlcpack/ci-qemu
```
:::

## 使用 TVMC Micro

TVMC 是一个命令行工具，也是 TVM Python 包的一部分。机器设置不同，访问此软件包的方式也不一样。多数情况下，可以直接使用 `tvmc` 命令。如果在 `$PYTHONPATH` 上将 TVM 作为 Python 模块，你可以使用 `python -m tvm.driver.tvmc`  命令来访问此驱动程序。简单起见，本教程使用 `tvmc` 命令。

检查是否安装了 TVMC 命令，运行如下命令：

``` bash
tvmc --help
```

使用 `tvmc compile` 子命令为 microtvm 编译模型，此命令的输出在后续步骤中与 `tvmc micro` 子命令一起使用。使用以下命令检查 TVMC Micro 是否可用：

``` bash
tvmc micro --help
```

使用 `tvmc micro` 执行的主要任务是 `create`、`build` 和 `flash`。要了解各个子命令下的特定选项，可使用 `tvmc micro <subcommand> --help`。本教程会使用每个子命令。

## 获取微模型

本教程使用 TFLite micro 的 Magic Wand 模型（一种深度卷积层模型，可通过加速度传感器识别手势）。

本教程使用 TFLite 格式的模型。

``` bash
wget https://github.com/tensorflow/tflite-micro/raw/main/tensorflow/lite/micro/examples/magic_wand/magic_wand.tflite
```

## 将 TFLite 模型编译为模型库格式

模型库格式（Model Library Format，简称 MLF）是 TVM 为微 target 提供的一种输出格式，MLF 是包含了 TVM 编译器输出的所有部分的 tarball，这些编译器输出可以用于 TVM 环境之外的微 target。更多信息请访问 [模型库格式](https://tvm.apache.org/docs/arch/model_library_format.html)。

为 `qemu_x86` Zephyr 板生成一个 MLF 文件，为 `magic_wand` TFLite 模型生成 MLF 输出：

``` bash
tvmc compile magic_wand.tflite \
     --target='c -keys=cpu -link-params=0 -model=host' \
     --runtime=crt \
     --runtime-crt-system-lib 1 \
     --executor='graph' \
     --executor-graph-link-params 0 \
     --output model.tar \
     --output-format mlf \
     --pass-config tir.disable_vectorize=1 \
     --disabled-pass=AlterOpLayout
```

这将生成一个包含 TVM 编译器输出文件的 `model.tar` 文件。若要为不同的 Zephyr 设备运行此命令，需要更新 `target`。例如，对于 `nrf5340dk_nrf5340_cpuapp` 板，target 是 `--target='c -keys=cpu -link-params=0 -model=nrf5340dk'`。

## 使用模型库格式创建 Zephyr 项目

使用 TVM Micro 子命令 `create` 生成 Zephyr 项目。将 MLF 格式、项目路径，以及项目选项传递给 `create` 子命令。每个平台（Zephyr/Arduino）的项目选项在其项目 API 服务器文件中定义。运行如下命令生成 Zephyr 项目：

``` bash
tvmc micro create \
    project \
    model.tar \
    zephyr \
    --project-option project_type=host_driven zephyr_board=qemu_x86
```

以上命令为 `qemu_x86` Zephyr 板生成一个 `Host-Driven` Zephyr 项目，在 Host-Driven 模板项目中，图执行器（Graph Executor）将在主机上运行，并通过使用 RPC 机制向设备发出命令，在 Zephyr 设备上运行模型执行。阅读有关[主机驱动执行](https://tvm.apache.org/docs/arch/microtvm_design.html#host-driven-execution)的更多信息。

获取有关 TVMC Micro `create` 子命令的更多信息，执行如下命令：

``` bash
tvmc micro create --help
```

## 使用 TVMC Micro 构建和烧录 Zephyr 项目

接下来使用如下命令构建 Zephyr 项目（包括用于运行微模型的 TVM 生成代码、用于在主机驱动模式下运行模型的 Zephyr 模板代码和 TVM runtime 源/头文件）。要构建项目：

``` bash
tvmc micro build \
    project \
    zephyr \
    --project-option zephyr_board=qemu_x86
```

以上命令将在 `project` 目录中构建项目，并在 `project/build` 下生成二进制文件，要为不同的 Zephyr 板构建 Zephyr 项目，需更改 `zephyr_board` 项目选项。

接下来把 Zephyr 二进制文件烧录到 Zephyr 设备。对于 `qemu_x86` Zephyr 板，因为要用到 QEMU，所以不会执行任何操作，但是对于物理硬件，此步骤不可省略。

``` bash
tvmc micro flash \
    project \
    zephyr \
    --project-option zephyr_board=qemu_x86
```

## 在微 Target 上运行微模型

与设备通信后，在设备上对编译好的模型和 TVM RPC 服务器进行编程。Zephyr 板等待主机打开通信通道。MicroTVM 设备通常使用串口通信（UART）进行通信 。要使用 TVMC 在设备上运行闪存模型，可通过 `tvmc run` 子命令并通过 `--device micro` 来指定设备类型，打开通信通道，使用主机上的 `Graph Executor` 设置输入值并在设备上运行完整模型，然后从设备获取输出。

``` bash
tvmc run \
    --device micro \
    project \
    --project-option zephyr_board=qemu_x86 \
    --fill-mode ones \
    --print-top 4

 # Output:
 #
 # INFO:__main__:b'[100%] [QEMU] CPU: qemu32,+nx,+pae\n'
 # remote: microTVM Zephyr runtime - running
 # INFO:__main__:b'[100%] Built target run\n'
 # [[3.         1.         2.         0.        ]
 # [0.47213247 0.41364592 0.07525456 0.03896701]]
```

具体来说，此命令将模型的输入全部设置为 1，并显示输出的四个值及其索引。

[下载 Python 源代码：micro_tvmc.py](https://tvm.apache.org/docs/_downloads/eb483c672b88006c331115968e0ffd9b/micro_tvmc.py)

[下载 Jupyter notebook：micro_tvmc.ipynb](https://tvm.apache.org/docs/_downloads/6e511f5a8ddbf12f2fca2dfadc0cc4a9/micro_tvmc.ipynb)
