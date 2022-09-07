---
title: 部署模型并与 TVM 集成
---

# 部署模型并与 TVM 集成

本节介绍如何将 TVM 部署到各种平台，以及如何将其与项目集成。

 ![https://tvm.apache.org/images/release/tvm_flexible.png](https://tvm.apache.org/images/release/tvm_flexible.png)

## 构建 TVM runtime 库

不同于传统的深度学习框架，TVM 堆栈分为两个主要组件：

* TVM compiler：负责模型的编译和优化。
* TVM runtime：在目标设备上运行。

集成编译后的模块并**不需要**在目标设备上构建整个 TVM，只需在你的电脑上构建 TVM 编译器堆栈，然后用来交叉编译要部署到目标设备上的模块。

这里只需利用可集成到各种平台的轻量级 runtime API 即可。

例如，可在基于 Linux 的嵌入式系统（如树莓派）上，运行以下命令来构建 runtime API：

``` bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make runtime
```

注意：`make runtime` 仅构建 runtime 库。

也可以交叉编译 runtime 库，但不要和嵌入式设备的交叉编译模型混淆。

若要包含其他 runtime（例如 OpenCL），可以修改 `config.cmake` 来启用这些选项。获取 TVM runtime 库后，就可以链接编译好的库了。

 ![https://raw.githubusercontent.com/tlc-pack/web-data/main/images/dev/tvm_deploy_crosscompile.svg](https://raw.githubusercontent.com/tlc-pack/web-data/main/images/dev/tvm_deploy_crosscompile.svg)

TVM 可针对不同架构（例如 `x64_64` 主机上的 `aarch64`）交叉编译模型（无论是否被 TVM 优化）。一旦模型被交叉编译，runtime 必须与目标架构兼容，才能运行交叉编译的模型。

## 为其他架构交叉编译 TVM runtime

在 [上面](https://tvm.apache.org/docs/how_to/deploy/index.html#build-tvm-runtime-on-target-device) 的示例中，runtime 库是在树莓派上编译的，与树莓派等目标设备相比，在拥有高性能芯片和充足资源的主机（如笔记本电脑、工作站）上生成 runtime 库的速度要快得多。为了交叉编译 runtime，必须安装目标设备的工具链。安装正确的工具链后，与原生编译相比，主要区别在于向 cmake 传递了一些额外的命令行参数来指定要使用的工具链。例如，在现代笔记本电脑（使用 8 个线程）上为 `aarch64` 构建 TVM runtime 库需要大约 20 秒，而在树莓派 4 上构建 runtime 需要约 10 分钟。

### aarch64 的交叉编译

``` bash
sudo apt-get update
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

``` bash
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_VERSION=1 \
    -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
    -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DMACHINE_NAME=aarch64-linux-gnu

make -j$(nproc) runtime
```

对于 ARM 裸机，用以下工具链（而不是 gcc-aarch64-linux-*）来安装非常方便：

``` bash
sudo apt-get install gcc-multilib-arm-linux-gnueabihf g++-multilib-arm-linux-gnueabihf
```

### RISC-V 的交叉编译

``` bash
sudo apt-get update
sudo apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
···

···
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_VERSION=1 \
    -DCMAKE_C_COMPILER=/usr/bin/riscv64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/riscv64-linux-gnu-g++ \
    -DCMAKE_FIND_ROOT_PATH=/usr/riscv64-linux-gnu \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DMACHINE_NAME=riscv64-linux-gnu

make -j$(nproc) runtime
```

`file` 命令可用于查询生成的 runtime 的架构。

``` bash
file libtvm_runtime.so
libtvm_runtime.so: ELF 64-bit LSB shared object, UCB RISC-V, version 1 (GNU/Linux), dynamically linked, BuildID[sha1]=e9ak845b3d7f2c126dab53632aea8e012d89477e, not stripped
```

## 针对目标设备优化和调优模型

在嵌入式设备上对 TVM 内核进行测试、调优和基准测试，最简单且推荐的方法是通过 TVM 的 RPC API。下面是相关教程的链接：

* [交叉编译和 RPC](../../user_tutorial/rpc)
* [在树莓派上部署预训练模型](https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_rasp.html#tutorial-deploy-model-on-rasp)

## 在目标设备上部署优化模型

完成调优和基准测试后，要在目标设备上以不依赖 RPC 的方式来部署模型。具体操作参考以下教程：

* [使用 C++ API 部署 TVM 模块](deploy_c++)
* [部署到 Android](deploy_android)
* [将 TVM 集成到项目中](integrate)
* [HLS 后端示例](hls)
* [Relay Arm®  计算库集成](relay_arm)
* [Relay TensorRT 集成](relay_tensorrt)
* [Vitis AI 集成](vitis_ai)
* [Relay BNNS 集成](relay_bnns)

## 其他部署方法

前面已经有许多针对特定设备的操作指南，其中包含 Python 代码的示例（可用 Jupyter Notebook 查看），这些操作指南描述了如何准备模型，并将其部署到支持的后端。

* [部署深度学习模型](deploy_models)