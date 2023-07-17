---
title: NNPACK Contrib 安装
sidebar_position: 3
---

# NNPACK Contrib 安装

[NNPACK](https://github.com/Maratyszcza/NNPACK) 是用于神经网络计算的加速包，可以在 x86-64、ARMv7 或 ARM64 架构的 CPU 上运行。使用 NNPACK，像 MXNet 这样的高级库可以加快多核 CPU 计算机（包括笔记本电脑和移动设备）上的执行速度。

:::note
由于 TVM 已经有原生调整的调度，这里的 NNPACK 主要是为了参考和比较。对于常规使用，原生调整的 TVM 实现更佳。
:::

TVM 支持 NNPACK 在卷积、最大池和全连接层中进行前向传播（仅限推理）。在本文档中，我们对如何将 NNPACK 与 TVM 一起使用进行了高级概述。

## 条件

NNPACK 的底层实现使用了多种加速方法，包括 fft 和 winograd。这些算法在某些特殊的批处理大小、内核大小和步幅设置上比其他算法效果更好，因此根据上下文，并非所有卷积、最大池或全连接层都可以由 NNPACK 提供支持。NNPACK 仅支持 Linux 和 OS X 系统，目前不支持 Windows。

## 构建/安装 NNPACK

如果训练后的模型满足使用 NNPACK 的一些条件，则可以构建支持 NNPACK 的 TVM。请按照以下简单步骤操作：

使用以下命令构建 NNPACK 共享库。 TVM 会动态链接 NNPACK。

注意：以下 NNPACK 安装指导已经在 Ubuntu 16.04 上进行了测试。

### 构建 Ninja

NNPACK 需要最新版本的 Ninja。所以我们需要从源代码安装 ninja。

``` bash
git clone git://github.com/ninja-build/ninja.git
cd ninja
./configure.py --bootstrap
```

设置环境变量 PATH 以告诉 bash 在哪里可以找到 ninja 可执行文件。例如，假设我们在主目录 \~ 上克隆了 ninja。然后我们可以在 \~/.bashrc 中添加以下行。

``` bash
export PATH="${PATH}:~/ninja"
```

### 构建 NNPACK

CMAKE 新版 NNPACK 单独下载 [Peach](https://github.com/Maratyszcza/PeachPy) 等依赖

注意：至少在 OS X 上，运行下面的 ninja install 会覆盖安装在 /usr/local/lib 中的 googletest 库。如果您再次构建 googletest 以替换 nnpack 副本，请务必将 -DBUILD_SHARED_LIBS=ON 传给 cmake。

``` bash
git clone --recursive https://github.com/Maratyszcza/NNPACK.git
cd NNPACK

# 在 CFLAG 和 CXXFLAG 中添加 PIC 选项以构建 NNPACK 共享库
sed -i "s|gnu99|gnu99 -fPIC|g" CMakeLists.txt
sed -i "s|gnu++11|gnu++11 -fPIC|g" CMakeLists.txt
mkdir build
cd build

# 生成 ninja 构建规则并在配置中添加共享库
cmake -G Ninja -D BUILD_SHARED_LIBS=ON ..
ninja
sudo ninja install

# 在你的 ldconfig 中添加 NNPACK 的 lib 文件夹
echo "/usr/local/lib" > /etc/ld.so.conf.d/nnpack.conf
sudo ldconfig
```

## 构建支持 NNPACK 的 TVM

``` bash
git clone --recursive https://github.com/apache/tvm tvm
```

* 在 config.cmake 中设置 *set(USE_NNPACK ON)*。
* 将 *NNPACK_PATH* 设置为 $(YOUR_NNPACK_INSTALL_PATH)
  配置后使用 make 构建 TVM

``` bash
make
```