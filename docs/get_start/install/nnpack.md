---
title: NNPACK Contrib 安装
sidebar_position: 3
---

[NNPACK](https://github.com/Maratyszcza/NNPACK) 是用于神经网络计算的加速包，可以在 x86-64、ARMv7 或 ARM64 架构的 CPU 上运行。使用 NNPACK，像 \_MXNet\_ 这样的高级库可以加快多核 CPU 计算机（包括笔记本电脑和移动设备）上的执行速度。

:::note
由于 TVM 已经有原生调整的调度，这里的 NNPACK 主要是为了参考和比较。对于常规使用，原生调整的 TVM 实现更佳。
:::

TVM 支持 NNPACK 在卷积、最大池化和全连接层中进行前向传播（仅限推理）。在本文档中，我们对如何将 NNPACK 与 TVM 一起使用进行了高级概述。

## 条件

The underlying implementation of NNPACK utilizes several acceleration
methods, including fft and winograd. These algorithms work better on
some special [batch size]{.title-ref}, [kernel size]{.title-ref}, and
[stride]{.title-ref} settings than on other, so depending on the
context, not all convolution, max-pooling, or fully-connected layers can
be powered by NNPACK. When favorable conditions for running NNPACKS are
not met,

NNPACK only supports Linux and OS X systems. Windows is not supported at
present.

NNPACK 的底层实现使用了多种加速方法，包括 fft 和 winograd。这些算法在某些特殊的批处理大小、内核大小和步幅设置上比其他算法效果更好，因此根据上下文，并非所有卷积、最大池或全连接层都可以由 NNPACK 提供支持。当没有满足运行 NNPACKS 的有利条件时，NNPACK 仅支持 Linux 和 OS X 系统。目前不支持 Windows。

## Build/Install NNPACK

If the trained model meets some conditions of using NNPACK, you can
build TVM with NNPACK support. Follow these simple steps:

uild NNPACK shared library with the following commands. TVM will link
NNPACK dynamically.

Note: The following NNPACK installation instructions have been tested on
Ubuntu 16.04.

### Build Ninja

NNPACK need a recent version of Ninja. So we need to install ninja from
source.

``` bash
git clone git://github.com/ninja-build/ninja.git
cd ninja
./configure.py --bootstrap
```

Set the environment variable PATH to tell bash where to find the ninja
executable. For example, assume we cloned ninja on the home directory
\~. then we can added the following line in \~/.bashrc.

``` bash
export PATH="${PATH}:~/ninja"
```

### Build NNPACK

The new CMAKE version of NNPACK download
[Peach](https://github.com/Maratyszcza/PeachPy) and other dependencies
alone

Note: at least on OS X, running [ninja install]{.title-ref} below will
overwrite googletest libraries installed in
[/usr/local/lib]{.title-ref}. If you build googletest again to replace
the nnpack copy, be sure to pass [-DBUILD_SHARED_LIBS=ON]{.title-ref} to
[cmake]{.title-ref}.

``` bash
git clone --recursive https://github.com/Maratyszcza/NNPACK.git
cd NNPACK
# Add PIC option in CFLAG and CXXFLAG to build NNPACK shared library
sed -i "s|gnu99|gnu99 -fPIC|g" CMakeLists.txt
sed -i "s|gnu++11|gnu++11 -fPIC|g" CMakeLists.txt
mkdir build
cd build
# Generate ninja build rule and add shared library in configuration
cmake -G Ninja -D BUILD_SHARED_LIBS=ON ..
ninja
sudo ninja install

# Add NNPACK lib folder in your ldconfig
echo "/usr/local/lib" > /etc/ld.so.conf.d/nnpack.conf
sudo ldconfig
```

## Build TVM with NNPACK support

``` bash
git clone --recursive https://github.com/apache/tvm tvm
```

-   Set [set(USE_NNPACK ON)]{.title-ref} in config.cmake.
-   Set [NNPACK_PATH]{.title-ref} to the \$(YOUR_NNPACK_INSTALL_PATH)

after configuration use [make]{.title-ref} to build TVM

``` bash
make
```
