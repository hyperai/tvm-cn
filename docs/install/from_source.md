---
title: 从源码安装
sidebar_position: 1
---

在各种系统中从 0 到 1 构建和安装 TVM 软件包括两个步骤：

1.  从 C++ 代码中构建共享库（Linux: `libtvm.so`; macOS:
    `libtvm.dylib`; Windows: `libtvm.dll`）。
3.  设置编程语言包（如 Python 包）。

下载 TVM 源代码，请访问 [下载页面](https://tvm.apache.org/download)。

## 开发者：从 GitHub 获取源代码

从 GitHub 上克隆源码仓库，请使用 `--recursive` 选项来初始化子模块。

``` bash
git clone --recursive https://github.com/apache/tvm tvm
```

Windows 用户可以打开 Git shell，并输入以下命令：

``` bash
git submodule init
git submodule update
```

## 构建共享库 {#build-shared-library}

我们的目标是构建共享库：

* 在 Linux 上，目标库是 *libtvm.so* 和 *libtvm_runtime.so*
* 在 MacOS 上，目标库是 *libtvm.dylib* 和 *libtvm_runtime.dylib*
* 在 Windows 上，目标库是 *libtvm.dll* 和 *libtvm_runtime.dll*

也可以只 [构建运行时库](/docs/how_to/deploy)。

`TVM` 库的最低构建要求是：

* 支持 C++17 的 C++ 编译器
   * GCC 7.1
   * Clang 5.0
   * Apple Clang 9.3
   * Visual Stuio 2019 (v16.7)
* CMake 3.10 或更高版本
* 构建 TVM 库时，我们推荐使用 LLVM，以启用所有功能。
* 如需使用 CUDA，请确保 CUDA 工具包的版本至少在 8.0 以上。注意：CUDA 旧版本升级后，请删除旧版本并重新启动。
* macOS 可安装 [Homebrew](https://brew.sh) 以方便安装和管理依赖。
* Python：推荐使用 3.7.X+ 和 3.8.X+ 版本，3.9.X+ 暂时[不支持](https://github.com/apache/tvm/issues/8577)。

在 Ubuntu/Debian 等 Linux 操作系统上，要安装这些依赖环境，请在终端执行：

``` bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

用 Homebrew 为搭载 Intel 或 M1 芯片的 macOS 安装所需的依赖，需遵循
Homebrew 指定的安装步骤，以保证正确安装和配置这些依赖：

``` bash
brew install gcc git cmake
brew install llvm
brew install python@3.8
```

使用 cmake 来构建库。TVM 的配置可以通过编辑 config.cmake
和/或在命令行传递 cmake flags 来修改：

-   如果没有安装 cmake，可访问 [官方网站](https://cmake.org/download/)
    下载最新版本。

-   创建一个构建目录，将 `cmake/config.cmake` 复制到该目录。

    ``` bash
    mkdir build
    cp cmake/config.cmake build
    ```

-   编辑 `build/config.cmake` 自定义编译选项

    -   对于 macOS 某些版本的 Xcode，需要在 LDFLAGS 中添加
        `-lc++abi`，以免出现链接错误。

    -   将 `set(USE_CUDA OFF)` 改为 `set(USE_CUDA ON)` 以启用 CUDA
        后端。对其他你想构建的后端和库（OpenCL，RCOM，METAL，VULKAN\...\...）做同样的处理。

    -   为了便于调试，请确保使用 `set(USE_GRAPH_EXECUTOR ON)` 和
        `set(USE_PROFILER ON)` 启用嵌入式图形执行器（embedded graph
        executor）和调试功能。

    -   如需用 IR 调试，可以设置 `set(USE_RELAY_DEBUG ON)`，同时设置环境变量 *TVM_LOG_DEBUG*。

        > ``` bash
        > export TVM_LOG_DEBUG="ir/transform.cc=1;relay/ir/transform.cc=1"
        > ```

-   TVM 需要 LLVM 用于 CPU 代码生成工具（Codegen）。推荐使用 LLVM 构建。

    -   使用 LLVM 构建时需要 LLVM 4.0 或更高版本。注意，默认的 apt 中的
        LLVM 版本可能低于 4.0。
    -   由于 LLVM 从源码构建需要很长时间，推荐从 [LLVM
        下载页面](http://releases.llvm.org/download.html)
        下载预构建版本。
        -   解压缩到某个特定位置，修改 `build/config.cmake` 以添加
            `set(USE_LLVM /path/to/your/llvm/bin/llvm-config)`
        -   或直接设置 `set(USE_LLVM ON)`，利用 CMake 搜索一个可用的
            LLVM 版本。
    -   也可以使用 [LLVM Ubuntu 每日构建](https://apt.llvm.org/)
        -   注意 apt-package 会在 `llvm-config`
            中附加版本号。例如，如果你安装了 LLVM 10 版本，则设置
            `set(USE_LLVM llvm-config-10)`
    -   PyTorch 的用户建议设置
        `set(USE_LLVM "/path/to/llvm-config --link-static")` 和
        `set(HIDE_PRIVATE_SYMBOLS ON)` 以避免 TVM 和 PyTorch
        使用的不同版本的 LLVM 之间潜在的符号冲突。
    -   某些支持平台上，[Ccache 编译器 Wrapper](https://ccache.dev/)
        可帮助减少 TVM 的构建时间。在 TVM 构建中启用 CCache 的方法包括：
        -   Ccache 的 Masquerade 模式。通常在 Ccache
            安装过程中启用。要让 TVM 在 masquerade 中使用
            Ccache，只需在配置 TVM 的构建系统时指定适当的 C/C++
            编译器路径。例如：`cmake -DCMAKE_CXX_COMPILER=/usr/lib/ccache/c++ ...`。
        -   Ccache 作为 CMake 的 C++ 编译器前缀。在配置 TVM
            的构建系统时，将 CMake 变量 `CMAKE_CXX_COMPILER_LAUNCHER`
            设置为一个合适的值，例如，`cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ...`。

-   构建 TVM 及相关库：

    ``` bash
    cd build
    cmake ..
    make -j4
    ```

    -   可以使用 Ninja 来加速构建

    ``` bash
    cd build
    cmake .. -G Ninja
    ninja
    ```

    -   在 TVM 的根目录下也有一个
        Makefile，它可以自动完成其中的几个步骤：创建构建目录，将默认的
        `config.cmake` 复制到该构建目录下，运行 cmake，并运行 make。

        构建目录可以用环境变量 `TVM_BUILD_PATH` 来指定。如果
        `TVM_BUILD_PATH` 没有设置，Makefile 就会假定应该使用 TVM 里面的
        `build` 目录。 由 `TVM_BUILD_PATH`
        指定的路径可以是绝对路径，也可以是相对于 TVM 根目录的路径。 如果
        `TVM_BUILD_PATH`
        被设置为一个以空格分隔的路径列表，则将创建所有列出的路径。

        如果使用另一个构建目录，那么应该在运行时设置环境变量
        `TVM_LIBRARY_PATH`，它指向编译后的 `libtvm.so` 和
        `libtvm_runtime.so` 的位置。 如果没有设置，TVM 将寻找相对于 TVM
        Python 模块的位置。与 `TVM_BUILD_PATH`
        不同，这必须是一个绝对路径。

    ``` bash
    # 在 "build" 目录下构建
    make

    # 替代位置，"build_debug"
    TVM_BUILD_PATH=build_debug make

    # 同时构建 "build_release" 和 "build_debug"
    TVM_BUILD_PATH="build_debug build_release" make

    # 使用调试构建
    TVM_LIBRARY_PATH=~/tvm/build_debug python3
    ```

如果一切顺利，我们就可以去查看 [Python 包的安装](#python-package-installation) 了。

### 使用 Conda 环境进行构建 {#build-with-conda}

Conda 可以用来获取运行 TVM 所需的必要依赖。如果没有安装 Conda，请参照
[Conda
安装指南](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
来安装 Miniconda 或 Anaconda。在 Conda 环境中运行以下命令：

``` bash
# 用 yaml 指定的依赖创建 Conda 环境
conda env create --file conda/build-environment.yaml
# 激活所创建的环境
conda activate tvm-build
```

上述命令将安装所有必要的构建依赖，如 CMake 和
LLVM。接下来可以运行上一节中的标准构建过程。

在 Conda 环境之外使用已编译的二进制文件，可将 LLVM 设置为静态链接模式
`set(USE_LLVM "llvm-config --link-static")`。
这样一来，生成的库就不会依赖于 Conda 环境中的动态 LLVM 库。

以上内容展示了如何使用 Conda 提供必要的依赖，从而构建
libtvm。如果已经使用 Conda 作为软件包管理器，并且希望直接将 TVM 作为
Conda 软件包来构建和安装，可以按照以下指导进行：

``` bash
conda build --output-folder=conda/pkg  conda/recipe
# 在启用 CUDA 的情况下运行 conda/build_cuda.sh 来构建
conda install tvm -c ./conda/pkg
```

### 在 Windows 上构建

TVM 支持通过 MSVC 使用 CMake 构建。需要有一个 Visual Studio 编译器。 VS
的最低版本为 **Visual Studio Enterprise 2019**（注意：查看针对 GitHub
Actions 的完整测试细节，请访问 [Windows 2019
Runner](https://github.com/actions/virtual-environments/blob/main/images/win/Windows2019-Readme.md)。
官方推荐 [使用 Conda 环境进行构建](#build-with-conda)，以获取必要的依赖及激活的 tvm-build 环境。）运行以下命令行：

``` bash
mkdir build
cd build
cmake -A x64 -Thost=x64 ..
cd ..
```

上述命令在构建目录下生成了解决方案文件。接着运行：

``` bash
cmake --build build --config Release -- /m
```

### 构建 ROCm 支持

目前，ROCm 只在 Linux 上支持，因此所有教程均以 Linux 为基础编写的。 -
设置 `set(USE_ROCM ON)`，将 ROCM_PATH 设置为正确的路径。 - 需要先从 ROCm
中安装 HIP runtime。确保安装系统中已经安装了 ROCm。 - 安装 LLVM
的最新稳定版本（v6.0.1），以及 LLD，确保 `ld.lld` 可以通过命令行获取。

## Python 包的安装 {#python-package-installation}

### TVM 包

本部分介绍利用 `virtualenv` 或 `conda` 等虚拟环境和软件包管理器，来管理
Python 软件包和依赖的方法。

Python 包位于 *tvm/python*。安装方法有两种：

* 方法1

    本方法适用于有可能修改**代码的开发者**。

    设置环境变量 *PYTHONPATH*，告诉 Python 在哪里可以找到这个库。例如，假设我们在 */path/to/tvm* 目录下克隆了 *tvm*，我们可以在 *~/.bashrc* 中添加以下代码：这使得拉取代码及重建项目时，无需再次调用 `setup`，这些变化就会立即反映出来

    ``` bash
    export TVM_HOME=/path/to/tvm
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
    ```

* 方法2

    通过 *setup.py* 安装 TVM 的 Python 绑定：

    ``` bash
    # 为当前用户安装 TVM 软件包
    # 注意：如果你通过 homebrew 安装了 Python，那么在安装过程中就不需要 --user
    #        它将被自动安装到你的用户目录下。
    #        在这种情况下，提供 --user 标志可能会在安装时引发错误。
    export MACOSX_DEPLOYMENT_TARGET=10.9  # 这是 mac 所需要的，以避免与 libstdc++ 的符号冲突
    cd python; python setup.py install --user; cd ..
    ```

### Python 依赖

注意，如果你想要安装到一个受管理的本地环境，如 `virtualenv`，则不需要
`--user` 标志。

-   必要的依赖：

``` bash
pip3 install --user numpy decorator attrs
```

-   使用 RPC 跟踪器

``` bash
pip3 install --user tornado
```

-   使用 auto-tuning 模块

``` bash
pip3 install --user tornado psutil xgboost cloudpickle
```

注意：在搭载 M1 芯片的 Mac 上，安装 xgboost / scipy
时可能遇到一些问题。scipy 和 xgboost 需要安装 openblas
等额外依赖。运行以下命令行，安装 scipy 和 xgboost 以及所需的依赖和配置：

``` bash
brew install openblas gfortran

pip install pybind11 cython pythran

export OPENBLAS=/opt/homebrew/opt/openblas/lib/

pip install scipy --no-use-pep517

pip install xgboost
```

## 安装 Contrib 库

[NNPACK Contrib 安装](nnpack)

## 启用 C++ 测试 {#C++_tests}

可以用 [Google Test](https://github.com/google/googletest) 来驱动 TVM
中的 C++ 测试。安装 GTest 最简单的方法是从源代码安装：

``` bash
git clone https://github.com/google/googletest
cd googletest
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make
sudo make install
```

安装成功后，可以用 `./tests/scripts/task_cpp_unittest.sh` 来构建和启动
C++ 测试，或者直接用 `make cpptest` 构建。
