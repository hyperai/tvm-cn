---

title: 从源码安装

---


本文档介绍如何从源代码构建和安装 TVM 软件包。


**目录**
* [步骤 1：安装依赖项](https://tvm.apache.org/docs/install/from_source.html#step-1-install-dependencies)
* [步骤 2：从 GitHub 获取源码](https://tvm.apache.org/docs/install/from_source.html#step-2-get-source-from-github)
* [步骤 3：配置与构建](https://tvm.apache.org/docs/install/from_source.html#step-3-configure-and-build)
* [步骤 4：验证安装](https://tvm.apache.org/docs/install/from_source.html#step-4-validate-installation)
* [步骤 5：额外 Python 依赖项](https://tvm.apache.org/docs/install/from_source.html#step-5-extra-python-dependencies)
* [高级构建配置](https://tvm.apache.org/docs/install/from_source.html#advanced-build-configuration)
   * [Ccache](https://tvm.apache.org/docs/install/from_source.html#ccache)
   * [在 Windows 上构建](https://tvm.apache.org/docs/install/from_source.html#building-on-windows)
   * [构建 ROCm 支持](https://tvm.apache.org/docs/install/from_source.html#building-rocm-support)
   * [启用 C++ 测试](https://tvm.apache.org/docs/install/from_source.html#enable-c-tests)


## [步骤 1：安装依赖项](https://tvm.apache.org/docs/install/from_source.html#id2)

Apache TVM 需要以下依赖项：
* CMake (>= 3.24.0)
* LLVM (建议 >= 15)
* Git
* **至少支持 C++ 17 标准的最新 C++ 编译器**
   * GCC 7.1
   * Clang 5.0
   * Apple Clang 9.3
   * Visual Studio 2019 (v16.7)
* Python (>= 3.8)
* (可选) Conda (强烈推荐)
* CMake（>= 3.24.0）
* LLVM（推荐 >= 15）
* Git
* **支持 C++ 17 的现代 C++ 编译器（最低要求）**
   * GCC 7.1
   * Clang 5.0
   * Apple Clang 9.3
   * Visual Studio 2019（v16.7）
* Python（>= 3.8）
* （可选）Conda（强烈推荐）


使用 Conda 是管理依赖项的最简单方式，它提供了跨平台的工具链（包括 LLVM）。要创建包含这些构建依赖项的环境，可以运行以下命令：

```plain
# 确保从一个干净的环境开始
conda env remove -n tvm-build-venv
# 创建包含构建依赖项的 Conda 环境
conda create -n tvm-build-venv -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    git \
    python=3.11
# 进入构建环境
conda activate tvm-build-venv
```


## [步骤 2：从 GitHub 获取源码](https://tvm.apache.org/docs/install/from_source.html#id3)

你也可以选择从 GitHub 克隆源码仓库：

```plain
git clone --recursive https://github.com/apache/tvm tvm
```


:::note

克隆 TVM 仓库时，必须使用 `--recursive` 标志，这会自动克隆子模块。如果忘记使用此标志，可以在 TVM 仓库的根目录中运行以下命令手动克隆子模块：

`git submodule update --init --recursive`

:::


## [步骤 3：配置与构建](https://tvm.apache.org/docs/install/from_source.html#id4)

创建一个构建目录并运行 CMake 进行配置。以下示例展示了如何构建：

```plain
cd tvm
rm -rf build && mkdir build && cd build
# 通过 CMake 选项指定构建配置
cp ../cmake/config.cmake .
```


可以通过在配置文件末尾追加以下标志来调整配置：

```plain
# 控制默认编译标志（可选值：Release、Debug、RelWithDebInfo）
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake

# LLVM 是编译器端的必需依赖项
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake

# GPU SDK，按需启用
echo "set(USE_CUDA   OFF)" >> config.cmake
echo "set(USE_METAL  OFF)" >> config.cmake
echo "set(USE_VULKAN OFF)" >> config.cmake
echo "set(USE_OPENCL OFF)" >> config.cmake

# 支持 cuBLAS、cuDNN、cutlass，按需启用
echo "set(USE_CUBLAS OFF)" >> config.cmake
echo "set(USE_CUDNN  OFF)" >> config.cmake
echo "set(USE_CUTLASS OFF)" >> config.cmake
```


:::note

`HIDE_PRIVATE_SYMBOLS` 是一个配置选项，用于启用 `-fvisibility=hidden` 标志。该标志可以避免 TVM 与 PyTorch 之间的符号冲突（由于两者可能使用不同版本的 LLVM）。


[CMAKE_BUILD_TYPE](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html) 控制默认编译标志：
* `Debug` 设置为 `-O0 -g`
* `RelWithDebInfo` 设置为 `-O2 -g -DNDEBUG`（推荐）
* `Release` 设置为 `-O3 -DNDEBUG`

:::


编辑完 `config.cmake` 后，运行以下命令开始构建：

```plain
cmake .. && cmake --build . --parallel $(nproc)
```


:::note

`nproc` 可能在某些系统中不可用，请替换为你的系统核心数。

:::


构建成功后，会在 `build/` 目录下生成 `libtvm` 和 `libtvm_runtime`。


退出构建环境 `tvm-build-venv` 后，可以通过以下两种方式将构建结果安装到你的环境中：
* **通过环境变量安装**：

```plain
export TVM_HOME=/path-to-tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```
* **通过 pip 安装本地项目**：

```plain
conda activate your-own-env
conda install python # 确保已安装 Python
export TVM_LIBRARY_PATH=/path-to-tvm/build
pip install -e /path-to-tvm/python
```


## [步骤 4：验证安装](https://tvm.apache.org/docs/install/from_source.html#id5)


由于 TVM 是一个支持多语言绑定的编译器基础设施，安装过程中容易出错。因此，强烈建议在使用前验证安装。


**步骤 1：定位 TVM Python 包**

以下命令可以确认 TVM 是否已正确安装为 Python 包，并显示其位置：

```plain
>>> python -c "import tvm; print(tvm.__file__)"
/some-path/lib/python3.11/site-packages/tvm/__init__.py
```


**步骤 2：确认使用的 TVM 库**

当维护多个 TVM 构建或安装时，需要确认 Python 包是否使用了正确的 `libtvm`：

```plain
>>> python -c "import tvm; print(tvm._ffi.base._LIB)"
<CDLL '/some-path/lib/python3.11/site-packages/tvm/libtvm.dylib', handle 95ada510 at 0x1030e4e50>
```


**步骤 3：检查 TVM 构建选项**

如果下游应用出现问题，可能是由于错误的 TVM 提交或构建标志。以下命令可以帮助排查：

```plain
>>> python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
... # 省略部分不相关选项
GIT_COMMIT_HASH: 4f6289590252a1cf45a4dc37bce55a25043b8338
HIDE_PRIVATE_SYMBOLS: ON
USE_LLVM: llvm-config --link-static
LLVM_VERSION: 15.0.7
USE_VULKAN: OFF
USE_CUDA: OFF
CUDA_VERSION: NOT-FOUND
USE_OPENCL: OFF
USE_METAL: ON
USE_ROCM: OFF
```


**步骤4. 检查设备检测**

可以通过以下命令了解 TVM 是否能够检测到您的设备

```plain
>>> python -c "import tvm; print(tvm.metal().exist)"
True # or False
>>> python -c "import tvm; print(tvm.cuda().exist)"
False # or True
>>> python -c "import tvm; print(tvm.vulkan().exist)"
False # or True
```


请注意，上述命令验证的是本地机器上实际设备的存在情况，供 TVM 运行时（而非编译器）正确执行。然而，TVM 编译器可以在不需要物理设备的情况下执行编译任务。只要具备必要的工具链（如 NVCC），TVM 就支持在没有实际设备的情况下进行交叉编译。


## [步骤 5. 额外 Python 依赖项](https://tvm.apache.org/docs/install/from_source.html#id6)

从源代码构建不会自动安装所有必要的 Python 依赖项。可以使用以下命令安装额外的 Python 依赖项：
* 必要依赖项：

```plain
pip3 install numpy
```
* 如需使用 RPC Tracker：

```plain
pip3 install tornado
```
* 如需使用自动调优模块：

```plain
pip3 install tornado psutil 'xgboost>=1.1.0' cloudpickle
```
## [高级构建配置](https://tvm.apache.org/docs/install/from_source.html#id7)

### [Ccache](https://tvm.apache.org/docs/install/from_source.html#id8)


在支持的平台上，[Ccache 编译器包装器](https://ccache.dev/) 可以显著减少 TVM 的构建时间（尤其是构建 [cutlass](https://github.com/NVIDIA/cutlass) 或 [flashinfer](https://github.com/flashinfer-ai/flashinfer) 时）。启用 Ccache 的方式如下：
* Leave `USE_CCACHE=AUTO` in `build/config.cmake`. CCache will be used if it is found.
* 在 `build/config.cmake` 中保留 `USE_CCACHE=AUTO`。如果找到 Ccache，则会自动使用。
* 启用 Ccache 的 Masquerade 模式（通常在安装 Ccache 时配置）。只需在配置 TVM 构建时指定 C/C++ 编译器路径即可，例如：`cmake -DCMAKE_CXX_COMPILER=/usr/lib/ccache/c++ ...`。
* 将 Ccache 作为 CMake 的 C++ 编译器前缀。配置时设置 `CMAKE_CXX_COMPILER_LAUNCHER`，例如：`cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ...`。


### [在 Windows 上构建](https://tvm.apache.org/docs/install/from_source.html#id9)


TVM 支持通过 MSVC 和 CMake 在 Windows 上构建。你需要安装 Visual Studio 编译器（最低要求：**Visual Studio Enterprise 2019**）。我们测试了 [GitHub Actions 的 Windows 2019 Runner](https://github.com/actions/virtual-environments/blob/main/images/win/Windows2019-Readme.md) 的配置，可访问该页面获取全部细节。

推荐按照 [步骤 1：安装依赖项](https://tvm.apache.org/docs/install/from_source.html#install-dependencies) 获取依赖项并激活 tvm-build 环境后，运行以下命令构建：

```plain
mkdir build
cd build
cmake ..
cd ..
```


以上命令会在 build 目录下生成解决方案文件，接着运行以下命令完成构建：

```plain
cmake --build build --config Release -- /m
```


### [构建 ROCm 支持](https://tvm.apache.org/docs/install/from_source.html#id10)

目前，ROCm 仅支持 Linux 平台。以下是配置步骤：
* 设置 `set(USE_ROCM ON)`，并将 ROCM_PATH 设置为正确的路径。
* 安装 ROCm 的 HIP 运行时，确保系统已正确安装 ROCm。
* 安装最新稳定版的 LLVM（如 v6.0.1）和 LLD，确保 `ld.lld` 可通过命令行调用。

### [启用 C++ 测试](https://tvm.apache.org/docs/install/from_source.html#id11)

TVM 使用 [Google Test](https://github.com/google/googletest) 驱动 C++ 测试，最简单的安装方式是从源码构建：

```plain
git clone https://github.com/google/googletest
cd googletest
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make
sudo make install
```


安装 GTest 后，可以通过 `./tests/scripts/task_cpp_unittest.sh` 运行 C++ 测试，或通过 `make cpptest` 仅构建测试。


