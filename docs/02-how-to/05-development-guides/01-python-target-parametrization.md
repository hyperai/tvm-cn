---

title: Python 目标参数化

---


## 概述

对于任何支持的开发环境，TVM 都应该生成数值正确的结果。因此，在编写验证数值输出的单元测试时，这些单元测试应在所有受支持的开发环境上执行。由于这是一个非常常见的使用场景，TVM 提供了辅助函数来参数化单元测试，使它们可以在所有已启用且具有兼容设备的目标上运行。


测试套件中的一个 Python 函数可以展开成多个参数化的单元测试，每个测试针对一个单一的目标设备。要运行一个测试，必须满足以下所有条件：


* 测试必须存在于已传递给 pytest 的文件或目录中。 

* 应用于函数的 pytest 标记（无论是显式标记还是通过目标参数化生成的标记）必须与传递给 pytest’s -m 参数的表达式兼容。 

* 对于使用 target 固件的参数化测试，目标必须出现在环境变量 TVM_TEST_TARGETS 中。 

* 对于使用 target 固件的参数化测试，config.cmake 中的构建配置必须启用对应的运行时。


## 单元测试文件内容

在多个目标上运行测试的推荐方法是对测试进行参数化。这可以通过使用装饰器 `@tvm.testing.parametrize_targets('target_1', 'target_2', ...)` 显式完成，并在函数中接受 `target` 或 `dev` 参数。该函数将针对列表中的每个目标运行一次，并分别报告每个目标的成功/失败情况。如果某个目标由于在 config.cmake 中被禁用，或因为没有合适的硬件而无法运行，则该目标将被标记为已跳过。


```plain
# 显式列出使用的目标
@tvm.testing.parametrize_target('llvm', 'cuda')
def test_function(target, dev):
    # 测试代码在这里
```


对于需要在所有目标上正常运行的测试，可以省略装饰器。任何接受 `target` 或 `dev` 参数的测试将自动在环境变量 `TVM_TEST_TARGETS` 中指定的所有目标上参数化运行。该参数化过程会为每个目标提供相同的通过/失败/跳过报告，同时允许测试套件轻松扩展以覆盖更多目标。


```plain
# 隐式参数化运行在 TVM_TEST_TARGETS 环境变量中的所有目标上
def test_function(target, dev):
    # 测试代码在这里
```


`@tvm.testing.parametrize_targets` 也可以用作裸装饰器来显式强调参数化，但没有额外效果。


```plain
# 显式参数化运行在 TVM_TEST_TARGETS 环境变量中的所有目标上
@tvm.testing.parametrize_targets
def test_function(target, dev):
    # 测试代码在这里
```


可以使用 `@tvm.testing.exclude_targets` 或 `@tvm.testing.known_failing_targets` 装饰器排除特定目标或标记预期失败的目标。有关其预期用例的更多信息，请参阅它们的文档字符串。


在某些情况下，可能需要跨多个参数进行参数化。例如，有些目标可能有多个实现方式需要测试。这种情况下，可以显式地对参数元组进行参数化，如下所示。这种写法中只会运行显式列出的目标，但每个目标仍会应用相应的 `@tvm.testing.requires_RUNTIME` 标记。

```plain
@pytest.mark.parametrize('target,impl', [
     ('llvm', cpu_implementation),
     ('cuda', gpu_implementation_small_batch),
     ('cuda', gpu_implementation_large_batch),
 ])
 def test_function(target, dev, impl):
     # 测试代码在这里
```


参数化功能是基于 `pytest marks` 实现的。每个测试函数都可以使用 [pytest marks](https://tvm.apache.org/docs/how_to/dev/pytest-marks) 进行装饰以添加元数据。最常用的标记如下：

* `@pytest.mark.gpu`：将函数标记为使用 GPU 能力。该标记本身无直接作用，但可配合命令行参数 `-m gpu` 或 `-m 'not gpu'` 使用，以限制 pytest 执行哪些测试。通常不单独使用，而是嵌入在其他单元测试装饰器中。

* `@tvm.testing.uses_gpu`：适用 `@pytest.mark.gpu`。应使用该装饰器标记那些可能使用 GPU 的测试（如果存在 GPU），仅在显式遍历 `tvm.testing.enabled_targets()` 的测试中需要该装饰器，但这已不再是推荐的写法。在使用 `tvm.testing.parametrize_targets()` 时，GPU 目标会自动带上此标记，无需显式添加。

* `@tvm.testing.requires_gpu`：适用 `@tvm.testing.uses_gpu`，并额外使用 `@pytest.mark.skipif` 标记，当无 GPU 时跳过该测试。

* `@tvm.testing.requires_RUNTIME`：一组装饰器（如 `@tvm.testing.requires_cuda`），如果某个运行时不可用（比如在 `config.cmake` 中被禁用，或缺乏兼容设备），就会跳过该测试。对于使用 GPU 的运行时，也包括 `@tvm.testing.requires_gpu`。


在使用目标参数化时，每次测试运行都会被装饰上与其目标对应的 `@tvm.testing.requires_RUNTIME`。因此，如果目标在 `config.cmake` 中被禁用或没有可用硬件，将会被明确标记为跳过。


还存在一个 `tvm.testing.enabled_targets()` 函数，它会根据环境变量 `TVM_TEST_TARGETS`、构建配置以及实际硬件返回所有已启用并可运行的目标。目前多数测试都显式地循环遍历该函数的返回值，但这不应用于新测试。这种写法在 pytest 输出中会悄悄跳过禁用的运行时，或无法运行的设备，而且测试一旦在某个目标失败就会中止运行，使得难以判断是该目标出错还是所有目标都失败。

```plain
# 旧式写法，不推荐使用
def test_function():
    for target,dev in tvm.testing.enabled_targets():
        # 测试代码在这里
```


## 本地运行

在本地运行 Python 单元测试，可以在 `${TVM_HOME}` 目录下使用命令 `pytest`。


* **环境变量**

   * `TVM_TEST_TARGETS` 应是一个以分号分隔的目标列表。如果未设置，将默认使用 `tvm.testing.DEFAULT_TEST_TARGETS` 中定义的目标。

    注意：如果 `TVM_TEST_TARGETS` 中不包含任何既被启用又有可用设备的目标，则测试将回退，仅在 `llvm` 目标上运行。 

   * `TVM_LIBRARY_PATH` 应指向 `libtvm.so` 库的路径。例如，可用于使用调试版本来运行测试。如果未设置，将在 TVM 源码目录下自动查找 `libtvm.so`。


* **命令行参数**

   *  指定文件夹或文件路径只会运行该文件夹或文件中的单元测试。这在避免在没有特定前端环境的系统上运行 `tests/python/frontend` 下测试时很有用。 

   * `-m` 参数只运行带有特定 `pytest` 标记的测试。最常见的用法是使用 `-m gpu` 仅运行标记为 `@pytest.mark.gpu` 的 GPU 测试。也可以使用 `-m 'not gpu'` 来运行不使用 GPU 的测试。
   
   注意：此过滤是在基于环境变量 `TVM_TEST_TARGETS` 选择目标之后执行的。即使指定了 `-m gpu`，如果 `TVM_TEST_TARGETS` 中不包含 GPU 目标，也不会运行 GPU 测试。


## 在本地 Docker 容器中运行

可以使用 `docker/bash.sh` 脚本在与 CI 使用的相同 Docker 镜像中运行单元测试。第一个参数应指定要运行的 Docker 镜像（例如 `docker/bash.sh ci_gpu`）。允许的镜像名称在 TVM 源代码目录中的 Jenkinsfile 文件顶部定义，并映射到 [tlcpack](https://hub.docker.com/u/tlcpack) 上的镜像。


如果不提供额外参数，Docker 镜像将启动一个交互式 bash 会话。如果传入脚本作为可选参数（如 `docker/bash.sh ci_gpu tests/scripts/task_python_unittest.sh`），该脚本将在 Docker 镜像中被执行。


**注意：** Docker 镜像包含所有系统依赖项，但不包含该系统的 `build/config.cmake` 配置文件。TVM 源目录将作为 Docker 镜像的主目录，因此默认会使用本地的 config/build 目录。一个可行的做法是分别维护 `build_local` 和 `build_docker` 目录，在进入或退出 Docker 时将 `build` 符号链接到相应目录。


## 在 CI 中运行

CI 中的所有流程都起始于 Jenkinsfile 中定义的任务。这些定义包括指定使用的 Docker 镜像、编译时配置和每个阶段所运行的测试。


* Docker 镜像

Jenkinsfile 中的每个任务（如“BUILD: CPU”）都会调用 `docker/bash.sh`。紧随其后的参数定义了 CI 中使用的 Docker 镜像，方式与本地执行一致。


* 编译时配置

Docker 镜像中并不包含 `config.cmake` 文件，因此这是每个 `BUILD` 任务的第一步。此步骤通过执行 `tests/scripts/task_config_build_*.sh` 脚本完成。具体使用哪个脚本取决于正在测试的构建方式，并在 Jenkinsfile 中指定。

每个 `BUILD` 任务最终都会打包一个库文件，供后续测试阶段使用。


* 测试运行方式

Jenkinsfile 中的 `Unit Test` 和 `Integration Test` 阶段定义了如何调用 `pytest`。每个任务都以解压先前在 `BUILD` 阶段编译的库文件开始，随后运行一个测试脚本（如 `tests/script/task_python_unittest.sh`）。这些脚本会设置要传递给 `pytest` 的文件/目录和命令行参数。

多个测试脚本使用了 `-m gpu` 选项，用于限定仅运行带有 `@pytest.mark.gpu` 标记的测试用例。


