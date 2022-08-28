---
title: Python Target 参数化
---

# Python Target 参数化

## 摘要

对于任何支持的 runtime，TVM 都应该输出正确的数字结果。因此，在编写验证数字输出的单元测试时，这些单元测试应该在所有支持的 runtime 上都能正常运行。由于这是一个非常常见的用例，TVM 的辅助函数可以对所有单元测试进行参数化，从而便于单元测试在所有启用并具有兼容设备的 target 上运行。

测试套件的单个 Python 函数，可以扩展为几个参数化单元测试，每个单元测试一个 target 设备。为了保证测试正常运行，以下所有条件必须为 True：

* 测试存在于已经传递给 *pytest* 的文件或目录中。
* 应用于函数的 pytest 标记，无论是显式还是通过 target 参数化，都必须与传递给 pytest 的 *-m* 参数的表达式兼容。
* 对于使用 target fixture 的参数化测试，target 必须出现在环境变量 *TVM_TEST_TARGETS* 中。
* 对于使用 *target* fixture 的参数化测试，config.cmake 中的构建配置必须启用相应的 runtime。

## 单元测试文件内容

在多个 target 上运行测试，推荐方法是通过参数化测试。对于一个固定的 target 列表，可以通过用 `@tvm.testing.parametrize_targets('target_1', 'target_2', ...)` 修饰同时接受 `target` 或 `dev` 作为函数参数来显式地完成。

该函数将为列出的每个 target 都运行一遍，并单独报告每个 target 的运行结果（成功/失败）。如果一个 target 因为在 *config.cmake* 中被禁用而无法运行，或者因为没有合适的硬件存在，那么这个 target 将被报告为跳过。

``` python
# 显式列出要使用的 target
@tvm.testing.parametrize_target('llvm', 'cuda')
def test_function(target, dev):
    # 测试代码写在这里
```

对于在所有 target 上都能正常运行的测试，可以省略装饰器。任何接收 `target` 或 `dev` 参数的测试，都将自动在 `TVM_TEST_TARGETS` 指定的所有 target 上进行参数化。参数化为每个 target 提供了相同的成功/失败/跳过报告，同时允许轻松扩展测试套件，以覆盖额外的 target。

``` python
# 隐式参数化以运行在所有 target 上
# 在环境变量 TVM_TEST_TARGETS 里
def test_function(target, dev):
    # 测试代码写在这里
```

`@tvm.testing.parametrize_targets` 也可以用作裸装饰器 (bare decorator) 来显式地进行参数化，但没有额外的效果。

``` python
# 隐式参数化以运行在所有 target 上
# 在环境变量 TVM_TEST_TARGETS 里
@tvm.testing.parametrize_targets
def test_function(target, dev):
    # 测试代码写在这里
```

可以使用 `@tvm.testing.exclude_targets` 或 `@tvm.testing.known_failing_targets` 装饰器，将特定 target 排除或标记为预期失败。更多信息，请参阅文档字符串。

在某些情况下，可能需要跨多个参数进行参数化。例如，可能存在一些待测试的 target-specific 实现方法，其中一些 target 的实现方法还不止一个。这可以通过显式地参数化参数元组来完成，如下所示。在这种情况下，只有显式地列出的 target 会运行，但它们仍会应用适当的 `@tvm.testing.requires_RUNTIME` 标记。

``` python
pytest.mark.parametrize('target,impl', [
    ('llvm', cpu_implementation),
    ('cuda', gpu_implementation_small_batch),
    ('cuda', gpu_implementation_large_batch),
])
def test_function(target, dev, impl):
    # 测试代码写在这里
```

参数化功能是在 pytest 标记之上实现的。每个测试函数都可以用 [pytest 标记](#) 装饰以包含元数据。最常用的标记如下：

* `@pytest.mark.gpu` - 将函数标记为使用 GPU 功能。这本身是没有效果的，但可以与命令行参数 `-m gpu` 或 `-m 'not gpu'` 搭配使用，从而限制 pytest 要执行哪些测试。这不应该单独调用，而应该是单元测试中使用的其他标记的一部分。
* `@tvm.testing.uses_gpu` - 应用 `@pytest.mark.gpu`。用于标记可能使用 GPU 的单元测试（如果有）。只有在显式循环 `tvm.testing.enabled_targets()` 的测试中，才需要这个装饰器，不过这已经不是编写单元测试的首选方法了（见下文）。使用 `tvm.testing.parametrize_targets()` 时，此装饰器对于 GPU target 是隐式的，不需要显式地应用。
* `@tvm.testing.requires_gpu` - 应用 `@tvm.testing.uses_gpu`，如果没有 GPU，还要标记这个测试应该被跳过 (`@pytest.mark.skipif`)。
* `@tvfm.testing.requires_RUNTIME` - 几个装饰器（例如 `@tvm.testing.requires_cuda`），如果指定 runtime 不可用，每个装饰器都会跳过测试。runtime 如果在 `config.cmake` 中被禁用，或是不存在兼容设备时，则该 runtime 不可用。对于使用 GPU 的 runtime，包含 `@tvm.testing.requires_gpu`。

使用参数化 target 时，每个测试运行都是用跟正在使用的 target 相对应的 `@tvm.testing.requires_RUNTIME` 修饰的。因此，如果某个 target 在 `config.cmake` 中被禁用，或没有合适的硬件可以运行，它将被显式列为跳过。

还有 `tvm.testing.enabled_targets()`，根据环境变量 `TVM_TEST_TARGETS`、构建配置和存在的物理硬件，返回所有在当前机器上启用和可运行的 target。大多数当前测试显式循环是 `enabled_targets()` 返回 target，但它无法应用于新测试。这种类型的 pytest 输出会自动跳过在 `config.cmake` 中禁用，或者没有运行设备的 runtime。此外，测试会在第一个失败的 target 上停止，这对于判断错误是发生在某一个还是所有 target 上，都很困难。

``` python
# 老的风格, 已弃用。
def test_function():
    for target,dev in tvm.testing.enabled_targets():
        # 测试代码写在这里
```

## 本地运行

要在本地运行 Python 单元测试，可以使用 `${TVM_HOME}` 目录中的命令 `pytest`。

* 环境变量
  
  * `TVM_TEST_TARGETS` 应该是一个用分号分隔的待运行 target 列表。如果未设置，默认是 `tvm.testing.DEFAULT_TEST_TARGETS` 中定义的 target。
    
    注意：如果 `TVM_TEST_TARGETS` 不包含任何已启用且具有该类型可访问设备的 target，则测试将回退到仅在 `llvm` target 上运行。

  * `TVM_LIBRARY_PATH` 应该是 `libtvm.so` 库的路径。例如，这可以用来借助调试版本运行测试。如果未设置，将搜索相对于 TVM 源目录的 `libtvm.so`。
  
* 命令行参数
  
  * 传递文件夹或文件的路径，将仅在该文件夹或文件中运行单元测试。这一点很实用，例如，避免在未安装特定前端的系统上，运行位于 `tests/python/frontend` 中的测试。
  
  * `-m` 参数仅运行带有特定 pytest 标记的单元测试。最常见的用法是使用 `m gpu` 仅运行标有 `@pytest.mark.gpu` 的测试，并使用 GPU 运行。通过传递 `m 'not gpu'`，它也可以用于仅运行不使用 GPU 的测试。
    
    注意：此过滤发生在基于 `TVM_TEST_TARGETS` 环境变量选定 target 之后。即使指定了 `-m gpu`，如果 `TVM_TEST_TARGETS` 不包含 GPU target，也不会运行任何 GPU 测试。

## 在本地 Docker 容器中运行

与在 CI 中的用法类似，`docker/bash.sh` 脚本可用于在同一 Docker 镜像中运行单元测试。第一个参数应指定要运行的 Docker 镜像（例如 `docker/bash.sh ci_gpu`）。允许的镜像名称在位于 TVM 源目录的 Jenkinsfile 顶部定义，并映射到 [tlcpack](https://hub.docker.com/u/tlcpack) 中的镜像。

如果没有给出额外的参数，Docker 镜像将被载入一个交互式 bash 会话。如果脚本作为可选参数传递（例如 `docker/bash.sh ci_gpu tests/scripts/task_python_unittest.sh`），则该脚本将在 Docker 镜像中执行。

注意：Docker 镜像包含所有系统依赖项，但不包括这些系统的 `build/config.cmake` 配置文件。 TVM 源目录用作 Docker 镜像的主目录，因此这将默认使用与本地配置相同的 config/build 目录。一种解决方案是单独维护 `build_local` 和 `build_docker` 目录，并在进入/退出 Docker 时，创建从 `build` 到相应文件夹的符号链接。

## 在 CI 中运行

CI 中的所有内容都从 Jenkinsfile 中的任务定义开始的。这包括定义使用哪个 Docker 镜像，编译时配置是什么，以及哪些阶段都各自包含哪些测试。

* Docker 镜像
  
  Jenkinsfile 的每个任务（例如 'BUILD: CPU'）都会调用 `docker/bash.sh`。调用 docker/bash.sh 后面的参数定义了 CI 中的 Docker 镜像，就本地类似。

* Compile-time 配置
  
  Docker 镜像没有内置 `config.cmake` 文件，因此这是每个 `BUILD` 任务的第一步。这一步是使用 `tests/scripts/task_config_build_*.sh` 脚本完成的。使用哪个脚本取决于正在测试的构建，还需要在 Jenkinsfile 中指定。

  每个 `BUILD` 任务都以打包一个供以后测试使用的库而结束。

* 运行哪些测试
  
  Jenkinsfile 的 `Unit Test` 和 `Integration Test` 阶段决定了如何调用 `pytest`。每个任务都是先解压一个编译库，这个库在先前的 `BUILD` 阶段已经编译过了。接下来运行测试脚本（如`tests/script/task_python_unittest.sh`）。这些脚本可以设定文件/文件夹，以及传递给 `pytest` 的命令行选项。
  
  其中一些脚本包含 `-m gpu` 选项，该选项将测试限制为仅运行包含 `@pytest.mark.gpu` 标记的测试。