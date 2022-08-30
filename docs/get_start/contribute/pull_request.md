---
title: 提交 Pull Request
sidebar_position: 2
---

## 准则

-  建议作者发送范围明确的 PR，以便在出现问题的时候便于 review 和 revert。因此，作者应该避免将多个不相关的修改 merge 到一个 PR 中。
-  提交 PR 之前，请将代码重新建立在 `main` 的最新版本上，运行以下代码：

    ``` bash
    git remote add upstream [url to tvm repo]
    git fetch upstream
    git rebase upstream/main
    ```

-   确保代码通过 lint 检查

    > ``` bash
    > # 虽然使用的 lint 命令应该与在 CI 中运行的相同，但此命令会重现
    > # 准确的 CI lint 过程（通常有助于调试 lint 脚本错误或避免手动安装工具）
    > python tests/scripts/ci.py lint
    >
    > # 运行所有的 lint 步骤。
    > docker/lint.sh
    >
    > # 要单独运行步骤，请在命令行中指定步骤名称。
    > # 一个不正确拼写的步骤名称会导致工具打印所有可用的步骤。
    > docker/lint.sh <step_name> ...
    > ```
    >
    > 若 clang-format lint 检查失败，运行 git-clang-format 自动对代码重新格式化：
    >
    > ``` bash
    > # 运行 clang-format 检查所有自 upstream/main 以来改变的文件
    > docker/bash.sh ci_lint ./tests/lint/git-clang-format.sh upstream/main
    > ```

-   添加测试用例，以涵盖补丁所引入的新功能或错误修复。

-   记录新写入的代码，更多信息请见 [文档](document)

-   创建一个 [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)，修复 CI 检查报告的问题。

-   请求其他贡献者帮你 review 代码，并根据其在 pull request 中用 `@` 标记的评论，来改进补丁。PR 标题中的标签会自动标记订阅用户，所以请确保在 PR 标题中加入并强调相关主题（例如：`[microTVM] 一个很酷的变化`，而不是 `microTVM 的一个很酷的变化`）。

    -   为了加快代码 review 进程，推荐 committer 之间彼此帮助 review 代码。
    -   代码 review 是一个引导过程，有助于提高贡献者的代码质量。我们应该积极主动地对待它，在 review 前尽可能地改进代码。我们高度重视那些不需要大量 review 就能进入的补丁。
    -   详细的指导方针，并总结实用的经验。

-   reviewer 批准 PR 后，才可以 merge。

## CI 环境

使用 Docker 镜像，创建能部署到多台机器上的稳定的 CI 环境。按照 [这个 issue 模板](https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-image.md&title=%5BCI+Image%5D+) 的步骤，更新 CI Docker 镜像。

## 测试 {#pr-testing}

尽管每个 pull request 都有自动运行单元测试的 hook，但推荐先在本地运行单元测试，以减少 reviewer 的负担并加快 review 过程。

### Docker（推荐）

`tests/scripts/ci.py` 在本地复制 CI 环境，并提供一个用户友好界面。在 CI 中使用的 Docker 镜像和脚本，可以直接用于运行测试。它是在不同的文件夹中保存构建的，这便于维护多个测试环境，无需每次都从头开始重建（例如，你可以测试 CPU 和 i386 的变化，同时保留增量重建）。

``` bash
# 查看所有可用的平台
python tests/scripts/ci.py --help
python tests/scripts/ci.py cpu --help

# 在 ci_cpu docker 容器中运行 CPU 构建（构建将被留在 build-cpu/ 文件夹中)
# 注意：CPU 和 GPU 的 Docker 镜像相当大，可能在第一次使用时需要一些时间来下载
python tests/scripts/ci.py cpu

# 在 ci_cpu docker 容器中运行 CPU 构建，然后运行 unittests
python tests/scripts/ci.py cpu --unittest

# 通过运行特定的测试快速迭代，并跳过每次的重建
python tests/scripts/ci.py cpu --skip-build --tests tests/python/unittest/test_tir_transform_inject_rolling_buffer.py::test_upscale

# 运行 CPU 构建，并将其放入容器中的一个 shell 中
python tests/scripts/ci.py cpu --interactive
```

我们会定期更新 Docker 镜像，随着时间的推移，陈旧的镜像可能造成磁盘空间的浪费。您可以使用以下命令删除当前检出的分支以及任何其他工作树中未使用的陈旧 Docker 镜像：

``` bash
docker/clear-stale-images.sh
```

有关更多选项，请参阅 --help。

### C++（本地）

运行 C++ 测试需要安装 gtest，按照 [启用 C++ 测试](../install/from_source#C++_tests) 中的说明进行安装

``` bash
# 假设您是在 tvm 源码根目录下
TVM_ROOT=`pwd`

./tests/scripts/task_cpp_unittest.sh
```

### Python（本地）

必要的依赖：

``` bash
pip install --user pytest Cython synr
```

如果希望运行所有的测试：

``` bash
# 构建 tvm
make

./tests/scripts/task_python_unittest.sh
```

If you want to run a single test:

``` bash
# 构建 tvm
make

# 让 python 知道在哪里可以找到 tvm 相关的库
export PYTHONPATH=python
rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

TVM_FFI=ctypes python -m pytest -v tests/python/unittest/test_pass_storage_rewrite.py

# 另外，如果您想运行单一的测试，例如在一个文件内的 test_all_elemwise。
TVM_FFI=ctypes python -m pytest -v -k "test_all_elemwise" tests/python/frontend/tflite/test_forward.py
```
