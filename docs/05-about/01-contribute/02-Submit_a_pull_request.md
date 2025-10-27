---

title: 提交 Pull Request

---

* [指南](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#%E6%8C%87%E5%8D%97)
* [提交信息指南](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#%E6%8F%90%E4%BA%A4%E4%BF%A1%E6%81%AF%E6%8C%87%E5%8D%97)
* [持续集成（CI）环境](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#ci-%E7%8E%AF%E5%A2%83)
* [测试](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#ci-%E7%8E%AF%E5%A2%83)
   * [ Docker（推荐）](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#docker%E6%8E%A8%E8%8D%90)
   * [C++（本地）](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#c%E6%9C%AC%E5%9C%B0)
   * [Python（本地）](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#python%E6%9C%AC%E5%9C%B0)


## 指南
*  我们建议开发者提交范围清晰的 PR，便于审查和在出现问题时撤销。因此，请避免将多个无关改动合并到单个 PR 中。
* 提交 PR 之前，请将你的代码基于最新的 `main` 分支进行 rebase，可以通过以下命令完成：

```plain
git remote add upstream [url to tvm repo]
git fetch upstream
git rebase upstream/main
```


确保代码通过代码风格检查（lint）：

```plain
# 虽然使用的 lint 命令应该和 CI 中一致，
# 但以下命令能精确复现 CI 的 lint 过程。
#（便于调试 lint 脚本错误或避免手动安装工具）：
python tests/scripts/ci.py lint


# 运行所有 lint 步骤：
docker/lint.sh


# 若需运行指定步骤，可在命令行传入步骤名称。拼写错误会打印出所有可用步骤：
# 如果步骤名称拼写错误，工具将显示所有可用的步骤。
docker/lint.sh <step_name> ...
```


如果 clang-format 检查未通过，可运行以下命令自动格式化：


```plain
# 运行 clang-format 从 upstream/main 检查所有改变的文件
docker/bash.sh ci_lint ./tests/lint/git-clang-format.sh --rev upstream/main
```
* 请为你引入的新功能或修复添加测试用例。 
*  为你编写的代码添加文档，详见 [文档指南](https://tvm.hyper.ai/docs/about/contribute/documentation)。
*  创建 [Pull Request](https://docs.github.com/zh/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)，并修复 CI 检查中发现的问题。 
* 邀请其他贡献者进行代码审查，并根据审查反馈改进补丁，可以通过 `@用户名` 的方式在 PR 中标记审查者。PR 标题中的标签会自动通知订阅用户，因此请确保标题包含相关标签（例如 `[microTVM] Add a cool change`，而不是 `a cool change for microTVM`）。更多关于标签和消息写法的内容请见下方的提交信息指南。
   *  为了更快获得审查，我们鼓励你也积极审查他人的代码，互帮互助。 
   *  代码审查是一个提升代码质量的过程，请积极对待，尽量在审查前完善你的代码。我们非常欢迎那些不需要反复审查就可以合并的补丁。 
   *  此指南总结了一些实用经验。 
*  当审查者通过 PR 后，该 PR 即可合并。 



## 提交信息指南

Apache TVM 使用 GitHub 进行补丁提交和代码审查，最终合并到主分支的提交信息（标题和正文）即为 PR 的标题和正文，因此需要根据审查反馈保持更新。


尽管这些规范主要针对 PR 的标题和正文，由于 GitHub 会自动从分支的提交中生成标题和正文，建议你从一开始写提交时就遵循这些规范，这样可以避免额外修改，也便于审查。


这些规范有助于统一格式，提升审查效率和代码库维护质量，例如便于搜索日志、回溯错误等。


PR/提交标题：
* 必须有标题（强制） 。
* 不要在标题中使用 GitHub 用户名（强制） 。
* 必须有标签，表示 PR 涉及的模块，如 [BugFix], [CI], [microTVM], [TVMC]。标签应使用驼峰式写法，如 [Fix], [Docker]，不要使用 [fix], [docker] 等。若多个标签，用多个方括号，如 [BugFix][CI] 。
* 使用祈使语气，如 “Add feature X”，而不是 “Added operator X” 。
* 正确使用大小写，如 Fix TVM use of OpenCL library 而不是 fix tvm use of opencl 
* 不要以句号结尾。


PR/提交正文：
* 必须有正文内容（强制） 。
* 不要在正文中使用 GitHub 用户名（强制） 。
* 避免使用「项目符号式」的提交信息正文：「项目符号式」（bullet）提交正文本身并不是问题，但如果只是简单地列出项目符号，而没有任何描述或解释，这种写法的效果和完全没有正文、理由或说明的提交没什么区别，同样糟糕。


对于与这些指南有轻微偏离的情况，社区通常会选择提醒贡献者遵守规范，而不是回滚或拒绝提交/拉取请求（PR）。


没有标题和/或正文的提交不被视为轻微偏离，因此必须避免。


最重要的是，提交信息的内容，尤其是正文部分，应该能够清晰传达修改的意图，因此应避免含糊其辞。例如，标题为 “Fix（修复）”、“Cleanup（清理）”、“Fix flaky test（修复不稳定测试）” 且正文为空的提交应避免使用。这样的提交会让审查者困惑于到底修复了什么，修改了什么，以及为何需要这个修改，从而拖慢审查进度。


下面是一个可供参考的良好提交信息示例：

```plain
[microTVM] Zephyr: Remove zephyr_board option from build, flash, and open_transport methods

Currently it’s necessary to pass the board type via ‘zephyr_board’ option to
the Project API build, flash, and open_transport methods.

However, since the board type is already configured when the project is
created (i.e. when the generate_project method is called), it’s possible to
avoid this redundancy by obtaining the board type from the project
configuration files.

This commit adds code to obtain the board type from the project CMake files,
removing this option from build, flash, and open_transport methods, so it’s
only necessary to specify the ‘zephyr_board’ option when calling
generate_project.

This commit also moves the ‘verbose’ and ‘west_cmd’ options from ‘build’
method to ‘generate_project’, reducing further the number of required options
when building a project, since the ‘build’ method is usually called more often
than the ‘generate_project’.
```


当一个新的 PR 被创建并开始审查时，审查者通常会提出修改意见。通常，作者会根据审查者的评论进行修改，并在初始提交的基础上追加新的提交。对于这些追加的提交，没有关于提交信息的特别建议。然而，如果这些追加的提交使得 PR 的标题和/或正文变得过时，那么作者有责任根据代码中的新改动同步更新 PR 的标题和正文（请记住，PR 的标题和正文将被用于生成最终提交信息，该信息将被合并至主干）。

提交者会在合并前尽可能修复提交信息中的问题，但他们有权提醒作者遵循相关规则，并鼓励其今后遵循。同时，他们也有权要求作者在 PR 标题和/或正文未被正确更新或修复时进行相应更新。


## CI 环境

我们使用 Docker 镜像来创建稳定的 CI 环境，这些环境可以被部署到多台机器上。

 请按照[此 issue 模板](https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-image.md&title=%5BCI+Image%5D+)中的步骤更新 CI Docker 镜像。


## 测试

尽管我们为每个 pull request 配置了自动运行的单元测试钩子，但我们仍强烈建议在本地预先运行单元测试，以减轻审查者负担并加快审查流程。


### Docker（推荐）

`tests/scripts/ci.py` 可在本地复现 CI 环境，并提供一个用户友好的界面。它使用与 CI 中相同的 Docker 镜像和脚本来直接运行测试。它还会将构建结果存放在不同文件夹中，使你可以保留多个测试环境而无需每次从零构建（例如，你可以在 CPU 和 i386 上分别测试变更，同时保留增量构建）。


```plain
# 查看所有可用平台。
python tests/scripts/ci.py --help
python tests/scripts/ci.py cpu --help


# 在 ci_cpu 容器中运行 CPU 构建（构建结果将保存在 build-cpu/ 文件夹中）。
# 注意：CPU 和 GPU 的 Docker 镜像体积较大，首次使用时可能需要较长时间下载。
python tests/scripts/ci.py cpu


# 在 ci_cpu 容器中运行 CPU 构建并运行单元测试。
python tests/scripts/ci.py cpu --unittest


# 快速迭代：运行特定测试并跳过每次重建。
python tests/scripts/ci.py cpu --skip-build --tests tests/python/tir-transform/test_tir_transform_inject_rolling_buffer.py::test_upscale


# 在容器中运行 CPU 构建并进入交互式 shell。
python tests/scripts/ci.py cpu --interactive
```


我们定期更新 Docker 镜像，随着时间推移，过时的镜像可能会不必要地占用磁盘空间。你可以使用以下命令删除当前分支及其他工作区未使用的过时镜像：

```plain
docker/clear-stale-images.sh
```


请参考 `--help` 查看更多选项。


### C++（本地）

运行 C++ 测试需要安装 gtest，请参考 [启用 C++ 测试](https://docs.github.com/zh/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) 中的说明。


```plain
# 假设你位于 tvm 源码根目录。
TVM_ROOT=`pwd`

./tests/scripts/task_cpp_unittest.sh
```


### Python（本地）

必要的依赖项：

```plain
pip install --user pytest Cython
```


如果你想运行所有测试：

```plain

# 构建 tvm
make

./tests/scripts/task_python_unittest.sh
```


如果你想运行单个测试：


```plain
# 构建 tvm。
make


# 告诉 python tvm 相关库的路径。
export PYTHONPATH=python
rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

TVM_FFI=ctypes python -m pytest -v tests/python/unittest/test_pass_storage_rewrite.py


# 如果你只想运行某个测试，例如文件中的 test_all_elemwise 测试。
TVM_FFI=ctypes python -m pytest -v -k "test_all_elemwise" tests/python/frontend/tflite/test_forward.py
```


