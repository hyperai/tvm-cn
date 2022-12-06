---
title: 使用 TVM 的 CI
sidebar_position: 8
---

# 使用 TVM 的 CI

TVM 用 Jenkins 在 [分支](https://ci.tlcpack.ai/job/tvm/)上运行 Linux 持续集成（CI）测试，并通过 [Jenkinsfile](https://github.com/apache/tvm/blob/main/Jenkinsfile) 中指定的构建配置 [pull request](https://ci.tlcpack.ai/job/tvm/view/change-requests/)。Windows 和 MacOS 的非关键任务在 GitHub Actions 中运行。

本页描述了贡献者和 committer 如何用 TVM 的 CI 来验证代码。可通过 [tlc-pack/ci](https://github.com/tlc-pack/ci) 仓库了解有关 TVM CI 设计的更多信息。

## 对 Contributor 而言

[Jenkins 的 BlueOcean 查看器](https://ci.tlcpack.ai/blue/organizations/jenkins/tvm/activity) 中的标准 CI 运行如下所示。 CI 运行通常需要几个小时才能完成，并且在 CI 完成之前无法 merge pull request（PR）。要诊断失败的步骤，请单击 failing pipeline stage，然后单击 failing step 来查看输出日志。

![The Jenkins UI for a CI run](https://github.com/tlc-pack/web-data/raw/main/images/contribute/ci.png)

### 调试失败

当 CI 由于某种原因失败时，诊断问题的方法有如下几种。

#### Jenkins 日志

失败了首先按照失败作业上的红色 X 查看 CI 日志。注意：

* Jenkins 默认不显示完整日志，在日志查看器的顶部有一个“Show complete log”按钮，点击可以查看纯文本日志。
* `pytest` 失败总结在日志的底部，但可能需要向上滚动查看，才能知道失败的实际原因。

#### 重现失败

大多数 TVM Python 测试可以按照 [Testing](pull_request#pr-testing) 中的描述在 `pytest` 下运行。

### 提交 Issue

[在 GitHub 上报告](https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-problem.md&title=%5BCI+Problem%5D+) CI 的 issue，应该提供相关工作、commit 或 PR 的链接。

## 对 Maintainer 而言

本节讨论 TVM maintainer 的工作流程。

### 让 CI 保持正确的流程

本节讨论让 CI 正常运行的一般流程。

#### 同时合并导致 CI 失败

开发者在 merge 前依靠 TVM CI 来获取 PR 的信号。有时，两个不同的 PR 能分别通过 CI，但同时会破坏 `main`。反过来，又会导致错误显示在，基于失败 commit 的无关 PR 上。可从 [GitHub](https://github.com/apache/tvm/commits/main) 的 commit 状态图标或通过 [Jenkins](https://ci.tlcpack.ai/blue/organizations/jenkins/tvm/activity?branch=main) 来查找失败的 commit。

这些情况由 committer 负责，也鼓励其他人去帮助他们。这种情况的常见处理方式有：

1. 回退有问题的 commit
2. 提交一个正向修复以解决问题

选择哪个选项取决于 committer 和 commit 作者。失败的 CI 会影响所有 TVM 开发者，所以应尽快修复。而当 PR 很大时，对其作者来说，回退尤其痛苦。

### 处理 Flakiness

如果 PR 上的失败看起来与你的更改无关，并且没有看到任何有关失败的报告，你可以搜索 [与 flaky 测试相关的最新 GitHub Issue](https://github.com/apache/tvm/issues?q=is%3Aissue+%5BCI+%E9%97%AE%E9%A2%98%5D+Flaky+%3E) 和 [提交新 Issue](https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-problem.md&title=%5BCI+Problem%5D+%3E) 来寻找解决方案。如果某个测试或者某类测试在多个 PR 或 main 上的 commit 引发了 flaky 失败，则应通过带有 [strict=False](https://docs.pytest.org/en/6.2.x/skipping.html#strict-parameter) 参数的 [pytest 的 @xfail 装饰器](https://docs.pytest.org/en/6.2.x/skipping.html#xfail-mark-test-functions-as-expected-to-fail) 和禁用 PR 链接的相关 issue 来禁用这个测试或者这类测试。

``` python
@pytest.mark.xfail(strict=False, reason="Flaky test: https://github.com/apache/tvm/issues/1234")
def test_something_flaky():
    pass
```

然后照常提交 PR：

``` bash
git add <test file>
git commit -m'[skip ci][ci] Disable flaky test: ``<test_name>``

See #<issue number>
'
gh pr create
```

### 跳过 CI

对于回退和小的正向修复，将 `[skip ci]` 添加到回退的 PR 标题会让 CI 仅运行 lint。committer 应该注意，他们 merge 跳过 CI 的 PR，只是为了修复 `main` 上的失败 ，而非 submitter 想用 CI 更快 merge 更改。首次构建时会检查 PR 标题（尤其是在 lint 时，所以此后的更改不影响 CI，并且需要另一个 `git push` 重新触发该任务）。

``` bash
# 回退最近一次提交，确保在提交的开头插入 '[skip ci]'
git revert HEAD
git checkout -b my_fix
# 在你推送分支后，照常创建 PR
git push my_repo
# 示例：在已有 PR 的分支上跳过 CI
# 将这个提交添加到已有的分支上会导致新的 CI 跳过 Jenkins 运行
git commit --allow-empty --message "[skip ci] Trigger skipped CI"
git push my_repo
```

### Docker 镜像

所有 CI 任务都在 Docker 容器（由 [docker/](https://github.com/apache/tvm/tree/main/docker) 文件夹中的文件构建）中运行其大部分的工作。这些文件通过 [docker-images-ci](https://ci.tlcpack.ai/job/docker-images-ci/) 任务每日在 Jenkins 中构建。这些容器的镜像在 [tlcpack Docker Hub](https://hub.docker.com/u/tlcpack) 中托管，并在 [Jenkinsfile.j2](https://github.com/apache/tvm/tree/main/Jenkinsfile.j2) 中引用。这些镜像可用标准的 Docker 命令在本地检查和运行。

### ci-docker-staging

[ci-docker-staging](https://github.com/apache/tvm/tree/ci-docker-staging) 分支对 Docker 镜像的更新和 `Jenkinsfile` 的更改进行测试。当构建从 fork 的仓库得到的 PR 时，Jenkins 使用除了`Jenkinsfile` 本身（来自基本分支）之外的 PR 中的代码。由于构建分支时会使用分支中的 `Jenkinsfile`，所以具有写权限的 committer 必须将 PR 推送到 apache/tvm 中的分支，从而正确测试 `Jenkinsfile` 更改。如果 PR 修改了 `Jenkinsfile`，必须 @ [committer](https://github.com/apache/tvm/tree/main/CONTRIBUTORS.md)，并要求他们把你的 PR 作为分支推送，从而测试更改。

### CI 监控轮换

有些测试也很不稳定，会因为与 PR 无关的原因而失败。 [CI 监控轮换](https://github.com/apache/tvm/wiki/CI-Monitoring-Runbook) 对这些故障进行监控，并在必要时禁用这些测试。编写测试的人负责修复好这些测试，并重新启用它们。
