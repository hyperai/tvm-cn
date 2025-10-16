---

title: 使用 TVM 的 CI

---

* [贡献者指南](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips#%E5%BC%BA%E5%88%B6%E6%8E%A8%E9%80%81%E7%9A%84%E5%90%8E%E6%9E%9C)
   * [调试失败](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips#%E5%BC%BA%E5%88%B6%E6%8E%A8%E9%80%81%E7%9A%84%E5%90%8E%E6%9E%9C)
      * [Jenkins 日志](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#jenkins-%E6%97%A5%E5%BF%97)
      * [重现失败](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E5%A4%8D%E7%8E%B0%E5%A4%B1%E8%B4%A5)
   * [报告问题](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E6%8A%A5%E5%91%8A%E9%97%AE%E9%A2%98)
* [维护者](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E7%BB%B4%E6%8A%A4%E8%80%85%E6%8C%87%E5%8D%97)[指南](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E7%BB%B4%E6%8A%A4%E8%80%85%E6%8C%87%E5%8D%97))
   * [保持 CI 通过的常规操作](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E4%BF%9D%E6%8C%81-ci-%E9%80%9A%E8%BF%87%E7%9A%84%E5%B8%B8%E8%A7%84%E6%93%8D%E4%BD%9C)
      * [同时合并导致的 CI 中断](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E5%90%8C%E6%97%B6%E5%90%88%E5%B9%B6%E5%AF%BC%E8%87%B4%E7%9A%84-ci-%E4%B8%AD%E6%96%AD)
   * [处理不稳定测试](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E5%A4%84%E7%90%86%E4%B8%8D%E7%A8%B3%E5%AE%9A%E6%B5%8B%E8%AF%95)
   * [跳过 CI](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E8%B7%B3%E8%BF%87-ci)
   * [Docker 镜像](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#docker-%E9%95%9C%E5%83%8F)
      * [更新 Docker 镜像标签](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E6%9B%B4%E6%96%B0-docker-%E9%95%9C%E5%83%8F%E6%A0%87%E7%AD%BE)
      * [添加新的 Docker 镜像](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E6%B7%BB%E5%8A%A0%E6%96%B0-docker-%E9%95%9C%E5%83%8F)
   * [CI 监控轮值](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E6%B7%BB%E5%8A%A0%E6%96%B0-docker-%E9%95%9C%E5%83%8F)



TVM 主要使用 Jenkins 运行 Linux 持续集成(CI)测试，通过 [Jenkinsfile](https://github.com/apache/tvm/blob/main/ci/jenkins/templates/) 中定义的构建配置，对[分支](https://ci.tlcpack.ai/job/tvm/)和[拉取请求](https://ci.tlcpack.ai/job/tvm/view/change-requests/)进行测试。Jenkins 是唯一会阻止代码合并的 CI 步骤。TVM 还使用 GitHub Actions 对 Windows 和 MacOS 进行最小化测试。


本文档描述贡献者和维护者如何使用 TVM 的 CI 系统验证代码。更多关于 TVM CI 设计的详细信息可在 [tlc-pack/ci](https://github.com/tlc-pack/ci) 仓库中找到。

## 贡献者指南

标准 CI 运行流程可通过 [Jenkins BlueOcean 界面](https://ci.tlcpack.ai/blue/organizations/jenkins/tvm/activity)查看。CI 运行通常需要几小时完成，拉取请求(PR)在 CI 成功完成前无法合并。诊断失败步骤时，可点击失败管道阶段，再进入失败步骤查看输出日志。


![图片](/img/docs/v21/contribute_08-Using_TVM's_Ci_1.png)

### 调试失败

当 CI 失败时，有几种诊断方法：


#### Jenkins 日志

首先查看 CI 日志，跟随失败任务上的红色"X"标记查看日志。注意：
* Jenkins 默认不显示完整日志，日志查看器顶部有"Show complete log"按钮可查看纯文本版本。
* `pytest` 失败摘要位于日志底部，但通常需要向上滚动查看实际失败原因。


#### 复现失败

大多数 TVM Python 测试使用 `pytest` 运行，可按[测试指南](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E6%B7%BB%E5%8A%A0%E6%96%B0-docker-%E9%95%9C%E5%83%8F)操作复现。


### 报告问题

CI 问题应[在 GitHub 提交问题](https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-problem.md&title=%255BCI+Problem%255D+)，并附上相关任务、提交或 PR 链接。

## 维护者指南

本节讨论 TVM 维护者使用的流程。

### 保持 CI 通过的常规操作


本节讨论保持 CI 通过的超过操作。


#### 同时合并导致的 CI 中断

开发者依赖 TVM CI 在合并前验证 PR。偶尔两个独立通过的 PR 合并到 main 后会导致问题，进而影响其他无关 PR。可通过 [GitHub](https://github.com/apache/tvm/commits/main) 提交状态图标或 [Jenkins](https://ci.tlcpack.ai/blue/organizations/jenkins/tvm/activity?branch=main) 识别问题提交。


此时合并 PR 的 TVM 提交者有责任修复 CI（鼓励其他人协助）。典型解决方案：

1. 回退问题提交

2. 提交正向修复


由提交者和作者决定采用哪种方案。CI 中断会影响所有 TVM 开发者，应尽快修复，而回退对大 PR 作者影响较大。

### 处理不稳定测试

如果 PR 出现与更改无关的失败，应搜索[最近的不稳定测试问题](https://github.com/apache/tvm/issues?q=is%253Aissue+%255BCI+Problem%255D+Flaky+)，若无相关报告则[新建问题](https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-problem.md&title=%255BCI+Problem%255D+)。如果某测试在多个 PR 或 main 提交中频繁失败，应使用 [pytest 的 @xfail 装饰器](https://docs.pytest.org/en/6.2.x/skipping.html#xfail-mark-test-functions-as-expected-to-fail)（设置 strict=False）禁用测试，并在禁用 PR 中链接相关问题。

```plain
@pytest.mark.xfail(strict=False, reason="Flaky test: https://github.com/apache/tvm/issues/1234")
    def test_something_flaky():
        pass
```


然后按常规提交 PR：

```plain
git add <test file>
git commit -m'[skip ci][ci] Disable flaky test: ``<test_name>``

See #<issue number>
'
gh pr create
```


### 跳过 CI

对于回退和简单正向修复，在 PR 标题添加 `[skip ci]` 会使 CI 只运行 lint。提交者应确保仅在修复 main 分支失败时合并跳过 CI 的 PR，而非为快速合并而跳过。PR 标题在构建首次运行时检查（特别是 lint 步骤），之后更改不会影响 CI，需要另一次 `git push` 重新触发。


```plain
# 回退 HEAD 提交，确保在提交主题开头插入'[skip ci]'。
git revert HEAD
git checkout -b my_fix

# 推送分支后按常规创建 PR。
git push my_repo

# 示例：在已有 PR 的分支上跳过 CI。
# 添加此提交将触发新的跳过 CI 运行。
git commit --allow-empty --message "[skip ci] Trigger skipped CI"
git push my_repo
```

### Docker 镜像

每个 CI 任务主要在 Docker 容器中运行，容器基于 [docker/](https://github.com/apache/tvm/tree/main/docker) 目录中的文件构建。


#### 更新 Docker 镜像标签

更新标签需要构建新镜像并上传到 Docker Hub，然后更新 [docker-images.ini](https://github.com/apache/tvm/tree/main/ci/jenkins/docker-images.ini) 中的标签以匹配 Docker Hub。


Docker 镜像通过 [tvm-docker](https://ci.tlcpack.ai/job/tvm-docker/) 每晚自动构建，通过 CI 后上传到 [tlcpackstaging](https://hub.docker.com/u/tlcpackstaging)。`main` 的合并后 CI 也会临时构建镜像上传到 `tlcpackstaging` 。存在从 `tlcpackstaging` 自动迁移到 `tlcpackstaging` 的流程，因此 CI 中可使用 `tlcpackstaging` 标签，成功合并后会自动迁移到 `tlcpack` 。更新步骤：



1. 合并更改 `docker/` 下 Dockerfiles 或 `docker/install` 脚本的 PR。

2. 等待：

   1. PR 的合并后 CI 完成并将新镜像上传到 [tlcpackstaging](https://hub.docker.com/u/tlcpackstaging)。

   2. 或等待每日镜像构建完成上传。

3. 在 [tlcpackstaging](https://hub.docker.com/u/tlcpackstaging) 找到新标签（如 `20221208-070144-22ff38dff`），更新 `ci/jenkins/docker-images.ini` 使用 tlcpack 账户下的 tlcpackstaging 标签（如 `tlcpack/ci-arm:20221208-070144-22ff38dff`）。提交 PR 等待 CI 验证新镜像。

4. 合并 `docker-images.ini` 更新 PR。main 的合并后 CI 完成后，`tlcpackstaging` 标签会自动重新上传到 `tlcpack。`


#### 添加新 Docker 镜像

可添加新 CI 镜像测试 TVM 在不同平台的表现。添加步骤：



1. 定义 `docker/Dockerfile.ci_foo` 和 `docker/install` 中的相关脚本。创建仅包含这些更改的 PR（无 `Jenkinsfile` 更改）。

      示例：[https://github.com/apache/tvm/pull/12230/files](https://github.com/apache/tvm/pull/12230/files)

2. 提交者验证镜像本地构建后审核/批准此 PR。

3. 提交者在  [https://hub.docker.com/u/tlcpack](https://hub.docker.com/u/tlcpack) 和 [https://hub.docker.com/u/tlcpackstaging](https://hub.docker.com/u/tlcpackstaging) 创建 ci-foo 仓库。

4. 在 tlcpack/ci 创建 ECR 仓库 PR：[https://github.com/tlc-pack/ci/pull/46/files](https://github.com/tlc-pack/ci/pull/46/files)。

5. 提交者创建并合并添加镜像到 `Jenkinsfile` 的 PR。

   1. 示例：[https://github.com/apache/tvm/pull/12369/files](https://github.com/apache/tvm/pull/12369/files)

   2. 注意：PR 必须从 apache/tvm 的分支创建，不能从 fork 仓库分支创建

6. 提交者将镜像添加到 tlcpack 的每日镜像重建/验证任务。

   1. 示例：[https://github.com/tlc-pack/tlcpack/pull/131](https://github.com/tlc-pack/tlcpack/pull/131)

### `ci-docker-staging`

[ci-docker-staging](https://github.com/apache/tvm/tree/ci-docker-staging) 分支通常用于测试 Docker 镜像更新和 `Jenkinsfile` 更改。从 fork 仓库的普通 PR 构建时，Jenkins 使用 PR 代码但 `Jenkinsfile` 来自基础分支。构建分支时使用分支中的 `Jenkinsfile`，因此有写权限的提交者必须将 PR 推送到 apache/tvm 的分支才能正确测试 `Jenkinsfile` 更改。如果 PR 更改 `Jenkinsfile`，请 @ [提交者](https://github.com/apache/tvm/tree/main/CONTRIBUTORS.md) 请求将你的 PR 作为分支推送测试。


### CI 监控轮值

部分测试不稳定可能因与 PR 无关的原因失败。[CI 监控轮值](https://github.com/apache/tvm/wiki/CI-Monitoring-Runbook) 监视这些失败并按需禁用测试。测试作者有责任最终修复并重新启用测试。


