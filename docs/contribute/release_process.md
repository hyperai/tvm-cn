---
title: 发版流程
sidebar_position: 9
---

TVM 中的版本 manager 角色意味着要负责以下这些事情：

* 准备发行说明
* 准备设置
* 准备候选版本
   * 截取版本分支
   * 通知社区时间
   * 在分支中做必要版本更新的代码修改
* 对版本投票
   * 创建候选版本
   * 召集投票和分类问题
* 完成并发布版本
   * 更新 TVM 网站
   * 完成发行说明
   * 宣布发布

## 准备发行说明

发行说明包含新功能、改进、错误修复、已知问题和弃用等。TVM 提供 [每月开发报告](https://discuss.tvm.ai/search?q=TVM%20Monthly%20%23Announcement)，收集每个月的开发进度。发行说明的编写者可能用得上这个。

建议在截取版本分支之前开一个 GitHub Issue 来收集发行说明初稿的反馈。

## 准备 GPG 密钥

如果已经上传了密钥，则可以跳过这部分。

参考 [https://www.apache.org/dev/openpgp.html#generate-key](https://www.apache.org/dev/openpgp.html#generate-key) 将生成的 gpg 密钥上传到公钥服务器。

通过 `gpg --export` 和 `gpg --import` 命令可以将 gpg 密钥传输到另一台机器上发布。

最后一步是使用你的代码签名密钥 [https://www.apache.org/dev/openpgp.html#export-public-key](https://www.apache.org/dev/openpgp.html#export-public-key) 更新 KEYS 文件。查看对 TVM 主分支及 ASF SVN 的更改，

``` bash
# 指定 --depth=files 参数将跳过检查已有文件夹
svn co --depth=files "https://dist.apache.org/repos/dist/dev/tvm" svn-tvm
cd svn-tvm
# 编辑 KEY 文件
svn ci --username $ASF_USERNAME --password "$ASF_PASSWORD" -m "Update KEYS"
# 更新 downloads.apache.org
svn rm --username $ASF_USERNAME --password "$ASF_PASSWORD" https://dist.apache.org/repos/dist/release/tvm/KEYS -m "Update KEYS"
svn cp --username $ASF_USERNAME --password "$ASF_PASSWORD" https://dist.apache.org/repos/dist/dev/tvm/KEYS https://dist.apache.org/repos/dist/release/tvm/ -m "Update KEYS"
```

## 截取一个候选版本

要截取一个候选版本，首先用选定的版本字符串截取一个分支。分支的名称应该使用基本发行版本号，而不是patch号。例如，要为 v0.11.0 截取一个候选版本，分支应该命名为 v0.11，标签为`v0.11.0.rc0`,一旦截取便推送到对应分子的HEAD。

``` bash
	git clone https://github.com/apache/tvm.git
	cd tvm/

	# 更新版本号
	# ...
	git add .
	git commit -m "Bump version numbers to v0.6.0"

	# 相关版本替换 v0.6
	git branch v0.6
	git push --set-upstream origin v0.6

	git tag v0.6.0.rc0
	git push origin refs/tags/v0.6.0.rc0
```

（*确保源代码中的版本号正确。*运行 `python3 version.py` 进行版本更新。）

转到 GitHub 仓库的 "releases" 选项卡，然后单击 "Draft a new release"，
-   检查版本号并确保 TVM 可以构建和运行单元测试来验证发行版本
-   以 “v1.0.0.rc0” 的形式提供发布标签，其中 0 表示它是第一个候选版本。标签必须严格匹配此模式：`v[0-9]+\.[0-9]+\.[0-9]+\.rc[0-9]`
-   单击 Target 选择提交：branch \> Recent commits \> \$commit_hash
-   将发行说明初稿复制并粘贴到说明框中
-   选择 “This is a pre-release”
-   点击 “Publish release”

注意：截取后 BRANCH 仍可以更改，而 TAG 是固定的。如果此版本要做任何更改，则必须创建一个新的 TAG。

删除以前的候选版本（如果有的话），

``` bash
git push --delete origin v0.6.0.rc1
```

创建源代码工程，

``` bash
git clone git@github.com:apache/tvm.git apache-tvm-src-v0.6.0.rc0
cd apache-tvm-src-v0.6.0.rc0
git checkout v0.6
git submodule update --init --recursive
git checkout v0.6.0.rc0
rm -rf .DS_Store
find . -name ".git*" -print0 | xargs -0 rm -rf
cd ..
brew install gnu-tar
gtar -czvf apache-tvm-src-v0.6.0.rc0.tar.gz apache-tvm-src-v0.6.0.rc0
```

使用 GPG 密钥对创建的工程进行签名。首先确保 GPG 使用正确的私钥，

``` bash
$ cat ~/.gnupg/gpg.conf
default-key F42xxxxxxxxxxxxxxx
```

创建 GPG 签名以及文件的哈希，

``` bash
gpg --armor --output apache-tvm-src-v0.6.0.rc0.tar.gz.asc --detach-sig apache-tvm-src-v0.6.0.rc0.tar.gz
shasum -a 512 apache-tvm-src-v0.6.0.rc0.tar.gz > apache-tvm-src-v0.6.0.rc0.tar.gz.sha512
```

## 更新 `main` 上的 TVM 版本

在截取一个发行候选版本后，务必在整个 main 上更新版本号。例如，如果我们发布了 `v0.10.0`，我们希望将代码库中的版本号从 `v0.10.dev0` 提升到 `v0.11.dev0`。如何执行此操作的示例可以在此处找到：https://github.com/apache/tvm/pull/12190。在最后的一个包含开发标签（例如 `v0.11.dev0`）准备发行时立即在 `main` 上的提交标签。此标签是必须的，以便每晚从 main 构建的包具有正确的版本号。

## 上传候选版本

编辑 GitHub 上的发布页面并上传前面步骤创建的工程。

版本 manager 还需要将工程上传到 ASF SVN，

``` bash
# 指定 --depth=files 参数将跳过检查已有文件夹
svn co --depth=files "https://dist.apache.org/repos/dist/dev/tvm" svn-tvm
cd svn-tvm
mkdir tvm-v0.6.0-rc0
# 将文件复制到其中
svn add tvm-0.6.0-rc0
svn ci --username $ASF_USERNAME --password "$ASF_PASSWORD" -m "Add RC"
```

## 筛选（cherry-pick）
在截取了一个发行分支后还没被投票之前，发行管理员可以从 main 分支中 cherry-pick 提交。由于 GitHub 保护发行分支，要将这些修复合并到发行分支（例如`v0.11`），发行管理员必须向发行分支提交一个 PR，并包含被 cherry-pick 的更改。该 PR 应大致与从`main`分支提交的原始 PR 相匹配，并附带有关为何 cherry-pick 该提交的额外细节。然后，社区会对这些 PR 进行标准的审查并合并。请注意，针对发行分支的这些 PR 必须`[签名/](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)。


## 对候选版本投票

第一次投票在 Apache TVM 开发者名单（[dev@tvm.apache.org](mailto:dev@tvm.apache.org)）上进行。为了获得更多关注，可以创建一个以 “[VOTE]” 开头的 GitHub Issue，它会自动镜像到 dev@。可以查看以前的投票帖子来了解它是如何进行的。电子邮件应遵循以下格式：

-   在电子邮件中提供版本说明初稿的链接
-   提供候选版本工程的链接
-   确保电子邮件为文本格式且链接正确

对于 dev@ 投票，必须至少有 3 个约束性 +1 投票，并且 +1 投票必须多于 -1 投票。投票完成后，发送一封包含总数的摘要电子邮件，主题类似于 \[VOTE\]\[RESULT\] \...。

在 ASF 中，投票“至少”开放 72 小时（3 天）。如果在这段时间内没有获得足够数量的约束性投票，将无法在投票截止日期关闭它，需要延期投票。

如果投票失败，社区需要相应地修改版本，创建新的候选版本并重新投票。

## 发布版本

投票通过后，要将二进制文件上传到 Apache 镜像，请将二进制文件从 dev 目录（这应该是它们被投票的地方）移动到发布目录。这种“移动”是将内容添加到实际发布目录的唯一方法。 （注：只有 PMC 可以移动到发布目录）

``` bash
export SVN_EDITOR=vim
svn mkdir https://dist.apache.org/repos/dist/release/tvm
svn mv https://dist.apache.org/repos/dist/dev/tvm/tvm-v0.6.0-rc2 https://dist.apache.org/repos/dist/release/tvm/tvm-v0.6.0

# 如果您已将签名密钥添加到 KEYS 文件中，请同时更新发布副本。
svn co --depth=files "https://dist.apache.org/repos/dist/release/tvm" svn-tvm
curl "https://dist.apache.org/repos/dist/dev/tvm/KEYS" > svn-tvm/KEYS
(cd svn-tvm && svn ci --username $ASF_USERNAME --password "$ASF_PASSWORD" -m"Update KEYS")
```

记得在 GitHub 上创建一个新版本 TAG（本例中为 v0.6.0）并删除预发布候选 TAG。

``` bash
git push --delete origin v0.6.0.rc2
```

## 更新 TVM 网站

网站仓库位于 [https://github.com/apache/tvm-site](https://github.com/apache/tvm-site)。向下载页面中添加版本工程以及 GPG 签名和 SHA 哈希。

之后，更新[下载页面](https://tvm.apache.org/download)提供最新发行版本。如何操作可参考[此处/](https://github.com/apache/tvm-site/pull/38)。

## 发布公告

向 [announce@apache.org](mailto:announce@apache.org) 和 [dev@tvm.apache.org](mailto:dev@tvm.apache.org) 发送公告邮件。公告应包括发布说明和下载页面的链接。

## 发布Patch
发布Patch应为修复关键错误而被保留。发布Patch必须经过与正常发行相同的流程，但发行管理员可以酌情选择缩短为期 24 小时的候选版本投票窗口，以确保修复迅速交付。每个修补版本应提升发布基础分支（例如`v0.11`）上的版本号，并为发布候选版本创建标签（`例如v0.11.1.rc0`）。