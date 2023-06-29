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

参考 <https://www.apache.org/dev/openpgp.html#generate-key> 将生成的 gpg 密钥上传到公钥服务器。

通过 `gpg --export` 和 `gpg --import` 命令可以将 gpg 密钥传输到另一台机器上发布。

最后一步是使用你的代码签名密钥 <https://www.apache.org/dev/openpgp.html#export-public-key> 更新 KEYS 文件。查看对 TVM 主分支及 ASF SVN 的更改，

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

要创建一个候选发行版本，首先需要使用选定的版本字符串创建一个分支。该分支的命名应该使用基本发行版本，而不包括补丁号。例如，要为`v0.11.0`创建一个候选版本，分支应该命名为`v0.11`，并在创建分支后将名为`v0.11.0.rc0`的标签(tag)推送到该分支的HEAD位置。

``` bash
git clone https://github.com/apache/tvm.git
cd tvm/

# 更新版本号
# ...
git add .
git commit -m "Bump version numbers to v0.6.0"

# 替换这里的"v0.6"为相应的版本号
git branch v0.6
git push --set-upstream origin v0.6

git tag v0.6.0.rc0
git push origin refs/tags/v0.6.0.rc0
```

要确保源代码中的版本号正确（例如：https://github.com/apache/tvm/pull/14300）。运行 `python3 version.py` 来更新版本。推送一个候选发行版本分支后，应立刻更新版本号。

转到 GitHub 仓库的 "releases" 选项卡，然后单击 "Draft a new release"，

-   检查版本号，并确保 TVM 可以构建和运行单元测试
-   以 “v1.0.0.rc0” 的形式提供发布标签，其中“0”表示它是第一个候选版本（该标签必须明确符合`v[0-9]+\.[0-9]+\.[0-9]+\.rc[0-9]`的形式）
-   单击 Target 选择提交：branch \> Recent commits \> \$commit_hash
-   将发行说明初稿复制并粘贴到说明框中
-   选择 “This is a pre-release”
-   点击 “Publish release”

注意：截取后分支(branch)仍可以更改，而标签(tag)是固定的。如果此版本要做任何更改，则必须创建一个新的标签。

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

## 更新`main`分支上的TVM版本

在创建了一个发行候选版本之后，请确保在整个`main`分支下的代码中更新版本号。例如，如果我们正发布的是`v0.10.0`版本，我们希望将代码库中的版本号从`v0.10.dev0`提升到`v0.11.dev0`。这里是一个有关如何进行此操作的[示例](https://github.com/apache/tvm/pull/12190)。
在包含在发布版本分支中的最后一个提交之后，立即给`main`分支上的新提交打上dev标签（例如`v0.11.dev0`），以用于下一次发布新版本。这样从`main`构建的每日构建版本才会具有正确的版本号。

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

## 选择性合并（Cherry-Picking）

在创建了发布分支但尚未对发布进行投票时，发布管理者可以从`main`分支上选择性地合并提交。由于发布分支在GitHub上受到保护，为了将这些修复合并到发布分支（例如 `v0.11`），发布管理者必须针对发布分支提交一个包含选择性合并更改的PR。该PR应与原本来自`main`分支的PR大致匹配，并附带有关为何选择性合并该提交的额外详情。社区随后对这些PR进行惯例的审查合并流程。针对发布分支的这些PR必须经过[签署](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)。

## 对候选版本投票

第一次投票在 Apache TVM 开发者名单（<dev@tvm.apache.org>）上进行。为了获得更多关注，可以创建一个以 “[VOTE]” 开头的 GitHub Issue，它会自动镜像到 dev@。可以查看以前的投票帖子来了解它是如何进行的。电子邮件应遵循以下格式：

-   在电子邮件中提供版本说明初稿的链接
-   提供候选版本工程的链接
-   确保电子邮件为文本格式且链接正确

对于 dev@ 投票，必须至少有 3 个约束性 +1 投票，并且 +1 投票必须多于 -1 投票。投票完成后，发送一封包含总数的摘要电子邮件，主题类似于 \[VOTE\]\[RESULT\] \...。

在 ASF 中，投票至少开放 72 小时（3 天）。如果在这段时间内没有获得足够数量的约束性投票，将无法在投票截止日期关闭它，需要延期投票。

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

网站仓库位于 <https://github.com/apache/tvm-site>。向下载页面中添加版本工程以及 GPG 签名和 SHA 哈希。
TVM的文档是持续更新的，所以需要上传一份相应版本的发行文档。如在你更新网站时，CI已从release中删除了文档，你可以在Jenkins上重新启动发行分支的CI构建。示例代码如下。

``` bash
git clone https://github.com/apache/tvm-site.git
pushd tvm-site
git checkout asf-site
pushd docs

# 创建发行文档目录
mkdir v0.9.0
pushd v0.9.0

# 从CI下载发行文档
# 在最近一次构建的发行分支的CI日志中找到这个URL
curl -LO https://tvm-jenkins-artifacts-prod.s3.us-west-2.amazonaws.com/tvm/v0.9.0/1/docs/docs.tgz
tar xf docs.tgz
rm docs.tgz

# 添加文档并推送
git add .
git commit -m "Add v0.9.0 docs"
git push
```

其后，修改[下载页](https://tvm.apache.org/download)来提供最新的release版本。这是一个[例子](https://github.com/apache/tvm-site/pull/38)。

## 发布公告

向 <announce@apache.org> 和 <dev@tvm.apache.org> 发送公告邮件。公告应包括发布说明和下载页面的链接。

## 发布补丁

为及时修复严重错误，我们保留了补丁版本的发布可能。补丁版本的发布流程必须与正常版本相同。
发布管理者自行考量这些补丁的发行。这些候选发行版本的投票窗口被缩减至24小时，以确保能快速交付这些修复。每个补丁版本应该在其基础发行分支（例如 ``v0.11``）和为候选发行版本创建的标签（例如 ``v0.11.1.rc0``）上增加版本号。
