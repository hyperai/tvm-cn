---
title: 发布流程
sidebar_position: 9
---

The release manager role in TVM means you are responsible for a few
different things:

-   Preparing release notes
-   Preparing your setup
-   Preparing for release candidates
    -   Cutting a release branch
    -   Informing the community of timing
    -   Making code changes in that branch with necessary version
        updates
-   Running the voting process for a release
    -   Creating release candidates
    -   Calling votes and triaging issues
-   Finalizing and posting a release:
    -   Updating the TVM website
    -   Finalizing release notes
    -   Announcing the release

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

要截取一个候选版本，首先用选定的版本字符串截取一个分支，例如，

``` bash
git clone https://github.com/apache/tvm.git
cd tvm/
git branch v0.6.0
git push --set-upstream origin v0.6.0
```

（*确保源代码中的版本号正确。*运行 `python3 version.py` 进行版本更新。）

转到 GitHub 仓库的 "releases" 选项卡，然后单击 "Draft a new release"，

-   以 “v1.0.0.rc0” 的形式提供发布标签，其中 0 表示它是第一个候选版本
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

## 对候选版本投票

第一次投票在 Apache TVM 开发者名单（<dev@tvm.apache.org>）上进行。为了获得更多关注，可以创建一个以 “[VOTE]” 开头的 GitHub Issue，它会自动镜像到 dev@。可以查看以前的投票帖子来了解它是如何进行的。电子邮件应遵循以下格式：

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

> ``` bash
> git push --delete origin v0.6.0.rc2
> ```

## 更新 TVM 网站

网站仓库位于 <https://github.com/apache/tvm-site>。向下载页面中添加版本工程以及 GPG 签名和 SHA 哈希。

## 发布公告

向 <announce@apache.org> 和 <dev@tvm.apache.org> 发送公告邮件。公告应包括发布说明和下载页面的链接。
