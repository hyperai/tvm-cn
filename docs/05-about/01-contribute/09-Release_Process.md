---

title: 发布流程

---

* [准备发布说明](/docs/about/contribute/Release_Process#%E5%87%86%E5%A4%87%E5%8F%91%E5%B8%83%E8%AF%B4%E6%98%8E)
* [准备候选版本](/docs/about/contribute/Release_Process#%E5%87%86%E5%A4%87%E5%80%99%E9%80%89%E7%89%88%E6%9C%AC)
* [准备 GPG 密钥](/docs/about/contribute/Release_Process#%E5%87%86%E5%A4%87-gpg-%E5%AF%86%E9%92%A5)
* [创建候选版本](/docs/about/contribute/Release_Process#%E5%88%9B%E5%BB%BA%E5%80%99%E9%80%89%E7%89%88%E6%9C%AC%E5%88%86%E6%94%AF)
* [在 main 上更新 TVM 版本](/docs/about/contribute/Release_Process#%E5%9C%A8-main-%E4%B8%8A%E6%9B%B4%E6%96%B0-tvm-%E7%89%88%E6%9C%AC)
* [上传候选版本](/docs/about/contribute/Release_Process#%E4%B8%8A%E4%BC%A0%E5%80%99%E9%80%89%E7%89%88%E6%9C%AC)
* [拣选提交](/docs/about/contribute/Release_Process#%E6%8B%A3%E9%80%89%E6%8F%90%E4%BA%A4)
* [发起候选版本投票](/docs/about/contribute/Release_Process#%E5%8F%91%E8%B5%B7%E5%80%99%E9%80%89%E7%89%88%E6%9C%AC%E6%8A%95%E7%A5%A8)
* [完成发布](/docs/about/contribute/Release_Process#%E5%AE%8C%E6%88%90%E5%8F%91%E5%B8%83)
* [更新 TVM 官网](/docs/about/contribute/Release_Process#%E6%9B%B4%E6%96%B0-tvm-%E5%AE%98%E7%BD%91)
* [发布公告](/docs/about/contribute/Release_Process#%E5%8F%91%E5%B8%83%E5%85%AC%E5%91%8A)
* [补丁发布](/docs/about/contribute/Release_Process#%E8%A1%A5%E4%B8%81%E5%8F%91%E5%B8%83)


TVM 项目的发布管理员需要负责以下工作内容：
* 准备发布说明。
* 搭建发布环境。
* 准备候选版本。
   * 创建发布分支。
   * 向社区同步时间安排。
   * 在分支中完成必要的版本更新。
* 组织发布投票流程。
   * 创建候选版本。
   * 发起投票并跟踪问题处理。
* 完成最终发布。
   * 更新 TVM 官网。
   * 定稿发布说明。
   * 发布公告。


## 准备发布说明

发布说明应包含新功能、改进、错误修复、已知问题和弃用内容等。TVM 提供的 [月度开发报告](https://discuss.tvm.ai/search?q=TVM%2520Monthly%2520%2523Announcement) 汇总了每月的开发进展，可作为编写参考。

建议在创建发布分支前，先在 GitHub 上创建 issue 收集对发布说明草案的反馈。可参考 `tests/scripts/release` 目录下的脚本作为起点。


## 准备候选版本

在正式发布前，可能需要对发布分支进行必要的代码修改。请确保所有版本号都已更新。

## 准备 GPG 密钥

如果已上传密钥，可跳过本节。


生成 GPG 密钥后，需要将其上传到公共密钥服务器。详情请参考 [https://www.apache.org/dev/openpgp.html#generate-key](https://www.apache.org/dev/openpgp.html#generate-key)。


如果需要在其他机器上进行发布，可以通过 `gpg --export` 和 `gpg --import` 命令迁移密钥。


最后一步是使用代码签名密钥更新 KEYS 文件：[Apache 公钥导出指南](https://www.apache.org/dev/openpgp.html#export-public-key)。将更改提交到 TVM 主分支以及 ASF SVN：


```plain
# --depth=files 可避免检出已有文件夹  。
svn co --depth=files "https://dist.apache.org/repos/dist/dev/tvm" svn-tvm
cd svn-tvm

# 编辑 KEYS 文件 。 
svn ci --username $ASF_USERNAME --password "$ASF_PASSWORD" -m "Update KEYS"

# 更新 downloads.apache.org（注意：只有 PMC 成员可更新 dist/release 目录）。
svn rm --username $ASF_USERNAME --password "$ASF_PASSWORD" https://dist.apache.org/repos/dist/release/tvm/KEYS -m "Update KEYS"
svn cp --username $ASF_USERNAME --password "$ASF_PASSWORD" https://dist.apache.org/repos/dist/dev/tvm/KEYS https://dist.apache.org/repos/dist/release/tvm/ -m "Update KEYS"
```


## 创建候选版本分支

以 v0.6 版本为例：
* 需要在一个 Pull Request 中提交两个 commit：第一个 commit 将版本号从 0.6.dev0 更新为 0.6.0；第二个 commit 将版本号从 0.6.0 更新为 0.7.dev0；Pull Request 标题需注明：[Dont Squash]。
* 合并后，基于第一个版本号更新 commit 创建分支。分支名称应使用基础版本号（不含补丁号），例如为 `v0.6.0` 创建候选版本时，分支应命名为 `v0.6`，并在创建后立即推送 `v0.6.0.rc0` 标签到分支头部。

```plain
git clone https://github.com/apache/tvm.git
cd tvm/


# 更新第一个 commit 的版本号。
# ...
git add .
git commit -m "Bump version numbers to v0.6.0"


# 更新第二个 commit 的版本号。
# ...
git add .
git commit -m "Bump version numbers to v0.7.dev0"

# After pull request merged

# Pull Request 合并后  
# 基于第一个 commit 创建分支。  
git checkout <first-commit-id>


# 将 v0.6 替换为实际版本号。  
git branch v0.6
git push --set-upstream origin v0.6

git tag v0.6.0.rc0
git push origin refs/tags/v0.6.0.rc0
```


确保源代码中的版本号正确（示例：[PR #14300](https://github.com/apache/tvm/pull/14300)）。运行 `python3 version.py` 更新版本号。创建候选版本分支后应立即更新版本号。


在 GitHub 仓库的 "Releases" 标签页点击 "Draft a new release"：
* 通过检查版本号并确保 TVM 能构建和运行单元测试来验证发布。
* 提供格式为 `v1.0.0.rc0` 的发布标签（0 表示第一个候选版本），标签必须完全匹配 `v[0-9]+\.[0-9]+\.[0-9]+\.rc[0-9]` 模式。
* 通过点击 Target: branch > Recent commits > $commit_hash 选择提交。
* 将发布说明草案复制到描述框中。
* 勾选 "This is a pre-release"。
* 点击 "Publish release"。


注意：创建分支后仍可进行更改，但标签是固定的。如果发布需要任何修改，必须创建新标签。

删除之前的候选版本（如适用）：

```plain
git push --delete origin v0.6.0.rc1
```


创建源代码：

```plain

# 将 v0.6.0 替换为实际版本号。  
git clone git@github.com:apache/tvm.git apache-tvm-src-v0.6.0
cd apache-tvm-src-v0.6.0
git checkout v0.6
git submodule update --init --recursive
git checkout v0.6.0.rc0
rm -rf .DS_Store
find . -name ".git*" -print0 | xargs -0 rm -rf
cd ..
brew install gnu-tar
gtar -czvf apache-tvm-src-v0.6.0.rc0.tar.gz apache-tvm-src-v0.6.0
```


使用 GPG 密钥签名制品。首先确保 GPG 使用正确的私钥：

```plain
$ cat ~/.gnupg/gpg.conf
default-key F42xxxxxxxxxxxxxxx
```


创建 GPG 签名和文件哈希：

```plain
gpg --armor --output apache-tvm-src-v0.6.0.rc0.tar.gz.asc --detach-sig apache-tvm-src-v0.6.0.rc0.tar.gz
shasum -a 512 apache-tvm-src-v0.6.0.rc0.tar.gz > apache-tvm-src-v0.6.0.rc0.tar.gz.sha512
```

## 在 main 上更新 TVM 版本

创建候选版本后，确保更新主分支（`main`）的版本号。例如，如果发布 `v0.10.0`，需要将代码库中的版本号从 `v0.10.dev0` 更新为 `v0.11.dev0`。示例参考：[PR #12190](https://github.com/apache/tvm/pull/12190)。在发布分支包含的最后一个 commit 之后立即为主分支打上开发标签（如 `v0.11.dev0`），这是确保从主分支构建的 nightly 包版本号正确的必要步骤。

## 上传候选版本

编辑 GitHub 的发布页面，上传前几步创建的制品。

发布管理员还需要将制品上传到 ASF SVN：

```plain
# the --depth=files will avoid checkout existing folders
svn co --depth=files "https://dist.apache.org/repos/dist/dev/tvm" svn-tvm
cd svn-tvm
mkdir tvm-v0.6.0-rc0
# copy files into it
svn add tvm-0.6.0-rc0
svn ci --username $ASF_USERNAME --password "$ASF_PASSWORD" -m "Add RC"
```

## 拣选提交

在创建发布分支后但投票结束前，发布管理员可以从主分支（`main`）拣选 commit。由于 GitHub 上的发布分支受保护，要将修复合并到发布分支（如 `v0.11`），发布管理员必须针对发布分支提交包含拣选更改的 PR。PR 应大致匹配原始 PR，并额外说明拣选原因。社区随后对这些 PR 进行常规评审和合并流程。注意：针对发布分支的 PR 必须进行[签名](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)。

## 发起候选版本投票

第一轮投票在 Apache TVM 开发者邮件列表（[dev@tvm.apache.org](https://mailto:dev@tvm.apache.org/)）进行。为获得更多关注，可以创建以 "[VOTE]" 开头的 GitHub issue，它会自动镜像到 dev@。参考过往投票线程了解流程。邮件应包含以下内容：
* 发布说明草案链接。
* 候选版本制品链接。
* 确保邮件为纯文本格式且链接正确。


dev@ 投票需要至少 3 张有效赞成票且赞成票多于反对票。投票结束后应发送汇总邮件说明结果，主题类似 [VOTE][RESULT] ...


在 ASF，投票至少开放 72 小时（3 天）。如果在此时间内未获得足够有效票数，不能关闭投票，需要延长截止时间。


如果投票未通过，社区需要相应修改发布内容：创建新的候选版本并重新进行投票流程。

## 完成发布

投票通过后，将二进制文件从 dev 目录（投票所在目录）移动到 release 目录。这是向正式发布目录添加内容的唯一方式（注意：只有 PMC 成员可移动到 release 目录）。

```plain
export SVN_EDITOR=vim
svn mkdir https://dist.apache.org/repos/dist/release/tvm
svn mv https://dist.apache.org/repos/dist/dev/tvm/tvm-v0.6.0-rc2 https://dist.apache.org/repos/dist/release/tvm/tvm-v0.6.0


# 如果已将签名密钥添加到 KEYS 文件，同时更新 release 副本  
svn co --depth=files "https://dist.apache.org/repos/dist/release/tvm" svn-tvm
curl "https://dist.apache.org/repos/dist/dev/tvm/KEYS" > svn-tvm/KEYS
(cd svn-tvm && svn ci --username $ASF_USERNAME --password "$ASF_PASSWORD" -m"Update KEYS")
```


记得在 GitHub 上创建新的发布标签（本例中为 v0.6.0）并删除预发布候选标签：

```plain
git push --delete origin v0.6.0.rc2
```


## 更新 TVM 官网

官网仓库位于 [https://github.com/apache/tvm-site](https://github.com/apache/tvm-site)。修改下载页面以包含发布制品、GPG 签名和 SHA 哈希。由于 TVM 文档持续更新，需上传固定的发布文档版本。如果更新官网时 CI 已删除发布的文档，可以在 Jenkins 上重新启动发布分支的 CI 构建。参考以下示例代码：

```plain
git clone https://github.com/apache/tvm-site.git
pushd tvm-site
git checkout asf-site
pushd docs


# 创建发布文档目录。  
mkdir v0.9.0
pushd v0.9.0



# 从 CI 下载发布文档 。 
# 通过检查发布分支最近构建的 CI 日志找到此 URL  
curl -LO https://tvm-jenkins-artifacts-prod.s3.us-west-2.amazonaws.com/tvm/v0.9.0/1/docs/docs.tgz
tar xf docs.tgz
rm docs.tgz


# 添加文档并推送。
git add .
git commit -m "Add v0.9.0 docs"
git push
```


之后，修改[下载页面](https://tvm.apache.org/download)以支持最新发布。示例可参考[此处](https://github.com/apache/tvm-site/pull/38)。


## 发布公告

向 [announce@apache.org](https://mailto:announce@apache.org/) 和 [dev@tvm.apache.org](https://mailto:dev@tvm.apache.org/) 发送公告邮件。公告应包含发布说明和下载页面的链接。

## 补丁发布

补丁版本应仅用于关键错误修复。补丁版本必须经过与常规发布相同的流程，但发布管理员可酌情将候选版本投票窗口缩短至 24 小时以确保快速交付修复。每个补丁版本应更新发布基础分支（如 `v0.11`）的版本号，并为候选版本创建标签（如 `v0.11.1.rc0`）。

