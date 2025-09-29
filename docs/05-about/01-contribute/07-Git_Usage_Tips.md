---

title: Git 使用技巧

---

* [如何解决与](https://tvm.apache.org/docs/contribute/git_howto.html#how-to-resolve-a-conflict-with-main) `main`[ 分支的冲突](https://tvm.apache.org/docs/contribute/git_howto.html#how-to-resolve-a-conflict-with-main)
* [如何合并多个提交](https://tvm.apache.org/docs/contribute/git_howto.html#how-to-combine-multiple-commits-into-one)
* [重置到最新的 main 分支](https://tvm.apache.org/docs/contribute/git_howto.html#reset-to-the-most-recent-main-branch)
* [恢复重置前的提交](https://tvm.apache.org/docs/contribute/git_howto.html#recover-a-previous-commit-after-reset)
* [仅将最新的 k 个提交到 main 分支](https://tvm.apache.org/docs/contribute/git_howto.html#apply-only-k-latest-commits-on-to-the-main)
* [强制推送的后果](https://tvm.apache.org/docs/contribute/git_howto.html#what-is-the-consequence-of-force-push)



以下是一些 Git 工作流的使用技巧。


## 如何解决与 `main` 分支的冲突

首先变基到最新的 main 分支：

```plain

# 前两步只需执行一次。
git remote add upstream [url to tvm repo]
git fetch upstream
git rebase upstream/main
```
* Git 可能会显示无法自动合并的冲突文件，例如 `conflicted.py`
   * 手动修改文件解决冲突。
   * 解决冲突后标记为已解决。

```plain
git add conflicted.py
```


继续变基操作。

```plain
git rebase --continue
```
最后推送到你的 fork 仓库（可能需要强制推送）。

```plain
git push --force
```

## 如何合并多个提交

有时我们需要将多个提交合并为一个（特别是后续提交只是对前一个提交的修正），以创建具有明确意义的 PR。操作步骤如下：


首先配置 Git 的默认编辑器（如果尚未配置）：

```plain
git config core.editor the-editor-you-like
```


假设要合并最后 3 个提交，输入下面命令。

```plain
git rebase -i HEAD~3
```
* 弹出的文本编辑器中，将第一个提交设为 `pick`，后续改为 `squash。`
* 保存文件后，会弹出另一个编辑器用于修改合并后的提交信息。
* 强制推送到你的 fork 仓库。

```plain
git push --force
```

## 重置到最新的 main 分支

你可以随时使用 git reset 将版本重置到最新的 main 分支。注意：**所有本地修改将会丢失**。因此仅在没有本地修改或 PR 刚被合并时使用。

```plain
git fetch origin main
git reset --hard FETCH_HEAD
```

## 恢复重置前的提交

有时我们可能错误地重置到错误的提交。此时可以使用以下命令查看最近提交列表：

```plain
git reflog
```


找到正确的哈希值后，再次使用 git reset 将 HEAD 指向正确的提交。

## 仅将最新的 k 个提交到 main 分支

有时只需要将最新的 k 个变更应用到 main 分支很有用。这种情况通常发生在你有其他 m 个提交已在这些 k 个提交之前被合并。直接基于 main 变基可能会导致前 m 个提交的冲突（这些可以安全丢弃）。


可以使用以下命令：


```plain
# k 为具体数字
# 使用 HEAD~2 表示最后 1 个提交
git rebase --onto upstream/main HEAD~k
```


然后可以强制推送到 main 分支。注意：上述命令将丢弃最后 k 个提交之前的所有提交。

## 强制推送的后果

前两个技巧需要强制推送，这是因为我们改变了提交路径。只要更改的提交只是你自己的，强制推送到你自己的 fork 仓库是没问题的。

