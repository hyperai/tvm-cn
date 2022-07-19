---
title: Git Usage Tips
---

::: {.contents depth="2" local=""}
:::

以下是 Git 工作流程的一些技巧。

## 如何解决与 `main` 的冲突

-   首先 rebase 到最近的 main

    ``` bash
    # 前两步如果操作过可以跳过
    git remote add upstream [url to tvm repo]
    git fetch upstream
    git rebase upstream/main
    ```

-   Git 会显示一些无法 merge 的冲突，比如 `conflicted.py`。

    -   手动修改文件以解决冲突。

    -   解决冲突后，将其标记为已解决

        ``` bash
        git add conflicted.py
        ```

-   然后可以通过以下命令继续 rebase

    ``` bash
    git rebase --continue
    ```

-   最后强制 push 到你的分支

    ``` bash
    git push --force
    ```

## 如何将多个 commit merge 为一个

要将多个 commit（尤其是后面的 commit 只是对前面 commit 的修改时）组合为一个 PR 可以按照以下步骤操作。

-   如果之前没有配置过 Git 的默认编辑器，请先进行配置

    ``` bash
    git config core.editor the-editor-you-like
    ```

-   假设要 merge 最新 3 个 commit，请输入以下命令

    ``` bash
    git rebase -i HEAD~3
    ```

-   它将弹出一个文本编辑器。将第一个 commit 设置为 `pick`，并将稍后的 commit 更改为 `squash`。
-   保存文件后，它会弹出另一个文本编辑器，要求修改 merge 的 commit 消息。

-   将更改强制 push 到你的分支。

    ``` bash
    git push --force
    ```

## 重置到最近的主分支

可用 git reset 重置为最新的主版本。注意，**所有本地更改都将丢失**。因此，只有在没有本地更改或 pull request 刚刚 merge 时才可以这样做。

``` bash
git fetch origin main
git reset --hard FETCH_HEAD
```

## 重置后恢复先前的 commit

有时我们可能会将分支重置为错误的 commit。发生这种情况时，可以使用以下命令显示最近 commit 的列表

``` bash
git reflog
```

获得正确的 hashtag 后，可以再次使用 git reset 将 head 更改为正确的 commit。

## 仅将 k 个最新 commit 应用于 main

当有其他 m 个 commit 在 k 个最新 commit 之前已经 merge 时，只在 main 上应用这 k 个最新 commit 很有用。直接对 main 进行 rebase 可能会导致这 m 个 commit 的 merge 冲突（可以安全地丢弃）。

可以改用以下命令：

``` bash
# k 是实数
# 将 HEAD~2 作为最新一个提交
git rebase --onto upstream/main HEAD~k
```

然后可以强制 push 到 main。注意，上述命令将丢弃最后 k 个 commit 之前的所有 commit。

## 强制 push 的后果是什么

前两个技巧需要强制 push，这是因为我们改变了 commit 的路径。如果更改的 commit 仅属于你自己，就可以强制 push 到你自己的分支。
