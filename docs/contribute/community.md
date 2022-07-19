---
title: TVM Community Guidelines
---

::: {.contents depth="2" local=""}
:::

TVM 采用 Apache 风格的模型，以优胜劣。在我们看来，创造一个包容性的社区（人人都可以使用，做贡献，也可以影响项目的走向）非常重要。有关当前贡献者列表，请参见 [CONTRIBUTORS.md](https://github.com/apache/tvm/blob/main/CONTRIBUTORS.md)。

## 一般开发流程

欢迎社区中的每个人发送补丁、文档以及为项目提出新方向。这里的关键指导方针是让社区中的每个人都能参与决策和发展。当提出重大更改时，应发送 RFC 以供社区讨论。我们鼓励利用诸如 issue，论坛和邮件的可归档渠道来公开讨论，以便社区中的每个人都可以参与并在以后回顾该过程。

代码 review 是保证代码质量的关键方法之一。高质量的代码 review 可以防止长期的技术问题累积，对项目的成功至关重要。pull request 在 review 完后才能 merge。具有相应领域专业知识的 committer 审查完 pull request 再 merge。相应的可以请求多个熟悉代码领域的 reviewer。我们鼓励贡献者自己请求代码 review 并帮助 review 彼此的代码——记住每个人都自愿为社区贡献自己的时间，高质量的代码 review 本身将花费与实际的代码贡献一样多的时间，如果这样做，你的代码也会快速得到 review。

社区应努力通过讨论就技术决策达成共识。我们希望 committer 和 PMC 以外交方式主持技术讨论，并在必要时提供具有明确技术推理的建议。

## Committers

committer 是被授予项目写入权限的人。committer 还需要监督他们负责的代码的 review 过程。贡献领域可以采取各种形式，包括代码贡献和代码 review、文档、宣传和对外合作。在一个高质量和健康的项目中，committer 的作用至关重要。社区积极从贡献者中寻找新的 committer。以下是帮助社区识别潜在 committer 的有用特征的列表：

- 对项目的持续贡献，包括对 RFC 的讨论、代码 review 和新特性的提出以及其他开发活动。熟悉并能够掌握项目的一个或多个领域。
- 贡献质量：高质量、可读性高的代码贡献表现为：无需大量代码 review，便可 merge pull request；有创建整洁、可维护的代码并包含良好的测试用例的历史；提供有用信息的代码 review，从而帮助其他贡献者更好地遵守标准。
- 社区参与：积极参与论坛交流，通过教程、讲座和对外合作来推广项目。我们鼓励 committer 广泛合作，例如与社区成员线上一起 review 代码和讨论项目设计。

[项目管理委员会（PMC）](https://projects.apache.org/committee.html?tvm) 由活跃的 committer 组成，负责主持讨论、管理项目发布并提名新的 committer/PMC 成员。候选人通常由 PMC 内部讨论提出，然后共识批准（即至少 3+1 票，并且没有否决权）。任何否决权都必须附有理由。PMC 应该通过维护社区实践和准则，把 TVM 打造成一个更好的社区。PMC 应尽可能提名自己组织之外的新候选人。

## Reviewers

reviewer 是积极为项目做出贡献并愿意参与新贡献的代码 review 的人，他们源自活跃的贡献者。committer 应明确征求 reviewer 的 review 意见。高质量的代码 review 可以防止长期的技术问题累积，这对项目的成功至关重要。项目的 pull request 必须由至少一名 reviewer review 才能 merge。