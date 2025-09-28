# Apache TVM 中文文档

[中文文档](https://tvm.hyper.ai/) |
[参与社区共建](CONTRIBUTING.md) |
[了解更多](https://hyper.ai/) |
[版本注释](NEWS.md)

TVM 是一个开源的深度学习编译器，适用于 CPU、GPU、ARM 等多种硬件架构，旨在使机器学习工程师能够在任意硬件后端，高效地运行并优化计算。

鉴于 TVM 相关的中文学习资料比较零散，不利于开发者系统性学习，我们在 GitHub 上创建了 TVM 文档翻译项目。

目前中文文档的版本是基于 TVM v0.10.0 进行的本土化，随着 TVM 官方版本及文档的更新，中文文档也会不断调整，你可以：

* 学习 TVM 中文文档，为翻译不准确或有歧义的地方 [提交 issue](https://github.com/hyperai/tvm-cn/issues) 或 [PR](https://github.com/hyperai/tvm-cn/pulls)
* 参与开源共建、追踪文档更新，并认领文档翻译，成为 TVM 中文文档贡献者
* 加入 TVM 中文社区、结识同行并参与更多讨论及交流

真诚希望能借此项目，为 TVM 中文社区的发展带来一些微小的帮助。

## 参与贡献

本地开发服务器需先安装 Node.js 以及 [pnpm](https://pnpm.io/installation)。

```bash
pnpm install
pnpm start
```

本项目基于 Docusaurus 构建，具体的 Markdown 格式要求请参考 [Docusaurus 文档](https://docusaurus.io/docs/docs-introduction)。

迁移图片，将第三方的外部图片按其完整路径进行迁移，例如图片：

```md
![图片](https://raw.githubusercontent.com/tvmai/tvmai.github.io/main/images/relay/dataflow.png)
```

请将其保存在项目中的如下路径：

```
static/img/docs/tvmai/tvmai.github.io/main/images/relay/dataflow.png
```

然后在文档中替换为：

```md
![图片](/img/docs/tvmai/tvmai.github.io/main/images/relay/dataflow.png)
```

生成 HTML 文件 (Deprecated)

```bash
sphinx-build -b html docs build
```

## 创建新版本

如果当前版本为 `0.12.0`，想升到 `0.13.0`，那么你需要先保存当前版本

```bash
pnpm run docusaurus docs:version 0.12.0
```

然后编辑 `docusaurus.config.ts` 中 `versions.current.label` 为最新版本 `0.13.0`
