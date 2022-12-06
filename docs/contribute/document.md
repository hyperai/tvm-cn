---
title: 文档指南
sidebar_position: 5
---

TVM 文档大致遵循 [Divio 技术文档写作指南](https://documentation.divio.com)。之所以选择该指南，是因为它是一个“简单、全面且几乎普遍适用的方案，且已在广泛的领域和应用中证明”。

本文档描述了 TVM 文档的组织结构，以及如何撰写新文档。参阅 [docs/README.md](https://github.com/apache/tvm/tree/main/docs#build-locally) 来了解如何构建文档。

## 四种文档类型

### 入门教程

这是引导新用户加入项目的分步指南。入门教程只需要让用户学会使用软件即可，无需让他们知道底层原理。底层原理将在其他类型的文档中讲解。入门教程侧重于成功的初体验，它是将新手转变为新用户和开发人员的最重要文档。

完整的端到端教程——从安装 TVM 和支持 ML 软件，到创建和训练模型，再到编译到不同的架构——将让新用户以尽可能最有效的方式来使用 TVM。它让初学者知道那些必备知识，这与操作指南形成鲜明对比，操作指南旨在回答有一定经验的用户会提出的问题。

教程必须是可复现且可靠的，因为不成功的体验会让用户寻找其他解决方案。

### 操作指南

这是有关如何解决特定问题的分步指南。用户可以提出有意义的问题，然后文档给出答案。这类文档举例来说大概是这样的：“如何为 ARM 架构编译一个优化模型？”或“如何编译和优化 TensorFlow 模型？”这些文档应该足够开放，以便用户可以看到如何将其应用于新的用例。实用性比完整性更重要。标题应该能让用户看出这个操作指南解决的是什么问题。

教程与操作指南有何不同？教程面向刚接触 TVM 的开发者，重点是成功地将他们引导到软件和社区之中。而操作指南则侧重于，在对它有基本理解的前提下，完成特定任务。教程假定用户没有任何先备知识，帮助他快速上手。操作指南则假设用户具备最基本的知识，旨在指导他完成特定任务。

### 参考

参考文档描述了软件的配置和操作方式。 API、关键函数、命令和接口都是参考文档的候选对象。它是让用户构建自己的界面和程序的技术手册。它以信息为导向，注重列表和描述。可以假设读者已经掌握了软件的工作原理，并且正在寻找特定问题的特定答案。理想情况下，参考文档应该和代码库具有相同的结构，并且尽可能是自动生成的。

### 架构指南

架构指南是关于某个主题的解释和背景材料。这些文档帮助用户理解应用环境。为什么事情是这样的？设计决策是什么，考虑了哪些替代方案，描述现有系统的 RFC 是什么？它包括了学术论文和软件相关出版物的链接。在这些文档中，你可以探索到矛盾点在哪里，并帮助读者理解软件是如何、以及为什么按照现在的方式构建的。

它既不是操作指南，也不是如何完成任务的描述。相反，这些文档聚焦能够帮助用户理解项目的更高级的概念。通常，这些是由项目的架构师和开发人员编写的，有助于帮助用户和开发人员更深入地了解为什么软件是这样工作的，以及如何以与底层设计原则一致的方式，来对它做出贡献。

### TVM 的特殊考量

TVM 社区有一些特殊考量，偏离了 Divio 的简单文档风格的原则。第一点就是用户和开发者社区之间经常存在重叠。很多项目用不同的系统记录开发者和用户的体验，但 TVM 将这两者放在一起，只在合适的时候做一些区分。因此，教程和操作指南将分为关注用户体验的“用户手册”和关注开发者体验的“开发手册”。

下一个考量是 TVM 社区中存在一些特殊主题，值得投入较多的关注。这些主题包括但不限于 microTVM 和 VTA。可以创建特殊的“主题指南”来索引现有材料，并提供相关超链接。

为方便新手，将制作一个特殊的“入门”部分，其中包含安装说明、为什么使用 TVM 以及其他初体验文档。

## 技术细节

我们将 [Sphinx](http://sphinx-doc.org) 作为主要文档。 Sphinx 支持 reStructuredText 和 markdown。我们鼓励使用功能更丰富的 reStructuredText。注意，Python 文档字符串和教程允许嵌入 reStructuredText 语法。

构建文档的指导请参阅 [docs/README.md](https://github.com/apache/tvm/tree/main/docs#build-locally)。

### Python 参考文档

使用 [numpydoc](https://numpydoc.readthedocs.io/en/latest/) 格式来记录函数和类。以下代码段提供了一个文档字符串示例。记录所有公共函数，必要时提供支持功能的使用示例（如下所示）。

``` python
def myfunction(arg1, arg2, arg3=3):
    """简单描述我的函数。

    Parameters
    ----------
    arg1 : Type1
        arg1 的描述

    arg2 : Type2
        arg2 的描述

    arg3 : Type3, 可选
        arg3 的描述

    Returns
    -------
    rv1 : RType1
        返回类型 1 的描述

    Examples
    --------
    .. code:: python

        # myfunction 的使用示例
        x = myfunction(1, 2)
    """
    return rv1
```

注意在文档的各个部分之间留空行。在上例中，`Parameters`、`Returns` 和 `Examples` 前面必须有一个空行，以便正确构建文档。要往文档里添加新功能时，需要将 [sphinx.autodoc](http://www.sphinx-doc.org/en/master/ext/autodoc.html) 规则添加到 [docs/reference/api/python](https://github.com/apache/tvm/tree/main/docs/reference/api/python) 中）。可以参考该文件夹下的已有的文件来了解如何添加功能。

### C++ 参考文档

使用 Doxygen 格式来记录 C++ 函数。以下片段展示了一个 C++ 文档字符串的示例。

``` c++
/*!
 * \brief 我的函数的简单描述
 * \param arg1：arg1 的描述
 * \param arg2：arg2 的描述
 * \returns 描述返回值
 */
int myfunction(int arg1, int arg2) {
  // 必要的时候添加一些注释来阐明内部逻辑
}
```

除了记录函数的用法外，我们还强烈建议贡献者添加有关代码逻辑的注释以提高可读性。

### Sphinx Gallery 操作指南

我们用 [sphinx-gallery](https://sphinx-gallery.github.io/) 构建了很多 Python 操作指南。可以在 [gallery](https://github.com/apache/tvm/tree/main/gallery) 下找到源代码。注意：注释块是用 reStructuredText 而不是 markdown 编写的，请注意语法。

操作指南代码将在我们的构建服务器上运行以生成文档页面。所以我们可能会有一个限制，比如无法访问远程 Raspberry Pi，在这种情况下，向教程中添加一个标志变量（如 `use_rasp`），并允许用户通过更改一个标志轻松切换到真实设备。然后用已有的环境来演示使用。

如果为操作指南添加一个新的分类，则需要添加对 [conf.py](https://github.com/apache/tvm/tree/main/docs/conf.py) 和 [how-to index](https://github.com/apache/tvm/tree/main/docs/how-to/index.rst) 的引用。

### 引用文档中的另一个位置

请使用 sphinx 的 `:ref:` 标记来引用同一文档中的另一个位置。

``` rst
.. _document-my-section-tag

My Section
----------

可以使用 :ref:`document-my-section-tag` 来引用 My Section。
```

### 带有图像/图形的文档

reStructuredText 的 [figure](https://docutils.sourceforge.io/docs/ref/rst/directives.html#figure) 和 [image](https://docutils.sourceforge.io/docs/ref/rst/directives.html#image) 元素允许文档包含图像 URL。

为 TVM 文档创建的图像文件应该在 <https://github.com/tlc-pack/web-data> 仓库中，而使用这些图像的 *.rst* 文件应该在 TVM 仓库（<https://github.com/apache/tvm>）。

这需要两个 GitHub Pull Request，一个用于图像文件，另一个用于 *.rst* 文件。contributor 与 reviewer 之间可能需要讨论来协调 review 的过程。

重要提示：使用上述的两个 pull request 时，请先在 <https://github.com/tlc-pack/web-data> 中 merge pull request，然后再在 <https://github.com/apache/tvm> 中 merge pull request。这有助于确保 TVM 在线文档中的所有 URL 链接都是有效的。
