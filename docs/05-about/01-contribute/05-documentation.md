---

title: 文档

---

* [四种文档类型](https://tvm.apache.org/docs/contribute/document.html#the-four-document-types)
   * [入门教程](https://tvm.apache.org/docs/contribute/document.html#introductory-tutorials)
   * [操作指南](https://tvm.apache.org/docs/contribute/document.html#how-to-guides)
   * [参考手册](https://tvm.apache.org/docs/contribute/document.html#reference)
   * [架构指南](https://tvm.apache.org/docs/contribute/document.html#architecture-guides)
   * [TVM 的特殊注意事项](https://tvm.apache.org/docs/contribute/document.html#special-considerations-for-tvm)
* [技术细节](https://tvm.apache.org/docs/contribute/document.html#technical-details)
   * [Python 参考文档](https://tvm.apache.org/docs/contribute/document.html#python-reference-documentation)
   * [C++ 参考文档](https://tvm.apache.org/docs/contribute/document.html#c-reference-documentation)
   * [Sphinx Gallery 操作指南](https://tvm.apache.org/docs/contribute/document.html#sphinx-gallery-how-tos)
   * [文档内部引用](https://tvm.apache.org/docs/contribute/document.html#refer-to-another-location-in-the-document)
   * [含图像/图表的文档](https://tvm.apache.org/docs/contribute/document.html#documents-with-images-figures)


TVM 文档大致遵循 [Divio 提出的正式文档风格](https://documentation.divio.com/)。选择这一体系是因为它「简单、全面且几乎普适。经过广泛领域和实践验证，具有极高的实用性。」


本文档描述了 TVM 文档的组织方式及如何编写新文档。关于文档构建的说明，请参阅 [docs/README.md](https://github.com/apache/tvm/tree/main/docs#build-locally)。



## 四种文档类型

### 入门教程


这是逐步引导新用户熟悉项目的指南。入门教程旨在让用户快速上手软件，而不必解释软件为何如此设计（这些内容可留待其他文档类型说明）。其核心是确保用户首次体验成功，是将新人转化为用户和开发者的最重要文档。一个完整的端到端教程——从安装 TVM 和支持性 ML 软件，到创建和训练模型，再到编译至不同架构——能让新用户以最高效的方式使用 TVM。教程教授初学者必备知识，这与操作指南（回答有经验用户的具体问题）形成对比。


教程必须可复现且可靠，因为失败会导致用户寻求其他解决方案。


### 操作指南

这是解决特定问题的分步指南。用户可提出明确问题，文档则提供答案。例如：「如何为 ARM 架构编译优化模型？」或「如何编译和优化 TensorFlow 模型？」这类文档应具有开放性，让用户能将其迁移至新场景。实用性比完整性更重要，标题需清晰说明解决的问题。


教程与操作指南有何不同？教程面向新开发者，聚焦于引导其成功接触软件和社区；而操作指南则基于用户已有基础认知，指导完成具体任务。教程助力入门，假设零基础；操作指南假设最低知识储备，旨在实现特定目标。


### 参考手册

参考文档描述软件的配置与操作方式。API、核心函数、命令和接口均属于此类。这些技术手册使用户能构建自己的程序和接口，以信息为导向，侧重列表和描述。可假设读者已理解软件原理，仅需特定问题的答案。理想情况下，参考文档应与代码结构一致，并尽可能自动生成。


### 架构指南

架构指南是对主题的背景解析材料，帮助理解应用环境的设计逻辑：为何如此设计？决策依据是什么？有哪些备选方案？相关 RFC 有哪些？包括学术论文和关联出版物链接。这类文档可探讨矛盾观点，帮助读者理解软件的构建逻辑。它不涉及具体操作，而是聚焦于帮助理解项目的高层概念。通常由项目架构师和开发者撰写，但也能帮助用户和开发者深入理解软件设计哲学，从而以符合底层原则的方式贡献代码。


### TVM 的特殊注意事项

TVM 社区存在需偏离 Divio 简单文档风格的特殊情况。首先，用户与开发者社区常有重叠。许多项目会分开记录两种体验，但本体系将统一考虑，仅在适当时区分。因此教程和操作指南将分为聚焦用户体验的「用户指南」和聚焦开发体验的「开发者指南」。


其次，TVM 社区存在需要额外关注的专题。为此可创建「专题指南」，索引现有材料并提供高效查阅的上下文。


为方便新人，将设立「入门」专区，包含安装说明、TVM 优势概述等首次体验文档。


## 技术细节

我们使用 [Sphinx](http://sphinx-doc.org/) 构建主文档。Sphinx 支持 reStructuredText 和 markdown，我们鼓励优先使用功能更丰富的 reStructuredText。注意 Python 文档字符串和教程允许嵌入 reStructuredText 语法。

文档构建说明见 [docs/README.md](https://github.com/apache/tvm/tree/main/docs#build-locally)。

### Python 参考文档

我们采用 [numpydoc](https://numpydoc.readthedocs.io/en/latest/) 格式记录函数和类。以下片段展示示例文档字符串。我们会完整记录所有公开函数，并在必要时提供使用示例（如下所示）：

```plain
def myfunction(arg1, arg2, arg3=3):
    """简要描述函数功能。
    Parameters
    ----------
    arg1 : Type1
        Description of arg1

    arg2 : Type2
        Description of arg2

    arg3 : Type3, optional
        Description of arg3

    Returns
    -------
    rv1 : RType1
        Description of return type one

    Examples
    --------
    .. code:: python

        
        # 函数使用示例。
        x = myfunction(1, 2)
    """
    return rv1
```


注意各章节间需留空行。上述示例中，`参数`、`返回值` 和 `示例` 前必须有空行以确保正确构建。要将新函数加入文档，需在 [docs/reference/api/python](https://github.com/apache/tvm/tree/main/docs/reference/api/python) 中添加 [sphinx.autodoc](http://www.sphinx-doc.org/en/master/ext/autodoc.html) 规则，可参考该目录下现有文件。


### C++ 参考文档

我们使用 doxygen 格式记录 C++ 函数。以下片段展示 C++ 文档字符串示例：

```plain
/*!
 * \brief Description of my function
 * \param arg1 Description of arg1
 * \param arg2 Description of arg2
 * \returns describe return value
 */
 /*!  
 * \brief 函数描述  
 * \param arg1 参数1说明  
 * \param arg2 参数2说明  
 * \returns 返回值描述  
 */  
int myfunction(int arg1, int arg2) {
  // When necessary, also add comment to clarify internal logics
  // 必要时添加注释阐明内部逻辑  
}
```


除函数用法外，强烈建议贡献者添加代码逻辑注释以提高可读性。

### 

### Sphinx Gallery 操作指南

我们使用 [sphinx-gallery](https://sphinx-gallery.github.io/) 构建多数 Python 操作指南，源码位于 [gallery](https://github.com/apache/tvm/tree/main/gallery)。需注意注释块需用 reStructuredText 而非 markdown 编写。


操作指南代码将在构建服务器上运行以生成文档页。若存在限制（如无法远程访问树莓派），可添加标志变量（如 `use_rasp`），允许用户通过修改标志切换至真实设备，同时用现有环境演示用法。

新增分类时，需在 [conf.py](https://github.com/apache/tvm/tree/main/docs/conf.py) 和 [操作指南索引](https://github.com/apache/tvm/tree/main/docs/how-to/index.rst) 中添加引用。

### 文档内部引用

请使用 sphinx 的 `:ref:` 标记引用同一文档的其他位置：

```plain
.. _document-my-section-tag


章节标题。
----------
可通过 :ref:`document-my-section-tag` 引用本章节。  

```


### 含图片/图表的文档

reStructuredText 的 [figure](https://docutils.sourceforge.io/docs/ref/rst/directives.html#figure) 和 [image](https://docutils.sourceforge.io/docs/ref/rst/directives.html#image) 元素支持插入图片 URL。


TVM 文档的图片文件应存放于 [https://github.com/tlc-pack/web-data](https://github.com/tlc-pack/web-data) 仓库，而引用这些图片的 .rst 文件应位于主仓库 [https://github.com/apache/tvm](https://github.com/apache/tvm)。


这需要两个 GitHub Pull Request：一个用于图片文件，另一个用于 .rst 文件。贡献者与审阅者可能需要协调合并顺序。


重要提示：按上述方式提交两个 PR 时，请先合并 [https://github.com/tlc-pack/web-data](https://github.com/tlc-pack/web-data) 的 PR，再合并 [TVM](https://github.com/apache/tvm) 的 PR。这能确保在线文档的所有链接有效。


