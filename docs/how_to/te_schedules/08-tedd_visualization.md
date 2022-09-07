---
title: 用 TEDD 进行可视化
---

# 用 TEDD 进行可视化

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_schedules/tedd.html#sphx-glr-download-how-to-work-with-schedules-tedd-py) 下载完整的示例代码
:::

**作者**：[Yongfeng Gu](https://github.com/yongfeng-nv)

本文介绍使用 TEDD（Tensor Expression Debug Display）对张量表达式进行可视化。

张量表达式使用原语进行调度，单个原语容易理解，但组合在一起时，就会变得复杂。在张量表达式中引入了调度原语的操作模型。

* 不同调度原语之间的交互，
* 调度原语对最终代码生成的影响。

操作模型基于数据流图、调度树和 IterVar 关系图。调度原语在这些计算图上进行操作。

TEDD 从给定的 schedule 中呈现这三个计算图，本教程演示了如何用 TEDD，以及如何解释渲染的计算图。

``` python
import tvm
from tvm import te
from tvm import topi
from tvm.contrib import tedd
```

## 使用 Bias 和 ReLU 定义和调度卷积

用 Bias 和 ReLU 为卷积构建一个张量表达式示例，首先连接 conv2d、add 和 relu TOPIs，然后创建一个 TOPI 通用 schedule。

``` python
batch = 1
in_channel = 256
in_size = 32
num_filter = 256
kernel = 3
stride = 1
padding = "SAME"
dilation = 1

A = te.placeholder((in_size, in_size, in_channel, batch), name="A")
W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W")
B = te.placeholder((1, num_filter, 1), name="bias")

with tvm.target.Target("llvm"):
    t_conv = topi.nn.conv2d_hwcn(A, W, stride, padding, dilation)
    t_bias = topi.add(t_conv, B)
    t_relu = topi.nn.relu(t_bias)
    s = topi.generic.schedule_conv2d_hwcn([t_relu])
```

## 使用 TEDD 渲染计算图

通过渲染计算图来查看计算及其调度方式。若在 Jupyter Notebook 中运行本教程，则可以用以下注释行来渲染 SVG 图形，让它直接在 Notebook 中显示。

``` python
tedd.viz_dataflow_graph(s, dot_file_path="/tmp/dfg.dot")
# tedd.viz_dataflow_graph(s, show_svg = True)
```

 ![图片](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/tedd_dfg.png)

第一个是数据流图。每个节点代表一个阶段，中间是名称和内存范围，两边是输入/输出信息。图中的边显示节点的依赖关系。

``` python
tedd.viz_schedule_tree(s, dot_file_path="/tmp/scheduletree.dot")
# tedd.viz_schedule_tree(s, show_svg = True)
```

上面渲染了调度树图。注意范围不可用的警告，它表明要调用 normalize() 来推断范围信息。跳过检查第一个调度树，推荐通过比较 normalize() 之前和之后的计算图来了解其影响。

``` python
s = s.normalize()
tedd.viz_schedule_tree(s, dot_file_path="/tmp/scheduletree2.dot")
# tedd.viz_schedule_tree(s, show_svg = True)
```

 ![图片](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/tedd_st.png)

仔细看第二个调度树，ROOT 下的每一个 block 代表一个阶段。阶段名称显示在顶行，计算显示在底行。中间行是 IterVars，外部越高，内部越低。

IterVar 行包含其索引、名称、类型和其他可选信息。以 W.shared 阶段为例，第一行是名称「W.shared」和内存范围「Shared」。它的计算是 `W(ax0, ax1, ax2, ax3)`。最外层循环 IterVar 是 ax0.ax1.fused.ax2.fused.ax3.fused.outer，以 kDataPar 的 0 为索引，绑定到 threadIdx.y，范围（min=0，ext=8）。

还可以用索引框的颜色来判断 IterVar 类型，如图所示。

如果一个阶段在任何其他阶段都没有计算，则它有直接到根节点的边；否则，它有一条边指向它所附加的 IterVar，例如 W.shared 在中间计算阶段附加到 rx.outer。

:::note
根据定义，IterVars 是内部节点，而计算是调度树中的叶节点。为提高可读性，省略了 IterVars 之间的边和一个阶段内的计算，使每个阶段都是一个块。
:::

``` python
tedd.viz_itervar_relationship_graph(s, dot_file_path="/tmp/itervar.dot")
# tedd.viz_itervar_relationship_graph(s, show_svg = True)
```

 ![图片](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/tedd_itervar_rel.png)

最后一个是 IterVar 关系图。每个子图代表一个阶段，包含 IterVar 节点和转换节点。

例如，W.shared 有三个拆分节点和三个融合节点。其余的是与调度树中的 IterVar 行格式相同的 IterVar 节点。 Root IterVars 是那些不受任何变换节点驱动的，例如 ax0；叶节点 IterVars 不驱动任何转换节点，并且具有非负索引，例如索引为 0 的 ax0.ax1.fused.ax2.fused.ax3.fused.outer。

## 总结

本教程演示 TEDD 的用法。用一个 TOPI 构建的示例来显示底层的 schedule，可在任何调度原语之前和之后用它来检查其效果。

[下载 Python 源代码：tedd.py](https://tvm.apache.org/docs/_downloads/c253040abc62eace272e406b7e1a4df5/tedd.py)

[下载 Jupyter Notebook：tedd.ipynb](https://tvm.apache.org/docs/_downloads/a7aff5918e1b86809a5bd1da8bef7229/tedd.ipynb)