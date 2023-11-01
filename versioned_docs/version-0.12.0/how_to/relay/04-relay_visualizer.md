---
title: 使用 Relay Visualizer 可视化 Relay
---

# 使用 Relay Visualizer 可视化 Relay

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_relay/using_relay_viz.html#sphx-glr-download-how-to-work-with-relay-using-relay-viz-py) 下载完整的示例代码
:::

**作者**：[Chi-Wei Wang](https://github.com/chiwwang)

Relay IR 模块可以包含很多操作。通常单个操作很容易理解，但放在一起可能会使计算图难以阅读。随着优化 pass 发挥作用，情况可能会变得更糟。

该实用程序定义了一组接口（包括解析器、绘图器/渲染器、计算图、节点和边）将 IR 模块可视化为节点和边，且提供了默认解析器。用户可以用自己的渲染器对计算图进行渲染。

这里用渲染器来渲染文本形式的计算图，它是一个轻量级、类似 AST 可视化工具（灵感来自 [clang ast-dump](https://clang.llvm.org/docs/IntroductionToTheClangAST.html)）。以下将介绍如何通过接口类来实现自定义的解析器和渲染器。

更多细节参考 `tvm.contrib.relay_viz`。

``` python
from typing import (
    Dict,
    Union,
    Tuple,
    List,
)
import tvm
from tvm import relay
from tvm.contrib import relay_viz
from tvm.contrib.relay_viz.interface import (
    VizEdge,
    VizNode,
    VizParser,
)
from tvm.contrib.relay_viz.terminal import (
    TermGraph,
    TermPlotter,
    TermVizParser,
)
```

## 定义具有多个 GlobalVar 的 Relay IR 模块

构建一个包含多个 `GlobalVar` 的 Relay IR 模块示例。定义一个 `add` 函数，并在 main 函数中调用。

``` python
data = relay.var("data")
bias = relay.var("bias")
add_op = relay.add(data, bias)
add_func = relay.Function([data, bias], add_op)
add_gvar = relay.GlobalVar("AddFunc")

input0 = relay.var("input0")
input1 = relay.var("input1")
input2 = relay.var("input2")
add_01 = relay.Call(add_gvar, [input0, input1])
add_012 = relay.Call(add_gvar, [input2, add_01])
main_func = relay.Function([input0, input1, input2], add_012)
main_gvar = relay.GlobalVar("main")

mod = tvm.IRModule({main_gvar: main_func, add_gvar: add_func})
```

## 在终端上使用 Relay Visualizer 渲染图形

终端可以用类似于 clang AST-dump 的文本显示 Relay IR 模块，可以看到 `AddFunc` 函数在 `main` 函数中调用了两次。

``` python
viz = relay_viz.RelayVisualizer(mod)
viz.render()
```

输出结果：

``` bash
@main([Var(input0), Var(input1), Var(input2)])
`--Call
   |--GlobalVar AddFunc
   |--Var(Input) name_hint: input2
   `--Call
      |--GlobalVar AddFunc
      |--Var(Input) name_hint: input0
      `--Var(Input) name_hint: input1
@AddFunc([Var(data), Var(bias)])
`--Call
   |--add
   |--Var(Input) name_hint: data
   `--Var(Input) name_hint: bias
```

## 为 Relay 类型自定义解析器

有时想要强调感兴趣的信息，或者针对特定用途对事物进行不同的解析，需要遵守接口来定制解析器，下面演示如何自定义 `relay.var` 的解析器，需要实现抽象接口 `tvm.contrib.relay_viz.interface.VizParser`。

``` python
class YourAwesomeParser(VizParser):
    def __init__(self):
        self._delegate = TermVizParser()

    def get_node_edges(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:

        if isinstance(node, relay.Var):
            node = VizNode(node_to_id[node], "AwesomeVar", f"name_hint {node.name_hint}")
            # 没有引入边缘。所以返回一个空列表
            return node, []

        # 将其他类型委托给其他解析器。
        return self._delegate.get_node_edges(node, relay_param, node_to_id)
```

将解析器和感兴趣的渲染器传递给 visualizer，这里只用终端渲染器。

``` python
viz = relay_viz.RelayVisualizer(mod, {}, TermPlotter(), YourAwesomeParser())
viz.render()
```

输出结果：

``` bash
@main([Var(input0), Var(input1), Var(input2)])
`--Call
   |--GlobalVar AddFunc
   |--AwesomeVar name_hint input2
   `--Call
      |--GlobalVar AddFunc
      |--AwesomeVar name_hint input0
      `--AwesomeVar name_hint input1
@AddFunc([Var(data), Var(bias)])
`--Call
   |--add
   |--AwesomeVar name_hint data
   `--AwesomeVar name_hint bias
```

## 围绕计算图和绘图器进行定制

除了解析器，还可以通过实现抽象类 `tvm.contrib.relay_viz.interface.VizGraph` 和  `tvm.contrib.relay_viz.interface.Plotter` 来自定义计算图和渲染器。下面重写了 `terminal.py` 中定义的 `TermGraph`。我们添加了一个 hook 在 `AwesomeVar` 上方复制，并让 `TermPlotter` 使用新类。

``` python
class AwesomeGraph(TermGraph):
    def node(self, viz_node):
        # 先添加节点
        super().node(viz_node)
        # 如果是 AwesomeVar，复制它。
        if viz_node.type_name == "AwesomeVar":
            duplicated_id = f"duplicated_{viz_node.identity}"
            duplicated_type = "double AwesomeVar"
            super().node(VizNode(duplicated_id, duplicated_type, ""))
            # 将复制的 var 连接到原始的
            super().edge(VizEdge(duplicated_id, viz_node.identity))

# 使用 `AwesomeGraph` 覆盖 TermPlotter 
class AwesomePlotter(TermPlotter):
    def create_graph(self, name):
        self._name_to_graph[name] = AwesomeGraph(name)
        return self._name_to_graph[name]

viz = relay_viz.RelayVisualizer(mod, {}, AwesomePlotter(), YourAwesomeParser())
viz.render()
```

输出结果：

``` bash
@main([Var(input0), Var(input1), Var(input2)])
`--Call
   |--GlobalVar AddFunc
   |--AwesomeVar name_hint input2
   |  `--double AwesomeVar
   `--Call
      |--GlobalVar AddFunc
      |--AwesomeVar name_hint input0
      |  `--double AwesomeVar
      `--AwesomeVar name_hint input1
         `--double AwesomeVar
@AddFunc([Var(data), Var(bias)])
`--Call
   |--add
   |--AwesomeVar name_hint data
   |  `--double AwesomeVar
   `--AwesomeVar name_hint bias
      `--double AwesomeVar
```

## 总结

本教程演示了 Relay Visualizer 的使用和自定义。 `tvm.contrib.relay_viz.RelayVisualizer` 类由 `interface.py` 中定义的接口组成。

目的是快速查看，然后修复迭代，构造函数参数尽可能简单，而自定义可以通过一组接口类进行。

[下载 Python 源代码：using_relay_viz.py](https://tvm.apache.org/docs/_downloads/cb089f2129f9829a01cc54eb81528811/using_relay_viz.py)

[下载 Jupyter Notebook：using_relay_viz.ipynb](https://tvm.apache.org/docs/_downloads/b954238c1884e83b45d2ae543d824f03/using_relay_viz.ipynb)