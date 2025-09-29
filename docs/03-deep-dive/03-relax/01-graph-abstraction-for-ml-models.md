---

title: 面向机器学习模型的图抽象

---



图抽象是机器学习（ML）编译器中用于表示和分析模型结构与数据流的关键技术。通过将模型抽象为图结构，编译器可以执行各种优化，以提升性能和效率。本教程将介绍图抽象的基础知识、Relax IR 的关键元素，以及它在机器学习编译器中如何启用优化能力。



## 什么是图抽象？

图抽象是将机器学习模型表示为有向图的过程，其中节点代表计算操作（例如矩阵乘法、卷积），边表示操作之间的数据流。这种抽象方式使编译器能够分析模型中不同部分之间的依赖关系与联系。

```plain
from tvm.script import relax as R

@R.function
def main(
    x: R.Tensor((1, 784), dtype="float32"),
    weight: R.Tensor((784, 256), dtype="float32"),
    bias: R.Tensor((256,), dtype="float32"),
) -> R.Tensor((1, 256), dtype="float32"):
    with R.dataflow():
        lv0 = R.matmul(x, weight)
        lv1 = R.add(lv0, bias)
        gv = R.nn.relu(lv1)
        R.output(gv)
    return gv
```



## Relax 的关键特性

Relax 是 Apache TVM Unity 策略中使用的图表示方法，它通过以下几个重要特性支持对机器学习模型的端到端优化：
* **头等符号形状（symbolic shape）：** Relax 使用符号形状来表示张量的维度，使得编译器能够在张量操作与函数调用之间全局追踪动态形状关系。 
*  **多层次抽象：** Relax 支持从高层的神经网络层到低层的张量操作的跨层次抽象，使得优化可以在模型的多个层级之间协同展开。 
*  **可组合的变换（Composable transformations）：** Relax 提供了可组合的变换框架，可以选择性地应用于模型的不同部分。这包括部分 lower（向低层转换）和部分特化（partial specialization）等能力，从而提供灵活的定制与优化手段。


这些特性共同赋予 Relax 在 Apache TVM 生态系统中强大而灵活的机器学习模型优化能力。


