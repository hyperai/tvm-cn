---

title: 理解 Relax 抽象层

---


Relax 是 Apache TVM Unity 策略中使用的一种图抽象方式，用于对机器学习模型进行端到端的优化。Relax 的主要目标是描述机器学习模型的结构与数据流，包括模型不同部分之间的依赖关系与连接方式，以及如何在硬件上执行该模型。


## 端到端模型执行

在本章节中，我们将以一个两层神经网络模型为例进行说明。该模型由两个线性操作和 ReLU 激活函数组成。


![图片](/img/docs/v21/03-deep-dive_03-relax_02-understand-relax-abstraction_1.png)


### 高层操作表示

我们先来看一个使用 Numpy 实现的高层模型代码：

```plain
def numpy_mlp(data, w0, b0, w1, b1):
    lv0 = data @ w0 + b0
    lv1 = np.maximum(lv0, 0)
    lv2 = lv1 @ w1 + b1
    return lv2
```


上述代码展示了如何使用数组操作进行端到端模型的执行。当然，我们也可以用 Relax 重写上述模型：

```plain
from tvm.script import relax as R

@R.function
def relax_mlp(
    data: R.Tensor(("n", 784), dtype="float32"),
    w0: R.Tensor((784, 128), dtype="float32"),
    b0: R.Tensor((128,), dtype="float32"),
    w1: R.Tensor((128, 10), dtype="float32"),
    b1: R.Tensor((10,), dtype="float32"),
) -> R.Tensor(("n", 10), dtype="float32"):
    with R.dataflow():
        lv0 = R.matmul(data, w0) + b0
        lv1 = R.nn.relu(lv0)
        lv2 = R.matmul(lv1, w1) + b1
        R.output(lv2)
    return lv2
```



### 底层集成

然而，从机器学习编译（MLC）的角度来看，我们还希望深入了解这些数组操作背后的底层细节。


为此，我们会使用更底层的 Numpy 实现方式进行说明：


我们将使用循环来代替数组函数，在必要时显式使用 `numpy.empty` 来分配数组，并进行传递。下面是该模型的底层 NumPy 实现示例：


```plain
def lnumpy_linear(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    n, m, K = X.shape[0], W.shape[1], X.shape[1]
    Y = np.empty((n, m), dtype="float32")
    for i in range(n):
        for j in range(m):
            for k in range(K):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[k, j]

    for i in range(n):
        for j in range(m):
            Z[i, j] = Y[i, j] + B[j]


def lnumpy_relu0(X: np.ndarray, Y: np.ndarray):
    n, m = X.shape
    for i in range(n):
        for j in range(m):
            Y[i, j] = np.maximum(X[i, j], 0)

def lnumpy_mlp(data, w0, b0, w1, b1):
    n = data.shape[0]
    lv0 = np.empty((n, 128), dtype="float32")
    lnumpy_matmul(data, w0, b0, lv0)

    lv1 = np.empty((n, 128), dtype="float32")
    lnumpy_relu(lv0, lv1)

    out = np.empty((n, 10), dtype="float32")
    lnumpy_matmul(lv1, w1, b1, out)
    return out
```


了解了这个底层的 NumPy 示例后，我们现在可以介绍对应的 Relax 抽象形式，用于端到端模型的执行。以下代码展示了使用 TVMScript 实现的同一模型：


```plain
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def linear(x: T.handle, w: T.handle, b: T.handle, z: T.handle):
        M, N, K = T.int64(), T.int64(), T.int64()
        X = T.match_buffer(x, (M, K), "float32")
        W = T.match_buffer(w, (K, N), "float32")
        B = T.match_buffer(b, (N,), "float32")
        Z = T.match_buffer(z, (M, N), "float32")
        Y = T.alloc_buffer((M, N), "float32")
        for i, j, k in T.grid(M, N, K):
            with T.block("Y"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[v_i, v_j] = T.float32(0.0)
                Y[v_i, v_j] = Y[v_i, v_j] + X[v_i, v_k] * W[v_k, v_j]
        for i, j in T.grid(M, N):
            with T.block("Z"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                Z[v_i, v_j] = Y[v_i, v_j] + B[v_j]

    @T.prim_func(private=True)
    def relu(x: T.handle, y: T.handle):
        M, N = T.int64(), T.int64()
        X = T.match_buffer(x, (M, N), "float32")
        Y = T.match_buffer(y, (M, N), "float32")
        for i, j in T.grid(M, N):
            with T.block("Y"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                Y[v_i, v_j] = T.max(X[v_i, v_j], T.float32(0.0))

    @R.function
    def main(
        x: R.Tensor(("n", 784), dtype="float32"),
        w0: R.Tensor((784, 256), dtype="float32"),
        b0: R.Tensor((256,), dtype="float32"),
        w1: R.Tensor((256, 10), dtype="float32"),
        b1: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor(("n", 10), dtype="float32"):
        cls = Module
        n = T.int64()
        with R.dataflow():
            lv = R.call_tir(cls.linear, (x, w0, b0), out_sinfo=R.Tensor((n, 256), dtype="float32"))
            lv1 = R.call_tir(cls.relu, (lv0,), out_sinfo=R.Tensor((n, 256), dtype="float32"))
            lv2 = R.call_tir(cls.linear, (lv1, w1, b1), out_sinfo=R.Tensor((b, 10), dtype="float32"))
            R.output(lv2)
        return lv2
```


上面的代码包括两类函数：原始张量函数（`T.prim_func`）和 Relax 函数（`R.function`）。Relax 函数是一种新型抽象，用于表示高层神经网络的执行过程。


请注意，上述 Relax 模块原生支持符号形状（symbolic shape），如 `main` 函数中张量形状里的`"n"`，**以及 `linear` 函数中的 `M`**、**`N`**、**`K`**。这是 Relax 抽象层的一项关键特性，使得编译器能够在张量操作符和函数调用之间，全局追踪动态形状关系。


再次并排查看 TVMScript 代码与底层 numpy 代码，并检查它们之间的对应关系是非常有帮助的，我们将逐步详细地分析它们。由于我们已经学习了基本的张量函数，这里我们将重点关注高层执行部分。



## Relax 的关键元素

本节将介绍 Relax 抽象的关键元素，以及它如何在机器学习编译器中实现优化。



### 结构信息

结构信息（Structure Info）是 Relax 中的一个新概念，用于表示 Relax 表达式的类型。它可以是 `TensorStructInfo`、`TupleStructInfo` 等等。在前面的例子中，我们使用 `TensorStructInfo`（在 TVMScript 中简写为 `R.Tensor`）来表示输入、输出和中间结果张量的形状和数据类型。


### R.call_tir


`R.call_tir` 是 Relax 中的新抽象，用于在同一个 IRModule 中调用底层的原始张量函数（primitive tensor functions）。这是 Relax 支持跨层抽象的一个关键特性，它使得我们可以从高层的神经网络层调用到低层的张量操作。以下是一个来自上文代码的示例：


```plain
lv = R.call_tir(cls.linear, (x, w0, b0), out_sinfo=R.Tensor((n, 256), dtype="float32"))
```


为了说明 `R.call_tir` 是如何工作的，我们来看一个等效的低层 numpy 实现：


```plain
lv0 = np.empty((n, 256), dtype="float32")
lnumpy_linear(x, w0, b0, lv0)
```


具体来说，`call_tir` 会先分配一个输出张量 res，然后将输入和输出一起传入 prim_func。执行 prim_func 后，结果会被写入 res，然后返回该结果。



**这种约定被称为destination passing（目标传递），** 其思想是输入与输出张量都在外部显式分配，然后传入低层函数。这种风格在底层库设计中非常常见，便于高层框架自行控制内存的分配。需要注意的是，并非所有张量操作都适用于这种风格（特别是当输出形状依赖于输入时）。尽管如此，在实际使用中，尽量采用这种风格编写低层函数通常是很有益的。


### 数据流块

在 Relax 函数中另一个重要元素是 R.dataflow() 的作用域注解。

```plain
with R.dataflow():
    lv = R.call_tir(cls.linear, (x, w0, b0), out_sinfo=R.Tensor((n, 256), dtype="float32"))
    lv1 = R.call_tir(cls.relu, (lv0,), out_sinfo=R.Tensor((n, 256), dtype="float32"))
    lv2 = R.call_tir(cls.linear, (lv1, w1, b1), out_sinfo=R.Tensor((b, 10), dtype="float32"))
    R.output(lv2)
```


在讨论 dataflow block 之前，我们先介绍 纯函数（pure） 和 有副作用（side-effect） 的概念。


如果一个函数满足以下条件，**就**可以被认为是纯函数或无副作用的：
* 它只读取输入，并通过输出返回结果； 
* 它不会更改程序的其他部分（例如增加全局计数器的值）。


例如，所有 `R.call_tir` 的函数都是纯函数，因为它们只从输入读取数据，并将结果写入新分配的输出张量中。然而，inplace 操作（就地修改） 则不属于纯函数，即具有副作用的函数，因为它们会修改已有的中间或输入张量。


数据流块（Dataflow block）是一种用于标记程序中计算图区域的方式。特别地，在数据流块中，所有操作都必须是无副作用的；而在数据流块之外的操作则可以具有副作用。


:::note

一个常见问题是：为什么我们需要手动标记数据流块，而不是自动推断？


主要有两个原因：
* 自动推断数据流块很困难，尤其是当涉及到对打包函数（如 cuBLAS 的集成调用）时容易出现不准确。手动标记数据流块可以帮助编译器更准确地理解程序的数据流结构并进行优化。 
*  许多优化只能在数据流块中进行。例如，算子融合（fusion）优化仅限于单个数据流块中的操作。如果编译器错误地推断了数据流边界，可能会错过关键的优化机会，从而影响程序性能。

:::


通过手动标记数据流块，我们可以确保编译器拥有最准确的信息，从而实现更有效的优化。


