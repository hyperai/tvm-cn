---

title: 理解 TensorIR 抽象

---



TensorIR 是 Apache TVM 中的张量程序抽象，它是标准的机器学习编译框架之一。张量程序抽象的主要目标是描述循环及其相关的硬件加速选项，包括线程、应用专用硬件指令和内存访问。


为了帮助我们解释，我们使用以下的张量计算序列作为激励示例。具体来说，对于两个 128×128 的矩阵 `A` 和 `B`，我们执行以下两个步骤的张量计算。

```plain
Y_{i, j} &= \sum_k A_{i, k} \times B_{k, j} \\
C_{i, j} &= \mathbb{relu}(Y_{i, j}) = \mathbb{max}(Y_{i, j}, 0)
```
​
上述计算类似于神经网络中常见的典型原始张量函数，即带有 ReLU 激活函数的线性层。我们使用 TensorIR 来表示上述计算，如下所示。


在我们调用 TensorIR 之前，让我们先用原生 Python 代码和 NumPy 来展示计算：


```plain
def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)
```


在理解了底层 NumPy 示例后，我们现在准备引入 TensorIR。下面的代码块展示了 `mm_relu` 的 TensorIR 实现。该代码使用一种叫做 TVMScript 的语言编写，这是一个嵌入在 Python AST 中的领域特定语言。


```plain
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
```


接下来，让我们分析一下上述 TensorIR 程序中的元素。


## 函数参数与缓冲区

**函数参数与 NumPy 函数中的参数相对应。**

```plain
# TensorIR
def mm_relu(A: T.Buffer((128, 128), "float32"),
            B: T.Buffer((128, 128), "float32"),
            C: T.Buffer((128, 128), "float32")):
    ...
# NumPy
def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    ...
```


在这里，`A`、`B` 和 `C` 使用了名为 `T.Buffer` 的类型，该类型的形状参数是 `(128, 128)`，数据类型是 `float32`。这些附加信息帮助可能的 MLC 进程生成专门化的代码，针对特定的形状和数据类型进行优化。


同样，TensorIR 还使用缓冲区类型来分配中间结果。

```plain
# TensorIR
Y = T.alloc_buffer((128, 128), dtype="float32")
# NumPy
Y = np.empty((128, 128), dtype="float32")
```


## 循环迭代

**循环迭代也有直接的对应关系。**

`T.grid` 是 TensorIR 中的一种语法，允许我们写出多个嵌套的迭代器。


```plain
# TensorIR with `T.grid`
for i, j, k in T.grid(128, 128, 128):
    ...
# TensorIR with `range`
for i in range(128):
    for j in range(128):
        for k in range(128):
            ...
# NumPy
for i in range(128):
    for j in range(128):
        for k in range(128):
            ...
```


## 计算块

一个重要的区别在于计算语句：TensorIR 引入了一个额外的构造，称为 `T.block`。

```plain
# TensorIR
with T.block("Y"):
    vi = T.axis.spatial(128, i)
    vj = T.axis.spatial(128, j)
    vk = T.axis.reduce(128, k)
    with T.init():
        Y[vi, vj] = T.float32(0)
    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
# NumPy
vi, vj, vk = i, j, k
if vk == 0:
    Y[vi, vj] = 0
Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
```


一个块代表 TensorIR 中的一个基本计算单元。重要的是，块包含的信息比标准 NumPy 代码要多。它包含了一组块轴 `vi, vj, vk)`，以及围绕这些轴的计算。


```plain
vi = T.axis.spatial(128, i)
vj = T.axis.spatial(128, j)
vk = T.axis.reduce(128, k)
```


以上三行声明了以下语法中关于块轴的关键属性。

```plain
[block_axis] = T.axis.[axis_type]([axis_range], [mapped_value])
```


这三行表达了以下内容：
*  它们指定了 `vi`、`vj`、`vk`（在这个例子中，绑定到 `i`、`j`、`k`）。 
*  它们声明了 `vi`、`vj`、`vk` 的原始范围（例如，`T.axis.spatial(128, i)` 中的 128）。 
*  它们宣布了迭代器的属性（空间轴、归约轴）。


## 块轴属性

让我们深入探讨块轴的属性。这些属性表示轴与当前计算的关系。块包含三个轴 `vi`、`vj` 和 `vk`，同时块读取缓冲区 `A[vi, vk]` 和 `B[vk, vj]`，并写入缓冲区 `Y[vi, vj]`。严格来说，块对 Y 进行了（归约）更新，我们暂时将其称为写入，因为我们不需要从另一个块获取 Y 的值。


重要的是，对于固定的 `vi` 和 `vj` 值，计算块会在 `Y` 的空间位置生成一个点值（`Y[vi, vj]`），该值独立于 `Y` 中其他位置（具有不同的 `vi` 和 `vj` 值）。我们可以将 `vi` 和 `vj` 称为空间轴，因为它们直接对应于块写入缓冲区空间区域的起始位置。参与归约的轴（`vk`）被指定为归约轴。


## 为什么在块中有额外的信息

一个重要的观察是，额外的信息（块轴的范围及其属性）使得块在执行迭代时是自包含的，独立于外部的 `i, j, k` 循环。


块轴信息还提供了额外的属性，帮助我们验证外部循环在执行计算时的正确性。例如，下面的代码块会因为循环期望一个大小为 128 的迭代器，但我们只将其绑定到一个大小为 127 的 for 循环，因此会导致错误。


```plain
# 错误的程序，由于循环和块迭代不匹配
for i in range(127):
    with T.block("C"):
        vi = T.axis.spatial(128, i)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        这里出现错误是因为迭代器大小不匹配
        ...
```

## 块轴绑定的语法

在每个块轴直接映射到外部循环迭代器的情况下，我们可以使用 `T.axis.remap` 在一行中声明块轴。

```plain
# SSR 表示每个轴的属性分别为 "spatial", "spatial", "reduce"
vi, vj, vk = T.axis.remap("SSR", [i, j, k])
```


这等价于：

```plain
vi = T.axis.spatial(range_of_i, i)
vj = T.axis.spatial(range_of_j, j)
vk = T.axis.reduce (range_of_k, k)
```


因此，我们还可以按以下方式编写程序：

```plain
@tvm.script.ir_module
class MyModuleWithAxisRemapSugar:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
```


