---

title: TensorIR 创建

---

:::note

本教程可通过 Google Colab 交互式运行！也可点击[此处](/docs/deep-dive/tensorir/tir_creation/#%E5%88%9B%E5%BB%BA%E5%8A%A8%E6%80%81%E5%BD%A2%E7%8A%B6%E5%87%BD%E6%95%B0)在本地运行 Jupyter Notebook。

[在 Google Colab 中打开](https://colab.research.google.com/github/apache/tvm-site/blob/asf-site/docs/_downloads/c43b2ae5210f95ce8dae6102e9b060fd/tir_creation.ipynb)

:::


本节将介绍在 Apache TVM Unity 中编写 TensorIR 函数的方法。此教程假设你已经了解 TensorIR 的基本概念。如果你不熟悉，请先阅读：[理解 TensorIR 抽象](/docs/deep-dive/tensorir/understand-tensorir-abstraction)。


:::note

本教程聚焦于构建**独立的** TensorIR 函数。这里介绍的技术并不是最终用户编译 Relax 模型所必需的。

:::


## 使用 TVMScript 创建 TensorIR

创建 TensorIR 函数最直接的方法是使用 TVMScript。TVMScript 是一种用于表示 TVM 中 TensorIR 的 Python 语言。


:::important

虽然 TVMScript 使用 Python 的语法和 AST（抽象语法树），并支持自动补全、代码检查等 Python 工具，但它并不是原生的 Python 语言，不能由 Python 解释器直接执行。

更准确地说，装饰器 **@tvm.script** 会提取被装饰函数的 Python AST，并将其解析为 TensorIR。

:::


### 标准格式

我们来看一个来自「[理解 TensorIR 抽象](/docs/deep-dive/tensorir/understand-tensorir-abstraction)」中的 `mm_relu` 示例。以下是完整的 `ir_module` 和 TVMScript 格式：

```plain
import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tir as T


@I.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    with T.block("Y"):
                        vi = T.axis.spatial(128, i)
                        vj = T.axis.spatial(128, j)
                        vk = T.axis.reduce(128, k)
                        T.reads(A[vi, vk], B[vk, vj])
                        T.writes(Y[vi, vj])
                        with T.init():
                            Y[vi, vj] = T.float32(0)
                        Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i in range(128):
            for j in range(128):
                with T.block("C"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j)
                    T.reads(Y[vi, vj])
                    T.writes(C[vi, vj])
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
```


### 使用语法糖简化

为了简化代码编写，我们可以使用以下语法：
*  使用 `T.grid` 来压缩嵌套循环
*  使用 `T.axis.remap` 来简化 block 迭代器注解
*  对于可以从 block 主体中推导出读写信息的 block，可以省略 `T.reads` 和 `T.writes`

```plain
@I.ir_module
class ConciseModule:
    @T.prim_func
    def mm_relu(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
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


我们可以通过以下代码验证两个模块是否等价：

```plain
print(tvm.ir.structural_equal(MyModule, ConciseModule))
```
输出：
```plain
True
```


### 与 Python 变量的交互

尽管 TVMScript 不能被 Python 解释器直接执行，但它可以与 Python 进行一定程度的交互。例如，我们可以使用 Python 变量来指定 TensorIR 的形状和数据类型。


```plain
# Python 变量
M = N = K = 128
dtype = "float32"


# 使用 TVMScript 定义的 IRModule
@I.ir_module
class ConciseModuleFromPython:
    @T.prim_func
    def mm_relu(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        Y = T.alloc_buffer((M, N), dtype)
        for i, j, k in T.grid(M, N, K):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.cast(T.float32(0), dtype)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(M, N):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.cast(T.float32(0), dtype))
```


检查等价性：

```plain
print(tvm.ir.structural_equal(ConciseModule, ConciseModuleFromPython))
```
输出：
```plain
True
```


### 使用动态形状的 TensorIR 函数

尽管 TVMScript 不能被 Python 解释器直接执行，但它可以与 Python 进行一定程度的交互。例如，我们可以使用 Python 变量来指定 TensorIR 的形状和数据类型。

```plain
@I.ir_module
class DynamicShapeModule:
    @T.prim_func
    def mm_relu(a: T.handle, b: T.handle, c: T.handle):
        # 动态形状定义
        M, N, K = T.int32(), T.int32(), T.int32()

        # 使用动态形状绑定输入缓冲区
        A = T.match_buffer(a, [M, K], dtype)
        B = T.match_buffer(b, [K, N], dtype)
        C = T.match_buffer(c, [M, N], dtype)
        Y = T.alloc_buffer((M, N), dtype)
        for i, j, k in T.grid(M, N, K):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.cast(T.float32(0), dtype)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(M, N):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.cast(T.float32(0), dtype))
```


接下来，我们来测试运行时的动态形状推理：

```plain
def evaluate_dynamic_shape(lib: tvm.runtime.Module, m: int, n: int, k: int):
    A = tvm.runtime.tensor(np.random.uniform(size=(m, k)).astype("float32"))
    B = tvm.runtime.tensor(np.random.uniform(size=(k, n)).astype("float32"))
    C = tvm.runtime.tensor(np.zeros((m, n), dtype="float32"))
    lib(A, B, C)
    return C.numpy()


# 只需编译一次
dyn_shape_lib = tvm.compile(DynamicShapeModule, target="llvm")
# 支持不同的输入形状
print(evaluate_dynamic_shape(dyn_shape_lib, m=4, n=4, k=4))
print(evaluate_dynamic_shape(dyn_shape_lib, m=64, n=64, k=128))
```
输出：
```plain
[[1.6744074  1.8393843  0.9076001  0.32640088]
 [1.3455076  1.5298209  0.75502187 0.32371795]
 [1.9979694  2.221868   1.0828729  0.43582058]
 [1.7054784  1.8512932  0.89285195 0.34154552]]
[[30.544813 29.938599 33.654526 ... 29.934391 30.73088  25.106636]
 [30.644558 31.062693 32.34803  ... 29.584583 32.756992 25.280499]
 [33.73643  33.23441  34.2736   ... 34.284283 35.100815 27.748833]
 ...
 [31.313179 30.462463 30.996958 ... 28.831778 32.279408 25.663143]
 [33.129818 31.630735 33.334507 ... 29.682335 32.925854 26.043703]
 [32.44726  30.645096 33.926357 ... 29.750242 32.810432 25.420698]]
```

## 使用 Tensor Expression 创建 TensorIR

通常情况下，我们不直接关注 TensorIR 的具体细节，而是更倾向于用一种更简洁的方式描述计算过程，这时 Tensor Expression（TE）就派上了用场。


Tensor Expression 是一种领域特定语言（DSL），它使用类似表达式的 API 来描述一系列计算过程。


:::note

Tensor 表达式（TE）包含了 TVM 堆栈中的两个组件：表达式和调度。表达式是领域特定语言，体现了计算模式，这正是我们在本节中讨论的内容。相反，TE 调度是传统的调度方法，已经被 TVM Unity 堆栈中的 TensorIR 调度所取代。

:::


### 创建静态形状函数

我们仍然使用上一小节中的 `mm_relu` 示例来演示如何使用 TE 创建函数。

```plain
from tvm import te

A = te.placeholder((128, 128), "float32", name="A")
B = te.placeholder((128, 128), "float32", name="B")
k = te.reduce_axis((0, 128), "k")
Y = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
C = te.compute((128, 128), lambda i, j: te.max(Y[i, j], 0), name="C")
```


在这里，`te.compute` 的函数签名是 `te.compute(output_shape, fcompute)`。其中 fcompute 函数用于描述每个索引位置上元素 `Y[i, j]` 的计算方式：

```plain
lambda i, j: te.sum(A[i, k] * B[k, j], axis=k)
```


上面 lambda 表达式定义了如下计算：$$Y_{i, j} = \sum_k A_{i, k} \times B_{k, j}$$。定义完计算后，我们就可以结合输入输出参数创建对应的 TensorIR 函数。这个例子中，我们希望构建一个具有两个输入参数 **A、B** 和一个输出参数 **C** 的函数。

```plain
te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})
TEModule = tvm.IRModule({"mm_relu": te_func})
TEModule.show()
```
输出：
```plain
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        Y = T.alloc_buffer((128, 128))
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(A[v_i, v_k], B[v_k, v_j])
                T.writes(Y[v_i, v_j])
                with T.init():
                    Y[v_i, v_j] = T.float32(0.0)
                Y[v_i, v_j] = Y[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(Y[v_i, v_j])
                T.writes(C[v_i, v_j])
                C[v_i, v_j] = T.max(Y[v_i, v_j], T.float32(0.0))

```


### 创建动态形状函数

我们也可以使用 Tensor Expression 创建动态形状的函数。唯一的区别是我们需要将输入张量的形状指定为符号变量。


```plain
# 定义符号变量
M, N, K = te.var("m"), te.var("n"), te.var("k")
A = te.placeholder((M, N), "float32", name="A")
B = te.placeholder((K, N), "float32", name="B")
k = te.reduce_axis((0, K), "k")
Y = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
C = te.compute((M, N), lambda i, j: te.max(Y[i, j], 0), name="C")

dyn_te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})
DynamicTEModule = tvm.IRModule({"mm_relu": dyn_te_func})
DynamicTEModule.show()
```
输出：
```plain
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def mm_relu(var_A: T.handle, var_B: T.handle, var_C: T.handle):
        T.func_attr({"tir.noalias": True})
        m, n = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, n))
        k = T.int32()
        B = T.match_buffer(var_B, (k, n))
        C = T.match_buffer(var_C, (m, n))
        # with T.block("root"):
        Y = T.alloc_buffer((m, n))
        for i, j, k_1 in T.grid(m, n, k):
            with T.block("Y"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k_1])
                T.reads(A[v_i, v_k], B[v_k, v_j])
                T.writes(Y[v_i, v_j])
                with T.init():
                    Y[v_i, v_j] = T.float32(0.0)
                Y[v_i, v_j] = Y[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
        for i, j in T.grid(m, n):
            with T.block("C"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(Y[v_i, v_j])
                T.writes(C[v_i, v_j])
                C[v_i, v_j] = T.max(Y[v_i, v_j], T.float32(0.0))
```

可右键另存为下载。

[下载 Jupyter Notebook：tir_creation.ipynb](https://tvm.apache.org/docs/_downloads/c43b2ae5210f95ce8dae6102e9b060fd/tir_creation.ipynb)

[下载 Python 源码：tir_creation.py](https://tvm.apache.org/docs/_downloads/69cc67b95e2c6b258f6dc9c7367fe71e/tir_creation.py)

[下载压缩包：tir_creation.zip](https://tvm.apache.org/docs/_downloads/be26483bb70b8468499a01c55e8e866c/tir_creation.zip)


