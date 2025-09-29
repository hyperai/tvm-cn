---

title: 转换

---


:::note

本教程可通过 Google Colab 交互式运行！也可点击[此处](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html#sphx-glr-download-deep-dive-tensor-ir-tutorials-tir-transformation-py)在本地运行 Jupyter Notebook。

[在 Google Colab 中打开](https://colab.research.google.com/github/apache/tvm-site/blob/asf-site/docs/_downloads/e2a9c4bfdec7a9365ef67c5335b1aaa4/tir_transformation.ipynb)

:::



在本节中，我们将深入编译流程的核心内容 —— 原始张量函数的转换（Transformation）。


在[上一节](https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html#tir-learning)中，我们展示了如何使用 TensorIR 编写 `mm_relu`。在实际应用中，同一个功能可能有多种实现方式，而不同实现可能会带来不同的性能表现。


:::note

本教程主要展示如何应用 TensorIR 的转换功能，而不是深入优化技巧本身。

:::



我们先回顾一下上一节中 `mm_relu` 的实现：

```plain
import tvm
from tvm.script import ir as I
from tvm.script import tir as T


@I.ir_module
class MyModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        Y = T.alloc_buffer((128, 128))
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


在进行转换之前，先评估一下原始实现的性能。

```plain
import numpy as np

a_np = np.random.uniform(size=(128, 128)).astype("float32")
b_np = np.random.uniform(size=(128, 128)).astype("float32")
c_np = a_np @ b_np

a_nd = tvm.runtime.tensor(a_np)
b_nd = tvm.runtime.tensor(b_np)
c_nd = tvm.runtime.tensor(np.zeros((128, 128), dtype="float32"))

def evaluate(mod: tvm.IRModule):
    lib = tvm.tir.build(mod, target="llvm")
    # check correctness
    # 检查正确性
    lib(a_nd, b_nd, c_nd)
    np.testing.assert_allclose(c_nd.numpy(), c_np, rtol=1e-5)
    # evaluate performance
    # 评估性能
    f_timer = lib.time_evaluator("main", tvm.cpu())
    print(f_timer(a_nd, b_nd, c_nd))


 evaluate(MyModule)

```
输出:
```plain
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
   2.3069       2.3069       2.3069       2.3069       0.0000
```


## 初始化计划

我们通过创建一个 Schedule 辅助类，并将提供的 MyModule 作为输入，来启动代码转换的过程：

```plain
sch = tvm.tir.Schedule(MyModule)
```



## 循环分块（Loop Tiling）

随后，我们执行必要的操作，以获取对块 Y 及其相关循环的引用：

```plain
block_Y = sch.get_block("Y")
i, j, k = sch.get_loops(block_Y)
```


我们接下来执行转换操作。第一个转换操作是将循环 `j` 分成两个嵌套的循环，其中内层循环的长度为 4。必须注意的是，转换过程是分步进行的；因此，若无意中重复执行该代码块，将会因变量 `j` 不存在而触发错误提示。

```plain
j0, j1 = sch.split(j, factors=[None, 8])
```


你可以查看转换后的结果，它保存在 `sch.mod` 中：

```plain
sch.mod.show()
```
输出：
```plain
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        Y = T.alloc_buffer((128, 128))
        for i, j_0, j_1, k in T.grid(128, 16, 8, 128):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j_0 * 8 + j_1)
                vk = T.axis.reduce(128, k)
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(Y[vi, vj])
                with T.init():
                    Y[vi, vj] = T.float32(0.0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(Y[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0.0))
```



完成初步转换后，我们得到了两个新循环 `j_0` 和 `j_1`，它们的取值范围分别为 32 和 4。接下来的操作是对这两个循环进行重排序：

```plain
sch.reorder(j0, k, j1)
sch.mod.show()
evaluate(sch.mod)
```
输出：
```plain
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        Y = T.alloc_buffer((128, 128))
        for i, j_0, k, j_1 in T.grid(128, 16, 128, 8):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j_0 * 8 + j_1)
                vk = T.axis.reduce(128, k)
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(Y[vi, vj])
                with T.init():
                    Y[vi, vj] = T.float32(0.0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(Y[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0.0))

Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
   0.8776       0.8776       0.8776       0.8776       0.0000
```


## 利用局部性

接下来，我们通过两个新的转换步骤来生成另一个变体。

首先使用原语 **reverse_compute_at**，将块 **C** 移动到块 **Y** 的某个内循环中：

```plain
block_C = sch.get_block("C")
sch.reverse_compute_at(block_C, j0)
sch.mod.show()
```
输出：
```plain
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        Y = T.alloc_buffer((128, 128))
        for i, j_0 in T.grid(128, 16):
            for k, j_1 in T.grid(128, 8):
                with T.block("Y"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 8 + j_1)
                    vk = T.axis.reduce(128, k)
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(Y[vi, vj])
                    with T.init():
                        Y[vi, vj] = T.float32(0.0)
                    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
            for ax0 in range(8):
                with T.block("C"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 8 + ax0)
                    T.reads(Y[vi, vj])
                    T.writes(C[vi, vj])
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0.0))
```


## 重写归约操作

到目前为止，归约操作的初始化和更新步骤仍然在同一个块体中。这种形式有利于进行循环变换，因为初始化和更新通常需要保持外部循环（如 `i` 和 `j`）同步。


在完成前面的循环变换后，我们可以通过 **decompose_reduction** 原语将 Y 的初始化操作和归约更新操作拆分开来：

```plain
sch.decompose_reduction(block_Y, k)
sch.mod.show()
evaluate(sch.mod)
```
输出：
```plain
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        Y = T.alloc_buffer((128, 128))
        for i, j_0 in T.grid(128, 16):
            for j_1_init in range(8):
                with T.block("Y_init"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 8 + j_1_init)
                    T.reads()
                    T.writes(Y[vi, vj])
                    Y[vi, vj] = T.float32(0.0)
            for k, j_1 in T.grid(128, 8):
                with T.block("Y_update"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 8 + j_1)
                    vk = T.axis.reduce(128, k)
                    T.reads(Y[vi, vj], A[vi, vk], B[vk, vj])
                    T.writes(Y[vi, vj])
                    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
            for ax0 in range(8):
                with T.block("C"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 8 + ax0)
                    T.reads(Y[vi, vj])
                    T.writes(C[vi, vj])
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0.0))

Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
   0.3313       0.3313       0.3313       0.3313       0.0000
```



## 追踪转换

TensorIR 的调度是过程化**语言**，转换是按步骤逐步执行的。我们可以通过打印调度或其历史记录来追踪这些转换。


我们已经通过 `sch.mod` 打印了调度，也可以通过 `sch.trace` 打印调度历史：

```plain
sch.trace.show()
```
输出：
```plain
# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="Y", func_name="main")
  l1, l2, l3 = sch.get_loops(block=b0)
  l4, l5 = sch.split(loop=l2, factors=[None, 8], preserve_unit_iters=True, disable_predication=False)
  sch.reorder(l4, l3, l5)
  b6 = sch.get_block(name="C", func_name="main")
  sch.reverse_compute_at(block=b6, loop=l4, preserve_unit_loops=False, index=-1)
  b7 = sch.decompose_reduction(block=b0, loop=l3)
```


或者，可以同时输出 IRModule 与调度历史：

```plain
sch.show()
```
输出：
```plain
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        Y = T.alloc_buffer((128, 128))
        for i, j_0 in T.grid(128, 16):
            for j_1_init in range(8):
                with T.block("Y_init"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 8 + j_1_init)
                    T.reads()
                    T.writes(Y[vi, vj])
                    Y[vi, vj] = T.float32(0.0)
            for k, j_1 in T.grid(128, 8):
                with T.block("Y_update"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 8 + j_1)
                    vk = T.axis.reduce(128, k)
                    T.reads(Y[vi, vj], A[vi, vk], B[vk, vj])
                    T.writes(Y[vi, vj])
                    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
            for ax0 in range(8):
                with T.block("C"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 8 + ax0)
                    T.reads(Y[vi, vj])
                    T.writes(C[vi, vj])
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0.0))

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="Y", func_name="main")
  l1, l2, l3 = sch.get_loops(block=b0)
  l4, l5 = sch.split(loop=l2, factors=[None, 8], preserve_unit_iters=True, disable_predication=False)
  sch.reorder(l4, l3, l5)
  b6 = sch.get_block(name="C", func_name="main")
  sch.reverse_compute_at(block=b6, loop=l4, preserve_unit_loops=False, index=-1)
  b7 = sch.decompose_reduction(block=b0, loop=l3)
```



[下载 Jupyter Notebook: tir_transformation.ipynb](https://tvm.apache.org/docs/_downloads/e2a9c4bfdec7a9365ef67c5335b1aaa4/tir_transformation.ipynb) 

[下载 Python 源码: tir_transformation.py](https://tvm.apache.org/docs/_downloads/335201bfd37b29f9d9fd7765217e7ba9/tir_transformation.py) 

[下载压缩包: tir_transformation.zip](https://tvm.apache.org/docs/_downloads/18ba0d2ee8120824175aaef66bc9c9bf/tir_transformation.zip)



