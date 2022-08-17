---
title: 外部张量函数
---

# 外部张量函数

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_schedules/extern_op.html#sphx-glr-download-how-to-work-with-schedules-extern-op-py) 下载完整的示例代码
:::

**作者**：[Tianqi Chen](https://tqchen.github.io/)

虽然 TVM 支持透明代码生成，但有时也需将手写的代码合并到流水线，例如对一些卷积核使用 cuDNN，并定义其余阶段。

原生 TVM 就支持黑盒函数调用。具体来说，TVM 支持所有与 DLPack 兼容的张量函数。这意味着可以使用 POD 类型（指针、整数、浮点数），或者将指向 DLTensor 的指针作为参数，调用任何函数。

``` python
from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np
from tvm.contrib import cblas
import tvm.testing

if not tvm.get_global_func("tvm.contrib.cblas.matmul", allow_missing=True):
    raise Exception("Not compiled with cblas support; can't build this tutorial")
```

## 使用外部张量函数

以下示例用 `te.extern` 来添加一个外部数组函数调用。外部调用声明了输出张量的 shape，第二个参数给出了输入列表。

用户需要提供一个描述如何对结果进行计算的函数。计算函数获取输入和输出的符号占位符列表，并返回执行语句。

这种情况只需调用一个注册的 TVM 函数，它会调用 CBLAS。TVM 不控制外部数组函数的内部，将其视为黑盒。可以进一步混合可调度的 TVM 函数，为结果添加偏差项。

``` python
n = 1024
l = 128
m = 235
bias = te.var("bias", dtype="float32")
A = te.placeholder((n, l), name="A")
B = te.placeholder((l, m), name="B")
C = te.extern(
    (n, m),
    [A, B],
    lambda ins, outs: tvm.tir.call_packed(
        "tvm.contrib.cblas.matmul", ins[0], ins[1], outs[0], False, False
    ),
    name="C",
)
D = te.compute(C.shape, lambda i, j: C[i, j] + bias, name="D")
s = te.create_schedule(D.op)
```

## 验证结果

验证结果是否符合预期。

``` python
dev = tvm.cpu(0)
f = tvm.build(s, [A, B, D, bias], "llvm")
a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), dev)
d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), dev)
bb = 10.0
f(a, b, d, bb)
tvm.testing.assert_allclose(d.numpy(), np.dot(a.numpy(), b.numpy()) + 10, rtol=1e-5)
```

## 外部 Contrib Wrappers

TVM 为外部调用提供了外部contrib Wrappers，以下代码与前面的示例等效。

``` python
from tvm.contrib import cblas

C = cblas.matmul(A, B)
D = te.compute(C.shape, lambda i, j: C[i, j] + bias, name="D")
s = te.create_schedule(D.op)
```

## 将 Python 函数 Hook 为 Extern

由于可以调用 TVM 中的任何 PackedFunc，所以可以用外部函数回调到 Python 中。

以下示例将一个 Python 函数注册到 TVM runtime 系统，并用它来完成一个阶段的计算，这使得 TVM 更加灵活。例如，可通过插入前端回调来检查中间结果，或将自定义代码与 TVM 混合。

``` python
@tvm.register_func("tvm.contrib.my_tvm_addone")
def my_tvm_addone(x, y):
    print("my_tvm_addone signatures: %s, %s" % (type(x), type(y)))
    tvm.nd.array(x.numpy() + 1).copyto(y)

A = te.placeholder((n,), name="A")
B = te.extern(
    A.shape,
    [A],
    lambda ins, outs: tvm.tir.call_packed("tvm.contrib.my_tvm_addone", ins[0], outs[0]),
    name="C",
)
s = te.create_schedule(B.op)
f = tvm.build(s, [A, B], "llvm")
a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), dev)
f(a, b)
tvm.testing.assert_allclose(b.numpy(), a.numpy() + 1, rtol=1e-5)
```

输出结果：

``` bash
my_tvm_addone signatures: <class 'tvm.runtime.ndarray.NDArray'>, <class 'tvm.runtime.ndarray.NDArray'>
```

## 总结

* TVM 通过 `te.extern` 调用外部张量函数。
* 对外部张量调用使用 contrib wrappers。
* 将前端函数 hook 为外部张量的回调。

[下载 Python 源代码：extern_op.py](https://tvm.apache.org/docs/_downloads/286e7f77f494a25312ac88e3f234822e/extern_op.py)

[下载 Jupyter Notebook：extern_op.ipynb](https://tvm.apache.org/docs/_downloads/8472bea81cf679760d7e4e77e895726f/extern_op.ipynb)