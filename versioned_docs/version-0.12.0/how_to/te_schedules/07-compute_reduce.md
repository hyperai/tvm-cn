---
title: 使用元组输入（Tuple Inputs）进行计算和归约
---

# 使用元组输入（Tuple Inputs）进行计算和归约

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_schedules/tuple_inputs.html#sphx-glr-download-how-to-work-with-schedules-tuple-inputs-py) 下载完整的示例代码
:::

**作者**：[Ziheng Jiang](https://github.com/ZihengJiang)

若要在单个循环中计算具有相同 shape 的多个输出，或执行多个值的归约，例如 `argmax`。这些问题可以通过元组输入来解决。

本教程介绍了 TVM 中元组输入的用法。

```plain
from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np
```

## 描述批量计算

对于 shape 相同的算子，若要在下一个调度过程中一起调度，可以将它们放在一起作为 `te.compute` 的输入。

``` python
n = te.var("n")
m = te.var("m")
A0 = te.placeholder((m, n), name="A0")
A1 = te.placeholder((m, n), name="A1")
B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] + 2, A1[i, j] * 3), name="B")

# 生成的 IR 代码：
s = te.create_schedule(B0.op)
print(tvm.lower(s, [A0, A1, B0, B1], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A0_1: handle, A1_1: handle, B_2: handle, B_3: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A0: Buffer(A0_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             A1: Buffer(A1_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto"),
             B: Buffer(B_4: Pointer(float32), float32, [(stride_2: int32*m)], [], type="auto"),
             B_1: Buffer(B_5: Pointer(float32), float32, [(stride_3: int32*m)], [], type="auto")}
  buffer_map = {A0_1: A0, A1_1: A1, B_2: B, B_3: B_1}
  preflattened_buffer_map = {A0_1: A0_3: Buffer(A0_2, float32, [m, n: int32], [stride, stride_4: int32], type="auto"), A1_1: A1_3: Buffer(A1_2, float32, [m, n], [stride_1, stride_5: int32], type="auto"), B_2: B_6: Buffer(B_4, float32, [m, n], [stride_2, stride_6: int32], type="auto"), B_3: B_7: Buffer(B_5, float32, [m, n], [stride_3, stride_7: int32], type="auto")} {
  for (i: int32, 0, m) {
    for (j: int32, 0, n) {
      B[((i*stride_2) + (j*stride_6))] = (A0[((i*stride) + (j*stride_4))] + 2f32)
      B_1[((i*stride_3) + (j*stride_7))] = (A1[((i*stride_1) + (j*stride_5))]*3f32)
    }
  }
}
```

## 使用协同输入（Collaborative Inputs）描述归约

有时需要多个输入来表达归约算子，并且输入会协同工作，例如 `argmax`。在归约过程中，`argmax` 要比较操作数的值，还需要保留操作数的索引，可用 `te.comm_reducer()` 表示：

``` python
# x 和 y 是归约的操作数，它们都是元组的索引和值。
def fcombine(x, y):
    lhs = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
    rhs = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
    return lhs, rhs

# 身份元素也要是一个元组，所以 `fidentity` 接收两种类型作为输入。
def fidentity(t0, t1):
    return tvm.tir.const(-1, t0), tvm.te.min_value(t1)

argmax = te.comm_reducer(fcombine, fidentity, name="argmax")

# 描述归约计算
m = te.var("m")
n = te.var("n")
idx = te.placeholder((m, n), name="idx", dtype="int32")
val = te.placeholder((m, n), name="val", dtype="int32")
k = te.reduce_axis((0, n), "k")
T0, T1 = te.compute((m,), lambda i: argmax((idx[i, k], val[i, k]), axis=k), name="T")

# 生成的 IR 代码：
s = te.create_schedule(T0.op)
print(tvm.lower(s, [idx, val, T0, T1], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(idx_1: handle, val_1: handle, T_2: handle, T_3: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {idx: Buffer(idx_2: Pointer(int32), int32, [(stride: int32*m: int32)], [], type="auto"),
             val: Buffer(val_2: Pointer(int32), int32, [(stride_1: int32*m)], [], type="auto"),
             T: Buffer(T_4: Pointer(int32), int32, [(stride_2: int32*m)], [], type="auto"),
             T_1: Buffer(T_5: Pointer(int32), int32, [(stride_3: int32*m)], [], type="auto")}
  buffer_map = {idx_1: idx, val_1: val, T_2: T, T_3: T_1}
  preflattened_buffer_map = {idx_1: idx_3: Buffer(idx_2, int32, [m, n: int32], [stride, stride_4: int32], type="auto"), val_1: val_3: Buffer(val_2, int32, [m, n], [stride_1, stride_5: int32], type="auto"), T_2: T_6: Buffer(T_4, int32, [m], [stride_2], type="auto"), T_3: T_7: Buffer(T_5, int32, [m], [stride_3], type="auto")} {
  for (i: int32, 0, m) {
    T[(i*stride_2)] = -1
    T_1[(i*stride_3)] = -2147483648
    for (k: int32, 0, n) {
      T[(i*stride_2)] = @tir.if_then_else((val[((i*stride_1) + (k*stride_5))] <= T_1[(i*stride_3)]), T[(i*stride_2)], idx[((i*stride) + (k*stride_4))], dtype=int32)
      T_1[(i*stride_3)] = @tir.if_then_else((val[((i*stride_1) + (k*stride_5))] <= T_1[(i*stride_3)]), T_1[(i*stride_3)], val[((i*stride_1) + (k*stride_5))], dtype=int32)
    }
  }
}
```

:::note
若对归约不熟悉，可以参考 [定义通用交换归约运算](https://tvm.apache.org/docs/how_to/work_with_schedules/reduction.html#general-reduction)。
:::

## 使用元组输入调度操作

虽然一次 batch 操作会有多个输出，但它们只能一起调度。

``` python
n = te.var("n")
m = te.var("m")
A0 = te.placeholder((m, n), name="A0")
B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] + 2, A0[i, j] * 3), name="B")
A1 = te.placeholder((m, n), name="A1")
C = te.compute((m, n), lambda i, j: A1[i, j] + B0[i, j], name="C")

s = te.create_schedule(C.op)
s[B0].compute_at(s[C], C.op.axis[0])
# 生成的 IR 代码：
print(tvm.lower(s, [A0, A1, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A0_1: handle, A1_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A0: Buffer(A0_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             A1: Buffer(A1_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [(stride_2: int32*m)], [], type="auto")}
  buffer_map = {A0_1: A0, A1_1: A1, C_1: C}
  preflattened_buffer_map = {A0_1: A0_3: Buffer(A0_2, float32, [m, n: int32], [stride, stride_3: int32], type="auto"), A1_1: A1_3: Buffer(A1_2, float32, [m, n], [stride_1, stride_4: int32], type="auto"), C_1: C_3: Buffer(C_2, float32, [m, n], [stride_2, stride_5: int32], type="auto")} {
  allocate(B.v0: Pointer(global float32), float32, [n]), storage_scope = global;
  allocate(B.v1: Pointer(global float32), float32, [n]), storage_scope = global;
  for (i: int32, 0, m) {
    for (j: int32, 0, n) {
      B.v0_1: Buffer(B.v0, float32, [n], [])[j] = (A0[((i*stride) + (j*stride_3))] + 2f32)
      B.v1_1: Buffer(B.v1, float32, [n], [])[j] = (A0[((i*stride) + (j*stride_3))]*3f32)
    }
    for (j_1: int32, 0, n) {
      C[((i*stride_2) + (j_1*stride_5))] = (A1[((i*stride_1) + (j_1*stride_4))] + B.v0_1[j_1])
    }
  }
}
```

## 总结

本教程介绍元组输入操作的用法。

* 描述常规的批量计算。
* 用元组输入描述归约操作。
* 注意，只能根据操作而不是张量来调度计算。

[下载 Python 源代码：tuple_inputs.py](https://tvm.apache.org/docs/_downloads/68abf665197871646fffcd0955bddad7/tuple_inputs.py)

[下载 Jupyter Notebook：tuple_inputs.ipynb](https://tvm.apache.org/docs/_downloads/a1417396e306d987107a7a39376ec261/tuple_inputs.ipynb)
