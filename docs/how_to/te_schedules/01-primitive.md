---
title: TVM 中的 Schedule 原语
---

# TVM 中的 Schedule 原语

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html#sphx-glr-download-how-to-work-with-schedules-schedule-primitives-py) 下载完整的示例代码
:::

**作者**：[Ziheng Jiang](https://github.com/ZihengJiang)

TVM 是一种用于高效构建内核的领域特定语言。

本教程展示了如何通过 TVM 提供的各种原语来调度计算。

``` python
from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np
```

计算相同结果的方法众多，然而，不同的方法会导致局部性和性能各异，因此 TVM 要求用户借助 **Schedule** 执行计算。

**Schedule** 是一组计算转换，可用于转换程序中的循环计算。

``` python
# 声明变量，供之后使用
n = te.var("n")
m = te.var("m")
```

Schedule 可由算子列表创建，它默认以行优先的方式串行计算张量。

``` python
# 声明一个矩阵元素乘法
A = te.placeholder((m, n), name="A")
B = te.placeholder((m, n), name="B")
C = te.compute((m, n), lambda i, j: A[i, j] * B[i, j], name="C")

s = te.create_schedule([C.op])
# lower 会将计算从定义转换为实际可调用的函数。
# 使用参数 `simple_mode=True` 会返回一个可读的类 C 的语句，这里用它来打印 schedule 结果。
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [(stride_2: int32*m)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [m, n: int32], [stride, stride_3: int32], type="auto"), B_1: B_3: Buffer(B_2, float32, [m, n], [stride_1, stride_4: int32], type="auto"), C_1: C_3: Buffer(C_2, float32, [m, n], [stride_2, stride_5: int32], type="auto")} {
  for (i: int32, 0, m) {
    for (j: int32, 0, n) {
      C[((i*stride_2) + (j*stride_5))] = (A[((i*stride) + (j*stride_3))]*B[((i*stride_1) + (j*stride_4))])
    }
  }
}
```

一个 Schedule 由多个 Stage 组成，一个 **Stage** 代表一个操作的 schedule。每个 stage 的调度都有多种方法：

## split

`split` 可根据 `factor` 将指定 axis 拆分为两个 axis。

``` python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] * 2, name="B")

s = te.create_schedule(B.op)
xo, xi = s[B].split(B.op.axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [m], [stride], type="auto"), B_1: B_3: Buffer(B_2, float32, [m], [stride_1], type="auto")} {
  for (i.outer: int32, 0, floordiv((m + 31), 32)) {
    for (i.inner: int32, 0, 32) {
      if @tir.likely((((i.outer*32) + i.inner) < m), dtype=bool) {
        let cse_var_1: int32 = ((i.outer*32) + i.inner)
        B[(cse_var_1*stride_1)] = (A[(cse_var_1*stride)]*2f32)
      }
    }
  }
}
```

也可用 `nparts` 来拆分 axis，它拆分 axis 的方式与 `factor` 相反。

``` python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i], name="B")

s = te.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], nparts=32)
print(tvm.lower(s, [A, B], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [m], [stride], type="auto"), B_1: B_3: Buffer(B_2, float32, [m], [stride_1], type="auto")} {
  for (i.outer: int32, 0, 32) {
    for (i.inner: int32, 0, floordiv((m + 31), 32)) {
      if @tir.likely(((i.inner + (i.outer*floordiv((m + 31), 32))) < m), dtype=bool) {
        B[((i.inner + (i.outer*floordiv((m + 31), 32)))*stride_1)] = A[((i.inner + (i.outer*floordiv((m + 31), 32)))*stride)]
      }
    }
  }
}
```

## tile

`tile` 可在两个 axis 上逐块执行计算。

``` python
A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j], name="B")

s = te.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
print(tvm.lower(s, [A, B], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [m, n: int32], [stride, stride_2: int32], type="auto"), B_1: B_3: Buffer(B_2, float32, [m, n], [stride_1, stride_3: int32], type="auto")} {
  for (i.outer: int32, 0, floordiv((m + 9), 10)) {
    for (j.outer: int32, 0, floordiv((n + 4), 5)) {
      for (i.inner: int32, 0, 10) {
        if @tir.likely((((i.outer*10) + i.inner) < m), dtype=bool) {
          for (j.inner: int32, 0, 5) {
            if @tir.likely((((j.outer*5) + j.inner) < n), dtype=bool) {
              let cse_var_2: int32 = ((j.outer*5) + j.inner)
              let cse_var_1: int32 = ((i.outer*10) + i.inner)
              B[((cse_var_1*stride_1) + (cse_var_2*stride_3))] = A[((cse_var_1*stride) + (cse_var_2*stride_2))]
            }
          }
        }
      }
    }
  }
}
```

## fuse

`fuse` 可将一个计算的两个连续 axis 融合。

``` python
A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j], name="B")

s = te.create_schedule(B.op)
# 首先调用 tile 平铺到四个 axis: (i.outer, j.outer, i.inner, j.inner)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# 然后将 (i.inner, j.inner) 融合成一个轴： (i.inner.j.inner.fused)
fused = s[B].fuse(xi, yi)
print(tvm.lower(s, [A, B], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [m, n: int32], [stride, stride_2: int32], type="auto"), B_1: B_3: Buffer(B_2, float32, [m, n], [stride_1, stride_3: int32], type="auto")} {
  for (i.outer: int32, 0, floordiv((m + 9), 10)) {
    for (j.outer: int32, 0, floordiv((n + 4), 5)) {
      for (i.inner.j.inner.fused: int32, 0, 50) {
        if @tir.likely((((i.outer*10) + floordiv(i.inner.j.inner.fused, 5)) < m), dtype=bool) {
          if @tir.likely((((j.outer*5) + floormod(i.inner.j.inner.fused, 5)) < n), dtype=bool) {
            let cse_var_2: int32 = ((j.outer*5) + floormod(i.inner.j.inner.fused, 5))
            let cse_var_1: int32 = ((i.outer*10) + floordiv(i.inner.j.inner.fused, 5))
            B[((cse_var_1*stride_1) + (cse_var_2*stride_3))] = A[((cse_var_1*stride) + (cse_var_2*stride_2))]
          }
        }
      }
    }
  }
}
```

## reorder

`reorder` 可按指定的顺序对 axis 重新排序。

``` python
A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j], name="B")

s = te.create_schedule(B.op)
# 首先调用 tile 平铺到四个轴: (i.outer, j.outer, i.inner, j.inner)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# 然后将 axis 重新排序：（i.inner，j.outer，i.outer，j.inner）
s[B].reorder(xi, yo, xo, yi)
print(tvm.lower(s, [A, B], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [m, n: int32], [stride, stride_2: int32], type="auto"), B_1: B_3: Buffer(B_2, float32, [m, n], [stride_1, stride_3: int32], type="auto")} {
  for (i.inner: int32, 0, 10) {
    for (j.outer: int32, 0, floordiv((n + 4), 5)) {
      for (i.outer: int32, 0, floordiv((m + 9), 10)) {
        if @tir.likely((((i.outer*10) + i.inner) < m), dtype=bool) {
          for (j.inner: int32, 0, 5) {
            if @tir.likely((((j.outer*5) + j.inner) < n), dtype=bool) {
              let cse_var_2: int32 = ((j.outer*5) + j.inner)
              let cse_var_1: int32 = ((i.outer*10) + i.inner)
              B[((cse_var_1*stride_1) + (cse_var_2*stride_3))] = A[((cse_var_1*stride) + (cse_var_2*stride_2))]
            }
          }
        }
      }
    }
  }
}
```

## bind

`bind` 可将指定 axis 与线程 axis 绑定，常用于 GPU 编程。

``` python
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: A[i] * 2, name="B")

s = te.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], factor=64)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
print(tvm.lower(s, [A, B], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*n: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*n)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [n], [stride], type="auto"), B_1: B_3: Buffer(B_2, float32, [n], [stride_1], type="auto")} {
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = floordiv((n + 63), 64);
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
  if @tir.likely((((blockIdx.x*64) + threadIdx.x) < n), dtype=bool) {
    B[(((blockIdx.x*64) + threadIdx.x)*stride_1)] = (A[(((blockIdx.x*64) + threadIdx.x)*stride)]*2f32)
  }
}
```

## compute_at

对于包含多个算子的 schedule，TVM 默认会分别计算 root 处的张量。

``` python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")

s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [(stride_2: int32*m)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [m], [stride], type="auto"), B_1: B_3: Buffer(B_2, float32, [m], [stride_1], type="auto"), C_1: C_3: Buffer(C_2, float32, [m], [stride_2], type="auto")} {
  for (i: int32, 0, m) {
    B[(i*stride_1)] = (A[(i*stride)] + 1f32)
  }
  for (i_1: int32, 0, m) {
    C[(i_1*stride_2)] = (B[(i_1*stride_1)]*2f32)
  }
}
```

`compute_at` 可将 B 的计算移动到 C 计算的首个 axis 中。

``` python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")

s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [(stride_2: int32*m)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [m], [stride], type="auto"), B_1: B_3: Buffer(B_2, float32, [m], [stride_1], type="auto"), C_1: C_3: Buffer(C_2, float32, [m], [stride_2], type="auto")} {
  for (i: int32, 0, m) {
    B[(i*stride_1)] = (A[(i*stride)] + 1f32)
    C[(i*stride_2)] = (B[(i*stride_1)]*2f32)
  }
}
```

## compute_inline

`compute_inline` 可将 stage 标记为 inline，然后扩展计算体，并将其插入到需要张量的地址。

``` python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")

s = te.create_schedule(C.op)
s[B].compute_inline()
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [(stride_2: int32*m)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [m], [stride], type="auto"), B_1: B_3: Buffer(B_2, float32, [m], [stride_1], type="auto"), C_1: C_3: Buffer(C_2, float32, [m], [stride_2], type="auto")} {
  for (i: int32, 0, m) {
    C[(i*stride_2)] = ((A[(i*stride)] + 1f32)*2f32)
  }
}
```

## compute_root

`compute_root` 可将一个 stage 的计算移动到 root。

``` python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")

s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
s[B].compute_root()
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [(stride_2: int32*m)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [m], [stride], type="auto"), B_1: B_3: Buffer(B_2, float32, [m], [stride_1], type="auto"), C_1: C_3: Buffer(C_2, float32, [m], [stride_2], type="auto")} {
  for (i: int32, 0, m) {
    B[(i*stride_1)] = (A[(i*stride)] + 1f32)
  }
  for (i_1: int32, 0, m) {
    C[(i_1*stride_2)] = (B[(i_1*stride_1)]*2f32)
  }
}
```

## 总结

本教程介绍了 TVM 中的 schedule 原语（使得用户可以轻松、灵活地调度计算）。

为提高内核的性能，一般的工作流程如下：

* 通过一系列操作描述你的计算。
* 使用原语来调度计算。
* 编译并运行，查看性能差异。
* 根据运行结果来调整 schedule。


[下载 Python 源代码：schedule_primitives.py](https://tvm.apache.org/docs/_downloads/da47fa2ad30c4b6921171c97e72f36a9/schedule_primitives.py)

[下载 Jupyter Notebook：schedule_primitives.ipynb](https://tvm.apache.org/docs/_downloads/b78f1a6e1b2c2fb073a791dc258a1d7d/schedule_primitives.ipynb)