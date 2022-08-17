---
title: 扫描和递归核
---

# 扫描和递归核

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_schedules/scan.html#sphx-glr-download-how-to-work-with-schedules-scan-py) 下载完整的示例代码
:::

**作者**：[Tianqi Chen](https://tqchen.github.io/)

下面介绍如何在 TVM 中进行递归计算（神经网络中的典型模式）。

``` python
from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np
```

TVM 用扫描算子来描述符号循环。以下扫描算子计算 X 列上的累积和。

扫描在张量的最高维度上进行。`s_state` 是描述扫描转换状态的占位符。`s_init` 描述如何初始化前 k 个时间步长，其第一个维度为 1，描述了如何初始化第一个时间步长的状态。

`s_update` 描述了如何更新时间步长 t 处的值，更新的值可通过状态占位符引用上一个时间步长的值。注意在当前或之后的时间步长引用 `s_state` 是无效的。

扫描包含状态占位符、初始值和更新描述。推荐列出扫描单元的输入，扫描的结果是一个张量—— `s_state` 在时域更新后的结果。

``` python
m = te.var("m")
n = te.var("n")
X = te.placeholder((m, n), name="X")
s_state = te.placeholder((m, n))
s_init = te.compute((1, n), lambda _, i: X[0, i])
s_update = te.compute((m, n), lambda t, i: s_state[t - 1, i] + X[t, i])
s_scan = tvm.te.scan(s_init, s_update, s_state, inputs=[X])
```

## 调度扫描单元

通过分别调度 update 和 init 部分来调度扫描体。注意，调度更新部分的第一个迭代维度是无效的。要在时间迭代上拆分，用户可以在 scan_op.scan_axis 上进行调度。

``` python
s = te.create_schedule(s_scan.op)
num_thread = 256
block_x = te.thread_axis("blockIdx.x")
thread_x = te.thread_axis("threadIdx.x")
xo, xi = s[s_init].split(s_init.op.axis[1], factor=num_thread)
s[s_init].bind(xo, block_x)
s[s_init].bind(xi, thread_x)
xo, xi = s[s_update].split(s_update.op.axis[1], factor=num_thread)
s[s_update].bind(xo, block_x)
s[s_update].bind(xi, thread_x)
print(tvm.lower(s, [X, s_scan], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(X_1: handle, scan_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {X: Buffer(X_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             scan: Buffer(scan_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto")}
  buffer_map = {X_1: X, scan_1: scan}
  preflattened_buffer_map = {X_1: X_3: Buffer(X_2, float32, [m, n: int32], [stride, stride_2: int32], type="auto"), scan_1: scan_3: Buffer(scan_2, float32, [m, n], [stride_1, stride_3: int32], type="auto")} {
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = floordiv((n + 255), 256);
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 256;
  if @tir.likely((((blockIdx.x*256) + threadIdx.x) < n), dtype=bool) {
    scan[(((blockIdx.x*256) + threadIdx.x)*stride_3)] = X[(((blockIdx.x*256) + threadIdx.x)*stride_2)]
  }
  for (scan.idx: int32, 0, (m - 1)) {
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = floordiv((n + 255), 256);
    attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 256;
    if @tir.likely((((blockIdx.x*256) + threadIdx.x) < n), dtype=bool) {
      let cse_var_1: int32 = (scan.idx + 1)
      scan[((cse_var_1*stride_1) + (((blockIdx.x*256) + threadIdx.x)*stride_3))] = (scan[((scan.idx*stride_1) + (((blockIdx.x*256) + threadIdx.x)*stride_3))] + X[((cse_var_1*stride) + (((blockIdx.x*256) + threadIdx.x)*stride_2))])
    }
  }
}
```

## 构建和验证

可以像其他 TVM 内核一样构建扫描内核，这里用 numpy 来验证结果的正确性。

``` python
fscan = tvm.build(s, [X, s_scan], "cuda", name="myscan")
dev = tvm.cuda(0)
n = 1024
m = 10
a_np = np.random.uniform(size=(m, n)).astype(s_scan.dtype)
a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(np.zeros((m, n), dtype=s_scan.dtype), dev)
fscan(a, b)
tvm.testing.assert_allclose(b.numpy(), np.cumsum(a_np, axis=0))
```

## 多阶段扫描单元

以上示例用 s_update 中的一个张量计算阶段描述了扫描单元，可以在扫描单元中使用多个张量级。

以下代码演示了有两个阶段操作的扫描单元中的扫描过程：

``` python
m = te.var("m")
n = te.var("n")
X = te.placeholder((m, n), name="X")
s_state = te.placeholder((m, n))
s_init = te.compute((1, n), lambda _, i: X[0, i])
s_update_s1 = te.compute((m, n), lambda t, i: s_state[t - 1, i] * 2, name="s1")
s_update_s2 = te.compute((m, n), lambda t, i: s_update_s1[t, i] + X[t, i], name="s2")
s_scan = tvm.te.scan(s_init, s_update_s2, s_state, inputs=[X])
```

这些中间张量可以正常调度。为了确保正确性，TVM 创建了一个组约束——禁用扫描循环之外的 compute_at 位置的扫描体。

``` python
s = te.create_schedule(s_scan.op)
xo, xi = s[s_update_s2].split(s_update_s2.op.axis[1], factor=32)
s[s_update_s1].compute_at(s[s_update_s2], xo)
```

输出结果：

``` bash
print(tvm.lower(s, [X, s_scan], simple_mode=True))
@main = primfn(X_1: handle, scan_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {X: Buffer(X_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             scan: Buffer(scan_2: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto")}
  buffer_map = {X_1: X, scan_1: scan}
  preflattened_buffer_map = {X_1: X_3: Buffer(X_2, float32, [m, n: int32], [stride, stride_2: int32], type="auto"), scan_1: scan_3: Buffer(scan_2, float32, [m, n], [stride_1, stride_3: int32], type="auto")} {
  allocate(s1: Pointer(global float32), float32, [32]), storage_scope = global {
    for (i: int32, 0, n) {
      scan[(i*stride_3)] = X[(i*stride_2)]
    }
    for (scan.idx: int32, 0, (m - 1)) {
      for (i.outer: int32, 0, floordiv((n + 31), 32)) {
        for (i_1: int32, 0, 32) {
          if @tir.likely((((i.outer*32) + i_1) < n), dtype=bool) {
            s1_1: Buffer(s1, float32, [32], [])[i_1] = (scan[((scan.idx*stride_1) + (((i.outer*32) + i_1)*stride_3))]*2f32)
          }
        }
        for (i.inner: int32, 0, 32) {
          if @tir.likely((((i.outer*32) + i.inner) < n), dtype=bool) {
            let cse_var_2: int32 = (scan.idx + 1)
            let cse_var_1: int32 = ((i.outer*32) + i.inner)
            scan[((cse_var_2*stride_1) + (cse_var_1*stride_3))] = (s1_1[i.inner] + X[((cse_var_2*stride) + (cse_var_1*stride_2))])
          }
        }
      }
    }
  }
}
```

## 多状态

对于像 RNN 这样的复杂应用，需要多个递归状态。扫描支持多个递归状态，以下示例演示如何构建具有两种状态的递归。

``` python
m = te.var("m")
n = te.var("n")
l = te.var("l")
X = te.placeholder((m, n), name="X")
s_state1 = te.placeholder((m, n))
s_state2 = te.placeholder((m, l))
s_init1 = te.compute((1, n), lambda _, i: X[0, i])
s_init2 = te.compute((1, l), lambda _, i: 0.0)
s_update1 = te.compute((m, n), lambda t, i: s_state1[t - 1, i] + X[t, i])
s_update2 = te.compute((m, l), lambda t, i: s_state2[t - 1, i] + s_state1[t - 1, 0])
s_scan1, s_scan2 = tvm.te.scan(
    [s_init1, s_init2], [s_update1, s_update2], [s_state1, s_state2], inputs=[X]
)
s = te.create_schedule(s_scan1.op)
print(tvm.lower(s, [X, s_scan1, s_scan2], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(X_1: handle, scan_2: handle, scan_3: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {X: Buffer(X_2: Pointer(float32), float32, [(stride: int32*m: int32)], [], type="auto"),
             scan: Buffer(scan_4: Pointer(float32), float32, [(stride_1: int32*m)], [], type="auto"),
             scan_1: Buffer(scan_5: Pointer(float32), float32, [(stride_2: int32*m)], [], type="auto")}
  buffer_map = {X_1: X, scan_2: scan, scan_3: scan_1}
  preflattened_buffer_map = {X_1: X_3: Buffer(X_2, float32, [m, n: int32], [stride, stride_3: int32], type="auto"), scan_2: scan_6: Buffer(scan_4, float32, [m, n], [stride_1, stride_4: int32], type="auto"), scan_3: scan_7: Buffer(scan_5, float32, [m, l: int32], [stride_2, stride_5: int32], type="auto")} {
  for (i: int32, 0, n) {
    scan[(i*stride_4)] = X[(i*stride_3)]
  }
  for (i_1: int32, 0, l) {
    scan_1[(i_1*stride_5)] = 0f32
  }
  for (scan.idx: int32, 0, (m - 1)) {
    for (i_2: int32, 0, n) {
      let cse_var_1: int32 = (scan.idx + 1)
      scan[((cse_var_1*stride_1) + (i_2*stride_4))] = (scan[((scan.idx*stride_1) + (i_2*stride_4))] + X[((cse_var_1*stride) + (i_2*stride_3))])
    }
    for (i_3: int32, 0, l) {
      scan_1[(((scan.idx + 1)*stride_2) + (i_3*stride_5))] = (scan_1[((scan.idx*stride_2) + (i_3*stride_5))] + scan[(scan.idx*stride_1)])
    }
  }
}
```

## 总结

本教程演示了如何使用扫描原语。

* 用 init 和 update 描述扫描。
* 将扫描单元当作正常 schedule 进行调度。
* 对于复杂的工作负载，在扫描单元中使用多个状态和步骤。

[下载 Python 源代码：scan.py](https://tvm.apache.org/docs/_downloads/8c7d8fd6a4b93bcff1f5573943dd02f4/scan.py)

[下载 Jupyter Notebook：scan.ipynb](https://tvm.apache.org/docs/_downloads/729378592a96230b4f7be71b44da43a4/scan.ipynb)