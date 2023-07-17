---
title: 规约（reduce）
---

# 规约（reduce）

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_schedules/reduction.html#sphx-glr-download-how-to-work-with-schedules-reduction-py) 下载完整的示例代码
:::

**作者**：[Tianqi Chen](https://tqchen.github.io/)

本文介绍如何在 TVM 中规约（reduce）。关联规约算子（如 sum/max/min）是线性代数运算的典型构造块。

``` python
from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np
```

## 描述行的总和

在 NumPy 语法中，计算行的总和可以写成 `B = numpy.sum(A, axis=1)`

下面几行描述了行求和操作。为创建一个规约公式，用 `te.reduce_axis` 声明了一个 reduction 轴，它接收规约的范围。 `te.sum` 接收要规约的表达式以及 reduction 轴，并计算声明范围内所有 k 值的总和。

等效的 C 代码如下：

``` c
for (int i = 0; i < n; ++i) {
  B[i] = 0;
  for (int k = 0; k < m; ++k) {
    B[i] = B[i] + A[i][k];
  }
}
```

``` python
n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), "k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
```

## Schedule 规约

有几种方法可以 Schedule Reduce，先打印出默认 Schedule 的 IR 代码。

``` python
s = te.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*n: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*n)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [n, m: int32], [stride, stride_2: int32], type="auto"), B_1: B_3: Buffer(B_2, float32, [n], [stride_1], type="auto")} {
  for (i: int32, 0, n) {
    B[(i*stride_1)] = 0f32
    for (k: int32, 0, m) {
      B[(i*stride_1)] = (B[(i*stride_1)] + A[((i*stride) + (k*stride_2))])
    }
  }
}
```

IR 代码与 C 代码非常相似，reduction 轴类似于普通轴，可以拆分。

以下代码按不同的因子将 B 的行轴和轴进行拆分，得到一个嵌套 reduction。

``` python
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
xo, xi = s[B].split(B.op.axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*n: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*n)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [n, m: int32], [stride, stride_2: int32], type="auto"), B_1: B_3: Buffer(B_2, float32, [n], [stride_1], type="auto")} {
  for (i.outer: int32, 0, floordiv((n + 31), 32)) {
    for (i.inner: int32, 0, 32) {
      if @tir.likely((((i.outer*32) + i.inner) < n), dtype=bool) {
        B[(((i.outer*32) + i.inner)*stride_1)] = 0f32
      }
      if @tir.likely((((i.outer*32) + i.inner) < n), dtype=bool) {
        for (k.outer: int32, 0, floordiv((m + 15), 16)) {
          for (k.inner: int32, 0, 16) {
            if @tir.likely((((k.outer*16) + k.inner) < m), dtype=bool) {
              let cse_var_1: int32 = ((i.outer*32) + i.inner)
              B[(cse_var_1*stride_1)] = (B[(cse_var_1*stride_1)] + A[((cse_var_1*stride) + (((k.outer*16) + k.inner)*stride_2))])
            }
          }
        }
      }
    }
  }
}
```

把 B 的行绑定到 GPU 线程，从而构建一个 GPU 内核。

``` python
s[B].bind(xo, te.thread_axis("blockIdx.x"))
s[B].bind(xi, te.thread_axis("threadIdx.x"))
print(tvm.lower(s, [A, B], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*n: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*n)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [n, m: int32], [stride, stride_2: int32], type="auto"), B_1: B_3: Buffer(B_2, float32, [n], [stride_1], type="auto")} {
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = floordiv((n + 31), 32);
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32 {
    if @tir.likely((((blockIdx.x*32) + threadIdx.x) < n), dtype=bool) {
      B[(((blockIdx.x*32) + threadIdx.x)*stride_1)] = 0f32
    }
    for (k.outer: int32, 0, floordiv((m + 15), 16)) {
      for (k.inner: int32, 0, 16) {
        if @tir.likely((((blockIdx.x*32) + threadIdx.x) < n), dtype=bool) {
          if @tir.likely((((k.outer*16) + k.inner) < m), dtype=bool) {
            B[(((blockIdx.x*32) + threadIdx.x)*stride_1)] = (B[(((blockIdx.x*32) + threadIdx.x)*stride_1)] + A[((((blockIdx.x*32) + threadIdx.x)*stride) + (((k.outer*16) + k.inner)*stride_2))])
          }
        }
      }
    }
  }
}
```

## 规约因式分解和并行化

构建规约时不能简单地在 reduction 轴上并行化，需要划分规约，将局部规约结果存储在数组中，然后再对临时数组进行规约。

rfactor 原语对计算进行了上述重写，在下面的调度中，B 的结果被写入一个临时结果 B.rf，分解后的维度成为 B.rf 的第一个维度。

``` python
s = te.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
print(tvm.lower(s, [A, B], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*n: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*n)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [n, m: int32], [stride, stride_2: int32], type="auto"), B_1: B_3: Buffer(B_2, float32, [n], [stride_1], type="auto")} {
  allocate(B.rf: Pointer(global float32), float32, [(n*16)]), storage_scope = global {
    for (k.inner: int32, 0, 16) {
      for (i: int32, 0, n) {
        B.rf_1: Buffer(B.rf, float32, [(16*n)], [])[((k.inner*n) + i)] = 0f32
        for (k.outer: int32, 0, floordiv((m + 15), 16)) {
          if @tir.likely((((k.outer*16) + k.inner) < m), dtype=bool) {
            B.rf_1[((k.inner*n) + i)] = (B.rf_1[((k.inner*n) + i)] + A[((i*stride) + (((k.outer*16) + k.inner)*stride_2))])
          }
        }
      }
    }
    for (ax0: int32, 0, n) {
      B[(ax0*stride_1)] = 0f32
      for (k.inner.v: int32, 0, 16) {
        B[(ax0*stride_1)] = (B[(ax0*stride_1)] + B.rf_1[((k.inner.v*n) + ax0)])
      }
    }
  }
}
```

B 的调度算子被重写为 B.f 的规约结果在第一个轴上的和。

``` python
print(s[B].op.body)
```

输出结果：

``` bash
[reduce(combiner=comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f]), source=[B.rf[k.inner.v, ax0]], init=[], axis=[iter_var(k.inner.v, range(min=0, ext=16))], where=(bool)1, value_index=0)]
```

## 跨线程规约

接下来可以在因子轴上进行并行化，这里 B 的 reduction 轴被标记为线程，如果唯一的 reduction 轴在设备中可以进行跨线程规约，则 TVM 允许将 reduction 轴标记为 thread。

也可以直接在规约轴上计算 BF。最终生成的内核会将行除以 blockIdx.x，将 threadIdx.y 列除以 threadIdx.x，最后对 threadIdx.x 进行跨线程规约。

``` python
xo, xi = s[B].split(s[B].op.axis[0], factor=32)
s[B].bind(xo, te.thread_axis("blockIdx.x"))
s[B].bind(xi, te.thread_axis("threadIdx.y"))
tx = te.thread_axis("threadIdx.x")
s[B].bind(s[B].op.reduce_axis[0], tx)
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
s[B].set_store_predicate(tx.var.equal(0))
fcuda = tvm.build(s, [A, B], "cuda")
print(fcuda.imported_modules[0].get_source())
```

输出结果：

``` c
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(512) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, int m, int n, int stride, int stride1, int stride2) {
  float B_rf[1];
  float red_buf0[1];
  B_rf[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < (m >> 4); ++k_outer) {
    if (((((int)blockIdx.x) * 32) + ((int)threadIdx.y)) < n) {
      B_rf[0] = (B_rf[0] + A[((((((int)blockIdx.x) * 32) + ((int)threadIdx.y)) * stride) + (((k_outer * 16) + ((int)threadIdx.x)) * stride1))]);
    }
  }
  for (int k_outer1 = 0; k_outer1 < (((m & 15) + 15) >> 4); ++k_outer1) {
    if (((((int)blockIdx.x) * 32) + ((int)threadIdx.y)) < n) {
      if (((((m >> 4) * 16) + (k_outer1 * 16)) + ((int)threadIdx.x)) < m) {
        B_rf[0] = (B_rf[0] + A[((((((int)blockIdx.x) * 32) + ((int)threadIdx.y)) * stride) + (((((m >> 4) * 16) + (k_outer1 * 16)) + ((int)threadIdx.x)) * stride1))]);
      }
    }
  }
  uint mask[1];
  float t0[1];
  red_buf0[0] = B_rf[0];
  mask[0] = (__activemask() & ((uint)(65535 << (((int)threadIdx.y) * 16))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 16), 32);
  if (((int)threadIdx.x) == 0) {
    B[(((((int)blockIdx.x) * 32) + ((int)threadIdx.y)) * stride2)] = red_buf0[0];
  }
}
```

结果内核与 NumPy 进行比较来验证结果内核的正确性。

``` python
nn = 128
dev = tvm.cuda(0)
a = tvm.nd.array(np.random.uniform(size=(nn, nn)).astype(A.dtype), dev)
b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), dev)
fcuda(a, b)
tvm.testing.assert_allclose(b.numpy(), np.sum(a.numpy(), axis=1), rtol=1e-4)
```

## 用二维规约描述卷积

在 TVM 中，用简单的二维规约来描述卷积（过滤器大小 = [3, 3]，步长 = [1, 1]）。

``` python
n = te.var("n")
Input = te.placeholder((n, n), name="Input")
Filter = te.placeholder((3, 3), name="Filter")
di = te.reduce_axis((0, 3), name="di")
dj = te.reduce_axis((0, 3), name="dj")
Output = te.compute(
    (n - 2, n - 2),
    lambda i, j: te.sum(Input[i + di, j + dj] * Filter[di, dj], axis=[di, dj]),
    name="Output",
)
s = te.create_schedule(Output.op)
print(tvm.lower(s, [Input, Filter, Output], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(Input_1: handle, Filter_1: handle, Output_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {Input: Buffer(Input_2: Pointer(float32), float32, [(stride: int32*n: int32)], [], type="auto"),
             Filter: Buffer(Filter_2: Pointer(float32), float32, [9], []),
             Output: Buffer(Output_2: Pointer(float32), float32, [((n - 2)*(n - 2))], [])}
  buffer_map = {Input_1: Input, Filter_1: Filter, Output_1: Output}
  preflattened_buffer_map = {Input_1: Input_3: Buffer(Input_2, float32, [n, n], [stride, stride_1: int32], type="auto"), Filter_1: Filter_3: Buffer(Filter_2, float32, [3, 3], []), Output_1: Output_3: Buffer(Output_2, float32, [(n - 2), (n - 2)], [])} {
  for (i: int32, 0, (n - 2)) {
    for (j: int32, 0, (n - 2)) {
      Output[((i*(n - 2)) + j)] = 0f32
      for (di: int32, 0, 3) {
        for (dj: int32, 0, 3) {
          Output[((i*(n - 2)) + j)] = (Output[((i*(n - 2)) + j)] + (Input[(((i + di)*stride) + ((j + dj)*stride_1))]*Filter[((di*3) + dj)]))
        }
      }
    }
  }
}
```

## 定义一般交换规约运算

除了 `te.sum`, `tvm.te.min` 和 `tvm.te.max` 等内置规约操作外，还可以通过 `te.comm_reducer` 定义交换规约操作。

``` python
n = te.var("n")
m = te.var("m")
product = te.comm_reducer(lambda x, y: x * y, lambda t: tvm.tir.const(1, dtype=t), name="product")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), name="k")
B = te.compute((n,), lambda i: product(A[i, k], axis=k), name="B")
```

:::note
执行涉及多个值的规约，例如 `argmax`，可以通过元组输入来完成。更多详细信息，请参阅 [使用协作输入描述规约](https://tvm.apache.org/docs/how_to/work_with_schedules/tuple_inputs.html#reduction-with-tuple-inputs)。
:::

## 总结

本教程演示了如何规约 schedule。

* 用 reduce_axis 描述规约。
* 如需并行性（parallelism），用 rfactor 来分解轴。
* 通过 `te.comm_reducer` 定义新的规约操作。


[下载 Python 源代码：reduction.py](https://tvm.apache.org/docs/_downloads/2a0982f8ca0176cb17713d28286536e4/reduction.py)

[下载 Jupyter Notebook：reduction.ipynb](https://tvm.apache.org/docs/_downloads/10d831d158490a9ee3abd1901806fc11/reduction.ipynb)
