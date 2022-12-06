---
title: TOPI 简介
---

# TOPI 简介

:::note
单击 [此处](https://tvm.apache.org/docs/tutorial/intro_topi.html#sphx-glr-download-tutorial-intro-topi-py) 下载完整的示例代码
:::

**作者**：[Ehsan M. Kermani](https://github.com/ehsanmok)

这是 TVM 算子清单（TOPI）的入门教程。 TOPI 提供了 numpy 风格的通用操作和 schedule，其抽象程度高于 TVM。本教程将介绍 TOPI 是如何使得 TVM 中的代码不那么样板化的。

``` python
import tvm
import tvm.testing
from tvm import te
from tvm import topi
import numpy as np
```

## 基本示例

让我们回顾一下行求和操作（例如 `B = numpy.sum(A, axis=1)`）。要计算二维 TVM 张量 A 的行之和，应指定符号运算以及 schedule，如下所示：

``` python
n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), "k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
s = te.create_schedule(B.op)
```

输入以下命令查看可读的 IR 代码：

``` bash
print(tvm.lower(s, [A], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*n: int32)], [], type="auto")}
  buffer_map = {A_1: A}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [n, m: int32], [stride, stride_1: int32], type="auto")} {
  allocate(B: Pointer(global float32), float32, [n]), storage_scope = global;
  for (i: int32, 0, n) {
    B_1: Buffer(B, float32, [n], [])[i] = 0f32
    for (k: int32, 0, m) {
      B_1[i] = (B_1[i] + A[((i*stride) + (k*stride_1))])
    }
  }
}
```

然而，必须为这样一个常用的操作定义 reduce 轴，并用 `te.compute` 定义显式计算。幸运的是，可以用 `topi.sum`（类似 `numpy.sum`）来替换这两行：

``` python
C = topi.sum(A, axis=1)
ts = te.create_schedule(C.op)
print(tvm.lower(ts, [A], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*n: int32)], [], type="auto")}
  buffer_map = {A_1: A}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [n, m: int32], [stride, stride_1: int32], type="auto")} {
  allocate(A_red: Pointer(global float32), float32, [n]), storage_scope = global;
  for (ax0: int32, 0, n) {
    A_red_1: Buffer(A_red, float32, [n], [])[ax0] = 0f32
    for (k1: int32, 0, m) {
      A_red_1[ax0] = (A_red_1[ax0] + A[((ax0*stride) + (k1*stride_1))])
    }
  }
}
```

## Numpy 风格的算子重载

可用 `topi.broadcast_add` 添加两个张量（其 shape 可广播，且是特定的）。TOPI 为此类常见操作提供了算子重载使其更简短。例如：

``` python
x, y = 100, 10
a = te.placeholder((x, y, y), name="a")
b = te.placeholder((y, y), name="b")
c = a + b  # 等价于 topi.broadcast_add
d = a * b  # 等价于 topi.broadcast_mul
```

TOPI 使用相同的语法重载，将原语 (*int, float*) 广播到张量 `d - 3.14`。

## 通用调度和融合操作

前面已经展示了 TOPI 如何使我们免于用低级 API 编写显式的计算过程，但调度过程还是和以前一样。TOPI 还基于给定的上下文提供了更高级的调度方案。可以仅用 `topi.generic.schedule_reduce` 调度下面以 `topi.sum` 结尾的一系列操作，以 CUDA 为例：

``` python
e = topi.elemwise_sum([c, d])
f = e / 2.0
g = topi.sum(f)
with tvm.target.cuda():
    sg = topi.cuda.schedule_reduce(g)
    print(tvm.lower(sg, [a, b], simple_mode=True))
```

输出结果：

``` bash
/workspace/python/tvm/target/target.py:377: UserWarning: Try specifying cuda arch by adding 'arch=sm_xx' to your target.
  warnings.warn("Try specifying cuda arch by adding 'arch=sm_xx' to your target.")
@main = primfn(a_1: handle, b_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {a: Buffer(a_2: Pointer(float32), float32, [10000], []),
             b: Buffer(b_2: Pointer(float32), float32, [100], [])}
  buffer_map = {a_1: a, b_1: b}
  preflattened_buffer_map = {a_1: a_3: Buffer(a_2, float32, [100, 10, 10], []), b_1: b_3: Buffer(b_2, float32, [10, 10], [])} {
  allocate(T_divide_red: Pointer(global float32), float32, [1]), storage_scope = global;
  attr [IterVar(threadIdx.x: int32, [0:1024], "ThreadIndex", "threadIdx.x")] "thread_extent" = 1024;
  allocate(T_divide_red.rf: Pointer(local float32), float32, [1]), storage_scope = local;
  allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local {
    T_divide_red.rf_1: Buffer(T_divide_red.rf, float32, [1], [], scope="local", align=4)[0] = 0f32
    for (k0.k1.fused.k2.fused.outer: int32, 0, 10) {
      if @tir.likely((((((k0.k1.fused.k2.fused.outer*64) + floordiv(threadIdx.x, 16)) < 625) && (((k0.k1.fused.k2.fused.outer*64) + floordiv(threadIdx.x, 16)) < 625)) && (((k0.k1.fused.k2.fused.outer*64) + floordiv(threadIdx.x, 16)) < 625)), dtype=bool) {
        T_divide_red.rf_1[0] = (T_divide_red.rf_1[0] + (((a[((k0.k1.fused.k2.fused.outer*1024) + threadIdx.x)] + b[((floordiv(floormod(((k0.k1.fused.k2.fused.outer*12) + floordiv(threadIdx.x, 2)), 50), 5)*10) + floormod(((k0.k1.fused.k2.fused.outer*4) + threadIdx.x), 10))]) + (a[((k0.k1.fused.k2.fused.outer*1024) + threadIdx.x)]*b[((floordiv(floormod(((k0.k1.fused.k2.fused.outer*12) + floordiv(threadIdx.x, 2)), 50), 5)*10) + floormod(((k0.k1.fused.k2.fused.outer*4) + threadIdx.x), 10))]))*0.5f32))
      }
    }
    attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
    @tir.tvm_thread_allreduce(1u32, T_divide_red.rf_1[0], True, reduce_temp0_1: Buffer(reduce_temp0, float32, [1], [], scope="local")[0], threadIdx.x, dtype=handle)
    if (threadIdx.x == 0) {
      T_divide_red_1: Buffer(T_divide_red, float32, [1], [], align=4)[0] = reduce_temp0_1[0]
    }
  }
}
```

如上所示，计算的调度阶段是累积的，可以输入以下命令来查看：

``` python
print(sg.stages)
```

输出结果：

``` bash
[stage(a, placeholder(a, 0x228afb00)), stage(b, placeholder(b, 0x22097c90)), stage(T_add, compute(T_add, body=[(a[ax0, ax1, ax2] + b[ax1, ax2])], axis=[iter_var(ax0, range(min=0, ext=100)), iter_var(ax1, range(min=0, ext=10)), iter_var(ax2, range(min=0, ext=10))], reduce_axis=[], tag=broadcast, attrs={})), stage(T_multiply, compute(T_multiply, body=[(a[ax0, ax1, ax2]*b[ax1, ax2])], axis=[iter_var(ax0, range(min=0, ext=100)), iter_var(ax1, range(min=0, ext=10)), iter_var(ax2, range(min=0, ext=10))], reduce_axis=[], tag=broadcast, attrs={})), stage(T_elemwise_sum, compute(T_elemwise_sum, body=[(T_add[ax0, ax1, ax2] + T_multiply[ax0, ax1, ax2])], axis=[iter_var(ax0, range(min=0, ext=100)), iter_var(ax1, range(min=0, ext=10)), iter_var(ax2, range(min=0, ext=10))], reduce_axis=[], tag=elemwise, attrs={})), stage(T_divide, compute(T_divide, body=[(T_elemwise_sum[ax0, ax1, ax2]/2f)], axis=[iter_var(ax0, range(min=0, ext=100)), iter_var(ax1, range(min=0, ext=10)), iter_var(ax2, range(min=0, ext=10))], reduce_axis=[], tag=elemwise, attrs={})), stage(T_divide_red.rf, compute(T_divide_red.rf, body=[reduce(combiner=comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f]), source=[T_divide[floordiv(floordiv((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)), 10), 10), floormod(floordiv((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)), 10), 10), floormod((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)), 10)]], init=[], axis=[iter_var(k0.k1.fused.k2.fused.outer, range(min=0, ext=10))], where=tir.likely((((floordiv(floordiv((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)), 10), 10) < 100) && (floordiv((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)), 10) < 1000)) && ((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)) < 10000))), value_index=0)], axis=[iter_var(k0.k1.fused.k2.fused.inner, range(min=0, ext=1024))], reduce_axis=[iter_var(k0.k1.fused.k2.fused.outer, range(min=0, ext=10))], tag=, attrs={})), stage(T_divide_red, compute(T_divide_red.repl, body=[reduce(combiner=comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f]), source=[T_divide_red.rf[k0.k1.fused.k2.fused.inner.v]], init=[], axis=[iter_var(k0.k1.fused.k2.fused.inner.v, range(min=0, ext=1024))], where=(bool)1, value_index=0)], axis=[], reduce_axis=[iter_var(k0.k1.fused.k2.fused.inner.v, range(min=0, ext=1024))], tag=, attrs={}))]
```

可通过与 `numpy` 结果对比来验证其正确性，如下所示：

``` python
func = tvm.build(sg, [a, b, g], "cuda")
dev = tvm.cuda(0)
a_np = np.random.uniform(size=(x, y, y)).astype(a.dtype)
b_np = np.random.uniform(size=(y, y)).astype(b.dtype)
g_np = np.sum(np.add(a_np + b_np, a_np * b_np) / 2.0)
a_nd = tvm.nd.array(a_np, dev)
b_nd = tvm.nd.array(b_np, dev)
g_nd = tvm.nd.array(np.zeros(g_np.shape, dtype=g_np.dtype), dev)
func(a_nd, b_nd, g_nd)
tvm.testing.assert_allclose(g_nd.numpy(), g_np, rtol=1e-5)
```

TOPI 还提供了常见神经网络操作，例如对优化的 schedule 进行 *softmax*：

``` python
tarray = te.placeholder((512, 512), name="tarray")
softmax_topi = topi.nn.softmax(tarray)
with tvm.target.Target("cuda"):
    sst = topi.cuda.schedule_softmax(softmax_topi)
    print(tvm.lower(sst, [tarray], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(tarray_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {tarray: Buffer(tarray_2: Pointer(float32), float32, [262144], [])}
  buffer_map = {tarray_1: tarray}
  preflattened_buffer_map = {tarray_1: tarray_3: Buffer(tarray_2, float32, [512, 512], [])} {
  allocate(T_softmax_norm: Pointer(global float32x4), float32x4, [65536]), storage_scope = global;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 512;
  allocate(normal_reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local;
  allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local;
  allocate(T_softmax_exp: Pointer(warp float32), float32, [512]), storage_scope = warp;
  allocate(normal_reduce_temp0_1: Pointer(local float32), float32, [1]), storage_scope = local;
  allocate(reduce_temp0_1: Pointer(local float32), float32, [1]), storage_scope = local {
    attr [IterVar(threadIdx.x: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32 {
      normal_reduce_temp0_2: Buffer(normal_reduce_temp0, float32, [1], [], scope="local")[0] = -3.40282e+38f32
      for (k.inner: int32, 0, 16) {
        normal_reduce_temp0_2[0] = max(normal_reduce_temp0_2[0], tarray[(((blockIdx.x*512) + (threadIdx.x*16)) + k.inner)])
      }
      attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
      @tir.tvm_thread_allreduce(1u32, normal_reduce_temp0_2[0], True, reduce_temp0_2: Buffer(reduce_temp0, float32, [1], [], scope="local")[0], threadIdx.x, dtype=handle)
      for (i1.inner.outer: int32, 0, 4) {
        let cse_var_1: int32 = (i1.inner.outer*4)
        T_softmax_exp_1: Buffer(T_softmax_exp, float32, [512], [], scope="warp")[ramp(((threadIdx.x*16) + cse_var_1), 1, 4)] = @tir.exp((tarray[ramp((((blockIdx.x*512) + (threadIdx.x*16)) + cse_var_1), 1, 4)] - broadcast(reduce_temp0_3: Buffer(reduce_temp0, float32, [1], [], scope="local", align=4)[0], 4)), dtype=float32x4)
      }
    }
    attr [IterVar(threadIdx.x, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32 {
      normal_reduce_temp0_3: Buffer(normal_reduce_temp0_1, float32, [1], [], scope="local")[0] = 0f32
      for (k.inner_1: int32, 0, 16) {
        normal_reduce_temp0_3[0] = (normal_reduce_temp0_3[0] + T_softmax_exp_1[((threadIdx.x*16) + k.inner_1)])
      }
      attr [meta[tir.CommReducer][1]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
      @tir.tvm_thread_allreduce(1u32, normal_reduce_temp0_3[0], True, reduce_temp0_4: Buffer(reduce_temp0_1, float32, [1], [], scope="local")[0], threadIdx.x, dtype=handle)
      for (i1.inner.outer_1: int32, 0, 4) {
        T_softmax_norm_1: Buffer(T_softmax_norm, float32x4, [65536], [])[(((blockIdx.x*128) + (threadIdx.x*4)) + i1.inner.outer_1)] = (T_softmax_exp_1[ramp(((threadIdx.x*16) + (i1.inner.outer_1*4)), 1, 4)] / broadcast(reduce_temp0_5: Buffer(reduce_temp0_1, float32, [1], [], scope="local", align=4)[0], 4))
      }
    }
  }
}
```

## 融合卷积

可将 `topi.nn.conv2d` 和 `topi.nn.relu` 融合在一起。

:::note
TOPI 函数都是通用函数，不同的后端实现性能优化的方式不同。所有的后端都必须在 compute 声明和 schedule 范围内调用它们。 TVM 会选择调用目标信息的正确函数。
:::

``` python
data = te.placeholder((1, 3, 224, 224))
kernel = te.placeholder((10, 3, 5, 5))

with tvm.target.Target("cuda"):
    conv = topi.cuda.conv2d_nchw(data, kernel, 1, 2, 1)
    out = topi.nn.relu(conv)
    sconv = topi.cuda.schedule_conv2d_nchw([out])
    print(tvm.lower(sconv, [data, kernel], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(placeholder_2: handle, placeholder_3: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {placeholder: Buffer(placeholder_4: Pointer(float32), float32, [150528], []),
             placeholder_1: Buffer(placeholder_5: Pointer(float32), float32, [750], [])}
  buffer_map = {placeholder_2: placeholder, placeholder_3: placeholder_1}
  preflattened_buffer_map = {placeholder_2: placeholder_6: Buffer(placeholder_4, float32, [1, 3, 224, 224], []), placeholder_3: placeholder_7: Buffer(placeholder_5, float32, [10, 3, 5, 5], [])} {
  allocate(compute: Pointer(global float32), float32, [501760]), storage_scope = global;
  attr [IterVar(blockIdx.z: int32, (nullptr), "ThreadIndex", "blockIdx.z")] "thread_extent" = 5;
  allocate(conv2d_nchw: Pointer(local float32), float32, [14]), storage_scope = local;
  allocate(pad_temp.shared: Pointer(shared float32), float32, [112]), storage_scope = shared;
  allocate(placeholder.shared: Pointer(shared float32), float32, [2]), storage_scope = shared;
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 224;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 2;
  attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
    conv2d_nchw_1: Buffer(conv2d_nchw, float32, [4], [], scope="local", align=8)[0] = 0f32
    conv2d_nchw_1[2] = 0f32
    conv2d_nchw_1[4] = 0f32
    conv2d_nchw_1[6] = 0f32
    conv2d_nchw_1[8] = 0f32
    conv2d_nchw_1[10] = 0f32
    conv2d_nchw_1[12] = 0f32
    conv2d_nchw_1[1] = 0f32
    conv2d_nchw_1[3] = 0f32
    conv2d_nchw_1[5] = 0f32
    conv2d_nchw_1[7] = 0f32
    conv2d_nchw_1[9] = 0f32
    conv2d_nchw_1[11] = 0f32
    conv2d_nchw_1[13] = 0f32
    for (rc.outer: int32, 0, 3) {
      for (ry.outer: int32, 0, 5) {
        attr [IterVar(threadIdx.z_1: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_1: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_1: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
          pad_temp.shared_1: Buffer(pad_temp.shared, float32, [112], [], scope="shared")[(threadIdx.x_1*7)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (1 <= ((blockIdx.x*56) + floordiv((threadIdx.x_1*7), 2)))), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 450)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 1)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (1 <= ((blockIdx.x*56) + floordiv(((threadIdx.x_1*7) + 1), 2)))), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 449)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 2)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 448)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 3)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 447)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 4)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 446)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 5)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 445)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 6)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 444)], 0f32, dtype=float32)
        }
        attr [IterVar(threadIdx.z_2: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_2: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_2: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
        if @tir.likely((threadIdx.x_2 < 2), dtype=bool) {
          placeholder.shared_1: Buffer(placeholder.shared, float32, [2], [], scope="shared", align=8)[threadIdx.x_2] = placeholder_1[((((blockIdx.z*150) + (threadIdx.x_2*75)) + (rc.outer*25)) + (ry.outer*5))]
        }
        conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[threadIdx.x]*placeholder.shared_1[0]))
        conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[(threadIdx.x + 16)]*placeholder.shared_1[0]))
        conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[(threadIdx.x + 32)]*placeholder.shared_1[0]))
        conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[(threadIdx.x + 48)]*placeholder.shared_1[0]))
        conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[(threadIdx.x + 64)]*placeholder.shared_1[0]))
        conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[(threadIdx.x + 80)]*placeholder.shared_1[0]))
        conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[(threadIdx.x + 96)]*placeholder.shared_1[0]))
        conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[threadIdx.x]*placeholder.shared_1[1]))
        conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[(threadIdx.x + 16)]*placeholder.shared_1[1]))
        conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[(threadIdx.x + 32)]*placeholder.shared_1[1]))
        conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[(threadIdx.x + 48)]*placeholder.shared_1[1]))
        conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[(threadIdx.x + 64)]*placeholder.shared_1[1]))
        conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[(threadIdx.x + 80)]*placeholder.shared_1[1]))
        conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[(threadIdx.x + 96)]*placeholder.shared_1[1]))
        attr [IterVar(threadIdx.z_1, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_1, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_1, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
          pad_temp.shared_1[(threadIdx.x_1*7)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (1 <= ((blockIdx.x*56) + floordiv(((threadIdx.x_1*7) + 1), 2)))), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 449)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 1)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 448)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 2)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 447)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 3)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 446)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 4)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 445)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 5)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 444)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 6)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 443)], 0f32, dtype=float32)
        }
        attr [IterVar(threadIdx.z_2, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_2, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
        if @tir.likely((threadIdx.x_2 < 2), dtype=bool) {
          placeholder.shared_1[threadIdx.x_2] = placeholder_1[(((((blockIdx.z*150) + (threadIdx.x_2*75)) + (rc.outer*25)) + (ry.outer*5)) + 1)]
        }
        conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[threadIdx.x]*placeholder.shared_1[0]))
        conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[(threadIdx.x + 16)]*placeholder.shared_1[0]))
        conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[(threadIdx.x + 32)]*placeholder.shared_1[0]))
        conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[(threadIdx.x + 48)]*placeholder.shared_1[0]))
        conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[(threadIdx.x + 64)]*placeholder.shared_1[0]))
        conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[(threadIdx.x + 80)]*placeholder.shared_1[0]))
        conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[(threadIdx.x + 96)]*placeholder.shared_1[0]))
        conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[threadIdx.x]*placeholder.shared_1[1]))
        conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[(threadIdx.x + 16)]*placeholder.shared_1[1]))
        conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[(threadIdx.x + 32)]*placeholder.shared_1[1]))
        conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[(threadIdx.x + 48)]*placeholder.shared_1[1]))
        conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[(threadIdx.x + 64)]*placeholder.shared_1[1]))
        conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[(threadIdx.x + 80)]*placeholder.shared_1[1]))
        conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[(threadIdx.x + 96)]*placeholder.shared_1[1]))
        attr [IterVar(threadIdx.z_1, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_1, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_1, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
          pad_temp.shared_1[(threadIdx.x_1*7)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 448)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 1)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 447)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 2)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 446)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 3)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 445)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 4)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 444)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 5)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 443)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 6)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 442)], 0f32, dtype=float32)
        }
        attr [IterVar(threadIdx.z_2, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_2, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
        if @tir.likely((threadIdx.x_2 < 2), dtype=bool) {
          placeholder.shared_1[threadIdx.x_2] = placeholder_1[(((((blockIdx.z*150) + (threadIdx.x_2*75)) + (rc.outer*25)) + (ry.outer*5)) + 2)]
        }
        conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[threadIdx.x]*placeholder.shared_1[0]))
        conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[(threadIdx.x + 16)]*placeholder.shared_1[0]))
        conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[(threadIdx.x + 32)]*placeholder.shared_1[0]))
        conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[(threadIdx.x + 48)]*placeholder.shared_1[0]))
        conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[(threadIdx.x + 64)]*placeholder.shared_1[0]))
        conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[(threadIdx.x + 80)]*placeholder.shared_1[0]))
        conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[(threadIdx.x + 96)]*placeholder.shared_1[0]))
        conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[threadIdx.x]*placeholder.shared_1[1]))
        conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[(threadIdx.x + 16)]*placeholder.shared_1[1]))
        conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[(threadIdx.x + 32)]*placeholder.shared_1[1]))
        conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[(threadIdx.x + 48)]*placeholder.shared_1[1]))
        conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[(threadIdx.x + 64)]*placeholder.shared_1[1]))
        conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[(threadIdx.x + 80)]*placeholder.shared_1[1]))
        conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[(threadIdx.x + 96)]*placeholder.shared_1[1]))
        attr [IterVar(threadIdx.z_1, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_1, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_1, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
          pad_temp.shared_1[(threadIdx.x_1*7)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 447)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 1)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 446)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 2)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 445)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 3)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 444)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 4)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 443)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 5)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 442)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 6)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (((blockIdx.x*56) + floordiv(((threadIdx.x_1*7) + 9), 2)) < 113)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 441)], 0f32, dtype=float32)
        }
        attr [IterVar(threadIdx.z_2, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_2, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
        if @tir.likely((threadIdx.x_2 < 2), dtype=bool) {
          placeholder.shared_1[threadIdx.x_2] = placeholder_1[(((((blockIdx.z*150) + (threadIdx.x_2*75)) + (rc.outer*25)) + (ry.outer*5)) + 3)]
        }
        conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[threadIdx.x]*placeholder.shared_1[0]))
        conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[(threadIdx.x + 16)]*placeholder.shared_1[0]))
        conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[(threadIdx.x + 32)]*placeholder.shared_1[0]))
        conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[(threadIdx.x + 48)]*placeholder.shared_1[0]))
        conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[(threadIdx.x + 64)]*placeholder.shared_1[0]))
        conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[(threadIdx.x + 80)]*placeholder.shared_1[0]))
        conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[(threadIdx.x + 96)]*placeholder.shared_1[0]))
        conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[threadIdx.x]*placeholder.shared_1[1]))
        conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[(threadIdx.x + 16)]*placeholder.shared_1[1]))
        conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[(threadIdx.x + 32)]*placeholder.shared_1[1]))
        conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[(threadIdx.x + 48)]*placeholder.shared_1[1]))
        conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[(threadIdx.x + 64)]*placeholder.shared_1[1]))
        conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[(threadIdx.x + 80)]*placeholder.shared_1[1]))
        conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[(threadIdx.x + 96)]*placeholder.shared_1[1]))
        attr [IterVar(threadIdx.z_1, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_1, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_1, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
          pad_temp.shared_1[(threadIdx.x_1*7)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 446)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 1)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 445)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 2)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 444)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 3)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 443)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 4)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 442)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 5)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (((blockIdx.x*56) + floordiv(((threadIdx.x_1*7) + 9), 2)) < 113)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 441)], 0f32, dtype=float32)
          pad_temp.shared_1[((threadIdx.x_1*7) + 6)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (((blockIdx.x*56) + floordiv((threadIdx.x_1*7), 2)) < 108)), placeholder[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 440)], 0f32, dtype=float32)
        }
        attr [IterVar(threadIdx.z_2, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_2, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
        if @tir.likely((threadIdx.x_2 < 2), dtype=bool) {
          placeholder.shared_1[threadIdx.x_2] = placeholder_1[(((((blockIdx.z*150) + (threadIdx.x_2*75)) + (rc.outer*25)) + (ry.outer*5)) + 4)]
        }
        conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[threadIdx.x]*placeholder.shared_1[0]))
        conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[(threadIdx.x + 16)]*placeholder.shared_1[0]))
        conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[(threadIdx.x + 32)]*placeholder.shared_1[0]))
        conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[(threadIdx.x + 48)]*placeholder.shared_1[0]))
        conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[(threadIdx.x + 64)]*placeholder.shared_1[0]))
        conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[(threadIdx.x + 80)]*placeholder.shared_1[0]))
        conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[(threadIdx.x + 96)]*placeholder.shared_1[0]))
        conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[threadIdx.x]*placeholder.shared_1[1]))
        conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[(threadIdx.x + 16)]*placeholder.shared_1[1]))
        conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[(threadIdx.x + 32)]*placeholder.shared_1[1]))
        conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[(threadIdx.x + 48)]*placeholder.shared_1[1]))
        conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[(threadIdx.x + 64)]*placeholder.shared_1[1]))
        conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[(threadIdx.x + 80)]*placeholder.shared_1[1]))
        conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[(threadIdx.x + 96)]*placeholder.shared_1[1]))
      }
    }
    compute_1: Buffer(compute, float32, [501760], [])[((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x)] = max(conv2d_nchw_1[0], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 16)] = max(conv2d_nchw_1[2], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 32)] = max(conv2d_nchw_1[4], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 48)] = max(conv2d_nchw_1[6], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 64)] = max(conv2d_nchw_1[8], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 80)] = max(conv2d_nchw_1[10], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 96)] = max(conv2d_nchw_1[12], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50176)] = max(conv2d_nchw_1[1], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50192)] = max(conv2d_nchw_1[3], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50208)] = max(conv2d_nchw_1[5], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50224)] = max(conv2d_nchw_1[7], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50240)] = max(conv2d_nchw_1[9], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50256)] = max(conv2d_nchw_1[11], 0f32)
    compute_1[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50272)] = max(conv2d_nchw_1[13], 0f32)
  }
}
```

## 总结

本教程已经展示了如下内容：

* 如何使用 TOPI API 操作 numpy 风格的算子。
* TOPI 如何促进上下文的通用 schedule 和算子融合，来生成优化的内核代码。

[下载 Python 源代码：intro_topi.py](https://tvm.apache.org/docs/_downloads/3a9b1d387f618487c8ccf6b8b78ae179/intro_topi.py)

[下载 Jupyter Notebook：intro_topi.ipynb](https://tvm.apache.org/docs/_downloads/63f9e50204143ea3c2d3593c72439b3d/intro_topi.ipynb)
