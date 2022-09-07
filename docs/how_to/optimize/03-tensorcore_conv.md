---
title: 如何使用 TensorCores 优化卷积
---

# 如何使用 TensorCores 优化卷积

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/optimize_operators/opt_conv_tensorcore.html#sphx-glr-download-how-to-optimize-operators-opt-conv-tensorcore-py) 下载完整的示例代码
:::

**作者**：[Siyuan Feng](https://github.com/Hzfengsy)

本教程演示如何在 TVM 中使用 TensorCores 编写高性能卷积调度。在这个例子中，会假设卷积输入的 batch 较大。强烈建议前置讲解 [如何在 GPU 上优化卷积](gpu_conv)。

## TensorCore 介绍

每个 Tensor Core 都提供了一个 4x4x4 矩阵处理数组，它使得 `D = A * B + C`，其中 A、B、C 和 D 是 4x4 矩阵，如图所示。矩阵乘法输入 A 和 B 是 FP16 矩阵，而累加矩阵 C 和 D 可以是 FP16 或 FP32 矩阵。

但是，CUDA 开发者只能使用 warp 级原语 `wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag)` 在张量核上执行 16x16x16 半精度矩阵乘法。调用矩阵乘法之前，开发者必须使用原始 `wmma::load_matrix_sync` 显式地将数据从内存加载到寄存器中。NVCC 编译器将该原语转换为多个内存加载指令。在运行时，每个线程从矩阵 A 加载 16 个元素，从 B 加载 16 个元素。

## 准备和算法

对具有 256 个通道和 14 x 14 维度的输入张量使用固定大小。batch size 为 256，卷积过滤器包含 512 个大小为 3 x 3 的过滤器，使用 stride size 为 1 和 padding size 为 1 进行卷积。在示例中，使用 NHWCnc 内存布局。以下代码定义了 TVM 中的卷积算法。

``` python
import tvm
from tvm import te
import numpy as np
from tvm.contrib import nvcc

# 输入和过滤器的大小
batch_size = 256
height = 14
width = 14
in_channels = 256
out_channels = 512
kernel_h = 3
kernel_w = 3
pad_h = 1
pad_w = 1
stride_h = 1
stride_w = 1

# TensorCore shape
block_size = 16

assert batch_size % block_size == 0
assert in_channels % block_size == 0
assert out_channels % block_size == 0

# 输入特征图：（N，H，W，IC，n，ic）
data_shape = (
    batch_size // block_size,
    height,
    width,
    in_channels // block_size,
    block_size,
    block_size,
)
# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    kernel_h,
    kernel_w,
    in_channels // block_size,
    out_channels // block_size,
    block_size,
    block_size,
)
# 输出特征图：(N, H, W, OC, n, oc)
output_shape = (
    batch_size // block_size,
    height,
    width,
    out_channels // block_size,
    block_size,
    block_size,
)

# Reduction axes
kh = te.reduce_axis((0, kernel_h), name="kh")
kw = te.reduce_axis((0, kernel_w), name="kw")
ic = te.reduce_axis((0, in_channels // block_size), name="ic")
ii = te.reduce_axis((0, block_size), name="ii")

# 算法
A = te.placeholder(data_shape, name="A", dtype="float16")
W = te.placeholder(kernel_shape, name="W", dtype="float16")
Apad = te.compute(
    (
        batch_size // block_size,
        height + 2 * pad_h,
        width + 2 * pad_w,
        in_channels // block_size,
        block_size,
        block_size,
    ),
    lambda n, h, w, i, nn, ii: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h - pad_h < height, w >= pad_w, w - pad_w < width),
        A[n, h - pad_h, w - pad_w, i, nn, ii],
        tvm.tir.const(0.0, "float16"),
    ),
    name="Apad",
)
Conv = te.compute(
    output_shape,
    lambda n, h, w, o, nn, oo: te.sum(
        Apad[n, h * stride_h + kh, w * stride_w + kw, ic, nn, ii].astype("float32")
        * W[kh, kw, ic, o, ii, oo].astype("float32"),
        axis=[ic, kh, kw, ii],
    ),
    name="Conv",
)

s = te.create_schedule(Conv.op)
s[Apad].compute_inline()
```

## 内存范围

传统的 GPU 调度有全局、共享和本地内存范围。为了支持 TensorCores，添加另外三个特殊的内存范围：`wmma.matrix_a`, `wmma.matrix_b` 和 `wmma.accumulator`。在硬件上，所有片段范围都存储在芯片上寄存器级别，与本地内存相同。

``` python
# 指定内存层次结构
AS = s.cache_read(Apad, "shared", [Conv])
WS = s.cache_read(W, "shared", [Conv])
AF = s.cache_read(AS, "wmma.matrix_a", [Conv])
WF = s.cache_read(WS, "wmma.matrix_b", [Conv])
ConvF = s.cache_write(Conv, "wmma.accumulator")
```

## 定义张量内联函数

实际上，TensorCore 是一种特殊的硬件操作。因此，可以只用 tensorize 将一个计算单元替换为 TensorCore 指令。首先需要定义张量内联函数。

TensorCore 中有四个基本操作：`fill_fragment`， `load_matrix`， `mma_sync` 和 `store_matrix`。由于 `fill_fragment` 和 `mma_sync` 都用于矩阵乘法，所以可以只写以下三个内联函数。

``` python
def intrin_wmma_load_matrix(scope):
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float16")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=256)
    C = te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                n,
                n,
                n,
                BC.elem_offset // 256,
                BA.access_ptr("r"),
                n,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})

def intrin_wmma_gemm():
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float16")
    B = te.placeholder((n, n), name="B", dtype="float16")
    k = te.reduce_axis((0, n), name="k")
    C = te.compute(
        (n, n),
        lambda ii, jj: te.sum(A[ii, k].astype("float") * B[k, jj].astype("float"), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="BA", scope="wmma.matrix_a", data_alignment=32, offset_factor=256
    )
    BB = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="BB", scope="wmma.matrix_b", data_alignment=32, offset_factor=256
    )
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="BC", scope="wmma.accumulator", data_alignment=32, offset_factor=256
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle", "tir.tvm_fill_fragment", BC.data, n, n, n, BC.elem_offset // 256, 0.0
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    BC.elem_offset // 256,
                    BA.data,
                    BA.elem_offset // 256,
                    BB.data,
                    BB.elem_offset // 256,
                    BC.data,
                    BC.elem_offset // 256,
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})

def intrin_wmma_store_matrix():
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator", data_alignment=32, offset_factor=256
    )
    C = te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="global", data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                n,
                n,
                n,
                BA.elem_offset // 256,
                BC.access_ptr("w"),
                n,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})
```

## 调度计算

要在 TVM 中使用 TensorCores，必须将计算调度到特定的结构中，从而匹配张量内联函数。和传统的 GPU 程序一样，也可以用共享内存来提升速度。如果对分块和共享内存有任何疑问，请参阅 [如何在 GPU 上优化卷积](gpu_conv)。

在这个例子中，每个块包含 2x4 个 warp，每个 warp 调用 4x2 TensorCore 指令。因此，每个 warp 的输出 shape 为 64x32，每个块输出 128x128 个 titles。由于共享内存空间的限制，一次只加载 2 个块（2x128x128 tiles）。

:::note
*Warp 级操作*

所有 TensorCore 指令都是 warp 级指令，这意味着一个 warp 中的所有 32 个线程应该同时执行此指令。

threadIdx.x extent=32 是解决此问题的最简单方法之一。除了那些直接或间接包含 TensorCore 内联函数的循环，可以将 threadIdx.x 绑定到任何循环。

这并不是唯一的解决方案，唯一应该做的就是确保一个 warp 中的所有线程都可以同时调用 TensorCore。
:::

``` python
# 定义 tile 大小
block_row_warps = 4
block_col_warps = 2
warp_row_tiles = 2
warp_col_tiles = 4
warp_size = 32
chunk = 2

block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")
thread_z = te.thread_axis("threadIdx.z")

nc, hc, wc, oc, nnc, ooc = Conv.op.axis
block_k = s[Conv].fuse(hc, wc)
s[Conv].bind(block_k, block_z)
nc, nci = s[Conv].split(nc, factor=warp_row_tiles)
block_i, nc = s[Conv].split(nc, factor=block_row_warps)
oc, oci = s[Conv].split(oc, factor=warp_col_tiles)
block_j, oc = s[Conv].split(oc, factor=block_col_warps)
s[Conv].reorder(block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
s[Conv].bind(block_i, block_x)
s[Conv].bind(block_j, block_y)
s[Conv].bind(nc, thread_y)
s[Conv].bind(oc, thread_z)

# 调度本地计算
s[ConvF].compute_at(s[Conv], oc)
n, h, w, o, nnf, oof = ConvF.op.axis
ko, ki = s[ConvF].split(ic, factor=chunk)
s[ConvF].reorder(ko, kh, ki, kw, n, o, nnf, oof, ii)

# 将中间计算移动到每个输出计算块中
s[AF].compute_at(s[ConvF], kw)
s[WF].compute_at(s[ConvF], kw)

# A 的共享内存调度
s[AS].compute_at(s[ConvF], kh)
n, h, w, i, nn, ii = AS.op.axis
tx, xo = s[AS].split(n, nparts=block_row_warps)
ty, yo = s[AS].split(xo, nparts=block_col_warps)
t = s[AS].fuse(nn, ii)
to, ti = s[AS].split(t, factor=warp_size)
s[AS].bind(tx, thread_y)
s[AS].bind(ty, thread_z)
s[AS].bind(ti, thread_x)

# W 的共享内存调度
s[WS].compute_at(s[ConvF], kh)
kh, kw, ic, o, ii, oo = WS.op.axis
tx, xo = s[WS].split(o, nparts=block_row_warps)
ty, yo = s[WS].split(xo, nparts=block_col_warps)
t = s[WS].fuse(ii, oo)
to, ti = s[WS].split(t, nparts=warp_size)
s[WS].bind(tx, thread_y)
s[WS].bind(ty, thread_z)
s[WS].bind(to, thread_x)
s[WS].vectorize(ti)
print(tvm.lower(s, [A, W, Conv], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, W_1: handle, Conv_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [12845056], []),
             W: Buffer(W_2: Pointer(float16), float16, [1179648], []),
             Conv: Buffer(Conv_2: Pointer(float32), float32, [25690112], [])}
  buffer_map = {A_1: A, W_1: W, Conv_1: Conv}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [16, 14, 14, 16, 16, 16], []), W_1: W_3: Buffer(W_2, float16, [3, 3, 16, 32, 16, 16], []), Conv_1: Conv_3: Buffer(Conv_2, float32, [16, 14, 14, 32, 16, 16], [])} {
  attr [IterVar(blockIdx.z: int32, (nullptr), "ThreadIndex", "blockIdx.z")] "thread_extent" = 196;
  allocate(Conv.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [2048]), storage_scope = wmma.accumulator;
  allocate(Apad.shared: Pointer(shared float16), float16, [12288]), storage_scope = shared;
  allocate(W.shared: Pointer(shared float16), float16, [12288]), storage_scope = shared;
  allocate(Apad.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [512]), storage_scope = wmma.matrix_a;
  allocate(W.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [1024]), storage_scope = wmma.matrix_b;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 2;
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 4;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 4;
  attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2 {
    for (n.c.init: int32, 0, 2) {
      for (o.c.init: int32, 0, 4) {
        for (nn.c.init: int32, 0, 16) {
          for (oo.c.init: int32, 0, 16) {
            Conv.wmma.accumulator_1: Buffer(Conv.wmma.accumulator, float32, [2048], [], scope="wmma.accumulator")[((((n.c.init*1024) + (o.c.init*256)) + (nn.c.init*16)) + oo.c.init)] = 0f32
          }
        }
      }
    }
    for (ic.outer: int32, 0, 8) {
      for (kh: int32, 0, 3) {
        for (ax2: int32, 0, 3) {
          for (ax3: int32, 0, 2) {
            for (ax4.ax5.fused.outer: int32, 0, 8) {
              let cse_var_2: int32 = (ax3*256)
              let cse_var_1: int32 = (ax4.ax5.fused.outer*32)
              attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
              Apad.shared_1: Buffer(Apad.shared, float16, [12288], [], scope="shared")[((((((threadIdx.y*3072) + (threadIdx.z*1536)) + (ax2*512)) + cse_var_2) + cse_var_1) + threadIdx.x)] = @tir.if_then_else(((((1 <= (floordiv(blockIdx.z, 14) + kh)) && ((floordiv(blockIdx.z, 14) + kh) < 15)) && (1 <= (ax2 + floormod(blockIdx.z, 14)))) && ((ax2 + floormod(blockIdx.z, 14)) < 15)), A[(((((((((((blockIdx.x*6422528) + (threadIdx.y*1605632)) + (threadIdx.z*802816)) + (kh*57344)) + (blockIdx.z*4096)) + (ax2*4096)) + (ic.outer*512)) + cse_var_2) + cse_var_1) + threadIdx.x) - 61440)], 0f16, dtype=float16)
            }
          }
        }
        for (ax1: int32, 0, 3) {
          for (ax2_1: int32, 0, 2) {
            attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
            W.shared_1: Buffer(W.shared, float16, [12288], [], scope="shared")[ramp((((((ax1*4096) + (ax2_1*2048)) + (threadIdx.y*512)) + (threadIdx.z*256)) + (threadIdx.x*8)), 1, 8)] = W[ramp(((((((((kh*393216) + (ax1*131072)) + (ic.outer*16384)) + (ax2_1*8192)) + (blockIdx.y*2048)) + (threadIdx.y*512)) + (threadIdx.z*256)) + (threadIdx.x*8)), 1, 8)]
          }
        }
        for (ic.inner: int32, 0, 2) {
          for (kw: int32, 0, 3) {
            for (ax0: int32, 0, 2) {
              for (ax4: int32, 0, 16) {
                for (ax5: int32, 0, 16) {
                  let cse_var_3: int32 = (ax4*16)
                  Apad.shared.wmma.matrix_a_1: Buffer(Apad.shared.wmma.matrix_a, float16, [512], [], scope="wmma.matrix_a")[(((ax0*256) + cse_var_3) + ax5)] = Apad.shared_1[((((((threadIdx.y*3072) + (ax0*1536)) + (kw*512)) + (ic.inner*256)) + cse_var_3) + ax5)]
                }
              }
            }
            for (ax3_1: int32, 0, 4) {
              for (ax4_1: int32, 0, 16) {
                for (ax5_1: int32, 0, 16) {
                  let cse_var_5: int32 = (ax3_1*256)
                  let cse_var_4: int32 = (ax4_1*16)
                  W.shared.wmma.matrix_b_1: Buffer(W.shared.wmma.matrix_b, float16, [1024], [], scope="wmma.matrix_b")[((cse_var_5 + cse_var_4) + ax5_1)] = W.shared_1[((((((kw*4096) + (ic.inner*2048)) + (threadIdx.z*1024)) + cse_var_5) + cse_var_4) + ax5_1)]
                }
              }
            }
            for (n.c: int32, 0, 2) {
              for (o.c: int32, 0, 4) {
                for (nn.c: int32, 0, 16) {
                  for (oo.c: int32, 0, 16) {
                    for (ii: int32, 0, 16) {
                      let cse_var_8: int32 = (o.c*256)
                      let cse_var_7: int32 = (nn.c*16)
                      let cse_var_6: int32 = ((((n.c*1024) + cse_var_8) + cse_var_7) + oo.c)
                      Conv.wmma.accumulator_1[cse_var_6] = (Conv.wmma.accumulator_1[cse_var_6] + (cast(float32, Apad.shared.wmma.matrix_a_1[(((n.c*256) + cse_var_7) + ii)])*cast(float32, W.shared.wmma.matrix_b_1[((cse_var_8 + (ii*16)) + oo.c)])))
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    for (n.inner: int32, 0, 2) {
      for (o.inner: int32, 0, 4) {
        for (nn: int32, 0, 16) {
          for (oo: int32, 0, 16) {
            let cse_var_10: int32 = (o.inner*256)
            let cse_var_9: int32 = (nn*16)
            Conv[(((((((((blockIdx.x*12845056) + (threadIdx.y*3211264)) + (n.inner*1605632)) + (blockIdx.z*8192)) + (blockIdx.y*2048)) + (threadIdx.z*1024)) + cse_var_10) + cse_var_9) + oo)] = Conv.wmma.accumulator_1[((((n.inner*1024) + cse_var_10) + cse_var_9) + oo)]
          }
        }
      }
    }
  }
}
```

## 将计算降级为内联函数

最后一个阶段将计算循环降级到 TensorCore 硬件内联函数，这是通过将 2D 卷积映射到张量内联函数实现的。

``` python
s[AF].tensorize(AF.op.axis[-2], intrin_wmma_load_matrix("wmma.matrix_a"))
s[WF].tensorize(WF.op.axis[-2], intrin_wmma_load_matrix("wmma.matrix_b"))
s[Conv].tensorize(nnc, intrin_wmma_store_matrix())
s[ConvF].tensorize(nnf, intrin_wmma_gemm())
print(tvm.lower(s, [A, W, Conv], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, W_1: handle, Conv_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [12845056], []),
             W: Buffer(W_2: Pointer(float16), float16, [1179648], []),
             Conv: Buffer(Conv_2: Pointer(float32), float32, [25690112], [])}
  buffer_map = {A_1: A, W_1: W, Conv_1: Conv}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [16, 14, 14, 16, 16, 16], []), W_1: W_3: Buffer(W_2, float16, [3, 3, 16, 32, 16, 16], []), Conv_1: Conv_3: Buffer(Conv_2, float32, [16, 14, 14, 32, 16, 16], [])} {
  attr [IterVar(blockIdx.z: int32, (nullptr), "ThreadIndex", "blockIdx.z")] "thread_extent" = 196;
  allocate(Conv.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [2048]), storage_scope = wmma.accumulator;
  allocate(Apad.shared: Pointer(shared float16), float16, [12288]), storage_scope = shared;
  allocate(W.shared: Pointer(shared float16), float16, [12288]), storage_scope = shared;
  allocate(Apad.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [512]), storage_scope = wmma.matrix_a;
  allocate(W.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [1024]), storage_scope = wmma.matrix_b;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 2;
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 4;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 4;
  attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2 {
    for (n.c.init: int32, 0, 2) {
      for (o.c.init: int32, 0, 4) {
        @tir.tvm_fill_fragment(Conv.wmma.accumulator, 16, 16, 16, ((n.c.init*4) + o.c.init), 0f32, dtype=handle)
      }
    }
    for (ic.outer: int32, 0, 8) {
      for (kh: int32, 0, 3) {
        for (ax2: int32, 0, 3) {
          for (ax3: int32, 0, 2) {
            for (ax4.ax5.fused.outer: int32, 0, 8) {
              let cse_var_2: int32 = (ax3*256)
              let cse_var_1: int32 = (ax4.ax5.fused.outer*32)
              attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
              Apad.shared_1: Buffer(Apad.shared, float16, [12288], [], scope="shared")[((((((threadIdx.y*3072) + (threadIdx.z*1536)) + (ax2*512)) + cse_var_2) + cse_var_1) + threadIdx.x)] = @tir.if_then_else(((((1 <= (floordiv(blockIdx.z, 14) + kh)) && ((floordiv(blockIdx.z, 14) + kh) < 15)) && (1 <= (ax2 + floormod(blockIdx.z, 14)))) && ((ax2 + floormod(blockIdx.z, 14)) < 15)), A[(((((((((((blockIdx.x*6422528) + (threadIdx.y*1605632)) + (threadIdx.z*802816)) + (kh*57344)) + (blockIdx.z*4096)) + (ax2*4096)) + (ic.outer*512)) + cse_var_2) + cse_var_1) + threadIdx.x) - 61440)], 0f16, dtype=float16)
            }
          }
        }
        for (ax1: int32, 0, 3) {
          for (ax2_1: int32, 0, 2) {
            attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
            W.shared_1: Buffer(W.shared, float16, [12288], [], scope="shared")[ramp((((((ax1*4096) + (ax2_1*2048)) + (threadIdx.y*512)) + (threadIdx.z*256)) + (threadIdx.x*8)), 1, 8)] = W[ramp(((((((((kh*393216) + (ax1*131072)) + (ic.outer*16384)) + (ax2_1*8192)) + (blockIdx.y*2048)) + (threadIdx.y*512)) + (threadIdx.z*256)) + (threadIdx.x*8)), 1, 8)]
          }
        }
        for (ic.inner: int32, 0, 2) {
          for (kw: int32, 0, 3) {
            for (ax0: int32, 0, 2) {
              @tir.tvm_load_matrix_sync(Apad.shared.wmma.matrix_a, 16, 16, 16, ax0, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), Apad.shared, ((((threadIdx.y*3072) + (ax0*1536)) + (kw*512)) + (ic.inner*256)), 256, 1, dtype=handle), 16, "row_major", dtype=handle)
            }
            for (ax3_1: int32, 0, 4) {
              @tir.tvm_load_matrix_sync(W.shared.wmma.matrix_b, 16, 16, 16, ax3_1, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), W.shared, ((((kw*4096) + (ic.inner*2048)) + (threadIdx.z*1024)) + (ax3_1*256)), 256, 1, dtype=handle), 16, "row_major", dtype=handle)
            }
            for (n.c: int32, 0, 2) {
              for (o.c: int32, 0, 4) {
                let cse_var_3: int32 = ((n.c*4) + o.c)
                @tir.tvm_mma_sync(Conv.wmma.accumulator, cse_var_3, Apad.shared.wmma.matrix_a, n.c, W.shared.wmma.matrix_b, o.c, Conv.wmma.accumulator, cse_var_3, dtype=handle)
              }
            }
          }
        }
      }
    }
    for (n.inner: int32, 0, 2) {
      for (o.inner: int32, 0, 4) {
        @tir.tvm_store_matrix_sync(Conv.wmma.accumulator, 16, 16, 16, ((n.inner*4) + o.inner), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), Conv_2, (((((((blockIdx.x*12845056) + (threadIdx.y*3211264)) + (n.inner*1605632)) + (blockIdx.z*8192)) + (blockIdx.y*2048)) + (threadIdx.z*1024)) + (o.inner*256)), 256, 2, dtype=handle), 16, "row_major", dtype=handle)
      }
    }
  }
}
```

## 生成 CUDA 内核

最后使用 TVM 生成和编译 CUDA 内核，并评估卷积的延迟。由于 TensorCores 仅支持 Compute Capability 7.0 或更高版本的 NVIDIA GPU，因此可能无法在构建服务器上运行。

``` bash
dev = tvm.cuda(0)
if nvcc.have_tensorcore(dev.compute_version):
    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
        func = tvm.build(s, [A, W, Conv], "cuda")
    a_np = np.random.uniform(size=data_shape).astype(A.dtype)
    w_np = np.random.uniform(size=kernel_shape).astype(W.dtype)
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    c = tvm.nd.array(np.zeros(output_shape, dtype=Conv.dtype), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print("conv2d with tensor core: %f ms" % (evaluator(a, w, c).mean * 1e3))
```

输出结果：

``` bash
conv2d with tensor core: 6.835711 ms
```

## 总结

本教程演示如何用 TVM 调度原语在特定 GPU 上调用 TensorCore。

[下载 Python 源代码：opt_conv_tensorcore.py](https://tvm.apache.org/docs/_downloads/7372db5919b5619bc34fde3434862bca/opt_conv_tensorcore.py)

[下载 Jupyter Notebook：opt_conv_tensorcore.ipynb](https://tvm.apache.org/docs/_downloads/7455981870c23c8c76482dedf33d8a42/opt_conv_tensorcore.ipynb)