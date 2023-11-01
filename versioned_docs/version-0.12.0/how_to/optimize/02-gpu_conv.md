---
title: 如何在 GPU 上优化卷积
---

# 如何在 GPU 上优化卷积

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/optimize_operators/opt_conv_cuda.html#sphx-glr-download-how-to-optimize-operators-opt-conv-cuda-py) 下载完整的示例代码
:::

**作者**：[Haichen Shen](https://homes.cs.washington.edu/\~haichen/)

本教程演示了如何在 TVM 中编写高性能卷积实现。以正方形大小的输入张量和滤波器为例，假设卷积输入的 batch 较大。在此示例中，使用不同的布局来存储数据，以实现更好的数据局部性。缓冲区布局是 HWCN，分别代表高度、宽度、通道、batch。

## 准备和算法

对具有 256 个通道和 14 x 14 维度的输入张量使用固定大小。batch size 为 256，卷积过滤器包含 512 个大小为 3 x 3 的过滤器，用步长为 1 和 padding size 为 1 进行卷积。以下代码定义了 TVM 中的卷积算法。

``` python
import numpy as np
import tvm
from tvm import te

# 输入和过滤器的大小
batch = 256
in_channel = 256
out_channel = 512
in_size = 14
kernel = 3
pad = 1
stride = 1

# 算法
A = te.placeholder((in_size, in_size, in_channel, batch), name="A")
W = te.placeholder((kernel, kernel, in_channel, out_channel), name="W")
out_size = (in_size - kernel + 2 * pad) // stride + 1
# Pad 输入
Apad = te.compute(
    (in_size + 2 * pad, in_size + 2 * pad, in_channel, batch),
    lambda yy, xx, cc, nn: tvm.tir.if_then_else(
        tvm.tir.all(yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size),
        A[yy - pad, xx - pad, cc, nn],
        tvm.tir.const(0.0, "float32"),
    ),
    name="Apad",
)
# 创建归约变量
rc = te.reduce_axis((0, in_channel), name="rc")
ry = te.reduce_axis((0, kernel), name="ry")
rx = te.reduce_axis((0, kernel), name="rx")
# 计算卷积
B = te.compute(
    (out_size, out_size, out_channel, batch),
    lambda yy, xx, ff, nn: te.sum(
        Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff], axis=[ry, rx, rc]
    ),
    name="B",
)
```

## 内存层次结构

首先指定缓冲区的内存层次结构。下图显示了 GPU 内存层次结构，与 CPU 内存层次结构的重要区别是 GPU 提供了一个称为共享内存的缓存缓冲区，由程序员管理。因此，如何最大化共享内存中的数据重用对于在 GPU 内核中实现高性能至关重要。

![图片](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/gpu_memory_hierarchy.png)

在本例中，将 Apad 和 W 加载到缓冲区 AA 和 WW 中（存储在共享内存中）。这些缓冲区稍后将由同一线程块中的所有线程共享以计算卷积，然后每个线程将自己的部分从共享缓冲区加载到它们的本地寄存器 AL 和 WL 中。BL 是输出 B 的本地缓存，也存储在线程本地寄存器中。

``` python
# 指定内存层次结构
s = te.create_schedule(B.op)
s[Apad].compute_inline()  # compute Apad inline
AA = s.cache_read(Apad, "shared", [B])
WW = s.cache_read(W, "shared", [B])
AL = s.cache_read(AA, "local", [B])
WL = s.cache_read(WW, "local", [B])
BL = s.cache_write(B, "local")
```

## 分块

以下代码将工作负载拆分为线程块和单独的线程，遵循矩阵乘法中的分块方案。如下图所示，给定一个像素坐标（y、x），一个线程块负责计算一个 block_factor x block_factor (64 x 64) 的区域，用于输出通道和 batch。由于共享内存空间的限制，每次只从 Apad 和 B 加载 step x block_factor (8 x 64) 数据到共享内存中的缓冲区。

![图片](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/conv_gpu_blocking.png)

``` python
# 平铺常量
tile = 8
num_thread = 8
block_factor = tile * num_thread
step = 8
vthread = 2

# 获取 GPU 线程索引
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")

# split 工作负载
hi, wi, fi, ni = s[B].op.axis
bz = s[B].fuse(hi, wi)
by, fi = s[B].split(fi, factor=block_factor)
bx, ni = s[B].split(ni, factor=block_factor)

# 将迭代变量绑定到 GPU 线程索引
s[B].bind(bz, block_z)
s[B].bind(by, block_y)
s[B].bind(bx, block_x)
```

## 虚拟线程分割

进一步将工作负载从线程块拆分为单个线程。为了避免 *memory bank conflict*，使用虚拟线程将区域分成 4 个部分，然后平铺成 8x8 的网格。因此，如下图所示，每个线程计算 4 个跨步网格，每个网格的大小为 4 x 4。

![图片](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/conv_gpu_vthread.png)

``` python
tyz, fi = s[B].split(fi, nparts=vthread)  # 虚拟线程 split
txz, ni = s[B].split(ni, nparts=vthread)  # 虚拟线程 split
ty, fi = s[B].split(fi, nparts=num_thread)
tx, ni = s[B].split(ni, nparts=num_thread)
s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

s[B].bind(tyz, thread_yz)
s[B].bind(txz, thread_xz)
s[B].bind(ty, thread_y)
s[B].bind(tx, thread_x)
```

## 协同获取（Cooperative Fetching）

如前所述，每个时间步长都要将 step x block_factor 数据从 GPU 全局内存传输到共享内存。为了减少每个线程的内存传输，以下代码让同一线程块中的线程协同从全局内存中获取相关数据。

``` python
# Schedule BL 本地写入
s[BL].compute_at(s[B], tx)
yi, xi, fi, ni = s[BL].op.axis
ry, rx, rc = s[BL].op.reduce_axis
rco, rci = s[BL].split(rc, factor=step)
s[BL].reorder(rco, ry, rx, rci, fi, ni)

# 将计算附加到迭代变量
s[AA].compute_at(s[BL], rx)
s[WW].compute_at(s[BL], rx)
s[AL].compute_at(s[BL], rci)
s[WL].compute_at(s[BL], rci)

# A 的共享内存负载调度
yi, xi, ci, ni = s[AA].op.axis
ty, ci = s[AA].split(ci, nparts=num_thread)
tx, ni = s[AA].split(ni, nparts=num_thread)
_, ni = s[AA].split(ni, factor=4)
s[AA].reorder(ty, tx, yi, xi, ci, ni)
s[AA].bind(ty, thread_y)
s[AA].bind(tx, thread_x)
s[AA].vectorize(ni)  # 向量化内存加载

# W 的共享内存负载调度
yi, xi, ci, fi = s[WW].op.axis
ty, ci = s[WW].split(ci, nparts=num_thread)
tx, fi = s[WW].split(fi, nparts=num_thread)
_, fi = s[WW].split(fi, factor=4)
s[WW].reorder(ty, tx, yi, xi, ci, fi)
s[WW].bind(ty, thread_y)
s[WW].bind(tx, thread_x)
s[WW].vectorize(fi)  # 向量化内存加载
```

## 生成 CUDA 内核

最后用 TVM 生成和编译 CUDA 内核，并评估卷积的延迟。

``` python
func = tvm.build(s, [A, W, B], "cuda")
dev = tvm.cuda(0)
a_np = np.random.uniform(size=(in_size, in_size, in_channel, batch)).astype(A.dtype)
w_np = np.random.uniform(size=(kernel, kernel, in_channel, out_channel)).astype(W.dtype)
a = tvm.nd.array(a_np, dev)
w = tvm.nd.array(w_np, dev)
b = tvm.nd.array(np.zeros((out_size, out_size, out_channel, batch), dtype=B.dtype), dev)
func(a, w, b)
evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("Convolution: %f ms" % (evaluator(a, w, b).mean * 1e3))
```

输出结果：

``` bash
Convolution: 54.146944 ms
```

[下载 Python 源代码：opt_conv_cuda.py](https://tvm.apache.org/docs/_downloads/3c5c85c3954f3110f16ca084e286f03a/opt_conv_cuda.py)

[下载 Jupyter notebook：opt_conv_cuda.ipynb](https://tvm.apache.org/docs/_downloads/854257a66df713b1f3f82eb3577f95e3/opt_conv_cuda.ipynb)
