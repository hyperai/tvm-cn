---
title: 使用 Tensorize 来利用硬件内联函数
---

# 使用 Tensorize 来利用硬件内联函数

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_schedules/tensorize.html#sphx-glr-download-how-to-work-with-schedules-tensorize-py) 下载完整的示例代码
:::

**作者**：[Yizhi Liu](https://github.com/yzhliu)

本文介绍如何在 TVM 中进行张量化。

通过使用调度原语 `tensorize`，可以用相应的内联函数替换一个计算单元，从而可以利用手写的 micro-kernels，以及扩展 TVM 支持新的硬件架构。

本教程的目的是展示 tensorize 的功能和用法，而非提供有效的解决方案。

``` python
from __future__ import absolute_import, print_function

import tvm
from tvm import te
import tvm.testing
import numpy as np
```

## 定义矩阵乘法

以矩阵乘法为例，Matmul 首先将两个矩阵之间的对应元素相乘，然后在某个轴上累加。以下代码描述了 TVM 中的计算 `A * B^T`。

``` python
N, M, L = 1024, 512, 64
A = te.placeholder((N, L), name="A")
B = te.placeholder((M, L), name="B")
k = te.reduce_axis((0, L), name="k")
C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k), name="C")
s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [65536], []),
             B: Buffer(B_2: Pointer(float32), float32, [32768], []),
             C: Buffer(C_2: Pointer(float32), float32, [524288], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 64], []), B_1: B_3: Buffer(B_2, float32, [512, 64], []), C_1: C_3: Buffer(C_2, float32, [1024, 512], [])} {
  for (i: int32, 0, 1024) {
    for (j: int32, 0, 512) {
      C[((i*512) + j)] = 0f32
      for (k: int32, 0, 64) {
        let cse_var_1: int32 = ((i*512) + j)
        C[cse_var_1] = (C[cse_var_1] + (A[((i*64) + k)]*B[((j*64) + k)]))
      }
    }
  }
}
```

## 调度 Matmul

假设有一个支持矩阵向量乘法 (GEMV) 作为硬件原语的加速器，它可以采用任意大小的 reduce 轴，但另一个轴需要不大于 16。我们需要分解 matmul 循环，使最里面的循环是 (16x64) GEMV。

``` python
factor = 16
x, y = C.op.axis
(z,) = C.op.reduce_axis
yo, yi = s[C].split(y, factor=factor)
s[C].reorder(x, yo, yi, z)
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

```plain
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [65536], []),
             B: Buffer(B_2: Pointer(float32), float32, [32768], []),
             C: Buffer(C_2: Pointer(float32), float32, [524288], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 64], []), B_1: B_3: Buffer(B_2, float32, [512, 64], []), C_1: C_3: Buffer(C_2, float32, [1024, 512], [])} {
  for (i: int32, 0, 1024) {
    for (j.outer: int32, 0, 32) {
      for (j.inner: int32, 0, 16) {
        C[(((i*512) + (j.outer*16)) + j.inner)] = 0f32
        for (k: int32, 0, 64) {
          let cse_var_1: int32 = (((i*512) + (j.outer*16)) + j.inner)
          C[cse_var_1] = (C[cse_var_1] + (A[((i*64) + k)]*B[(((j.outer*1024) + (j.inner*64)) + k)]))
        }
      }
    }
  }
}
```

如上面打印的 IR 所示，内部循环 `j.inner` 与 `k` 共同构成 GEMV 的计算——在最里面的两个循环中，索引 `i` 是固定的，对矩阵 `A` 的访问只取决于 `k`，这使得 `A` 的访问模式是一个「向量」。可以张量化 `j.inner`，从而利用假定硬件的 GEMV 指令。

## 定义 GEMV Tensorization 内联函数

调度张量之前，首先定义 GEMV 的内联函数。它包括两部分：第一部分是 GEMV 的计算定义， TVM 使用它来匹配原始 Matmul schedule 中的计算模式；二是指定如何在设备上执行 GEMV，在下面的 `intrin_func` 中完成。

``` python
def intrin_gemv(m, l):
    a = te.placeholder((l,), name="a")
    b = te.placeholder((m, l), name="b")
    k = te.reduce_axis((0, l), name="k")
    c = te.compute((m,), lambda i: te.sum(a[k] * b[i, k], axis=k), name="c")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[te.var("s1"), 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "gemv_update",
                cc.access_ptr("w"),
                aa.access_ptr("r"),
                bb.access_ptr("r"),
                m,
                l,
                bb.strides[0],
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})
```

这里 `te.decl_tensor_intrin` 声明了如何执行计算 `c.op`。我们的实现只是接收输入和输出，将它们转换为指针，并提供外部函数调用。

注意，tensorization 需要用户指定 `offset_factor`，TVM 通过这个信息知道数据在原始数据结构的起始地址与传给 tensorize 的偏移量之间是否对齐，因此它有机会通过向量化加载进行优化。简单起见，将因子设置为 1。

为输入和输出声明 buffer，这并不是必需的，但如此一来，我们就能受益于 buffer 提供的额外信息了。例如，将 `bb.strides[0]` 作为参数，传给外部函数 `gemv_update`。现在 `bb.strides[0] == l`，稍后将看到它们与更复杂的 schedules 有何不同。

注意，将 `te.var("s1")` 作为 `B` 的第一个步长维度。若可以推断步长——在这种情况下，TVM 知道张量 B 是紧凑的，因此步长是 `[L, 1]`——这样的占位符可以让 TVM 自动绑定推断值。

``` python
gemv = intrin_gemv(factor, L)
s[C].tensorize(yi, gemv)
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [65536], []),
             B: Buffer(B_2: Pointer(float32), float32, [32768], []),
             C: Buffer(C_2: Pointer(float32), float32, [524288], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 64], []), B_1: B_3: Buffer(B_2, float32, [512, 64], []), C_1: C_3: Buffer(C_2, float32, [1024, 512], [])} {
  for (i: int32, 0, 1024) {
    for (j.outer: int32, 0, 32) {
      @tir.call_extern("gemv_update", @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), C_2, ((i*512) + (j.outer*16)), 16, 2, dtype=handle), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), A_2, (i*64), 64, 1, dtype=handle), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), B_2, (j.outer*1024), 1024, 1, dtype=handle), 16, 64, 64, dtype=int32)
    }
  }
}
```

通过张量化 `yi`，最里面的两个循环现在被之前定义的内联函数所取代。为了构建和运行模块，定义外部函数 `gemv_update`（GEMV 的简单实现，仅用于演示）。

``` python
def gemv_impl():
    cc_code = """
      extern "C" int gemv_update(float *cc, float *aa, float *bb, int m, int l, int stride) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < l; ++j) {
                cc[i] += aa[j] * bb[i * stride + j];
            }
        }
        return 0;
      }
    """
    from tvm.contrib import utils, clang

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # 从 C 源代码创建 LLVM ir
    ll_code = clang.create_llvm(cc_code, output=ll_path)
    return ll_code
```

执行张量化 GEMV 前，利用编译指示属性 `import_llvm` 导入 llvm 内联 asm。

``` python
s[C].pragma(x, "import_llvm", gemv_impl())
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [65536], []),
             B: Buffer(B_2: Pointer(float32), float32, [32768], []),
             C: Buffer(C_2: Pointer(float32), float32, [524288], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 64], []), B_1: B_3: Buffer(B_2, float32, [512, 64], []), C_1: C_3: Buffer(C_2, float32, [1024, 512], [])} {
  attr [IterVar(i: int32, (nullptr), "DataPar", "")] "pragma_import_llvm" = "; ModuleID = '/tmp/tmpnmkhyqx0/input0.cc'\nsource_filename = \"/tmp/tmpnmkhyqx0/input0.cc\"\ntarget datalayout = \"e-m:e-i64:64-f80:128-n8:16:32:64-S128\"\ntarget triple = \"x86_64-pc-linux-gnu\"\n\n; Function Attrs: noinline nounwind optnone uwtable\ndefine dso_local i32 @gemv_update(float*, float*, float*, i32, i32, i32) #0 {\n  %7 = alloca float*, align 8\n  %8 = alloca float*, align 8\n  %9 = alloca float*, align 8\n  %10 = alloca i32, align 4\n  %11 = alloca i32, align 4\n  %12 = alloca i32, align 4\n  %13 = alloca i32, align 4\n  %14 = alloca i32, align 4\n  store float* %0, float** %7, align 8\n  store float* %1, float** %8, align 8\n  store float* %2, float** %9, align 8\n  store i32 %3, i32* %10, align 4\n  store i32 %4, i32* %11, align 4\n  store i32 %5, i32* %12, align 4\n  store i32 0, i32* %13, align 4\n  br label %15\n\n15:                                               ; preds = %50, %6\n  %16 = load i32, i32* %13, align 4\n  %17 = load i32, i32* %10, align 4\n  %18 = icmp slt i32 %16, %17\n  br i1 %18, label %19, label %53\n\n19:                                               ; preds = %15\n  store i32 0, i32* %14, align 4\n  br label %20\n\n20:                                               ; preds = %46, %19\n  %21 = load i32, i32* %14, align 4\n  %22 = load i32, i32* %11, align 4\n  %23 = icmp slt i32 %21, %22\n  br i1 %23, label %24, label %49\n\n24:                                               ; preds = %20\n  %25 = load float*, float** %8, align 8\n  %26 = load i32, i32* %14, align 4\n  %27 = sext i32 %26 to i64\n  %28 = getelementptr inbounds float, float* %25, i64 %27\n  %29 = load float, float* %28, align 4\n  %30 = load float*, float** %9, align 8\n  %31 = load i32, i32* %13, align 4\n  %32 = load i32, i32* %12, align 4\n  %33 = mul nsw i32 %31, %32\n  %34 = load i32, i32* %14, align 4\n  %35 = add nsw i32 %33, %34\n  %36 = sext i32 %35 to i64\n  %37 = getelementptr inbounds float, float* %30, i64 %36\n  %38 = load float, float* %37, align 4\n  %39 = fmul float %29, %38\n  %40 = load float*, float** %7, align 8\n  %41 = load i32, i32* %13, align 4\n  %42 = sext i32 %41 to i64\n  %43 = getelementptr inbounds float, float* %40, i64 %42\n  %44 = load float, float* %43, align 4\n  %45 = fadd float %44, %39\n  store float %45, float* %43, align 4\n  br label %46\n\n46:                                               ; preds = %24\n  %47 = load i32, i32* %14, align 4\n  %48 = add nsw i32 %47, 1\n  store i32 %48, i32* %14, align 4\n  br label %20\n\n49:                                               ; preds = %20\n  br label %50\n\n50:                                               ; preds = %49\n  %51 = load i32, i32* %13, align 4\n  %52 = add nsw i32 %51, 1\n  store i32 %52, i32* %13, align 4\n  br label %15\n\n53:                                               ; preds = %15\n  ret i32 0\n}\n\nattributes #0 = { noinline nounwind optnone uwtable \"correctly-rounded-divide-sqrt-fp-math\"=\"false\" \"disable-tail-calls\"=\"false\" \"less-precise-fpmad\"=\"false\" \"min-legal-vector-width\"=\"0\" \"no-frame-pointer-elim\"=\"true\" \"no-frame-pointer-elim-non-leaf\" \"no-infs-fp-math\"=\"false\" \"no-jump-tables\"=\"false\" \"no-nans-fp-math\"=\"false\" \"no-signed-zeros-fp-math\"=\"false\" \"no-trapping-math\"=\"false\" \"stack-protector-buffer-size\"=\"8\" \"target-cpu\"=\"x86-64\" \"target-features\"=\"+cx8,+fxsr,+mmx,+sse,+sse2,+x87\" \"unsafe-fp-math\"=\"false\" \"use-soft-float\"=\"false\" }\n\n!llvm.module.flags = !{!0}\n!llvm.ident = !{!1}\n\n!0 = !{i32 1, !\"wchar_size\", i32 4}\n!1 = !{!\"clang version 9.0.0-2~ubuntu18.04.2 (tags/RELEASE_900/final)\"}\n";
  for (i, 0, 1024) {
    for (j.outer: int32, 0, 32) {
      @tir.call_extern("gemv_update", @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), C_2, ((i*512) + (j.outer*16)), 16, 2, dtype=handle), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), A_2, (i*64), 64, 1, dtype=handle), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), B_2, (j.outer*1024), 1024, 1, dtype=handle), 16, 64, 64, dtype=int32)
    }
  }
}
```

最后，将张量化的版本与 `numpy.dot` 生成的版本进行比较，确保实现是正确的。

``` python
func = tvm.build(s, [A, B, C], target="llvm", name="gemv")

from tvm.topi.utils import get_const_tuple

dtype = A.dtype
dev = tvm.device("cpu", 0)
a = np.random.uniform(size=get_const_tuple(A.shape)).astype(dtype)
b = np.random.uniform(size=get_const_tuple(B.shape)).astype(dtype)
c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=dtype), dev)
func(tvm.nd.array(a, dev), tvm.nd.array(b, dev), c)
tvm.testing.assert_allclose(c.numpy(), np.dot(a, b.T), rtol=1e-3)
```

## 更新 tensorize 的 reduce

前面已经了解了 tensorize 的基本概念，接下来看一个更复杂的案例。

假设加速器只能让一个向量和一个矩阵相乘，其中向量大小不大于 16。鉴于这样的硬件约束，需要按如下方式将 reduce 轴拆分：

``` python
zo, zi = s[C].split(z, factor=factor)
s[C].reorder(x, yo, zo, yi, zi)
```

由于 tensorize 内联函数目前只覆盖了 reduce 轴的一部分，而非使用「body」函数，因此 TVM 需要一个 `reduce_reset` 函数（在 reduce for 循环之前调用），以及一个 `reduce_update` 函数（定义了「update」计算策略）。

``` python
def gemv_impl():
    cc_code = """
      extern "C" int gemv_update(float *cc, float *aa, float *bb, int m, int l, int stride) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < l; ++j) {
                cc[i] += aa[j] * bb[i * stride + j];
            }
        }
        return 0;
      }
      extern "C" int gemv_reset(float *cc, int m) {
        for (int i = 0; i < m; ++i) {
            cc[i] = 0.0;
        }
        return 0;
      }
    """
    from tvm.contrib import utils, clang

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # 从 C 源代码创建 LLVM ir
    ll_code = clang.create_llvm(cc_code, output=ll_path)
    return ll_code

def intrin_gemv(m, l):
    a = te.placeholder((l,), name="a")
    b = te.placeholder((m, l), name="b")
    k = te.reduce_axis((0, l), name="k")
    c = te.compute((m,), lambda i: te.sum(a[k] * b[i, k], axis=k), name="c")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[te.var("s1"), 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    "gemv_update",
                    cc.access_ptr("w"),
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    m,
                    l,
                    bb.strides[0],
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern("int32", "gemv_reset", cc.access_ptr("w"), m))
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})
```

注意，`intrin_func` 返回一个三元组：`(body, reduce_reset, reduce_update)`。如果 tensorization 包括所有 reduce 轴，将调用函数 `body()`，否则 `reduce_reset()` 和 `reduce_update()` 将一起使用。

示例中 `body()` 和 `reduce_update()` 的实现相同，在其他情况下，硬件对这两个函数的指令可能不同。此外，可以看到 `bb.strides[0]` 由于平铺而与 `l` 不同。

对 squared GEMV 进行 tensorize，构建并检查结果。

``` python
gemv = intrin_gemv(factor, factor)
s[C].tensorize(yi, gemv)
s[C].pragma(yo, "import_llvm", gemv_impl())

func = tvm.build(s, [A, B, C], target="llvm", name="gemv")
a = np.random.uniform(size=get_const_tuple(A.shape)).astype(dtype)
b = np.random.uniform(size=get_const_tuple(B.shape)).astype(dtype)
c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=dtype), dev)
func(tvm.nd.array(a, dev), tvm.nd.array(b, dev), c)
tvm.testing.assert_allclose(c.numpy(), np.dot(a, b.T), rtol=1e-3)
```

## 总结

本教程演示了 TVM 中 tensorize 内联函数的用法。tensorize 提供了一种方法，使得用户通过 micro-kernels 获得完全优化调度。例如，Intel CPU 上的 INT8 量化使用 tensorize 直接调用 AVX 指令。此外，它还使 TVM 能够编译为 ASIC - 查看 [VTA: Versatile Tensor Accelerator](https://tvm.apache.org/docs/topic/vta/index.html#vta-index) 获取详细信息。文档还演示了如何使用内联汇编导入，使用户轻松地将 asm 注入到调度中。

[下载 Python 源代码：tensorize.py](https://tvm.apache.org/docs/_downloads/428c6201e29ce74e73c6b41eee589f62/tensorize.py)

[下载 Jupyter Notebook：tensorize.ipynb](https://tvm.apache.org/docs/_downloads/3b5e41b16a898b72d18127ebe2182c66/tensorize.ipynb)