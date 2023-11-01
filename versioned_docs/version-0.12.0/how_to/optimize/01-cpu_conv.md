---
title: 如何在 CPU 上优化 GEMM
---

# 如何在 CPU 上优化 GEMM

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/optimize_operators/opt_gemm.html#sphx-glr-download-how-to-optimize-operators-opt-gemm-py) 下载完整的示例代码
:::

**作者**：[Jian Weng](https://github.com/were), [Ruofei Yu](https://github.com/yuruofeifei)

TVM 提供抽象接口，允许用户分别描述算法和算法的实现（所谓的调度）。通常，以高性能调度编写算法会破坏算法的可读性和模块化。此外，尝试各种看似有希望的 schedules 非常耗时。在 TVM 的帮助下，可以有效地尝试这些 schedules 以提高性能。

本教程将演示如何用 TVM 优化矩阵乘法，并通过 18 行代码实现比 baseline 快 200 倍的性能。

在 CPU 上执行的密集计算应用程序有两个重要的优化：

1. 提高内存访问的 cache 命中率。高 cache 命中率可以加速复杂的数值计算和热点内存访问。需要将原始内存访问模式转换为适合 cache 策略的模式。
2. SIMD（单指令多数据），或者称之为向量处理单元，每次都会处理一小批数据，而不是单个网格。需要将循环体中的数据访问模式转换为统一模式，以便 LLVM 后端可以将其降级到 SIMD。

实际上，本教程中使用的所有方法都在这个 [repo](https://github.com/flame/how-to-optimize-gemm) 中提到了。其中一些已被 TVM 抽象自动应用，但有一些由于 TVM 的限制，不能被简单地应用。

下面提到的所有实验结果，都是在配备 Intel i7-4770HQ CPU 的 2015 年 15 英寸 MacBook 上执行的，所有 x86 CPU 的高速缓存行大小应为 64 字节。

## 准备和 baseline

本教程演示如何使用 TVM 优化矩阵乘法。实际演示前，首先定义这些变量。然后编写一个 baseline 实现，这是在 TVM 中编写矩阵乘法的最简单方法。

``` python
import tvm
import tvm.testing
from tvm import te
import numpy
import timeit

# 矩阵的大小
# (M, K) x (K, N)
# 可自由尝试不同的 shapes，有时 TVM 优化在 MKL 中的表现优于 numpy。
M = 1024
K = 1024
N = 1024

# tvm 中的默认张量类型
dtype = "float32"

# 为 SIMD 使用英特尔 AVX2（高级向量扩展）ISA
# 要获得最佳性能，更改以下行
# 为 llvm -mcpu=core-avx2，或者使用的特定类型的 CPU
target = "llvm"
dev = tvm.device(target, 0)

# 用于测试的随机生成张量
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

np_repeat = 100
np_runing_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_runing_time / np_repeat))

answer = numpy.dot(a.numpy(), b.numpy())

# 算法
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

# 默认 schedule
s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("Baseline: %f" % evaluator(a, b, c).mean)
```

输出结果：

``` bash
Numpy running time: 0.018437
Baseline: 3.336375
```

在 TVM 中，始终可以检查较低级别的 IR 以调试或优化 schedule。这是使用 baseline schedule 生成的 IR。

``` python
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (m: int32, 0, 1024) {
    for (n: int32, 0, 1024) {
      C[((m*1024) + n)] = 0f32
      for (k: int32, 0, 1024) {
        let cse_var_2: int32 = (m*1024)
        let cse_var_1: int32 = (cse_var_2 + n)
        C[cse_var_1] = (C[cse_var_1] + (A[(cse_var_2 + k)]*B[((k*1024) + n)]))
      }
    }
  }
}
```

## 分块

提高缓存命中率的一个重要技巧是分块——数据块将逐块计算。块内的内存访问是一个局部性的小邻域。本教程选择 32 作为分块因子，因此该块将填充 32 * 32 * sizeof(float) ，即总大小为 32KB 的缓存中的 4KB（L1 数据缓存）。

``` python
bn = 32
kfactor = 4
s = te.create_schedule(C.op)

# 通过循环 tiling 进行分块
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

# 将 reduction 域提升到分块循环之外
s[C].reorder(mo, no, ko, ki, mi, ni)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

# 通过简单地将循环 32x32 分块，并将 ko、ki 提升到分块循环之外，
# 可以看到与 baseline 相比，加速有很大提升。
evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt1: %f" % evaluator(a, b, c).mean)
```

输出结果：

``` bash
Opt1: 0.307321
```

分块后生成的 IR：

``` bash
print(tvm.lower(s, [A, B, C], simple_mode=True))
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (m.outer: int32, 0, 32) {
    for (n.outer: int32, 0, 32) {
      for (m.inner.init: int32, 0, 32) {
        for (n.inner.init: int32, 0, 32) {
          C[((((m.outer*32768) + (m.inner.init*1024)) + (n.outer*32)) + n.inner.init)] = 0f32
        }
      }
      for (k.outer: int32, 0, 256) {
        for (k.inner: int32, 0, 4) {
          for (m.inner: int32, 0, 32) {
            for (n.inner: int32, 0, 32) {
              let cse_var_3: int32 = (n.outer*32)
              let cse_var_2: int32 = ((m.outer*32768) + (m.inner*1024))
              let cse_var_1: int32 = ((cse_var_2 + cse_var_3) + n.inner)
              C[cse_var_1] = (C[cse_var_1] + (A[((cse_var_2 + (k.outer*4)) + k.inner)]*B[((((k.outer*4096) + (k.inner*1024)) + cse_var_3) + n.inner)]))
            }
          }
        }
      }
    }
  }
}
```

## 向量化

另一个重要技巧是向量化，当内存访问模式一致时，编译器可以检测到这种模式并将连续内存传递给向量处理器。TVM 中可以用 vectorize 接口来提示编译器这种模式，这样就可以进行加速。

本教程选择向量化内部循环 row data（对缓存更友好）。

``` python
s = te.create_schedule(C.op)
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

s[C].reorder(mo, no, ko, ki, mi, ni)

# 向量化
s[C].vectorize(ni)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt2: %f" % evaluator(a, b, c).mean)
```

输出结果：

``` bash
Opt2: 0.349439
```

向量化后生成的 IR：

``` python
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (m.outer: int32, 0, 32) {
    for (n.outer: int32, 0, 32) {
      for (m.inner.init: int32, 0, 32) {
        C[ramp((((m.outer*32768) + (m.inner.init*1024)) + (n.outer*32)), 1, 32)] = broadcast(0f32, 32)
      }
      for (k.outer: int32, 0, 256) {
        for (k.inner: int32, 0, 4) {
          for (m.inner: int32, 0, 32) {
            let cse_var_3: int32 = (n.outer*32)
            let cse_var_2: int32 = ((m.outer*32768) + (m.inner*1024))
            let cse_var_1: int32 = (cse_var_2 + cse_var_3)
            C[ramp(cse_var_1, 1, 32)] = (C[ramp(cse_var_1, 1, 32)] + (broadcast(A[((cse_var_2 + (k.outer*4)) + k.inner)], 32)*B[ramp((((k.outer*4096) + (k.inner*1024)) + cse_var_3), 1, 32)]))
          }
        }
      }
    }
  }
}
```

## 循环置换

查看上面的 IR，可以看到内部循环的 row data 对于 B 和 C 都是向量化的。接下来查看 A 的访问模式。在当前调度中，A 是逐列访问的，但它对缓存不友好。如果改变 ki 和内轴 mi 的嵌套循环顺序，A 矩阵的访问模式对缓存更友好。

``` python
s = te.create_schedule(C.op)
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

# 重新排序
s[C].reorder(mo, no, ko, mi, ki, ni)
s[C].vectorize(ni)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt3: %f" % evaluator(a, b, c).mean)
```

输出结果：

``` bash
Opt3: 0.115375
```

循环置换后生成的 IR：

``` python
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (m.outer: int32, 0, 32) {
    for (n.outer: int32, 0, 32) {
      for (m.inner.init: int32, 0, 32) {
        C[ramp((((m.outer*32768) + (m.inner.init*1024)) + (n.outer*32)), 1, 32)] = broadcast(0f32, 32)
      }
      for (k.outer: int32, 0, 256) {
        for (m.inner: int32, 0, 32) {
          for (k.inner: int32, 0, 4) {
            let cse_var_3: int32 = (n.outer*32)
            let cse_var_2: int32 = ((m.outer*32768) + (m.inner*1024))
            let cse_var_1: int32 = (cse_var_2 + cse_var_3)
            C[ramp(cse_var_1, 1, 32)] = (C[ramp(cse_var_1, 1, 32)] + (broadcast(A[((cse_var_2 + (k.outer*4)) + k.inner)], 32)*B[ramp((((k.outer*4096) + (k.inner*1024)) + cse_var_3), 1, 32)]))
          }
        }
      }
    }
  }
}
```

## 数组打包

另一个重要的技巧是数组打包，对多维数组的存储进行重新排序，展平并存储在一维内存中，方便顺序访问。

![图片](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/array-packing.png)

注意：此图是数组打包工作原理的一般说明。

可以用数组打包来解决 B 的访问模式。观察展平后 B 的数组访问模式，当迭代 K 维时，它不是顺序的。可以用维度 [K][N] 对 B 重新排序，使其具有 [N/bn][K][bn] 维度，其中 bn 是分块因子，也是内循环中 B 的向量大小。

这种重新排序将 N 拆分为两个维度——bigN（N/bn）和 littleN（bn）——新维度 [N/bn][K][bn] 匹配 B 从外部到内部循环的索引（no, ko, ki, ni) 在展平后导致 B 的顺序访问模式。

``` python
# 我们必须稍微重新编写算法。
packedB = te.compute(
    (N / bn, K, bn), lambda bigN, k, littleN: B[k, bigN * bn + littleN], name="packedB"
)
C = te.compute(
    (M, N),
    lambda m, n: te.sum(A[m, k] * packedB[n // bn, k, tvm.tir.indexmod(n, bn)], axis=k),
    name="C",
)

s = te.create_schedule(C.op)

mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

s[C].reorder(mo, no, ko, mi, ki, ni)
s[C].vectorize(ni)

bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt4: %f" % evaluator(a, b, c).mean)
```

输出结果：

``` bash
Opt4: 0.109499
```

数组打包后生成的 IR：

``` python
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  allocate(packedB: Pointer(global float32x32), float32x32, [32768]), storage_scope = global {
    for (bigN: int32, 0, 32) "parallel" {
      for (k: int32, 0, 1024) {
        packedB_1: Buffer(packedB, float32x32, [32768], [])[((bigN*1024) + k)] = B[ramp(((k*1024) + (bigN*32)), 1, 32)]
      }
    }
    for (m.outer: int32, 0, 32) {
      for (n.outer: int32, 0, 32) {
        for (m.inner.init: int32, 0, 32) {
          C[ramp((((m.outer*32768) + (m.inner.init*1024)) + (n.outer*32)), 1, 32)] = broadcast(0f32, 32)
        }
        for (k.outer: int32, 0, 256) {
          for (m.inner: int32, 0, 32) {
            for (k.inner: int32, 0, 4) {
              let cse_var_3: int32 = ((m.outer*32768) + (m.inner*1024))
              let cse_var_2: int32 = (k.outer*4)
              let cse_var_1: int32 = (cse_var_3 + (n.outer*32))
              C[ramp(cse_var_1, 1, 32)] = (C[ramp(cse_var_1, 1, 32)] + (broadcast(A[((cse_var_3 + cse_var_2) + k.inner)], 32)*packedB_1[(((n.outer*1024) + cse_var_2) + k.inner)]))
            }
          }
        }
      }
    }
  }
}
```

## 块的写缓存

分块后，程序会逐块将结果写入 C（访问模式不是顺序的），因此，可以使用顺序缓存数组来保存块结果，并在所有块结果准备好时写入 C。

``` python
s = te.create_schedule(C.op)

# 分配写缓存
CC = s.cache_write(C, "global")
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

# 写缓存在 no 被计算
s[CC].compute_at(s[C], no)

# 新的内轴
mc, nc = s[CC].op.axis

(kaxis,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(kaxis, factor=kfactor)
s[CC].reorder(ko, mc, ki, nc)
s[CC].vectorize(nc)

# TODO: 添加单独的优化步骤，来讨论循环展开
# unrolling 是一种循环优化策略，可以减少分支
# 预测失败，以及增加并发执行的机会
# 展开 kfactor 循环
s[CC].unroll(ki)

bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt5: %f" % evaluator(a, b, c).mean)
```

输出结果：

``` bash
Opt5: 0.110823
```

分块后生成的 IR：

``` bash
print(tvm.lower(s, [A, B, C], simple_mode=True))
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  allocate(packedB: Pointer(global float32x32), float32x32, [32768]), storage_scope = global;
  allocate(C.global: Pointer(global float32), float32, [1024]), storage_scope = global {
    for (bigN: int32, 0, 32) "parallel" {
      for (k: int32, 0, 1024) {
        packedB_1: Buffer(packedB, float32x32, [32768], [])[((bigN*1024) + k)] = B[ramp(((k*1024) + (bigN*32)), 1, 32)]
      }
    }
    for (m.outer: int32, 0, 32) {
      for (n.outer: int32, 0, 32) {
        for (m.c.init: int32, 0, 32) {
          C.global_1: Buffer(C.global, float32, [1024], [])[ramp((m.c.init*32), 1, 32)] = broadcast(0f32, 32)
        }
        for (k.outer: int32, 0, 256) {
          for (m.c: int32, 0, 32) {
            let cse_var_4: int32 = (k.outer*4)
            let cse_var_3: int32 = (m.c*32)
            let cse_var_2: int32 = ((n.outer*1024) + cse_var_4)
            let cse_var_1: int32 = (((m.outer*32768) + (m.c*1024)) + cse_var_4)
             {
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[cse_var_1], 32)*packedB_1[cse_var_2]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 1)], 32)*packedB_1[(cse_var_2 + 1)]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 2)], 32)*packedB_1[(cse_var_2 + 2)]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 3)], 32)*packedB_1[(cse_var_2 + 3)]))
            }
          }
        }
        for (m.inner: int32, 0, 32) {
          for (n.inner: int32, 0, 32) {
            C[((((m.outer*32768) + (m.inner*1024)) + (n.outer*32)) + n.inner)] = C.global_1[((m.inner*32) + n.inner)]
          }
        }
      }
    }
  }
}
```

## 并行化

此外，还可以利用多核处理器进行线程级并行化。

``` python
s = te.create_schedule(C.op)

CC = s.cache_write(C, "global")

mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

s[CC].compute_at(s[C], no)

mc, nc = s[CC].op.axis

(kaxis,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(kaxis, factor=kfactor)
s[CC].reorder(ko, mc, ki, nc)
s[CC].vectorize(nc)
s[CC].unroll(ki)

# 并行
s[C].parallel(mo)

bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=50)
opt6_time = evaluator(a, b, c).mean
print("Opt6: %f" % opt6_time)
```

输出结果：

``` bash
Opt6: 0.144875
```

并行化后生成的 IR：

``` python
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  allocate(packedB: Pointer(global float32x32), float32x32, [32768]), storage_scope = global {
    for (bigN: int32, 0, 32) "parallel" {
      for (k: int32, 0, 1024) {
        packedB_1: Buffer(packedB, float32x32, [32768], [])[((bigN*1024) + k)] = B[ramp(((k*1024) + (bigN*32)), 1, 32)]
      }
    }
    for (m.outer: int32, 0, 32) "parallel" {
      allocate(C.global: Pointer(global float32), float32, [1024]), storage_scope = global;
      for (n.outer: int32, 0, 32) {
        for (m.c.init: int32, 0, 32) {
          C.global_1: Buffer(C.global, float32, [1024], [])[ramp((m.c.init*32), 1, 32)] = broadcast(0f32, 32)
        }
        for (k.outer: int32, 0, 256) {
          for (m.c: int32, 0, 32) {
            let cse_var_4: int32 = (k.outer*4)
            let cse_var_3: int32 = (m.c*32)
            let cse_var_2: int32 = ((n.outer*1024) + cse_var_4)
            let cse_var_1: int32 = (((m.outer*32768) + (m.c*1024)) + cse_var_4)
             {
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[cse_var_1], 32)*packedB_1[cse_var_2]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 1)], 32)*packedB_1[(cse_var_2 + 1)]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 2)], 32)*packedB_1[(cse_var_2 + 2)]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 3)], 32)*packedB_1[(cse_var_2 + 3)]))
            }
          }
        }
        for (m.inner: int32, 0, 32) {
          for (n.inner: int32, 0, 32) {
            C[((((m.outer*32768) + (m.inner*1024)) + (n.outer*32)) + n.inner)] = C.global_1[((m.inner*32) + n.inner)]
          }
        }
      }
    }
  }
}
```

## 总结

应用上述简单优化后，仅用 18 行代码，就可以达到使用 MKL *numpy* 性能的 60%。注意，网页上的输出反映了非专有 Docker 容器上的运行时间，是*不可靠*的。推荐自己运行本教程，观察 TVM 的性能提升。

[下载 Python 源代码：opt_gemm.py](https://tvm.apache.org/docs/_downloads/96137df89d8034b548f407123ec50ce9/opt_gemm.py)

[下载 Jupyter Notebook：opt_gemm.ipynb](https://tvm.apache.org/docs/_downloads/0f8d36b3ffd04a5a08089dc671eb788e/opt_gemm.ipynb)
