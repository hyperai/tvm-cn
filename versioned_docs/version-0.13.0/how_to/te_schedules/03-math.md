---
title: 内联及数学函数
---

# 内联及数学函数

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_schedules/intrin_math.html#sphx-glr-download-how-to-work-with-schedules-intrin-math-py) 下载完整的示例代码
:::

**作者**：[Tianqi Chen](https://tqchen.github.io/)

尽管 TVM 支持基本的算术运算，但很多时候，也需要复杂的内置函数，例如 `exp` 取指函数。

这些函数是依赖 target 系统的，并且在不同 target 平台中可能具有不同的名称。本教程会学习到如何调用这些 target-specific 函数，以及如何通过 TVM 内联 API 统一接口。

``` python
from __future__ import absolute_import, print_function

import numpy as np
import tvm
from tvm import te
from tvm.ir import register_op_attr, register_intrin_lowering
```

## 直接声明外部数学调用

调用 target-specific 函数最直接方法，就是通过 TVM 中的 extern 函数调用构造。以下示例用 `tvm.tir.call_pure_extern` 来调用 `__expf` 函数（仅在 CUDA 下可用）。

``` python
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: tvm.tir.call_pure_extern("float32", "__expf", A[i]), name="B")
s = te.create_schedule(B.op)
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
f = tvm.build(s, [A, B], "cuda", name="myexp")
print(f.imported_modules[0].get_source())
```

输出结果：

``` c
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
extern "C" __global__ void __launch_bounds__(64) myexp_kernel0(float* __restrict__ B, float* __restrict__ A, int n, int stride, int stride1) {
  if (((int)blockIdx.x) < (n >> 6)) {
    B[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride)] = __expf(A[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1)]);
  } else {
    if (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < n) {
      B[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride)] = __expf(A[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1)]);
    }
  }
}
```

## 统一内联调用

以上代码验证了直接外部调用可用于 device-specific 的函数。但上述方式仅适用于带有浮点类型的 CUDA target。理想情况下，我们希望写一套代码，即可适用于任何设备以及任何数据类型。

TVM 内联函数为用户提供了实现机制，且推荐用这个方法来解决问题。以下代码用的是 te.exp，它创建了一个内联调用 `tvm.te.exp()` 来做指数。

``` python
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: te.exp(A[i]), name="B")
s = te.create_schedule(B.op)
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
fcuda = tvm.build(s, [A, B], "cuda", name="myexp")
print(fcuda.imported_modules[0].get_source())
```

输出结果：

``` c
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
extern "C" __global__ void __launch_bounds__(64) myexp_kernel0(float* __restrict__ B, float* __restrict__ A, int n, int stride, int stride1) {
  if (((int)blockIdx.x) < (n >> 6)) {
    B[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride)] = __expf(A[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1)]);
  } else {
    if (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < n) {
      B[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride)] = __expf(A[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1)]);
    }
  }
}
```

该代码适用于 CUDA 和 opencl，相同的 te.exp 也可用于 float64 数据类型。

``` python
fopencl = tvm.build(s, [A, B], "opencl", name="myexp")
print(fopencl.imported_modules[0].get_source())
```

输出结果：

``` bash
// Function: myexp_kernel0
__kernel void myexp_kernel0(__global float* restrict B, __global float* restrict A, int n, int stride, int stride1) {
  if (((int)get_group_id(0)) < (n >> 6)) {
    B[(((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) * stride)] = exp(A[(((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) * stride1)]);
  } else {
    if (((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) < n) {
      B[(((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) * stride)] = exp(A[(((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) * stride1)]);
    }
  }
}
```

## 内联函数降级规则

当调用 `tvm.te.exp()` 时，TVM 会创建一个 intrinsic Call Expr。TVM 使用转换规则（transformation rules），将内联调用（intrinsic call）转换为特定设备的外部调用（extern calls）。

TVM 支持在运行时自定义规则，以下示例为 `exp` 自定义 CUDA 降级规则。

``` python
def my_cuda_math_rule(op):
    """自定义 CUDA 内联函数降级规则"""
    assert isinstance(op, tvm.tir.Call)
    name = op.op.name
    assert name.startswith("tir.")
    dispatch_name = name[4:]
    if op.dtype == "float32":
        # 调用浮点函数
        return tvm.tir.call_pure_extern("float32", "%sf" % dispatch_name, op.args[0])
    elif op.dtype == "float64":
        # 调用双精度函数
        return tvm.tir.call_pure_extern("float32", dispatch_name, op.args[0])
    else:
        # 不能转换，返回自身。
        return op

register_intrin_lowering("tir.exp", target="cuda", f=my_cuda_math_rule, level=99)
```

输出结果：

``` bash
<function my_cuda_math_rule at 0x7f7017159dd0>
```

用选项覆盖现有规则，从而将规则注册到 TVM。注意，打印代码与之前代码的区别：新规则用数学函数 `expf`，而不是快速数学版本 `__expf`。

``` python
fcuda = tvm.build(s, [A, B], "cuda", name="myexp")
print(fcuda.imported_modules[0].get_source())
```

输出结果：

``` c
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
extern "C" __global__ void __launch_bounds__(64) myexp_kernel0(float* __restrict__ B, float* __restrict__ A, int n, int stride, int stride1) {
  if (((int)blockIdx.x) < (n >> 6)) {
    B[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride)] = expf(A[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1)]);
  } else {
    if (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < n) {
      B[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride)] = expf(A[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1)]);
    }
  }
}
```

## 添加内联函数

对于 TVM 未提供的内联函数，用户可以借助内联规则系统，添加新的内联函数。以下是将内联函数 `mylog` 添加到系统的示例：

``` python
def mylog(x):
    """自定义日志内联函数"""
    return tvm.tir.call_intrin(x.dtype, "tir.mylog", x)

def my_cuda_mylog_rule(op):
    """CUDA 降级日志的规则"""
    if op.dtype == "float32":
        return tvm.tir.call_pure_extern("float32", "logf", op.args[0])
    elif op.dtype == "float64":
        return tvm.tir.call_pure_extern("float64", "log", op.args[0])
    else:
        return op

# 新的注册操作是通过注册操作的属性来触发的
register_op_attr("tir.mylog", "TCallEffectKind", tvm.tir.CallEffectKind.Pure)
register_intrin_lowering("tir.mylog", target="cuda", f=my_cuda_mylog_rule, level=99)

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: mylog(A[i]), name="B")
s = te.create_schedule(B.op)
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
fcuda = tvm.build(s, [A, B], "cuda", name="mylog")
print(fcuda.imported_modules[0].get_source())
```

输出结果：

``` c
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
extern "C" __global__ void __launch_bounds__(64) mylog_kernel0(float* __restrict__ B, float* __restrict__ A, int n, int stride, int stride1) {
  if (((int)blockIdx.x) < (n >> 6)) {
    B[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride)] = logf(A[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1)]);
  } else {
    if (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < n) {
      B[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride)] = logf(A[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1)]);
    }
  }
}
```

## 总结

* TVM 能调用依赖 target 的外部数学函数。
* 用内联函数为函数定义统一的接口。
* 有关 TVM 中更多可用的内联函数，查看 `tvm.tir`。
* 通过自定义规则，从而自定义内联行为。


[下载 Python 源代码：intrin_math.py](https://tvm.apache.org/docs/_downloads/d9089082842c138d4c81335f88c60c82/intrin_math.py)

[下载 Jupyter Notebook：intrin_math.ipynb](https://tvm.apache.org/docs/_downloads/1e482ba1190961191e3a0bdbd0585faa/intrin_math.ipynb)
