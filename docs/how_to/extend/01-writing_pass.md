---
title: 编写自定义 Pass
---

# 编写自定义 Pass

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/extend_tvm/low_level_custom_pass.html#sphx-glr-download-how-to-extend-tvm-low-level-custom-pass-py) 下载完整的示例代码
:::

**作者**：[Jian Weng](https://were.github.io/)

TVM 是一个抽象出机器学习加速器异质性的框架，有时用户希望自定义一些分析和 IR 转换，使得 TVM 适应自己的专用硬件。本教程介绍如何在 TVM 中编写自定义 Pass。

## 先决条件

阅读本教程前，假设读者已经熟悉以下主题：

* 在 TVM 中编写算法并对其进行调度，若不熟悉，请参阅示例教程如 [如何在 CPU 上优化 GEMM](/docs/how_to/optimize/cpu_conv)。
* 熟悉 HalideIR 的基本结构，若不熟悉，请参阅 `HalideIR/src/ir/IR.h` 了解定义了 IR 节点的哪些属性。
* 访问器设计模式，若不熟悉，请参阅 [Python AST 模块](https://docs.python.org/3/library/ast.html) 以查看 AST 访问器的实现原理。
* Schedule 如何降低为 IRModule 类或 LLVM 模块。若不熟悉，请参阅 `python/tvm/build_module.py` 获取相关基础知识。

``` python
import tvm
from tvm import te
import numpy as np
```

首先编写一个简单的向量加法，并用默认 schedule 构建。然后，使用自定义的降低 pass 而非调度原语，来直接操作 IR。

``` python
n = tvm.tir.const(128, "int32")
a = te.placeholder((n,), name="a")
b = te.placeholder((n,), name="b")
c = te.compute((n,), lambda i: a[i] + b[i], name="c")

sch = te.create_schedule(c.op)
ir = tvm.lower(sch, [a, b, c])
print(ir)
```

输出结果：

``` bash
@main = primfn(a_1: handle, b_1: handle, c_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {a: Buffer(a_2: Pointer(float32), float32, [128], []),
             b: Buffer(b_2: Pointer(float32), float32, [128], []),
             c: Buffer(c_2: Pointer(float32), float32, [128], [])}
  buffer_map = {a_1: a, b_1: b, c_1: c}
  preflattened_buffer_map = {a_1: a_3: Buffer(a_2, float32, [128], []), b_1: b_3: Buffer(b_2, float32, [128], []), c_1: c_3: Buffer(c_2, float32, [128], [])} {
  for (i: int32, 0, 128) {
    c[i] = (a[i] + b[i])
  }
}
```

## 编写 Pass

本质上，「IR 转换 pass」是将语句映射到新语句的函数。因此，我们要定义这个向量化函数，并逐步实现它。

TVM 为用户提供了两个类来分析和转换 IR。

### IR 访问器

可以用 `tvm.tir.stmt_functor.post_order_visit(stmt, func)` 从 Halide IR 中收集信息。 `func` 是一个回调函数，会在退出当前 IR 节点之前调用，即 post-order visit。然后存储 IR 访问的结果，因为 `func` 的返回值将被忽略。

:::note
必须用数组来存储 IR 访问的结果。值甚至是一个单变量。这主要是由于 Python-C runtime 的限制，每次递归都会刷新变量值，但会保留数组值。
:::

``` python
loops = []

def find_width8(op):
    """查找范围可以被 8 整除的所有「tir.For」节点。"""
    if isinstance(op, tvm.tir.For):
        if isinstance(op.extent, tvm.tir.IntImm):
            if op.extent.value % 8 == 0:
                loops.append(op)
```

### IR 转换

转换接口与访问器接口略有不同。访问器中只有一个后序回调，但转换访问器同时支持前序回调和后序回调。若要保留原始 IR 节点，只需返回 None。若要将当前节点更改为某个节点，使用 TVM IR maker 接口构建，并返回这个值。

:::note
若调用 pre-order 函数后返回一个非 None 的值，则将跳过 post-order 函数。
:::

``` python
def vectorize8(op):
    """Split 可以向量化 `find_width8` 中的循环。"""
    if op in loops:
        extent = op.extent.value
        name = op.loop_var.name
        lo, li = te.var(name + ".outer"), te.var(name + ".inner")
        body = tvm.tir.stmt_functor.substitute(op.body, {op.loop_var: lo * 8 + li})
        body = tvm.tir.For(li, 0, 8, tvm.tir.ForKind.VECTORIZED, body)
        body = tvm.tir.For(lo, 0, extent // 8, tvm.tir.ForKind.SERIAL, body)
        return body
    return None

@tvm.tir.transform.prim_func_pass(opt_level=0)
def vectorize(f, mod, ctx):
    global loops

    tvm.tir.stmt_functor.post_order_visit(f.body, find_width8)

    if not loops:
        return f

    # 最后一个列表参数表示将转换哪些类型的节点。
    # 在这种情况下，只有 `For` 节点会调用 `vectorize8`
    return f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, None, vectorize8, ["tir.For"]))
```

## 对接低层（Glue to Lowering）

到目前为止，已经完成了这个 IR 转换 pass 的编写。接下来将这个 pass 和 TVM 的底层 pass 对接。

在这种情况下，通过**元组列表**作为参数提供给 `tir.add_lower_pass`，将上面编写的 pass 注入 TVM 标准较低级的 pass。「元组」表示降级的不同阶段。 TVM 中有四个阶段的降级，每个阶段完成后，都会调用自定义的阶段。

:::note
**以下是每个阶段完成的基本转换：**

* 阶段 0 生成原始 IR 和循环级别。
* 阶段 1 扁平化数组存储。
* 阶段 2 转换循环，如展开、矢量化和线程绑定。
* 阶段 3 清理工作。
:::

因此，这个转换 pass 适合放在第 1 阶段之后。

```python
with tvm.transform.PassContext(config={"tir.add_lower_pass": [(1, vectorize)]}):
    print(tvm.lower(sch, [a, b, c]))
```

输出结果：

``` bash
@main = primfn(a_1: handle, b_1: handle, c_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {a: Buffer(a_2: Pointer(float32), float32, [128], []),
             b: Buffer(b_2: Pointer(float32), float32, [128], []),
             c: Buffer(c_2: Pointer(float32), float32, [128], [])}
  buffer_map = {a_1: a, b_1: b, c_1: c}
  preflattened_buffer_map = {a_1: a_3: Buffer(a_2, float32, [128], []), b_1: b_3: Buffer(b_2, float32, [128], []), c_1: c_3: Buffer(c_2, float32, [128], [])} {
  for (i.outer: int32, 0, 16) {
    let cse_var_1: int32 = (i.outer*8)
    c[ramp(cse_var_1, 1, 8)] = (a[ramp(cse_var_1, 1, 8)] + b[ramp(cse_var_1, 1, 8)])
  }
}
```

## 快速回顾

快速回顾本教程有关编写自定义 IR 转换 pass：

* 用 `tvm.tir.stmt_functor.post_order_visit` 收集每个 IR 节点的信息。
* 用 `tvm.tir.stmt_functor.ir_transform` 转换 IR 节点。
* 总结以上两点来编写一个 IR 转换函数。
* 用 `tvm.transform.PassContext` 将此函数放入 TVM 降级 pass。

[下载 Python 源代码：low_level_custom_pass.py](https://tvm.apache.org/docs/_downloads/caa649473e845a115a0397a2855fd356/low_level_custom_pass.py)

[下载 Jupyter Notebook：low_level_custom_pass.ipynb](https://tvm.apache.org/docs/_downloads/d58ec306b89044968adefb49e6552378/low_level_custom_pass.ipynb)
