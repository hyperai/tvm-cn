---
title: 使用张量表达式处理算子
---

# 使用张量表达式处理算子

:::note
单击 [此处](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html#sphx-glr-download-tutorial-tensor-expr-get-started-py) 下载完整的示例代码
:::

**作者**：[Tianqi Chen](https://tqchen.github.io/)

本教程重点关注 TVM 如何用张量表达式（TE）来定义张量计算并应用循环优化。 TE 用纯函数式语言描述张量计算（即每个函数表达式都不会产生副作用（side effect））。从 TVM 的整体来看，Relay 将计算描述为一组算子，每个算子都可以表示为一个 TE 表达式，其中每个 TE 表达式接收输入张量并产生一个输出张量。

这是 TVM 中张量表达式语言的入门教程。 TVM 使用特定领域的张量表达式来进行有效的内核构建。下面将通过两个使用张量表达式语言的示例，来演示基本工作流程。第一个例子介绍了 TE 和带有向量加法的调度。通过逐步讲解如何用 TE 优化矩阵乘法，对第二个例子的这些概念进行了扩展。后续涉及到更高阶的 TVM 功能教程，将基于此矩阵乘法示例。

## 示例 1：在 TE 中为 CPU 编写并调度向量加法

以下 Python 示例展示了如何实现用于向量加法的 TE，以及针对 CPU 的调度。首先初始化一个 TVM 环境：

``` python
import tvm
import tvm.testing
from tvm import te
import numpy as np
```

为了提高性能，可以指定目标 CPU。如果使用的是 LLVM，可输入命令 `llc --version` 来获取 CPU 类型，并通过查看 `/proc/cpuinfo` 获取处理器支持的其他扩展。例如，如果 CPU 支持 AVX-512 指令，则可以使用 `llvm -mcpu=skylake-avx512`。

``` python
tgt = tvm.target.Target(target="llvm", host="llvm")
```

### 描述向量计算

TVM 采用张量语义，每个中间结果表示为一个多维数组。用户需要描述生成张量的计算规则。首先定义一个符号变量 `n` 来表示 shape。然后定义两个占位符张量，`A` 和 `B`，给定 shape 都为 `(n,)`。然后用 `compute` 操作描述结果张量 `C`。

`compute` 定义了一个计算，其输出符合指定的张量 shape，计算将在 lambda 函数定义的张量中的每个位置执行。注意，虽然 `n` 是一个变量，但它定义了 `A`、`B` 和 `C` 张量之间的一致的 shape。注意，此过程只是在声明应该如何进行计算，不会发生实际的计算。

``` python
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
```

:::note Lambda 函数
`te.compute` 方法的第二个参数是执行计算的函数。此示例使用匿名函数（也称 `lambda` 函数）来定义计算，在本例中是对 `A` 和 `B` 的第 `i` 个元素进行加法运算。
:::

### 为计算创建默认 schedule

虽然上面描述了计算规则，但为适应不同的设备，可用不同的方式来计算 `C`。对于具有多个轴（axis）的张量，可以选择首先迭代哪个轴，或将计算拆分到不同的线程。 TVM 要求用户提供 schedule，这是对如何执行计算的描述。 TE 中的调度操作可以在其他操作中更改循环顺序、跨不同线程拆分计算以及将数据块组合在一起。调度背后的一个重要概念是它们只描述如何执行计算，因此同一 TE 的不同调度将产生相同的结果。

TVM 允许创建一个通过按行迭代计算 `C` 的 schedule：

``` python
for (int i = 0; i < n; ++i) {
  C[i] = A[i] + B[i];
}
s = te.create_schedule(C.op)
```

### 编译和评估默认 schedule

用 TE 表达式和 schedule 可为目标语言和架构（本例为 LLVM 和 CPU）生成可运行的代码。TVM 提供 schedule、schedule 中的 TE 表达式列表、target 和 host，以及正在生成的函数名。输出结果是一个类型擦除函数（type-erased function），可直接从 Python 调用。

下面将使用 `tvm.build` 创建函数。 build 函数接收 schedule、所需的函数签名（包括输入和输出）以及要编译到的目标语言。

``` python
fadd = tvm.build(s, [A, B, C], tgt, name="myadd")
```

输出结果：

``` plain
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

运行该函数，并将输出与 numpy 中的相同计算进行比较。编译后的 TVM 函数提供了一个任何语言都可调用的 C API。首先创建 TVM 编译调度的目标设备（本例为 CPU）。这个例子的目标设备为 LLVM CPU。然后初始化设备中的张量，并执行自定义的加法操作。为了验证计算是否正确，可将 C 张量的输出结果与 numpy 执行相同计算的结果进行比较。

``` python
dev = tvm.device(tgt.kind.name, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
```

创建一个辅助函数来运行 TVM 生成的代码的配置文件，从而比较这个版本和 numpy 的运行速度。

``` python
import timeit

np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "n = 32768\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(n, 1).astype(dtype)\n"
    "b = numpy.random.rand(n, 1).astype(dtype)\n",
    stmt="answer = a + b",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))

def evaluate_addition(func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    n = 32768
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))

    log.append((optimization, mean_time))

log = [("numpy", np_running_time / np_repeat)]
evaluate_addition(fadd, tgt, "naive", log=log)
```

输出结果：

``` plain
Numpy running time: 0.000008
naive: 0.000006
```

### 更新 schedule 以使用并行性

前面已经讲解了 TE 的基础知识，下面将更深入地了解调度的作用，以及如何使用它们来优化不同架构的张量表达式。schedule 是用多种方式来转换表达式的一系列步骤。当调度应用于 TE 中的表达式时，输入和输出保持不变，但在编译时，表达式的实现会发生变化。默认调度的这种张量加法是串行运行的，不过可以很容易实现在所有处理器线程上并行化。将并行调度操作应用于我们的计算：

``` python
s[C].parallel(C.op.axis[0])
```

`tvm.lower` 命令会生成 TE 的中间表示（IR），以及相应的 schedule。应用不同的调度操作时降低表达式，可以看到调度对计算顺序的影响。用标志 `simple_mode=True` 来返回可读的 C 风格语句：

``` python
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*n: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*n)], [], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [(stride_2: int32*n)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [n], [stride], type="auto"), B_1: B_3: Buffer(B_2, float32, [n], [stride_1], type="auto"), C_1: C_3: Buffer(C_2, float32, [n], [stride_2], type="auto")} {
  for (i: int32, 0, n) "parallel" {
    C[(i*stride_2)] = (A[(i*stride)] + B[(i*stride_1)])
  }
}
```

TVM 现在可以在独立的线程上运行这些代码块。下面将并行编译并运行这个新的 schedule：

``` python
fadd_parallel = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")
fadd_parallel(a, b, c)

tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

evaluate_addition(fadd_parallel, tgt, "parallel", log=log)
```

输出结果：

``` plain
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
parallel: 0.000006
```

### 更新 schedule 以使用向量化

现代 CPU 可以对浮点值执行 SIMD 操作，利用这一优势，可将另一个调度应用于计算表达式中。实现这一点需要多个步骤：首先，用拆分调度原语将调度拆分为内部循环和外部循环。内部循环可用向量化调度原语来调用 SIMD 指令，然后可用并行调度原语对外部循环进行并行化。选择拆分因子作为 CPU 上的线程数量。

``` python
# 重新创建 schedule, 因为前面的例子在并行操作中修改了它
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)

# 这个因子应该和适合 CPU 的线程数量匹配。
# 这会因架构差异而有所不同，不过好的规则是
# 将这个因子设置为 CPU 可用内核数量。
factor = 4

outer, inner = s[C].split(C.op.axis[0], factor=factor)
s[C].parallel(outer)
s[C].vectorize(inner)

fadd_vector = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")

evaluate_addition(fadd_vector, tgt, "vector", log=log)

print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
vector: 0.000024
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [(stride: int32*n: int32)], [], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [(stride_1: int32*n)], [], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [(stride_2: int32*n)], [], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [n], [stride], type="auto"), B_1: B_3: Buffer(B_2, float32, [n], [stride_1], type="auto"), C_1: C_3: Buffer(C_2, float32, [n], [stride_2], type="auto")} {
  for (i.outer: int32, 0, floordiv((n + 3), 4)) "parallel" {
    for (i.inner.s: int32, 0, 4) {
      if @tir.likely((((i.outer*4) + i.inner.s) < n), dtype=bool) {
        let cse_var_1: int32 = ((i.outer*4) + i.inner.s)
        C[(cse_var_1*stride_2)] = (A[(cse_var_1*stride)] + B[(cse_var_1*stride_1)])
      }
    }
  }
}
```

### 比较不同的 schedule

现在可以对不同的 schedule 进行比较：

``` python
baseline = log[0][1]
print("%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20)))
for result in log:
    print(
        "%s\t%s\t%s"
        % (result[0].rjust(20), str(result[1]).rjust(20), str(result[1] / baseline).rjust(20))
    )

```

输出结果：

``` plain
Operator                  Timing             Performance
   numpy    7.816320003257716e-06                    1.0
   naive    6.4633000000000004e-06    0.8268980795702072
parallel              6.0509e-06       0.774136677807214
  vector             2.39039e-05      3.0582038593656913
```

:::note Code Specialization
由前面可知，`A`、`B` 和 `C` 的声明都采用相同的 shape 参数 `n`。基于此，TVM 只需将单个 shape 参数传递给内核（如上面打印的设备代码所示），这是 code specialization 的一种形式。

在宿主机上，TVM 自动生成检查代码来检查参数中的约束。因此，如果将不同 shape 的数组传递给 fadd，则会引发错误。

此外，代码还可以实现更多 specialization。例如，在计算声明中写成 `n = tvm.runtime.convert(1024)`，而不是 `n = te.var("n")`。生成的函数只会接收长度为 1024 的向量。
:::

经过上述步骤，我们已经对向量加法算子（vector addition operator）进行了定义、调度和编译，接下来在 TVM runtime 执行。将算子保存为一个库，之后可以在 TVM runtime 中加载。

### GPU 的目标向量加法（可选）

TVM 适用于多种架构。下面的示例将针对 GPU 的向量加法进行编译：

``` python
# 要运行这个代码, 更改为 `run_cuda = True`
# 注意：默认这个示例不在 CI 文档上运行

run_cuda = False
if run_cuda:
    # 将这个 target 改为你 GPU 的正确后端。例如：NVIDIA GPUs：cuda；Radeon GPUS：rocm；opencl：OpenCL
    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")

    # 重新创建 schedule
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    print(type(C))

    s = te.create_schedule(C.op)

    bx, tx = s[C].split(C.op.axis[0], factor=64)

    ################################################################################
    # 最终必须将迭代轴 bx 和 tx 和 GPU 计算网格绑定。
    # 原生 schedule 对 GPU 是无效的, 这些是允许我们生成可在 GPU 上运行的代码的特殊构造

    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    ######################################################################
    # 编译
    # -----------
    # 指定 schedule 后, 可将它编译为 TVM 函数。
    # 默认 TVM 编译为可直接从 Python 端调用的类型擦除函数。
    #
    # 下面将用 tvm.build 来创建函数。
    # build 函数接收 schedule、所需的函数签名（包括输入和输出）以及要编译到的目标语言。
    #
    # fadd 的编译结果是 GPU 设备函数（如果利用了 GPU）以及调用 GPU 函数的主机 wrapper。
    # fadd 是生成的主机 wrapper 函数，它包含对内部生成的设备函数的引用。

    fadd = tvm.build(s, [A, B, C], target=tgt_gpu, name="myadd")

    ################################################################################
    # 编译后的 TVM 函数提供了一个任何语言都可调用的 C API。
    #
    # 我们在 Python 中提供了最小数组 API 来进行快速测试以及制作原型。
    # 数组 API 基于 `DLPack <https://github.com/dmlc/dlpack>`_ 标准。
    #
    # - 首先创建 GPU 设备。
    # - 然后 tvm.nd.array 将数据复制到 GPU 上。
    # - `fadd` 运行真实的计算。
    # - `numpy()` 将 GPU 数组复制回 CPU 上（然后验证正确性）。
    #
    # 注意将数据复制进出内存是必要步骤。

    dev = tvm.device(tgt_gpu.kind.name, 0)

    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    ################################################################################
    # 检查生成的 GPU 代码
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 可以在 TVM 中检查生成的代码。tvm.build 的结果是一个 TVM 模块。fadd 是包含主机模块的主机 wrapper，对 CUDA（GPU）函数来说它还包含设备模块。
    #
    # 下面的代码从设备模块中取出并打印内容代码。

    if (
        tgt_gpu.kind.name == "cuda"
        or tgt_gpu.kind.name == "rocm"
        or tgt_gpu.kind.name.startswith("opencl")
    ):
        dev_module = fadd.imported_modules[0]
        print("-----GPU code-----")
        print(dev_module.get_source())
    else:
        print(fadd.get_source())
```

## 保存和加载已编译的模块

除了 runtime 编译外，还可将编译后的模块保存到文件中，之后再进行加载。

以下代码首先执行：

* 将编译的主机模块保存到目标文件中。
* 然后将设备模块保存到 ptx 文件中。
* cc.create_shared 调用编译器（gcc）来创建共享库。

``` python
from tvm.contrib import cc
from tvm.contrib import utils

temp = utils.tempdir()
fadd.save(temp.relpath("myadd.o"))
if tgt.kind.name == "cuda":
    fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
if tgt.kind.name == "rocm":
    fadd.imported_modules[0].save(temp.relpath("myadd.hsaco"))
if tgt.kind.name.startswith("opencl"):
    fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
print(temp.listdir())
```

输出结果：

``` plain
['myadd.o', 'myadd.so']
```

模块存储格式：

CPU（主机）模块直接保存为共享库（.so）。设备代码有多种自定义格式。在我们的示例中，设备代码存储在 ptx 以及元数据 json 文件中。它们可以分别导入，从而实现单独加载和链接。

### 加载编译模块

可从文件系统中加载编译好的模块并运行代码。下面的代码分别加载主机和设备模块，并将它们链接在一起。可以验证新加载的函数是否有效。

``` python
fadd1 = tvm.runtime.load_module(temp.relpath("myadd.so"))
if tgt.kind.name == "cuda":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.ptx"))
    fadd1.import_module(fadd1_dev)

if tgt.kind.name == "rocm":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.hsaco"))
    fadd1.import_module(fadd1_dev)

if tgt.kind.name.startswith("opencl"):
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.cl"))
    fadd1.import_module(fadd1_dev)

fadd1(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
```

### 将所有内容打包到一个库中

上面的示例分别存储了设备和主机代码。 TVM 还支持将所有内容导出为一个共享库。在后台，将设备模块打包成二进制 blob，并将它们与主机代码链接在一起。目前支持 Metal、OpenCL 和 CUDA 模块的打包。

``` python
fadd.export_library(temp.relpath("myadd_pack.so"))
fadd2 = tvm.runtime.load_module(temp.relpath("myadd_pack.so"))
fadd2(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
```

:::note runtime API 和线程安全
TVM 的编译模块不依赖于 TVM 编译器，它们只依赖于最小的 runtime 库。 TVM runtime 库封装了设备驱动程序，并在编译后的函数中提供线程安全和 device-agnostic 的调用。

这意味着，可以从任何线程，在任何已经编译了代码的 GPU 上调用已编译的 TVM 函数。
:::

## 生成 OpenCL 代码

TVM 为多个后端提供代码生成功能。可以生成在 CPU 后端运行的 OpenCL 代码或 LLVM 代码。

下面的代码块首先生成 OpenCL 代码，然后在 OpenCL 设备上创建数组，最后验证代码的正确性。

``` python
if tgt.kind.name.startswith("opencl"):
    fadd_cl = tvm.build(s, [A, B, C], tgt, name="myadd")
    print("------opencl code------")
    print(fadd_cl.imported_modules[0].get_source())
    dev = tvm.cl(0)
    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd_cl(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
```

:::note TE 调度原语
TVM 的调度原语包括：
* split：将指定的轴按定义的因子拆分为两个轴。
* tile：通过定义的因子将计算拆分到两个轴上。
* fuse：将一个计算的两个连续轴融合。
* reorder：可以将计算的轴重新排序为定义的顺序。
* bind：可以将计算绑定到特定线程，在 GPU 编程中很有用。
* compute_at：TVM 默认将在函数的最外层或根部计算张量。 compute_at 指定应该在另一个算子的第一个计算轴上计算一个张量。
* compute_inline：当标记为 inline 时，计算将被扩展，然后插入到需要张量的地址中。
* compute_root：将计算移动到函数的最外层或根部。这意味着当前阶段的计算完全完成后才可进入下一阶段。

原语的完整描述参考 [Schedule Primitives](https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html#schedule-primitives) 文档。
:::

## 示例 2：使用 TE 手动优化矩阵乘法

现在来看第二个更高级的示例，演示了 TVM 如何仅使用 18 行 Python 代码，将常见的矩阵乘法运算速度提高 18 倍。

**矩阵乘法是计算密集型运算。为取得良好的 CPU 性能，有两个重要的优化：**

1. 提高内存访问的缓存命中率。高缓存命中率可以加速复杂的数值计算和热点内存访问。这需要将原始内存访问模式转换为适合缓存策略的模式。
2. SIMD（单指令多数据），又称向量处理单元。在每个循环中，SIMD 可以处理一小批数据，而不是处理单个值。这需要将循环体中的数据访问模式转换为统一模式，以便 LLVM 后端可将其降低到 SIMD。

本教程使用的技术出自此 [仓库](https://github.com/flame/how-to-optimize-gemm)。其中一些已经自动被 TVM 抽象地应用，但另一些由于 TVM 的限制而无法自动应用。

### 准备和性能基线

首先收集有关矩阵乘法的 numpy 实现的性能数据：

``` python
import tvm
import tvm.testing
from tvm import te
import numpy

# 矩阵的大小
# (M, K) x (K, N)
# 可尝试不同的 shape，TVM 优化的性能有时比 numpy + MKL 更好
M = 1024
K = 1024
N = 1024

# TVM 默认张量数据类型
dtype = "float32"

# 你可能想调整 target 使其和你的任何 CPU 向量扩展匹配
# 例如，如果你为 SIMD 用的是 Intel AVX2（高级向量扩展）ISA，把下面这行换成 `llvm -mcpu=core-avx2` 可以取得最佳性能（或者你所用 CPU 的具体类型）
# 记住你用的是 llvm, 可以用 `llc --version` 命令来获取 CPU 类型，也可以查看 `/proc/cpuinfo` 来获取你处理器支持的更多扩展

target = tvm.target.Target(target="llvm", host="llvm")
dev = tvm.device(target.kind.name, 0)

# 为测试随机生成的张量
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

# 重复执行矩阵乘法以获得默认 numpy 实现的性能基线
np_repeat = 100
np_running_time = timeit.timeit(
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
print("Numpy running time: %f" % (np_running_time / np_repeat))

answer = numpy.dot(a.numpy(), b.numpy())
```

输出结果：

``` plain
Numpy running time: 0.019016
```

用 TVM TE 编写一个基本的矩阵乘法，并验证它是否产生与 numpy 实现相同的结果。再编写一个函数来辅助评估调度优化的性能：

``` python
# 用 TE 的 TVM 矩阵乘法
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

# 默认 schedule
s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name="mmult")

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

def evaluate_operation(s, vars, target, name, optimization, log):
    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))
    log.append((optimization, mean_time))

log = []

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="none", log=log)
```

输出结果：

``` plain
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
none: 3.256676
```

查看用 TVM 底层函数的算子和默认调度的中间表示。注意，该实现本质上是矩阵乘法的简单实现，在 A 和 B 矩阵的索引上使用三个嵌套循环。

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
  for (x: int32, 0, 1024) {
    for (y: int32, 0, 1024) {
      C[((x*1024) + y)] = 0f32
      for (k: int32, 0, 1024) {
        let cse_var_2: int32 = (x*1024)
        let cse_var_1: int32 = (cse_var_2 + y)
        C[cse_var_1] = (C[cse_var_1] + (A[(cse_var_2 + k)]*B[((k*1024) + y)]))
      }
    }
  }
}
```

### 优化一：块操作

提高缓存命中率的一个重要技巧是块操作，可以在其中构造内存访问，使块内部是一个具有高内存局部性的小邻域。本教程选择一个 32 的块因子，使得一个块将填充 32*32*sizeof(float) 的内存区域。这对应于 4KB 的缓存大小，而 L1 缓存的参考缓存大小为 32 KB。

首先为 `C` 操作创建一个默认 schedule，然后用指定的块因子对其应用 `tile` 调度原语，调度原语返回向量 `[x_outer, y_outer, x_inner, y_inner]`（从最外层到最内层的结果循环的顺序）。然后得到操作输出的归约轴，并用因子 4 对其执行拆分操作。这个因子并不直接影响现在正在处理的块优化，但在应用向量化时很有用。

既然操作已经块级化了，可对计算进行重新排序，将归约操作放到计算的最外层循环中，保证块数据保留在缓存中。完成 schedule 后，就可以构建和测试与原始 schedule 相比的性能。

``` bash
bn = 32

# 通过循环切分实现块级化
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# 将归约域提升到块循环外
s[C].reorder(xo, yo, ko, ki, xi, yi)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="blocking", log=log)
```

输出结果：

``` plain
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
blocking: 0.297447
```

利用缓存对计算重新排序，可发现计算性能的显着提高。现在，打印内部表示，并将其与原始表示进行比较：

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
  for (x.outer: int32, 0, 32) {
    for (y.outer: int32, 0, 32) {
      for (x.inner.init: int32, 0, 32) {
        for (y.inner.init: int32, 0, 32) {
          C[((((x.outer*32768) + (x.inner.init*1024)) + (y.outer*32)) + y.inner.init)] = 0f32
        }
      }
      for (k.outer: int32, 0, 256) {
        for (k.inner: int32, 0, 4) {
          for (x.inner: int32, 0, 32) {
            for (y.inner: int32, 0, 32) {
              let cse_var_3: int32 = (y.outer*32)
              let cse_var_2: int32 = ((x.outer*32768) + (x.inner*1024))
              let cse_var_1: int32 = ((cse_var_2 + cse_var_3) + y.inner)
              C[cse_var_1] = (C[cse_var_1] + (A[((cse_var_2 + (k.outer*4)) + k.inner)]*B[((((k.outer*4096) + (k.inner*1024)) + cse_var_3) + y.inner)]))
            }
          }
        }
      }
    }
  }
}
```

### 优化二：向量化

另一个重要的优化技巧是向量化。当内存访问模式一致时，编译器可以检测到这种模式，并将连续内存传递给 SIMD 向量处理器。利用 TVM 中这个硬件特性，可用 `vectorize` 接口来提示编译器这个模式。

本教程选择对内部循环行数据进行向量化，因为它在之前的优化中已经可以很好地缓存了。

``` python
# Apply the vectorization optimization
s[C].vectorize(yi)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="vectorization", log=log)

# The generalized IR after vectorization
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
vectorization: 0.332722
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (x.outer: int32, 0, 32) {
    for (y.outer: int32, 0, 32) {
      for (x.inner.init: int32, 0, 32) {
        C[ramp((((x.outer*32768) + (x.inner.init*1024)) + (y.outer*32)), 1, 32)] = broadcast(0f32, 32)
      }
      for (k.outer: int32, 0, 256) {
        for (k.inner: int32, 0, 4) {
          for (x.inner: int32, 0, 32) {
            let cse_var_3: int32 = (y.outer*32)
            let cse_var_2: int32 = ((x.outer*32768) + (x.inner*1024))
            let cse_var_1: int32 = (cse_var_2 + cse_var_3)
            C[ramp(cse_var_1, 1, 32)] = (C[ramp(cse_var_1, 1, 32)] + (broadcast(A[((cse_var_2 + (k.outer*4)) + k.inner)], 32)*B[ramp((((k.outer*4096) + (k.inner*1024)) + cse_var_3), 1, 32)]))
          }
        }
      }
    }
  }
}
```

### 优化三：循环置换

查看上面的 IR，可看到内部循环行数据被向量化，并且 B 被转换为 PackedB（通过内部循环的 *(float32x32*)B2* 部分可明显看出）。 PackedB 的遍历现在是顺序的。在当前 schedule 中，A 是逐列访问的，这对缓存不利。如果我们改变 *ki* 和内轴 *xi* 的嵌套循环顺序，A 矩阵的访问模式将更利于缓存。

``` python
s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# re-ordering
# 重新排序
s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

evaluate_operation(
    s, [A, B, C], target=target, name="mmult", optimization="loop permutation", log=log
)

# Again, print the new generalized IR
# 再次打印新生成的 IR
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
loop permutation: 0.114844
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (x.outer: int32, 0, 32) {
    for (y.outer: int32, 0, 32) {
      for (x.inner.init: int32, 0, 32) {
        C[ramp((((x.outer*32768) + (x.inner.init*1024)) + (y.outer*32)), 1, 32)] = broadcast(0f32, 32)
      }
      for (k.outer: int32, 0, 256) {
        for (x.inner: int32, 0, 32) {
          for (k.inner: int32, 0, 4) {
            let cse_var_3: int32 = (y.outer*32)
            let cse_var_2: int32 = ((x.outer*32768) + (x.inner*1024))
            let cse_var_1: int32 = (cse_var_2 + cse_var_3)
            C[ramp(cse_var_1, 1, 32)] = (C[ramp(cse_var_1, 1, 32)] + (broadcast(A[((cse_var_2 + (k.outer*4)) + k.inner)], 32)*B[ramp((((k.outer*4096) + (k.inner*1024)) + cse_var_3), 1, 32)]))
          }
        }
      }
    }
  }
}
```

### 优化四：数组打包

另一个重要的技巧是数组打包。它对数组的存储维度进行重新排序，将某个维度上的连续访问模式转换为展开后的顺序模式。

 ![https://github.com/dmlc/web-data/raw/main/tvm/tutorial/array-packing.png](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/array-packing.png)

如上图所示，将计算块级化（block）后，可以看到 B 的数组访问模式（展开后）是有规律但不连续的。我们期望经过一些转换后，可得到一个持续访问模式。通过将 `[16] [16]` 数组重新排序为 `[16/4] [16][4]` 数组，当从打包数组中获取相应值时，B 的访问模式是顺序的。

考虑到 B 的新包装，必须从一个新的默认 schedule 开始来实现这一点。值得讨论的是：TE 是一种用于编写优化算子的强大且富有表现力的语言，但它通常需要一些关于你正在编写的底层算法、数据结构和硬件目标的知识。本教程后面将讨论，如何借助 TVM 完成部分任务。下面继续新的优化 schedule：

``` bash
# We have to re-write the algorithm slightly.
# 我们必须稍作改动以重写算法。
packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
C = te.compute(
    (M, N),
    lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
    name="C",
)

s = te.create_schedule(C.op)

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="array packing", log=log)

# Here is the generated IR after array packing.
# 数组打包后生成的 IR。
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
array packing: 0.106744
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  allocate(packedB: Pointer(global float32x32), float32x32, [32768]), storage_scope = global {
    for (x: int32, 0, 32) "parallel" {
      for (y: int32, 0, 1024) {
        packedB_1: Buffer(packedB, float32x32, [32768], [])[((x*1024) + y)] = B[ramp(((y*1024) + (x*32)), 1, 32)]
      }
    }
    for (x.outer: int32, 0, 32) {
      for (y.outer: int32, 0, 32) {
        for (x.inner.init: int32, 0, 32) {
          C[ramp((((x.outer*32768) + (x.inner.init*1024)) + (y.outer*32)), 1, 32)] = broadcast(0f32, 32)
        }
        for (k.outer: int32, 0, 256) {
          for (x.inner: int32, 0, 32) {
            for (k.inner: int32, 0, 4) {
              let cse_var_3: int32 = ((x.outer*32768) + (x.inner*1024))
              let cse_var_2: int32 = (k.outer*4)
              let cse_var_1: int32 = (cse_var_3 + (y.outer*32))
              C[ramp(cse_var_1, 1, 32)] = (C[ramp(cse_var_1, 1, 32)] + (broadcast(A[((cse_var_3 + cse_var_2) + k.inner)], 32)*packedB_1[(((y.outer*1024) + cse_var_2) + k.inner)]))
            }
          }
        }
      }
    }
  }
}
```

### 优化五：通过缓存优化块写入

到目前为止，所有的优化都集中在有效地访问和计算来自 A 和 B 矩阵的数据，从而计算 C 矩阵。分块优化后，算子会逐块将结果写入C，访问模式不是顺序的。可用顺序缓存数组来解决这个问题，用 cache_write、compute_at 和 unroll 的组合来保存块结果，并在所有块结果准备好时写入 C。

``` python
s = te.create_schedule(C.op)

# Allocate write cache
# 分配写缓存
CC = s.cache_write(C, "global")

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

# Write cache is computed at yo
# 写缓存在 yo 处被计算
s[CC].compute_at(s[C], yo)

# New inner axes
# 新的内部轴
xc, yc = s[CC].op.axis

(k,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(k, factor=4)
s[CC].reorder(ko, xc, ki, yc)
s[CC].unroll(ki)
s[CC].vectorize(yc)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="block caching", log=log)

# Here is the generated IR after write cache blocking.
# 写缓存块级化后生成的 IR。
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
block caching: 0.108552
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  allocate(packedB: Pointer(global float32x32), float32x32, [32768]), storage_scope = global;
  allocate(C.global: Pointer(global float32), float32, [1024]), storage_scope = global {
    for (x: int32, 0, 32) "parallel" {
      for (y: int32, 0, 1024) {
        packedB_1: Buffer(packedB, float32x32, [32768], [])[((x*1024) + y)] = B[ramp(((y*1024) + (x*32)), 1, 32)]
      }
    }
    for (x.outer: int32, 0, 32) {
      for (y.outer: int32, 0, 32) {
        for (x.c.init: int32, 0, 32) {
          C.global_1: Buffer(C.global, float32, [1024], [])[ramp((x.c.init*32), 1, 32)] = broadcast(0f32, 32)
        }
        for (k.outer: int32, 0, 256) {
          for (x.c: int32, 0, 32) {
            let cse_var_4: int32 = (k.outer*4)
            let cse_var_3: int32 = (x.c*32)
            let cse_var_2: int32 = ((y.outer*1024) + cse_var_4)
            let cse_var_1: int32 = (((x.outer*32768) + (x.c*1024)) + cse_var_4)
             {
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[cse_var_1], 32)*packedB_1[cse_var_2]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 1)], 32)*packedB_1[(cse_var_2 + 1)]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 2)], 32)*packedB_1[(cse_var_2 + 2)]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 3)], 32)*packedB_1[(cse_var_2 + 3)]))
            }
          }
        }
        for (x.inner: int32, 0, 32) {
          for (y.inner: int32, 0, 32) {
            C[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] = C.global_1[((x.inner*32) + y.inner)]
          }
        }
      }
    }
  }
}
```

### 优化六：并行化

到目前为止，仅设计了用单核来计算。几乎所有现代处理器都有多个内核，计算可以从并行计算中受益。最后的优化将利用线程级并行（thread-level parallelization）。

``` python
# parallel
# 并行化
s[C].parallel(xo)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(
    s, [A, B, C], target=target, name="mmult", optimization="parallelization", log=log
)

# Here is the generated IR after parallelization.
# 并行化后生成的 IR。
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
parallelization: 0.141811
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  allocate(packedB: Pointer(global float32x32), float32x32, [32768]), storage_scope = global {
    for (x: int32, 0, 32) "parallel" {
      for (y: int32, 0, 1024) {
        packedB_1: Buffer(packedB, float32x32, [32768], [])[((x*1024) + y)] = B[ramp(((y*1024) + (x*32)), 1, 32)]
      }
    }
    for (x.outer: int32, 0, 32) "parallel" {
      allocate(C.global: Pointer(global float32), float32, [1024]), storage_scope = global;
      for (y.outer: int32, 0, 32) {
        for (x.c.init: int32, 0, 32) {
          C.global_1: Buffer(C.global, float32, [1024], [])[ramp((x.c.init*32), 1, 32)] = broadcast(0f32, 32)
        }
        for (k.outer: int32, 0, 256) {
          for (x.c: int32, 0, 32) {
            let cse_var_4: int32 = (k.outer*4)
            let cse_var_3: int32 = (x.c*32)
            let cse_var_2: int32 = ((y.outer*1024) + cse_var_4)
            let cse_var_1: int32 = (((x.outer*32768) + (x.c*1024)) + cse_var_4)
             {
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[cse_var_1], 32)*packedB_1[cse_var_2]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 1)], 32)*packedB_1[(cse_var_2 + 1)]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 2)], 32)*packedB_1[(cse_var_2 + 2)]))
              C.global_1[ramp(cse_var_3, 1, 32)] = (C.global_1[ramp(cse_var_3, 1, 32)] + (broadcast(A[(cse_var_1 + 3)], 32)*packedB_1[(cse_var_2 + 3)]))
            }
          }
        }
        for (x.inner: int32, 0, 32) {
          for (y.inner: int32, 0, 32) {
            C[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] = C.global_1[((x.inner*32) + y.inner)]
          }
        }
      }
    }
  }
}
```

### 矩阵乘法示例总结

仅用 18 行代码应用上述简单优化后，生成的代码开始接近带有数学内核库（MKL）的 numpy 的性能。由于一直在记录性能，因此可比较结果：

``` python
baseline = log[0][1]
print("%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20)))
for result in log:
    print(
        "%s\t%s\t%s"
        % (result[0].rjust(20), str(result[1]).rjust(20), str(result[1] / baseline).rjust(20))
    )
```

输出结果：

``` plain
        Operator                  Timing             Performance
            none      3.2566761794000003                     1.0
        blocking     0.29744742350000003     0.09133466366152523
   vectorization     0.33272212060000006     0.10216616644437143
loop permutation     0.11484386070000001     0.03526413262283832
   array packing     0.10674374140000001    0.032776897523678926
   block caching            0.1085523429     0.03333224948388923
 parallelization            0.1418105982     0.04354458054411991
```

注意，网页上的输出反映的是非专用 Docker 容器上的运行时间，这是不可靠的。强烈推荐你运行本教程，观察 TVM 获得的性能提升，并仔细研究每个示例，来了解对矩阵乘法运算所做的迭代改进。

## 总结

如前所述，如何使用 TE 和调度原语来应用优化，需要一些底层架构和算法的知识。但是，TE 是能搜索潜在优化的、更复杂算法的基础。学完本章节对 TE 的介绍，现在可以开始探索 TVM 如何自动化调度优化过程。

本教程用向量加法和矩阵乘法这两个示例讲解了 TVM 张量表达式（TE）的工作流程。一般工作流程如下：

* 通过一系列操作描述计算。
* 描述如何用调度原语进行计算。
* 编译成想要的目标函数。
* 保存之后要加载的函数（可选）。
  接下来的教程将详细介绍矩阵乘法示例，并展示如何用可调参数构建矩阵乘法和其他操作的通用模板，从而自动优化特定平台的计算。

[下载 Python 源代码：tensor_expr_get_started.py](https://tvm.apache.org/docs/_downloads/40a01cffb015a67aaec0fad7e27cf80d/tensor_expr_get_started.py)

[下载 Jupyter Notebook：tensor_expr_get_started.ipynb](https://tvm.apache.org/docs/_downloads/4459ebf5b03d332f7f380abdaef81c05/tensor_expr_get_started.ipynb)
