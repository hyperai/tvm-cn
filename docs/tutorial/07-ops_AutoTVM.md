---
title: 用 Schedule 模板和 AutoTVM 优化算子
---

# 用 Schedule 模板和 AutoTVM 优化算子

:::note
注意：单击 [此处](https://tvm.apache.org/docs/tutorial/autotvm_matmul_x86.html#sphx-glr-download-tutorial-autotvm-matmul-x86-py) 下载完整的示例代码
:::

**作者**：[Lianmin Zheng](https://github.com/merrymercy)，[Chris Hoge](https://github.com/hogepodge)

本教程将展示如何用 TVM 张量表达式（TE）语言编写 schedule 模板，并通过 AutoTVM 对模板进行搜索，从而找到最佳 schedule。这个自动优化张量计算的过程被称为 Auto-Tuning。

本教程基于前面的 [TE 编写矩阵乘法教程](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html) 设立。

auto-tuning 包括两个步骤：

* 第一步：定义搜索空间。
* 第二步：运行搜索算法来探索这个空间。

通过本教程可以了解如何在 TVM 中执行这两个步骤。整个工作流程由一个矩阵乘法示例来说明。

:::note
注意，本教程不会在 Windows 或最新版本的 macOS 上运行。如需运行，请将本教程的主体放在 `if __name__ == "__main__":` 代码块中。
:::

## 安装依赖

要在 TVM 中使用 autotvm 包，需安装一些额外的依赖。

``` bash
pip3 install --user psutil xgboost cloudpickle
```

为了让 TVM 在调优过程中运行更快，建议使用 Cython 作为 TVM 的 FFI。在 TVM 的根目录下，执行：

``` bash
pip3 install --user cython
sudo make cython3
```

现在我们一起来看如何用 Python 代码实现。首先导入所需的包：

``` python
import logging
import sys

import numpy as np
import tvm
from tvm import te
import tvm.testing

# 模块名叫 `autotvm`
from tvm import autotvm
```

## TE 的基本矩阵乘法

回想一下用 TE 进行矩阵乘法的基本实现，下面做一些改变。将矩阵乘法放在 Python 函数定义中。简单起见，重点关注拆分的优化，将重新排序的块大小设为固定值。

``` python
def matmul_basic(N, L, M, dtype):

    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # 调度
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    yo, yi = s[C].split(y, 8)
    xo, xi = s[C].split(x, 8)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]
```

## 用 AutoTVM 进行矩阵乘法

前面的调度代码用常量“8”作为循环切分因子，但是它可能不是最佳的。因为最佳的循环切分因子取决于真实的硬件环境和输入 shape。

如果希望调度代码能够在更广泛的输入 shape 和目标硬件上可移植，最好定义一组候选值，并根据目标硬件上的评估结果选择最佳值。

autotvm 中可以为这种值定义一个可调参数，或者一个 "knob"。

## 基本矩阵乘法模板

以下示例将演示，如何为 *split* 调度操作的 block 大小创建一个可调的参数集。

``` python
# Matmul V1: 列出候选值
@autotvm.template("tutorial/matmul_v1")  # 1. 使用装饰器
def matmul_v1(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # 调度
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # 2. 获取 config 对象
    cfg = autotvm.get_config()

    # 3. 定义搜索空间
    cfg.define_knob("tile_y", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_x", [1, 2, 4, 8, 16])

    # 4. 根据 config 进行调度
    yo, yi = s[C].split(y, cfg["tile_y"].val)
    xo, xi = s[C].split(x, cfg["tile_x"].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]
```

下面将对前面的调度代码作出四个修改，然后得到一个可调的“模板”。一一解释这些修改：

1. 使用装饰器将此函数标记为简单模板。
2. 获取 config 对象：将 `cfg` 视为此函数的参数，但我们以另外的方式获取它。cfg 参数使得这个函数不再是一个确定的 schedule。将不同的配置传递给这个函数，可以得到不同的 schedule。这种使用配置对象的函数称为“模板”。

   为使模板函数更精炼，可在单个函数中定义参数搜索空间：
   1. 用一组值来定义搜索空间。将 `cfg` 转为 `ConfigSpace` 对象，收集此函数中的所有可调 knob，然后从中构建一个搜索空间。
   2. 根据空间中的实体进行调度。将 `cfg` 转为 `ConfigEntity` 对象，当它被转为 `ConfigEntity` 后，会忽略所有空间定义 API（即 `cfg.define_XXXXX(...)`），但会存储所有可调 knob 的确定值，并根据这些值进行调度。

   在 auto-tuning 的过程中，首先用 `ConfigSpace` 对象调用这个模板来构建搜索空间，然后在构建的空间中用不同的 `ConfigEntity` 调用这个模板，来得到不同的 schedule。最后，我们将评估由不同 schedule 生成的代码，然后选择最佳的 schedule。
4. 定义两个可调 knob。第一个是 `tile_y`，它有 5 个可能值。第二个是 `tile_x`，它和前者具有相同的可能值。这两个 knob 是独立的，所以它们跨越大小为 25 = 5x5 的搜索空间。
5. 配置 knob 被传递给 `split` 调度操作，然后可以根据之前在 `cfg` 中定义的 5x5 确定值进行调度。

## 带有高级参数 API 的矩阵乘法模板

前面的模板手动列出了 konb 的所有可能值，它是用来定义空间的最底层 API，显示列出了要搜索的参数空间。这里推荐使用另一组更高级的 API，它可以更简单、更智能地定义搜索空间。

下面的示例用 `ConfigSpace.define_split` 来定义拆分 knob。它列举了所有可能的拆分 axis 和构造空间的方法。

同时，`ConfigSpace.define_reorder` 用于对 knob 重新排序，`ConfigSpace.define_annotate` 用于对展开、向量化、线程绑定等进行注释 。当高级 API 无法满足你的需求时，可以回退使用底层 API。

``` python
@autotvm.template("tutorial/matmul")
def matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # 调度
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### 开始定义控件 #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### 结束定义空间 #####

    # 根据 config 进行调度
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]
```

:::note 关于 `cfg.define_split` 的更多解释
在此模板中，`cfg.define_split("tile_y", y, num_outputs=2)` 枚举了所有可能的组合（以 y 的长度为因子，将 y 轴分成两个轴）。例如，如果 y 的长度为 32 并且想以 32 为因子将它拆分为两个轴，那么（外轴长度，内轴长度）有 6 个可能的值，即 (32, 1)，(16, 2)，(8, 4)，(4, 8)，(2, 16) 或 (1, 32)。这些也是 *tile_y* 的 6 个可能值。

调度过程中，`cfg["tile_y"]` 是一个 `SplitEntity` 对象。我们将外轴和内轴的长度存储在 `cfg['tile_y'].size` （有两个元素的元组）中。这个模板使用 `yo, yi = cfg['tile_y'].apply(s, C, y)` 来应用它。其实等价于 `yo, yi = s[C].split(y, cfg["tile_y"].size[1])` 或 `yo, yi = s[C].split(y, nparts=cfg['tile_y"].size[0])`。

cfg.apply API 的优点是它使多级拆分（即当 num_outputs >= 3 时）变得更加简单。
:::

## 第 2 步：使用 AutoTVM 优化矩阵乘法

第 1 步编写的矩阵乘法模板，可对拆分的 schedule 中的块大小进行参数化。通过第 1 步，可以实现对这个参数空间进行搜索。下一步是选择一个调优器来指导如何对空间进行探索。

### TVM 的自动调优器

调优器的任务可用以下伪代码来描述：

``` python
ct = 0
while ct < max_number_of_trials:
    propose a batch of configs
    measure this batch of configs on real hardware and get results
    ct += batch_size
```

调优器可采取不同的策略来计划下一批配置，包括：

* `tvm.autotvm.tuner.RandomTuner` ：以随机顺序枚举空间
* `tvm.autotvm.tuner.GridSearchTuner` ：以网格搜索顺序枚举空间
* `tvm.autotvm.tuner.GATuner` ：使用遗传算法搜索空间
* `tvm.autotvm.tuner.XGBTuner` ：用基于模型的方法训练一个 XGBoost 模型，来预测降级 IR 的速度，并根据预测值选择下一批配置。

可根据空间大小、时间预算和其他因素来选择调优器。例如，如果你的空间非常小（小于 1000），则网格搜索调优器或随机调优器就够了。如果你的空间在 10^9 级别（CUDA GPU 上的 conv2d 算子的空间大小），XGBoostTuner 可以更有效地探索并找到更好的配置。

### 开始调优

下面继续矩阵乘法的示例。首先创建一个调优任务，然后检查初始的搜索空间。下面示例中是 512x512 的矩阵乘法，空间大小为 10x10=100。注意，任务和搜索空间与选择的调优器无关。

``` python
N, L, M = 512, 512, 512
task = autotvm.task.create("tutorial/matmul", args=(N, L, M, "float32"), target="llvm")
print(task.config_space)
```

输出结果：

``` bash
ConfigSpace (len=100, space_map=
   0 tile_y: Split(policy=factors, product=512, num_outputs=2) len=10
   1 tile_x: Split(policy=factors, product=512, num_outputs=2) len=10
)
```

然后定义如何评估生成的代码，并且选择一个调优器。由于我们的空间很小，所以随机调优器就可以。

本教程只做 10 次试验进行演示。实际上可以根据自己的时间预算进行更多试验。调优结果会记录到日志文件中。这个文件可用于选择之后发现的调优器的最佳配置。

``` python
# 记录 config（为了将 tuning 日志打印到屏幕）
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
```

评估配置有两个步骤：构建和运行。默认用所有 CPU core 来编译程序。然后依次进行评估。为了减少方差，对 5 次评估结果取平均值。

``` python
measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

# 用 RandomTuner 开始调优, 日志记录到 `matmul.log` 文件中
# 可用 XGBTuner 来替代.
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(
    n_trial=10,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("matmul.log")],
)
```

输出结果：

``` bash
waiting for device...
device available
Get devices for measurement successfully!
No: 1   GFLOPS: 8.48/8.48       result: MeasureResult(costs=(0.0316434228,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.638512134552002, timestamp=1657225928.6342561)        [('tile_y', [-1, 1]), ('tile_x', [-1, 256])],None,80
No: 2   GFLOPS: 2.30/8.48       result: MeasureResult(costs=(0.1165478966,), error_no=MeasureErrorNo.NO_ERROR, all_cost=2.0105199813842773, timestamp=1657225930.6636436)       [('tile_y', [-1, 4]), ('tile_x', [-1, 8])],None,32
No: 3   GFLOPS: 11.82/11.82     result: MeasureResult(costs=(0.0227097348,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.5589795112609863, timestamp=1657225931.7059512)       [('tile_y', [-1, 64]), ('tile_x', [-1, 32])],None,56
No: 4   GFLOPS: 1.66/11.82      result: MeasureResult(costs=(0.1616202114,), error_no=MeasureErrorNo.NO_ERROR, all_cost=2.6911513805389404, timestamp=1657225934.9635096)       [('tile_y', [-1, 1]), ('tile_x', [-1, 4])],None,20
No: 5   GFLOPS: 3.65/11.82      result: MeasureResult(costs=(0.073561817,), error_no=MeasureErrorNo.NO_ERROR, all_cost=1.3051848411560059, timestamp=1657225936.3988533)        [('tile_y', [-1, 256]), ('tile_x', [-1, 16])],None,48
No: 6   GFLOPS: 1.85/11.82      result: MeasureResult(costs=(0.1452834464,), error_no=MeasureErrorNo.NO_ERROR, all_cost=2.5179028511047363, timestamp=1657225938.961955)        [('tile_y', [-1, 512]), ('tile_x', [-1, 4])],None,29
No: 7   GFLOPS: 0.87/11.82      result: MeasureResult(costs=(0.30933780240000003,), error_no=MeasureErrorNo.NO_ERROR, all_cost=5.067087888717651, timestamp=1657225944.589149)  [('tile_y', [-1, 512]), ('tile_x', [-1, 2])],None,19
No: 8   GFLOPS: 10.53/11.82     result: MeasureResult(costs=(0.025489421,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.5452830791473389, timestamp=1657225945.1592515)        [('tile_y', [-1, 4]), ('tile_x', [-1, 64])],None,62
No: 9   GFLOPS: 1.58/11.82      result: MeasureResult(costs=(0.16960762680000002,), error_no=MeasureErrorNo.NO_ERROR, all_cost=2.8109781742095947, timestamp=1657225948.0900776)        [('tile_y', [-1, 2]), ('tile_x', [-1, 2])],None,11
No: 10  GFLOPS: 2.42/11.82      result: MeasureResult(costs=(0.11083148779999999,), error_no=MeasureErrorNo.NO_ERROR, all_cost=1.8757600784301758, timestamp=1657225950.0266354)        [('tile_y', [-1, 4]), ('tile_x', [-1, 4])],None,22
```

调优完成后，可从日志文件中选择具有最佳评估性能的配置，并用相应参数来编译 schedule。快速验证 schedule 是否产生了正确的结果，可直接在 `autotvm.apply_history_best` 上下文中调用 `matmul` 函数，它会用参数查询调度上下文，然后可用相同的参数获取最优配置。

``` python
# 从日志文件中应用历史最佳
with autotvm.apply_history_best("matmul.log"):
    with tvm.target.Target("llvm"):
        s, arg_bufs = matmul(N, L, M, "float32")
        func = tvm.build(s, arg_bufs)

# 验证正确性
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)
```

输出结果：

``` plain
Finish loading 10 records
```

## 总结

本教程展示了如何构建算子模板，使得 TVM 能够搜索参数空间，并选择优化的调度配置。为了更深入地了解其工作原理，推荐基于 :ref: *张量表达式入门 <tensor_expr_get_started>_* 教程中演示的调度操作，向调度添加新的搜索参数。接下来的章节将演示 AutoScheduler，它是TVM 中一种优化常用算子的方法，同时无需用户提供自定义的模板。

[下载 Python 源代码：autotvm_matmul_x86.py](https://tvm.apache.org/docs/_downloads/8e7bbc9dbdda76ac573b24606b41c006/autotvm_matmul_x86.py)

[下载 Jupyter Notebook：autotvm_matmul_x86.ipynb](https://tvm.apache.org/docs/_downloads/37bbf9e2065ec8deeb64a8d9fa0755bc/autotvm_matmul_x86.ipynb)
