---
title: 使用 Auto-scheduling 优化算子
---

# 使用 Auto-scheduling 优化算子

:::note
单击 [此处](https://tvm.apache.org/docs/tutorial/auto_scheduler_matmul_x86.html#sphx-glr-download-tutorial-auto-scheduler-matmul-x86-py) 下载完整的示例代码
:::

**作者**：[Lianmin Zheng](https://github.com/merrymercy)，[Chengfan Jia](https://github.com/jcf94/)

本教程将展示 TVM 的 Auto Scheduling 功能，如何在不编写自定义模板的情况下，找到最佳 schedule。

与基于模板的 [AutoTVM](https://tvm.apache.org/docs/tutorial/autotvm_matmul_x86.html) 依赖手工模板来定义搜索空间不同，auto-scheduler 不需要任何模板。用户只需编写计算声明，无需任何 schedule 命令或模板。auto-scheduler 可以自动生成一个大的搜索空间，并在空间中找到最优 schedule。

本教程中使用矩阵乘法作为示例。

:::note
注意，本教程不会在 Windows 或最新版本的 macOS 上运行。如需运行，请将本教程的主体放在 `if __name__ == "__main__":` 代码块中。
:::

``` python
import os

import numpy as np
import tvm
from tvm import te, auto_scheduler
```

## 定义矩阵乘法

首先，定义一个带有偏置加法的矩阵乘法。注意，这里使用了 TVM 张量表达式语言中的标准操作。主要区别在于函数定义上方使用了 `register_workload` 装饰器。该函数应返回输入/输出张量列表。通过这些张量，auto-scheduler 可以得到整个计算图。

``` python
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]
```

## 创建搜索任务

定义函数后，可以为 auto_scheduler 创建要搜索的任务。我们为这个矩阵乘法指定了特定的参数，如这里是两个大小为 1024x1024 的矩阵乘法。然后我们创建一个 N=L=M=1024 和 dtype="float32" 的搜索任务

:::note 使用自定义 target 提高性能
为让 TVM 充分利用特定的硬件平台，需要手动指定 CPU 功能。例如：

* 启用 AVX2：将下面的 `llvm` 替换为 `llvm -mcpu=core-avx2`
* 启用 AVX-512：将下面的 `llvm` 替换为 `llvm -mcpu=skylake-avx512`
:::

``` python
target = tvm.target.Target("llvm")
N = L = M = 1024
task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(N, L, M, "float32"), target=target)

# 检查计算图
print("Computational DAG:")
print(task.compute_dag)
```

输出结果：

``` bash
Computational DAG:
A = PLACEHOLDER [1024, 1024]
B = PLACEHOLDER [1024, 1024]
matmul(i, j) += (A[i, k]*B[k, j])
C = PLACEHOLDER [1024, 1024]
out(i, j) = (matmul[i, j] + C[i, j])
```

## 设置 auto-scheduler 的参数

接下来，为 auto-scheduler 设置参数。

* `num_measure_trials` 表示搜索过程中可用的测试试验次数。本教程为了快速演示只进行了 10 次试验。实际上，1000 是搜索收敛的最佳数量。可以根据自己的时间预算进行更多试验。
* 此外，我们用 `RecordToFile` 将测试记录记录到文件 `matmul.json` 中。测试记录可用于查询历史最佳、恢复搜索以及以后进行更多分析。
* 有关更多参数，参见 `TuningOptions`

``` python
log_file = "matmul.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
```

## 开始搜索

准备好所有输入就可以开始搜索，让 auto-scheduler 发挥它的作用。经过一些测试试验后，可从日志文件中加载最佳  schedule 并应用。

``` python
# 运行 auto-tuning (搜索)
task.tune(tune_option)
# 应用最佳 schedule
sch, args = task.apply_best(log_file)
```

## 检查优化的 schedule

auto-scheduling 完成后，可将 schedule 降级来查看 IR。auto-scheduler 执行合适的优化，包括多级循环切分、布局转换、并行化、向量化、展开和算子融合。

``` python
print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))
```

输出结果：

``` bash
Lowered TIR:
@main = primfn(A_1: handle, B_1: handle, C_1: handle, out_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], []),
             out: Buffer(out_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C, out_1: out}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], []), out_1: out_3: Buffer(out_2, float32, [1024, 1024], [])} {
  allocate(auto_scheduler_layout_transform: Pointer(global float32), float32, [1048576]), storage_scope = global {
    for (ax0.ax1.fused.ax2.fused: int32, 0, 128) "parallel" {
      for (ax4: int32, 0, 256) {
        for (ax6: int32, 0, 4) {
          for (ax7: int32, 0, 8) {
            auto_scheduler_layout_transform_1: Buffer(auto_scheduler_layout_transform, float32, [1048576], [])[((((ax0.ax1.fused.ax2.fused*8192) + (ax4*32)) + (ax6*8)) + ax7)] = B[((((ax4*4096) + (ax6*1024)) + (ax0.ax1.fused.ax2.fused*8)) + ax7)]
          }
        }
      }
    }
    for (i.outer.outer.j.outer.outer.fused: int32, 0, 16384) "parallel" {
      allocate(matmul: Pointer(global float32x8), float32x8, [4]), storage_scope = global;
      for (i.outer.inner: int32, 0, 2) {
        matmul_1: Buffer(matmul, float32x8, [4], [])[0] = broadcast(0f32, 8)
        matmul_1[1] = broadcast(0f32, 8)
        matmul_1[2] = broadcast(0f32, 8)
        matmul_1[3] = broadcast(0f32, 8)
        for (k.outer: int32, 0, 256) {
          for (k.inner: int32, 0, 4) {
            let cse_var_2: int32 = (((floormod(i.outer.outer.j.outer.outer.fused, 128)*8192) + (k.outer*32)) + (k.inner*8))
            let cse_var_1: int32 = ((((floordiv(i.outer.outer.j.outer.outer.fused, 128)*8192) + (i.outer.inner*4096)) + (k.outer*4)) + k.inner)
             {
              matmul_1[0] = (matmul_1[0] + (broadcast(A[cse_var_1], 8)*auto_scheduler_layout_transform_1[ramp(cse_var_2, 1, 8)]))
              matmul_1[1] = (matmul_1[1] + (broadcast(A[(cse_var_1 + 1024)], 8)*auto_scheduler_layout_transform_1[ramp(cse_var_2, 1, 8)]))
              matmul_1[2] = (matmul_1[2] + (broadcast(A[(cse_var_1 + 2048)], 8)*auto_scheduler_layout_transform_1[ramp(cse_var_2, 1, 8)]))
              matmul_1[3] = (matmul_1[3] + (broadcast(A[(cse_var_1 + 3072)], 8)*auto_scheduler_layout_transform_1[ramp(cse_var_2, 1, 8)]))
            }
          }
        }
        for (i.inner: int32, 0, 4) {
          let cse_var_3: int32 = ((((floordiv(i.outer.outer.j.outer.outer.fused, 128)*8192) + (i.outer.inner*4096)) + (i.inner*1024)) + (floormod(i.outer.outer.j.outer.outer.fused, 128)*8))
          out[ramp(cse_var_3, 1, 8)] = (matmul_1[i.inner] + C[ramp(cse_var_3, 1, 8)])
        }
      }
    }
  }
}
```

## 检查正确性并评估性能

构建二进制文件并检查其正确性和性能。

``` python
func = tvm.build(sch, args, target)
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = np.random.uniform(size=(N, M)).astype(np.float32)
out_np = a_np.dot(b_np) + c_np

dev = tvm.cpu()
a_tvm = tvm.nd.array(a_np, device=dev)
b_tvm = tvm.nd.array(b_np, device=dev)
c_tvm = tvm.nd.array(c_np, device=dev)
out_tvm = tvm.nd.empty(out_np.shape, device=dev)
func(a_tvm, b_tvm, c_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
)
```

输出结果：

``` bash
Execution time of this operator: 93.286 ms
```

## 使用记录文件

在搜索过程中，所有测试记录都保存到记录文件 `matmul.json` 中。测试记录可用于重新应用搜索结果、恢复搜索和执行其他分析。

下面是从文件中加载最佳 schedule，并打印等效的 Python schedule API 的例子。可用于调试和学习 auto-scheduler 的行为。

``` python
print("Equivalent python schedule:")
print(task.print_best(log_file))
```

输出结果：

``` bash
Equivalent python schedule:
matmul_i, matmul_j, matmul_k = tuple(matmul.op.axis) + tuple(matmul.op.reduce_axis)
out_i, out_j = tuple(out.op.axis) + tuple(out.op.reduce_axis)
matmul_i_o_i, matmul_i_i = s[matmul].split(matmul_i, factor=4)
matmul_i_o_o_i, matmul_i_o_i = s[matmul].split(matmul_i_o_i, factor=1)
matmul_i_o_o_o, matmul_i_o_o_i = s[matmul].split(matmul_i_o_o_i, factor=2)
matmul_j_o_i, matmul_j_i = s[matmul].split(matmul_j, factor=8)
matmul_j_o_o_i, matmul_j_o_i = s[matmul].split(matmul_j_o_i, factor=1)
matmul_j_o_o_o, matmul_j_o_o_i = s[matmul].split(matmul_j_o_o_i, factor=1)
matmul_k_o, matmul_k_i = s[matmul].split(matmul_k, factor=4)
s[matmul].reorder(matmul_i_o_o_o, matmul_j_o_o_o, matmul_i_o_o_i, matmul_j_o_o_i, matmul_k_o, matmul_i_o_i, matmul_j_o_i, matmul_k_i, matmul_i_i, matmul_j_i)
out_i_o_i, out_i_i = s[out].split(out_i, factor=4)
out_i_o_o, out_i_o_i = s[out].split(out_i_o_i, factor=2)
out_j_o_i, out_j_i = s[out].split(out_j, factor=8)
out_j_o_o, out_j_o_i = s[out].split(out_j_o_i, factor=1)
s[out].reorder(out_i_o_o, out_j_o_o, out_i_o_i, out_j_o_i, out_i_i, out_j_i)
s[matmul].compute_at(s[out], out_j_o_i)
out_i_o_o_j_o_o_fused = s[out].fuse(out_i_o_o, out_j_o_o)
s[out].parallel(out_i_o_o_j_o_o_fused)
s[matmul].pragma(matmul_i_o_o_o, "auto_unroll_max_step", 8)
s[matmul].pragma(matmul_i_o_o_o, "unroll_explicit", True)
s[matmul].vectorize(matmul_j_i)
s[out].vectorize(out_j_i)
```

恢复搜索则更为复杂，需要自己创建搜索策略和 cost model，并通过日志文件恢复搜索策略和 cost model 的状态。下面的示例进行了 5 次试验来恢复它们的状态：

``` python
def resume_search(task, log_file):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5, measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    task.tune(tune_option, search_policy=search_policy)

resume_search(task, log_file)
```

输出结果：

``` bash
Resume search:
/usr/local/lib/python3.7/dist-packages/xgboost/training.py:17: UserWarning: Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
  warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
```

## 最后的说明和总结

本教程展示了如何在不指定搜索模板的情况下，使用 TVM Auto-Scheduler 自动优化矩阵乘法。从张量表达式 (TE) 语言开始，演示了一系列关于 TVM 如何优化计算操作的示例。

[下载 Python 源代码：auto_scheduler_matmul_x86.py](https://tvm.apache.org/docs/_downloads/eac4389b114db015e95cb3cdf8b86b83/auto_scheduler_matmul_x86.py)

[下载 Jupyter Notebook：auto_scheduler_matmul_x86.ipynb](https://tvm.apache.org/docs/_downloads/246d4b8509474fd9046e69f6cc9b7f87/auto_scheduler_matmul_x86.ipynb)
