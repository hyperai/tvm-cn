---
title: 使用自定义调度规则 (Sketch Rule) 在 CPU 上自动调度稀疏矩阵乘法
---

# 使用自定义草图规则 (Sketch Rule) 在 CPU 上自动调度稀疏矩阵乘法

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_sparse_x86.html#sphx-glr-download-how-to-tune-with-autoscheduler-tune-sparse-x86-py) 下载完整的示例代码
:::

**作者**：[Chengfan Jia](https://github.com/jcf94/)

本文介绍如何用 auto-scheduler 来调优 CPU 的稀疏矩阵乘法。

auto-scheduler 旨在自动探索给定计算声明的最佳性能调度。有时需要尝试一些特殊的操作，auto-scheduler 的默认调度规则 (Sketch Rule) 可能不能很好的支持这些操作，会导致性能不佳。auto-scheduler 目前允许用户提供一个 CustomSketch 来覆盖这些情况。

本教程使用稀疏矩阵乘法作为示例，演示如何实现自定义调度规则，并将其插入 auto-scheduler 的搜索策略。

注意，本教程无法在 Windows 或最新版本的 macOS 上运行。如需运行，请将本教程的主体放在 `if __name__ == "__main__":` 代码块中。

``` python
import os
import numpy as np

import tvm
import tvm.testing
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix
```

## 定义计算

首先用几个 relu 和 bias 相加来定义一个稀疏 matmul 的计算，该函数返回输入/输出张量列表，auto-scheduler 可以从这些张量中得到整个计算图。

``` python
@auto_scheduler.register_workload
def sparse_dense(M, N, K, w_data_shape, w_indices_shape, w_indptr_shape, dtype):
    X = te.placeholder(shape=(M, K), dtype=dtype)
    W_data = te.placeholder(shape=w_data_shape, dtype=dtype)
    W_indices = te.placeholder(shape=w_indices_shape, dtype="int32")
    W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32")
    B = te.placeholder(shape=(M, N), dtype=dtype)

    out = topi.nn.sparse_dense(topi.nn.relu(X), W_data, W_indices, W_indptr)
    out = te.compute((M, N), lambda i, j: out[i, j] + B[i, j], name="BiasAdd")
    out = topi.nn.relu(out)

    return [X, W_data, W_indices, W_indptr, B, out]
```

## 稀疏工作负载 (sparse workload) 的特殊步骤

在调度调优期间，auto-scheduler 使用随机输入来测试生成的调度的性能。虽然不能直接使用随机数组作为稀疏运算的输入，但「indices」和「indptr」数组对于计算很有用。

为解决这个问题，将它们注册为特殊的 buffer，然后在测试程序时加载。更多详细信息，参阅 *tvm.auto_scheduler.measure.py* 。

``` python
# 定义稀疏计算的基本 shape
M = 128
K = 256
N = 512
BS_R = 16
BS_C = 1
density = 0.6

# 用 numpy 生成测试数据
X_np = np.random.randn(M, K).astype("float32")
X_np = np.maximum(np.zeros((M, K), dtype="float32"), X_np)  # Relu
W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
W_np = W_sp_np.todense()
Y_np = X_np @ W_np.T  # 处理矩阵乘法
B_np = np.random.randn(M, N).astype("float32")
Y_np = Y_np + B_np  # Bias add
Y_np = np.maximum(np.zeros((M, N), dtype="float32"), Y_np)  # Relu
```

## 创建搜索任务

接下来创建一个搜索任务 M=N=K=512，dtype=「float32」。如果你的机器支持 avx 指令，你可以：

* 将下面的「llvm」替换为「llvm -mcpu=core-avx2」来启用 AVX2
* 将下面的「llvm」替换为「llvm -mcpu=skylake-avx512」来启用 AVX-512

``` python
target = tvm.target.Target("llvm")

# 将稀疏数据注册到任务输入
prefix = "sparse_dense_bsr_%d_%d_%d_%d_%d_%d_" % (
    N,
    K,
    BS_R,
    BS_C,
    W_sp_np.indices.shape[0],
    W_sp_np.indptr.shape[0],
)
task = tvm.auto_scheduler.SearchTask(
    func=sparse_dense,
    args=(M, N, K, W_sp_np.data.shape, W_sp_np.indices.shape, W_sp_np.indptr.shape, "float32"),
    target=target,
    task_inputs={
        prefix + "W_data": runtime.ndarray.array(W_sp_np.data),
        prefix + "W_indices": runtime.ndarray.array(W_sp_np.indices),
        prefix + "W_indptr": runtime.ndarray.array(W_sp_np.indptr),
    },
    task_inputs_save_to_file=True,
)

# 检查计算图
print("Computational DAG:")
print(task.compute_dag)
```

输出结果：

``` bash
Computational DAG:
placeholder = PLACEHOLDER [33]
placeholder = PLACEHOLDER [4916, 16, 1]
placeholder = PLACEHOLDER [4916]
placeholder = PLACEHOLDER [128, 256]
compute(i0, i1) = max(placeholder[i0, i1], 0f)
compute(i, nb_j, j) += (placeholder[(placeholder[nb_j] + elem_idx), j, c]*compute[i, (placeholder[(placeholder[nb_j] + elem_idx)] + c)])
compute(m, n) = compute[m, floordiv(n, 16), floormod(n, 16)]
placeholder = PLACEHOLDER [128, 512]
BiasAdd(i, j) = (compute[i, j] + placeholder[i, j])
compute(i0, i1) = max(BiasAdd[i0, i1], 0f)
```

## 为稀疏密集算子 (sparse dense op) 编写自定义草图 (sketch)

在调优之前，需要为稀疏密集操作定义 CustomSketchRule。

CustomSketchRule 由两部分组成：条件函数和应用函数。

* 条件函数：描述应用此调度规则的时间。例如，通过匹配名称和标签将规则应用于稀疏操作。
* 应用函数：描述生成初始草图的方式。可以用 auto-scheduler 提供的循环状态 API 来实现。

``` python
def meet_condition_func(search_policy, state, stage_id):
    state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if state.stages[stage_id].op.tag in [
        "sparse_dense_sp_rhs_bsrmm",
        "sparse_dense_sp_rhs_bsrmm_block",
    ]:
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
    else:
        return auto_scheduler.PreloadCustomSketchRule.PASS

def apply_func(search_policy, state, stage_id):
    ret = []
    s0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if s0.stages[stage_id].op.tag == "sparse_dense_sp_rhs_bsrmm_block":
        return [s0.state_object, stage_id - 1]

    sparse_dense = s0.stages[stage_id].op
    sparse_dense_block = s0.stages[stage_id - 1].op
    assert sparse_dense.tag == "sparse_dense_sp_rhs_bsrmm"
    assert sparse_dense_block.tag == "sparse_dense_sp_rhs_bsrmm_block"

    # 设置计算块的默认消费者
    consumer = sparse_dense

    # 若稀疏密集有单个元素消费者
    # 可以计算内联稀疏密集输出阶段
    consumers = _ffi_api.SearchPolicyUtilsGetConsumers(
        search_policy.search_task, s0.state_object, stage_id
    )
    if len(consumers) == 1:
        consumer_id = int(consumers.items()[0][0])
        if _ffi_api.SearchPolicyUtilsIsElementwiseMatch(
            search_policy.search_task, s0.state_object, stage_id, consumer_id
        ):
            consumer = s0.stages[consumer_id].op
            s0.compute_inline(sparse_dense)

    i, nb_j, j, row_offset, c = s0[sparse_dense_block].iters
    m, n = s0[consumer].iters
    i0, i1, i2 = s0.split(sparse_dense_block, i, [None, None])
    m0, m1 = s0.follow_split(consumer, m, len(s0.transform_steps) - 1, 1)
    j0, j1 = s0.split(sparse_dense_block, nb_j, [None])
    n0, n1 = s0.follow_split(consumer, n, len(s0.transform_steps) - 1, 1)
    s0.reorder(sparse_dense_block, [i0, j0, i1, j1, row_offset, i2, j, c])
    s0.reorder(consumer, [m0, n0, m1, n1])
    s0.compute_at(sparse_dense_block, consumer, n0)

    ret.append([s0.state_object, stage_id - 2])

    return ret
```

接下来，为插入自定义草图的 auto-scheduler 设置参数。

* `num_measure_trials` 是搜索过程中可以使用的测试次数（根据自己的时间预算调整这个参数），为快速演示，本教程只进行了 10 次试验。在实践中，推荐使用 1000 以得到收敛结果。
* 此外，使用 `RecordToFile` 将测试记录转储到 *sparse_dense.json* 文件中，测试记录可用于查询历史最佳、恢复搜索以及以后进行更多分析。
* 有关更多参数，参见 `auto_scheduler.TuningOptions`
* 接下来创建一个 `auto_scheduler.SketchPolicy` 对象，并将自定义调度规则添加为 *init_search_callbacks*。

``` python
log_file = "sparse_dense.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

search_policy = auto_scheduler.SketchPolicy(
    task,
    program_cost_model=auto_scheduler.XGBModel(),
    init_search_callbacks=[
        auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "SparseDense")
    ],
)
```

## 运行搜索

现在已经准备好所有的输入。接下来开始搜索，经过一些测试后，可以从日志文件中加载最佳调度并应用。

``` python
# 运行自动调优（搜索）
# 注意：不在网页服务器中运行调优，因为它需要的时间太长。
# 取消以下行的注释，来运行它。
task.tune(tune_option, search_policy)

# 应用最佳 schedule
sch, args = task.apply_best(log_file)
```

输出结果：

``` bash
/usr/local/lib/python3.7/dist-packages/numpy/core/_methods.py:43: RuntimeWarning: invalid value encountered in reduce
  return umr_minimum(a, axis, None, out, keepdims, initial, where)
/usr/local/lib/python3.7/dist-packages/numpy/core/_methods.py:39: RuntimeWarning: invalid value encountered in reduce
  return umr_maximum(a, axis, None, out, keepdims, initial, where)
```

自动调度后对 schedule 降级，来查看 IR。auto-scheduler 正确执行优化，包括多级平铺、布局转换、并行化、向量化、展开和算子融合。

``` python
print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))
```

输出结果：

``` bash
Lowered TIR:
@main = primfn(placeholder_5: handle, placeholder_6: handle, placeholder_7: handle, placeholder_8: handle, placeholder_9: handle, compute_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {placeholder: Buffer(placeholder_10: Pointer(float32), float32, [32768], []),
             placeholder_1: Buffer(placeholder_11: Pointer(float32), float32, [78656], []),
             placeholder_2: Buffer(placeholder_12: Pointer(int32), int32, [4916], []),
             placeholder_3: Buffer(placeholder_13: Pointer(int32), int32, [33], []),
             placeholder_4: Buffer(placeholder_14: Pointer(float32), float32, [65536], []),
             compute: Buffer(compute_2: Pointer(float32), float32, [65536], [])}
  buffer_map = {placeholder_5: placeholder, placeholder_6: placeholder_1, placeholder_7: placeholder_2, placeholder_8: placeholder_3, placeholder_9: placeholder_4, compute_1: compute}
  preflattened_buffer_map = {placeholder_5: placeholder_15: Buffer(placeholder_10, float32, [128, 256], []), compute_1: compute_3: Buffer(compute_2, float32, [128, 512], []), placeholder_9: placeholder_16: Buffer(placeholder_14, float32, [128, 512], []), placeholder_7: placeholder_17: Buffer(placeholder_12, int32, [4916], []), placeholder_6: placeholder_18: Buffer(placeholder_11, float32, [4916, 16, 1], []), placeholder_8: placeholder_19: Buffer(placeholder_13, int32, [33], [])} {
  for (i0.outer: int32, 0, 32) "parallel" {
    allocate(compute_4: Pointer(global float32), float32, [64]), storage_scope = global;
    for (i1.outer: int32, 0, 32) {
      compute_5: Buffer(compute_4, float32, [64], [])[0] = 0f32
      compute_5[1] = 0f32
      compute_5[2] = 0f32
      compute_5[3] = 0f32
      compute_5[4] = 0f32
      compute_5[5] = 0f32
      compute_5[6] = 0f32
      compute_5[7] = 0f32
      compute_5[8] = 0f32
      compute_5[9] = 0f32
      compute_5[10] = 0f32
      compute_5[11] = 0f32
      compute_5[12] = 0f32
      compute_5[13] = 0f32
      compute_5[14] = 0f32
      compute_5[15] = 0f32
      compute_5[16] = 0f32
      compute_5[17] = 0f32
      compute_5[18] = 0f32
      compute_5[19] = 0f32
      compute_5[20] = 0f32
      compute_5[21] = 0f32
      compute_5[22] = 0f32
      compute_5[23] = 0f32
      compute_5[24] = 0f32
      compute_5[25] = 0f32
      compute_5[26] = 0f32
      compute_5[27] = 0f32
      compute_5[28] = 0f32
      compute_5[29] = 0f32
      compute_5[30] = 0f32
      compute_5[31] = 0f32
      compute_5[32] = 0f32
      compute_5[33] = 0f32
      compute_5[34] = 0f32
      compute_5[35] = 0f32
      compute_5[36] = 0f32
      compute_5[37] = 0f32
      compute_5[38] = 0f32
      compute_5[39] = 0f32
      compute_5[40] = 0f32
      compute_5[41] = 0f32
      compute_5[42] = 0f32
      compute_5[43] = 0f32
      compute_5[44] = 0f32
      compute_5[45] = 0f32
      compute_5[46] = 0f32
      compute_5[47] = 0f32
      compute_5[48] = 0f32
      compute_5[49] = 0f32
      compute_5[50] = 0f32
      compute_5[51] = 0f32
      compute_5[52] = 0f32
      compute_5[53] = 0f32
      compute_5[54] = 0f32
      compute_5[55] = 0f32
      compute_5[56] = 0f32
      compute_5[57] = 0f32
      compute_5[58] = 0f32
      compute_5[59] = 0f32
      compute_5[60] = 0f32
      compute_5[61] = 0f32
      compute_5[62] = 0f32
      compute_5[63] = 0f32
      for (elem_idx: int32, 0, (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])) {
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[0] = (compute_5[0] + (placeholder_1[((placeholder_3[i1.outer]*16) + (elem_idx*16))]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[1] = (compute_5[1] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 1)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[2] = (compute_5[2] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 2)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[3] = (compute_5[3] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 3)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[4] = (compute_5[4] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 4)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[5] = (compute_5[5] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 5)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[6] = (compute_5[6] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 6)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[7] = (compute_5[7] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 7)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[8] = (compute_5[8] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 8)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[9] = (compute_5[9] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 9)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[10] = (compute_5[10] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 10)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[11] = (compute_5[11] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 11)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[12] = (compute_5[12] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 12)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[13] = (compute_5[13] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 13)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[14] = (compute_5[14] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 14)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[15] = (compute_5[15] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 15)]*max(placeholder[((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)])], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[16] = (compute_5[16] + (placeholder_1[((placeholder_3[i1.outer]*16) + (elem_idx*16))]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[17] = (compute_5[17] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 1)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[18] = (compute_5[18] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 2)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[19] = (compute_5[19] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 3)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[20] = (compute_5[20] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 4)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[21] = (compute_5[21] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 5)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[22] = (compute_5[22] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 6)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[23] = (compute_5[23] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 7)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[24] = (compute_5[24] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 8)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[25] = (compute_5[25] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 9)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[26] = (compute_5[26] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 10)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[27] = (compute_5[27] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 11)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[28] = (compute_5[28] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 12)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[29] = (compute_5[29] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 13)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[30] = (compute_5[30] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 14)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[31] = (compute_5[31] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 15)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 256)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[32] = (compute_5[32] + (placeholder_1[((placeholder_3[i1.outer]*16) + (elem_idx*16))]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[33] = (compute_5[33] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 1)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[34] = (compute_5[34] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 2)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[35] = (compute_5[35] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 3)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[36] = (compute_5[36] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 4)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[37] = (compute_5[37] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 5)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[38] = (compute_5[38] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 6)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[39] = (compute_5[39] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 7)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[40] = (compute_5[40] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 8)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[41] = (compute_5[41] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 9)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[42] = (compute_5[42] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 10)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[43] = (compute_5[43] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 11)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[44] = (compute_5[44] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 12)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[45] = (compute_5[45] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 13)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[46] = (compute_5[46] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 14)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[47] = (compute_5[47] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 15)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 512)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[48] = (compute_5[48] + (placeholder_1[((placeholder_3[i1.outer]*16) + (elem_idx*16))]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[49] = (compute_5[49] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 1)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[50] = (compute_5[50] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 2)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[51] = (compute_5[51] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 3)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[52] = (compute_5[52] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 4)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[53] = (compute_5[53] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 5)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[54] = (compute_5[54] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 6)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[55] = (compute_5[55] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 7)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[56] = (compute_5[56] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 8)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[57] = (compute_5[57] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 9)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[58] = (compute_5[58] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 10)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[59] = (compute_5[59] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 11)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[60] = (compute_5[60] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 12)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[61] = (compute_5[61] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 13)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[62] = (compute_5[62] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 14)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
        if @tir.likely((elem_idx < (placeholder_3[(i1.outer + 1)] - placeholder_3[i1.outer])), dtype=bool) {
          compute_5[63] = (compute_5[63] + (placeholder_1[(((placeholder_3[i1.outer]*16) + (elem_idx*16)) + 15)]*max(placeholder[(((i0.outer*1024) + placeholder_2[(placeholder_3[i1.outer] + elem_idx)]) + 768)], 0f32)))
        }
      }
      for (i0.inner: int32, 0, 4) {
        let cse_var_1: int32 = (((i0.outer*2048) + (i0.inner*512)) + (i1.outer*16))
        compute[ramp(cse_var_1, 1, 16)] = max((compute_5[ramp((i0.inner*16), 1, 16)] + placeholder_4[ramp(cse_var_1, 1, 16)]), broadcast(0f32, 16))
      }
    }
  }
}
```

## 检查正确性并评估性能

构建二进制文件，并检查其正确性和性能。

``` python
func = tvm.build(sch, args, target)

dev = tvm.cpu()

X_tvm = tvm.nd.array(X_np, device=dev)
W_data_tvm = tvm.nd.array(W_sp_np.data, device=dev)
W_indices_tvm = tvm.nd.array(W_sp_np.indices, device=dev)
W_indptr_tvm = tvm.nd.array(W_sp_np.indptr, device=dev)
B_tvm = tvm.nd.array(B_np, device=dev)
Y_tvm = tvm.nd.empty(Y_np.shape, device=dev)

func(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, B_tvm, Y_tvm)

# 检查结果
tvm.testing.assert_allclose(Y_np, Y_tvm.numpy(), atol=1e-4, rtol=1e-4)

# 评估执行时间。
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (
        np.median(evaluator(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, B_tvm, Y_tvm).results)
        * 1000
    )
)
```

输出结果：

``` bash
Execution time of this operator: 2.616 ms
```

:::note
调优结果示例

``` bash
----------------------------------------------------------------------
Lowered TIR:
primfn(placeholder_5: handle, placeholder_6: handle, placeholder_7: handle, placeholder_8: handle, placeholder_9: handle, compute_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {placeholder_2: Buffer(placeholder_10: Pointer(float32), float32, [9831, 16, 1], []),
             placeholder_4: Buffer(placeholder_11: Pointer(int32), int32, [33], []),
             placeholder_3: Buffer(placeholder_12: Pointer(float32), float32, [512, 512], []),
             compute: Buffer(compute_2: Pointer(float32), float32, [512, 512], []),
             placeholder_1: Buffer(placeholder_13: Pointer(float32), float32, [512, 512], []),
             placeholder: Buffer(placeholder_14: Pointer(int32), int32, [9831], [])}
  buffer_map = {placeholder_7: placeholder, placeholder_9: placeholder_1, placeholder_6: placeholder_2, compute_1: compute, placeholder_5: placeholder_3, placeholder_8: placeholder_4} {
  for (i0.outer.i1.outer.fused: int32, 0, 1024) "parallel" {
    attr [compute_3: Pointer(float32)] "storage_scope" = "global";
    allocate(compute_3, float32, [256]) {
      for (nb_j.inner: int32, 0, 2) {
        for (i.inner.init: int32, 0, 8) {
          for (j.init: int32, 0, 16) {
            compute_3[(((i.inner.init*32) + (nb_j.inner*16)) + j.init)] = 0f32
          }
        }
        for (elem_idx: int32, 0, ((int32*)placeholder_11[(((floormod(i0.outer.i1.outer.fused, 16)*2) + nb_j.inner) + 1)] - (int32*)placeholder_11[((floormod(i0.outer.i1.outer.fused, 16)*2) + nb_j.inner)])) {
          for (i.inner: int32, 0, 8) {
            for (j: int32, 0, 16) {
              compute_3[(((i.inner*32) + (nb_j.inner*16)) + j)] = ((float32*)compute_3[(((i.inner*32) + (nb_j.inner*16)) + j)] + ((float32*)placeholder_10[((((int32*)placeholder_11[((floormod(i0.outer.i1.outer.fused, 16)*2) + nb_j.inner)]*16) + (elem_idx*16)) + j)]*max((float32*)placeholder_12[(((floordiv(i0.outer.i1.outer.fused, 16)*4096) + (i.inner*512)) + (int32*)placeholder_14[((int32*)placeholder_11[((floormod(i0.outer.i1.outer.fused, 16)*2) + nb_j.inner)] + elem_idx)])], 0f32)))
            }
          }
        }
      }
      for (i0.inner: int32, 0, 8) {
        compute_2[ramp((((floordiv(i0.outer.i1.outer.fused, 16)*4096) + (i0.inner*512)) + (floormod(i0.outer.i1.outer.fused, 16)*32)), 1, 32)] = max(((float32x32*)compute_3[ramp((i0.inner*32), 1, 32)] + (float32x32*)placeholder_13[ramp((((floordiv(i0.outer.i1.outer.fused, 16)*4096) + (i0.inner*512)) + (floormod(i0.outer.i1.outer.fused, 16)*32)), 1, 32)]), broadcast(0f32, 32))
      }
    }
  }
}
```

[下载 Python 源代码：tune_sparse_x86.py](https://tvm.apache.org/docs/_downloads/07733b6b2cc4df026fce525285e8f538/tune_sparse_x86.py)

[下载 Jupyter Notebook：tune_sparse_x86.ipynb](https://tvm.apache.org/docs/_downloads/293f8d0753933b706a0b588f909fe38a/tune_sparse_x86.ipynb)
