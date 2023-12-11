---
title: 为 GPU 自动调度卷积层
---

# 为 GPU 自动调度卷积层

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_conv2d_layer_cuda.html#sphx-glr-download-how-to-tune-with-autoscheduler-tune-conv2d-layer-cuda-py) 下载完整的示例代码
:::

**作者**：[Lianmin Zheng](https://github.com/merrymercy), [Chengfan Jia](https://github.com/jcf94/)

本文介绍如何为 GPU 使用 auto-scheduler。

与 [AutoTVM](/docs/how_to/autotune) 不同，AutoTVM 依赖手动模板来定义搜索空间，而 auto-scheduler 不需要任何模板。用户只需编写计算声明，无需任何调度命令或模板。auto-scheduler 可以自动生成一个大的搜索空间，并在空间中找到合适的调度。

``` python
import os
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
```

## 定义计算

首先定义卷积层的计算，该函数返回输入/输出张量列表，从这些张量中，auto-scheduler 可以得到整个计算图。

``` python
@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]
```

## 创建搜索任务

然后为 ResNet 中的最后一个卷积层创建一个搜索任务。

``` python
target = tvm.target.Target("cuda")

# 使用 ResNet-50 中的最后一层
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
task = auto_scheduler.SearchTask(
    func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
)

# 检查计算图
print("Computational DAG:")
print(task.compute_dag)
```

输出结果：

``` bash
Computational DAG:
data = PLACEHOLDER [1, 512, 7, 7]
pad_temp(i0, i1, i2, i3) = tir.if_then_else(((((i2 >= 1) && (i2 < 8)) && (i3 >= 1)) && (i3 < 8)), data[i0, i1, (i2 - 1), (i3 - 1)], 0f)
kernel = PLACEHOLDER [512, 512, 3, 3]
conv2d_nchw(nn, ff, yy, xx) += (pad_temp[nn, rc, (yy + ry), (xx + rx)]*kernel[ff, rc, ry, rx])
bias = PLACEHOLDER [1, 512, 1, 1]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nchw[ax0, ax1, ax2, ax3] + bias[ax0, ax1, 0, 0])
compute(i0, i1, i2, i3) = max(T_add[i0, i1, i2, i3], 0f)
```

接下来为 auto-scheduler 设置参数，它们主要指定在搜索过程中如何进行测试。

* `measure_ctx` 启动不同进程从而使测试隔离。它保护主进程在测试期间免受 GPU 崩溃影响，并避免其他 runtime 冲突。
* `min_repeat_ms` 定义每次测试中一次“重复”的最短持续时间。这样可以预热 GPU，从而获得准确测试结果。通常推荐将其值设置为 >= 300 ms。
* `num_measure_trials` 是搜索过程中可以使用的测试次数。为了快速演示，本教程中只进行了 10 次。推荐实际运行时试验 1000 次，也可以根据时间预算进行更多试验。
* 此外，使用 `RecordToFile` 将测试记录转储到文件 conv2d.json 中。测试记录可用于查询历史最佳、恢复搜索以及以后进行更多分析。
* 有关更多参数，参见 `auto_scheduler.TuningOptions`, `auto_scheduler.LocalRPCMeasureContext`。

``` python
log_file = "conv2d.json"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
```

输出结果：

``` bash
Get devices for measurement successfully!
```

## 运行搜索

准备好所有输入后，可以开始搜索，让 auto-scheduler 发挥作用。经过一些测试试验后，可以从日志文件中加载最佳 schedule 并进行应用。

``` python
# 运行自动调优（搜索）
# 我们不再在服务器上运行调优，因为太耗时间了
# 去掉下行代码的注释自行运行
# task.tune(tune_option)
# 应用最佳 schedule
sch, args = task.apply_best(log_file)
# 终止测试过程
del measure_ctx
```

在自动调度后降低调度以查看 IR，auto-scheduler 正确执行优化，包括多级平铺、协作获取、展开和算子融合。

``` python
print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))
```

输出结果：

``` bash
Lowered TIR:
@main = primfn(data_1: handle, kernel_1: handle, bias_1: handle, compute_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {data: Buffer(data_2: Pointer(float32), float32, [25088], []),
             kernel: Buffer(kernel_2: Pointer(float32), float32, [2359296], []),
             bias: Buffer(bias_2: Pointer(float32), float32, [512], []),
             compute: Buffer(compute_2: Pointer(float32), float32, [25088], [])}
  buffer_map = {data_1: data, kernel_1: kernel, bias_1: bias, compute_1: compute}
  preflattened_buffer_map = {data_1: data_3: Buffer(data_2, float32, [1, 512, 7, 7], []), kernel_1: kernel_3: Buffer(kernel_2, float32, [512, 512, 3, 3], []), bias_1: bias_3: Buffer(bias_2, float32, [1, 512, 1, 1], []), compute_1: compute_3: Buffer(compute_2, float32, [1, 512, 7, 7], [])} {
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 28;
  allocate(conv2d_nchw: Pointer(local float32), float32, [14]), storage_scope = local;
  allocate(pad_temp.shared: Pointer(shared float32), float32, [72]), storage_scope = shared;
  allocate(kernel.shared: Pointer(shared float32), float32, [3072]), storage_scope = shared;
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64 {
    conv2d_nchw_1: Buffer(conv2d_nchw, float32, [14], [], scope="local", align=32)[0] = 0f32
    conv2d_nchw_1[1] = 0f32
    conv2d_nchw_1[2] = 0f32
    conv2d_nchw_1[3] = 0f32
    conv2d_nchw_1[4] = 0f32
    conv2d_nchw_1[5] = 0f32
    conv2d_nchw_1[6] = 0f32
    conv2d_nchw_1[7] = 0f32
    conv2d_nchw_1[8] = 0f32
    conv2d_nchw_1[9] = 0f32
    conv2d_nchw_1[10] = 0f32
    conv2d_nchw_1[11] = 0f32
    conv2d_nchw_1[12] = 0f32
    conv2d_nchw_1[13] = 0f32
    for (rc.outer.outer: int32, 0, 64) {
      for (ry.outer.outer: int32, 0, 3) {
        let cse_var_2: int32 = (rc.outer.outer*72)
        let cse_var_1: int32 = (ry.outer.outer*3)
         {
          attr [IterVar(threadIdx.x_1: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64 {
            if @tir.likely((threadIdx.x_1 < 18), dtype=bool) {
              pad_temp.shared_1: Buffer(pad_temp.shared, float32, [72], [], scope="shared")[(threadIdx.x_1*4)] = @tir.if_then_else(((((1 <= (ry.outer.outer + floormod(blockIdx.x, 7))) && ((ry.outer.outer + floormod(blockIdx.x, 7)) < 8)) && (1 <= floormod((threadIdx.x_1*4), 9))) && (floormod((threadIdx.x_1*4), 9) < 8)), data[((((((rc.outer.outer*392) + (floordiv((threadIdx.x_1*4), 9)*49)) + (ry.outer.outer*7)) + (floormod(blockIdx.x, 7)*7)) + floormod((threadIdx.x_1*4), 9)) - 8)], 0f32, dtype=float32)
            }
            if @tir.likely((threadIdx.x_1 < 18), dtype=bool) {
              pad_temp.shared_1[((threadIdx.x_1*4) + 1)] = @tir.if_then_else(((((1 <= (ry.outer.outer + floormod(blockIdx.x, 7))) && ((ry.outer.outer + floormod(blockIdx.x, 7)) < 8)) && (1 <= floormod(((threadIdx.x_1*4) + 1), 9))) && (floormod(((threadIdx.x_1*4) + 1), 9) < 8)), data[((((((rc.outer.outer*392) + (floordiv(((threadIdx.x_1*4) + 1), 9)*49)) + (ry.outer.outer*7)) + (floormod(blockIdx.x, 7)*7)) + floormod(((threadIdx.x_1*4) + 1), 9)) - 8)], 0f32, dtype=float32)
            }
            if @tir.likely((threadIdx.x_1 < 18), dtype=bool) {
              pad_temp.shared_1[((threadIdx.x_1*4) + 2)] = @tir.if_then_else(((((1 <= (ry.outer.outer + floormod(blockIdx.x, 7))) && ((ry.outer.outer + floormod(blockIdx.x, 7)) < 8)) && (1 <= floormod(((threadIdx.x_1*4) + 2), 9))) && (floormod(((threadIdx.x_1*4) + 2), 9) < 8)), data[((((((rc.outer.outer*392) + (floordiv(((threadIdx.x_1*4) + 2), 9)*49)) + (ry.outer.outer*7)) + (floormod(blockIdx.x, 7)*7)) + floormod(((threadIdx.x_1*4) + 2), 9)) - 8)], 0f32, dtype=float32)
            }
            if @tir.likely((threadIdx.x_1 < 18), dtype=bool) {
              pad_temp.shared_1[((threadIdx.x_1*4) + 3)] = @tir.if_then_else(((((1 <= (ry.outer.outer + floormod(blockIdx.x, 7))) && ((ry.outer.outer + floormod(blockIdx.x, 7)) < 8)) && (1 <= floormod(((threadIdx.x_1*4) + 3), 9))) && (floormod(((threadIdx.x_1*4) + 3), 9) < 8)), data[((((((rc.outer.outer*392) + (floordiv(((threadIdx.x_1*4) + 3), 9)*49)) + (ry.outer.outer*7)) + (floormod(blockIdx.x, 7)*7)) + floormod(((threadIdx.x_1*4) + 3), 9)) - 8)], 0f32, dtype=float32)
            }
          }
          attr [IterVar(threadIdx.x_2: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1: Buffer(kernel.shared, float32, [3072], [], scope="shared")[threadIdx.x_2] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 64)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 64), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 128)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 128), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 192)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 36864)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 256)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 256), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 320)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 320), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 384)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 73728)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 448)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 448), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 512)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 512), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 576)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 110592)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 640)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 640), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 704)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 704), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 768)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 147456)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 832)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 832), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 896)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 896), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 960)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 184320)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1024)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1024), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1088)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1088), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1152)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 221184)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1216)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1216), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1280)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1280), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1344)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 258048)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1408)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1408), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1472)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1472), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1536)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 294912)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1600)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1600), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1664)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1664), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1728)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 331776)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1792)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1792), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1856)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1856), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1920)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 368640)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 1984)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 1984), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2048)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 2048), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2112)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 405504)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2176)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 2176), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2240)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 2240), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2304)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 442368)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2368)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 2368), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2432)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 2432), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2496)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 479232)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2560)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 2560), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2624)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 2624), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2688)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 516096)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2752)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 2752), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2816)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 2816), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2880)] = kernel[(((((((floordiv(blockIdx.x, 7)*589824) + (floordiv(threadIdx.x_2, 24)*4608)) + cse_var_2) + (floordiv(floormod(threadIdx.x_2, 24), 3)*9)) + cse_var_1) + floormod(threadIdx.x_2, 3)) + 552960)]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 2944)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 2944), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 16), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 1), 3))]
          attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
          kernel.shared_1[(threadIdx.x_2 + 3008)] = kernel[((((((floordiv(blockIdx.x, 7)*589824) + (floordiv((threadIdx.x_2 + 3008), 24)*4608)) + cse_var_2) + (floordiv(floormod((threadIdx.x_2 + 8), 24), 3)*9)) + cse_var_1) + floormod((threadIdx.x_2 + 2), 3))]
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[0]*kernel.shared_1[(threadIdx.x*48)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[9]*kernel.shared_1[((threadIdx.x*48) + 3)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[1]*kernel.shared_1[(threadIdx.x*48)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[10]*kernel.shared_1[((threadIdx.x*48) + 3)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[2]*kernel.shared_1[(threadIdx.x*48)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[11]*kernel.shared_1[((threadIdx.x*48) + 3)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[3]*kernel.shared_1[(threadIdx.x*48)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[12]*kernel.shared_1[((threadIdx.x*48) + 3)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[4]*kernel.shared_1[(threadIdx.x*48)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[13]*kernel.shared_1[((threadIdx.x*48) + 3)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[5]*kernel.shared_1[(threadIdx.x*48)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[14]*kernel.shared_1[((threadIdx.x*48) + 3)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[6]*kernel.shared_1[(threadIdx.x*48)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[15]*kernel.shared_1[((threadIdx.x*48) + 3)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[0]*kernel.shared_1[((threadIdx.x*48) + 24)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[9]*kernel.shared_1[((threadIdx.x*48) + 27)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[1]*kernel.shared_1[((threadIdx.x*48) + 24)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[10]*kernel.shared_1[((threadIdx.x*48) + 27)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[2]*kernel.shared_1[((threadIdx.x*48) + 24)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[11]*kernel.shared_1[((threadIdx.x*48) + 27)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[3]*kernel.shared_1[((threadIdx.x*48) + 24)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[12]*kernel.shared_1[((threadIdx.x*48) + 27)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[4]*kernel.shared_1[((threadIdx.x*48) + 24)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[13]*kernel.shared_1[((threadIdx.x*48) + 27)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[5]*kernel.shared_1[((threadIdx.x*48) + 24)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[14]*kernel.shared_1[((threadIdx.x*48) + 27)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[6]*kernel.shared_1[((threadIdx.x*48) + 24)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[15]*kernel.shared_1[((threadIdx.x*48) + 27)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[1]*kernel.shared_1[((threadIdx.x*48) + 1)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[10]*kernel.shared_1[((threadIdx.x*48) + 4)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[2]*kernel.shared_1[((threadIdx.x*48) + 1)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[11]*kernel.shared_1[((threadIdx.x*48) + 4)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[3]*kernel.shared_1[((threadIdx.x*48) + 1)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[12]*kernel.shared_1[((threadIdx.x*48) + 4)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[4]*kernel.shared_1[((threadIdx.x*48) + 1)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[13]*kernel.shared_1[((threadIdx.x*48) + 4)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[5]*kernel.shared_1[((threadIdx.x*48) + 1)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[14]*kernel.shared_1[((threadIdx.x*48) + 4)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[6]*kernel.shared_1[((threadIdx.x*48) + 1)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[15]*kernel.shared_1[((threadIdx.x*48) + 4)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[7]*kernel.shared_1[((threadIdx.x*48) + 1)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[16]*kernel.shared_1[((threadIdx.x*48) + 4)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[1]*kernel.shared_1[((threadIdx.x*48) + 25)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[10]*kernel.shared_1[((threadIdx.x*48) + 28)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[2]*kernel.shared_1[((threadIdx.x*48) + 25)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[11]*kernel.shared_1[((threadIdx.x*48) + 28)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[3]*kernel.shared_1[((threadIdx.x*48) + 25)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[12]*kernel.shared_1[((threadIdx.x*48) + 28)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[4]*kernel.shared_1[((threadIdx.x*48) + 25)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[13]*kernel.shared_1[((threadIdx.x*48) + 28)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[5]*kernel.shared_1[((threadIdx.x*48) + 25)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[14]*kernel.shared_1[((threadIdx.x*48) + 28)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[6]*kernel.shared_1[((threadIdx.x*48) + 25)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[15]*kernel.shared_1[((threadIdx.x*48) + 28)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[7]*kernel.shared_1[((threadIdx.x*48) + 25)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[16]*kernel.shared_1[((threadIdx.x*48) + 28)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[2]*kernel.shared_1[((threadIdx.x*48) + 2)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[11]*kernel.shared_1[((threadIdx.x*48) + 5)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[3]*kernel.shared_1[((threadIdx.x*48) + 2)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[12]*kernel.shared_1[((threadIdx.x*48) + 5)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[4]*kernel.shared_1[((threadIdx.x*48) + 2)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[13]*kernel.shared_1[((threadIdx.x*48) + 5)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[5]*kernel.shared_1[((threadIdx.x*48) + 2)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[14]*kernel.shared_1[((threadIdx.x*48) + 5)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[6]*kernel.shared_1[((threadIdx.x*48) + 2)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[15]*kernel.shared_1[((threadIdx.x*48) + 5)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[7]*kernel.shared_1[((threadIdx.x*48) + 2)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[16]*kernel.shared_1[((threadIdx.x*48) + 5)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[8]*kernel.shared_1[((threadIdx.x*48) + 2)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[17]*kernel.shared_1[((threadIdx.x*48) + 5)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[2]*kernel.shared_1[((threadIdx.x*48) + 26)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[11]*kernel.shared_1[((threadIdx.x*48) + 29)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[3]*kernel.shared_1[((threadIdx.x*48) + 26)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[12]*kernel.shared_1[((threadIdx.x*48) + 29)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[4]*kernel.shared_1[((threadIdx.x*48) + 26)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[13]*kernel.shared_1[((threadIdx.x*48) + 29)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[5]*kernel.shared_1[((threadIdx.x*48) + 26)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[14]*kernel.shared_1[((threadIdx.x*48) + 29)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[6]*kernel.shared_1[((threadIdx.x*48) + 26)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[15]*kernel.shared_1[((threadIdx.x*48) + 29)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[7]*kernel.shared_1[((threadIdx.x*48) + 26)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[16]*kernel.shared_1[((threadIdx.x*48) + 29)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[8]*kernel.shared_1[((threadIdx.x*48) + 26)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[17]*kernel.shared_1[((threadIdx.x*48) + 29)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[18]*kernel.shared_1[((threadIdx.x*48) + 6)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[27]*kernel.shared_1[((threadIdx.x*48) + 9)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[19]*kernel.shared_1[((threadIdx.x*48) + 6)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[28]*kernel.shared_1[((threadIdx.x*48) + 9)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[20]*kernel.shared_1[((threadIdx.x*48) + 6)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[29]*kernel.shared_1[((threadIdx.x*48) + 9)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[21]*kernel.shared_1[((threadIdx.x*48) + 6)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[30]*kernel.shared_1[((threadIdx.x*48) + 9)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[22]*kernel.shared_1[((threadIdx.x*48) + 6)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[31]*kernel.shared_1[((threadIdx.x*48) + 9)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[23]*kernel.shared_1[((threadIdx.x*48) + 6)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[32]*kernel.shared_1[((threadIdx.x*48) + 9)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[24]*kernel.shared_1[((threadIdx.x*48) + 6)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[33]*kernel.shared_1[((threadIdx.x*48) + 9)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[18]*kernel.shared_1[((threadIdx.x*48) + 30)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[27]*kernel.shared_1[((threadIdx.x*48) + 33)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[19]*kernel.shared_1[((threadIdx.x*48) + 30)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[28]*kernel.shared_1[((threadIdx.x*48) + 33)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[20]*kernel.shared_1[((threadIdx.x*48) + 30)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[29]*kernel.shared_1[((threadIdx.x*48) + 33)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[21]*kernel.shared_1[((threadIdx.x*48) + 30)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[30]*kernel.shared_1[((threadIdx.x*48) + 33)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[22]*kernel.shared_1[((threadIdx.x*48) + 30)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[31]*kernel.shared_1[((threadIdx.x*48) + 33)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[23]*kernel.shared_1[((threadIdx.x*48) + 30)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[32]*kernel.shared_1[((threadIdx.x*48) + 33)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[24]*kernel.shared_1[((threadIdx.x*48) + 30)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[33]*kernel.shared_1[((threadIdx.x*48) + 33)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[19]*kernel.shared_1[((threadIdx.x*48) + 7)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[28]*kernel.shared_1[((threadIdx.x*48) + 10)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[20]*kernel.shared_1[((threadIdx.x*48) + 7)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[29]*kernel.shared_1[((threadIdx.x*48) + 10)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[21]*kernel.shared_1[((threadIdx.x*48) + 7)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[30]*kernel.shared_1[((threadIdx.x*48) + 10)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[22]*kernel.shared_1[((threadIdx.x*48) + 7)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[31]*kernel.shared_1[((threadIdx.x*48) + 10)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[23]*kernel.shared_1[((threadIdx.x*48) + 7)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[32]*kernel.shared_1[((threadIdx.x*48) + 10)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[24]*kernel.shared_1[((threadIdx.x*48) + 7)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[33]*kernel.shared_1[((threadIdx.x*48) + 10)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[25]*kernel.shared_1[((threadIdx.x*48) + 7)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[34]*kernel.shared_1[((threadIdx.x*48) + 10)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[19]*kernel.shared_1[((threadIdx.x*48) + 31)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[28]*kernel.shared_1[((threadIdx.x*48) + 34)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[20]*kernel.shared_1[((threadIdx.x*48) + 31)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[29]*kernel.shared_1[((threadIdx.x*48) + 34)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[21]*kernel.shared_1[((threadIdx.x*48) + 31)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[30]*kernel.shared_1[((threadIdx.x*48) + 34)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[22]*kernel.shared_1[((threadIdx.x*48) + 31)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[31]*kernel.shared_1[((threadIdx.x*48) + 34)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[23]*kernel.shared_1[((threadIdx.x*48) + 31)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[32]*kernel.shared_1[((threadIdx.x*48) + 34)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[24]*kernel.shared_1[((threadIdx.x*48) + 31)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[33]*kernel.shared_1[((threadIdx.x*48) + 34)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[25]*kernel.shared_1[((threadIdx.x*48) + 31)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[34]*kernel.shared_1[((threadIdx.x*48) + 34)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[20]*kernel.shared_1[((threadIdx.x*48) + 8)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[29]*kernel.shared_1[((threadIdx.x*48) + 11)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[21]*kernel.shared_1[((threadIdx.x*48) + 8)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[30]*kernel.shared_1[((threadIdx.x*48) + 11)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[22]*kernel.shared_1[((threadIdx.x*48) + 8)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[31]*kernel.shared_1[((threadIdx.x*48) + 11)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[23]*kernel.shared_1[((threadIdx.x*48) + 8)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[32]*kernel.shared_1[((threadIdx.x*48) + 11)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[24]*kernel.shared_1[((threadIdx.x*48) + 8)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[33]*kernel.shared_1[((threadIdx.x*48) + 11)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[25]*kernel.shared_1[((threadIdx.x*48) + 8)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[34]*kernel.shared_1[((threadIdx.x*48) + 11)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[26]*kernel.shared_1[((threadIdx.x*48) + 8)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[35]*kernel.shared_1[((threadIdx.x*48) + 11)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[20]*kernel.shared_1[((threadIdx.x*48) + 32)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[29]*kernel.shared_1[((threadIdx.x*48) + 35)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[21]*kernel.shared_1[((threadIdx.x*48) + 32)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[30]*kernel.shared_1[((threadIdx.x*48) + 35)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[22]*kernel.shared_1[((threadIdx.x*48) + 32)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[31]*kernel.shared_1[((threadIdx.x*48) + 35)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[23]*kernel.shared_1[((threadIdx.x*48) + 32)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[32]*kernel.shared_1[((threadIdx.x*48) + 35)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[24]*kernel.shared_1[((threadIdx.x*48) + 32)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[33]*kernel.shared_1[((threadIdx.x*48) + 35)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[25]*kernel.shared_1[((threadIdx.x*48) + 32)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[34]*kernel.shared_1[((threadIdx.x*48) + 35)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[26]*kernel.shared_1[((threadIdx.x*48) + 32)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[35]*kernel.shared_1[((threadIdx.x*48) + 35)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[36]*kernel.shared_1[((threadIdx.x*48) + 12)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[45]*kernel.shared_1[((threadIdx.x*48) + 15)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[37]*kernel.shared_1[((threadIdx.x*48) + 12)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[46]*kernel.shared_1[((threadIdx.x*48) + 15)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[38]*kernel.shared_1[((threadIdx.x*48) + 12)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[47]*kernel.shared_1[((threadIdx.x*48) + 15)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[39]*kernel.shared_1[((threadIdx.x*48) + 12)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[48]*kernel.shared_1[((threadIdx.x*48) + 15)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[40]*kernel.shared_1[((threadIdx.x*48) + 12)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[49]*kernel.shared_1[((threadIdx.x*48) + 15)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[41]*kernel.shared_1[((threadIdx.x*48) + 12)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[50]*kernel.shared_1[((threadIdx.x*48) + 15)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[42]*kernel.shared_1[((threadIdx.x*48) + 12)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[51]*kernel.shared_1[((threadIdx.x*48) + 15)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[36]*kernel.shared_1[((threadIdx.x*48) + 36)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[45]*kernel.shared_1[((threadIdx.x*48) + 39)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[37]*kernel.shared_1[((threadIdx.x*48) + 36)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[46]*kernel.shared_1[((threadIdx.x*48) + 39)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[38]*kernel.shared_1[((threadIdx.x*48) + 36)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[47]*kernel.shared_1[((threadIdx.x*48) + 39)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[39]*kernel.shared_1[((threadIdx.x*48) + 36)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[48]*kernel.shared_1[((threadIdx.x*48) + 39)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[40]*kernel.shared_1[((threadIdx.x*48) + 36)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[49]*kernel.shared_1[((threadIdx.x*48) + 39)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[41]*kernel.shared_1[((threadIdx.x*48) + 36)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[50]*kernel.shared_1[((threadIdx.x*48) + 39)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[42]*kernel.shared_1[((threadIdx.x*48) + 36)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[51]*kernel.shared_1[((threadIdx.x*48) + 39)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[37]*kernel.shared_1[((threadIdx.x*48) + 13)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[46]*kernel.shared_1[((threadIdx.x*48) + 16)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[38]*kernel.shared_1[((threadIdx.x*48) + 13)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[47]*kernel.shared_1[((threadIdx.x*48) + 16)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[39]*kernel.shared_1[((threadIdx.x*48) + 13)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[48]*kernel.shared_1[((threadIdx.x*48) + 16)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[40]*kernel.shared_1[((threadIdx.x*48) + 13)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[49]*kernel.shared_1[((threadIdx.x*48) + 16)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[41]*kernel.shared_1[((threadIdx.x*48) + 13)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[50]*kernel.shared_1[((threadIdx.x*48) + 16)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[42]*kernel.shared_1[((threadIdx.x*48) + 13)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[51]*kernel.shared_1[((threadIdx.x*48) + 16)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[43]*kernel.shared_1[((threadIdx.x*48) + 13)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[52]*kernel.shared_1[((threadIdx.x*48) + 16)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[37]*kernel.shared_1[((threadIdx.x*48) + 37)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[46]*kernel.shared_1[((threadIdx.x*48) + 40)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[38]*kernel.shared_1[((threadIdx.x*48) + 37)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[47]*kernel.shared_1[((threadIdx.x*48) + 40)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[39]*kernel.shared_1[((threadIdx.x*48) + 37)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[48]*kernel.shared_1[((threadIdx.x*48) + 40)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[40]*kernel.shared_1[((threadIdx.x*48) + 37)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[49]*kernel.shared_1[((threadIdx.x*48) + 40)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[41]*kernel.shared_1[((threadIdx.x*48) + 37)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[50]*kernel.shared_1[((threadIdx.x*48) + 40)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[42]*kernel.shared_1[((threadIdx.x*48) + 37)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[51]*kernel.shared_1[((threadIdx.x*48) + 40)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[43]*kernel.shared_1[((threadIdx.x*48) + 37)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[52]*kernel.shared_1[((threadIdx.x*48) + 40)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[38]*kernel.shared_1[((threadIdx.x*48) + 14)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[47]*kernel.shared_1[((threadIdx.x*48) + 17)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[39]*kernel.shared_1[((threadIdx.x*48) + 14)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[48]*kernel.shared_1[((threadIdx.x*48) + 17)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[40]*kernel.shared_1[((threadIdx.x*48) + 14)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[49]*kernel.shared_1[((threadIdx.x*48) + 17)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[41]*kernel.shared_1[((threadIdx.x*48) + 14)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[50]*kernel.shared_1[((threadIdx.x*48) + 17)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[42]*kernel.shared_1[((threadIdx.x*48) + 14)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[51]*kernel.shared_1[((threadIdx.x*48) + 17)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[43]*kernel.shared_1[((threadIdx.x*48) + 14)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[52]*kernel.shared_1[((threadIdx.x*48) + 17)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[44]*kernel.shared_1[((threadIdx.x*48) + 14)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[53]*kernel.shared_1[((threadIdx.x*48) + 17)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[38]*kernel.shared_1[((threadIdx.x*48) + 38)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[47]*kernel.shared_1[((threadIdx.x*48) + 41)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[39]*kernel.shared_1[((threadIdx.x*48) + 38)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[48]*kernel.shared_1[((threadIdx.x*48) + 41)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[40]*kernel.shared_1[((threadIdx.x*48) + 38)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[49]*kernel.shared_1[((threadIdx.x*48) + 41)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[41]*kernel.shared_1[((threadIdx.x*48) + 38)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[50]*kernel.shared_1[((threadIdx.x*48) + 41)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[42]*kernel.shared_1[((threadIdx.x*48) + 38)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[51]*kernel.shared_1[((threadIdx.x*48) + 41)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[43]*kernel.shared_1[((threadIdx.x*48) + 38)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[52]*kernel.shared_1[((threadIdx.x*48) + 41)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[44]*kernel.shared_1[((threadIdx.x*48) + 38)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[53]*kernel.shared_1[((threadIdx.x*48) + 41)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[54]*kernel.shared_1[((threadIdx.x*48) + 18)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[63]*kernel.shared_1[((threadIdx.x*48) + 21)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[55]*kernel.shared_1[((threadIdx.x*48) + 18)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[64]*kernel.shared_1[((threadIdx.x*48) + 21)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[56]*kernel.shared_1[((threadIdx.x*48) + 18)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[65]*kernel.shared_1[((threadIdx.x*48) + 21)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[57]*kernel.shared_1[((threadIdx.x*48) + 18)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[66]*kernel.shared_1[((threadIdx.x*48) + 21)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[58]*kernel.shared_1[((threadIdx.x*48) + 18)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[67]*kernel.shared_1[((threadIdx.x*48) + 21)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[59]*kernel.shared_1[((threadIdx.x*48) + 18)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[68]*kernel.shared_1[((threadIdx.x*48) + 21)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[60]*kernel.shared_1[((threadIdx.x*48) + 18)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[69]*kernel.shared_1[((threadIdx.x*48) + 21)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[54]*kernel.shared_1[((threadIdx.x*48) + 42)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[63]*kernel.shared_1[((threadIdx.x*48) + 45)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[55]*kernel.shared_1[((threadIdx.x*48) + 42)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[64]*kernel.shared_1[((threadIdx.x*48) + 45)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[56]*kernel.shared_1[((threadIdx.x*48) + 42)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[65]*kernel.shared_1[((threadIdx.x*48) + 45)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[57]*kernel.shared_1[((threadIdx.x*48) + 42)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[66]*kernel.shared_1[((threadIdx.x*48) + 45)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[58]*kernel.shared_1[((threadIdx.x*48) + 42)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[67]*kernel.shared_1[((threadIdx.x*48) + 45)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[59]*kernel.shared_1[((threadIdx.x*48) + 42)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[68]*kernel.shared_1[((threadIdx.x*48) + 45)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[60]*kernel.shared_1[((threadIdx.x*48) + 42)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[69]*kernel.shared_1[((threadIdx.x*48) + 45)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[55]*kernel.shared_1[((threadIdx.x*48) + 19)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[64]*kernel.shared_1[((threadIdx.x*48) + 22)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[56]*kernel.shared_1[((threadIdx.x*48) + 19)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[65]*kernel.shared_1[((threadIdx.x*48) + 22)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[57]*kernel.shared_1[((threadIdx.x*48) + 19)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[66]*kernel.shared_1[((threadIdx.x*48) + 22)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[58]*kernel.shared_1[((threadIdx.x*48) + 19)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[67]*kernel.shared_1[((threadIdx.x*48) + 22)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[59]*kernel.shared_1[((threadIdx.x*48) + 19)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[68]*kernel.shared_1[((threadIdx.x*48) + 22)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[60]*kernel.shared_1[((threadIdx.x*48) + 19)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[69]*kernel.shared_1[((threadIdx.x*48) + 22)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[61]*kernel.shared_1[((threadIdx.x*48) + 19)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[70]*kernel.shared_1[((threadIdx.x*48) + 22)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[55]*kernel.shared_1[((threadIdx.x*48) + 43)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[64]*kernel.shared_1[((threadIdx.x*48) + 46)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[56]*kernel.shared_1[((threadIdx.x*48) + 43)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[65]*kernel.shared_1[((threadIdx.x*48) + 46)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[57]*kernel.shared_1[((threadIdx.x*48) + 43)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[66]*kernel.shared_1[((threadIdx.x*48) + 46)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[58]*kernel.shared_1[((threadIdx.x*48) + 43)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[67]*kernel.shared_1[((threadIdx.x*48) + 46)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[59]*kernel.shared_1[((threadIdx.x*48) + 43)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[68]*kernel.shared_1[((threadIdx.x*48) + 46)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[60]*kernel.shared_1[((threadIdx.x*48) + 43)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[69]*kernel.shared_1[((threadIdx.x*48) + 46)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[61]*kernel.shared_1[((threadIdx.x*48) + 43)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[70]*kernel.shared_1[((threadIdx.x*48) + 46)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[56]*kernel.shared_1[((threadIdx.x*48) + 20)]))
          conv2d_nchw_1[0] = (conv2d_nchw_1[0] + (pad_temp.shared_1[65]*kernel.shared_1[((threadIdx.x*48) + 23)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[57]*kernel.shared_1[((threadIdx.x*48) + 20)]))
          conv2d_nchw_1[1] = (conv2d_nchw_1[1] + (pad_temp.shared_1[66]*kernel.shared_1[((threadIdx.x*48) + 23)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[58]*kernel.shared_1[((threadIdx.x*48) + 20)]))
          conv2d_nchw_1[2] = (conv2d_nchw_1[2] + (pad_temp.shared_1[67]*kernel.shared_1[((threadIdx.x*48) + 23)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[59]*kernel.shared_1[((threadIdx.x*48) + 20)]))
          conv2d_nchw_1[3] = (conv2d_nchw_1[3] + (pad_temp.shared_1[68]*kernel.shared_1[((threadIdx.x*48) + 23)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[60]*kernel.shared_1[((threadIdx.x*48) + 20)]))
          conv2d_nchw_1[4] = (conv2d_nchw_1[4] + (pad_temp.shared_1[69]*kernel.shared_1[((threadIdx.x*48) + 23)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[61]*kernel.shared_1[((threadIdx.x*48) + 20)]))
          conv2d_nchw_1[5] = (conv2d_nchw_1[5] + (pad_temp.shared_1[70]*kernel.shared_1[((threadIdx.x*48) + 23)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[62]*kernel.shared_1[((threadIdx.x*48) + 20)]))
          conv2d_nchw_1[6] = (conv2d_nchw_1[6] + (pad_temp.shared_1[71]*kernel.shared_1[((threadIdx.x*48) + 23)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[56]*kernel.shared_1[((threadIdx.x*48) + 44)]))
          conv2d_nchw_1[7] = (conv2d_nchw_1[7] + (pad_temp.shared_1[65]*kernel.shared_1[((threadIdx.x*48) + 47)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[57]*kernel.shared_1[((threadIdx.x*48) + 44)]))
          conv2d_nchw_1[8] = (conv2d_nchw_1[8] + (pad_temp.shared_1[66]*kernel.shared_1[((threadIdx.x*48) + 47)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[58]*kernel.shared_1[((threadIdx.x*48) + 44)]))
          conv2d_nchw_1[9] = (conv2d_nchw_1[9] + (pad_temp.shared_1[67]*kernel.shared_1[((threadIdx.x*48) + 47)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[59]*kernel.shared_1[((threadIdx.x*48) + 44)]))
          conv2d_nchw_1[10] = (conv2d_nchw_1[10] + (pad_temp.shared_1[68]*kernel.shared_1[((threadIdx.x*48) + 47)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[60]*kernel.shared_1[((threadIdx.x*48) + 44)]))
          conv2d_nchw_1[11] = (conv2d_nchw_1[11] + (pad_temp.shared_1[69]*kernel.shared_1[((threadIdx.x*48) + 47)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[61]*kernel.shared_1[((threadIdx.x*48) + 44)]))
          conv2d_nchw_1[12] = (conv2d_nchw_1[12] + (pad_temp.shared_1[70]*kernel.shared_1[((threadIdx.x*48) + 47)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[62]*kernel.shared_1[((threadIdx.x*48) + 44)]))
          conv2d_nchw_1[13] = (conv2d_nchw_1[13] + (pad_temp.shared_1[71]*kernel.shared_1[((threadIdx.x*48) + 47)]))
        }
      }
    }
    for (i1.inner: int32, 0, 2) {
      for (i3.inner: int32, 0, 7) {
        compute[(((((floordiv(blockIdx.x, 7)*6272) + (threadIdx.x*98)) + (i1.inner*49)) + (floormod(blockIdx.x, 7)*7)) + i3.inner)] = max((conv2d_nchw_1[((i1.inner*7) + i3.inner)] + bias[(((floordiv(blockIdx.x, 7)*128) + (threadIdx.x*2)) + i1.inner)]), 0f32)
      }
    }
  }
}
```

## 检查正确性并评估性能

构建二进制文件，并验证其正确性和性能。

``` python
func = tvm.build(sch, args, target)

# 检查正确性
data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
bias_np = np.random.uniform(size=(1, CO, 1, 1)).astype(np.float32)
conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)
out_np = np.maximum(conv_np + bias_np, 0.0)

dev = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
bias_tvm = tvm.nd.array(bias_np, device=dev)
out_tvm = tvm.nd.empty(out_np.shape, device=dev)
func(data_tvm, weight_tvm, bias_tvm, out_tvm)

# 检查结果
np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

# 评估执行时间
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
)
```

输出结果：

``` bash
Execution time of this operator: 0.361 ms
```

## 使用记录文件

在搜索过程中，所有测试记录（可用于重新应用搜索结果、恢复搜索和执行其他分析）都转储到记录文件「conv2d.json」中。

下面示例中从文件中加载最佳调度，打印等效的 Python 调度 API 和 CUDA 源代码（可用于调试和分析自动调度程序的行为）。

``` python
print("Equivalent python schedule:")
print(task.print_best(log_file, print_mode="schedule"))

print("CUDA source code:")
print(task.print_best(log_file, print_mode="cuda"))
```

输出结果：

``` bash
Equivalent python schedule:
pad_temp_i0, pad_temp_i1, pad_temp_i2, pad_temp_i3 = tuple(pad_temp.op.axis) + tuple(pad_temp.op.reduce_axis)
conv2d_nchw_nn, conv2d_nchw_ff, conv2d_nchw_yy, conv2d_nchw_xx, conv2d_nchw_rc, conv2d_nchw_ry, conv2d_nchw_rx = tuple(conv2d_nchw.op.axis) + tuple(conv2d_nchw.op.reduce_axis)
T_add_ax0, T_add_ax1, T_add_ax2, T_add_ax3 = tuple(T_add.op.axis) + tuple(T_add.op.reduce_axis)
compute_i0, compute_i1, compute_i2, compute_i3 = tuple(compute.op.axis) + tuple(compute.op.reduce_axis)
s[T_add].compute_inline()
conv2d_nchw_nn_o_i, conv2d_nchw_nn_i = s[conv2d_nchw].split(conv2d_nchw_nn, factor=1)
conv2d_nchw_nn_o_o_i, conv2d_nchw_nn_o_i = s[conv2d_nchw].split(conv2d_nchw_nn_o_i, factor=1)
conv2d_nchw_nn_o_o_o_i, conv2d_nchw_nn_o_o_i = s[conv2d_nchw].split(conv2d_nchw_nn_o_o_i, factor=1)
conv2d_nchw_nn_o_o_o_o, conv2d_nchw_nn_o_o_o_i = s[conv2d_nchw].split(conv2d_nchw_nn_o_o_o_i, factor=1)
conv2d_nchw_ff_o_i, conv2d_nchw_ff_i = s[conv2d_nchw].split(conv2d_nchw_ff, factor=1)
conv2d_nchw_ff_o_o_i, conv2d_nchw_ff_o_i = s[conv2d_nchw].split(conv2d_nchw_ff_o_i, factor=2)
conv2d_nchw_ff_o_o_o_i, conv2d_nchw_ff_o_o_i = s[conv2d_nchw].split(conv2d_nchw_ff_o_o_i, factor=64)
conv2d_nchw_ff_o_o_o_o, conv2d_nchw_ff_o_o_o_i = s[conv2d_nchw].split(conv2d_nchw_ff_o_o_o_i, factor=1)
conv2d_nchw_yy_o_i, conv2d_nchw_yy_i = s[conv2d_nchw].split(conv2d_nchw_yy, factor=1)
conv2d_nchw_yy_o_o_i, conv2d_nchw_yy_o_i = s[conv2d_nchw].split(conv2d_nchw_yy_o_i, factor=1)
conv2d_nchw_yy_o_o_o_i, conv2d_nchw_yy_o_o_i = s[conv2d_nchw].split(conv2d_nchw_yy_o_o_i, factor=1)
conv2d_nchw_yy_o_o_o_o, conv2d_nchw_yy_o_o_o_i = s[conv2d_nchw].split(conv2d_nchw_yy_o_o_o_i, factor=1)
conv2d_nchw_xx_o_i, conv2d_nchw_xx_i = s[conv2d_nchw].split(conv2d_nchw_xx, factor=1)
conv2d_nchw_xx_o_o_i, conv2d_nchw_xx_o_i = s[conv2d_nchw].split(conv2d_nchw_xx_o_i, factor=7)
conv2d_nchw_xx_o_o_o_i, conv2d_nchw_xx_o_o_i = s[conv2d_nchw].split(conv2d_nchw_xx_o_o_i, factor=1)
conv2d_nchw_xx_o_o_o_o, conv2d_nchw_xx_o_o_o_i = s[conv2d_nchw].split(conv2d_nchw_xx_o_o_o_i, factor=1)
conv2d_nchw_rc_o_i, conv2d_nchw_rc_i = s[conv2d_nchw].split(conv2d_nchw_rc, factor=2)
conv2d_nchw_rc_o_o, conv2d_nchw_rc_o_i = s[conv2d_nchw].split(conv2d_nchw_rc_o_i, factor=4)
conv2d_nchw_ry_o_i, conv2d_nchw_ry_i = s[conv2d_nchw].split(conv2d_nchw_ry, factor=1)
conv2d_nchw_ry_o_o, conv2d_nchw_ry_o_i = s[conv2d_nchw].split(conv2d_nchw_ry_o_i, factor=1)
conv2d_nchw_rx_o_i, conv2d_nchw_rx_i = s[conv2d_nchw].split(conv2d_nchw_rx, factor=1)
conv2d_nchw_rx_o_o, conv2d_nchw_rx_o_i = s[conv2d_nchw].split(conv2d_nchw_rx_o_i, factor=3)
s[conv2d_nchw].reorder(conv2d_nchw_nn_o_o_o_o, conv2d_nchw_ff_o_o_o_o, conv2d_nchw_yy_o_o_o_o, conv2d_nchw_xx_o_o_o_o, conv2d_nchw_nn_o_o_o_i, conv2d_nchw_ff_o_o_o_i, conv2d_nchw_yy_o_o_o_i, conv2d_nchw_xx_o_o_o_i, conv2d_nchw_nn_o_o_i, conv2d_nchw_ff_o_o_i, conv2d_nchw_yy_o_o_i, conv2d_nchw_xx_o_o_i, conv2d_nchw_rc_o_o, conv2d_nchw_ry_o_o, conv2d_nchw_rx_o_o, conv2d_nchw_rc_o_i, conv2d_nchw_ry_o_i, conv2d_nchw_rx_o_i, conv2d_nchw_nn_o_i, conv2d_nchw_ff_o_i, conv2d_nchw_yy_o_i, conv2d_nchw_xx_o_i, conv2d_nchw_rc_i, conv2d_nchw_ry_i, conv2d_nchw_rx_i, conv2d_nchw_nn_i, conv2d_nchw_ff_i, conv2d_nchw_yy_i, conv2d_nchw_xx_i)
compute_i0_o_i, compute_i0_i = s[compute].split(compute_i0, factor=1)
compute_i0_o_o_i, compute_i0_o_i = s[compute].split(compute_i0_o_i, factor=1)
compute_i0_o_o_o, compute_i0_o_o_i = s[compute].split(compute_i0_o_o_i, factor=1)
compute_i1_o_i, compute_i1_i = s[compute].split(compute_i1, factor=2)
compute_i1_o_o_i, compute_i1_o_i = s[compute].split(compute_i1_o_i, factor=64)
compute_i1_o_o_o, compute_i1_o_o_i = s[compute].split(compute_i1_o_o_i, factor=1)
compute_i2_o_i, compute_i2_i = s[compute].split(compute_i2, factor=1)
compute_i2_o_o_i, compute_i2_o_i = s[compute].split(compute_i2_o_i, factor=1)
compute_i2_o_o_o, compute_i2_o_o_i = s[compute].split(compute_i2_o_o_i, factor=1)
compute_i3_o_i, compute_i3_i = s[compute].split(compute_i3, factor=7)
compute_i3_o_o_i, compute_i3_o_i = s[compute].split(compute_i3_o_i, factor=1)
compute_i3_o_o_o, compute_i3_o_o_i = s[compute].split(compute_i3_o_o_i, factor=1)
s[compute].reorder(compute_i0_o_o_o, compute_i1_o_o_o, compute_i2_o_o_o, compute_i3_o_o_o, compute_i0_o_o_i, compute_i1_o_o_i, compute_i2_o_o_i, compute_i3_o_o_i, compute_i0_o_i, compute_i1_o_i, compute_i2_o_i, compute_i3_o_i, compute_i0_i, compute_i1_i, compute_i2_i, compute_i3_i)
s[conv2d_nchw].compute_at(s[compute], compute_i3_o_i)
kernel_shared = s.cache_read(kernel, "shared", [conv2d_nchw])
kernel_shared_ax0, kernel_shared_ax1, kernel_shared_ax2, kernel_shared_ax3 = tuple(kernel_shared.op.axis)
s[kernel_shared].compute_at(s[conv2d_nchw], conv2d_nchw_rx_o_o)
pad_temp_shared = s.cache_read(pad_temp, "shared", [conv2d_nchw])
pad_temp_shared_ax0, pad_temp_shared_ax1, pad_temp_shared_ax2, pad_temp_shared_ax3 = tuple(pad_temp_shared.op.axis)
s[pad_temp_shared].compute_at(s[conv2d_nchw], conv2d_nchw_rx_o_o)
s[pad_temp].compute_inline()
compute_i0_o_o_o_i1_o_o_o_fused_i2_o_o_o_fused_i3_o_o_o_fused = s[compute].fuse(compute_i0_o_o_o, compute_i1_o_o_o, compute_i2_o_o_o, compute_i3_o_o_o)
s[compute].bind(compute_i0_o_o_o_i1_o_o_o_fused_i2_o_o_o_fused_i3_o_o_o_fused, te.thread_axis("blockIdx.x"))
compute_i0_o_o_i_i1_o_o_i_fused_i2_o_o_i_fused_i3_o_o_i_fused = s[compute].fuse(compute_i0_o_o_i, compute_i1_o_o_i, compute_i2_o_o_i, compute_i3_o_o_i)
s[compute].bind(compute_i0_o_o_i_i1_o_o_i_fused_i2_o_o_i_fused_i3_o_o_i_fused, te.thread_axis("vthread"))
compute_i0_o_i_i1_o_i_fused_i2_o_i_fused_i3_o_i_fused = s[compute].fuse(compute_i0_o_i, compute_i1_o_i, compute_i2_o_i, compute_i3_o_i)
s[compute].bind(compute_i0_o_i_i1_o_i_fused_i2_o_i_fused_i3_o_i_fused, te.thread_axis("threadIdx.x"))
kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused = s[kernel_shared].fuse(kernel_shared_ax0, kernel_shared_ax1, kernel_shared_ax2, kernel_shared_ax3)
kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o, kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_i = s[kernel_shared].split(kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused, factor=1)
s[kernel_shared].vectorize(kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_i)
kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_o, kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_i = s[kernel_shared].split(kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o, factor=64)
s[kernel_shared].bind(kernel_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_i, te.thread_axis("threadIdx.x"))
pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused = s[pad_temp_shared].fuse(pad_temp_shared_ax0, pad_temp_shared_ax1, pad_temp_shared_ax2, pad_temp_shared_ax3)
pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o, pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_i = s[pad_temp_shared].split(pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused, factor=4)
s[pad_temp_shared].vectorize(pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_i)
pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_o, pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_i = s[pad_temp_shared].split(pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o, factor=64)
s[pad_temp_shared].bind(pad_temp_shared_ax0_ax1_fused_ax2_fused_ax3_fused_o_i, te.thread_axis("threadIdx.x"))
s[conv2d_nchw].pragma(conv2d_nchw_nn_o_o_o_o, "auto_unroll_max_step", 512)
s[conv2d_nchw].pragma(conv2d_nchw_nn_o_o_o_o, "unroll_explicit", True)

CUDA source code:

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
extern "C" __global__ void __launch_bounds__(64) default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[14];
  __shared__ float pad_temp_shared[72];
  __shared__ float kernel_shared[3072];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    for (int ry_outer_outer = 0; ry_outer_outer < 3; ++ry_outer_outer) {
      __syncthreads();
      if (((int)threadIdx.x) < 18) {
        pad_temp_shared[(((int)threadIdx.x) * 4)] = (((((1 <= (ry_outer_outer + (((int)blockIdx.x) % 7))) && ((ry_outer_outer + (((int)blockIdx.x) % 7)) < 8)) && (1 <= ((((int)threadIdx.x) * 4) % 9))) && (((((int)threadIdx.x) * 4) % 9) < 8)) ? data[((((((rc_outer_outer * 392) + (((((int)threadIdx.x) * 4) / 9) * 49)) + (ry_outer_outer * 7)) + ((((int)blockIdx.x) % 7) * 7)) + ((((int)threadIdx.x) * 4) % 9)) - 8)] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 18) {
        pad_temp_shared[((((int)threadIdx.x) * 4) + 1)] = (((((1 <= (ry_outer_outer + (((int)blockIdx.x) % 7))) && ((ry_outer_outer + (((int)blockIdx.x) % 7)) < 8)) && (1 <= (((((int)threadIdx.x) * 4) + 1) % 9))) && ((((((int)threadIdx.x) * 4) + 1) % 9) < 8)) ? data[((((((rc_outer_outer * 392) + ((((((int)threadIdx.x) * 4) + 1) / 9) * 49)) + (ry_outer_outer * 7)) + ((((int)blockIdx.x) % 7) * 7)) + (((((int)threadIdx.x) * 4) + 1) % 9)) - 8)] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 18) {
        pad_temp_shared[((((int)threadIdx.x) * 4) + 2)] = (((((1 <= (ry_outer_outer + (((int)blockIdx.x) % 7))) && ((ry_outer_outer + (((int)blockIdx.x) % 7)) < 8)) && (1 <= (((((int)threadIdx.x) * 4) + 2) % 9))) && ((((((int)threadIdx.x) * 4) + 2) % 9) < 8)) ? data[((((((rc_outer_outer * 392) + ((((((int)threadIdx.x) * 4) + 2) / 9) * 49)) + (ry_outer_outer * 7)) + ((((int)blockIdx.x) % 7) * 7)) + (((((int)threadIdx.x) * 4) + 2) % 9)) - 8)] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 18) {
        pad_temp_shared[((((int)threadIdx.x) * 4) + 3)] = (((((1 <= (ry_outer_outer + (((int)blockIdx.x) % 7))) && ((ry_outer_outer + (((int)blockIdx.x) % 7)) < 8)) && (1 <= (((((int)threadIdx.x) * 4) + 3) % 9))) && ((((((int)threadIdx.x) * 4) + 3) % 9) < 8)) ? data[((((((rc_outer_outer * 392) + ((((((int)threadIdx.x) * 4) + 3) / 9) * 49)) + (ry_outer_outer * 7)) + ((((int)blockIdx.x) % 7) * 7)) + (((((int)threadIdx.x) * 4) + 3) % 9)) - 8)] : 0.000000e+00f);
      }
      kernel_shared[((int)threadIdx.x)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3))];
      kernel_shared[(((int)threadIdx.x) + 64)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 64) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 128)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 128) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 192)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 36864)];
      kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 256) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 320)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 320) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 384)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 73728)];
      kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 448) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 512) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 576)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 110592)];
      kernel_shared[(((int)threadIdx.x) + 640)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 640) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 704)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 704) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 768)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 147456)];
      kernel_shared[(((int)threadIdx.x) + 832)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 832) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 896) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 960)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 184320)];
      kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1024) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 1088)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1088) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 221184)];
      kernel_shared[(((int)threadIdx.x) + 1216)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1216) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1280) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 258048)];
      kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1408) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 1472)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1472) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 294912)];
      kernel_shared[(((int)threadIdx.x) + 1600)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1600) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1664) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 1728)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 331776)];
      kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1792) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 1856)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1856) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 368640)];
      kernel_shared[(((int)threadIdx.x) + 1984)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 1984) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2048) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 2112)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 405504)];
      kernel_shared[(((int)threadIdx.x) + 2176)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2176) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2240) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 442368)];
      kernel_shared[(((int)threadIdx.x) + 2368)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2368) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 2432)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2432) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 2496)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 479232)];
      kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2560) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 2624)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2624) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 516096)];
      kernel_shared[(((int)threadIdx.x) + 2752)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2752) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2816) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      kernel_shared[(((int)threadIdx.x) + 2880)] = kernel[((((((((((int)blockIdx.x) / 7) * 589824) + ((((int)threadIdx.x) / 24) * 4608)) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + (((int)threadIdx.x) % 3)) + 552960)];
      kernel_shared[(((int)threadIdx.x) + 2944)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 2944) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 16) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 1) % 3))];
      kernel_shared[(((int)threadIdx.x) + 3008)] = kernel[(((((((((int)blockIdx.x) / 7) * 589824) + (((((int)threadIdx.x) + 3008) / 24) * 4608)) + (rc_outer_outer * 72)) + ((((((int)threadIdx.x) + 8) % 24) / 3) * 9)) + (ry_outer_outer * 3)) + ((((int)threadIdx.x) + 2) % 3))];
      __syncthreads();
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.x) * 48)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.x) * 48) + 3)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[1] * kernel_shared[(((int)threadIdx.x) * 48)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[10] * kernel_shared[((((int)threadIdx.x) * 48) + 3)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.x) * 48)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 48) + 3)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[3] * kernel_shared[(((int)threadIdx.x) * 48)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 48) + 3)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.x) * 48)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 48) + 3)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[5] * kernel_shared[(((int)threadIdx.x) * 48)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 48) + 3)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.x) * 48)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 48) + 3)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.x) * 48) + 24)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.x) * 48) + 27)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[1] * kernel_shared[((((int)threadIdx.x) * 48) + 24)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[10] * kernel_shared[((((int)threadIdx.x) * 48) + 27)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.x) * 48) + 24)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 48) + 27)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[3] * kernel_shared[((((int)threadIdx.x) * 48) + 24)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 48) + 27)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 48) + 24)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 48) + 27)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 48) + 24)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 48) + 27)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 48) + 24)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 48) + 27)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[1] * kernel_shared[((((int)threadIdx.x) * 48) + 1)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[10] * kernel_shared[((((int)threadIdx.x) * 48) + 4)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.x) * 48) + 1)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 48) + 4)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[3] * kernel_shared[((((int)threadIdx.x) * 48) + 1)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 48) + 4)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 48) + 1)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 48) + 4)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 48) + 1)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 48) + 4)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 48) + 1)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 48) + 4)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[7] * kernel_shared[((((int)threadIdx.x) * 48) + 1)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[16] * kernel_shared[((((int)threadIdx.x) * 48) + 4)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[1] * kernel_shared[((((int)threadIdx.x) * 48) + 25)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[10] * kernel_shared[((((int)threadIdx.x) * 48) + 28)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.x) * 48) + 25)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 48) + 28)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[3] * kernel_shared[((((int)threadIdx.x) * 48) + 25)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 48) + 28)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 48) + 25)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 48) + 28)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 48) + 25)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 48) + 28)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 48) + 25)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 48) + 28)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[7] * kernel_shared[((((int)threadIdx.x) * 48) + 25)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[16] * kernel_shared[((((int)threadIdx.x) * 48) + 28)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.x) * 48) + 2)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 48) + 5)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[3] * kernel_shared[((((int)threadIdx.x) * 48) + 2)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 48) + 5)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 48) + 2)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 48) + 5)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 48) + 2)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 48) + 5)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 48) + 2)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 48) + 5)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[7] * kernel_shared[((((int)threadIdx.x) * 48) + 2)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[16] * kernel_shared[((((int)threadIdx.x) * 48) + 5)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.x) * 48) + 2)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.x) * 48) + 5)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.x) * 48) + 26)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.x) * 48) + 29)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[3] * kernel_shared[((((int)threadIdx.x) * 48) + 26)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[12] * kernel_shared[((((int)threadIdx.x) * 48) + 29)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.x) * 48) + 26)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.x) * 48) + 29)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[5] * kernel_shared[((((int)threadIdx.x) * 48) + 26)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[14] * kernel_shared[((((int)threadIdx.x) * 48) + 29)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.x) * 48) + 26)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.x) * 48) + 29)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[7] * kernel_shared[((((int)threadIdx.x) * 48) + 26)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[16] * kernel_shared[((((int)threadIdx.x) * 48) + 29)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.x) * 48) + 26)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.x) * 48) + 29)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[18] * kernel_shared[((((int)threadIdx.x) * 48) + 6)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[27] * kernel_shared[((((int)threadIdx.x) * 48) + 9)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[19] * kernel_shared[((((int)threadIdx.x) * 48) + 6)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[28] * kernel_shared[((((int)threadIdx.x) * 48) + 9)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[20] * kernel_shared[((((int)threadIdx.x) * 48) + 6)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[29] * kernel_shared[((((int)threadIdx.x) * 48) + 9)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 48) + 6)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[30] * kernel_shared[((((int)threadIdx.x) * 48) + 9)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[22] * kernel_shared[((((int)threadIdx.x) * 48) + 6)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[31] * kernel_shared[((((int)threadIdx.x) * 48) + 9)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 48) + 6)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[32] * kernel_shared[((((int)threadIdx.x) * 48) + 9)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[24] * kernel_shared[((((int)threadIdx.x) * 48) + 6)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[33] * kernel_shared[((((int)threadIdx.x) * 48) + 9)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[18] * kernel_shared[((((int)threadIdx.x) * 48) + 30)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[27] * kernel_shared[((((int)threadIdx.x) * 48) + 33)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[19] * kernel_shared[((((int)threadIdx.x) * 48) + 30)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[28] * kernel_shared[((((int)threadIdx.x) * 48) + 33)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[20] * kernel_shared[((((int)threadIdx.x) * 48) + 30)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[29] * kernel_shared[((((int)threadIdx.x) * 48) + 33)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 48) + 30)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[30] * kernel_shared[((((int)threadIdx.x) * 48) + 33)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[22] * kernel_shared[((((int)threadIdx.x) * 48) + 30)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[31] * kernel_shared[((((int)threadIdx.x) * 48) + 33)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 48) + 30)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[32] * kernel_shared[((((int)threadIdx.x) * 48) + 33)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[24] * kernel_shared[((((int)threadIdx.x) * 48) + 30)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[33] * kernel_shared[((((int)threadIdx.x) * 48) + 33)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[19] * kernel_shared[((((int)threadIdx.x) * 48) + 7)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[28] * kernel_shared[((((int)threadIdx.x) * 48) + 10)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[20] * kernel_shared[((((int)threadIdx.x) * 48) + 7)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[29] * kernel_shared[((((int)threadIdx.x) * 48) + 10)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 48) + 7)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[30] * kernel_shared[((((int)threadIdx.x) * 48) + 10)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[22] * kernel_shared[((((int)threadIdx.x) * 48) + 7)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[31] * kernel_shared[((((int)threadIdx.x) * 48) + 10)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 48) + 7)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[32] * kernel_shared[((((int)threadIdx.x) * 48) + 10)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[24] * kernel_shared[((((int)threadIdx.x) * 48) + 7)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[33] * kernel_shared[((((int)threadIdx.x) * 48) + 10)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[25] * kernel_shared[((((int)threadIdx.x) * 48) + 7)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[34] * kernel_shared[((((int)threadIdx.x) * 48) + 10)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[19] * kernel_shared[((((int)threadIdx.x) * 48) + 31)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[28] * kernel_shared[((((int)threadIdx.x) * 48) + 34)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[20] * kernel_shared[((((int)threadIdx.x) * 48) + 31)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[29] * kernel_shared[((((int)threadIdx.x) * 48) + 34)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 48) + 31)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[30] * kernel_shared[((((int)threadIdx.x) * 48) + 34)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[22] * kernel_shared[((((int)threadIdx.x) * 48) + 31)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[31] * kernel_shared[((((int)threadIdx.x) * 48) + 34)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 48) + 31)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[32] * kernel_shared[((((int)threadIdx.x) * 48) + 34)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[24] * kernel_shared[((((int)threadIdx.x) * 48) + 31)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[33] * kernel_shared[((((int)threadIdx.x) * 48) + 34)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[25] * kernel_shared[((((int)threadIdx.x) * 48) + 31)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[34] * kernel_shared[((((int)threadIdx.x) * 48) + 34)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[20] * kernel_shared[((((int)threadIdx.x) * 48) + 8)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[29] * kernel_shared[((((int)threadIdx.x) * 48) + 11)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 48) + 8)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[30] * kernel_shared[((((int)threadIdx.x) * 48) + 11)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[22] * kernel_shared[((((int)threadIdx.x) * 48) + 8)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[31] * kernel_shared[((((int)threadIdx.x) * 48) + 11)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 48) + 8)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[32] * kernel_shared[((((int)threadIdx.x) * 48) + 11)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[24] * kernel_shared[((((int)threadIdx.x) * 48) + 8)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[33] * kernel_shared[((((int)threadIdx.x) * 48) + 11)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[25] * kernel_shared[((((int)threadIdx.x) * 48) + 8)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[34] * kernel_shared[((((int)threadIdx.x) * 48) + 11)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[26] * kernel_shared[((((int)threadIdx.x) * 48) + 8)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[35] * kernel_shared[((((int)threadIdx.x) * 48) + 11)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[20] * kernel_shared[((((int)threadIdx.x) * 48) + 32)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[29] * kernel_shared[((((int)threadIdx.x) * 48) + 35)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[21] * kernel_shared[((((int)threadIdx.x) * 48) + 32)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[30] * kernel_shared[((((int)threadIdx.x) * 48) + 35)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[22] * kernel_shared[((((int)threadIdx.x) * 48) + 32)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[31] * kernel_shared[((((int)threadIdx.x) * 48) + 35)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[23] * kernel_shared[((((int)threadIdx.x) * 48) + 32)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[32] * kernel_shared[((((int)threadIdx.x) * 48) + 35)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[24] * kernel_shared[((((int)threadIdx.x) * 48) + 32)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[33] * kernel_shared[((((int)threadIdx.x) * 48) + 35)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[25] * kernel_shared[((((int)threadIdx.x) * 48) + 32)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[34] * kernel_shared[((((int)threadIdx.x) * 48) + 35)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[26] * kernel_shared[((((int)threadIdx.x) * 48) + 32)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[35] * kernel_shared[((((int)threadIdx.x) * 48) + 35)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[36] * kernel_shared[((((int)threadIdx.x) * 48) + 12)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[45] * kernel_shared[((((int)threadIdx.x) * 48) + 15)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[37] * kernel_shared[((((int)threadIdx.x) * 48) + 12)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[46] * kernel_shared[((((int)threadIdx.x) * 48) + 15)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[38] * kernel_shared[((((int)threadIdx.x) * 48) + 12)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[47] * kernel_shared[((((int)threadIdx.x) * 48) + 15)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[39] * kernel_shared[((((int)threadIdx.x) * 48) + 12)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[48] * kernel_shared[((((int)threadIdx.x) * 48) + 15)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[40] * kernel_shared[((((int)threadIdx.x) * 48) + 12)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[49] * kernel_shared[((((int)threadIdx.x) * 48) + 15)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[41] * kernel_shared[((((int)threadIdx.x) * 48) + 12)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[50] * kernel_shared[((((int)threadIdx.x) * 48) + 15)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[42] * kernel_shared[((((int)threadIdx.x) * 48) + 12)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[51] * kernel_shared[((((int)threadIdx.x) * 48) + 15)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[36] * kernel_shared[((((int)threadIdx.x) * 48) + 36)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[45] * kernel_shared[((((int)threadIdx.x) * 48) + 39)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[37] * kernel_shared[((((int)threadIdx.x) * 48) + 36)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[46] * kernel_shared[((((int)threadIdx.x) * 48) + 39)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[38] * kernel_shared[((((int)threadIdx.x) * 48) + 36)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[47] * kernel_shared[((((int)threadIdx.x) * 48) + 39)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[39] * kernel_shared[((((int)threadIdx.x) * 48) + 36)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[48] * kernel_shared[((((int)threadIdx.x) * 48) + 39)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[40] * kernel_shared[((((int)threadIdx.x) * 48) + 36)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[49] * kernel_shared[((((int)threadIdx.x) * 48) + 39)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[41] * kernel_shared[((((int)threadIdx.x) * 48) + 36)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[50] * kernel_shared[((((int)threadIdx.x) * 48) + 39)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[42] * kernel_shared[((((int)threadIdx.x) * 48) + 36)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[51] * kernel_shared[((((int)threadIdx.x) * 48) + 39)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[37] * kernel_shared[((((int)threadIdx.x) * 48) + 13)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[46] * kernel_shared[((((int)threadIdx.x) * 48) + 16)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[38] * kernel_shared[((((int)threadIdx.x) * 48) + 13)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[47] * kernel_shared[((((int)threadIdx.x) * 48) + 16)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[39] * kernel_shared[((((int)threadIdx.x) * 48) + 13)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[48] * kernel_shared[((((int)threadIdx.x) * 48) + 16)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[40] * kernel_shared[((((int)threadIdx.x) * 48) + 13)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[49] * kernel_shared[((((int)threadIdx.x) * 48) + 16)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[41] * kernel_shared[((((int)threadIdx.x) * 48) + 13)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[50] * kernel_shared[((((int)threadIdx.x) * 48) + 16)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[42] * kernel_shared[((((int)threadIdx.x) * 48) + 13)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[51] * kernel_shared[((((int)threadIdx.x) * 48) + 16)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[43] * kernel_shared[((((int)threadIdx.x) * 48) + 13)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[52] * kernel_shared[((((int)threadIdx.x) * 48) + 16)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[37] * kernel_shared[((((int)threadIdx.x) * 48) + 37)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[46] * kernel_shared[((((int)threadIdx.x) * 48) + 40)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[38] * kernel_shared[((((int)threadIdx.x) * 48) + 37)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[47] * kernel_shared[((((int)threadIdx.x) * 48) + 40)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[39] * kernel_shared[((((int)threadIdx.x) * 48) + 37)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[48] * kernel_shared[((((int)threadIdx.x) * 48) + 40)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[40] * kernel_shared[((((int)threadIdx.x) * 48) + 37)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[49] * kernel_shared[((((int)threadIdx.x) * 48) + 40)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[41] * kernel_shared[((((int)threadIdx.x) * 48) + 37)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[50] * kernel_shared[((((int)threadIdx.x) * 48) + 40)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[42] * kernel_shared[((((int)threadIdx.x) * 48) + 37)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[51] * kernel_shared[((((int)threadIdx.x) * 48) + 40)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[43] * kernel_shared[((((int)threadIdx.x) * 48) + 37)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[52] * kernel_shared[((((int)threadIdx.x) * 48) + 40)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[38] * kernel_shared[((((int)threadIdx.x) * 48) + 14)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[47] * kernel_shared[((((int)threadIdx.x) * 48) + 17)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[39] * kernel_shared[((((int)threadIdx.x) * 48) + 14)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[48] * kernel_shared[((((int)threadIdx.x) * 48) + 17)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[40] * kernel_shared[((((int)threadIdx.x) * 48) + 14)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[49] * kernel_shared[((((int)threadIdx.x) * 48) + 17)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[41] * kernel_shared[((((int)threadIdx.x) * 48) + 14)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[50] * kernel_shared[((((int)threadIdx.x) * 48) + 17)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[42] * kernel_shared[((((int)threadIdx.x) * 48) + 14)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[51] * kernel_shared[((((int)threadIdx.x) * 48) + 17)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[43] * kernel_shared[((((int)threadIdx.x) * 48) + 14)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[52] * kernel_shared[((((int)threadIdx.x) * 48) + 17)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[44] * kernel_shared[((((int)threadIdx.x) * 48) + 14)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[53] * kernel_shared[((((int)threadIdx.x) * 48) + 17)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[38] * kernel_shared[((((int)threadIdx.x) * 48) + 38)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[47] * kernel_shared[((((int)threadIdx.x) * 48) + 41)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[39] * kernel_shared[((((int)threadIdx.x) * 48) + 38)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[48] * kernel_shared[((((int)threadIdx.x) * 48) + 41)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[40] * kernel_shared[((((int)threadIdx.x) * 48) + 38)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[49] * kernel_shared[((((int)threadIdx.x) * 48) + 41)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[41] * kernel_shared[((((int)threadIdx.x) * 48) + 38)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[50] * kernel_shared[((((int)threadIdx.x) * 48) + 41)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[42] * kernel_shared[((((int)threadIdx.x) * 48) + 38)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[51] * kernel_shared[((((int)threadIdx.x) * 48) + 41)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[43] * kernel_shared[((((int)threadIdx.x) * 48) + 38)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[52] * kernel_shared[((((int)threadIdx.x) * 48) + 41)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[44] * kernel_shared[((((int)threadIdx.x) * 48) + 38)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[53] * kernel_shared[((((int)threadIdx.x) * 48) + 41)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[54] * kernel_shared[((((int)threadIdx.x) * 48) + 18)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[63] * kernel_shared[((((int)threadIdx.x) * 48) + 21)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[55] * kernel_shared[((((int)threadIdx.x) * 48) + 18)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[64] * kernel_shared[((((int)threadIdx.x) * 48) + 21)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[56] * kernel_shared[((((int)threadIdx.x) * 48) + 18)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[65] * kernel_shared[((((int)threadIdx.x) * 48) + 21)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[57] * kernel_shared[((((int)threadIdx.x) * 48) + 18)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[66] * kernel_shared[((((int)threadIdx.x) * 48) + 21)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[58] * kernel_shared[((((int)threadIdx.x) * 48) + 18)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[67] * kernel_shared[((((int)threadIdx.x) * 48) + 21)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[59] * kernel_shared[((((int)threadIdx.x) * 48) + 18)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[68] * kernel_shared[((((int)threadIdx.x) * 48) + 21)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[60] * kernel_shared[((((int)threadIdx.x) * 48) + 18)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[69] * kernel_shared[((((int)threadIdx.x) * 48) + 21)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[54] * kernel_shared[((((int)threadIdx.x) * 48) + 42)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[63] * kernel_shared[((((int)threadIdx.x) * 48) + 45)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[55] * kernel_shared[((((int)threadIdx.x) * 48) + 42)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[64] * kernel_shared[((((int)threadIdx.x) * 48) + 45)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[56] * kernel_shared[((((int)threadIdx.x) * 48) + 42)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[65] * kernel_shared[((((int)threadIdx.x) * 48) + 45)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[57] * kernel_shared[((((int)threadIdx.x) * 48) + 42)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[66] * kernel_shared[((((int)threadIdx.x) * 48) + 45)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[58] * kernel_shared[((((int)threadIdx.x) * 48) + 42)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[67] * kernel_shared[((((int)threadIdx.x) * 48) + 45)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[59] * kernel_shared[((((int)threadIdx.x) * 48) + 42)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[68] * kernel_shared[((((int)threadIdx.x) * 48) + 45)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[60] * kernel_shared[((((int)threadIdx.x) * 48) + 42)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[69] * kernel_shared[((((int)threadIdx.x) * 48) + 45)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[55] * kernel_shared[((((int)threadIdx.x) * 48) + 19)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[64] * kernel_shared[((((int)threadIdx.x) * 48) + 22)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[56] * kernel_shared[((((int)threadIdx.x) * 48) + 19)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[65] * kernel_shared[((((int)threadIdx.x) * 48) + 22)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[57] * kernel_shared[((((int)threadIdx.x) * 48) + 19)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[66] * kernel_shared[((((int)threadIdx.x) * 48) + 22)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[58] * kernel_shared[((((int)threadIdx.x) * 48) + 19)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[67] * kernel_shared[((((int)threadIdx.x) * 48) + 22)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[59] * kernel_shared[((((int)threadIdx.x) * 48) + 19)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[68] * kernel_shared[((((int)threadIdx.x) * 48) + 22)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[60] * kernel_shared[((((int)threadIdx.x) * 48) + 19)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[69] * kernel_shared[((((int)threadIdx.x) * 48) + 22)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[61] * kernel_shared[((((int)threadIdx.x) * 48) + 19)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[70] * kernel_shared[((((int)threadIdx.x) * 48) + 22)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[55] * kernel_shared[((((int)threadIdx.x) * 48) + 43)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[64] * kernel_shared[((((int)threadIdx.x) * 48) + 46)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[56] * kernel_shared[((((int)threadIdx.x) * 48) + 43)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[65] * kernel_shared[((((int)threadIdx.x) * 48) + 46)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[57] * kernel_shared[((((int)threadIdx.x) * 48) + 43)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[66] * kernel_shared[((((int)threadIdx.x) * 48) + 46)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[58] * kernel_shared[((((int)threadIdx.x) * 48) + 43)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[67] * kernel_shared[((((int)threadIdx.x) * 48) + 46)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[59] * kernel_shared[((((int)threadIdx.x) * 48) + 43)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[68] * kernel_shared[((((int)threadIdx.x) * 48) + 46)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[60] * kernel_shared[((((int)threadIdx.x) * 48) + 43)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[69] * kernel_shared[((((int)threadIdx.x) * 48) + 46)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[61] * kernel_shared[((((int)threadIdx.x) * 48) + 43)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[70] * kernel_shared[((((int)threadIdx.x) * 48) + 46)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[56] * kernel_shared[((((int)threadIdx.x) * 48) + 20)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[65] * kernel_shared[((((int)threadIdx.x) * 48) + 23)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[57] * kernel_shared[((((int)threadIdx.x) * 48) + 20)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[66] * kernel_shared[((((int)threadIdx.x) * 48) + 23)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[58] * kernel_shared[((((int)threadIdx.x) * 48) + 20)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[67] * kernel_shared[((((int)threadIdx.x) * 48) + 23)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[59] * kernel_shared[((((int)threadIdx.x) * 48) + 20)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[68] * kernel_shared[((((int)threadIdx.x) * 48) + 23)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[60] * kernel_shared[((((int)threadIdx.x) * 48) + 20)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[69] * kernel_shared[((((int)threadIdx.x) * 48) + 23)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[61] * kernel_shared[((((int)threadIdx.x) * 48) + 20)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[70] * kernel_shared[((((int)threadIdx.x) * 48) + 23)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[62] * kernel_shared[((((int)threadIdx.x) * 48) + 20)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[71] * kernel_shared[((((int)threadIdx.x) * 48) + 23)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[56] * kernel_shared[((((int)threadIdx.x) * 48) + 44)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[65] * kernel_shared[((((int)threadIdx.x) * 48) + 47)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[57] * kernel_shared[((((int)threadIdx.x) * 48) + 44)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[66] * kernel_shared[((((int)threadIdx.x) * 48) + 47)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[58] * kernel_shared[((((int)threadIdx.x) * 48) + 44)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[67] * kernel_shared[((((int)threadIdx.x) * 48) + 47)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[59] * kernel_shared[((((int)threadIdx.x) * 48) + 44)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[68] * kernel_shared[((((int)threadIdx.x) * 48) + 47)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[60] * kernel_shared[((((int)threadIdx.x) * 48) + 44)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[69] * kernel_shared[((((int)threadIdx.x) * 48) + 47)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[61] * kernel_shared[((((int)threadIdx.x) * 48) + 44)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[70] * kernel_shared[((((int)threadIdx.x) * 48) + 47)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[62] * kernel_shared[((((int)threadIdx.x) * 48) + 44)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[71] * kernel_shared[((((int)threadIdx.x) * 48) + 47)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    for (int i3_inner = 0; i3_inner < 7; ++i3_inner) {
      compute[((((((((int)blockIdx.x) / 7) * 6272) + (((int)threadIdx.x) * 98)) + (i1_inner * 49)) + ((((int)blockIdx.x) % 7) * 7)) + i3_inner)] = max((conv2d_nchw[((i1_inner * 7) + i3_inner)] + bias[((((((int)blockIdx.x) / 7) * 128) + (((int)threadIdx.x) * 2)) + i1_inner)]), 0.000000e+00f);
    }
  }
}
```

以下是恢复搜索的例子。这种情况下要自行创建搜索策略和 cost 模型，并通过日志文件恢复搜索策略和 cost 模型的状态。以下示例中，我们恢复状态，并多进行 5 次训练。

``` python
def resume_search(task, log_file):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    task.tune(tune_option, search_policy=search_policy)

    # 终止测试过程
    del measure_ctx

#我们不再在服务器上运行调优，因为太耗时间了
#去掉下行代码的注释自行运行
#resume_search(task, log_file)
```

输出结果：

``` bash
Resume search:
/usr/local/lib/python3.7/dist-packages/xgboost/training.py:17: UserWarning: Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
  warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
Get devices for measurement successfully!
```

**脚本总运行时长：**（ 3 分 12.949 秒）

[下载 Python 源代码：tune_conv2d_layer_cuda.py](https://tvm.apache.org/docs/_downloads/e3e540f3b477c0c52d8eb73e674e8ffd/tune_conv2d_layer_cuda.py)

[下载 Jupyter notebook：tune_conv2d_layer_cuda.ipynb](https://tvm.apache.org/docs/_downloads/5f1f7bd7d90710fd404f7bcdc4965622/tune_conv2d_layer_cuda.ipynb)
