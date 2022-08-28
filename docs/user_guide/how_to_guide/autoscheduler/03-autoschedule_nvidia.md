---
title: 为 NVIDIA GPU 自动调度神经网络
---

# 为 NVIDIA GPU 自动调度神经网络

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_network_cuda.html#sphx-glr-download-how-to-tune-with-autoscheduler-tune-network-cuda-py) 下载完整的示例代码
:::

**作者**：[Lianmin Zheng](https://github.com/merrymercy)

针对特定设备和工作负载的自动调优对于获得最佳性能至关重要。本文介绍如何使用 auto-scheduler 为 NVIDIA GPU 调优整个神经网络。

为自动调优神经网络，需要将网络划分为小的子图并独立调优。每个子图被视为一个搜索任务，任务调度器对时间进行切片并动态地为这些任务分配时间资源，并预测每个任务对端到端执行时间的影响，优先考虑最能减少执行时间的任务。

对于每个子图，使用 `tvm/python/topi` 中的计算声明来获取张量表达式形式的计算 DAG。然后用 auto-scheduler 来构建这个 DAG 的搜索空间，并搜索合适的调度（低级优化）。

与基于 template 的 [AutoTVM](../autotune)（依赖手动 template 来定义搜索空间的） 不同，auto-scheduler 无需任何调度 template。换言之，auto-scheduler 只使用 `tvm/python/topi` 中的计算声明，不使用现有的调度 template。

注意，本教程无法在 Windows 或最新版本的 macOS 上运行。如需运行，请将本教程的主体放在 `if __name__ == "__main__":` 代码块中。

``` python
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
```

## 定义网络

首先，要用 Relay 前端 API 定义网络。可以从 `tvm.relay.testing` 加载一些预定义的网络。也可以从 MXNet、ONNX、PyTorch 和 TensorFlow 加载模型（参见 [前端教程](../compile)）。

对于卷积神经网络，尽管 auto-scheduler 可以在任何布局下正常运行，但通过 NHWC 布局实现的性能最佳。auto-scheduler 对 NHWC 布局进行了很多优化，因此推荐将模型转换为 NHWC 布局，从而得以使用 auto-scheduler。可用 [ConvertLayout](https://tvm.apache.org/docs/arch/convert_layout.html#convert-layout-usage) pass 在 TVM 中进行布局转换。

``` python
def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler 更适合 NHWC 布局
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # MXNet 模型的示例
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    return mod, params, input_shape, output_shape

# 定义神经网络和编译目标
network = "resnet-18"
batch_size = 1
layout = "NHWC"
target = tvm.target.Target("cuda")
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
```

## 提取搜索任务

接下来，从网络中提取搜索任务及其权重。任务的权重是任务的子图在整个网络中出现的次数。通过使用权重，可以将网络的端到端延迟近似为 `sum(latency[t] * weight[t])`，其中 `latency[t]` 是任务的延迟，而 `weight[t]` 是任务的权重，任务调度器仅针对该目标进行优化。

``` python
# 从网络中提取任务
print("Extract tasks...")
mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
```

输出结果：

``` bash
Extract tasks...
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
========== Task 0  (workload key: ["8654f16aeddf785bad9f028164b3a48d", [1, 56, 56, 64], [1, 1, 64, 64], [1, 56, 56, 64]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 64, 64]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])

========== Task 1  (workload key: ["c4500b4e2fd04e695c32d2f31bbdc14a", [1, 28, 28, 128], [4, 4, 128, 128], [1, 28, 28, 128], [1, 1, 1, 128], [1, 28, 28, 128]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 29)) && (i2 >= 1)) && (i2 < 29)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 196), ((floormod(floordiv(p, 14), 14)*2) + eps), ((floormod(p, 14)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 128, 128]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*14)*14) + (floordiv(h, 2)*14)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 28, 28, 128]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 2  (workload key: ["06f578e6519a86e85028eecf4de64b25", [1, 56, 56, 64], [1, 1, 64, 128], [1, 28, 28, 128]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 64, 128]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])

========== Task 3  (workload key: ["b8b52b9be9df6102466a22a014c44c1f", [1, 14, 14, 256], [4, 4, 256, 256], [1, 1, 1, 256], [1, 14, 14, 256]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 15)) && (i2 >= 1)) && (i2 < 15)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 49), ((floormod(floordiv(p, 7), 7)*2) + eps), ((floormod(p, 7)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 256, 256]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*7)*7) + (floordiv(h, 2)*7)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 4  (workload key: ["e4cdf917b876dbdd64488c3818d9c141", [1, 28, 28, 128], [4, 4, 128, 128], [1, 1, 1, 128], [1, 28, 28, 128]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 29)) && (i2 >= 1)) && (i2 < 29)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 196), ((floormod(floordiv(p, 14), 14)*2) + eps), ((floormod(p, 14)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 128, 128]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*14)*14) + (floordiv(h, 2)*14)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 5  (workload key: ["d730bcd28f0920f6b97245e2a11bd8d6", [1, 7, 7, 512], [4, 4, 512, 512], [1, 7, 7, 512], [1, 7, 7, 512]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 512]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 8)) && (i2 >= 1)) && (i2 < 8)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 16), ((floormod(floordiv(p, 4), 4)*2) + eps), ((floormod(p, 4)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 512, 512]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*4)*4) + (floordiv(h, 2)*4)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 7, 7, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])

========== Task 6  (workload key: ["b818b53148cd450f86569dfc3e04cb8a", [1, 56, 56, 64], [6, 6, 64, 64], [1, 1, 1, 64], [1, 56, 56, 64]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 196), ((floormod(floordiv(p, 14), 14)*4) + eps), ((floormod(p, 14)*4) + nu), ci]
B(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 6) == 5)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 6) == 4)),  ..(OMITTED)..  (floormod(j, 6) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 6) == 0)), 1f, 0f))))))))))))))))))))))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [6, 6, 64, 64]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 4) == 2)),  ..(OMITTED)..  6) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 4), floormod(w, 4), ((((n*14)*14) + (floordiv(h, 4)*14)) + floordiv(w, 4)), co]
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 7  (workload key: ["ad6cecbf5d85cb1cda3c2bb7af170211", [1, 7, 7, 512], [4, 4, 512, 512], [1, 7, 7, 512], [1, 1, 1, 512], [1, 1, 1, 512], [1, 7, 7, 512]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 512]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 8)) && (i2 >= 1)) && (i2 < 8)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 16), ((floormod(floordiv(p, 4), 4)*2) + eps), ((floormod(p, 4)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 512, 512]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*4)*4) + (floordiv(h, 2)*4)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 7, 7, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_multiply(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3]*placeholder[ax0, 0, 0, ax3])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (T_multiply[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 8  (workload key: ["f3b6c10fcc6ce01ff01add933e4d21e9", [1, 14, 14, 256], [4, 4, 256, 256], [1, 14, 14, 256], [1, 1, 1, 256], [1, 14, 14, 256]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 15)) && (i2 >= 1)) && (i2 < 15)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 49), ((floormod(floordiv(p, 7), 7)*2) + eps), ((floormod(p, 7)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 256, 256]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*7)*7) + (floordiv(h, 2)*7)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 14, 14, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 9  (workload key: ["d7b65649a4dd54becea0a52aabbc5af5", [1, 1000], [1, 1000]]) ==========
placeholder = PLACEHOLDER [1, 1000]
T_softmax_maxelem(i0) max= placeholder[i0, k]
T_softmax_exp(i0, i1) = tir.exp((placeholder[i0, i1] - T_softmax_maxelem[i0]))
T_softmax_expsum(i0) += T_softmax_exp[i0, k]
T_softmax_norm(i0, i1) = (T_softmax_exp[i0, i1]/T_softmax_expsum[i0])

========== Task 10  (workload key: ["69115f188984ae34ede37c3b8ca40b43", [1, 7, 7, 512], [1, 1, 1, 512]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 512]
tensor(ax0, ax1, ax2, ax3) += placeholder[ax0, ((ax1*7) + rv0), ((ax2*7) + rv1), ax3]
tensor(ax0, ax1, ax2, ax3) = (tensor[ax0, ax1, ax2, ax3]/(float32((select((bool)1, ((ax1 + 1)*7), (((ax1 + 1)*7) + 1)) - (ax1*7)))*float32((select((bool)1, ((ax2 + 1)*7), (((ax2 + 1)*7) + 1)) - (ax2*7)))))

========== Task 11  (workload key: ["3a69f9fbc63760d99e36b4c17b3bfc57", [1, 7, 7, 512], [4, 4, 512, 512], [1, 1, 1, 512], [1, 7, 7, 512]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 512]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 8)) && (i2 >= 1)) && (i2 < 8)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 16), ((floormod(floordiv(p, 4), 4)*2) + eps), ((floormod(p, 4)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 512, 512]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*4)*4) + (floordiv(h, 2)*4)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 12  (workload key: ["06f578e6519a86e85028eecf4de64b25", [1, 28, 28, 128], [1, 1, 128, 256], [1, 14, 14, 256]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 128, 256]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])

========== Task 13  (workload key: ["96daaa9daa1b41bc383b7c05ce8b58de", [1, 14, 14, 256], [3, 3, 256, 512], [1, 1, 1, 512], [1, 7, 7, 512]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
pad_temp(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 15)) && (i2 >= 1)) && (i2 < 15)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 256, 512]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 14  (workload key: ["dac19035dd5fe9424ee8617421b9c817", [1, 28, 28, 128], [4, 4, 128, 128], [1, 28, 28, 128], [1, 28, 28, 128]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 29)) && (i2 >= 1)) && (i2 < 29)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 196), ((floormod(floordiv(p, 14), 14)*2) + eps), ((floormod(p, 14)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 128, 128]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*14)*14) + (floordiv(h, 2)*14)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 28, 28, 128]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])

========== Task 15  (workload key: ["96daaa9daa1b41bc383b7c05ce8b58de", [1, 28, 28, 128], [3, 3, 128, 256], [1, 1, 1, 256], [1, 14, 14, 256]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
pad_temp(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 29)) && (i2 >= 1)) && (i2 < 29)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 128, 256]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 16  (workload key: ["1e3c4211ffd2f2db91078ae4d04b779d", [1, 56, 56, 64], [6, 6, 64, 64], [1, 56, 56, 64], [1, 1, 1, 64], [1, 56, 56, 64]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 196), ((floormod(floordiv(p, 14), 14)*4) + eps), ((floormod(p, 14)*4) + nu), ci]
B(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 6) == 5)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 6) == 4)),  ..(OMITTED)..  (floormod(j, 6) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 6) == 0)), 1f, 0f))))))))))))))))))))))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [6, 6, 64, 64]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 4) == 2)),  ..(OMITTED)..  6) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 4), floormod(w, 4), ((((n*14)*14) + (floordiv(h, 4)*14)) + floordiv(w, 4)), co]
placeholder = PLACEHOLDER [1, 56, 56, 64]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 17  (workload key: ["96daaa9daa1b41bc383b7c05ce8b58de", [1, 224, 224, 3], [7, 7, 3, 64], [1, 1, 1, 64], [1, 112, 112, 64]]) ==========
placeholder = PLACEHOLDER [1, 224, 224, 3]
pad_temp(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 3) && (i1 < 227)) && (i2 >= 3)) && (i2 < 227)), placeholder[i0, (i1 - 3), (i2 - 3), i3], 0f)
placeholder = PLACEHOLDER [7, 7, 3, 64]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 18  (workload key: ["3ea73fb9b0364374730d09e068821f95", [1, 56, 56, 64], [6, 6, 64, 64], [1, 56, 56, 64], [1, 56, 56, 64]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 196), ((floormod(floordiv(p, 14), 14)*4) + eps), ((floormod(p, 14)*4) + nu), ci]
B(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 6) == 5)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 6) == 4)),  ..(OMITTED)..  (floormod(j, 6) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 6) == 0)), 1f, 0f))))))))))))))))))))))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [6, 6, 64, 64]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 4) == 2)),  ..(OMITTED)..  6) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 4), floormod(w, 4), ((((n*14)*14) + (floordiv(h, 4)*14)) + floordiv(w, 4)), co]
placeholder = PLACEHOLDER [1, 56, 56, 64]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])

========== Task 19  (workload key: ["d374e472bd9d8164892b9e28a0a8cb59", [1, 14, 14, 256], [4, 4, 256, 256], [1, 14, 14, 256], [1, 14, 14, 256]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 15)) && (i2 >= 1)) && (i2 < 15)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 49), ((floormod(floordiv(p, 7), 7)*2) + eps), ((floormod(p, 7)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 256, 256]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*7)*7) + (floordiv(h, 2)*7)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 14, 14, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])

========== Task 20  (workload key: ["64b98c71af70a904fdbb81d7d4188d84", [1, 112, 112, 64], [1, 1, 1, 64], [1, 56, 56, 64]]) ==========
placeholder = PLACEHOLDER [1, 112, 112, 64]
pad_temp(ax0, ax1, ax2, ax3) = tir.if_then_else(((((ax1 >= 1) && (ax1 < 113)) && (ax2 >= 1)) && (ax2 < 113)), placeholder[ax0, (ax1 - 1), (ax2 - 1), ax3], -3.40282e+38f)
tensor(ax0, ax1, ax2, ax3) max= pad_temp[ax0, ((ax1*2) + rv0), ((ax2*2) + rv1), ax3]
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (tensor[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 21  (workload key: ["06f578e6519a86e85028eecf4de64b25", [1, 14, 14, 256], [1, 1, 256, 512], [1, 7, 7, 512]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 512]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])

========== Task 22  (workload key: ["7d44c6e3c81cd80f61ff2265b2bae89a", [1, 512], [1000, 512], [1, 1000], [1, 1000]]) ==========
placeholder = PLACEHOLDER [1, 512]
placeholder = PLACEHOLDER [1000, 512]
T_matmul_NT(i, j) += (placeholder[i, k]*placeholder[j, k])
placeholder = PLACEHOLDER [1, 1000]
T_add(ax0, ax1) = (T_matmul_NT[ax0, ax1] + placeholder[ax0, ax1])

========== Task 23  (workload key: ["96daaa9daa1b41bc383b7c05ce8b58de", [1, 56, 56, 64], [3, 3, 64, 128], [1, 1, 1, 128], [1, 28, 28, 128]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
pad_temp(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 64, 128]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)
```

## 开始调优

接下来为调优和启动搜索任务设置一些选项

* `measure_ctx` 启动不同的测试过程以提供隔离。在测试期间保护主进程免受 GPU 崩溃并避免其他 runtime 冲突。
* `min_repeat_ms` 定义每次测试中一次“重复”的最短持续时间，可以预热 GPU 以获得准确测试结果，通常，推荐设置值  >= 300 ms。
* `num_measure_trials` 是调优期间可以使用的测试次数（根据自己的时间预算调整这个参数），若要快速演示，可将其设置为较小的数字（例如 200）。推荐将其设置为 `900 * len(tasks)` 左右，以便使搜索收敛。比如 resnet-18 有 24 个任务，所以可以设置为 20000。
* 此外，使用 `RecordToFile` 将测试记录转储到日志文件中，测试记录可用于历史最佳查询、恢复搜索以及进行后续分析。
* 更多参数参见 `auto_scheduler.TuningOptions`，`auto_scheduler.LocalRunner`。

``` python
def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # 将此更改为 20000 以达到最佳性能
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

# 不在网页服务器中运行调优，因为它需要的时间太长。
# 取消注释运行下面行。
# run_tuning()
```

:::note
解释调优过程中打印的信息

在调优过程中，控制台上会打印很多用于调试的信息，最重要的信息是任务调度程序的输出，下表是输出示例。

``` bash
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.005 |           0.88 |     64 |
|    1 |        0.010 |          99.10 |     64 |
|    2 |        0.006 |           0.00 |     64 |
|    3 |        0.145 |         979.78 |    384 |
|    4 |        0.130 |        1097.02 |    384 |
|    5 |        0.143 |         992.69 |    384 |
|    6 |        0.076 |        1526.86 |    192 |
|    7 |        0.115 |         999.44 |    320 |
|    8 |        0.079 |        1449.39 |    320 |
|    9 |        0.122 |         938.73 |    384 |
|   10 |        0.063 |        1832.98 |    192 |
|   11 |        0.072 |        1763.62 |    256 |
|   12 |        0.062 |        2036.40 |    192 |
|   13 |        0.068 |        1874.44 |    192 |
|   14 |        0.049 |        2346.50 |    128 |
|   15 |        0.076 |        1694.31 |    256 |
|   16 |        0.067 |        1933.30 |    448 |
|   17 |        0.076 |        1680.90 |    256 |
|   18 |        0.022 |          98.43 |     64 |
|   19 |        0.076 |        3112.55 |    192 |
|   20 |        0.013 |        2026.44 |     64 |
|   21 |        0.011 |        1136.69 |     64 |
|   22 |        0.013 |         992.47 |     64 |
|   23 |        0.020 |         627.56 |     64 |
-------------------------------------------------
Estimated total latency: 1.587 ms  Trials: 4992  Used time : 13296 s  Next ID: 3
```

此表列出了所有任务的延迟和（预估）速度，还列出了所有任务的测试分配。最后一行打印了这些任务的总加权延迟，可以粗略估计网络的端到端执行时间。最后一行还打印了测试试验的总数、自动调优所花费的总时间以及下一个要调优的任务的 ID。

还有一些「tvm::Error」错误，因为 auto-scheduler 会尝试一些无效的调度。若调优继续运行，则可以忽略这些错误，因为这些错误与主进程隔离。
:::

:::note
提前终止调优

可以通过强制终止此进程来提前终止调优，只要在日志文件中为每个任务获得至少一个有效的调度，就能够进行编译（下面的部分）。
:::

## 编译及评估

自动调优后，用找到的最佳调度来编译网络。在自动调优期间，所有测试记录都被转储到日志文件中，可以读取日志文件加载最佳调度。

``` python
# 用历史最佳编译
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# 创建图执行器
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)

# 评估
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
```

输出结果：

``` bash
Compile...
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
Evaluate inference time cost...
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
  10.0003       9.9944      10.0327       9.9738       0.0244
```

## 其他技巧

1. 在调优过程中，auto-scheduler 需要编译许多程序，并从中提取特征。这部分会占用大量 CPU 资源，所以推荐使用多核的高性能 CPU，加快搜索速度。
2. 可以用 `python3 -m tvm.auto_scheduler.measure_record --mode distill -i log.json` 提取大日志文件，并仅保存最有用的记录。
3. 可以从以前的日志文件恢复搜索，只需要在函数 `run_tuning` 中创建任务调度程序时添加一个新参数 `load_log_file`。比如，`tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)`
4. 若有多个 target CPU，则可以将所有这些 CPU 用于并行化测试。查看这 [部分](https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_cuda.html#tutorials-autotvm-scale-up-rpc-tracker) 了解如何使用 RPC 跟踪器和 RPC 服务器。要在 auto-scheduler 中使用 RPC 跟踪器，请将 `TuningOptions` 中的 runner 替换为 `auto_scheduler.RPCRunner`。
   
[下载 Python 源代码：tune_network_cuda.py](https://tvm.apache.org/docs/_downloads/eafe360d52540634c9eea0fa89e804bd/tune_network_cuda.py)

[下载 Jupyter Notebook：tune_network_cuda.ipynb](https://tvm.apache.org/docs/_downloads/af264436d049e3cd84803b67b6620b63/tune_network_cuda.ipynb)