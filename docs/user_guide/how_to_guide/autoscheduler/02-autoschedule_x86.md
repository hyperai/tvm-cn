---
title: 为 x86 CPU 自动调度神经网络
---

# 为 x86 CPU 自动调度神经网络

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_network_x86.html#sphx-glr-download-how-to-tune-with-autoscheduler-tune-network-x86-py) 下载完整的示例代码
:::

**作者**：[Lianmin Zheng](https://github.com/merrymercy)，[Chengfan Jia](https://github.com/jcf94/)

针对特定设备和工作负载自动调优，对于获取最佳性能至关重要。本文介绍如何使用 auto-scheduler 为 x86 CPU 调优整个神经网络。

为自动调优神经网络，将网络划分为小的子图并独立调优。每个子图被视为一个搜索任务，任务调度器对时间进行切片并动态地为这些任务分配时间资源，并预测每个任务对端到端执行时间的影响，优先考虑最能减少执行时间的任务。

对于每个子图，使用 `tvm/python/topi` 中的计算声明来获取张量表达式形式的计算 DAG。然后使用 auto-scheduler 来构建这个 DAG 的搜索空间并搜索合适的调度（底层优化）。

与基于模板的 [AutoTVM](../autotune)（依赖手动模板来定义搜索空间的） 不同，auto-scheduler 不需要任何调度模板。换言之，auto-scheduler 只使用 `tvm/python/topi` 中的计算声明，不使用现有的调度模板。

注意，本教程无法在 Windows 或最新版本的 macOS 上运行。如需运行，请将本教程的主体放在 `if __name__ == "__main__":` 代码块中。

``` python
import numpy as np

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor
```

## 定义网络

首先，要使用 Relay 前端 API 定义网络。可以从 `tvm.relay.testing` 加载一些预定义的网络。也可以从 MXNet、ONNX、PyTorch 和 TensorFlow 加载模型（参见 [前端教程](../compile) ）。

对于卷积神经网络，尽管 auto-scheduler 可以在任何布局下正常运行，但通过 NHWC 布局实现的性能最佳。auto-scheduler 对 NHWC 布局进行了很多优化，因此推荐将模型转换为 NHWC 布局，从而得以使用 auto-scheduler。可用 [ConvertLayout](https://tvm.apache.org/docs/arch/convert_layout.html#convert-layout-usage) pass 在 TVM 中进行布局转换。

``` python
def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
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

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
        )
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, bs_r=4, random_params=True)

    return mod, params, input_shape, output_shape

# 定义神经网络和编译 target。
# 若 target 机器支持 avx512 指令，
# 使用 "llvm -mcpu=skylake-avx512" 替换 "llvm -mcpu=core-avx2"
network = "resnet-50"
use_sparse = False
batch_size = 1
layout = "NHWC"
target = tvm.target.Target("llvm -mcpu=core-avx2")
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
```

## 提取搜索任务

接下来，从网络中提取搜索任务及其权重。任务的权重是任务的子图在整个网络中出现的次数。通过使用权重，可以将网络的端到端延迟近似为 `sum(latency[t] * weight[t])`，其中 `latency[t]` 是任务的延迟，而 `weight[t]` 是任务的权重，任务调度器仅针对该目标进行优化。

``` python
# 从网络中提取任务
print("Get model...")
mod, params, input_shape, output_shape = get_network(
    network,
    batch_size,
    layout,
    dtype=dtype,
    use_sparse=use_sparse,
)
print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
```

输出结果：

``` bash
Get model...
Extract tasks...
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
========== Task 0  (workload key: ["8654f16aeddf785bad9f028164b3a48d", [1, 56, 56, 64], [1, 1, 64, 256], [1, 56, 56, 256]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 64, 256]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])

========== Task 1  (workload key: ["b9a4f9bd1416ba25810cb3de27628ace", [1, 14, 14, 256], [1, 1, 256, 1024], [1, 14, 14, 1024], [1, 1, 1, 1024], [1, 14, 14, 1024]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 1024]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 14, 14, 1024]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
placeholder = PLACEHOLDER [1, 1, 1, 1024]
T_add(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 2  (workload key: ["12cb81d4ad0a81be02dedf09d1ac8391", [1, 14, 14, 256], [1, 1, 256, 1024], [1, 14, 14, 1024], [1, 14, 14, 1024]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 1024]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 14, 14, 1024]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])

========== Task 3  (workload key: ["56af7508fbcdf6d851892b1e8434667b", [1, 14, 14, 1024], [1, 1, 1024, 512], [1, 1, 1, 512], [1, 7, 7, 512]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 1024]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 1024, 512]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 4  (workload key: ["06f578e6519a86e85028eecf4de64b25", [1, 56, 56, 256], [1, 1, 256, 512], [1, 28, 28, 512]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 256]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 512]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])

========== Task 5  (workload key: ["c68f92478eb18145106184c587d212b6", [1, 14, 14, 256], [6, 6, 256, 256], [1, 1, 1, 256], [1, 14, 14, 256]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 15)) && (i2 >= 1)) && (i2 < 15)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 16), ((floormod(floordiv(p, 4), 4)*4) + eps), ((floormod(p, 4)*4) + nu), ci]
B(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 6) == 5)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 6) == 4)),  ..(OMITTED)..  (floormod(j, 6) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 6) == 0)), 1f, 0f))))))))))))))))))))))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [6, 6, 256, 256]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 4) == 2)),  ..(OMITTED)..  6) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 4), floormod(w, 4), ((((n*4)*4) + (floordiv(h, 4)*4)) + floordiv(w, 4)), co]
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 6  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 14, 14, 1024], [1, 1, 1024, 256], [1, 1, 1, 256], [1, 14, 14, 256]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 1024]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 1024, 256]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 7  (workload key: ["12cb81d4ad0a81be02dedf09d1ac8391", [1, 28, 28, 128], [1, 1, 128, 512], [1, 28, 28, 512], [1, 28, 28, 512]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 128, 512]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 28, 28, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])

========== Task 8  (workload key: ["12cb81d4ad0a81be02dedf09d1ac8391", [1, 56, 56, 64], [1, 1, 64, 256], [1, 56, 56, 256], [1, 56, 56, 256]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 64, 256]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 56, 56, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])

========== Task 9  (workload key: ["ecec634b4882c5731f86cce3109db636", [1, 28, 28, 128], [6, 6, 128, 128], [1, 1, 1, 128], [1, 28, 28, 128]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 29)) && (i2 >= 1)) && (i2 < 29)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 49), ((floormod(floordiv(p, 7), 7)*4) + eps), ((floormod(p, 7)*4) + nu), ci]
B(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 6) == 5)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 6) == 4)),  ..(OMITTED)..  (floormod(j, 6) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 6) == 0)), 1f, 0f))))))))))))))))))))))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [6, 6, 128, 128]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 4) == 2)),  ..(OMITTED)..  6) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 4), floormod(w, 4), ((((n*7)*7) + (floordiv(h, 4)*7)) + floordiv(w, 4)), co]
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 10  (workload key: ["d7b65649a4dd54becea0a52aabbc5af5", [1, 1000], [1, 1000]]) ==========
placeholder = PLACEHOLDER [1, 1000]
T_softmax_maxelem(i0) max= placeholder[i0, k]
T_softmax_exp(i0, i1) = tir.exp((placeholder[i0, i1] - T_softmax_maxelem[i0]))
T_softmax_expsum(i0) += T_softmax_exp[i0, k]
T_softmax_norm(i0, i1) = (T_softmax_exp[i0, i1]/T_softmax_expsum[i0])

========== Task 11  (workload key: ["69115f188984ae34ede37c3b8ca40b43", [1, 7, 7, 2048], [1, 1, 1, 2048]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 2048]
tensor(ax0, ax1, ax2, ax3) += placeholder[ax0, ((ax1*7) + rv0), ((ax2*7) + rv1), ax3]
tensor(ax0, ax1, ax2, ax3) = (tensor[ax0, ax1, ax2, ax3]/(float32((select((bool)1, ((ax1 + 1)*7), (((ax1 + 1)*7) + 1)) - (ax1*7)))*float32((select((bool)1, ((ax2 + 1)*7), (((ax2 + 1)*7) + 1)) - (ax2*7)))))

========== Task 12  (workload key: ["12cb81d4ad0a81be02dedf09d1ac8391", [1, 7, 7, 512], [1, 1, 512, 2048], [1, 7, 7, 2048], [1, 7, 7, 2048]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 512]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 2048]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 7, 7, 2048]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])

========== Task 13  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 28, 28, 512], [1, 1, 512, 128], [1, 1, 1, 128], [1, 28, 28, 128]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 512]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 128]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 14  (workload key: ["56af7508fbcdf6d851892b1e8434667b", [1, 28, 28, 512], [1, 1, 512, 256], [1, 1, 1, 256], [1, 14, 14, 256]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 512]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 256]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 15  (workload key: ["06f578e6519a86e85028eecf4de64b25", [1, 28, 28, 512], [1, 1, 512, 1024], [1, 14, 14, 1024]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 512]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 1024]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])

========== Task 16  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 7, 7, 2048], [1, 1, 2048, 512], [1, 1, 1, 512], [1, 7, 7, 512]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 2048]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 2048, 512]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 17  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 56, 56, 256], [1, 1, 256, 64], [1, 1, 1, 64], [1, 56, 56, 64]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 256]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 64]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 18  (workload key: ["b9a4f9bd1416ba25810cb3de27628ace", [1, 56, 56, 64], [1, 1, 64, 256], [1, 56, 56, 256], [1, 1, 1, 256], [1, 56, 56, 256]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 64, 256]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 56, 56, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 19  (workload key: ["b9a4f9bd1416ba25810cb3de27628ace", [1, 28, 28, 128], [1, 1, 128, 512], [1, 28, 28, 512], [1, 1, 1, 512], [1, 28, 28, 512]]) ==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 128, 512]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 28, 28, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 20  (workload key: ["86551f1a74663d3ceafd5884659d3478", [1, 7, 7, 512], [3, 3, 512, 512], [1, 1, 1, 512], [1, 7, 7, 512]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 512]
pad_temp(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 8)) && (i2 >= 1)) && (i2 < 8)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 512, 512]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 21  (workload key: ["86551f1a74663d3ceafd5884659d3478", [1, 56, 56, 64], [3, 3, 64, 64], [1, 1, 1, 64], [1, 56, 56, 64]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
pad_temp(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 64, 64]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 22  (workload key: ["56af7508fbcdf6d851892b1e8434667b", [1, 56, 56, 256], [1, 1, 256, 128], [1, 1, 1, 128], [1, 28, 28, 128]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 256]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 128]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 23  (workload key: ["ff5ea7f814e5c497bb685e7385cf7159", [1, 7, 7, 512], [1, 1, 512, 2048], [1, 7, 7, 2048], [1, 1, 1, 2048], [1, 1, 1, 2048], [1, 7, 7, 2048]]) ==========
placeholder = PLACEHOLDER [1, 7, 7, 512]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 2048]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 7, 7, 2048]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
placeholder = PLACEHOLDER [1, 1, 1, 2048]
T_multiply(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3]*placeholder[ax0, 0, 0, ax3])
placeholder = PLACEHOLDER [1, 1, 1, 2048]
T_add(ax0, ax1, ax2, ax3) = (T_multiply[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 24  (workload key: ["96daaa9daa1b41bc383b7c05ce8b58de", [1, 224, 224, 3], [7, 7, 3, 64], [1, 1, 1, 64], [1, 112, 112, 64]]) ==========
placeholder = PLACEHOLDER [1, 224, 224, 3]
pad_temp(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 3) && (i1 < 227)) && (i2 >= 3)) && (i2 < 227)), placeholder[i0, (i1 - 3), (i2 - 3), i3], 0f)
placeholder = PLACEHOLDER [7, 7, 3, 64]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 25  (workload key: ["06f578e6519a86e85028eecf4de64b25", [1, 14, 14, 1024], [1, 1, 1024, 2048], [1, 7, 7, 2048]]) ==========
placeholder = PLACEHOLDER [1, 14, 14, 1024]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 1024, 2048]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])

========== Task 26  (workload key: ["7d44c6e3c81cd80f61ff2265b2bae89a", [1, 2048], [1000, 2048], [1, 1000], [1, 1000]]) ==========
placeholder = PLACEHOLDER [1, 2048]
placeholder = PLACEHOLDER [1000, 2048]
T_matmul_NT(i, j) += (placeholder[i, k]*placeholder[j, k])
placeholder = PLACEHOLDER [1, 1000]
T_add(ax0, ax1) = (T_matmul_NT[ax0, ax1] + placeholder[ax0, ax1])

========== Task 27  (workload key: ["64b98c71af70a904fdbb81d7d4188d84", [1, 112, 112, 64], [1, 1, 1, 64], [1, 56, 56, 64]]) ==========
placeholder = PLACEHOLDER [1, 112, 112, 64]
pad_temp(ax0, ax1, ax2, ax3) = tir.if_then_else(((((ax1 >= 1) && (ax1 < 113)) && (ax2 >= 1)) && (ax2 < 113)), placeholder[ax0, (ax1 - 1), (ax2 - 1), ax3], -3.40282e+38f)
tensor(ax0, ax1, ax2, ax3) max= pad_temp[ax0, ((ax1*2) + rv0), ((ax2*2) + rv1), ax3]
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (tensor[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

========== Task 28  (workload key: ["1037be767e8e18197e87653d81c34558", [1, 56, 56, 64], [1, 1, 64, 64], [1, 1, 1, 64], [1, 56, 56, 64]]) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
pad_temp(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 64, 64]
conv2d_nhwc(nn, yy, xx, ff) += (pad_temp[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (conv2d_nhwc[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)
```

## 开始调优

接下来为调优和启动搜索任务设置一些选项

* `num_measure_trials` 是调优期间可以使用的测试次数（根据自己的时间预算调整这个参数），若要进行快速演示，可将其设置为较小的数字（例如 200）。推荐将其设置为 `800 * len(tasks)` 左右，以便使搜索收敛。比如 ResNet-50 有 29 个任务，所以可以设置为 20000。
* 此外，使用 `RecordToFile` 将测试记录转储到日志文件中，测试记录可用于历史最佳查询、恢复搜索以及进行后续分析。
* 更多参数参见 `auto_scheduler.TuningOptions`，`auto_scheduler.LocalRunner`。

``` python
def run_tuning():
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # 将此更改为 20000 以达到最佳性能
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    if use_sparse:
        from tvm.topi.sparse.utils import sparse_sketch_rules

        search_policy = [
            auto_scheduler.SketchPolicy(
                task,
                program_cost_model=auto_scheduler.XGBModel(),
                init_search_callbacks=sparse_sketch_rules(),
            )
            for task in tasks
        ]

        tuner.tune(tune_option, search_policy=search_policy)
    else:
        tuner.tune(tune_option)

# 不在网页服务器中运行调优，因为它需要的时间太长。
# 取消注释运行以下行。
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
|    0 |        0.010 |           0.40 |     64 |
|    1 |        0.087 |          47.19 |     64 |
|    2 |        0.008 |          -0.00 |     64 |
|    3 |        0.177 |         582.07 |     64 |
|    4 |        0.268 |         862.37 |    256 |
|    5 |        0.166 |         621.13 |    128 |
|    6 |        0.170 |         605.10 |    128 |
|    7 |        0.128 |         403.20 |     64 |
|    8 |        0.189 |         545.71 |     64 |
|    9 |        0.231 |        1001.01 |    448 |
|   10 |        0.155 |         664.80 |    256 |
|   11 |        0.155 |         662.86 |    256 |
|   12 |        0.119 |         434.08 |     64 |
|   13 |        0.199 |         522.13 |     64 |
|   14 |        0.235 |         986.56 |    320 |
|   15 |        0.149 |         689.13 |    128 |
|   16 |        0.155 |         664.80 |    192 |
|   17 |        0.151 |         340.64 |     64 |
|   18 |        0.176 |         597.55 |    128 |
|   19 |        0.220 |        1054.37 |    192 |
|   20 |        0.150 |         686.01 |    128 |
|   21 |        0.159 |         650.88 |    128 |
|   22 |        0.073 |         358.19 |     64 |
|   23 |        0.031 |          70.63 |     64 |
|   24 |        0.251 |         947.73 |    128 |
|   25 |        0.157 |         652.47 |    128 |
|   26 |        0.215 |         954.84 |    128 |
|   27 |        0.237 |         868.92 |    128 |
|   28 |        0.266 |         774.06 |    128 |
-------------------------------------------------
Estimated total latency: 10.016 ms      Trials: 3992    Used time : 1131 s      Next ID: 15
```

此表列出了所有任务的延迟和（预估）速度，还列出了所有任务的测试分配。最后一行打印了这些任务的总加权延迟，可以粗略估计网络的端到端执行时间。最后一行还打印了测试的总数、自动调优所花费的总时间以及下一个要调优的任务的 ID。

还有一些「tvm::Error」错误，因为 auto-scheduler 会尝试一些无效的调度。若调优继续运行，则可以忽略这些错误，因为这些错误与主进程隔离。
:::

:::note
提前终止调优

可以通过强制终止此进程来提前终止调优，只要在日志文件中为每个任务获得至少一个有效的调度，就能够进行编译（下面的部分）。
:::

## 编译及评估

自动调优后，可以用找到的最佳调度来编译网络。在自动调优期间，所有测试记录都被转储到日志文件中，可以读取日志文件加载最佳调度。

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
  762.1013     761.7971     762.8241     761.6828      0.5132
```

## 其他技巧

1. 在调优过程中，auto-scheduler 需要编译许多程序并从中提取特征。这部分会占用大量 CPU 资源，所以推荐使用多核的高性能 CPU，加快搜索速度。
2. 可以使用 `python3 -m tvm.auto_scheduler.measure_record --mode distill -i log.json` 提取大日志文件并仅保存最有用的记录。
3. 可以从以前的日志文件恢复搜索，只需要在函数 `run_tuning` 中创建任务调度程序时添加一个新参数 `load_log_file`。比如，`tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)`
4. 若有多个 target CPU，则可以将所有这些 CPU 用于并行化测试。查看这 [部分](https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_cuda.html#tutorials-autotvm-scale-up-rpc-tracker) 了解如何使用 RPC 跟踪器和 RPC 服务器。要在 auto-scheduler 中使用 RPC 跟踪器，请将 `TuningOptions` 中的 runner 替换为 `auto_scheduler.RPCRunner`。

**脚本总运行时长：**（ 1 分 22.035 秒）

[下载 Python 源代码：tune_network_x86.py](https://tvm.apache.org/docs/_downloads/e416b94ca1090b0897c0f6e0df95b911/tune_network_x86.py)

[下载 Jupyter Notebook：tune_network_x86.ipynb](https://tvm.apache.org/docs/_downloads/ad2a7f55d615d188ad664d56696815a6/tune_network_x86.ipynb)