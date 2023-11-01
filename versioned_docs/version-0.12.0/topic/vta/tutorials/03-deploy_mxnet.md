---
title: 在 VTA 上部署来自 MxNet 的预训练视觉模型
---

# 在 VTA 上部署来自 MxNet 的预训练视觉模型

:::note
单击 [此处](https://tvm.apache.org/docs/topic/vta/tutorials/frontend/deploy_classification.html#sphx-glr-download-topic-vta-tutorials-frontend-deploy-classification-py) 下载完整的示例代码
:::

**作者**：[Thierry Moreau](https://homes.cs.washington.edu/\~moreau/)

本教程提供了一个端到端 demo，介绍了如何在 VTA 加速器设计上运行 ImageNet 分类推理，执行 ImageNet 分类任务。它展示了 Relay 作为一个前端编译器，可以执行量化（VTA 仅支持 int8/32 推理）以及计算图打包（为了在 core 中启用张量），从而为硬件 target 修改计算图。

## 安装依赖

要在 TVM 中使用 autotvm 包，需要安装额外的依赖（如果用的是 Python2，请将「3」更改为「2」）：

``` bash
pip3 install --user mxnet requests "Pillow<7"
```

在 Python 代码中导入包：

``` python
from __future__ import absolute_import, print_function

import argparse, json, os, requests, sys, time
from io import BytesIO
from os.path import join, isfile
from PIL import Image

from mxnet.gluon.model_zoo import vision
import numpy as np
from matplotlib import pyplot as plt

import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download
from tvm.contrib.debugger import debug_executor
from tvm.relay import transform

import vta
from vta.testing import simulator
from vta.top import graph_pack

# 确保 TVM 是使用 RPC=1 编译的
assert tvm.runtime.enabled("rpc")
```

## 定义平台和模型 targets

对比在 CPU 与 VTA 上执行，并定义模型。

``` python
# 从 3rdparty/vta-hw/config/vta_config.json 文件加载 VTA 参数
env = vta.get_env()

# 设置 ``device=arm_cpu`` 在 CPU 上运行推理
# 设置 ``device=vta`` 在 FPGA 上运行推理
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# 查找何时开始/结束位打包的字典
pack_dict = {
    "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet101_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
}

# 要编译的 Gluon 模型的名称
# ``start_pack`` 和 ``stop_pack`` 标签指示在哪里
# 开始和结束计算图打包 Relay pass：换言之，
# 从哪里开始和完成转移到 VTA。
model = "resnet18_v1"
assert model in pack_dict
```

## 获取远程执行

当 target 是「pynq」时，重新配置 FPGA 和 runtime。若 target 是「sim」，则在本地执行。

``` python
if env.TARGET not in ["sim", "tsim", "intelfocl"]:
    # 若设置了环境变量，则从跟踪器节点获取远程。
    # 要设置跟踪器，参考「自动调优 VTA 的卷积网络」教程。
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    # 否则，若有一个想直接从主机编程的设备，
    # 请确保已将以下变量设置为你板子的 IP
    device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
    device_port = os.environ.get("VTA_RPC_PORT", "9091")
    if not tracker_host or not tracker_port:
        remote = rpc.connect(device_host, int(device_port))
    else:
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, int(tracker_port), timeout=10000
        )

    # 重新配置 JIT runtime 和 FPGA。
    # 可以通过传递比特流文件的路径而非 None，用自定义比特流对 FPGA 进行编程
    reconfig_start = time.time()
    vta.reconfig_runtime(remote)
    vta.program_fpga(remote, bitstream=None)
    reconfig_time = time.time() - reconfig_start
    print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

# 在模拟模式下，本地托管 RPC 服务器。
else:
    remote = rpc.LocalSession()

    if env.TARGET in ["intelfocl"]:
        # 编写 intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

# 从远程获取执行上下文
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
```

## 构建推理图执行器

从 Gluon model zoo 选取视觉模型，并用 Relay 编译。编译步骤如下：

1. 从 MxNet 到 Relay 模块的前端转换。
2. 应用 8 位量化：这里跳过第一个 conv 层和 dense 层，它们都将在 CPU 上以 fp32 执行。
3. 执行计算图打包，更改张量化的数据布局。
4. 执行常量折叠，减少算子的数量（例如，消除 batch norm multiply）。
5. 对目标文件执行 Relay 构建。
6. 将目标文件加载到远程（FPGA 设备）。
7. 生成图执行器 *m*。

``` python
# 加载预先配置的 AutoTVM schedules
with autotvm.tophub.context(target):
    # 为 ImageNet 分类器输入填充 shape 和数据类型字典
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # 下架 gluon 模型，并转换为 Relay
    gluon_model = vision.get_model(model, pretrained=True)

    # 测试构建开始时间
    build_start = time.time()

    # 开始前端编译
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # 更新 shape 和类型字典
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    if target.device_name == "vta":
        # 在 Relay 中执行量化
        # 注意：将 opt_level 设置为 3 ，折叠 batch norm
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                mod = relay.quantize.quantize(mod, params=params)
            # 对 VTA 目标进行图打包和常量折叠
            assert env.BLOCK_IN == env.BLOCK_OUT
            # 若 target 是 intelfocl 或 sim，则进行设备注释
            relay_prog = graph_pack(
                mod["main"],
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=pack_dict[model][0],
                stop_name=pack_dict[model][1],
                device_annot=(env.TARGET == "intelfocl"),
            )
    else:
        relay_prog = mod["main"]

    # 在禁用 AlterOpLayout 的情况下，编译 Relay 程序
    if target.device_name != "vta":
        with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params
            )
    else:
        if env.TARGET == "intelfocl":
            # 在 cpu 和 vta 上运行多个 target
            target = {"cpu": env.target_vta_cpu, "ext_dev": target}
        with vta.build_config(
            opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}
        ):
            graph, lib, params = relay.build(
                relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params
            )

    # 测试 Relay 构建时间
    build_time = time.time() - build_start
    print(model + " inference graph built in {0:.2f}s!".format(build_time))

    # 将推理库发送到远程 RPC 服务器
    temp = utils.tempdir()
    lib.export_library(temp.relpath("graphlib.tar"))
    remote.upload(temp.relpath("graphlib.tar"))
    lib = remote.load_module("graphlib.tar")

    if env.TARGET == "intelfocl":
        ctxes = [remote.ext_dev(0), remote.cpu(0)]
        m = graph_executor.create(graph, lib, ctxes)
    else:
        # 计算图 runtime
        m = graph_executor.create(graph, lib, ctx)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:267: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
/workspace/python/tvm/relay/build_module.py:411: DeprecationWarning: Please use input parameter mod (tvm.IRModule) instead of deprecated parameter mod (tvm.relay.function.Function)
  DeprecationWarning,
/workspace/vta/tutorials/frontend/deploy_classification.py:213: DeprecationWarning: legacy graph executor behavior of producing json / lib / params will be removed in the next release. Please see documents of tvm.contrib.graph_executor.GraphModule for the  new recommended usage.
  relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params
resnet18_v1 inference graph built in 22.98s!
```

## 执行图像分类推理

对来自 ImageNet 的图像样本进行分类。只需下载类别文件、*synset.txt* 和输入测试图像。

``` python
# 下载 ImageNet 类别
categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
categ_fn = "synset.txt"
download.download(join(categ_url, categ_fn), categ_fn)
synset = eval(open(categ_fn).read())

# 下载测试图像
image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
image_fn = "cat.png"
download.download(image_url, image_fn)

# 为推理准备测试图像
image = Image.open(image_fn).resize((224, 224))
plt.imshow(image)
plt.show()
image = np.array(image) - np.array([123.0, 117.0, 104.0])
image /= np.array([58.395, 57.12, 57.375])
image = image.transpose((2, 0, 1))
image = image[np.newaxis, :]
image = np.repeat(image, env.BATCH, axis=0)

# 设置网络参数和输入
m.set_input(**params)
m.set_input("data", image)

# 执行推理，并收集执行统计信息
# 更多信息：:py:method:`tvm.runtime.Module.time_evaluator`
num = 4  # 为单个测试运行模块的次数
rep = 3  # 测试次数（我们从中得出标准差）
timer = m.module.time_evaluator("run", ctx, number=num, repeat=rep)

if env.TARGET in ["sim", "tsim"]:
    simulator.clear_stats()
    timer()
    sim_stats = simulator.stats()
    print("\nExecution statistics:")
    for k, v in sim_stats.items():
        # 由于多次执行工作流程，需要对统计数据归一化
        # 注意，总有一次预运行
        # 因此将整体统计数据除以 (num * rep + 1)
        print("\t{:<16}: {:>16}".format(k, v // (num * rep + 1)))
else:
    tcost = timer()
    std = np.std(tcost.results) * 1000
    mean = tcost.mean * 1000
    print("\nPerformed inference in %.2fms (std = %.2f) for %d samples" % (mean, std, env.BATCH))
    print("Average per sample inference time: %.2fms" % (mean / env.BATCH))

# 获取分类结果
tvm_output = m.get_output(0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0)))
for b in range(env.BATCH):
    top_categories = np.argsort(tvm_output.numpy()[b])
    # 报告前 5 个分类结果
    print("\n{} prediction for sample {}".format(model, b))
    print("\t#1:", synset[top_categories[-1]])
    print("\t#2:", synset[top_categories[-2]])
    print("\t#3:", synset[top_categories[-3]])
    print("\t#4:", synset[top_categories[-4]])
    print("\t#5:", synset[top_categories[-5]])
    # 这只检查前 5 个类别之一是一种猫；这绝不是有关量化如何影响分类准确性的准确评估，而是在捕捉对 CI 准确性的量化 pass 的变化。
    cat_detected = False
    for k in top_categories[-5:]:
        if "cat" in synset[k]:
            cat_detected = True
    assert cat_detected
```

![deploy classification](https://tvm.apache.org/docs/_images/sphx_glr_deploy_classification_001.png)

输出结果：

``` bash
Execution statistics:
        inp_load_nbytes :          5549568
        wgt_load_nbytes :         12763136
        acc_load_nbytes :          6051840
        uop_load_nbytes :            22864
        out_store_nbytes:          2433536
        gemm_counter    :          6623232
        alu_counter     :           699328

resnet18_v1 prediction for sample 0
        #1: tiger cat
        #2: Egyptian cat
        #3: tabby, tabby cat
        #4: lynx, catamount
        #5: weasel
```

[下载 Python 源代码：deploy_classification.py](https://tvm.apache.org/docs/_downloads/9e8de33a5822b31748bfd76861009f92/deploy_classification.py)

[下载 Jupyter Notebook：deploy_classification.ipynb](https://tvm.apache.org/docs/_downloads/95395e118195f25266654dd8fbf487d4/deploy_classification.ipynb)