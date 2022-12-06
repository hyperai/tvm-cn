---
title: 部署 Single Shot Multibox Detector（SSD）模型
---

# 部署 Single Shot Multibox Detector（SSD）模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/deploy_models/deploy_ssd_gluoncv.html#sphx-glr-download-how-to-deploy-models-deploy-ssd-gluoncv-py) 下载完整的示例代码
:::

**作者**：[Yao Wang](https://github.com/kevinthesun)，[Leyuan Wang](https://github.com/Laurawly)

本文介绍如何用 TVM 部署 SSD 模型。这里将使用 GluonCV 预训练的 SSD 模型，并将其转换为 Relay IR。

``` python
import tvm
from tvm import te

from matplotlib import pyplot as plt
from tvm import relay
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils
```

输出结果：

``` bash
/usr/local/lib/python3.7/dist-packages/gluoncv/__init__.py:40: UserWarning: Both `mxnet==1.6.0` and `torch==1.11.0+cpu` are installed. You might encounter increased GPU memory footprint if both framework are used at the same time.
  warnings.warn(f'Both `mxnet=={mx.__version__}` and `torch=={torch.__version__}` are installed. '
```

## 初步参数设置

:::note
现在支持在 CPU 和 GPU 上编译 SSD。

为取得 CPU 上的最佳推理性能，需要根据设备修改 target 参数——对于 x86 CPU：参考 [为 x86 CPU 自动调整卷积网络](/docs/how_to/autotune/autotuning_x86) 来调整；对于 arm CPU：参考 [为 ARM CPU 自动调整卷积网络](/docs/how_to/autotune/autotuning_arm) 来调整。

为在 Intel 显卡上取得最佳推理性能，将 target 参数修改为 `opencl -device=intel_graphics` 。注意：在 Mac 上使用 Intel 显卡时，target 要设置为 `opencl` ，因为 Mac 上不支持 Intel 子组扩展。

为取得基于 CUDA 的 GPU 上的最佳推理性能，将 target 参数修改为 `cuda`；对于基于 OPENCL 的 GPU，将 target 参数修改为 `opencl`，然后根据设备来修改设备参数。
:::

``` python
supported_model = [
    "ssd_512_resnet50_v1_voc",
    "ssd_512_resnet50_v1_coco",
    "ssd_512_resnet101_v2_voc",
    "ssd_512_mobilenet1.0_voc",
    "ssd_512_mobilenet1.0_coco",
    "ssd_300_vgg16_atrous_voc" "ssd_512_vgg16_atrous_coco",
]

model_name = supported_model[0]
dshape = (1, 3, 512, 512)
```

下载并预处理 demo 图像：

``` python
im_fname = download_testdata(
    "https://github.com/dmlc/web-data/blob/main/" + "gluoncv/detection/street_small.jpg?raw=true",
    "street_small.jpg",
    module="data",
)
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
```

为 CPU 转换和编译模型：

``` python
block = model_zoo.get_model(model_name, pretrained=True)

def build(target):
    mod, params = relay.frontend.from_mxnet(block, {"data": dshape})
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    return lib
```

输出结果：

``` bash
/usr/local/lib/python3.7/dist-packages/mxnet/gluon/block.py:1389: UserWarning: Cannot decide type for the following arguments. Consider providing them as input:
        data: None
  input_sym_arg_type = in_param.infer_type()[0]
Downloading /workspace/.mxnet/models/ssd_512_resnet50_v1_voc-9c8b225a.zip from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/ssd_512_resnet50_v1_voc-9c8b225a.zip...

  0%|          | 0/132723 [00:00<?, ?KB/s]
  2%|1         | 2429/132723 [00:00<00:05, 24288.69KB/s]
  8%|8         | 10888/132723 [00:00<00:02, 59757.20KB/s]
 14%|#4        | 18798/132723 [00:00<00:01, 68586.60KB/s]
 21%|##        | 27307/132723 [00:00<00:01, 75099.17KB/s]
 27%|##7       | 35836/132723 [00:00<00:01, 78765.91KB/s]
 33%|###3      | 44460/132723 [00:00<00:01, 81298.98KB/s]
 40%|###9      | 53075/132723 [00:00<00:00, 82882.32KB/s]
 46%|####6     | 61612/132723 [00:00<00:00, 83671.87KB/s]
 53%|#####2    | 69980/132723 [00:00<00:00, 82355.51KB/s]
 59%|#####9    | 78462/132723 [00:01<00:00, 83105.52KB/s]
 65%|######5   | 86777/132723 [00:01<00:00, 79179.66KB/s]
 72%|#######1  | 95291/132723 [00:01<00:00, 80915.06KB/s]
 78%|#######7  | 103417/132723 [00:01<00:00, 62776.56KB/s]
 84%|########4 | 111967/132723 [00:01<00:00, 68364.35KB/s]
 90%|########9 | 119368/132723 [00:01<00:00, 44237.04KB/s]
 96%|#########6| 127829/132723 [00:01<00:00, 51926.12KB/s]
100%|##########| 132723/132723 [00:02<00:00, 64946.94KB/s]
```

创建 TVM runtime，并进行推理，注意：

``` text
Use target = "cuda -libs" to enable thrust based sort, if you
enabled thrust during cmake by -DUSE_THRUST=ON.
```

``` python
def run(lib, dev):
    # 构建 TVM runtime
    m = graph_executor.GraphModule(lib["default"](dev))
    tvm_input = tvm.nd.array(x.asnumpy(), device=dev)
    m.set_input("data", tvm_input)
    # 执行
    m.run()
    # 得到输出
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs

for target in ["llvm", "cuda"]:
    dev = tvm.device(target, 0)
    if dev.exist:
        lib = build(target)
        class_IDs, scores, bounding_boxs = run(lib, dev)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

显示结果：

``` python
ax = utils.viz.plot_bbox(
    img,
    bounding_boxs.numpy()[0],
    scores.numpy()[0],
    class_IDs.numpy()[0],
    class_names=block.classes,
)
plt.show()
```

 ![图片](https://tvm.apache.org/docs/_images/sphx_glr_deploy_ssd_gluoncv_001.png)

**脚本总运行时长：**（ 2 分 32.231 秒）

[下载 Python 源代码：deploy_ssd_gluoncv.py](https://tvm.apache.org/docs/_downloads/cccb17d28e5e8b2e94ea8cd5ec59f6ed/deploy_ssd_gluoncv.py)

[下载 Jupyter Notebook：deploy_ssd_gluoncv.ipynb](https://tvm.apache.org/docs/_downloads/d92aacfae35477bed0f7f60aa8d2714e/deploy_ssd_gluoncv.ipynb)
