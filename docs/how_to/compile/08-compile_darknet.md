---
title: 在 DarkNet 模型中编译 YOLO-V2 和 YOLO-V3
---

# 在 DarkNet 模型中编译 YOLO-V2 和 YOLO-V3

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/compile_models/from_darknet.html#sphx-glr-download-how-to-compile-models-from-darknet-py) 下载完整的示例代码
:::

**作者**：[Siju Samuel](https://siju-samuel.github.io/)

本文介绍如何用 TVM 部署 DarkNet 模型。所有必需的模型和库都可通过脚本从 Internet 下载。此脚本运行带有边界框的 YOLO-V2 和 YOLO-V3 模型。DarkNet 解析依赖 CFFI 和 CV2 库，因此执行脚本前要安装这两个库。

``` bash
pip install cffi
pip install opencv-python
```

``` python
# numpy 和 matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys

# tvm 和 relay
import tvm
from tvm import te
from tvm import relay
from ctypes import *
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet
```

## 选择模型

模型有：‘yolov2’、‘yolov3’ 或 ‘yolov3-tiny’

``` python
# 模型名称
MODEL_NAME = "yolov3"
```

## 下载所需文件

第一次编译的话需要下载 cfg 和 weights 文件。

``` python
CFG_NAME = MODEL_NAME + ".cfg"
WEIGHTS_NAME = MODEL_NAME + ".weights"
REPO_URL = "https://github.com/dmlc/web-data/blob/main/darknet/"
CFG_URL = REPO_URL + "cfg/" + CFG_NAME + "?raw=true"
WEIGHTS_URL = "https://pjreddie.com/media/files/" + WEIGHTS_NAME

cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")

# 下载并加载 DarkNet 库
if sys.platform in ["linux", "linux2"]:
    DARKNET_LIB = "libdarknet2.0.so"
    DARKNET_URL = REPO_URL + "lib/" + DARKNET_LIB + "?raw=true"
elif sys.platform == "darwin":
    DARKNET_LIB = "libdarknet_mac2.0.so"
    DARKNET_URL = REPO_URL + "lib_osx/" + DARKNET_LIB + "?raw=true"
else:
    err = "Darknet lib is not supported on {} platform".format(sys.platform)
    raise NotImplementedError(err)

lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
dtype = "float32"
batch_size = 1

data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape_dict = {"data": data.shape}
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
```

输出结果：

``` bash
Converting darknet to relay functions...
```

## 将计算图导入到 Relay 中

编译模型：

``` python
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape = {"data": data.shape}
print("Compiling the model...")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

[neth, netw] = shape["data"][2:]  # 当前图像 shape 是 608x608
```

输出结果：

``` bash
Compiling the model...
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 加载测试图像

``` python
test_image = "dog.jpg"
print("Loading the test image...")
img_url = REPO_URL + "data/" + test_image + "?raw=true"
img_path = download_testdata(img_url, test_image, "data")

data = tvm.relay.testing.darknet.load_image(img_path, netw, neth)
```

输出结果：

``` bash
Loading the test image...
```

## 在 TVM Runtime 上执行

这个过程与其他示例的相同。

``` python
from tvm.contrib import graph_executor

m = graph_executor.GraphModule(lib["default"](dev))

# 设置输入
m.set_input("data", tvm.nd.array(data.astype(dtype)))
# 执行
print("Running the test image...")

# 检测
# 阈值
thresh = 0.5
nms_thresh = 0.45

m.run()
# 得到输出
tvm_out = []
if MODEL_NAME == "yolov2":
    layer_out = {}
    layer_out["type"] = "Region"
    # 获取区域层属性（n、out_c、out_h、out_w、classes、coords 和 background）
    layer_attr = m.get_output(2).numpy()
    layer_out["biases"] = m.get_output(1).numpy()
    out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0], layer_attr[2], layer_attr[3])
    layer_out["output"] = m.get_output(0).numpy().reshape(out_shape)
    layer_out["classes"] = layer_attr[4]
    layer_out["coords"] = layer_attr[5]
    layer_out["background"] = layer_attr[6]
    tvm_out.append(layer_out)
elif MODEL_NAME == "yolov3":
    for i in range(3):
        layer_out = {}
        layer_out["type"] = "Yolo"
        # 获取 yolo 层属性（n、out_c、out_h、out_w、classes 和 total）
        layer_attr = m.get_output(i * 4 + 3).numpy()
        layer_out["biases"] = m.get_output(i * 4 + 2).numpy()
        layer_out["mask"] = m.get_output(i * 4 + 1).numpy()
        out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0], layer_attr[2], layer_attr[3])
        layer_out["output"] = m.get_output(i * 4).numpy().reshape(out_shape)
        layer_out["classes"] = layer_attr[4]
        tvm_out.append(layer_out)
elif MODEL_NAME == "yolov3-tiny":
    for i in range(2):
        layer_out = {}
        layer_out["type"] = "Yolo"
        # 获取 yolo 层属性（n、out_c、out_h、out_w、classes 和 total）
        layer_attr = m.get_output(i * 4 + 3).numpy()
        layer_out["biases"] = m.get_output(i * 4 + 2).numpy()
        layer_out["mask"] = m.get_output(i * 4 + 1).numpy()
        out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0], layer_attr[2], layer_attr[3])
        layer_out["output"] = m.get_output(i * 4).numpy().reshape(out_shape)
        layer_out["classes"] = layer_attr[4]
        tvm_out.append(layer_out)
        thresh = 0.560

# 检测，并画出边界框
img = tvm.relay.testing.darknet.load_image_color(img_path)
_, im_h, im_w = img.shape
dets = tvm.relay.testing.yolo_detection.fill_network_boxes(
    (netw, neth), (im_w, im_h), thresh, 1, tvm_out
)
last_layer = net.layers[net.n - 1]
tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)

coco_name = "coco.names"
coco_url = REPO_URL + "data/" + coco_name + "?raw=true"
font_name = "arial.ttf"
font_url = REPO_URL + "data/" + font_name + "?raw=true"
coco_path = download_testdata(coco_url, coco_name, module="data")
font_path = download_testdata(font_url, font_name, module="data")

with open(coco_path) as f:
    content = f.readlines()

names = [x.strip() for x in content]

tvm.relay.testing.yolo_detection.show_detections(img, dets, thresh, names, last_layer.classes)
tvm.relay.testing.yolo_detection.draw_detections(
    font_path, img, dets, thresh, names, last_layer.classes
)
plt.imshow(img.transpose(1, 2, 0))
plt.show()
```

 ![from darknet](https://tvm.apache.org/docs/_images/sphx_glr_from_darknet_001.png)

输出结果：

``` bash
Running the test image...
class:['dog 0.994'] left:127 top:227 right:316 bottom:533
class:['truck 0.9266'] left:471 top:83 right:689 bottom:169
class:['bicycle 0.9984'] left:111 top:113 right:577 bottom:447
```

**脚本总运行时长：**（1 分 1.020 秒）

[下载 Python 源代码：from_darknet.py](https://tvm.apache.org/docs/_downloads/7716f96385bd5abb6e822041e285be54/from_darknet.py)

[下载 Jupyter Notebook：from_darknet.ipynb](https://tvm.apache.org/docs/_downloads/f97d815b408ef3f4d6bcb3e073c2d4dd/from_darknet.ipynb)