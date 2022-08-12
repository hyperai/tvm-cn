---
title: 编译 TensorFlow 模型
---

# 编译 TensorFlow 模型

注意：单击 [此处](https://tvm.apache.org/docs/how_to/compile_models/from_tensorflow.html#sphx-glr-download-how-to-compile-models-from-tensorflow-py) 下载完整的示例代码

本文介绍了如何用 TVM 部署 TensorFlow 模型。

首先安装 TensorFlow Python 模块（可参考 https://www.tensorflow.org/install）。

```plain
# 导入 tvm 和 relay
import tvm
from tvm import te
from tvm import relay

# 导入 os 和 numpy
import numpy as np
import os.path

# 导入 TensorFlow
import tensorflow as tf



# 让 TensorFlow 将 GPU 内存限制为实际需要的内存，而非占用所有可用的内存。
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# 本教程这样做，对 sphinx-gallery 更友好。
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("tensorflow will use experimental.set_memory_growth(True)")
    except RuntimeError as e:
        print("experimental.set_memory_growth option is not available: {}".format(e))



try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# TensorFlow 实用函数
import tvm.relay.testing.tf as tf_testing

# 模型相关文件的基本位置
repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"

# 测试图像
img_name = "elephant-299.jpg"
image_url = os.path.join(repo_base, img_name)
```

## 教程

参考 docs/frontend/tensorflow.md，获取 TensorFlow 中各种模型的更多信息。

```plain
model_name = "classify_image_graph_def-with_shapes.pb"
model_url = os.path.join(repo_base, model_name)

# 图像标签图
map_proto = "imagenet_2012_challenge_label_map_proto.pbtxt"
map_proto_url = os.path.join(repo_base, map_proto)

# 可读的标签文本
label_map = "imagenet_synset_to_human_label_map.txt"
label_map_url = os.path.join(repo_base, label_map)

# target 设置
# 用下面这些注释为 cuda 构建
# target = tvm.target.Target("cuda", host="llvm")
# layout = "NCHW"
# dev = tvm.cuda(0)
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
```

## 下载所需文件

下载上述列出的文件：

```plain
from tvm.contrib.download import download_testdata

img_path = download_testdata(image_url, img_name, module="data")
model_path = download_testdata(model_url, model_name, module=["tf", "InceptionV1"])
map_proto_path = download_testdata(map_proto_url, map_proto, module="data")
label_path = download_testdata(label_map_url, label_map, module="data")
```

## 导入模型

从 protobuf 文件创建 TensorFlow 计算图定义。

```plain
with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name="")
    # 调用函数将计算图定义导入默认计算图。
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # 给计算图添加 shape
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")
```

## 解码图像

注意：

TensorFlow 前端导入不支持 JpegDecode 等预处理操作。 JpegDecode 被绕过（只返回源节点），因此我们只向 TVM 提供解码后的帧。

```plain
from PIL import Image

image = Image.open(img_path).resize((299, 299))

x = np.array(image)
```

## 将计算图导入 Relay

将 TensorFlow 计算图定义导入到 Relay 前端。

**结果：**

sym：给定 TensorFlow protobuf 的 Relay 表达式。

params：从 TensorFlow 参数 (tensor protobuf) 转换而来的参数。

```plain
shape_dict = {"DecodeJpeg/contents": x.shape}
dtype_dict = {"DecodeJpeg/contents": "uint8"}
mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

print("Tensorflow protobuf imported to relay frontend.")
```

输出结果：

```plain
/workspace/python/tvm/relay/frontend/tensorflow.py:535: UserWarning: Ignore the passed shape. Shape in graphdef will be used for operator DecodeJpeg/contents.
  "will be used for operator %s." % node.name
/workspace/python/tvm/relay/frontend/tensorflow_ops.py:1009: UserWarning: DecodeJpeg: It's a pass through, please handle preprocessing before input
  warnings.warn("DecodeJpeg: It's a pass through, please handle preprocessing before input")
Tensorflow protobuf imported to relay frontend.
```

## Relay 构建

用给定的输入规范，将计算图编译为 LLVM target。

**结果：**

graph：编译后的最终计算图。

params：编译后的最终参数。

lib：target 库（可用 TVM runtime 部署到 target 上） 。

```plain
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
```

输出结果：

```plain
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 在 TVM 上执行可移植计算图

接下来在 target 上部署编译好的模型：

```plain
from tvm.contrib import graph_executor

dtype = "uint8"
m = graph_executor.GraphModule(lib["default"](dev))
# 设置输入
m.set_input("DecodeJpeg/contents", tvm.nd.array(x.astype(dtype)))
# 执行
m.run()
# 得到输出
tvm_output = m.get_output(0, tvm.nd.empty(((1, 1008)), "float32"))
```

## 处理输出

将 InceptionV1 模型的输出处理为人类可读文本。

```plain
predictions = tvm_output.numpy()
predictions = np.squeeze(predictions)

# 创建节点 ID --> 英文字符串查找
node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path, uid_lookup_path=label_path)

# 打印 TVM 输出的前 5 个预测。
top_k = predictions.argsort()[-5:][::-1]
for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print("%s (score = %.5f)" % (human_string, score))
```

输出结果：

```plain
African elephant, Loxodonta africana (score = 0.61481)
tusker (score = 0.30387)
Indian elephant, Elephas maximus (score = 0.03343)
banana (score = 0.00023)
rapeseed (score = 0.00021)
```

## 在 TensorFlow 上推理

在 TensorFlow 上运行对应的模型：

```plain
def create_graph():
    """从已保存的 GraphDef 文件创建一个计算图，并返回 saver。"""
    # 从已保存的 graph_def.pb 创建图形
    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        # 调用函数将计算图定义导入默认计算图。
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)



def run_inference_on_image(image):
    """在图像上进行推理。

    参数
    ----------
    image: String 类型
        图像文件名。

    返回值
    -------
        无
    """
    if not tf_compat_v1.gfile.Exists(image):
        tf.logging.fatal("File does not exist %s", image)
    image_data = tf_compat_v1.gfile.GFile(image, "rb").read()

    # 从已保存的 GraphDef 创建计算图。
    create_graph()

    with tf_compat_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name("softmax:0")
        predictions = sess.run(softmax_tensor, {"DecodeJpeg/contents:0": image_data})

        predictions = np.squeeze(predictions)

        # 创建节点 ID --> 英文字符查找
        node_lookup = tf_testing.NodeLookup(
            label_lookup_path=map_proto_path, uid_lookup_path=label_path
        )

        # 打印 TensorFlow 的前 5 个预测。
        top_k = predictions.argsort()[-5:][::-1]
        print("===== TENSORFLOW RESULTS =======")
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print("%s (score = %.5f)" % (human_string, score))



run_inference_on_image(img_path)
```

输出结果：

```plain
===== TENSORFLOW RESULTS =======
African elephant, Loxodonta africana (score = 0.58394)
tusker (score = 0.33909)
Indian elephant, Elephas maximus (score = 0.03186)
banana (score = 0.00022)
desk (score = 0.00019)
```

**脚本总运行时长：** （1 分 6.352 秒）

`下载 Python 源代码：from_tensorflow.py`

`下载 Jupyter Notebook：from_tensorflow.ipynb`