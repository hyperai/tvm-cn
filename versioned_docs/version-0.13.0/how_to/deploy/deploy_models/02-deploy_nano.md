---
title: 在 Jetson Nano 上部署预训练模型
---

# 在 Jetson Nano 上部署预训练模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_nano.html#sphx-glr-download-how-to-deploy-models-deploy-model-on-nano-py) 下载完整的示例代码
:::

**作者**：[BBuf](https://github.com/BBuf)

此教程介绍如何用 Relay 编译 ResNet 模型，并将其部署到 Jetson Nano。

``` python
import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, graph_executor as runtime
from tvm.contrib.download import download_testdata
```

## 在 Jetson Nano 上构建 TVM Runtime

第一步是在远程设备上构建 TVM runtime。

:::note
本节和下一节中的所有指令都应在目标设备（例如 Jetson Nano）及 Linux 上执行。
:::

由于我们是在本地机器上进行编译，远程设备仅用于运行生成的代码，因此只需在远程设备上构建 TVM runtime。

``` bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make runtime -j4
```

:::note
如果要用 Jetson Nano 的 GPU 进行推理，需要在 *config.cmake* 中开启 CUDA 选项，即 *set(USE_CUDA ON)*。
:::

runtime 构建完成后，在 `~/.bashrc` 文件中设置环境变量。可以用 `vi ~/.bashrc` 命令来编辑 `~/.bashrc`，添加下面这行代码（假设 TVM 目录在 `~/tvm` 中）：

``` bash
export PYTHONPATH=$PYTHONPATH:~/tvm/python
```

执行 `source ~/.bashrc` 来更新环境变量。

## 在设备上设置 RPC 服务器

若要启动 RPC 服务器，请在远程设备（示例中为 Jetson Nano）上运行以下命令：

``` bash
python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9091
```

看到如下结果，则表示 RPC 服务器启动成功：

``` bash
INFO:RPCServer:bind to 0.0.0.0:9091
```

## 准备预训练模型

注意：确保主机已经（用 LLVM）安装了完整的 TVM。

使用 [MXNet Gluon 模型集合](https://mxnet.apache.org/api/python/gluon/model_zoo.html) 中的预训练模型。更多有关这部分的信息详见 [编译 MXNet 模型](../../compile/compile_mxnet) 教程。

``` python
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
import numpy as np

# 获取模型
block = get_model("resnet18_v1", pretrained=True)
```

为了测试模型，下载一张猫的图片，并转换其格式：

``` python
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_name = "cat.png"
img_path = download_testdata(img_url, img_name, module="data")
image = Image.open(img_path).resize((224, 224))

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)
```

用 synset 将 ImageNet 类的标签转换为人类更容易理解的单词。

``` python
synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())
```

以下代码可将 Gluon 模型移植到可移植的计算图上：

``` python
# 在 mxnet.gluon 中支持 MXNet 静态图（符号）和 HybridBlock
shape_dict = {"data": x.shape}
mod, params = relay.frontend.from_mxnet(block, shape_dict)
# 添加 softmax 算子提高概率
func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)
```

以下是一些基本的数据工作负载配置：

``` python
batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
```

## 编译计算图

用计算图的配置和参数调用 `relay.build()` 函数，从而编译计算图。但是，不能在具有 ARM 指令集的设备上部署 x86 程序。除了用来指定深度学习工作负载的参数 `net` 和 `params`，Relay 还需要知道目标设备的编译选项。不同选项会导致性能不同。

如果在 x86 服务器上运行示例，可将其设置为 `llvm`。如果在 Jetson Nano 上运行，需要将其设置为 `nvidia/jetson-nano`。若要在真实设备上运行，需将 `local_demo` 设置为 False。

``` python
local_demo = True

if local_demo:
    target = tvm.target.Target("llvm")
else:
    target = tvm.target.Target("nvidia/jetson-nano")
    assert target.kind.name == "cuda"
    assert target.attrs["arch"] == "sm_53"
    assert target.attrs["shared_memory_per_block"] == 49152
    assert target.attrs["max_threads_per_block"] == 1024
    assert target.attrs["thread_warp_size"] == 32
    assert target.attrs["registers_per_block"] == 32768

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target, params=params)

# 在 `relay.build` 之后，会得到三个返回值：计算图，库和新参数，因为我们做了一些优化，它们会改变参数，但模型的结果不变。

# 将库保存在本地临时目录中。
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.tar")
lib.export_library(lib_fname)
```

输出结果：

``` bash
/workspace/python/tvm/relay/build_module.py:348: DeprecationWarning: Please use input parameter mod (tvm.IRModule) instead of deprecated parameter mod (tvm.relay.function.Function)
  DeprecationWarning,
/workspace/python/tvm/driver/build_module.py:267: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

## 通过 RPC 远程部署模型

利用 RPC 可将模型从主机部署到远程设备。

``` python
# 从远程设备获取 RPC session。
if local_demo:
    remote = rpc.LocalSession()
else:
    # 下面是教程环境，把这个改成你目标设备的 IP 地址
    host = "192.168.1.11"
    port = 9091
    remote = rpc.connect(host, port)

# 将库上传到远程设备，并加载它
remote.upload(lib_fname)
rlib = remote.load_module("net.tar")

# 创建远程 runtime 模块
if local_demo:
    dev = remote.cpu(0)
else:
    dev = remote.cuda(0)

module = runtime.GraphModule(rlib["default"](dev))
# 设置输入数据
module.set_input("data", tvm.nd.array(x.astype("float32")))
# 运行
module.run()
# 获取输出
out = module.get_output(0)
# 得到排第一的结果
top1 = np.argmax(out.numpy())
print("TVM prediction top-1: {}".format(synset[top1]))
```

输出结果：

``` bash
TVM prediction top-1: tiger cat
```

[下载 Python 源代码：deploy_model_on_nano.py](https://tvm.apache.org/docs/_downloads/3fde0fe8b31bf786dec2a01858372eae/deploy_model_on_nano.py)

[下载 Jupyter Notebook：deploy_model_on_nano.ipynb](https://tvm.apache.org/docs/_downloads/cafefaac0e14b00fd7644da616cab35a/deploy_model_on_nano.ipynb)