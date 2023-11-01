---
title: 使用 TVMC Python 入门：TVM 的高级 API
---

:::note
注意：单击 [此处](https://tvm.apache.org/docs/tutorial/tvmc_python.html#sphx-glr-download-tutorial-tvmc-python-py) 下载完整的示例代码
:::

**作者**：[Jocelyn Shiue](https://github.com/CircleSpin)

本节将介绍针对 TVM 初学者设计的脚本工具。

开始前如果没有下载示例模型，需要先通过终端下载 resnet 模型：

``` bash
mkdir myscripts
cd myscripts
wget https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx
mv resnet50-v2-7.onnx my_model.onnx
touch tvmcpythonintro.py
```

用你熟悉的文本编辑器来编辑 Python 文件。

## 第 0 步：导入

``` python
from tvm.driver import tvmc
```

## 第 1 步：加载模型

将模型导入 TVMC。这一步将机器学习模型从支持的框架，转换为 TVM 的高级图形表示语言 —— Relay。这是为 TVM 中的所有模型统一起点。目前支持的框架：Keras、ONNX、TensorFlow、TFLite 和 PyTorch。

``` python
model = tvmc.load('my_model.onnx') # 第 1 步：加载
```

查看 Relay，可运行 `model.summary()`。

所有框架都支持用 shape_dict 参数覆盖输入 shape。对于大多数框架，这是可选的；但对 PyTorch 是必需的，因为 TVM 无法自动搜索它。

``` python
model = tvmc.load('my_model.onnx', shape_dict={'input1' : [1, 2, 3, 4], 'input2' : [1, 2, 3, 4]}) #第一步: 加载 + shape_dict
```

推荐通过 [netron](https://netron.app/) 查看模型 input/shape_dict。打开模型后，单击第一个节点查看输入部分中的 name 和 shape。

## 第 2 步：编译

模型现在是用 Relay 表示的，下一步是将其编译到要运行的硬件（称为 target）。这个编译过程将模型从 Relay，翻译成目标机器可理解的底层语言。

编译模型需要一个 tvm.target 字符串。查看 [文档](https://tvm.apache.org/docs/api/python/target.html) 了解有关 tvm.targets 及其选项的更多信息。一些例子如下：

1. cuda (英伟达 GPU)
2. llvm (CPU)
3. llvm -mcpu=cascadelake（英特尔 CPU）

``` python
package = tvmc.compile(model, target="llvm") # 第 2 步：编译
```

编译完成后返回一个 package。

## 第 3 步：运行

编译后的 package 可在目标硬件上运行。设备输入选项有：CPU、Cuda、CL、Metal 和 Vulkan。

``` python
result = tvmc.run(package, device="cpu") # 第 3 步：运行
```

用 `print(result)` 打印结果。

## 第 1.5 步：调优（可选并推荐）

通过调优可进一步提高运行速度。此可选步骤用机器学习来查看模型（函数）中的每个操作，并找到一种更快的方法来运行它。这一步通过 cost 模型，以及对可能的 schedule 进行基准化来实现。

这里的 target 与编译过程用到的 target 是相同的。

``` python
tvmc.tune(model, target="llvm") # 第 1.5 步：可选 Tune
```

终端输出如下所示：

``` bash
[Task  1/13]  Current/Best:   82.00/ 106.29 GFLOPS | Progress: (48/769) | 18.56 s
[Task  1/13]  Current/Best:   54.47/ 113.50 GFLOPS | Progress: (240/769) | 85.36 s
.....
```

出现的 UserWarnings 可忽略。调优会使最终结果运行更快，但调优过程会耗费几个小时的时间。

参阅下面的“保存调优结果”部分，若要应用结果，务必将调优结果传给编译。

``` python
tvmc.compile(model, target="llvm", tuning_records = "records.log") # 第 2 步：编译
```

## 保存并在终端中启动进程

``` bash
python my_tvmc_script.py
```

## 示例结果

``` bash
Time elapsed for training: 18.99 s
Execution time summary:
mean (ms)   max (ms)   min (ms)   std (ms)
  25.24      26.12      24.89       0.38



Output Names:
['output_0']
```

## TVMC 附加功能

## 保存模型

加载模型（第 1 步）后可保存 Relay 版本来提高之后的工作效率。模型将被储存在你指定的位置，随后可以被转换过的语法使用。

``` python
model = tvmc.load('my_model.onnx') # 第 1 步：加载
model.save(desired_model_path)
```

## 保存 package

模型编译完成（第 2 步）后，可将 package 保存下来。

``` python
tvmc.compile(model, target="llvm", package_path="whatever") # 第 2 步：编译

new_package = tvmc.TVMCPackage(package_path="whatever")
result = tvmc.run(new_package, device="cpu") # 第 3 步：运行
```

## 使用 Autoscheduler

使用下一代 TVM，运行速度会更快。schedule 的搜索空间以前是手写的，而现在是自动生成的。 （了解更多： [1](https://tvm.apache.org/2021/03/03/intro-auto-scheduler)，[2](https://arxiv.org/abs/2006.06762)）

``` python
tvmc.tune(model, target="llvm", enable_autoscheduler = True)
```

## 保存调优结果

把调优结果保存在文件中，方便以后复用。

* 方法 1:
   ``` python
   log_file = "hello.json"

   # 运行 tuning
   tvmc.tune(model, target="llvm", tuning_records=log_file)

   ...

   # 运行 tuning，然后复用 tuning 的结果
   tvmc.tune(model, target="llvm",prior_records=log_file)
   ```

* 方法 2:
   ``` python
   # 运行 tuning
   tuning_records = tvmc.tune(model, target="llvm")

   ...

   # 运行 tuning，然后复用 tuning 的结果
   tvmc.tune(model, target="llvm",prior_records=tuning_records)
   ```

## 对更复杂的模型调优

如果 T 的打印类似 `.........T.T..T..T..T.T.T.T.T.T.`，则增加搜索时间范围：

``` python
tvmc.tune(model, trials=10000, timeout=10,)
```

## 为远程设备编译模型

为不在本地计算机上的硬件进行编译时，TVMC 支持使用远程过程调用（RPC）。要设置 RPC 服务器，可参考本 [文档](https://tvm.apache.org/docs/tutorials/get_started/cross_compilation_and_rpc.html) 中的“在设备上设置 RPC 服务器”部分。

TVMC 脚本包括以下内容，并进行了相应调整：

``` python
tvmc.tune(
     model,
     target=target, # 编译 target 为字符串 // 要编译的设备
     target_host=target_host, # 主机处理器
     hostname=host_ip_address, # 远程基准测试时使用的 RPC 跟踪器的 IP 地址
     port=port_number, # 要连接的 RPC 跟踪器的端口。默认为 9090。
     rpc_key=your_key, # 目标设备的 RPC 跟踪器密钥。提供 rpc_tracker 时需要
)
```

[`下载 Python 源代码：tvmc_python.py`](https://tvm.apache.org/docs/_downloads/10724e9ad9c29faa223c1d5eab6dbef9/tvmc_python.py)

[`下载 Jupyter Notebook：tvmc_python.ipynb`](https://tvm.apache.org/docs/_downloads/8d55b8f991fb704002f768367ce2d1a2/tvmc_python.ipynb)
