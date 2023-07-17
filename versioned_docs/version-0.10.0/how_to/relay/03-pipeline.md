---
title: 在 Relay 中使用 Pipeline Executor
---

# 在 Relay 中使用 Pipeline Executor

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_relay/using_pipeline_executor.html#sphx-glr-download-how-to-work-with-relay-using-pipeline-executor-py) 下载完整的示例代码
:::

**作者**：[Hua Jiang](https://github.com/huajsj)

本教程介绍如何将「Pipeline Executor」与 Relay 配合使用。

``` python
import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm import relay
from tvm.relay import testing
import tvm.testing
from tvm.contrib.cutlass import (
    has_cutlass,
    num_cutlass_partitions,
    finalize_modules,
    finalize_modules_vm,
)

img_size = 8
```

## 创建一个简单的网络，这个网络也可以是一个预训练的模型。

创建一个由 convolution、batch normalization、dense 和 ReLU activation 组成的网络用于演示。

``` python
def get_network():
    out_channels = 16
    batch_size = 1
    data = relay.var("data", relay.TensorType((batch_size, 3, img_size, img_size), "float16"))
    dense_weight = relay.var(
        "dweight", relay.TensorType((batch_size, 16 * img_size * img_size), "float16")
    )
    weight = relay.var("weight")
    second_weight = relay.var("second_weight")
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")
    simple_net = relay.nn.conv2d(
        data=data, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    )
    simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    simple_net = relay.nn.relu(simple_net)
    simple_net = relay.nn.batch_flatten(simple_net)
    simple_net = relay.nn.dense(simple_net, dense_weight)
    simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)
    data_shape = (batch_size, 3, img_size, img_size)
    net, params = testing.create_workload(simple_net)
    return net, params, data_shape

net, params, data_shape = get_network()
```

## 将网络拆分成两个子图。

这个来自单元测试的名为「graph_split」的函数只是一个例子。用户可以创建自定义逻辑来拆分计算图。

``` python
import inspect
import os

tutorial_dir = os.path.dirname(inspect.getfile(lambda: None))
os.sys.path.append(os.path.join(tutorial_dir, "../../../tests/python/relay"))
from test_pipeline_executor import graph_split
```

将网络拆分成两个子图。

``` python
split_config = [{"op_name": "nn.relu", "op_index": 0}]
subgraphs = graph_split(net["main"], split_config, params)
```

生成的子图如下所示。

``` bash
"""
#subgraphs[0])

 def @main(%data: Tensor[(1, 3, img_size, img_size), float16]) {
  %0 = nn.conv2d(%data, meta[relay.Constant][0] /* ty=Tensor[(16, 3, 3, 3), float16] */, padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3]) /* ty=Tensor[(1, 16, img_size, img_size), float16] */;
  %1 = nn.batch_norm(%0, meta[relay.Constant][1] /* ty=Tensor[(16), float16] */, meta[relay.Constant][2] /* ty=Tensor[(16), float16]*/, meta[relay.Constant][3] /* ty=Tensor[(16), float16] */, meta[relay.Constant][4] /* ty=Tensor[(16), float16] */) /* ty=(Tensor[(1,16, img_size, img_size), float16], Tensor[(16), float16], Tensor[(16), float16]) */;
  %2 = %1.0;
  nn.relu(%2) /* ty=Tensor[(1, 16, img_size, img_size), float16] */
 }

#subgraphs[1]

 def @main(%data_n_0: Tensor[(1, 16, 8, 8), float16] /* ty=Tensor[(1, 16, 8, 8), float16] */) {
  %0 = nn.batch_flatten(%data_n_0) /* ty=Tensor[(1, 1024), float16] */;
  nn.dense(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 1024), float16] */, units=None) /* ty=Tensor[(1, 1), float16] */
 }

"""
```

## 用 cutlass target 构建子图。

``` python
cutlass = tvm.target.Target(
    {
        "kind": "cutlass",
        "sm": int(tvm.target.Target("cuda").arch.split("_")[1]),
        "use_3xtf32": True,
        "split_k_slices": [1],
        "profile_all_alignments": False,
        "find_first_valid": True,
        "use_multiprocessing": True,
        "use_fast_math": False,
        "tmp_dir": "./tmp",
    },
    host=tvm.target.Target("llvm"),
)

def cutlass_build(mod, target, params=None, target_host=None, mod_name="default"):
    target = [target, cutlass]
    lib = relay.build_module.build(
        mod, target=target, params=params, target_host=target_host, mod_name=mod_name
    )
    return lib
```

## 使用 pipeline executor 在 pipeline 中运行两个子图。

在 cmake 中将 `USE_PIPELINE_EXECUTOR` 和 `USE_CUTLASS` 设置为 ON。

``` python
from tvm.contrib import graph_executor, pipeline_executor, pipeline_executor_build
```

创建子图 pipeline 配置。将子图模块与 target 关联起来。使用 CUTLASS BYOC 构建第二个子图模块。

``` python
mod0, mod1 = subgraphs[0], subgraphs[1]
# 将 cutlass 作为 codegen。
mod1 = partition_for_cutlass(mod1)
```

获取 pipeline executor 配置对象。

``` python
pipe_config = pipeline_executor_build.PipelineConfig()
```

设置子图模块的编译 target。

``` python
pipe_config[mod0].target = "llvm"
pipe_config[mod0].dev = tvm.cpu(0)
```

将第二个子图模块的编译 target 设置为 cuda。

``` python
pipe_config[mod1].target = "cuda"
pipe_config[mod1].dev = tvm.device("cuda", 0)
pipe_config[mod1].build_func = cutlass_build
pipe_config[mod1].export_cc = "nvcc"
# 通过连接子图模块创建 pipeline。
# 全局输入将被转发到第一个名为 mod0 的模块的输入接口
pipe_config["input"]["data"].connect(pipe_config[mod0]["input"]["data"])
# mod0 的第一个输出会转发到 mod1 的输入接口
pipe_config[mod0]["output"][0].connect(pipe_config[mod1]["input"]["data_n_0"])
# mod1 的第一个输出将是第一个全局输出。
pipe_config[mod1]["output"][0].connect(pipe_config["output"][0])
```

pipeline 配置如下：

``` bash
"""
print(pipe_config)
 Inputs
  |data: mod0:data

 output
  |output(0) : mod1.output(0)

 connections
  |mod0.output(0)-> mod1.data_n_0
"""
```

## 构建 pipeline executor。

``` python
with tvm.transform.PassContext(opt_level=3):
    pipeline_mod_factory = pipeline_executor_build.build(pipe_config)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:267: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

将参数配置导出到一个文件中。

``` python
directory_path = tvm.contrib.utils.tempdir().temp_dir
os.makedirs(directory_path, exist_ok=True)
config_file_name = pipeline_mod_factory.export_library(directory_path)
```

## 使用 load 函数创建和初始化 PipelineModule。

``` python
pipeline_module = pipeline_executor.PipelineModule.load_library(config_file_name)
```

## 运行 pipeline executor。

分配输入数据。

``` python
data = np.random.uniform(-1, 1, size=data_shape).astype("float16")
pipeline_module.set_input("data", tvm.nd.array(data))
```

以 pipeline 模式运行两个子图，异步或同步获取输出。以下示例为同步获取输出。

``` python
pipeline_module.run()
outputs = pipeline_module.get_output()
```

## 使用 graph_executor 进行验证。

用 graph_executor 依次运行这两个子图，得到输出。

``` python
target = "llvm"
dev0 = tvm.device(target, 0)
lib0 = relay.build_module.build(mod0, target, params=params)
module0 = runtime.GraphModule(lib0["default"](dev0))
cuda = tvm.target.Target("cuda", host=tvm.target.Target("llvm"))
lib1 = relay.build_module.build(mod1, [cuda, cutlass], params=params)
lib1 = finalize_modules(lib1, "compile.so", "./tmp")

dev1 = tvm.device("cuda", 0)

module1 = runtime.GraphModule(lib1["default"](dev1))

module0.set_input("data", data)
module0.run()
out_shape = (1, 16, img_size, img_size)
out = module0.get_output(0, tvm.nd.empty(out_shape, "float16"))
module1.set_input("data_n_0", out)
module1.run()
out_shape = (1, 1)
out = module1.get_output(0, tvm.nd.empty(out_shape, "float16"))
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:267: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

验证结果。

``` python
tvm.testing.assert_allclose(outputs[0].numpy(), out.numpy())
```

[下载 Python 源代码：using_pipeline_executor.py](https://tvm.apache.org/docs/_downloads/29c30a5341c6aa08601b51791150fa4b/using_pipeline_executor.py)

[下载 Jupyter Notebook：using_pipeline_executor.ipynb](https://tvm.apache.org/docs/_downloads/f407f66fb8174d0d4ec37407af1128d6/using_pipeline_executor.ipynb)