# Relay TensorRT 集成

**作者**：[Trevor Morris](https://github.com/trevor-m)

## 介绍

NVIDIA TensorRT 是一个用于优化深度学习推理的库。这种集成尽可能多地将算子从 Relay 迁移到 TensorRT，无需对 schedule 调优，即可提升 NVIDIA GPU 的性能。

本教程演示如何安装 TensorRT 以及如何构建 TVM，来启用 TensorRT BYOC 和 runtime。此外，还给出了示例代码，演示了如何用 TensorRT 编译和运行 ResNet-18 模型，以及如何配置编译和 runtime 设置。最后，还记录了支持的算子，以及如何扩展集成来支持其他算子。

## 安装 TensorRT

若要下载 TensorRT，需要创建一个 NVIDIA 开发者帐户，可参考 NVIDIA 的文档来了解更多信息：https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html。若有 Jetson 设备（如 TX1、TX2、Xavier 或 Nano），则 TensorRT 可能已由 JetPack SDK 安装到设备了 。

安装 TensorRT 的两种方法：

* 通过 deb 或 rpm 包系统安装。
* 通过 tar 文件安装。

用 tar 文件的安装方法，必须将解压后的 tar 路径传到 USE_TENSORRT_RUNTIME=/path/to/TensorRT 中；用系统安装的方法，USE_TENSORRT_RUNTIME=ON 会自动定位安装路径。

## 使用 TensorRT 支持构建 TVM

TVM 的 TensorRT 集成有两个单独的构建标志，这些标志启用了交叉编译：USE_TENSORRT_CODEGEN=ON —— 在支持 TensorRT 的主机上构建模块； USE_TENSORRT_RUNTIME=ON —— 使得边界设备上的 TVM runtime 能够执行 TensorRT 模块。若要编译并执行具有相同 TVM 构建的模型，则应启用这两者。

* USE_TENSORRT_CODEGEN=ON/OFF - 使得无需任何 TensorRT 库即可编译 TensorRT 模块。
* USE_TENSORRT_RUNTIME=ON/OFF/path-to-TensorRT - 启用 TensorRT runtime 模块，它用已安装的 TensorRT 库来构建 TVM。

config.cmake 文件中的设置示例：

``` cmake
set(USE_TENSORRT_CODEGEN ON)
set(USE_TENSORRT_RUNTIME /home/ubuntu/TensorRT-7.0.0.11)
```

## 使用 TensorRT 构建和部署 ResNet-18

从 MXNet ResNet-18 模型中创建 Relay 计算图：

``` python
import tvm
from tvm import relay
import mxnet
from mxnet.gluon.model_zoo.vision import get_model

dtype = "float32"
input_shape = (1, 3, 224, 224)
block = get_model('resnet18_v1', pretrained=True)
mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
```

为 TensorRT 计算图进行注释和分区，TensorRT 集成的所有操作都被标记并迁移到 TensorRT，其余的操作将通过常规的 TVM CUDA 编译和代码生成。

``` python
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
mod, config = partition_for_tensorrt(mod, params)
```

用 partition_for_tensorrt 返回的新模块和配置来构建 Relay 计算图。target 必须始终是 CUDA target。`partition_for_tensorrt` 会自动填充配置中所需的值，因此无需修改它——只需将其传给 PassContext，就可在编译时被读取到。

``` python
target = "cuda"
with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
    lib = relay.build(mod, target=target, params=params)
```

导出模块：

``` python
lib.export_library('compiled.so')
```

在目标机器上加载模块并运行推理，这个过程必须确保启用 `USE_TENSORRT_RUNTIME`。第一次运行时因为要构建 TensorRT 引擎，所以需要较长时间。

``` python
dev = tvm.cuda(0)
loaded_lib = tvm.runtime.load_module('compiled.so')
gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))
input_data = np.random.uniform(0, 1, input_shape).astype(dtype)
gen_module.run(data=input_data)
```

## 分区和编译设置

有些选项可在 `partition_for_tensorrt` 中配置：

* `version` - 用 (major, minor, patch) 元组表示的 TensorRT 版本。若在 USE_TENSORRT_RUNTIME=ON 条件下编译 TVM，则用链接的 TensorRT 版本。这个版本会影响哪些算子可以分区到 TensorRT。
* `use_implicit_batch` - 使用 TensorRT 隐式批处理模式（默认为 true）。设置为 false 会启用显式批处理模式，这种方式会扩大支持的算子（包括那些修改批处理维度的算子），但会降低某些模型的性能。
* `remove_no_mac_subgraphs` - 提高性能的启发式方法。若子图没有任何乘-加操作 (multiply-accumulate operation)，则删除已为 TensorRT 分区的子图。删除的子图将通过 TVM 的标准编译。
* `max_workspace_size` - 允许每个子图用于创建 TensorRT 引擎的工作空间 size（以字节为单位）。它可在运行时被覆盖。更多信息请参阅 TensorRT 文档。

## Runtime 设置

还有一些额外的选项，可在运行时用环境变量进行配置。

* 自动 FP16 转换 - 设置环境变量 `TVM_TENSORRT_USE_FP16=1`，从而自动将模型的 TensorRT 组件转换为 16 位浮点精度。此设置可提高性能，但可能会导致模型精度略有下降。
* 缓存 TensorRT 引擎 - runtime 会在第一次推理时调用 TensorRT API 来构建引擎。这个过程会花费很多时间，因此可设置 `TVM_TENSORRT_CACHE_DIR` 指向磁盘上的目录，这个目录保存构建的引擎。这样下次加载模型时指定相同的目录，runtime 就会加载已经构建的引擎，从而避免较长的预热时间。每个模型的目录是唯一的。
* TensorRT 有一个参数，用来配置模型中每一层可用的最大暂存空间量。最好使用最高值，它不会导致内存不足。可用 `TVM_TENSORRT_MAX_WORKSPACE_SIZE` 指定要用的工作区size（以字节为单位）来覆盖它。
* 对于包含动态 batch 维度的模型，变量 `TVM_TENSORRT_MULTI_ENGINE` 可用于确定如何在 runtime 中创建 TensorRT 引擎。默认模式下 `TVM_TENSORRT_MULTI_ENGINE=0`，在内存中每次维护一个引擎。如果输入出现更高的 batch size，则用新的 max_batch_size 设置重新构建引擎——该引擎与所有 batch size（从 1 到 max_batch_size）兼容。此模式减少了运行时使用的内存量。第二种模式，`TVM_TENSORRT_MULTI_ENGINE=1` 将构建一个独特的 TensorRT 引擎，该引擎针对遇到的每个 batch size 进行了优化。这种模式下性能更佳，但内存消耗也会更多。

## 支持的算子

| Relay节点 | 备注 |
|:---|:---|
| nn.relu |    |
| sigmoid |    |
| tanh |    |
| nn.batch_norm |    |
| nn.layer_norm |    |
| nn.softmax |    |
| nn.conv1d |    |
| nn.conv2d |    |
| nn.dense |    |
| nn.bias_add |    |
| add |    |
| subtract |    |
| multiply |    |
| divide |    |
| power |    |
| maximum |    |
| minimum |    |
| nn.max_pool2d |    |
| nn.avg_pool2d |    |
| nn.global_max_pool2d |    |
| nn.global_avg_pool2d |    |
| exp |    |
| log |    |
| sqrt |    |
| abs |    |
| negative |    |
| nn.batch_flatten |    |
| expand_dims |    |
| squeeze |    |
| concatenate |    |
| nn.conv2d_transpose |    |
| transpose |    |
| layout_transform |    |
| reshape |    |
| nn.pad |    |
| sum |    |
| prod |    |
| max |    |
| min |    |
| mean |    |
| nn.adaptive_max_pool2d |    |
| nn.adaptive_avg_pool2d |    |
| nn.batch_matmul |    |
| clip | 需要 TensorRT 的版本为 5.1.5 及以上 |
| nn.leaky_relu | 需要 TensorRT 的版本为 5.1.5 及以上 |
| sin | 需要 TensorRT 的版本为 5.1.5 及以上 |
| cos | 需要 TensorRT 的版本为 5.1.5 及以上 |
| atan | 需要 TensorRT 的版本为 5.1.5 及以上 |
| ceil | 需要 TensorRT 的版本为 5.1.5 及以上 |
| floor | 需要 TensorRT 的版本为 5.1.5 及以上 |
| split | 需要 TensorRT 的版本为 5.1.5 及以上 |
| strided_slice | 需要 TensorRT 的版本为 5.1.5 及以上 |
| nn.conv3d | 需要 TensorRT 的版本为 6.0.1 及以上 |
| nn.max_pool3d | 需要 TensorRT 的版本为 6.0.1 及以上 |
| nn.avg_pool3d | 需要 TensorRT 的版本为 6.0.1 及以上 |
| nn.conv3d_transpose | 需要 TensorRT 的版本为 6.0.1 及以上 |
| erf | 需要 TensorRT 的版本为 7.0.0 及以上 |

## 添加新算子

添加对新算子的支持，需要对一系列文件进行修改：

* *src/runtime/contrib/tensorrt/tensorrt_ops.cc* 创建一个新的算子转换器类实现 `TensorRTOpConverter` 接口。必须实现构造函数，并指定有多少输入，以及它们是张量还是权重。还必须实现 `Convert` 方法来执行转换。通过使用参数中的输入、属性和网络来添加新的 TensorRT 层，然后产生层输出。你可以使用示例中已有的转换器。最后，将新算子转换器注册到 `GetOpConverters()` 映射中。
* *python/relay/op/contrib/tensorrt.py* 这个文件包含了 TensorRT 的注解规则（决定了支持哪些算子及其属性）。必须为 Relay 算子注册一个注解函数，并根据属性的返回值为 true 还是 false 来指定转换器支持哪些属性。
* *tests/python/contrib/test_tensorrt.py* 为给定算子添加单元测试。


