# Relay BNNS 集成

**作者**：[Egor Churaev](https://github.com/echuraev)

## 介绍

Apple BNNS 库由一组函数构成，这些函数用来构建推理（和训练）过程中的神经网络。macOS、iOS、tvOS 和 watchOS 支持 Apple BNNS。 BNNS 提供在这些平台上支持的所有 CPU 上执行的原语，并针对高性能和低能耗进行了优化。这种集成将尽可能多的算子从 Relay 迁移到 BNNS。

BNNS runtime 是平台 API 的一部分，且在现代所有 Apple 的操作系统上都可用。使用 BNNS 的应用程序不依赖额外的外部依赖。

BNNS 函数使用了 Apple 尚未公开的私有硬件功能，例如 AMX Apple CPU 扩展。

本教程演示了如何在 BNNS codegen 和 runtime 启用的情况下构建 TVM，还给出了用 BNNS runtime 编译和运行模型的示例代码，最后，还记录了支持的算子。

## 使用 BNNS 支持构建 TVM

打开 USE_BNNS 标志可将 TVM BNNS codegen 和 TVM BNNS runtime 打开。

* USE_BNNS=ON/OFF - 将子图迁移到 BNNS 原语，并把 TVM 库链接到 BNNS runtime 模块。
  启用此标志会搜索当前 target SDK（所需最低版本为 macOS 11.0、iOS 14.0、tvOS 14.0 和 watchOS 7.0）上的默认加速框架。

config.cmake 文件的设置示例：

``` cmake
set(USE_BNNS ON)
```

## Relay 计算图的 BNNS 分区

传递模块进行编译前，必须对迁移到 BNNS 执行的操作进行注释。由 *partition_for_bnns* 注释的所有操作都将被迁移到 BNNS 上执行，其余的操作将通过 LLVM 进行编译和生成代码。

重要提示：BNNS 仅支持具有恒定权重的原语，因此必须将常量映射到 Relay 表示中的相关张量抽象。若要冻结张量，并将它们作为常量进行操作，则需要用特殊标志「freeze_params=True」调用 ONNX 导入器或手动绑定执行器。所有 Relay 导入器默认都不会这样做。若将 params 字典作为参数传递，可用「partition_for_bnns」来实现。

``` python
from tvm.relay.op.contrib.bnns import partition_for_bnns
model = partition_for_bnns(model, params=params)
```

## 迁移到 BNNS 执行操作的输入数据布局

BNNS 内核仅支持平面格式的输入数据，分区器需要 NCHW 输入布局（用于 conv2d 输入）。

要对混合输入布局的模型进行 BNNS 集成，应在模块传递给 *partition_for_bnns* 之前对其进行转换。布局转换只发生在显式枚举类型的操作中。根据拓扑结构，可能会围绕 conv2d 将常规数据重新排序为交织布局和平面布局。这样做会使得性能损失，并影响执行时间。建议分析整个拓扑结构，并扩展以下列表，将所有中间张量转换为 NCHW 数据布局。

输入布局更改的示例：

``` python
# 对于具有 NHWC 输入布局的模型
with tvm.transform.PassContext(opt_level=3):
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"],
                                         "nn.bias_add": ["NCHW", "default"],
                                         "nn.relu": ["NCHW"]})(mod)
```

## 示例：使用 BNNS 构建和部署 Mobilenet v2 1.0

从 MXNet Mobilenet v2 1.0 模型创建 Relay 计算图：

``` python
import tvm
from tvm import relay
import mxnet
from mxnet.gluon.model_zoo.vision import get_model

dtype = "float32"
input_shape = (1, 3, 224, 224)
block = get_model('mobilenetv2_1.0', pretrained=True)
module, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
```

标记要迁移到 BNNS 原语的计算图部分，BNNS 集成支持的所有操作都将由 BNNS 调用处理，其余操作将通过常规 TVM LLVM 编译和生成代码。

然后将新模块编译到所需的 Apple 平台：

``` python
from tvm.relay.op.contrib.bnns import partition_for_bnns

# target for macOS Big Sur 11.1:
target = "llvm -mtriple=x86_64-apple-darwin20.2.0"

model = partition_for_bnns(model, params=params)  # to markup operations to be offloaded to BNNS
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(model, target=target, params=params)
```

导出模块：

``` python
lib.export_library('compiled.dylib')
```

用（在启用 `USE_BNNS` 条件下构建的） TVM 在目标机器上加载模块，并运行推理：

``` python
import tvm
import numpy as np
from tvm.contrib import graph_executor

dev = tvm.cpu(0)
loaded_lib = tvm.runtime.load_module('compiled.dylib')
gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

dtype = "float32"
input_shape = (1, 3, 224, 224)
input_data = np.random.uniform(0, 1, input_shape).astype(dtype)
gen_module.run(data=input_data)
```

## 支持的算子

| **Relay 节点** | **备注** |
|:---|:---|
| nn.conv2d |    |
| nn.batch_norm | BNNS 集成仅支持 nn.conv2d-batch_norm 模式 |
| nn.dense |    |
| nn.batch_matmul |    |
| nn.bias_add | BNNS 集成仅支持 nn.conv2d 或 nn.dense 中融合的偏置部分 |
| add | BNNS 集成仅支持 nn.conv2d 或 nn.dense 中融合的偏置部分 |
| nn.relu | BNNS 集成仅支持 nn.conv2d 或 nn.dense 中融合的偏置部分 |
| nn.gelu | BNNS 集成仅支持 nn.conv2d 或 nn.dense 中融合的偏置部分 |


