# Relay Arm® 计算库集成

**作者**：[Luke Hutton](https://github.com/lhutton1)

## 介绍

Arm 计算库 (ACL) 是一个开源项目，它为 Arm CPU 和 GPU 提供了加速内核。目前，集成将算子迁移到 ACL 以在库中使用手工制作的汇编程序例程。通过将选择算子从 Relay 计算图迁移到 ACL，可在此类设备上实现性能提升。

## 安装 Arm 计算库

安装 Arm 计算库前，了解要构建的架构非常重要。一种方法是使用 *lscpu*，并查找 CPU 的“模型名称”，然后，可以使用它通过在线查看来确定架构。

TVM 目前只支持 v21.08 版本的 ACL，构建和安装所需的库的推荐方法如下：

* 使用位于 *docker/install/ubuntu_download_arm_compute_lib_binaries.sh* 的脚本，为 *target_lib* 指定的架构和扩展下载 ACL 二进制文件，它们将安装到 *install_path* 表示的位置。
* 或从 https://github.com/ARM-software/ComputeLibrary/releases 下载预构建的二进制文件。 使用此包时，要为所需的架构和扩展选择二进制文件，并确保它们对 CMake 可见：

   ``` bash
     cd <acl-prebuilt-package>/lib
     mv ./<architecture-and-extensions-required>/* .
   ```

这两种情况都要将 USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR 设置为 ACL 包所在的路径。 CMake 会在 /path-to-acl/，/path-to-acl/lib 和 /path-to-acl/build 中查找所需的二进制文件。如何使用这些配置选项，请参阅下一小节。

## 使用 ACL support 构建

当前的实现在 CMake 中有两个单独的构建选项。这种拆分的原因是 ACL 不能在 x86 机器上使用。但是，我们仍希望在 x86 机器上编译 ACL runtime 模块。

* USE_ARM_COMPUTE_LIB=ON/OFF - 启用此标志能添加对编译 ACL runtime 模块的支持。
* USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR=ON/OFF/path-to-acl - 启用此标志将允许图执行器计算 ACL 迁移的函数。
  
这些标志可根据你的设置应用于不同的场景。例如，若要在 x86 机器上编译 ACL 模块，并通过 RPC 在远程 Arm 设备上运行，则需要在 x86 机器上设置 USE_ARM_COMPUTE_LIB=ON，在远程 AArch64 设备上设置 USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR=ON。

默认这两个选项都设置为 OFF。设置 USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR=ON 意味着 CMake 会在默认位置和 /path-to-tvm-project/acl/ 目录下搜索 ACL 二进制文件（参阅 https://cmake.org/cmake/help/v3.4/command/find_library.html）。若要设置搜索 ACL 的路径，可在 ON 的位置指定。

这些标志应在 config.cmake 文件中进行设置，如：

``` cmake
set(USE_ARM_COMPUTE_LIB ON)
set(USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR /path/to/acl)
```

## 使用

:::note
此部分可能与 API 的更改不同步。
:::

创建一个 Relay 计算图（单个算子或整个计算图），使得任何 Relay 计算图都可以作为输入。ACL 集成只会选择支持的算子进行迁移，而其他的将通过 TVM 计算。（本例用的是单个 max_pool2d 算子）。

``` python
import tvm
from tvm import relay

data_type = "float32"
data_shape = (1, 14, 14, 512)
strides = (2, 2)
padding = (0, 0, 0, 0)
pool_size = (2, 2)
layout = "NHWC"
output_shape = (1, 7, 7, 512)

data = relay.var('data', shape=data_shape, dtype=data_type)
out = relay.nn.max_pool2d(data, pool_size=pool_size, strides=strides, layout=layout, padding=padding)
module = tvm.IRModule.from_expr(out)
```

为 ACL 的计算图进行注释和分区：

``` python
from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
module = partition_for_arm_compute_lib(module)
```

构建 Relay 计算图：

``` python
target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(module, target=target)
```

导出模块：

``` bash
lib_path = '~/lib_acl.so'
cross_compile = 'aarch64-linux-gnu-c++'
lib.export_library(lib_path, cc=cross_compile)
```

必须在 Arm 设备上运行推理。若在 x86 设备上编译，在 AArch64 上运行，则需要借助 RPC 机制（参考 [RPC 机制的使用教程](../../user_tutorial/rpc)）。

``` python
dev = tvm.cpu(0)
loaded_lib = tvm.runtime.load_module('lib_acl.so')
gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))
d_data = np.random.uniform(0, 1, data_shape).astype(data_type)
map_inputs = {'data': d_data}
gen_module.set_input(**map_inputs)
gen_module.run()
```

## 更多示例

以上示例仅展示了如何用 ACL 迁移单个 Maxpool2D 的基本示例。若要查看所有实现的算子和网络的更多示例，参阅 tests：*tests/python/contrib/test_arm_compute_lib*（可修改 *test_config.json* 来配置如何在 *infrastructure.py* 中创建远程设备，从而配置 runtime 测试的运行方式。

以 *test_config.json* 的配置为例：

* connection_type - RPC 连接的类型。选项：local、tracker 和 remote。
* host - 要连接的主机设备。
* port - 连接时使用的端口。
* target - 用于编译的 target。
* device_key - 通过 tracker 连接时的设备密钥。
* cross_compile - 连接非 arm 平台时交叉编译器的路径，例如 aarch64-linux-gnu-g++。

``` json
{
  "connection_type": "local",
  "host": "127.0.0.1",
  "port": 9090,
  "target": "llvm -mtriple=aarch64-linux-gnu -mattr=+neon",
  "device_key": "",
  "cross_compile": ""
}
```

## 支持的算子

| Relay 节点 | 备注 |
|:---|:---|
| nn.conv2d | **fp32:** <br/> Simple: nn.conv2d Composite: nn.pad?, nn.conv2d, nn.bias_add?, nn.relu? <br/> 支持深度和普通卷积（内核为 3x3 或 5x5 且步幅为 1x1 或 2x2 时），不支持分组卷积。``` |
| qnn.conv2d | **uint8:** <br/>     Composite: nn.pad?, nn.conv2d, nn.bias_add?, nn.relu?, qnn.requantizeNormal <br/> 支持深度和普通卷积（当内核为 3x3 或 5x5，步长为 1x1 或 2x2 时），不支持分组卷积。 |
| nn.dense | **fp32:** <br/>     Simple: nn.dense Composite: nn.dense, nn.bias_add? |
| qnn.dense | **uint8:** <br/>     Composite: qnn.dense, nn.bias_add?, qnn.requantize |
| nn.max_pool2d | fp32, uint8 |
| nn.global_max_pool2d | fp32, uint8 |
| nn.avg_pool2d | **fp32:** <br/>     Simple: nn.avg_pool2d <br/> **uint8:** <br/>     Composite: cast(int32), nn.avg_pool2d, cast(uint8) |
| nn.global_avg_pool2d | **fp32:** <br/>     Simple: nn.global_avg_pool2d <br/> **uint8:** <br/>     Composite: cast(int32), nn.avg_pool2d, cast(uint8) |
| power(of 2) + nn.avg_pool2d + sqrt | L2 池化的一种特殊情况。 <br/> **fp32:** <br/>     Composite: power(of 2), nn.avg_pool2d, sqrt |
| reshape | fp32, uint8 |
| maximum | fp32 |
| add | fp32 |
| qnn.add | uint8 |

:::note
复合算子由映射到单个 Arm 计算库的算子组成。从 Arm 计算库的角度来看，可以将其视为单个融合算子。“?”是构成复合算子的一系列算子中的可选算子。
:::

## 添加新算子

添加新算子需要修改多处，本节将分享需要修改的内容和位置，但不会深入探讨单个算子的复杂性（这个问题留给开发者思考）。

下面是要修改的几个文件：

* python/relay/op/contrib/arm_compute_lib.py：定义了要用 op.register 装饰器迁移的算子——意味着注释 pass 认为此算子可迁移 ACL。
* src/relay/backend/contrib/arm_compute_lib/codegen.cc：实现 Create[OpName]JSONNode 的方法；声明算子如何由 JSON 表示，可用来创建 ACL 模块。
* src/runtime/contrib/arm_compute_lib/acl_runtime.cc：实现 Create[OpName]Layer 方法；定义如何用 JSON 表示来创建 ACL 函数；只定义了如何将 JSON 表示转换为 ACL API。
* tests/python/contrib/test_arm_compute_lib：为给定的算子添加单元测试。


