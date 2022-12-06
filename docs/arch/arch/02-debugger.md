---
title: 调试器
sidebar_position: 120
---

# 调试器

TVM 调试器是一个用于调试 TVM 计算图执行过程的接口。它有助于访问 TVM runtime 的计算图结构和张量值。

## 调试交换格式

### 1. 计算图

由 Relay 以 JSON 序列化格式构建的优化图按原样转储。包含有关计算图的全部信息。UX 可以直接使用此图，也可以将此图转换为 UX 可以理解的格式。

Graph JSON 格式解释如下

1. `nodes` 节点是占位符，或是 json 中的计算节点。节点存储为列表。一个节点包含以下信息：
   * `op` - 操作类型，`null` 表示它是一个占位符/变量/输入节点，`tvm_op` 表示该节点可以执行
   * `name` - 节点名称
   * `inputs` - 算子的输入位置，Inputs 是一个（nodeid、index、version）元组列表。（可选）
   * `attrs` - 包含以下信息的节点属性
      * `flatten_data` - 执行前是否需要对数据进行展开
      * `func_name` - 融合函数名，对应 Relay 编译过程生成的 lib 的符号。
      * `num_inputs` - 节点的输入数量
      * `num_outputs`  - 节点产生的输出数量
2. `arg_nodes`：arg_nodes 是节点的索引列表，它是计算图的占位符/变量/输入或常量/参数。
3. `heads`：heads 是计算图输出的条目列表。
4. `node_row_ptr`：node_row_ptr 存储前向传播路径的历史记录，因此可以在推理任务中跳过构建整个计算图。
5. `attrs`：attrs 包含版本号或类似的有用信息。
   * `storage_id` - 存储布局中每个节点的内存插槽 id。
   * `dtype` - 每个节点的数据类型（枚举值）。
   * `dltype` - 按顺序排列的每个节点的数据类型。
   * `shape` - 每个节点的 shape 为 k 阶。
   * `device_index` - 计算图中每个条目的设备分配。

转储图示例：

``` json
{
  "nodes": [                                    # 节点列表
    {
      "op": "null",                             # operation type = null，这是一个占位符/变量/输入或常量/参数节点
      "name": "x",                              # 参数节点的名称
      "inputs": []                              # 此节点的输入，这里为 none，因为这是一个参数节点
    },
    {
      "op": "tvm_op",                           # operation type = tvm_op，这个节点可以执行
      "name": "relu0",                          # 节点名称
      "attrs": {                                # 节点的属性
        "flatten_data": "0",                    # 此数据是否需要展开
        "func_name": "fuse_l2_normalize_relu",  # 融合函数名，对应编译过程生成的 lib 的符号
        "num_inputs": "1",                      # 此节点的输入数量
        "num_outputs": "1"                      # 此节点产生的输出数量
      },
      "inputs": [[0, 0, 0]]                     # 此操作的输入位置
    }
  ],
  "arg_nodes": [0],                             # 其中所有节点都是参数节点
  "node_row_ptr": [0, 1, 2],                    # 用于更快深度优先搜索的行索引
  "heads": [[1, 0, 0]],                         # 此操作的输出节点的位置
  "attrs": {                                    # 计算图的属性
    "storage_id": ["list_int", [1, 0]],         # 存储布局中每个节点的内存插槽 ID
    "dtype": ["list_int", [0, 0]],              # 每个节点的数据类型（枚举值）
    "dltype": ["list_str", [                    # 按顺序排列的每个节点的数据类型
        "float32",
        "float32"]],
    "shape": ["list_shape", [                   # 每个节点的 shape 为 k 阶
        [1, 3, 20, 20],
        [1, 3, 20, 20]]],
    "device_index": ["list_int", [1, 1]],       # 按顺序为每个节点分配设备
  }
}
```

### 2. 张量转储

执行返回的张量是 `tvm.ndarray` 类型。所有张量都将以序列化格式保存为二进制字节。结果二进制字节可以通过 API「load_params」加载。

**加载参数示例**

``` python
with open(path_params, “rb”) as fi:
    loaded_params = bytearray(fi.read())
module.load_params(loaded_params)
```

## 如何使用调试器

1. 在 `config.cmake` 中把 `USE_PROFILER` 标志设置为 `ON`

   ``` cmake
   # 是否开启额外的计算图调试功能
   set(USE_PROFILER ON)
   ```

2. 执行 `make` tvm，生成 `libtvm_runtime.so`

3. 在前端脚本文件中导入 `GraphModuleDebug`，`from tvm.contrib.debugger.debug_executor import GraphModuleDebug`，而非 `from tvm.contrib import graph_executor`。

   ``` python
   from tvm.contrib.debugger.debug_executor import GraphModuleDebug
   m = GraphModuleDebug(
       lib["debug_create"]("default", dev),
       [dev],
       lib.graph_json,
       dump_root="/tmp/tvmdbg",
   )
   # 设置输入
   m.set_input('data', tvm.nd.array(data.astype(dtype)))
   m.set_input(**params)
   # 执行
   m.run()
   tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).numpy()
   ```

4. 如果之前和共享对象文件/动态链接库一样，**用的是 `lib.export_library("network.so")` 将网络导出到外部库**，调试 runtime 的初始化会略有不同。

   ``` python
   lib = tvm.runtime.load_module("network.so")
   m = graph_executor.create(lib["get_graph_json"](), lib, dev, dump_root="/tmp/tvmdbg")
   # 设置输入
   m.set_input('data', tvm.nd.array(data.astype(dtype)))
   m.set_input(**params)
   # 执行
   m.run()
   tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).numpy()
   ```

输出被转储到 `/tmp` 文件夹中的临时文件夹，或创建 runtime 时指定的文件夹。

## 样本输出

以下是调试器的输出示例：

``` bash
Node Name               Ops                                                                  Time(us)   Time(%)  Start Time       End Time         Shape                Inputs  Outputs
---------               ---                                                                  --------   -------  ----------       --------         -----                ------  -------
1_NCHW1c                fuse___layout_transform___4                                          56.52      0.02     15:24:44.177475  15:24:44.177534  (1, 1, 224, 224)     1       1
_contrib_conv2d_nchwc0  fuse__contrib_conv2d_NCHWc                                           12436.11   3.4      15:24:44.177549  15:24:44.189993  (1, 1, 224, 224, 1)  2       1
relu0_NCHW8c            fuse___layout_transform___broadcast_add_relu___layout_transform__    4375.43    1.2      15:24:44.190027  15:24:44.194410  (8, 1, 5, 5, 1, 8)   2       1
_contrib_conv2d_nchwc1  fuse__contrib_conv2d_NCHWc_1                                         213108.6   58.28    15:24:44.194440  15:24:44.407558  (1, 8, 224, 224, 8)  2       1
relu1_NCHW8c            fuse___layout_transform___broadcast_add_relu___layout_transform__    2265.57    0.62     15:24:44.407600  15:24:44.409874  (64, 1, 1)           2       1
_contrib_conv2d_nchwc2  fuse__contrib_conv2d_NCHWc_2                                         104623.15  28.61    15:24:44.409905  15:24:44.514535  (1, 8, 224, 224, 8)  2       1
relu2_NCHW2c            fuse___layout_transform___broadcast_add_relu___layout_transform___1  2004.77    0.55     15:24:44.514567  15:24:44.516582  (8, 8, 3, 3, 8, 8)   2       1
_contrib_conv2d_nchwc3  fuse__contrib_conv2d_NCHWc_3                                         25218.4    6.9      15:24:44.516628  15:24:44.541856  (1, 8, 224, 224, 8)  2       1
reshape1                fuse___layout_transform___broadcast_add_reshape_transpose_reshape    1554.25
```
