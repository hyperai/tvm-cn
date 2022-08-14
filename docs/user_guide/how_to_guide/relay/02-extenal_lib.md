---
title: 在 Relay 中使用外部库
---

# 在 Relay 中使用外部库

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_relay/using_external_lib.html#sphx-glr-download-how-to-work-with-relay-using-external-lib-py) 下载完整的示例代码
:::

**作者**：[Masahiro Masuda](https://github.com/masahi)，[Truman Tian](https://github.com/SiNZeRo)

本文介绍如何将 cuDNN 或 cuBLAS 等外部库与 Relay 一起使用。

Relay 内部用 TVM 来生成 target-specific 的代码。例如，TVM 使用 CUDA 后端为用户提供的网络中的所有层生成 CUDA 内核。有时也可将各个供应商开发的外部库合并到 Relay 中，TVM 有一种机制可以透明地调用这些库——对于 Relay 用户，只需要设置一个适当的 target 字符串。

使用 Relay 的外部库前，用你要用的库构建 TVM。例如，要用 cuDNN，需启用 *cmake/config.cmake* 中的 USE_CUDNN 选项，必要时要指定 cuDNN 包含和库目录。

首先导入 Relay 和 TVM。

``` python
import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm import relay
from tvm.relay import testing
import tvm.testing
```

## 创建一个简单网络

下面创建一个简单网络进行演示，它由 convolution，batch normalization 和 ReLU activation 组成。

``` python
out_channels = 16
batch_size = 1

data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
weight = relay.var("weight")
bn_gamma = relay.var("bn_gamma")
bn_beta = relay.var("bn_beta")
bn_mmean = relay.var("bn_mean")
bn_mvar = relay.var("bn_var")

simple_net = relay.nn.conv2d(
    data=data, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
)
simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
simple_net = relay.nn.relu(simple_net)
simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

data_shape = (batch_size, 3, 224, 224)
net, params = testing.create_workload(simple_net)
```

## 使用 CUDA 后端构建和运行

正常使用 CUDA 后端构建和运行这个网络。设置日志记录级别为 DEBUG，Relay 计算图编译的结果将作为伪代码转储。

``` python
import logging

logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion

target = "cuda"
lib = relay.build_module.build(net, target, params=params)

dev = tvm.device(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = runtime.GraphModule(lib["default"](dev))
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_cuda = out.numpy()
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

生成的伪代码应如下。注意 bias add，batch normalization 和 ReLU activation 是如何融合到卷积核中的。 TVM 从这个表示中生成一个单一的融合内核。

``` python
produce tensor {
  // attr [iter_var(blockIdx.z, , blockIdx.z)] thread_extent = 1
  // attr [compute] storage_scope = "local"
  allocate compute[float32 * 32]
  // attr [pad_temp.shared] storage_scope = "shared"
  allocate pad_temp.shared[float32 * 180]
  // attr [placeholder.shared] storage_scope = "shared"
  allocate placeholder.shared[float32 * 144]
  // attr [iter_var(blockIdx.y, , blockIdx.y)] thread_extent = 28
  // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 14
  // attr [iter_var(threadIdx.z, , threadIdx.z)] thread_extent = 4
  // attr [iter_var(threadIdx.y, , threadIdx.y)] thread_extent = 1
  // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 16
  produce compute {
    compute[0] = 0.000000f
    compute[1] = 0.000000f
    compute[2] = 0.000000f
    compute[3] = 0.000000f
    compute[4] = 0.000000f
    compute[5] = 0.000000f
    compute[6] = 0.000000f
    compute[7] = 0.000000f
    compute[8] = 0.000000f
    compute[9] = 0.000000f
    compute[10] = 0.000000f
    compute[11] = 0.000000f
    compute[12] = 0.000000f
    compute[13] = 0.000000f
    compute[14] = 0.000000f
    compute[15] = 0.000000f
    compute[16] = 0.000000f
    compute[17] = 0.000000f
    compute[18] = 0.000000f
    compute[19] = 0.000000f
    compute[20] = 0.000000f
    compute[21] = 0.000000f
    compute[22] = 0.000000f
    compute[23] = 0.000000f
    compute[24] = 0.000000f
    compute[25] = 0.000000f
    compute[26] = 0.000000f
    compute[27] = 0.000000f
    compute[28] = 0.000000f
    compute[29] = 0.000000f
    compute[30] = 0.000000f
    compute[31] = 0.000000f
    for (rc.outer, 0, 3) {
      produce pad_temp.shared {
        // attr [iter_var(threadIdx.z, , threadIdx.z)] thread_extent = 4
        // attr [iter_var(threadIdx.y, , threadIdx.y)] thread_extent = 1
        // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 16
        if (likely(((threadIdx.z*15) < (60 - threadIdx.x)))) {
          if (likely((threadIdx.x < 15))) {
            pad_temp.shared[(((((threadIdx.z*15) + threadIdx.x)/60)*180) + ((((((threadIdx.z*15) + threadIdx.x)/6) % 10)*18) + ((((threadIdx.z*3) + threadIdx.x)*3) % 18)))] = tvm_if_then_else((((((1 - ((((threadIdx.z*15) + threadIdx.x)/6) % 10)) <= (blockIdx.y*8)) && ((blockIdx.y*8) < (225 - ((((threadIdx.z*15) + threadIdx.x)/6) % 10)))) && ((1 - ((((threadIdx.z*3) + threadIdx.x)*3) % 18)) <= (blockIdx.x*16))) && ((blockIdx.x*16) < (225 - ((((threadIdx.z*3) + threadIdx.x)*3) % 18)))), placeholder[((((((((blockIdx.y*112) + blockIdx.x) + (rc.outer*3136)) + ((((threadIdx.z*15) + threadIdx.x)/60)*9408))*16) + ((((threadIdx.z*3) + threadIdx.x)*3) % 18)) + (((((threadIdx.z*15) + threadIdx.x)/6) % 10)*224)) + -225)], 0.000000f)
            pad_temp.shared[(((((((threadIdx.z*15) + threadIdx.x)*3) + 1)/180)*180) + ((((((((threadIdx.z*15) + threadIdx.x)*3) + 1)/18) % 10)*18) + (((((threadIdx.z*3) + threadIdx.x)*3) + 1) % 18)))] = tvm_if_then_else((((((1 - ((((((threadIdx.z*15) + threadIdx.x)*3) + 1)/18) % 10)) <= (blockIdx.y*8)) && ((blockIdx.y*8) < (225 - ((((((threadIdx.z*15) + threadIdx.x)*3) + 1)/18) % 10)))) && ((1 - (((((threadIdx.z*3) + threadIdx.x)*3) + 1) % 18)) <= (blockIdx.x*16))) && ((blockIdx.x*16) < (225 - (((((threadIdx.z*3) + threadIdx.x)*3) + 1) % 18)))), placeholder[((((((((blockIdx.y*112) + blockIdx.x) + (rc.outer*3136)) + ((((((threadIdx.z*15) + threadIdx.x)*3) + 1)/180)*9408))*16) + (((((threadIdx.z*3) + threadIdx.x)*3) + 1) % 18)) + (((((((threadIdx.z*15) + threadIdx.x)*3) + 1)/18) % 10)*224)) + -225)], 0.000000f)
            pad_temp.shared[(((((((threadIdx.z*15) + threadIdx.x)*3) + 2)/180)*180) + ((((((((threadIdx.z*15) + threadIdx.x)*3) + 2)/18) % 10)*18) + (((((threadIdx.z*3) + threadIdx.x)*3) + 2) % 18)))] = tvm_if_then_else((((((1 - ((((((threadIdx.z*15) + threadIdx.x)*3) + 2)/18) % 10)) <= (blockIdx.y*8)) && ((blockIdx.y*8) < (225 - ((((((threadIdx.z*15) + threadIdx.x)*3) + 2)/18) % 10)))) && ((1 - (((((threadIdx.z*3) + threadIdx.x)*3) + 2) % 18)) <= (blockIdx.x*16))) && ((blockIdx.x*16) < (225 - (((((threadIdx.z*3) + threadIdx.x)*3) + 2) % 18)))), placeholder[((((((((blockIdx.y*112) + blockIdx.x) + (rc.outer*3136)) + ((((((threadIdx.z*15) + threadIdx.x)*3) + 2)/180)*9408))*16) + (((((threadIdx.z*3) + threadIdx.x)*3) + 2) % 18)) + (((((((threadIdx.z*15) + threadIdx.x)*3) + 2)/18) % 10)*224)) + -225)], 0.000000f)
          }
        }
      }
      produce placeholder.shared {
        // attr [iter_var(threadIdx.z, , threadIdx.z)] thread_extent = 4
        // attr [iter_var(threadIdx.y, , threadIdx.y)] thread_extent = 1
        // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 16
        if (likely(((threadIdx.z*4) < (16 - (threadIdx.x/3))))) {
          if (likely(((threadIdx.z*12) < (48 - threadIdx.x)))) {
            if (likely((threadIdx.x < 12))) {
              placeholder.shared[(((((threadIdx.z*4) + (threadIdx.x/3))*3) + (threadIdx.x % 3))*3)] = placeholder[(((((rc.outer + (threadIdx.z*12)) + ((threadIdx.x/3)*3))*3) + (threadIdx.x % 3))*3)]
              placeholder.shared[((((((threadIdx.z*4) + (threadIdx.x/3))*3) + (threadIdx.x % 3))*3) + 1)] = placeholder[((((((rc.outer + (threadIdx.z*12)) + ((threadIdx.x/3)*3))*3) + (threadIdx.x % 3))*3) + 1)]
              placeholder.shared[((((((threadIdx.z*4) + (threadIdx.x/3))*3) + (threadIdx.x % 3))*3) + 2)] = placeholder[((((((rc.outer + (threadIdx.z*12)) + ((threadIdx.x/3)*3))*3) + (threadIdx.x % 3))*3) + 2)]
            }
          }
        }
      }
      compute[0] = (compute[0] + (pad_temp.shared[threadIdx.x]*placeholder.shared[(threadIdx.z*36)]))
      compute[1] = (compute[1] + (pad_temp.shared[(threadIdx.x + 18)]*placeholder.shared[(threadIdx.z*36)]))
      compute[2] = (compute[2] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[(threadIdx.z*36)]))
      compute[3] = (compute[3] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[(threadIdx.z*36)]))
      compute[4] = (compute[4] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[(threadIdx.z*36)]))
      compute[5] = (compute[5] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[(threadIdx.z*36)]))
      compute[6] = (compute[6] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[(threadIdx.z*36)]))
      compute[7] = (compute[7] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[(threadIdx.z*36)]))
      compute[8] = (compute[8] + (pad_temp.shared[threadIdx.x]*placeholder.shared[((threadIdx.z*36) + 9)]))
      compute[9] = (compute[9] + (pad_temp.shared[(threadIdx.x + 18)]*placeholder.shared[((threadIdx.z*36) + 9)]))
      compute[10] = (compute[10] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 9)]))
      compute[11] = (compute[11] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 9)]))
      compute[12] = (compute[12] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 9)]))
      compute[13] = (compute[13] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 9)]))
      compute[14] = (compute[14] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 9)]))
      compute[15] = (compute[15] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 9)]))
      compute[16] = (compute[16] + (pad_temp.shared[threadIdx.x]*placeholder.shared[((threadIdx.z*36) + 18)]))
      compute[17] = (compute[17] + (pad_temp.shared[(threadIdx.x + 18)]*placeholder.shared[((threadIdx.z*36) + 18)]))
      compute[18] = (compute[18] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 18)]))
      compute[19] = (compute[19] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 18)]))
      compute[20] = (compute[20] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 18)]))
      compute[21] = (compute[21] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 18)]))
      compute[22] = (compute[22] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 18)]))
      compute[23] = (compute[23] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 18)]))
      compute[24] = (compute[24] + (pad_temp.shared[threadIdx.x]*placeholder.shared[((threadIdx.z*36) + 27)]))
      compute[25] = (compute[25] + (pad_temp.shared[(threadIdx.x + 18)]*placeholder.shared[((threadIdx.z*36) + 27)]))
      compute[26] = (compute[26] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 27)]))
      compute[27] = (compute[27] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 27)]))
      compute[28] = (compute[28] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 27)]))
      compute[29] = (compute[29] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 27)]))
      compute[30] = (compute[30] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 27)]))
      compute[31] = (compute[31] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 27)]))
      compute[0] = (compute[0] + (pad_temp.shared[(threadIdx.x + 1)]*placeholder.shared[((threadIdx.z*36) + 1)]))
      compute[1] = (compute[1] + (pad_temp.shared[(threadIdx.x + 19)]*placeholder.shared[((threadIdx.z*36) + 1)]))
      compute[2] = (compute[2] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 1)]))
      compute[3] = (compute[3] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 1)]))
      compute[4] = (compute[4] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 1)]))
      compute[5] = (compute[5] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 1)]))
      compute[6] = (compute[6] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 1)]))
      compute[7] = (compute[7] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 1)]))
      compute[8] = (compute[8] + (pad_temp.shared[(threadIdx.x + 1)]*placeholder.shared[((threadIdx.z*36) + 10)]))
      compute[9] = (compute[9] + (pad_temp.shared[(threadIdx.x + 19)]*placeholder.shared[((threadIdx.z*36) + 10)]))
      compute[10] = (compute[10] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 10)]))
      compute[11] = (compute[11] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 10)]))
      compute[12] = (compute[12] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 10)]))
      compute[13] = (compute[13] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 10)]))
      compute[14] = (compute[14] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 10)]))
      compute[15] = (compute[15] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 10)]))
      compute[16] = (compute[16] + (pad_temp.shared[(threadIdx.x + 1)]*placeholder.shared[((threadIdx.z*36) + 19)]))
      compute[17] = (compute[17] + (pad_temp.shared[(threadIdx.x + 19)]*placeholder.shared[((threadIdx.z*36) + 19)]))
      compute[18] = (compute[18] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 19)]))
      compute[19] = (compute[19] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 19)]))
      compute[20] = (compute[20] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 19)]))
      compute[21] = (compute[21] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 19)]))
      compute[22] = (compute[22] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 19)]))
      compute[23] = (compute[23] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 19)]))
      compute[24] = (compute[24] + (pad_temp.shared[(threadIdx.x + 1)]*placeholder.shared[((threadIdx.z*36) + 28)]))
      compute[25] = (compute[25] + (pad_temp.shared[(threadIdx.x + 19)]*placeholder.shared[((threadIdx.z*36) + 28)]))
      compute[26] = (compute[26] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 28)]))
      compute[27] = (compute[27] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 28)]))
      compute[28] = (compute[28] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 28)]))
      compute[29] = (compute[29] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 28)]))
      compute[30] = (compute[30] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 28)]))
      compute[31] = (compute[31] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 28)]))
      compute[0] = (compute[0] + (pad_temp.shared[(threadIdx.x + 2)]*placeholder.shared[((threadIdx.z*36) + 2)]))
      compute[1] = (compute[1] + (pad_temp.shared[(threadIdx.x + 20)]*placeholder.shared[((threadIdx.z*36) + 2)]))
      compute[2] = (compute[2] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 2)]))
      compute[3] = (compute[3] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 2)]))
      compute[4] = (compute[4] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 2)]))
      compute[5] = (compute[5] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 2)]))
      compute[6] = (compute[6] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 2)]))
      compute[7] = (compute[7] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 2)]))
      compute[8] = (compute[8] + (pad_temp.shared[(threadIdx.x + 2)]*placeholder.shared[((threadIdx.z*36) + 11)]))
      compute[9] = (compute[9] + (pad_temp.shared[(threadIdx.x + 20)]*placeholder.shared[((threadIdx.z*36) + 11)]))
      compute[10] = (compute[10] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 11)]))
      compute[11] = (compute[11] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 11)]))
      compute[12] = (compute[12] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 11)]))
      compute[13] = (compute[13] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 11)]))
      compute[14] = (compute[14] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 11)]))
      compute[15] = (compute[15] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 11)]))
      compute[16] = (compute[16] + (pad_temp.shared[(threadIdx.x + 2)]*placeholder.shared[((threadIdx.z*36) + 20)]))
      compute[17] = (compute[17] + (pad_temp.shared[(threadIdx.x + 20)]*placeholder.shared[((threadIdx.z*36) + 20)]))
      compute[18] = (compute[18] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 20)]))
      compute[19] = (compute[19] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 20)]))
      compute[20] = (compute[20] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 20)]))
      compute[21] = (compute[21] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 20)]))
      compute[22] = (compute[22] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 20)]))
      compute[23] = (compute[23] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 20)]))
      compute[24] = (compute[24] + (pad_temp.shared[(threadIdx.x + 2)]*placeholder.shared[((threadIdx.z*36) + 29)]))
      compute[25] = (compute[25] + (pad_temp.shared[(threadIdx.x + 20)]*placeholder.shared[((threadIdx.z*36) + 29)]))
      compute[26] = (compute[26] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 29)]))
      compute[27] = (compute[27] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 29)]))
      compute[28] = (compute[28] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 29)]))
      compute[29] = (compute[29] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 29)]))
      compute[30] = (compute[30] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 29)]))
      compute[31] = (compute[31] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 29)]))
      compute[0] = (compute[0] + (pad_temp.shared[(threadIdx.x + 18)]*placeholder.shared[((threadIdx.z*36) + 3)]))
      compute[1] = (compute[1] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 3)]))
      compute[2] = (compute[2] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 3)]))
      compute[3] = (compute[3] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 3)]))
      compute[4] = (compute[4] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 3)]))
      compute[5] = (compute[5] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 3)]))
      compute[6] = (compute[6] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 3)]))
      compute[7] = (compute[7] + (pad_temp.shared[(threadIdx.x + 144)]*placeholder.shared[((threadIdx.z*36) + 3)]))
      compute[8] = (compute[8] + (pad_temp.shared[(threadIdx.x + 18)]*placeholder.shared[((threadIdx.z*36) + 12)]))
      compute[9] = (compute[9] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 12)]))
      compute[10] = (compute[10] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 12)]))
      compute[11] = (compute[11] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 12)]))
      compute[12] = (compute[12] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 12)]))
      compute[13] = (compute[13] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 12)]))
      compute[14] = (compute[14] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 12)]))
      compute[15] = (compute[15] + (pad_temp.shared[(threadIdx.x + 144)]*placeholder.shared[((threadIdx.z*36) + 12)]))
      compute[16] = (compute[16] + (pad_temp.shared[(threadIdx.x + 18)]*placeholder.shared[((threadIdx.z*36) + 21)]))
      compute[17] = (compute[17] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 21)]))
      compute[18] = (compute[18] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 21)]))
      compute[19] = (compute[19] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 21)]))
      compute[20] = (compute[20] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 21)]))
      compute[21] = (compute[21] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 21)]))
      compute[22] = (compute[22] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 21)]))
      compute[23] = (compute[23] + (pad_temp.shared[(threadIdx.x + 144)]*placeholder.shared[((threadIdx.z*36) + 21)]))
      compute[24] = (compute[24] + (pad_temp.shared[(threadIdx.x + 18)]*placeholder.shared[((threadIdx.z*36) + 30)]))
      compute[25] = (compute[25] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 30)]))
      compute[26] = (compute[26] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 30)]))
      compute[27] = (compute[27] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 30)]))
      compute[28] = (compute[28] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 30)]))
      compute[29] = (compute[29] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 30)]))
      compute[30] = (compute[30] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 30)]))
      compute[31] = (compute[31] + (pad_temp.shared[(threadIdx.x + 144)]*placeholder.shared[((threadIdx.z*36) + 30)]))
      compute[0] = (compute[0] + (pad_temp.shared[(threadIdx.x + 19)]*placeholder.shared[((threadIdx.z*36) + 4)]))
      compute[1] = (compute[1] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 4)]))
      compute[2] = (compute[2] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 4)]))
      compute[3] = (compute[3] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 4)]))
      compute[4] = (compute[4] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 4)]))
      compute[5] = (compute[5] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 4)]))
      compute[6] = (compute[6] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 4)]))
      compute[7] = (compute[7] + (pad_temp.shared[(threadIdx.x + 145)]*placeholder.shared[((threadIdx.z*36) + 4)]))
      compute[8] = (compute[8] + (pad_temp.shared[(threadIdx.x + 19)]*placeholder.shared[((threadIdx.z*36) + 13)]))
      compute[9] = (compute[9] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 13)]))
      compute[10] = (compute[10] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 13)]))
      compute[11] = (compute[11] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 13)]))
      compute[12] = (compute[12] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 13)]))
      compute[13] = (compute[13] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 13)]))
      compute[14] = (compute[14] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 13)]))
      compute[15] = (compute[15] + (pad_temp.shared[(threadIdx.x + 145)]*placeholder.shared[((threadIdx.z*36) + 13)]))
      compute[16] = (compute[16] + (pad_temp.shared[(threadIdx.x + 19)]*placeholder.shared[((threadIdx.z*36) + 22)]))
      compute[17] = (compute[17] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 22)]))
      compute[18] = (compute[18] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 22)]))
      compute[19] = (compute[19] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 22)]))
      compute[20] = (compute[20] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 22)]))
      compute[21] = (compute[21] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 22)]))
      compute[22] = (compute[22] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 22)]))
      compute[23] = (compute[23] + (pad_temp.shared[(threadIdx.x + 145)]*placeholder.shared[((threadIdx.z*36) + 22)]))
      compute[24] = (compute[24] + (pad_temp.shared[(threadIdx.x + 19)]*placeholder.shared[((threadIdx.z*36) + 31)]))
      compute[25] = (compute[25] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 31)]))
      compute[26] = (compute[26] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 31)]))
      compute[27] = (compute[27] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 31)]))
      compute[28] = (compute[28] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 31)]))
      compute[29] = (compute[29] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 31)]))
      compute[30] = (compute[30] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 31)]))
      compute[31] = (compute[31] + (pad_temp.shared[(threadIdx.x + 145)]*placeholder.shared[((threadIdx.z*36) + 31)]))
      compute[0] = (compute[0] + (pad_temp.shared[(threadIdx.x + 20)]*placeholder.shared[((threadIdx.z*36) + 5)]))
      compute[1] = (compute[1] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 5)]))
      compute[2] = (compute[2] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 5)]))
      compute[3] = (compute[3] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 5)]))
      compute[4] = (compute[4] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 5)]))
      compute[5] = (compute[5] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 5)]))
      compute[6] = (compute[6] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 5)]))
      compute[7] = (compute[7] + (pad_temp.shared[(threadIdx.x + 146)]*placeholder.shared[((threadIdx.z*36) + 5)]))
      compute[8] = (compute[8] + (pad_temp.shared[(threadIdx.x + 20)]*placeholder.shared[((threadIdx.z*36) + 14)]))
      compute[9] = (compute[9] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 14)]))
      compute[10] = (compute[10] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 14)]))
      compute[11] = (compute[11] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 14)]))
      compute[12] = (compute[12] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 14)]))
      compute[13] = (compute[13] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 14)]))
      compute[14] = (compute[14] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 14)]))
      compute[15] = (compute[15] + (pad_temp.shared[(threadIdx.x + 146)]*placeholder.shared[((threadIdx.z*36) + 14)]))
      compute[16] = (compute[16] + (pad_temp.shared[(threadIdx.x + 20)]*placeholder.shared[((threadIdx.z*36) + 23)]))
      compute[17] = (compute[17] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 23)]))
      compute[18] = (compute[18] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 23)]))
      compute[19] = (compute[19] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 23)]))
      compute[20] = (compute[20] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 23)]))
      compute[21] = (compute[21] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 23)]))
      compute[22] = (compute[22] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 23)]))
      compute[23] = (compute[23] + (pad_temp.shared[(threadIdx.x + 146)]*placeholder.shared[((threadIdx.z*36) + 23)]))
      compute[24] = (compute[24] + (pad_temp.shared[(threadIdx.x + 20)]*placeholder.shared[((threadIdx.z*36) + 32)]))
      compute[25] = (compute[25] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 32)]))
      compute[26] = (compute[26] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 32)]))
      compute[27] = (compute[27] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 32)]))
      compute[28] = (compute[28] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 32)]))
      compute[29] = (compute[29] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 32)]))
      compute[30] = (compute[30] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 32)]))
      compute[31] = (compute[31] + (pad_temp.shared[(threadIdx.x + 146)]*placeholder.shared[((threadIdx.z*36) + 32)]))
      compute[0] = (compute[0] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 6)]))
      compute[1] = (compute[1] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 6)]))
      compute[2] = (compute[2] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 6)]))
      compute[3] = (compute[3] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 6)]))
      compute[4] = (compute[4] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 6)]))
      compute[5] = (compute[5] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 6)]))
      compute[6] = (compute[6] + (pad_temp.shared[(threadIdx.x + 144)]*placeholder.shared[((threadIdx.z*36) + 6)]))
      compute[7] = (compute[7] + (pad_temp.shared[(threadIdx.x + 162)]*placeholder.shared[((threadIdx.z*36) + 6)]))
      compute[8] = (compute[8] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 15)]))
      compute[9] = (compute[9] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 15)]))
      compute[10] = (compute[10] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 15)]))
      compute[11] = (compute[11] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 15)]))
      compute[12] = (compute[12] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 15)]))
      compute[13] = (compute[13] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 15)]))
      compute[14] = (compute[14] + (pad_temp.shared[(threadIdx.x + 144)]*placeholder.shared[((threadIdx.z*36) + 15)]))
      compute[15] = (compute[15] + (pad_temp.shared[(threadIdx.x + 162)]*placeholder.shared[((threadIdx.z*36) + 15)]))
      compute[16] = (compute[16] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 24)]))
      compute[17] = (compute[17] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 24)]))
      compute[18] = (compute[18] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 24)]))
      compute[19] = (compute[19] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 24)]))
      compute[20] = (compute[20] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 24)]))
      compute[21] = (compute[21] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 24)]))
      compute[22] = (compute[22] + (pad_temp.shared[(threadIdx.x + 144)]*placeholder.shared[((threadIdx.z*36) + 24)]))
      compute[23] = (compute[23] + (pad_temp.shared[(threadIdx.x + 162)]*placeholder.shared[((threadIdx.z*36) + 24)]))
      compute[24] = (compute[24] + (pad_temp.shared[(threadIdx.x + 36)]*placeholder.shared[((threadIdx.z*36) + 33)]))
      compute[25] = (compute[25] + (pad_temp.shared[(threadIdx.x + 54)]*placeholder.shared[((threadIdx.z*36) + 33)]))
      compute[26] = (compute[26] + (pad_temp.shared[(threadIdx.x + 72)]*placeholder.shared[((threadIdx.z*36) + 33)]))
      compute[27] = (compute[27] + (pad_temp.shared[(threadIdx.x + 90)]*placeholder.shared[((threadIdx.z*36) + 33)]))
      compute[28] = (compute[28] + (pad_temp.shared[(threadIdx.x + 108)]*placeholder.shared[((threadIdx.z*36) + 33)]))
      compute[29] = (compute[29] + (pad_temp.shared[(threadIdx.x + 126)]*placeholder.shared[((threadIdx.z*36) + 33)]))
      compute[30] = (compute[30] + (pad_temp.shared[(threadIdx.x + 144)]*placeholder.shared[((threadIdx.z*36) + 33)]))
      compute[31] = (compute[31] + (pad_temp.shared[(threadIdx.x + 162)]*placeholder.shared[((threadIdx.z*36) + 33)]))
      compute[0] = (compute[0] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 7)]))
      compute[1] = (compute[1] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 7)]))
      compute[2] = (compute[2] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 7)]))
      compute[3] = (compute[3] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 7)]))
      compute[4] = (compute[4] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 7)]))
      compute[5] = (compute[5] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 7)]))
      compute[6] = (compute[6] + (pad_temp.shared[(threadIdx.x + 145)]*placeholder.shared[((threadIdx.z*36) + 7)]))
      compute[7] = (compute[7] + (pad_temp.shared[(threadIdx.x + 163)]*placeholder.shared[((threadIdx.z*36) + 7)]))
      compute[8] = (compute[8] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 16)]))
      compute[9] = (compute[9] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 16)]))
      compute[10] = (compute[10] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 16)]))
      compute[11] = (compute[11] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 16)]))
      compute[12] = (compute[12] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 16)]))
      compute[13] = (compute[13] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 16)]))
      compute[14] = (compute[14] + (pad_temp.shared[(threadIdx.x + 145)]*placeholder.shared[((threadIdx.z*36) + 16)]))
      compute[15] = (compute[15] + (pad_temp.shared[(threadIdx.x + 163)]*placeholder.shared[((threadIdx.z*36) + 16)]))
      compute[16] = (compute[16] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 25)]))
      compute[17] = (compute[17] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 25)]))
      compute[18] = (compute[18] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 25)]))
      compute[19] = (compute[19] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 25)]))
      compute[20] = (compute[20] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 25)]))
      compute[21] = (compute[21] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 25)]))
      compute[22] = (compute[22] + (pad_temp.shared[(threadIdx.x + 145)]*placeholder.shared[((threadIdx.z*36) + 25)]))
      compute[23] = (compute[23] + (pad_temp.shared[(threadIdx.x + 163)]*placeholder.shared[((threadIdx.z*36) + 25)]))
      compute[24] = (compute[24] + (pad_temp.shared[(threadIdx.x + 37)]*placeholder.shared[((threadIdx.z*36) + 34)]))
      compute[25] = (compute[25] + (pad_temp.shared[(threadIdx.x + 55)]*placeholder.shared[((threadIdx.z*36) + 34)]))
      compute[26] = (compute[26] + (pad_temp.shared[(threadIdx.x + 73)]*placeholder.shared[((threadIdx.z*36) + 34)]))
      compute[27] = (compute[27] + (pad_temp.shared[(threadIdx.x + 91)]*placeholder.shared[((threadIdx.z*36) + 34)]))
      compute[28] = (compute[28] + (pad_temp.shared[(threadIdx.x + 109)]*placeholder.shared[((threadIdx.z*36) + 34)]))
      compute[29] = (compute[29] + (pad_temp.shared[(threadIdx.x + 127)]*placeholder.shared[((threadIdx.z*36) + 34)]))
      compute[30] = (compute[30] + (pad_temp.shared[(threadIdx.x + 145)]*placeholder.shared[((threadIdx.z*36) + 34)]))
      compute[31] = (compute[31] + (pad_temp.shared[(threadIdx.x + 163)]*placeholder.shared[((threadIdx.z*36) + 34)]))
      compute[0] = (compute[0] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 8)]))
      compute[1] = (compute[1] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 8)]))
      compute[2] = (compute[2] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 8)]))
      compute[3] = (compute[3] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 8)]))
      compute[4] = (compute[4] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 8)]))
      compute[5] = (compute[5] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 8)]))
      compute[6] = (compute[6] + (pad_temp.shared[(threadIdx.x + 146)]*placeholder.shared[((threadIdx.z*36) + 8)]))
      compute[7] = (compute[7] + (pad_temp.shared[(threadIdx.x + 164)]*placeholder.shared[((threadIdx.z*36) + 8)]))
      compute[8] = (compute[8] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 17)]))
      compute[9] = (compute[9] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 17)]))
      compute[10] = (compute[10] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 17)]))
      compute[11] = (compute[11] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 17)]))
      compute[12] = (compute[12] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 17)]))
      compute[13] = (compute[13] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 17)]))
      compute[14] = (compute[14] + (pad_temp.shared[(threadIdx.x + 146)]*placeholder.shared[((threadIdx.z*36) + 17)]))
      compute[15] = (compute[15] + (pad_temp.shared[(threadIdx.x + 164)]*placeholder.shared[((threadIdx.z*36) + 17)]))
      compute[16] = (compute[16] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 26)]))
      compute[17] = (compute[17] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 26)]))
      compute[18] = (compute[18] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 26)]))
      compute[19] = (compute[19] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 26)]))
      compute[20] = (compute[20] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 26)]))
      compute[21] = (compute[21] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 26)]))
      compute[22] = (compute[22] + (pad_temp.shared[(threadIdx.x + 146)]*placeholder.shared[((threadIdx.z*36) + 26)]))
      compute[23] = (compute[23] + (pad_temp.shared[(threadIdx.x + 164)]*placeholder.shared[((threadIdx.z*36) + 26)]))
      compute[24] = (compute[24] + (pad_temp.shared[(threadIdx.x + 38)]*placeholder.shared[((threadIdx.z*36) + 35)]))
      compute[25] = (compute[25] + (pad_temp.shared[(threadIdx.x + 56)]*placeholder.shared[((threadIdx.z*36) + 35)]))
      compute[26] = (compute[26] + (pad_temp.shared[(threadIdx.x + 74)]*placeholder.shared[((threadIdx.z*36) + 35)]))
      compute[27] = (compute[27] + (pad_temp.shared[(threadIdx.x + 92)]*placeholder.shared[((threadIdx.z*36) + 35)]))
      compute[28] = (compute[28] + (pad_temp.shared[(threadIdx.x + 110)]*placeholder.shared[((threadIdx.z*36) + 35)]))
      compute[29] = (compute[29] + (pad_temp.shared[(threadIdx.x + 128)]*placeholder.shared[((threadIdx.z*36) + 35)]))
      compute[30] = (compute[30] + (pad_temp.shared[(threadIdx.x + 146)]*placeholder.shared[((threadIdx.z*36) + 35)]))
      compute[31] = (compute[31] + (pad_temp.shared[(threadIdx.x + 164)]*placeholder.shared[((threadIdx.z*36) + 35)]))
    }
  }
  tensor[(((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x)] = max(((compute[0]*placeholder[(threadIdx.z*4)]) + placeholder[(threadIdx.z*4)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 224)] = max(((compute[1]*placeholder[(threadIdx.z*4)]) + placeholder[(threadIdx.z*4)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 448)] = max(((compute[2]*placeholder[(threadIdx.z*4)]) + placeholder[(threadIdx.z*4)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 672)] = max(((compute[3]*placeholder[(threadIdx.z*4)]) + placeholder[(threadIdx.z*4)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 896)] = max(((compute[4]*placeholder[(threadIdx.z*4)]) + placeholder[(threadIdx.z*4)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 1120)] = max(((compute[5]*placeholder[(threadIdx.z*4)]) + placeholder[(threadIdx.z*4)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 1344)] = max(((compute[6]*placeholder[(threadIdx.z*4)]) + placeholder[(threadIdx.z*4)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 1568)] = max(((compute[7]*placeholder[(threadIdx.z*4)]) + placeholder[(threadIdx.z*4)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 50176)] = max(((compute[8]*placeholder[((threadIdx.z*4) + 1)]) + placeholder[((threadIdx.z*4) + 1)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 50400)] = max(((compute[9]*placeholder[((threadIdx.z*4) + 1)]) + placeholder[((threadIdx.z*4) + 1)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 50624)] = max(((compute[10]*placeholder[((threadIdx.z*4) + 1)]) + placeholder[((threadIdx.z*4) + 1)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 50848)] = max(((compute[11]*placeholder[((threadIdx.z*4) + 1)]) + placeholder[((threadIdx.z*4) + 1)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 51072)] = max(((compute[12]*placeholder[((threadIdx.z*4) + 1)]) + placeholder[((threadIdx.z*4) + 1)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 51296)] = max(((compute[13]*placeholder[((threadIdx.z*4) + 1)]) + placeholder[((threadIdx.z*4) + 1)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 51520)] = max(((compute[14]*placeholder[((threadIdx.z*4) + 1)]) + placeholder[((threadIdx.z*4) + 1)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 51744)] = max(((compute[15]*placeholder[((threadIdx.z*4) + 1)]) + placeholder[((threadIdx.z*4) + 1)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 100352)] = max(((compute[16]*placeholder[((threadIdx.z*4) + 2)]) + placeholder[((threadIdx.z*4) + 2)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 100576)] = max(((compute[17]*placeholder[((threadIdx.z*4) + 2)]) + placeholder[((threadIdx.z*4) + 2)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 100800)] = max(((compute[18]*placeholder[((threadIdx.z*4) + 2)]) + placeholder[((threadIdx.z*4) + 2)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 101024)] = max(((compute[19]*placeholder[((threadIdx.z*4) + 2)]) + placeholder[((threadIdx.z*4) + 2)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 101248)] = max(((compute[20]*placeholder[((threadIdx.z*4) + 2)]) + placeholder[((threadIdx.z*4) + 2)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 101472)] = max(((compute[21]*placeholder[((threadIdx.z*4) + 2)]) + placeholder[((threadIdx.z*4) + 2)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 101696)] = max(((compute[22]*placeholder[((threadIdx.z*4) + 2)]) + placeholder[((threadIdx.z*4) + 2)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 101920)] = max(((compute[23]*placeholder[((threadIdx.z*4) + 2)]) + placeholder[((threadIdx.z*4) + 2)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 150528)] = max(((compute[24]*placeholder[((threadIdx.z*4) + 3)]) + placeholder[((threadIdx.z*4) + 3)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 150752)] = max(((compute[25]*placeholder[((threadIdx.z*4) + 3)]) + placeholder[((threadIdx.z*4) + 3)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 150976)] = max(((compute[26]*placeholder[((threadIdx.z*4) + 3)]) + placeholder[((threadIdx.z*4) + 3)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 151200)] = max(((compute[27]*placeholder[((threadIdx.z*4) + 3)]) + placeholder[((threadIdx.z*4) + 3)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 151424)] = max(((compute[28]*placeholder[((threadIdx.z*4) + 3)]) + placeholder[((threadIdx.z*4) + 3)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 151648)] = max(((compute[29]*placeholder[((threadIdx.z*4) + 3)]) + placeholder[((threadIdx.z*4) + 3)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 151872)] = max(((compute[30]*placeholder[((threadIdx.z*4) + 3)]) + placeholder[((threadIdx.z*4) + 3)]), 0.000000f)
  tensor[((((((blockIdx.y*112) + blockIdx.x) + (threadIdx.z*12544))*16) + threadIdx.x) + 152096)] = max(((compute[31]*placeholder[((threadIdx.z*4) + 3)]) + placeholder[((threadIdx.z*4) + 3)]), 0.000000f)
}
```

## 将 cuDNN 用于卷积层

将选项 "-libs=cudnn" 附加到 target 字符串，从而用 cuDNN 将卷积核替换为 cuDNN。

``` python
net, params = testing.create_workload(simple_net)
target = "cuda -libs=cudnn"  # use cudnn for convolution
lib = relay.build_module.build(net, target, params=params)

dev = tvm.device(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = runtime.GraphModule(lib["default"](dev))
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_cudnn = out.numpy()
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

注意，若用 cuDNN，Relay 无法将卷积与其后面的层融合。因为层融合发生在 TVM internal representation (IR) 级别。 Relay 将外部库视为黑盒，因此无法将它们与 TVM IR 融合。

下面的伪代码显示了 cuDNN 卷积 + bias add + batch norm + ReLU 变成了两个计算阶段，一个用于 cuDNN 调用，另一个用于其余操作。

``` python
// attr [y] storage_scope = "global"
allocate y[float32 * 802816]
produce y {
  // attr [0] extern_scope = 0
  tvm_call_packed("tvm.contrib.cudnn.conv2d.forward", 1, 0, 1, 1, 1, 1, 1, 1, 1, tvm_stack_make_array(placeholder, tvm_stack_make_shape(1, 3, 224, 224), 0, 4, 0.000000f, 0), tvm_stack_make_array(placeholder, tvm_stack_make_shape(16, 3, 3, 3), 0, 4, 0.000000f, 0), tvm_stack_make_array(y, tvm_stack_make_shape(1, 16, 224, 224), 0, 4, 0.000000f, 0))
}
produce tensor {
  // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 256
  // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 512
  for (ax0.ax1.fused.ax2.fused.ax3.fused.outer, 0, 7) {
    if (likely(((blockIdx.x*512) < ((802816 - (ax0.ax1.fused.ax2.fused.ax3.fused.outer*131072)) - threadIdx.x)))) {
      tensor[(((((((blockIdx.x*512) + threadIdx.x) + (ax0.ax1.fused.ax2.fused.ax3.fused.outer*131072))/802816)*802816) + (((((((blockIdx.x*512) + threadIdx.x) + (ax0.ax1.fused.ax2.fused.ax3.fused.outer*131072))/224) % 224)*224) + ((((blockIdx.x*64) + threadIdx.x) + (ax0.ax1.fused.ax2.fused.ax3.fused.outer*32)) % 224))) + ((((((blockIdx.x*512) + threadIdx.x) + (ax0.ax1.fused.ax2.fused.ax3.fused.outer*131072))/50176) % 16)*50176))] = max(((y[(((((((blockIdx.x*512) + threadIdx.x) + (ax0.ax1.fused.ax2.fused.ax3.fused.outer*131072))/802816)*802816) + (((((((blockIdx.x*512) + threadIdx.x) + (ax0.ax1.fused.ax2.fused.ax3.fused.outer*131072))/224) % 224)*224) + ((((blockIdx.x*64) + threadIdx.x) + (ax0.ax1.fused.ax2.fused.ax3.fused.outer*32)) % 224))) + ((((((blockIdx.x*512) + threadIdx.x) + (ax0.ax1.fused.ax2.fused.ax3.fused.outer*131072))/50176) % 16)*50176))]*placeholder[(((((blockIdx.x*512) + threadIdx.x) + (ax0.ax1.fused.ax2.fused.ax3.fused.outer*131072))/50176) % 16)]) + placeholder[(((((blockIdx.x*512) + threadIdx.x) + (ax0.ax1.fused.ax2.fused.ax3.fused.outer*131072))/50176) % 16)]), 0.000000f)
    }
  }
}
```

## 验证结果

检查两次运行的结果是否匹配。

``` python
tvm.testing.assert_allclose(out_cuda, out_cudnn, rtol=1e-5)
```

## 结论

本教程介绍了 cuDNN 与 Relay 的使用，此外还支持 cuBLAS。若启用了 cuBLAS，它将在全连接层 (relay.dense) 内使用。若要用 cuBLAS，请将 target 字符串设置为 "cuda -libs=cublas"。也可以将 cuDNN 和 cuBLAS 与 "cuda -libs=cudnn,cublas" 一起使用。

对于 ROCm 后端，支持 MIOpen 和 rocBLAS。将 target 设置为 "rocm -libs=miopen,rocblas" 以启用它们。

使用外部库的注意事项：

首先，使用外部库可能会限制 TVM 和 Relay 的使用。例如，MIOpen 目前只支持 NCHW 布局和 fp32 数据类型，因此不能在 TVM 中使用其他布局或数据类型。

其次，外部库限制了计算图编译期间算子融合的可能性，如上所示。TVM 和 Relay 旨在通过联合算子级别和计算图级别优化，在各种硬件上实现最佳性能。为了实现这个目标，应该继续为 TVM 和 Relay 开发更好的优化，同时在必要时使用外部库回退到现有实现。

[下载 Python 源代码：using_external_lib.py](https://tvm.apache.org/docs/_downloads/d8509b0a8e7db9031303c1a1f6fd1e70/using_external_lib.py)

[下载 Jupyter Notebook：using_external_lib.ipynb](https://tvm.apache.org/docs/_downloads/edc9d28c4fbc249e2e7b78002af63b84/using_external_lib.ipynb)