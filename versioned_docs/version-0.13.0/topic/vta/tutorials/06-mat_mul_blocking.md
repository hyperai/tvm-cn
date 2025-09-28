---
title: 矩阵乘法分块
---

# 矩阵乘法分块

:::note
单击 [此处](https://tvm.apache.org/docs/topic/vta/tutorials/optimize/matrix_multiply_opt.html#sphx-glr-download-topic-vta-tutorials-optimize-matrix-multiply-opt-py) 下载完整的示例代码
:::

**作者**：[Thierry Moreau](https://homes.cs.washington.edu/\~moreau/)

本教程概述了如何用 TVM 在 VTA 设计上有效地映射矩阵乘法。推荐先学习 [简单矩阵乘法](mat_mul) 教程。

本教程演示 TVM 调度优化，将大型神经网络算子分解为更小的块，使得可以在有限的硬件加速器资源内实现计算。

## RPC 设置

首先对 Pynq 的 FPGA 进行编程，并构建其 RPC runtime。

``` python
from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import vta
import numpy as np
from tvm import rpc
from tvm.contrib import utils
from vta.testing import simulator

# 从 3rdparty/vta-hw/config/vta_config.json 文件加载 VTA 参数
env = vta.get_env()

# 从 OS 环境中读取 Pynq RPC 主机 IP 地址和端口号
host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
port = int(os.environ.get("VTA_RPC_PORT", "9091"))

# 在 Pynq 上配置比特流和 runtime 系统，匹配 vta_config.json 文件指定的 VTA 配置。
if env.TARGET == "pynq":
    # 确保 TVM 是用 RPC=1 编译的
    assert tvm.runtime.enabled("rpc")
    remote = rpc.connect(host, port)

    # 重新配置 JIT runtime
    vta.reconfig_runtime(remote)

    # 用预编译的 VTA 比特流对 FPGA 进行编程。
    # 可以通过传递比特流文件的路径而非 None，用自定义比特流对 FPGA 进行编程。
    vta.program_fpga(remote, bitstream=None)

# 在模拟模式下，本地托管 RPC 服务器。
elif env.TARGET in ["sim", "tsim"]:
    remote = rpc.LocalSession()
```

## 计算声明

第一步，描述矩阵乘法计算。

将矩阵乘法定义为全连接层中可以找到的计算，由其 batch size、输入通道和输出通道定义。它们必须是 VTA 张量 shape 的整数倍：分别为 `BATCH`、`BLOCK_IN` 和 `BLOCK_OUT`。

在矩阵乘法中添加了额外的算子，这些算子对输出进行移位和裁剪，从而模拟定点卷积，然后进行校正线性激活。下面描述全连接层的 TVM 数据流图：

![/img/docs/uwsampl/web-data/main/vta/tutorial/fc_dataflow.png](/img/docs/uwsampl/web-data/main/vta/tutorial/fc_dataflow.png)

由于这种计算太大，无法一次全部放入 VTA 的芯片缓冲区。因此，在调度阶段，依靠计算分块策略将计算分解为可管理的块。

``` python
# 全连接层维度：1024 x 1024
batch_size = 1
in_channels = 1024
out_channels = 1024
assert batch_size % env.BATCH == 0
assert in_channels % env.BLOCK_IN == 0
assert out_channels % env.BLOCK_OUT == 0

# 推导平铺的输入张量 shape
data_shape = (batch_size // env.BATCH, in_channels // env.BLOCK_IN, env.BATCH, env.BLOCK_IN)
weight_shape = (
    out_channels // env.BLOCK_OUT,
    in_channels // env.BLOCK_IN,
    env.BLOCK_OUT,
    env.BLOCK_IN,
)
output_shape = (batch_size // env.BATCH, out_channels // env.BLOCK_OUT, env.BATCH, env.BLOCK_OUT)
num_ops = in_channels * out_channels * batch_size * 2

# Reduction 轴
ic = te.reduce_axis((0, in_channels // env.BLOCK_IN), name="ic")
ic_tns = te.reduce_axis((0, env.BLOCK_IN), name="ic_tns")

# 输入占位符张量
data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
weight = te.placeholder(weight_shape, name="weight", dtype=env.wgt_dtype)

# 复制缓冲区
data_buf = te.compute(data_shape, lambda *i: data(*i), "data_buf")
weight_buf = te.compute(weight_shape, lambda *i: weight(*i), "weight_buf")

# 声明矩阵乘法计算
res_gemm = te.compute(
    output_shape,
    lambda bo, co, bi, ci: te.sum(
        data_buf[bo, ic, bi, ic_tns].astype(env.acc_dtype)
        * weight_buf[co, ic, ci, ic_tns].astype(env.acc_dtype),
        axis=[ic, ic_tns],
    ),
    name="res_gem",
)

# 为定点归一化添加移位阶段
res_shr = te.compute(output_shape, lambda *i: res_gemm(*i) >> env.INP_WIDTH, name="res_shr")

# 在（0，输入最大值）之间应用裁剪
inp_max = (1 << (env.INP_WIDTH - 1)) - 1
res_max = te.compute(output_shape, lambda *i: tvm.te.max(res_shr(*i), 0), "res_max")
res_min = te.compute(output_shape, lambda *i: tvm.te.min(res_max(*i), inp_max), "res_min")

# 返回结果前，对输入数据类型进行类型转换
res = te.compute(output_shape, lambda *i: res_min(*i).astype(env.inp_dtype), name="res")
```

## 调度计算

下面将研究用有效方式将矩阵乘法映射到 VTA 所需的一组调度转换。包括：

* 计算分块
* 降级到 VTA 硬件内联函数

``` python
# 创建 TVM schedule
s = te.create_schedule(res.op)
# 查看默认的 TVM schedule
print(tvm.lower(s, [data, weight, res], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(data_1: handle, weight_1: handle, res_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {data: Buffer(data_2: Pointer(int8), int8, [1024], []),
             weight: Buffer(weight_2: Pointer(int8), int8, [1048576], []),
             res: Buffer(res_2: Pointer(int8), int8, [1024], [])}
  buffer_map = {data_1: data, weight_1: weight, res_1: res}
  preflattened_buffer_map = {data_1: data_3: Buffer(data_2, int8, [1, 64, 1, 16], []), weight_1: weight_3: Buffer(weight_2, int8, [64, 64, 16, 16], []), res_1: res_3: Buffer(res_2, int8, [1, 64, 1, 16], [])} {
  allocate(data_buf: Pointer(global int8), int8, [1024]), storage_scope = global;
  allocate(weight_buf: Pointer(global int8), int8, [1048576]), storage_scope = global;
  allocate(res_gem: Pointer(global int32), int32, [1024]), storage_scope = global {
    for (i1: int32, 0, 64) {
      for (i3: int32, 0, 16) {
        let cse_var_1: int32 = ((i1*16) + i3)
        data_buf_1: Buffer(data_buf, int8, [1024], [])[cse_var_1] = data[cse_var_1]
      }
    }
    for (i0: int32, 0, 64) {
      for (i1_1: int32, 0, 64) {
        for (i2: int32, 0, 16) {
          for (i3_1: int32, 0, 16) {
            let cse_var_2: int32 = ((((i0*16384) + (i1_1*256)) + (i2*16)) + i3_1)
            weight_buf_1: Buffer(weight_buf, int8, [1048576], [])[cse_var_2] = weight[cse_var_2]
          }
        }
      }
    }
    for (co: int32, 0, 64) {
      for (ci: int32, 0, 16) {
        res_gem_1: Buffer(res_gem, int32, [1024], [])[((co*16) + ci)] = 0
        for (ic: int32, 0, 64) {
          for (ic_tns: int32, 0, 16) {
            let cse_var_3: int32 = ((co*16) + ci)
            res_gem_1[cse_var_3] = (res_gem_1[cse_var_3] + (cast(int32, data_buf_1[((ic*16) + ic_tns)])*cast(int32, weight_buf_1[((((co*16384) + (ic*256)) + (ci*16)) + ic_tns)])))
          }
        }
      }
    }
    for (i1_2: int32, 0, 64) {
      for (i3_2: int32, 0, 16) {
        let cse_var_4: int32 = ((i1_2*16) + i3_2)
        res_gem_2: Buffer(res_gem, int32, [1024], [])[cse_var_4] = @tir.shift_right(res_gem_1[cse_var_4], 8, dtype=int32)
      }
    }
    for (i1_3: int32, 0, 64) {
      for (i3_3: int32, 0, 16) {
        let cse_var_5: int32 = ((i1_3*16) + i3_3)
        res_gem_3: Buffer(res_gem, int32, [1024], [])[cse_var_5] = max(res_gem_2[cse_var_5], 0)
      }
    }
    for (i1_4: int32, 0, 64) {
      for (i3_4: int32, 0, 16) {
        let cse_var_6: int32 = ((i1_4*16) + i3_4)
        res_gem_4: Buffer(res_gem, int32, [1024], [])[cse_var_6] = min(res_gem_3[cse_var_6], 127)
      }
    }
    for (i1_5: int32, 0, 64) {
      for (i3_5: int32, 0, 16) {
        let cse_var_7: int32 = ((i1_5*16) + i3_5)
        res[cse_var_7] = cast(int8, res_gem_4[cse_var_7])
      }
    }
  }
}
```

### 对计算分块

默认情况下，2D 卷积对于激活或内核权重来说太大，无法一次性同时装入 VTA 的芯片缓冲区。将 (1, 1024) x (1024, 1024) 矩阵乘法分块为更小的 (1, 256) x (256, 256) 矩阵乘法，使得中间张量可以适合加速器的芯片上 SRAM。这种方法类似于为提高缓存命中率，应用于 CPU 和 GPU 的分块技术。

沿每个轴执行分块（由于正在执行单个 batch 推理，batch 轴未被处理）。还将最里面的张量轴保持原样，使得 TVM 模式匹配张量。下图展示了计算调度上的分块结果：

![/img/docs/uwsampl/web-data/main/vta/tutorial/blocking.png](/img/docs/uwsampl/web-data/main/vta/tutorial/blocking.png)

:::note
循环拆分和重新排序后的代码等价于以下伪代码。由于以下示例只执行单个 batch 推理，因此将忽略 batch 轴：

``` python
for (int oc_out = 0; oc_out < 4; ++oc_out) {
  // Initialization loop
  // 初始化循环
  for (int oc_inn = 0; oc_inn < 16; ++oc_inn) {
   for (int oc_tns = 0; oc_tns < 16; ++oc_tns) {
    int j = (oc_out * 16 + oc_inn) * 16 + oc_tns;
    C[0][j] = 0;
   }
  }
  for (int ic_out = 0; ic_out < 4; ++ic_out) {
   // Block loop
   // 块循环
   for (int oc_inn = 0; oc_inn < 16; ++oc_inn) {
    for (int ic_inn = 0; ic_inn < 16; ++ic_inn) {
     // Tensorization loop
     // 张量循环
     for (int oc_tns = 0; oc_tns < 16; ++oc_tns) {
      for (int ic_tns = 0; ic_tns < 16; ++ic_tns) {
       int i = (ic_out * 16 + ic_inn) * 16 + ic_tns;
       int j = (oc_out * 16 + oc_inn) * 16 + oc_tns;
       C[0][i] = C[0][i] + A[0][i] * B[j][i];
      }
     }
    }
   }
  }
 }
}
```
:::

``` python
# 定义平铺大小（用 VTA 张量 shape 大小的倍数表示）
b_block = 1 // env.BATCH
i_block = 256 // env.BLOCK_IN
o_block = 256 // env.BLOCK_OUT

# 沿空间和输出通道维度平铺输出张量
# （因为默认进行单个 batch 推理，沿 batch 维度的拆分没有效果）
b, oc, b_tns, oc_tns = s[res].op.axis
b_out, b_inn = s[res].split(b, b_block)
oc_out, oc_inn = s[res].split(oc, o_block)
s[res].reorder(b_out, oc_out, b_inn, oc_inn)

# 将中间计算移动到每个输出计算块中
s[res_gemm].compute_at(s[res], oc_out)
s[res_shr].compute_at(s[res], oc_out)
s[res_max].compute_at(s[res], oc_out)
s[res_min].compute_at(s[res], oc_out)

# 沿 reduction 轴（输入通道）应用额外的循环分割
b_inn, oc_inn, b_tns, oc_tns = s[res_gemm].op.axis
ic_out, ic_inn = s[res_gemm].split(ic, i_block)

# 对轴重新排序。将 ic_out 轴移出卷积循环，沿 reduction 轴阻塞。
s[res_gemm].reorder(ic_out, b_inn, oc_inn, ic_inn, b_tns, oc_tns, ic_tns)

# 查看阻塞后的当前 TVM schedule
print(tvm.lower(s, [data, weight, res], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(data_1: handle, weight_1: handle, res_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {data: Buffer(data_2: Pointer(int8), int8, [1024], []),
             weight: Buffer(weight_2: Pointer(int8), int8, [1048576], []),
             res: Buffer(res_2: Pointer(int8), int8, [1024], [])}
  buffer_map = {data_1: data, weight_1: weight, res_1: res}
  preflattened_buffer_map = {data_1: data_3: Buffer(data_2, int8, [1, 64, 1, 16], []), weight_1: weight_3: Buffer(weight_2, int8, [64, 64, 16, 16], []), res_1: res_3: Buffer(res_2, int8, [1, 64, 1, 16], [])} {
  allocate(data_buf: Pointer(global int8), int8, [1024]), storage_scope = global;
  allocate(weight_buf: Pointer(global int8), int8, [1048576]), storage_scope = global;
  allocate(res_gem: Pointer(global int32), int32, [256]), storage_scope = global {
    for (i1: int32, 0, 64) {
      for (i3: int32, 0, 16) {
        let cse_var_1: int32 = ((i1*16) + i3)
        data_buf_1: Buffer(data_buf, int8, [1024], [])[cse_var_1] = data[cse_var_1]
      }
    }
    for (i0: int32, 0, 64) {
      for (i1_1: int32, 0, 64) {
        for (i2: int32, 0, 16) {
          for (i3_1: int32, 0, 16) {
            let cse_var_2: int32 = ((((i0*16384) + (i1_1*256)) + (i2*16)) + i3_1)
            weight_buf_1: Buffer(weight_buf, int8, [1048576], [])[cse_var_2] = weight[cse_var_2]
          }
        }
      }
    }
    for (i1.outer: int32, 0, 4) {
      for (co.init: int32, 0, 16) {
        for (ci.init: int32, 0, 16) {
          res_gem_1: Buffer(res_gem, int32, [256], [])[((co.init*16) + ci.init)] = 0
        }
      }
      for (ic.outer: int32, 0, 4) {
        for (co: int32, 0, 16) {
          for (ic.inner: int32, 0, 16) {
            for (ci: int32, 0, 16) {
              for (ic_tns: int32, 0, 16) {
                let cse_var_3: int32 = ((co*16) + ci)
                res_gem_1[cse_var_3] = (res_gem_1[cse_var_3] + (cast(int32, data_buf_1[(((ic.outer*256) + (ic.inner*16)) + ic_tns)])*cast(int32, weight_buf_1[((((((i1.outer*262144) + (co*16384)) + (ic.outer*4096)) + (ic.inner*256)) + (ci*16)) + ic_tns)])))
              }
            }
          }
        }
      }
      for (i1_2: int32, 0, 16) {
        for (i3_2: int32, 0, 16) {
          let cse_var_4: int32 = ((i1_2*16) + i3_2)
          res_gem_2: Buffer(res_gem, int32, [256], [])[cse_var_4] = @tir.shift_right(res_gem_1[cse_var_4], 8, dtype=int32)
        }
      }
      for (i1_3: int32, 0, 16) {
        for (i3_3: int32, 0, 16) {
          let cse_var_5: int32 = ((i1_3*16) + i3_3)
          res_gem_3: Buffer(res_gem, int32, [256], [])[cse_var_5] = max(res_gem_2[cse_var_5], 0)
        }
      }
      for (i1_4: int32, 0, 16) {
        for (i3_4: int32, 0, 16) {
          let cse_var_6: int32 = ((i1_4*16) + i3_4)
          res_gem_4: Buffer(res_gem, int32, [256], [])[cse_var_6] = min(res_gem_3[cse_var_6], 127)
        }
      }
      for (i1.inner: int32, 0, 16) {
        for (i3_5: int32, 0, 16) {
          let cse_var_7: int32 = (i1.inner*16)
          res[(((i1.outer*256) + cse_var_7) + i3_5)] = cast(int8, res_gem_4[(cse_var_7 + i3_5)])
        }
      }
    }
  }
}
```

### 将拷贝降级到 DMA 传输

接下来，将缓冲区范围设置为相应的芯片 VTA SRAM 缓冲区。将加载循环移动到 2D 卷积计算循环中，暂存内存加载，使其适合芯片上 SRAM 缓冲区。最后，用 DMA 拷贝编译指示来注释加载/存储循环外轴，从而在 VTA 上执行大容量内存传输。

``` python
# 设置 SRAM 缓冲区的范围
s[data_buf].set_scope(env.inp_scope)
s[weight_buf].set_scope(env.wgt_scope)
s[res_gemm].set_scope(env.acc_scope)
s[res_shr].set_scope(env.acc_scope)
s[res_min].set_scope(env.acc_scope)
s[res_max].set_scope(env.acc_scope)

# 块数据和权重缓存读取
s[data_buf].compute_at(s[res_gemm], ic_out)
s[weight_buf].compute_at(s[res_gemm], ic_out)

# 用 DMA 拷贝编译指示操作 DRAM->SRAM
s[data_buf].pragma(s[data_buf].op.axis[0], env.dma_copy)
s[weight_buf].pragma(s[weight_buf].op.axis[0], env.dma_copy)

# 在 SRAM->DRAM 操作上，使用 DMA 拷贝编译指示（这意味着这些拷贝应沿 b_inn 或结果轴 2 执行）
s[res].pragma(s[res].op.axis[2], env.dma_copy)
```

### 将计算降级到 VTA 计算内联函数

最后一个阶段是降级计算循环到 VTA 硬件内联函数，这是通过将 2D 卷积映射到张量内联函数，并将移位和裁剪计算映射到向量 ALU 实现的。

``` python
# 在 batch 张量平铺轴上应用张量化
s[res_gemm].tensorize(b_tns, env.gemm)

# 在移位和裁剪操作上添加 ALU 编译指示
s[res_shr].pragma(s[res_shr].op.axis[0], env.alu)
s[res_min].pragma(s[res_min].op.axis[0], env.alu)
s[res_max].pragma(s[res_max].op.axis[0], env.alu)

# 将内存负载/存储降级为 DMA 拷贝内联函数，并将计算降级为 VTA 计算内联函数后，查看最终降低的 TVM schedule。
print(vta.lower(s, [data, weight, res], simple_mode=True))
```

输出结果：

``` bash
@main = primfn(data_1: handle, weight_1: handle, res_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {data: Buffer(data_2: Pointer(int8), int8, [1024], []),
             weight: Buffer(weight_2: Pointer(int8), int8, [1048576], []),
             res: Buffer(res_2: Pointer(int8), int8, [1024], [])}
  buffer_map = {data_1: data, weight_1: weight, res_1: res}
  preflattened_buffer_map = {data_1: data_3: Buffer(data_2, int8, [1, 64, 1, 16], []), weight_1: weight_3: Buffer(weight_2, int8, [64, 64, 16, 16], []), res_1: res_3: Buffer(res_2, int8, [1, 64, 1, 16], [])} {
  @tir.vta.coproc_dep_push(3, 2, dtype=int32)
  for (i1.outer: int32, 0, 4) {
    attr [IterVar(vta: int32, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 2 {
      @tir.vta.coproc_dep_pop(3, 2, dtype=int32)
      attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushGEMMOp" {
        @tir.call_extern("VTAUopLoopBegin", 16, 1, 0, 0, dtype=int32)
        @tir.vta.uop_push(0, 1, 0, 0, 0, 0, 0, 0, dtype=int32)
        @tir.call_extern("VTAUopLoopEnd", dtype=int32)
      }
      @tir.vta.coproc_dep_push(2, 1, dtype=int32)
    }
    for (ic.outer: int32, 0, 4) {
      let cse_var_1: int32 = (ic.outer*16)
       {
        attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 1 {
          @tir.vta.coproc_dep_pop(2, 1, dtype=int32)
          @tir.call_extern("VTALoadBuffer2D", @tir.tvm_thread_context(@tir.vta.command_handle(, dtype=handle), dtype=handle), data_2, cse_var_1, 16, 1, 16, 0, 0, 0, 0, 0, 2, dtype=int32)
          @tir.call_extern("VTALoadBuffer2D", @tir.tvm_thread_context(@tir.vta.command_handle(, dtype=handle), dtype=handle), weight_2, ((i1.outer*1024) + cse_var_1), 16, 16, 64, 0, 0, 0, 0, 0, 1, dtype=int32)
          @tir.vta.coproc_dep_push(1, 2, dtype=int32)
        }
        attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 2 {
          @tir.vta.coproc_dep_pop(1, 2, dtype=int32)
          attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushGEMMOp" {
            @tir.call_extern("VTAUopLoopBegin", 16, 1, 0, 16, dtype=int32)
            @tir.call_extern("VTAUopLoopBegin", 16, 0, 1, 1, dtype=int32)
            @tir.vta.uop_push(0, 0, 0, 0, 0, 0, 0, 0, dtype=int32)
            @tir.call_extern("VTAUopLoopEnd", dtype=int32)
            @tir.call_extern("VTAUopLoopEnd", dtype=int32)
          }
          @tir.vta.coproc_dep_push(2, 1, dtype=int32)
        }
      }
    }
    @tir.vta.coproc_dep_pop(2, 1, dtype=int32)
    attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 2 {
      attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushALUOp" {
        @tir.call_extern("VTAUopLoopBegin", 16, 1, 1, 0, dtype=int32)
        @tir.vta.uop_push(1, 0, 0, 0, 0, 3, 1, 8, dtype=int32)
        @tir.call_extern("VTAUopLoopEnd", dtype=int32)
      }
      attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushALUOp" {
        @tir.call_extern("VTAUopLoopBegin", 16, 1, 1, 0, dtype=int32)
        @tir.vta.uop_push(1, 0, 0, 0, 0, 1, 1, 0, dtype=int32)
        @tir.call_extern("VTAUopLoopEnd", dtype=int32)
      }
      attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_uop_scope" = "VTAPushALUOp" {
        @tir.call_extern("VTAUopLoopBegin", 16, 1, 1, 0, dtype=int32)
        @tir.vta.uop_push(1, 0, 0, 0, 0, 0, 1, 127, dtype=int32)
        @tir.call_extern("VTAUopLoopEnd", dtype=int32)
      }
      @tir.vta.coproc_dep_push(2, 3, dtype=int32)
    }
    attr [IterVar(vta, (nullptr), "ThreadIndex", "vta")] "coproc_scope" = 3 {
      @tir.vta.coproc_dep_pop(2, 3, dtype=int32)
      for (i1.inner: int32, 0, 16) {
        @tir.call_extern("VTAStoreBuffer2D", @tir.tvm_thread_context(@tir.vta.command_handle(, dtype=handle), dtype=handle), i1.inner, 4, res_2, ((i1.outer*16) + i1.inner), 1, 1, 1, dtype=int32)
      }
      @tir.vta.coproc_dep_push(3, 2, dtype=int32)
    }
  }
  @tir.vta.coproc_sync(, dtype=int32)
  @tir.vta.coproc_dep_pop(3, 2, dtype=int32)
}
```

## TVM 编译和验证

指定 schedule 后，可以将其编译为 TVM 函数。保存模块，然后可以通过 RPC 发送。运行这个函数，并根据 numpy 实现对其进行验证，以确保正确性。

``` python
# 编译 TVM 模块
my_gemm = vta.build(
    s, [data, weight, res], tvm.target.Target("ext_dev", host=env.target_host), name="my_gemm"
)
temp = utils.tempdir()
my_gemm.save(temp.relpath("gemm.o"))
remote.upload(temp.relpath("gemm.o"))
f = remote.load_module("gemm.o")

# 获取远程设备上下文
ctx = remote.ext_dev(0)

# 在 (-128, 128] 的 int 范围内随机初始化数据和权重数组
data_np = np.random.randint(-128, 128, size=(batch_size, in_channels)).astype(data.dtype)
weight_np = np.random.randint(-128, 128, size=(out_channels, in_channels)).astype(weight.dtype)

# 将数据和权重数组从 2D 打包为 4D 打包布局
data_packed = data_np.reshape(
    batch_size // env.BATCH, env.BATCH, in_channels // env.BLOCK_IN, env.BLOCK_IN
).transpose((0, 2, 1, 3))
weight_packed = weight_np.reshape(
    out_channels // env.BLOCK_OUT, env.BLOCK_OUT, in_channels // env.BLOCK_IN, env.BLOCK_IN
).transpose((0, 2, 1, 3))

# 用 tvm.nd.array 将输入/输出数组格式化为 DLPack 标准
data_nd = tvm.nd.array(data_packed, ctx)
weight_nd = tvm.nd.array(weight_packed, ctx)
res_nd = tvm.nd.array(np.zeros(output_shape).astype(res.dtype), ctx)

# 清除统计
if env.TARGET in ["sim", "tsim"]:
    simulator.clear_stats()

# 调用模块进行计算
f(data_nd, weight_nd, res_nd)

# 针对 numpy 实现进行验证
res_ref = np.dot(data_np.astype(env.acc_dtype), weight_np.T.astype(env.acc_dtype))
res_ref = res_ref >> env.INP_WIDTH
res_ref = np.clip(res_ref, 0, inp_max)
res_ref = res_ref.astype(res.dtype)
res_ref = res_ref.reshape(
    batch_size // env.BATCH, env.BATCH, out_channels // env.BLOCK_OUT, env.BLOCK_OUT
).transpose((0, 2, 1, 3))
np.testing.assert_equal(res_ref, res_nd.numpy())

# 打印统计
if env.TARGET in ["sim", "tsim"]:
    sim_stats = simulator.stats()
    print("Execution statistics:")
    for k, v in sim_stats.items():
        print("\t{:<16}: {:>16}".format(k, v))

print("Successful blocked matrix multiply test!")
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:267: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
Execution statistics:
        inp_load_nbytes :             4096
        wgt_load_nbytes :          1048576
        acc_load_nbytes :                0
        uop_load_nbytes :               20
        out_store_nbytes:             1024
        gemm_counter    :             4096
        alu_counter     :              192
Successful blocked matrix multiply test!
```

## 总结

本教程演示了 TVM 调度原语如何实现矩阵乘法示例的计算分块，进而能够将任意大的计算映射到有限的硬件加速器资源上。

[下载 Python 源代码：matrix_multiply_opt.py](https://tvm.apache.org/docs/_downloads/822e9d945c0bbf1cf23fc4f53c1b7906/matrix_multiply_opt.py)

[下载 Jupyter Notebook：matrix_multiply_opt.ipynb](https://tvm.apache.org/docs/_downloads/4d3f955a709b320db0d42740fead8ac1/matrix_multiply_opt.ipynb)
