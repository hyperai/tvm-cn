---
title: VTA 配置
---

# VTA 配置

VTA 堆栈包含硬件加速器堆栈和基于 TVM 的软件堆栈。VTA 具有开箱即用的灵活性：通过修改高级配置文件 `3rdparty/vta-hw/config/vta_config.json`，用户可以更改张量内联函数的 shape、时钟频率、流水线、数据类型宽度和芯片缓冲区大小。

## 参数概述

下表解释了 `vta_config.json` 文件中列出的参数。

| **属性** | **格式** | **描述** |
|:---|:---|:---|
| TARGET | String | TVM 设备 target。 |
| HW_VER | String | VTA 硬件版本号。 |
| LOG_INP_WIDTH | Int (log2) | 输入数据类型有符号整数宽度。 |
| LOG_WGT_WIDTH | Int (log2) | 权重数据类型有符号整数宽度。 |
| LOG_ACC_WIDTH | Int (log2) | 累加器数据类型有符号整数宽度。 |
| LOG_BATCH | Int (log2) | VTA 矩阵乘以固有输入/输出维度 0。 |
| LOG_BLOCK | Int (log2) | VTA 矩阵乘以内部维度。 |
| LOG_UOP_BUFF_SIZE | Int (log2) | 以字节为单位的微操作片上缓冲区。 |
| LOG_INP_BUFF_SIZE | Int (log2) | 以字节为单位输入芯片上缓冲区。 |
| LOG_WGT_BUFF_SIZE | Int (log2) | 以字节为单位的芯片上缓冲区权重。 |
| LOG_ACC_BUFF_SIZE | Int (log2) | 以字节为单位的累加器芯片上缓冲区。 |

:::note
当参数名称以 `LOG` 开头时，表示它描述的值只能表示为 2 的幂。因此，用这些参数的 log2 值来描述它们。例如，为了描述输入数据类型的 8 位整数宽度，将 `LOG_INP_WIDTH` 设置为 3，即 log2(8)。类似地，要描述 64kB 微操作缓冲区，将 `LOG_UOP_BUFF_SIZE` 设置为 16。
:::

每个参数的详细信息：

* `TARGET`：可设置为 `"pynq"`、`"ultra96"`、`"sim"`（快速模拟器）或 `"tsim"`（用验证器来循环精确模拟）。
* `HW_VER`：硬件版本，每次 VTA 硬件设计更改时都会增加。此参数用于唯一标识硬件比特流。
* `LOG_BATCH`：等价于 shape (A, B) x (B, C) 的乘积中的 A，或者通常来说是，内部张量计算的 batch 维度。
* `LOG_BLOCK`：相当于 shape (A, B) x (B, C) 中的 B 和 C，或者通常来说是，内部张量计算的输入/输出通道维度。