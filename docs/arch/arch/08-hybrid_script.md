---
title: 混合前端开发者指南
sidebar_position: 180
---

# 混合前端开发者指南

对于这样的开发者：

1. 尝试编写一些 TVM 尚未支持的初步模式，[混合前端语言参考手册](https://tvm.apache.org/docs/reference/langref/hybrid_script.html#hybrid-langref-label) 可能很有用。
2. 想要知道模块的实现细节，那么请详细阅读本节！

## 特征

### 软件仿真

在软件仿真中，最有趣的是 `tvm.te.hybrid.script` 装饰器。它的作用：

1. 导入 runtime 变量
2. 根据传递的参数重载函数

如果我说的不对请纠正我：我认为上面 1. 的实现很危险，但别无选择。我只是将这些名称添加到 Python 字典 `func.__global__` 中，`func` 调用完后，将清除这些名称。

重载很简单：装饰器会检查参数的类型，并决定实际要调用哪个函数。

### 后端编译

编译是一个大模块，更多细节参阅 `python/tvm/te/hybrid/`。第一阶段确定用法，或者更准确地说，是每个变量的声明，第二阶段进行实际的 IR 生成。

### 属性

目前**仅**支持张量的 shape 属性。参考 `python/tvm/te/hybrid/parser.py` 中的 `visit_Subscript`，了解更多详细信息。这是一种侵入式的解决方案，只在下标处检查了属性。

### 循环

HalideIR 中的循环共有 4 种类型：`serial`，`unrolled`，`parallel` 和 `vectorized`。

:::note
与 HalideIR 不同，在 `loop_type(a, b)` 中，`a` 是起点，`b` 是迭代的行程计数。这里的 `loop_type(a, b)` 表示 `[a, b)`。因此，当将其降级到 HalideIR 时，要执行 `start, extent = a, b - a`。
:::

:::note
它们在 HalideIR 中是被动形式的枚举。这里因为已准备好运行它们，所以用主动形式来注解循环。
:::

### 变量

因为 `HalideIR` 中没有变量，因此所有可变的变量都会被降级为大小为 1 的数组。它将变量的第一次存储作为其声明。

### 数学内联函数

目前支持这些数学内联函数：`log`，`exp`，`sigmoid`，`tanh`，`power` 和 `popcount`。数学内联函数由装饰器导入。大多数内联函数都是借用库的实现，除了 `popcount` 和 `sigmoid` 是手动实现的。

### 转换

可以用关键字 `uint8`，`uint16`，`uint32`，`uint64`，`int8`，`int16`，`int32`，`int64`，`float16`，`float32` 和 `float64` 来转换值。