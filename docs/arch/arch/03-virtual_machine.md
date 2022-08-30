---
title: 向 TVM 中添加虚拟机：Relay 虚拟机
sidebar_position: 130
---

# 向 TVM 中添加虚拟机：Relay 虚拟机

Relay 是一种新的程序表示形式，实现了很多机器学习程序的表示和优化。然而，在支持了一组更具表现力的程序的同时，也引入了几个新的执行挑战。

Relay 解释器可以执行完整的语言，但有明显的局限性——它不适合生产部署。它被构造为一个执行 AST 遍历（用来运行程序）的低效解释器。这种方法在概念上很简单，但效率很低，因为 AST 遍历严重依赖于间接层。

编译动态代码还存在其他挑战，例如动态调度和分配、完全动态的张量 shape 和控制流。解释器为它们提供了简单的解决方案，但没有一个是完美的。

第二种执行机制是已有的图执行器。为将 Relay 程序定位到这里，将它们中的一小部分编译为旧的计算图格式，并在 runtime 上执行。图执行器仅在非常有限的 Relay 程序子集上提供了快速的执行体验。

一种非标准的替代方法是 Relay 的 ahead-of-time 编译器，它将 Relay 程序编译为包含提前实现的共享库。ahead-of-time 编译器提供了较好的性能，但难以扩展和检测，只能通过修改代码生成和优化机制来实现。

Relay 虚拟机旨在成为一个平衡了这些竞争方法的框架，提供一个动态执行环境——它通过灵活的扩展机制与其他方法（如提前编译）进行扩展、检测和集成。

虚拟机是为了取得部署和执行 Relay 程序时性能和灵活性之间的平衡，同时不损失 TVM 的优势。

虚拟机 (VM) 设计是编程语言和系统中一个经过充分研究的领域，成熟的嵌入式编程语言都有多种虚拟机设计。以前的语言虚拟机设计针对传统程序的执行配置文件进行了大量适配。传统程序处理小的标量值，并由大量底层指令组成。

由于指令的数量很多，因此指令的执行和调度必须非常高效。机器学习的上下文中，主要用（相对）较少的高级指令来处理张量值。机器学习 (ML) 程序的 cost 中心是对大量输入的耗时算子的调用，比如 GEMM 或卷积。由于 ML 程序呈现的执行配置文件，标量虚拟机中的微优化显得没那么重要了。

TVM 很好地支持了视觉模型，但也希望能够支持更广泛的模型。图执行器能够利用输入图的完全静态特性，来执行积极的优化，比如完全静态分配，以及最佳内存重用。若引入的模型要利用控制流、递归、动态 shapes 和动态分配，就必须改变执行的工作方式。选择 Relay 虚拟机合情合理。

本文档的其余部分提供了 Relay 虚拟机设计及其指令集的高级概述。

## 设计

虚拟机的设计侧重于简单性，同时不牺牲性能。要实现这点，必须专注于设计张量虚拟机，而非标量虚拟机。

在张量虚拟机的设置中，进行了以下优化：对象的低成本「分配」（通过避免实际分配）、静态片段的重用，以及进行动态 shape 的能力（即锯齿张量）。

### 指令集

指令集和指令表示的选择是虚拟机最关键的设计决策。当前的指令表示是包含操作码和数据载荷的标记联合体。重要的设计决策是指令的抽象级别（RISC 与 CISC），以及获取数据的方式（固定宽度指令编码 vs. 可变长度编码）。当前版本更接近于 CISC，具有像 AllocTensor 这样的复杂指令，并且由于包含 shape 作为指令的一部分，因此其长度可变。当前的指令集的级别较高，基本和 Relay 中的高级操作对应。

#### Ret

**参数**：

```plain
RegName dst
RegName result
```

将 `result` 寄存器中的对象返回到调用者的 `dst` 寄存器。

#### InvokePacked

**参数**：

```plain
Index packed_index
Index arity
Index output_size
RegName* packed_args
```

调用 `packed_index` 表示的打包函数。`arity` 和 `output_size` 用于通知虚拟机预期有多少输入和输出。`packed_args` 存储了参数寄存器的列表。注意：`Index` 是 `int64_t` 的别名，在其他指令中也会用到。

#### AllocTensor

**参数**：

```plain
RegName dst
RegName storage
uint32_t ndim
int64_t* shape
DLDataType dtype
```

从给定的存储块 `storage` 分配一个张量值，这个张量值使用常量 shape（存储在 `shape` 中）和 `dtype`。结果保存到 `dst` 寄存器中。

#### AllocTensorReg

**参数**：

```plain
RegName dst
RegName storage
RegName shape_register
DLDataType dtype
```

从给定的存储块（存储在 `storage` 中）分配适当的 shape 的张量值（存储在 `shape_register` 中）和 `dtype`。结果保存到 `dst` 寄存器中。

#### AllocStorage

**参数**：

```plain
RegName dst
RegName size
RegName alignment
DLDataType dtype_hint
```

用给定的 `size`、`alignment` 和数据类型 `dtype_hint` 分配存储块。分配的存储块存储在 `dst` 寄存器中。

#### AllocADT

**参数**：

```plain
RegName dst
Index tag
Index num_fields
RegName* datatype_fields
```

用 `datatype_fields` 寄存器中的 `num_fields` 条目，分配带有 `tag` 标记的数据类型。结果保存到 `dst` 寄存器中。

#### AllocClosure

**参数**：

```plain
RegName dst
Index clo_index
Index num_freevar
RegName* free_vars;
```

用 `clo_index` 的 VMFunction 作为其代码分配一个闭包，并从 `free_vars` 中的寄存器分配 `num_freevar` 条目。结果保存到 `dst` 寄存器中。

#### GetField

**参数**：

```plain
RegName dst
RegName object
Index field_index
```

从 `object` 中获取 `field_index` 索引的字段值。并将结果保存到 `dst` 寄存器中。

#### If

**参数**：

```plain
RegName test
RegName target
Index true_offset
Index false_offset
```

检查 `test` 寄存器中的对象是否等于 `target`。若相等，则通过 `true_offset` 进行相对跳转，否则通过 `false_offset` 进行相对跳转。

#### GetTag

**参数**：

```plain
RegName object
RegName dst
```

获取 `object` 寄存器中 ADT 对象的对象标签。并将结果保存到 `dst` 寄存器中。

#### Fatal

虚拟机执行失败。

#### Goto

**参数**：

```plain
Index pc_offset
```

通过 `pc_offset` 进行无条件相对跳转。

#### Invoke

**参数**：

```plain
Index func_index
```

在 `func_index` 中调用函数，使用 VMFunction 的 arity 字段中包含的参数数量。

#### InvokeClosure

**参数**：

```plain
RegName closure
Index num_closure_args
RegName* closure_args
```

调用 `closure`，使用闭包的 VMFunction 中声明的参数数量。

#### LoadConst

**参数**：

```plain
RegName dst
Index const_index
```

从常量池中加载 `const_index` 处的常量。结果保存到 `dst` 寄存器中。

#### LoadConsti

**参数**：

```plain
Index val
RegName dst
```

将整型常量 `val` 加载到 `dst` 寄存器中。结果是一个秩为 0 的张量。

### 对象表示

用对象来表示虚拟机使用的对象。

目前，`NDArray`、`ADT` 和 `Closure` 这三种类型的对象分别用于表示张量、元组/列表和闭包数据。更多详细信息，可以分别在 [include/tvm/runtime/ndarray.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/ndarray.h)，[include/tvm/runtime/vm/vm.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/vm.h) 和 [include/tvm/runtime/container.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/container.h) 中找到。

### 堆栈和状态

Relay 虚拟机维护一个栈帧 (stack frame)，其中包含如何恢复之前的调用的信息。寄存器被分配在每个函数的连续空间（虚拟寄存器文件）中。

跟踪一组调用的 Relay 函数，一个指向其字节码的指针，以及字节码的偏移量（称为程序计数器）。

``` c++
struct VirtualMachine {
  ...
  std::vector<VMFrame> frames;
  ...
  // 当前函数。
  size_t func_index;
  // 指向当前函数指令的指针。
  const Instruction* code;
  // 当前程序计数器相对于代码指针。
  size_t pc;
  ...
};
```

### 调度循环

虚拟机的一个关键部分是调度循环。调度循环通常主导虚拟机的执行时间，而实验后发现 Relay 并非如此。实现一个简单的 `switch/goto` 调度循环——基于指令操作码进行调度。

这个循环由 `VirtualMachine::Run()` 实现。

### 虚拟机编译器

这个基础架构的一个重要组成部分是将 Relay 的完整 IR 编译成字节码序列的编译器。虚拟机编译器将 `tvm::relay::Module` 转换为 `tvm::relay::vm::Executable`。可执行文件包含一组编译函数（在 `tvm::relay::vm::Function` 中）。这些函数包含有关函数的元数据，及其编译的字节码。可以通过 `tvm::relay::vm::VirtualMachine` 对象加载和运行发出的可执行对象。有关数据结构的完整定义，参见 [include/tvm/runtime/vm/executable.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/executable.h) 和 [include/tvm/runtime/vm/vm.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/vm.h)。

### 优化

虚拟机编译器要进行很多优化。每一个都被实现为 pass，由 Relay pass 管理器管理 。

标有 *TODO* 的优化尚未实现。

* A-范式
* Lambda 提升（参见 [src/relay/vm/lambda_lift.cc](https://github.com/apache/tvm/blob/main/src/relay/backend/vm/lambda_lift.cc)）
* 内联原语（参见 [src/relay/vm/inline_primitives.cc](https://github.com/apache/tvm/blob/main/src/relay/backend/vm/inline_primitives.cc)）
* 常量池布局（参见 [src/relay/backend/vm/compiler.cc](https://github.com/apache/tvm/blob/main/src/relay/backend/vm/compiler.cc)）
* 尾调用优化 (TODO)
* 存活性分析 (TODO)

### 序列化

必须对 Relay 虚拟机编译器生成的可执行文件序列化和反序列化，因为可能要将模型保存到磁盘，然后执行推理。在此之前，Relay 已经在 json 文件中为图执行器生成了一个序列化的表单。但是，相同的格式不能直接用于虚拟机，因为它发出的是字节码，而非计算图样式的程序。可执行文件的序列化本质上需要处理模型特定的（即权重和内核）和虚拟机相关的（即字节码和全局函数名称）数据。

对于内核，可以方便地利用现有的 TVM 架构，来保存和加载编译好的库模块。这里只关注用二进制格式来序列化其他几个组件，这些组件按以下顺序组织：

* 全局部分。这一节包含虚拟机使用的全局变量（函数名称）。
* 常量部分。这一节用于存储虚拟机的常量池（即模型的权重）。
* 原语名称部分。引入这一节是为了归纳由虚拟机调用的原语算子名称列表，即以 `fused_` 开头的名称。原语名称用作符号，从而在编译的内核库中查找函数指针。
* 代码部分。包括字节码在内的虚拟机函数位于这一节中。调度循环遍历此部分以获取执行指令。
  
因此，不同于包含权重 (.params)、图 json (.json) 和编译内核库 (.so) 的图执行器 artifact，序列化的可执行 artifact 由 Relay 对象文件 (.ro) 和编译内核组成 (.so)。

实现的 `save` 函数将可执行文件存储到磁盘，并序列化为上述格式。同时，`load_exec` 函数用于加载序列化的内核二进制以及可执行相关的二进制代码，这些二进制代码之后也会用于实例化虚拟机对象。更多示例，参阅 [test_vm_serialization.py](https://github.com/apache/tvm/blob/main/tests/python/relay/test_vm_serialization.py) 文件。

### 未解决的问题

#### 如何处理动态 shape？

随着 Relay（TVM 的编译器）的升级，TVM 对动态 shape 的支持也在不断发展。推荐在 TVM 的论坛 (https://discuss.tvm.apache.org/) 中获取有关动态 shape 支持的最新进展。

#### 如何修改虚拟机来支持某些代码路径的 JIT 编译？

在代码生成空间中，仍有许多权衡因素需要分析。虚拟机的设计非常灵活，因此可以对其进行修改，供将来的实验使用。

#### 如何支持异构执行？

假设已经对合适的设备副本进行了注解，异构执行应该可以开箱即用。为正确执行此操作，要运行设备注解和拷贝 pass。