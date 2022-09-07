---
title: 设备/Target 交互
sidebar_position: 150
---

# 设备/Target 交互

本文档适用于有兴趣了解 TVM 框架如何与特定设备 API 交互的开发者，或希望实现对新 API 或新硬件的支持的开发者。

所有新的 runtime 环境都必须实现的三个主要方面：

* [DeviceAPI](#tvm-target-specific-device-api) 类为特定设备提供句柄，以及用于与其交互的 API。它定义了一个通用接口，用于查询设备参数（例如可用内存、线程数等）和执行简单操作（例如，从主机复制内存，或在设备上的缓冲区之间复制）。
* [Target](#tvm-target-specific-target) 类描述了运行函数的设备。它既对 target 代码生成器公开，也对优化 pass 公开。
* [target 代码生成器](#tvm-target-specific-codegen) 从 IRModule 构造了一个 [模块](/docs/arch/arch/runtimes#module)，它由一个或多个 [PackedFunc](/docs/arch/arch/runtimes#PackedFunc) 组成。

## DeviceAPI {#tvm-target-specific-device-api}

`DeviceAPI` 表示特定硬件设备 API 的句柄。（例如，`CUDADeviceAPI` 通过 CUDA 框架处理所有的交互。）大多数 `DeviceAPI` 方法接收一个 `device_id` 参数，来指定访问哪个设备。Python 中通常用 `tvm.runtime.device()` 函数访问它们，这个函数返回特定设备的句柄，通过特定 API 访问。（例如，`tvm.runtime.device()` 通过 CUDA API 访问物理设备 `0`。）

* 属性查询 - `GetAttr` 允许查询不同的设备特定的参数，例如设备名称、线程数等。[device_api.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/device_api.h) 中的 `enum DeviceAttrKind` 定义了可以查询的参数。有些设备不支持部分可查询参数。若无法查询某个参数（例如 Vulkan 上的 `kMaxClockRate`），或者某个参数不适用（例如 CPU 上的 `kWarpSize`），那么查询返回 `nullptr`。
* 设置活动设备 - `SetDevice` 应将某个特定设备设为活动设备。若要在设备上执行 target 特定的 codegen 生成的 `PackedFunc` ，应该在活动设备上运行。
* 内存管理 - 用于在设备上分配和释放内存的程序。
   * 分配数据空间 - `AllocDataSpace` 和 `FreeDataSpace` 在设备上分配和释放空间。这些分配可以作为输入和输出提供给算子，并构成算子计算图的主要数据流。它们必须能够在主机和数据空间之间传输数据。返回值是一个不透明的 `void*`。虽然某些实现返回一个内存地址，但这不是必需的，并且 `void*` 可能是不透明句柄，只能由生成它的设备后端解释。 `void*` 用作其他后端特定的函数的参数，例如 `CopyDataFromTo`。
   * 分配工作空间 - `AllocWorkspace` 和 `FreeWorkspace` 在设备上分配和释放空间。不同于数据空间，它们用于存储算子定义中的中间值，并且不需要传输到主机设备，或从主机设备传输。若 `DeviceAPI` 子类没有实现这些方法，它们会默认调用相应的 `DataSpace` 函数。
   * 复制数据 - `CopyDataFromTo` 应该将数据从一个位置复制到另一个位置。副本的类型由 `dev_from` 和 `dev_to` 参数确定。实现应支持在单个设备上将内存从 CPU 复制到设备、从设备复制到 CPU，以及从一个缓冲区复制到另一个缓冲区。若源位置或目标位置在 CPU 上，则对应的 `void*` 指向一个 CPU 地址，这个地址可以传递给 `memcpy`。若源位置或目标位置在设备上，则相应的 `void*` 之前已由 `AllocDataSpace` 或 `AllocWorkspace` 生成。

   这些副本排队等待在特定的 `TVMStreamHandle` 上执行。但是，该实现不应该假定在 `CopyDataFromTo` 调用完成后，CPU 缓冲区仍然有效或可访问。
* 执行流管理 - 用于处理 `TVMStreamHandle` 的程序，它表示用于执行命令的并行执行流。
   * 创建流 - `CreateStream` 和 `FreeStream` 为执行流分配/释放句柄。若设备仅实现一个命令队列，则 `CreateStream` 返回 `nullptr`。
   * 设置活动流 - `SetStream` 将流设置为活动的。在活动期间，若特定于 target 的 code gen 生成的 `PackedFunc` 需要在设备上执行，则应将工作提交到活动流。
   * 同步到 CPU - `StreamSync` 将执行流同步到 CPU。`StreamSync` 调用一次性返回所有的内存转换，以及在调用完成前提交的计算。
   * 在流之间同步 - `SyncStreamFromTo` 在源流和目标流之间引入同步屏障 (synchronization barrier)。即在源流完成当前排队的所有命令前，目标流不会超出当前排队的命令。

TVM 框架若要使用新的 DeviceAPI，应该按照以下步骤注册：

1. 创建一个函数，它实例化新 DeviceAPI，并返回一个指向它的指针：

   ``` c++
   FooDeviceAPI* FooDeviceAPI::Global() {
     static FooDeviceAPI inst;
     return &inst;
   }
   ```

2. 将函数注册到 TVM 注册表：

   ``` c++
   TVM_REGISTER_GLOBAL("device_api.foo").set_body_typed(FooDeviceAPI::Global);
   ```

1. 为新的 DeviceAPI 添加一个进入 [c_runtime_api.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/c_runtime_api.h) 中的 `TVMDeviceExtType` 枚举的入口。该值是一个未使用值，它大于 `DLDeviceType::kDLExtDev`，但小于 `DeviceAPIManager::kMaxDeviceAPI`。
2. 给 [device_api.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/device_api.h) 中的 `DeviceName` 添加一个案例，从而将枚举值转换为字符串表示形式。这个字符串表示应该和 `TVM_REGISTER_GLOBAL` 的名称匹配。
3. 将入口添加到 `tvm.runtime.Device` 的 `MASK2STR` 和 `STR2MASK` 字典，获取新的枚举值。

## Target 定义 {#tvm-target-specific-target}

`Target` 对象是属性（包含物理设备、其硬件/驱动程序限制，及其功能）的查找表。在优化和代码生成阶段都可以访问 `Target`。虽然相同的 `Target` 类适用于所有 runtime target，但每个 runtime target 可能需要添加特定于 target 的选项。

在 [target_kind.cc](https://github.com/apache/tvm/blob/main/src/target/target_kind.cc) 中，为 `TVM_REGISTER_TARGET_KIND` 添加一个新的声明，传递新 target 的字符串名称，以及该 target 运行设备的 `TVMDeviceExtType` 或 `DLDeviceType` 枚举值。通常，target 名称和设备名称匹配。（例如，`"cuda"` target 在 `kDLCUDA` 设备上运行。）但也有例外，例如多个不同的代码生成 targets 可以在同一个物理设备上运行。 （例如，`"llvm"` 和 `"c"` targets 都在 `kDLCPU` 设备类型上运行。）

特定 target 种类的所有选项都使用 `add_attr_option` 函数添加，具有可选的默认值。可以用 `set_target_parser` 来添加 *Target*  解析器，来处理那些动态基于其他参数，或是从设备属性查询到的参数。

这个参数定义了一个解析器，可以解析 target 的字符串描述。这是在 C++ 的 `Target::Target(const String&)` 构造函数中实现的，它接收 JSON 格式的字符串，通常用 `tvm.target.Target` 这个 Python 对象来调用。例如， `tvm.target.Target('{"kind": "cuda", "max_num_threads": 1024}')` 会创建一个 `cuda` target，同时覆盖默认的最大线程数。

代码生成器中可以用 C++ 中的 `target->GetAttr<T>(param_name)`，或是 Python 中的 `target.attrs` 字典来访问 target 属性。

## Target 代码生成器 {#tvm-target-specific-codegen}

代码生成器采用优化的 `IRModule`，并将其转换为可执行表示。每个代码生成器注册后，才能被 TVM 框架使用。这是通过注册 `"target.build.foo"` 函数来完成的，其中 `foo` 与上面的 `TVM_REGISTER_TARGET_KIND` 定义中使用的名称相同。

``` c++
tvm::runtime::Module GeneratorFooCode(IRModule mod, Target target);
TVM_REGISTER_GLOBAL("target.build.foo").set_body_typed(GeneratorFooCode);
```

代码生成器有两个参数：第一个是要编译的 `IRModule`，第二个是描述代码要在哪个设备运行的 `Target`参数。因为执行编译的环境与执行代码的环境不一定相同，所以代码生成器不应该在设备本身上执行任何属性的查找，而是应该访问存储在 `Target` 中的参数。

输入 `IRModule` 中的每个函数都可以在输出 `runtime::Module` 中按名称访问。
