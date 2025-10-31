---

title: 设备 / Target 交互

---

本文档面向希望了解 TVM 框架如何与特定设备 API 进行交互的开发者，或希望为新的 API 或新硬件添加支持的开发者。

对于任何新的运行时环境，需要实现三个主要部分：

- `DeviceAPI <tvm-target-specific-device-api>`{.interpreted-text role="ref"} 类提供对特定设备的句柄，以及用于与其交互的 API。它定义了一套通用接口，用于查询设备参数（例如：可用内存、线程数量等），以及执行简单操作（例如：从主机复制内存，或在设备缓冲区之间复制数据）。
- `Target <tvm-target-specific-target>`{.interpreted-text role="ref"} 类包含将要运行函数的设备描述。它同时暴露给目标代码生成器和优化 Pass。
- `目标代码生成器 <tvm-target-specific-codegen>`{.interpreted-text role="ref"} 从 IRModule 构建一个由一个或多个 `PackedFunc <tvm-runtime-system-packed-func>`{.interpreted-text role="ref"} 组成的 `Module <tvm-runtime-system-module>`{.interpreted-text role="ref"}。

## DeviceAPI {#tvm-target-specific-device-api}

`DeviceAPI`（设备 API）表示对特定硬件设备 API 的访问句柄。（例如，`CUDADeviceAPI` 处理所有通过 CUDA 框架的交互。）大多数 `DeviceAPI` 方法都接受一个 `device_id` 参数，用于指定访问哪个设备。

在 Python 中，通常使用 `tvm.runtime.device`{.interpreted-text role="py:func"} 函数访问特定设备，该函数返回指定 API 所访问设备的句柄。（例如，`tvm.runtime.device('cuda', 0)` 表示访问通过 CUDA API 访问的物理设备 `0`。）

- **属性查询** — `GetAttr` 用于查询不同的设备特定参数，例如设备名称、线程数量等。可查询的参数定义在
  `enum DeviceAttrKind`，文件位置：
  [device_api.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/device_api.h)。
  并非所有参数都适用于所有设备。如果某个参数无法查询（例如 Vulkan 上的 `kMaxClockRate`），或不适用（例如 CPU 上的 `kWarpSize`），应返回 `nullptr`。
- **设置活动设备** — `SetDevice` 应将某个设备设置为当前活动设备。如果目标代码生成器生成的 `PackedFunc` 需要在设备上执行，该执行应发生在当前活动设备上。
- **内存管理** — 用于在设备上分配和释放内存的工具函数。
  - **分配数据空间** — `AllocDataSpace` 和 `FreeDataSpace` 用于在设备上分配和释放数据存储空间。这些空间可作为算子输入和输出，并构成算子图的主要数据流。必须支持主机与数据空间之间的数据传输。返回值为不透明指针 `void*`。某些实现返回真实地址，但这不是必须的，该指针也可能是仅可由设备后端解释的句柄。该 `void*` 将作为参数传递给其他后端函数（例如 `CopyDataFromTo`）。
  - **分配工作空间** — `AllocWorkspace` 和 `FreeWorkspace` 用于分配和释放工作区。这些区域用于算子内部中间值存储，不要求可与主机传输。如果子类未实现，则默认调用对应的数据空间分配函数。
  - **数据复制** — `CopyDataFromTo` 应在不同位置之间复制数据。复制类型由 `dev_from` 和 `dev_to` 决定。实现应该支持将内存从CPU复制到设备，从设备复制到CPU，以及在单个设备上从一个缓冲区复制到另一个缓冲区。如果源或目标位于 CPU，则指针为可直接用于 `memcpy` 的主机地址；如果位于设备，则指针必定由 `AllocDataSpace` 或 `AllocWorkspace` 生成。  
    这些复制会排队在某个 `TVMStreamHandle` 流中执行。但是实现不应假设 CPU 缓冲区在函数返回后仍然有效或可访问。
- **执行流管理** — 管理 `TVMStreamHandle`（执行命令的并行流）。
  - **创建流** — `CreateStream` / `FreeStream` 负责分配和释放执行流。如果设备只有单一指令队列，则 `CreateStream` 应返回 `nullptr`。
  - **设置活动流** — `SetStream` 用于将某个流设置为当前活跃流。目标代码生成器生成的函数执行时应提交到该流。
  - **同步到 CPU** — `StreamSync` 应同步流，使之在执行完成前阻塞返回。
  - **流间同步** — `SyncStreamFromTo` 应在两个流之间插入同步屏障，使目标流在源流执行完当前排队命令前无法继续执行。

为了使 TVM 能够使用新的 DeviceAPI，需要执行以下注册步骤：

1. 创建一个实例化 DeviceAPI 并返回其指针的函数：

    ```cpp
    FooDeviceAPI* FooDeviceAPI::Global() {
      static FooDeviceAPI inst;
      return &inst;
    }
    ```

2. 在 TVM 注册表中注册：

    ```cpp
    TVM_FFI_STATIC_INIT_BLOCK() {
      namespace refl = tvm::ffi::reflection;
      refl::GlobalDef().def("device_api.foo", FooDeviceAPI::Global);
    }
    ```

<!-- -->

3. 在 [base.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/base.h) 的 `TVMDeviceExtType` 枚举中为新的 DeviceAPI 添加条目。值需大于 `DLDeviceType::kDLExtDev`，且小于 `DeviceAPIManager::kMaxDeviceAPI`。
4. 在 [device_api.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/device_api.h) 的 `DeviceName` 中添加对应枚举 → 字符串映射，该字符串需与 `GlobalDef().def` 中一致。
5. 在 `tvm.runtime.Device`的 `_DEVICE_TYPE_TO_NAME` 与 `_DEVICE_NAME_TO_TYPE` 字典中添加对应映射。

## Target 定义 

`Target` 对象是有关物理设备、其硬件/驱动限制和能力的属性查询表。`Target` 可在优化阶段和代码生成阶段使用。虽然所有运行时共享相同的 `Target` 类，但不同运行时可能需要额外的 target 特定属性。

在 [target_kind.cc](https://github.com/apache/tvm/blob/main/src/target/target_kind.cc) 中使用 `TVM_REGISTER_TARGET_KIND` 注册新的 target，需传入 target 名称，以及对应运行设备的 `TVMDeviceExtType` 或 `DLDeviceType`。通常情况下，target 名称和设备名称一致（如 `"cuda"` 运行于 `kDLCUDA`），但也有例外（例如 `"llvm"` 与 `"c"` 目标都运行于 `kDLCPU`）。

所有 target 选项通过 `add_attr_option` 添加，可带默认值。可以使用 `set_target_parser` 添加解析器，用于处理依赖其他参数或硬件属性的动态参数。

该参数解析器定义了如何从字符串格式构造 target。这由 `Target::Target(const String&)` 构造函数执行，该构造函数接受 JSON 格式字符串，通常通过 Python：

```python
tvm.target.Target('{"kind": "cuda", "max_num_threads": 1024}')
```

在代码生成器中，可通过以下方式访问 target 属性：

- C++：`target->GetAttr<T>(param_name)`
- Python：`target.attrs`

## Target 代码生成器 

代码生成器将优化后的 `IRModule` 转换为可执行表示。每个代码生成器必须注册到 TVM 框架中，其名称为：

```
"target.build.foo"
```

其中 `foo` 与先前 `TVM_REGISTER_TARGET_KIND` 中的名称一致。

示例：

```cpp
tvm::runtime::Module GeneratorFooCode(IRModule mod, Target target);
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.foo", GeneratorFooCode);
}
```

代码生成器有两个参数。第一个是要编译的`IRModule`，第二个是描述代码应该运行在哪个设备上的目标 `Target`。由于编译环境不一定与执行环境相同，因此代码生成器**不应直接向设备查询属性**，而应始终使用 `Target` 中的属性。

输入 `IRModule` 中的每个函数都应在输出的 `runtime::Module` 中可通过名称访问。
