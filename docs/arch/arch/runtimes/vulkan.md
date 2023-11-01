---
title: Vulkan Runtime
---

# Vulkan Runtime

TVM 支持用 Vulkan 计算着色器（Vulkan compute shaders）来查询。所有计算内核都被编译成一个 SPIR-V 着色器，然后可以用 TVM 接口来调用它。

## Vulkan 的功能和限制

由于不同的 Vulkan 实现可能启用不同的可选功能，或具有不同的物理限制，因此代码生成必须知道哪些功能是可用的。这与 [Vulkan 功能表](#tvm-table-vulkan-capabilities) 中的特定 Vulkan 功能/限制一一对应。若未指定，TVM 会假定某个功能不可用，或者某个限制是 Vulkan 规范在 [必需限制](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits-minmax) 部分中保证的最小值。

这些参数应在定义 [Target](https://tvm.apache.org/docs/arch/device_target_interactions.html#tvm-target-specific-target) 时明确指定，或是可从设备中查询。若要从设备查询，可用特殊参数 `-from_device=N` 从 id 为 `N` 的设备中，查询所有 vulkan 设备参数。任何显式指定的附加参数，都会覆盖从设备查询的参数。

| **Target 参数** | **要求的 Vulkan 版本/扩**展 | **查询参数** | **默认值** |
|:---|:---|:---|:---|
| supported_subgroup_operations | Vulkan 1.1+ | VkPhysicalDeviceSubgroupProperties::supportedOperations | 0 (interpreted as [VkSubgroupFeatureFlagBits](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSubgroupFeatureFlagBits.html)) |
| max_push_constants_size |    | VkPhysicalDeviceLimits::maxPushConstantsSize | 128 bytes |
| max_uniform_buffer_range |    | VkPhysicalDeviceLimits::maxUniformBufferRange | 16384 bytes |
| max_storage_buffer_range |    | VkPhysicalDeviceLimits::maxStorageBufferRange | 227bytes |
| max_per_stage_descriptor_storage_buffer |    | VkPhysicalDeviceLimits::maxPerStageDescriptorStorageBuffers | 4 |
| supports_storage_buffer_storage_class | VK_KHR_storage_buffer_storage_class |    | false |
| supports_storage_buffer_8bit_access | VK_KHR_8bit_storage | VkPhysicalDevice8BitStorageFeaturesKHR::storageBuffer8BitAccess | false |
| supports_storage_buffer_16bit_access | VK_KHR_16bit_storage | VkPhysicalDevice16BitStorageFeaturesKHR::storageBuffer16BitAccess | false |
| supports_float16 | VK_KHR_shader_float16_int8 | VkPhysicalDeviceShaderFloat16Int8FeaturesKHR::shaderFloat16 | false |
| supports_float64 |    | VkPhysicalDeviceFeatures::shaderFloat64 | false |
| supports_int8 | VK_KHR_shader_float16_int8 | VkPhysicalDeviceShaderFloat16Int8FeaturesKHR::shaderInt8 | false |
| supports_int16 |    | VkPhysicalDeviceFeatures::shaderInt16 | false |
| supports_int64 |    | VkPhysicalDeviceFeatures::shaderInt64 | false |

截至 2021 年 5 月，还有一些 Vulkan 的实现没支持。例如，要支持 64 位整数。若不支持 Vulkan target，则会在 SPIR-V 代码生成期间报错。我们正努力消除这些限制，并支持其他 Vulkan 实现。

## SPIR-V 功能

某些特定于设备的功能还对应于 SPIR-V 功能或扩展，它们必须在着色器中声明，或对应于要使用某个功能所需的最低 SPIR-V 版本。TVM 生成的着色器将声明执行编译好的计算图所需的最小扩展/功能集，以及 SPIR-V 的最小允许版本。

若着色器生成需要 `Target` 中未启用的功能或扩展，则会引发异常。

| **Target 参数** | **要求的 SPIR-V 版本/扩展** | **声明的功能** |
|:---|:---|:---|
| supported_subgroup_operations | SPIR-V 1.3+ | Varies, see [VkSubgroupFeatureFlagBits](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSubgroupFeatureFlagBits.html) |
| supports_storage_buffer_storage_class | SPV_KHR_storage_buffer_storage_class |    |
| supports_storage_buffer_8bit_access | SPV_KHR_8bit_storage | StorageBuffer8BitAccess |
| supports_storage_buffer_16bit_access | SPV_KHR_16bit_storage | StorageBuffer16BitAccess |
| supports_float16 |    | Float16 |
| supports_float64 |    | Float64 |
| supports_int8 |    | Int8 |
| supports_int16 |    | Int16 |
| supports_int64 |    | Int64 |

## Vulkan 特定的环境变量

SPIR-V 代码生成和 Vulkan runtime 都有可以修改某些 runtime 行为的环境变量。这些变量用于调试，既可以轻松地测试特定代码路径，也可以根据需要输出更多信息。

若环境变量被设置为非零整数，则所有布尔标志都为真。未设置的变量、整数零或空字符串，都是错误的布尔标志。

* `TVM_VULKAN_DISABLE_PUSH_DESCRIPTOR` - 布尔标志。若为 True，TVM 将显式分配描述符，并且不会用 [VK_KHR_push_descriptor](https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_push_descriptor.html) 或 [VK_KHR_descriptor_update_template](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_descriptor_update_template.html) 扩展。若为 False，TVM 将根据它们的可用性，来决定是否使用这些扩展。
* `TVM_VULKAN_DISABLE_DEDICATED_ALLOCATION` - 布尔标志。若为 True，TVM 不会将内存分配标记为专用分配，并且不会使用 [VK_KHR_dedicated_allocation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_dedicated_allocation.html) 扩展。若为 False，TVM 将根据该缓冲区的 [VkMemoryDedicatedRequirements](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkMemoryDedicatedRequirements.html)，来决定是否将内存分配标记为专用。
* `TVM_VULKAN_ENABLE_VALIDATION_LAYERS` - 布尔标志。若为 True，TVM 将启用设备支持的 [Vulkan 验证层](https://github.com/KhronosGroup/Vulkan-LoaderAndValidationLayers/blob/master/layers/README.md)。若为 False，则不启用任何验证层。
* `TVM_VULKAN_DISABLE_SHADER_VALIDATION` - 布尔标志。若为 True，则跳过使用 [spvValidate](https://github.com/KhronosGroup/SPIRV-Tools#validator) 完成的 SPIR-V 着色器验证。若为 False（默认），则 TVM 生成的所有 SPIR-V 着色器都使用 [spvValidate](https://github.com/KhronosGroup/SPIRV-Tools#validator) 进行验证。
* `TVM_VULKAN_DEBUG_SHADER_SAVEPATH` - 目录的路径。若设置为非空字符串，Vulkan 代码生成器会将 tir、二进制 SPIR-V 和反汇编的 SPIR-V 着色器保存到此目录，用于调试。
