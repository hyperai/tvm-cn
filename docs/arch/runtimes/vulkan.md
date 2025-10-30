# Vulkan 运行时


TVM 支持使用 Vulkan 计算着色器来执行任务。 每个计算内核都会被编译成一个 SPIR-V 着色器，然后可通过 TVM 接口进行调用。

## Vulkan 功能与限制 
由于不同的 Vulkan 实现可能启用了不同的可选特性，或具有不同的物理限制，
代码生成必须了解可用的特性。这些特性对应于特定的 Vulkan 能力与限制，如 `Vulkan Capabilities Table <tvm-table-vulkan-capabilities>`{.interpreted-text role="ref"} 所示。 若未指定，TVM 会假定该能力不可用，或该限制为 Vulkan 规范中 [Required Limits](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits-minmax) 一节所定义的最小保证值。

这些参数既可以在定义 `Target <tvm-target-specific-target>`{.interpreted-text role="ref"} 时显式指定， 也可以从设备中查询。若要从设备查询，可使用特殊参数
`-from_device=N`，以从设备 ID `N` 查询所有 Vulkan 参数。
任何额外显式指定的参数将覆盖从设备查询到的参数。

| 参数名称（Target Parameter）                | 所需 Vulkan 版本/扩展                      | 查询的 Vulkan 参数结构体字段                                                                 | 默认值 |
|--------------------------------------------|--------------------------------------------|------------------------------------------------------------------------------------------------|--------|
| `supported_subgroup_operations`（支持的子群操作） | Vulkan 1.1+                                | `VkPhysicalDeviceSubgroupProperties::supportedOperations`                                      | 0（对应子群特性标志位 VkSubgroupFeatureFlagBits） |
| `max_push_constants_size`（最大 Push 常量大小） |                                            | `VkPhysicalDeviceLimits::maxPushConstantsSize`                                                 | 128 字节 |
| `max_uniform_buffer_range`（最大 Uniform Buffer 范围） |                                        | `VkPhysicalDeviceLimits::maxUniformBufferRange`                                                | 16384 字节 |
| `max_storage_buffer_range`（最大 Storage Buffer 范围） |                                       | `VkPhysicalDeviceLimits::maxStorageBufferRange`                                                | 2^27 字节 |
| `max_per_stage_descriptor_storage_buffer`（每阶段可用 Storage Buffer 描述符数量） |           | `VkPhysicalDeviceLimits::maxPerStageDescriptorStorageBuffers`                                  | 4 |
| `supports_storage_buffer_storage_class`（支持 Storage Buffer 类型） | `VK_KHR_storage_buffer_storage_class`      | （无需查询，取决于扩展是否启用）                                                               | false |
| `supports_storage_buffer_8bit_access`（支持 8 位 Storage Buffer 访问） | `VK_KHR_8bit_storage`                     | `VkPhysicalDevice8BitStorageFeaturesKHR::storageBuffer8BitAccess`                              | false |
| `supports_storage_buffer_16bit_access`（支持 16 位 Storage Buffer 访问） | `VK_KHR_16bit_storage`                   | `VkPhysicalDevice16BitStorageFeaturesKHR::storageBuffer16BitAccess`                            | false |
| `supports_float16`（支持 float16 浮点类型） | `VK_KHR_shader_float16_int8`               | `VkPhysicalDeviceShaderFloat16Int8FeaturesKHR::shaderFloat16`                                  | false |
| `supports_float64`（支持 float64 浮点类型） |                                            | `VkPhysicalDeviceFeatures::shaderFloat64`                                                      | false |
| `supports_int8`（支持 int8 类型）          | `VK_KHR_shader_float16_int8`               | `VkPhysicalDeviceShaderFloat16Int8FeaturesKHR::shaderInt8`                                     | false |
| `supports_int16`（支持 int16 类型）        |                                            | `VkPhysicalDeviceFeatures::shaderInt16`                                                        | false |
| `supports_int64`（支持 int64 类型）        |                                            | `VkPhysicalDeviceFeatures::shaderInt64`                                                        | false |

截至 2021 年 5 月，并非所有 Vulkan 实现都受到支持。 例如，需要支持 64
位整数。若 Vulkan 目标不受支持， 在生成 SPIR-V 代码时将会报错。
目前也在努力消除此类限制，以支持更多 Vulkan 实现。

## SPIR-V 功能 

某些设备特性也对应于 SPIR-V 的功能或扩展，必须在着色器中声明，或要求使用最低版本的 SPIR-V。 TVM 生成的着色器会声明执行所需的最小扩展、功能以及最低 SPIR-V 版本。

如果着色器生成需要的能力或扩展在 `Target` 中未启用，将会抛出异常。


| 参数名称（Target Parameter）                | 所需 SPIR-V 版本/扩展                      | 声明的功能（Capability） |
|--------------------------------------------|--------------------------------------------|--------------------------|
| `supported_subgroup_operations`（支持的子群操作） | SPIR-V 1.3+                                | 视具体子群特性而定（参考 VkSubgroupFeatureFlagBits） |
| `supports_storage_buffer_storage_class`（支持 Storage Buffer 类） | SPV_KHR_storage_buffer_storage_class        | （使用该扩展隐式启用） |
| `supports_storage_buffer_8bit_access`（支持 8 位存储缓冲访问） | SPV_KHR_8bit_storage                        | StorageBuffer8BitAccess |
| `supports_storage_buffer_16bit_access`（支持 16 位存储缓冲访问） | SPV_KHR_16bit_storage                      | StorageBuffer16BitAccess |
| `supports_float16`（支持 Float16 浮点类型） |                                            | Float16 |
| `supports_float64`（支持 Float64 浮点类型） |                                            | Float64 |
| `supports_int8`（支持 Int8 类型）          |                                            | Int8 |
| `supports_int16`（支持 Int16 类型）        |                                            | Int16 |
| `supports_int64`（支持 Int64 类型）        |                                            | Int64 |


## Vulkan 特定环境变量

SPIR-V 代码生成器和 Vulkan 运行时均可通过环境变量修改部分运行时行为。
这些变量主要用于调试，以便更轻松地测试特定代码路径或输出更多信息。
所有布尔类型变量在设置为非零整数时视为"真"。 未设置、设为 0
或空字符串时，视为"假"。

-   `TVM_VULKAN_DISABLE_PUSH_DESCRIPTOR` ------ 布尔变量。 若为真，TVM
    将显式分配描述符，而不使用
    [VK_KHR_push_descriptor](https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_push_descriptor.html)
    或
    [VK_KHR_descriptor_update_template](https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_descriptor_update_template.html)
    扩展。 若为假，TVM 会根据扩展的可用性自动决定是否使用。
-   `TVM_VULKAN_DISABLE_DEDICATED_ALLOCATION` ------ 布尔变量。
    若为真，TVM 不会将内存分配标记为"专用分配"， 也不会使用
    [VK_KHR_dedicated_allocation](https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_dedicated_allocation.html)
    扩展。 若为假，TVM 会依据
    [VkMemoryDedicatedRequirements](https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkMemoryDedicatedRequirements.html)
    判断是否应将内存标记为专用分配。
-   `TVM_VULKAN_ENABLE_VALIDATION_LAYERS` ------ 布尔变量。 若为真，TVM
    会启用设备支持的 [Vulkan validation
    layers](https://github.com/KhronosGroup/Vulkan-LoaderAndValidationLayers/blob/master/layers/README.md)。
    若为假，则不会启用任何验证层。
-   `TVM_VULKAN_DISABLE_SHADER_VALIDATION` ------ 布尔变量。
    若为真，将跳过使用
    [spvValidate](https://github.com/KhronosGroup/SPIRV-Tools#validator)
    进行的 SPIR-V 着色器验证。 若为假（默认），TVM 生成的所有 SPIR-V
    着色器都将通过
    [spvValidate](https://github.com/KhronosGroup/SPIRV-Tools#validator)
    进行验证。
-   `TVM_VULKAN_DEBUG_SHADER_SAVEPATH` ------ 目录路径。
    若设置为非空字符串，Vulkan 代码生成器会将 TIR、二进制 SPIR-V
    以及反汇编后的 SPIR-V 着色器保存到此目录，用于调试。
