---
title: Vulkan Runtime
---

TVM supports using Vulkan compute shaders to execute queries. Each
computational kernel is compiled into a SPIR-V shader, which can then be
called using the TVM interface.

## Vulkan Features, Limits {#tvm-runtime-vulkan-features}

Since different Vulkan implementations may enable different optional
features or have different physical limits, the code generation must
know which features are available to use. These correspond to specific
Vulkan capabilities/limits as in
`Vulkan Capabilities Table <tvm-table-vulkan-capabilities>`{.interpreted-text
role="ref"}. If unspecified, TVM assumes that a capability is not
available, or that a limit is the minimum guaranteed by the Vulkan spec
in the [Required
Limits](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits-minmax)
section.

These parameters can be either explicitly specific when defining a
`Target <tvm-target-specific-target>`{.interpreted-text role="ref"}, or
can be queried from a device. To query from a device, the special
parameter `-from_device=N` can be used to query all vulkan device
parameters from device id `N`. Any additional parameters explicitly
specified will override the parameters queried from the device.

  Target Parameter                            Required Vulkan Version/Extension     Parameter Queried                                                     Default Value
  ------------------------------------------- ------------------------------------- --------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------
  `supported_subgroup_operations`             Vulkan 1.1+                           `VkPhysicalDeviceSubgroupProperties::supportedOperations`             0 (interpreted as [VkSubgroupFeatureFlagBits](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSubgroupFeatureFlagBits.html))
  `max_push_constants_size`                                                         `VkPhysicalDeviceLimits::maxPushConstantsSize`                        128 bytes
  `max_uniform_buffer_range`                                                        `VkPhysicalDeviceLimits::maxUniformBufferRange`                       16384 bytes
  `max_storage_buffer_range`                                                        `VkPhysicalDeviceLimits::maxStorageBufferRange`                       2^27^bytes
  `max_per_stage_descriptor_storage_buffer`                                         `VkPhysicalDeviceLimits::maxPerStageDescriptorStorageBuffers`         4
  `supports_storage_buffer_storage_class`     VK_KHR_storage_buffer_storage_class                                                                         false
  `supports_storage_buffer_8bit_access`       VK_KHR_8bit_storage                   `VkPhysicalDevice8BitStorageFeaturesKHR::storageBuffer8BitAccess`     false
  `supports_storage_buffer_16bit_access`      VK_KHR_16bit_storage                  `VkPhysicalDevice16BitStorageFeaturesKHR::storageBuffer16BitAccess`   false
  `supports_float16`                          VK_KHR_shader_float16_int8            `VkPhysicalDeviceShaderFloat16Int8FeaturesKHR::shaderFloat16`         false
  `supports_float64`                                                                `VkPhysicalDeviceFeatures::shaderFloat64`                             false
  `supports_int8`                             VK_KHR_shader_float16_int8            `VkPhysicalDeviceShaderFloat16Int8FeaturesKHR::shaderInt8`            false
  `supports_int16`                                                                  `VkPhysicalDeviceFeatures::shaderInt16`                               false
  `supports_int64`                                                                  `VkPhysicalDeviceFeatures::shaderInt64`                               false

  : Vulkan Capabilities

As of May 2021, not all Vulkan implementations are supported. For
example, support for 64-bit integers is required. If a Vulkan target is
not supported, an error message should be issued during SPIR-V code
generation. Efforts are also underway to remove these requirements and
support additional Vulkan implementations.

## SPIR-V Capabilities {#tvm-runtime-vulkan-spirv-capabilities}

Some of the device-specific capabilities also correspond to SPIR-V
capabilities or extensions that must be declared in the shader, or a
minimum SPIR-V version required in order to use a feature. The
TVM-generated shaders will declare the minimum set of
extensions/capabilities and the minimum allowed version of SPIR-V that
are needed to execute the compiled graph.

If the shader generation requires a capability or extension that is not
enabled in the `Target`, an exception will be raised.

  Target Parameter                          Required SPIR-V Version/Extension      Declared Capability
  ----------------------------------------- -------------------------------------- -----------------------------------------------------------------------------------------------------------------------------------------------
  `supported_subgroup_operations`           SPIR-V 1.3+                            Varies, see [VkSubgroupFeatureFlagBits](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSubgroupFeatureFlagBits.html)
  `supports_storage_buffer_storage_class`   SPV_KHR_storage_buffer_storage_class   
  `supports_storage_buffer_8bit_access`     SPV_KHR_8bit_storage                   StorageBuffer8BitAccess
  `supports_storage_buffer_16bit_access`    SPV_KHR_16bit_storage                  StorageBuffer16BitAccess
  `supports_float16`                                                               Float16
  `supports_float64`                                                               Float64
  `supports_int8`                                                                  Int8
  `supports_int16`                                                                 Int16
  `supports_int64`                                                                 Int64

  : Vulkan Capabilities

## Vulkan-Specific Environment Variables

Both the SPIR-V code generation and the Vulkan runtime have environment
variables that can modify some of the runtime behavior. These are
intended for debugging purposes, both to more easily test specific code
paths, and to output more information as needed. All boolean flags are
true if the environment variable is set to a non-zero integer. An unset
variable, the integer zero, or an empty string are all false boolean
flags.

-   `TVM_VULKAN_DISABLE_PUSH_DESCRIPTOR` - A boolean flag. If true, TVM
    will explicitly allocate descriptors, and will not use the
    [VK_KHR_push_descriptor](https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_push_descriptor.html)
    or
    [VK_KHR_descriptor_update_template](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_descriptor_update_template.html)
    extensions. If false, TVM will decide whether to use these
    extensions based on their availability.
-   `TVM_VULKAN_DISABLE_DEDICATED_ALLOCATION` - A boolean flag. If true,
    TVM will not mark memory allocations as being dedicated allocations,
    and will not use the
    [VK_KHR_dedicated_allocation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_dedicated_allocation.html)
    extension. If false, TVM will decide whether memory allocations
    should be marked as dedicated based on the
    [VkMemoryDedicatedRequirements](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkMemoryDedicatedRequirements.html)
    for that buffer.
-   `TVM_VULKAN_ENABLE_VALIDATION_LAYERS` - A boolean flag. If true, TVM
    will enable [Vulkan validation
    layers](https://github.com/KhronosGroup/Vulkan-LoaderAndValidationLayers/blob/master/layers/README.md)
    that the device supports. If false, no validation layers are
    enabled.
-   `TVM_VULKAN_DISABLE_SHADER_VALIDATION` - A boolean flag. If true,
    the SPIR-V shader validation done with
    [spvValidate](https://github.com/KhronosGroup/SPIRV-Tools#validator)
    is skipped. If false (default), all SPIR-V shaders generated by TVM
    are validated with
    [spvValidate](https://github.com/KhronosGroup/SPIRV-Tools#validator).
-   `TVM_VULKAN_DEBUG_SHADER_SAVEPATH` - A path to a directory. If set
    to a non-empty string, the Vulkan codegen will save tir, binary
    SPIR-V, and disassembled SPIR-V shaders to this directory, to be
    used for debugging purposes.
