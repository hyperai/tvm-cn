---

title: tvm.target

---


目标描述和 codgen 模块。


TVM 的目标字符串格式为。`<target_kind> [-option=value]...`。


:::note

选项列表包括：
* **-device=<device name>**
设备名称。
* **-mtriple=<target triple>** 指定目标三元组，这对于交叉编译很有用。
* **-mcpu=<cpuname>** 指定当前架构中的特定芯片以生成代码。默认情况下，这是从目标三元组推断并自动检测到当前架构的。
* **-mattr=a1,+a2,-a3,…** 覆盖或控制目标的具体属性，例如是否启用 SIMD 操作。默认属性集由当前 CPU 设置。
* **-mabi=<abi>** 为指定的 ABI 生成代码，例如「lp64d」。
* **-system-lib** 构建 TVM 系统库模块。系统库是一个全局模块，在程序启动时包含自注册的函数。用户可以使用 tvm.runtime.system_lib 获取该模块。在禁止使用动态加载 API（如 dlopen）的环境中，它非常有用。只要程序链接了结果代码，系统库就会可用。
* 我们可以使用 `tvm.target.Target()` 从目标字符串创建一个 tvm.target.Target。我们还可以使用本模块中的其他特定函数来创建特定的目标。

:::

**类：**

|[Target](/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone)(target[, host])|目标设备信息，通过TVM API使用。|
|:----|:----|
|[TargetKind](/docs/api-reference/python-api/tvm-target#class-tvmtargettargetkind)|一种编译目标。|
|[VirtualDevice](/docs/api-reference/python-api/tvm-target#classtvmtargetvirtualdevicedevicenonetargetnonememory_scope)([device, target, memory_scope])|编译时表示数据在运行时的存储位置以及如何编译代码来计算它。|


**函数：**

|[create](/docs/api-reference/python-api/tvm-target#tvmtargetcreatetarget)(target)|已弃用。|
|:----|:----|
|[cuda](/docs/api-reference/python-api/tvm-target#tvmtargetcudamodelunknown-archnone-optionsnone)([model, arch, options])|返回一个 cuda 目标。|
|[rocm](/docs/api-reference/python-api/tvm-target#tvmtargetrocmmodelunknown-optionsnone)([model, options])|返回 ROCM 目标。|
|[mali](/docs/api-reference/python-api/tvm-target#tvmtargetmalimodelunknown-optionsnone)([model, options])|返回 ARM Mali GPU 目标。|
|[intel_graphics](/docs/api-reference/python-api/tvm-target#tvmtargetintel_graphicsmodelunknown-optionsnone)([model, options])|返回 Intel Graphics 目标。|
|[arm_cpu](/docs/api-reference/python-api/tvm-target#tvmtargetarm_cpumodelunknown-optionsnone)([model, options])|返回 ARM CPU 目标。|
|[rasp](/docs/api-reference/python-api/tvm-target#tvmtargetraspoptionsnone)([options])|返回 Raspberry 3b 目标。|
|[bifrost](/docs/api-reference/python-api/tvm-target#tvmtargetbifrostmodelunknown-optionsnone)([model, options])|返回 ARM Mali GPU 目标（Bifrost 架构）。|
|[riscv_cpu](/docs/api-reference/python-api/tvm-target#tvmtargetriscv_cpumodelsifive-u54-optionsnone)([model, options])|返回 RISC-V CPU 目标。|
|[hexagon](/docs/api-reference/python-api/tvm-target#tvmtargethexagoncpu_verv68-kwargs)([cpu_ver])|返回 Hexagon 目标。|
|[stm32](/docs/api-reference/python-api/tvm-target#tvmtargetstm32seriesunknown-optionsnone)([series, options])|返回 STM32 目标。|
|[adreno](/docs/api-reference/python-api/tvm-target#tvmtargetadrenomodelunknown-optionsnone-clmlfalse)([model, options, clml])|返回 Qualcomm GPU 目标。|
|[make_compilation_config](/docs/api-reference/python-api/tvm-target#tvmtargetmake_compilation_configctxttargettarget_hostnone)(ctxt, target[, …])|返回适用于 target 和 target_host 的 CompilationConfig，使用与标准构建接口相同的表示约定。|
|[list_tags](/docs/api-reference/python-api/tvm-target#tvmtargetlist_tagsdictstrtargetnone)()|返回标签字典，将每个标签名称映射到其对应的目标。|

## *class* tvm.target.Target(*target*, *host=None*)


目标设备信息，通过 TVM API 使用。


:::note

您可以使用构造函数或以下函数创建目标：
* `tvm.target.arm_cpu()`创建 arm_cpu 目标。
* `tvm.target.cuda()`创建 CUDA 目标。
* `tvm.target.rocm()`创建 ROCM 目标。
* `tvm.target.mali()`创建 Mali 目标。
* `tvm.target.intel_graphics()`创建 Intel Graphics 目标。

:::


**方法：**

|[from_device](/docs/api-reference/python-api/tvm-target#static-from_devicedevicestr-device--target)(device)|检测与给定设备关联的目标。|
|:----|:----|
|[current](/docs/api-reference/python-api/tvm-target#static-currentallow_nonetrue)([allow_none])|返回当前目标。|
|[get_kind_attr](/docs/api-reference/python-api/tvm-target#get_kind_attrattr_name)(attr_name)|获取有关目标类型的附加属性。|
|[get_target_device_type](/docs/api-reference/python-api/tvm-target#get_target_device_type)()|返回此目标的设备类型。|
|[list_kinds](/docs/api-reference/python-api/tvm-target#static-list_kinds)()|返回可用目标名称的列表。|
|[canon_target](/docs/api-reference/python-api/tvm-target#static-canon_targettarget)(target)|给定一个类似目标的对象，返回代表它的 TVM Target 对象。|
|[canon_target_and_host](/docs/api-reference/python-api/tvm-target#static-canon_target_and_hosttarget-target_hostnone)(target[, target_host])|返回一个 TVM Target，用于表示 target 和 target_host。|
|[canon_multi_target](/docs/api-reference/python-api/tvm-target#static-canon_multi_targetmulti_targets)(multi_targets)|给定一个类似目标的对象，或者类似目标对象的类似集合的对象，返回代表该对象的 TVM 目标对象的 TVM 数组。|
|[canon_multi_target_and_host](/docs/api-reference/python-api/tvm-target#static-canon_multi_target_and_hosttarget-target_hostnone)(target[, …])|返回一个 TVM Array<Target>，用于表示 target 和 target_host。|
|[canon_target_map_and_host](/docs/api-reference/python-api/tvm-target#static-canon_target_map_and_hosttarget_map-target_hostnone)(target_map[, …])|将 target_map 作为从 TVM Target 的规范形式到 IRModules 的映射返回。|
|[target_or_current](/docs/api-reference/python-api/tvm-target#static-target_or_currenttarget)(target)|返回目标，如果目标为 None，则返回环境中的当前目标。|


**属性：**

|[arch](/docs/api-reference/python-api/tvm-target#property-arch)|如果存在，则返回目标的 cuda arch。|
|:----|:----|
|[max_num_threads](/docs/api-reference/python-api/tvm-target#property-max_num_threads)|如果存在，则返回目标的 max_num_threads。|
|[max_block_size_x](/docs/api-reference/python-api/tvm-target#property-max_block_size_x)|如果存在，则返回目标 x 维度上的最大块大小。|
|[max_block_size_y](/docs/api-reference/python-api/tvm-target#property-max_block_size_y)|如果存在，则返回目标 y 维度上的最大块大小。|
|[thread_warp_size]()|如果存在，则返回目标的thread_warp_size。|
|[model](/docs/api-reference/python-api/tvm-target#property-model)|如果存在，则返回目标模型。|
|[mcpu](/docs/api-reference/python-api/tvm-target#property-mcpu)|如果存在，则返回目标的 mcpu。|
|[mattr](/docs/api-reference/python-api/tvm-target#property-mattr)|如果存在，则返回目标的 mattr。|

### *static* from_device(*device:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| Device*) → [Target](/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone)


检测与指定设备关联的目标。如果设备不存在，则会引发错误。
* **参数：dev** (*Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Device****]*)  – 用于检测目标的设备。支持的设备类型：[“cuda”, “metal”, “rocm”, “vulkan”, “opencl”, “cpu”]。
* **返回：target**  – 检测到的目标。
* **返回类型：**[Target](/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnonet)。


### *static* current(*allow_none=True*)

返回当前目标。
* **参数：allow_none** ([bool](https://docs.python.org/3/library/functions.html#bool))  – 是否允许当前目标为无。
* **Raises:** 如果当前目标未设置，则抛出 ValueError。

### *property* arch

如果存在，则返回目标的 cuda arch。


### *property* max_num_threads

如果存在，则返回目标的 max_num_threads。


### *property* max_block_size_x

如果存在，则返回目标 x 维度上的最大块大小。


### *property* max_block_size_y

如果存在，则返回目标 y 维度上的最大块大小。


### *property* thread_warp_size

如果存在，则返回目标的 thread_warp_size。


### *property* model

如果存在，则返回目标模型。


### *property* mcpu

如果存在，则返回目标的 mcpu。


### *property* mattr

如果存在，则返回目标的 mattr。


### get_kind_attr(*attr_name*)

获取有关目标类型的附加属性。
* **参数：attr_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 属性名称。
* **返回：value**  – 属性值。
* **返回类型：**[object](https://docs.python.org/3/library/functions.html#object)。


### get_target_device_type()

返回此目标的设备类型。


### *static* list_kinds()

返回可用目标名称的列表。


### *static* canon_target(*target*)
给定一个类似目标的对象，返回代表它的 TVM Target 对象。可从以下对象转换：None（转换为 None）。现有的 TVM Target 对象。字符串，例如“cuda”或“cuda -arch=sm*80”。Python 字典，例如 {“kind”: “cuda”, “arch”: “sm*80” }。


### *static* canon_target_and_host(*target*, *target_host=None*)

返回 TVM Target 捕获 target 和 target_host。同时返回规范格式的地址。给定的目标可以是 Target.canon_target 可识别的任何格式。如果指定了 target_host，则 target.canon_target 可识别的任何格式。如果指定了 target_host，它将被设置为结果 Target 对象中的「host」（并发出警告）。


请注意，此方法不支持异构编译目标。


### *static* canon_multi_target(*multi_targets*)

给定一个类似目标的对象，或一个类似目标对象的集合对象，返回一个代表该对象的 TVM Target 对象的 TVM 数组。可从以下类型转换：None（转换为 None）。canon_target 可识别形式的单个类似目标的对象。canon_target 可识别形式的类似目标对象的 Python 列表或 TVM 数组。一个 Python 字典或 TVM 映射，用于将表示设备类型的 TVM IntImm 对象转换为 canon_target 可识别形式的类似目标的对象。（这是一种表示异构目标的传统方法。键会被忽略。）


### *static* canon_multi_target_and_host(*target*, *target_host=None*)

返回一个包含 target 和 target_host 的 TVM Array。给定的 target 可以是 Target.canon_multi_target 可识别的任何形式。如果指定了 target_host，则 target.canon_target 可以识别的任何形式。如果指定了 target_host，它将在每个结果 Target 对象中设置为「host」（并发出警告）。


### *static* canon_target_map_and_host(*target_map*, *target_host=None*)

返回 target_map，该映射以规范形式从 TVM Target 到 IRModules。输入 target_map 的键可以是 Target.canon_target 可识别的任何形式。同样，如果指定了 target_host，则 target_host 也可以是 Target.canon_target 可识别的任何形式。最终的 target_map 键将以规范形式捕获 target_host。同时，返回的 target_host 也以规范形式存在。


### *static* target_or_current(*target*)

返回目标，如果目标为 None，则返回环境中的当前目标。


## tvm.target.create(*target*)

已弃用。直接使用构造函数 `tvm.target.Target`。


## *class* tvm.target.TargetKind

一种编译目标。


**属性：**

|[options](/docs/api-reference/python-api/tvm-target#property-options)|返回可用选项名称和类型的字典。|
|:----|:----|


**方法：**

|[options_from_name](/docs/api-reference/python-api/tvm-target#static-options_from_namekind_namestr)(kind_name)|从 TargetKind 名称返回可用选项名称和类型的字典。|
|:----|:----|

### *property* options
返回可用选项名称和类型的字典。


### *static* options_from_name(*kind_name:*[str](https://docs.python.org/3/library/stdtypes.html#str))

从 TargetKind 名称返回可用选项名称和类型的字典。


## tvm.target.cuda(*model='unknown'*, *arch=None*, *options=None*)

返回一个 cuda 目标。
* **参数：**
   * **model** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：cuda 设备的型号（例如 1080ti）。
   * **arch** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：cuda 架构（例如 sm*61）。
   * **ptions** ([str](https://docs.python.org/3/library/stdtypes.html#str) *或*[str](https://docs.python.org/3/library/stdtypes.html#str)*列表*)  ：附加[选项](https://docs.python.org/3/library/stdtypes.html#str)。


## tvm.target.rocm(*model='unknown'*, *options=None*)

返回 ROCM 目标。
* **参数：**
   * **model** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：该设备的型号。
   * **options** ([str](https://docs.python.org/3/library/stdtypes.html#str) *或*[str](https://docs.python.org/3/library/stdtypes.html#str)*[列表](https://docs.python.org/3/library/stdtypes.html#list)*) ：附加选项。


## tvm.target.mali(*model='unknown'*, *options=None*)

返回 ARM Mali GPU 目标。
* **参数：**
   * **model** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：该设备的型号。
   * **options** ([str](https://docs.python.org/3/library/stdtypes.html#str) *或*[str](https://docs.python.org/3/library/stdtypes.html#str)*[列表](https://docs.python.org/3/library/stdtypes.html#list)*) ：附加选项。


## tvm.target.intel_graphics(*model='unknown'*, *options=None*)

返回 Intel Graphics 目标。
* **参数：**
   * **model** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：该设备的型号。
   * **options** ([str](https://docs.python.org/3/library/stdtypes.html#str) *或*[str](https://docs.python.org/3/library/stdtypes.html#str)*[列表](https://docs.python.org/3/library/stdtypes.html#list)*) ：附加选项。


## tvm.target.arm_cpu(*model='unknown'*, *options=None*)

返回 ARM CPU 目标。若没有预调操作参数，此函数也会下载。
* **参数：**
   * **model** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：arm 板的 SoC 名称或电话名称。
   * **options** ([str](https://docs.python.org/3/library/stdtypes.html#str) *或*[str](https://docs.python.org/3/library/stdtypes.html#str)*[列表](https://docs.python.org/3/library/stdtypes.html#list)*) ：附加选项。


## tvm.target.rasp(*options=None*)

返回 Raspberry 3b 目标。
* **参数：options** ([str](https://docs.python.org/3/library/stdtypes.html#str) *或*[str](https://docs.python.org/3/library/stdtypes.html#str)*[列表](https://docs.python.org/3/library/stdtypes.html#list)*) ：附加选项。


## tvm.target.bifrost(*model='unknown'*, *options=None*)

返回 ARM Mali GPU 目标（Bifrost 架构）。
* **参数：options** ([str](https://docs.python.org/3/library/stdtypes.html#str) *或*[str](https://docs.python.org/3/library/stdtypes.html#str)*[列表](https://docs.python.org/3/library/stdtypes.html#list)*) ：附加选项。

## tvm.target.riscv_cpu(*model='sifive-u54'*, *options=None*)


返回 RISC-V CPU 目标。默认值：sifive-u54 rv64gc
* **参数：**
   * **model** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：CPU 名称。
   * **options** ([str](https://docs.python.org/3/library/stdtypes.html#str) *或*[str](https://docs.python.org/3/library/stdtypes.html#str)*[列表](https://docs.python.org/3/library/stdtypes.html#list)*) ：附加选项。


## tvm.target.hexagon(*cpu_ver='v68'*, ***kwargs*)

返回 Hexagon 目标。
* **参数：**
   * **cpu_ver** ([str](https://docs.python.org/3/library/stdtypes.html#str)*(**default: "v68"****)*) ：用于代码生成的 CPU 版本。并非所有允许的 cpu str 都是有效的，LLVM 将会抛出错误。
   * **parameters** (*Recognized keyword*)。
   * **-----------------------------**
   * **hvx** ([int](https://docs.python.org/3/library/functions.html#int)*(**default: 128****)*) ：HVX 向量的大小（以字节为单位）。值为 0 表示禁用 HVX 代码生成。
   * **llvm_options** ([str](https://docs.python.org/3/library/stdtypes.html#str) *或*[str](https://docs.python.org/3/library/stdtypes.html#str)*[列表](https://docs.python.org/3/library/stdtypes.html#list)(**default: None****)*) ：用户定义的编译器参数。
   * **use_qfloat** ([bool](https://docs.python.org/3/library/functions.html#bool)*(**default: True for cpu_ver >= v68****,False otherwise)*) ：是否使用 QFloat HVX 指令。
   * **use_ieee_fp** ([bool](https://docs.python.org/3/library/functions.html#bool)*(**default: False****)*) ：是否使用 IEEE HVX 指令。
   * **num_cores** ([int](https://docs.python.org/3/library/functions.html#int)*(**default: 4****)*) ：HVX 线程数。此属性是元调度程序所必需的。
   * **vtcm_capacity** ([int](https://docs.python.org/3/library/functions.html#int)*(**default: 0****)*) ：Hexagon VTCM 容量限制。如果值为 0，则容量被视为无限制。
   * **Note** (HVX 中的浮点支持需要 LLVM 14 及以上版本)。


## tvm.target.stm32(*series='unknown'*, *options=None*)

返回 STM32 目标。
* **参数：**
   * **series** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：STM32 开发板系列的名称，例如 stm32H7xx 或 stm32F4xx。
   * **options** ([str](https://docs.python.org/3/library/stdtypes.html#str) *或*[str](https://docs.python.org/3/library/stdtypes.html#str)*[列表](https://docs.python.org/3/library/stdtypes.html#list)*) ：附加选项。


## tvm.target.adreno(*model='unknown'*, *options=None*, *clml=False*)

返回 Qualcomm GPU 目标。:param model: 此设备的型号:type model: str:param options: 附加选项:type options: str 或 str 列表。


## ***class*tvm.target.VirtualDevice(*device=None*,*target=None*,*memory_scope=''*)**

编译时数据存储位置和代码编译方式的表现形式，用于在运行时存储数据。


**属性：**


|[device_type_int](/docs/api-reference/python-api/tvm-target#propertydevice_type_int)|虚拟设备的类型。|
|:----|:----|
|[memory_scope](/docs/api-reference/python-api/tvm-target#propertymemory_scope)|关于内存的面积。|
|[target](/docs/api-reference/python-api/tvm-target#propertytarget)|描述如何为虚拟设备编译的目标。|
|[virtual_device_id](/docs/api-reference/python-api/tvm-target#propertyvirtual_device_id)|虚拟设备的设备 ID。|


### ***property*device_type_int**

虚拟设备的类型。


### ***property*memory_scope**

相对于虚拟设备存储数据的内存区域。


### ***property*target**
描述如何为虚拟设备编译的目标。


### ***property*virtual_device_id**

虚拟设备的设备 ID。


## **tvm.target.make_compilation_config(*ctxt*,*target*,*target_host=None*)**

返回适用于 target 和 target_host 的 CompilationConfig，使用与标准构建接口相同的表示约定。仅用于单元测试。


## **tvm.target.list_tags()→**[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[Target](/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone)**] |**[None](https://docs.python.org/3/library/constants.html#None)

返回一个包含标签的字典，将每个标签名映射到其对应的目标。
* **返回:** tag_dict：标签字典，将每个标签名映射到其对应的目标。如果 TVM 以仅运行时模式构建，则为 None。
* **返回类型:** Optional[Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), [Target](/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone)]]。
