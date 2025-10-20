---

title: tvm.ir

---


适用于所有 IR 变体的通用数据结构


**类：**

|[Attrs](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#classtvmirattrs)|属性节点，主要用于定义运算符的属性|
|:----|:----|
|[DictAttrs](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirdictattrs)|字典属性|
|[EnvFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirenvfunc)|环境函数|
|[Node](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirnode)|所有 IR 节点的基类|
|[SourceName](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirsourcenamename)(name)|源位置的标识符|
|[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)(source_name, line, end_line, column, ...)|指定源程序中的位置|
|[SequentialSpan](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirsequentialspanspans)(spans)|源跨度序列|
|[Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany)(input_list)|数组容器|
|[Map](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirmapinput_dictmappinganyany)(input_dict)|映射容器|
|[BaseExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirbaseexpr)|所有表达式的基类|
|[GlobalVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none)(name_hint[, type_annot])|IR 中的全局变量|
|[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)|所有原始表达式的基类|
|[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)(begin[, end, span])|表示 TVM 中的范围|
|[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)|所有非原始表达式的基类|
|[BaseFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#classtvmirbasefunc)|所有函数的基类|
|[CallingConv](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmircallingconvvalue)(value)|可能的调用约定种类|
|[GlobalInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirglobalinfo)|所有可出现在 IR 中的全局信息的基础节点|
|[DummyGlobalInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirdummyglobalinfo)()||
|[VDevice](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirvdevicetargetnone-vdevice_idint-0-memory_scopestr-global)([target, vdevice_id, memory_scope])||
|[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)([functions, attrs, global_infos])|包含函数和类型定义的 IRModule|
|[Op](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirop)()|IR 中的原始运算符|
|[FuncType](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirfunctypearg_types-ret_type)(arg_types, ret_type)|函数类型|
|[PointerType](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirpointertypeelement_type-storage_scope)(element_type[, storage_scope])|低级 TIR 中使用的 PointerType|
|[PrimType](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimtypedtype)(dtype)|低级 IR 中的原始数据类型|
|[TupleType](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirtupletypefields)(fields)|元组值的类型|
|[Type](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirtype)|所有类型的基类|


**函数：**

|[make_node](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#tvmirmake_nodetype_key-kwargs)(type_key, **kwargs)|根据类型键和字段创建一个新的 IR 节点|
|:----|:----|
|[assert_structural_equal](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#tvmirassert_structural_equallhsrhsmap_free_varsfalse)(lhs, rhs[, …])|断言 lhs 和 rhs 在结构上彼此相等|
|[load_json](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#tvmirload_jsonjson_strobject)(json_str)|从 json_str 加载 tvm 对象|
|[save_json](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#tvmirsave_jsonnode--str)(node)|将 tvm 对象保存为 json 字符串|
|[structural_equal](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#tvmirstructural_equallhsrhsmap_free_varsfalse)(lhs, rhs[, map_free_vars])|检查 lhs 和 rhs 的结构相等性|
|[structural_hash](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#tvmirstructural_hashnodemap_free_varsfalse)(node[, map_free_vars])|计算节点的结构哈希|
|[register_intrin_lowering](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#tvmirregister_intrin_loweringop_nametargetfnonelevel10)(op_name, target, *)|寄存器操作数降低函数|
|[register_op_attr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#tvmirregister_op_attrop_nameattr_keyvaluenonelevel10)(op_name, attr_key[, value, …])|通过名称注册一个运算符的运算符属性|

## ***class*tvm.ir.Attrs**

属性节点，主要用于定义运算符的属性。


由在 Python 端注册的函数使用，例如 compute、schedule 和 alter_layout。Attrs 作为第一个参数传递给这些函数。


**方法：**

|[get_int_tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#get_int_tuplekey)(key)|获取键的 Python int 元组|
|:----|:----|
|[get_int](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#get_intkey)(key)|获取某个键的 Python int 值|
|[get_str](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#get_strkey)(key)|获取某个键的 Python int 值|

### **get_int_tuple(*key*)**
获取键的 Python 整数元组。
* **参数:         key** ([str](https://docs.python.org/3/library/stdtypes.html#str))
* **返回:         value  值**
* **返回类型:**[int](https://docs.python.org/3/library/functions.html#int)[元组](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)



### **get_int(*key*)**
获取键的 Python int 值。
* **参数:          key** ([str](https://docs.python.org/3/library/stdtypes.html#str))
* **返回:          value  值**
* **返回类型：**[int](https://docs.python.org/3/library/functions.html#int)


### **get_str(*key*)**

获取键的 Python 整数值。
* **参数:          key** ([str](https://docs.python.org/3/library/stdtypes.html#str))
* **返回:          value  值**
* **返回类型:**[int](https://docs.python.org/3/library/functions.html#int)


## *class* tvm.ir.DictAttrs
字典属性。


**方法：**

|[keys](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#keys)()|获取属性中的名称列表|
|:----|:----|
|[get](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#getkey-defaultnone)(key[, default])|获取具有默认值的元素|
|[items](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#items)()|从映射上获取项目|

### keys()

获取属性中的名称列表。
* **返回：keys**  – 键列表
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str) 列表

### get(*key*, *default=None*)

获取具有默认值的元素。

### items()

从映射上获取项目。

## tvm.ir.make_node(*type_key*, **kwargs*)

根据类型键和字段创建一个新的 IR 节点。
* **参数：**
   * **type_key** ([str](https://docs.python.org/3/library/stdtypes.html#str))：节点的类型键
   * ***kwargs** ([dict](https://docs.python.org/3/library/stdtypes.html#dict)) ：节点的字段
* **返回：node** ：相应的 IR 节点
* **返回类型：**[Node](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirnode)


:::Note

如果创建的节点是 AttrsNode 的实例，那么创建函数还将运行 Attrs 支持的边界检查和默认值设置

:::


示例


以下代码构造一个 IntImm 对象

```python
x = tvm.ir.make_node("IntImm", dtype="int32", value=10, span=None)
assert isinstance(x, tvm.tir.IntImm)
assert x.value == 10
```
## *class* tvm.ir.EnvFunc

环境函数。


这是一个全局函数对象，可以通过其名称进行序列化


**方法：**

|[get](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#static-getname)(name)|获取静态环境函数|
|:----|:----|

### *static* get(*name*)
获取静态环境函数
* **参数：name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：函数的名称


## *class* tvm.ir.Node
所有 IR 节点的基类。

## *class* tvm.ir.SourceName(*name*)

源位置的标识符。
* **参数：name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：源的名称


## *class* tvm.ir.Span(*source_name*, *line*, *end_line*, *column*, *end_column*)

指定源程序中的位置。
* **参数：**
   * **source** ([SourceName](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirsourcenamename))  – 源名称
   * **lineno** ([int](https://docs.python.org/3/library/functions.html#int))  – 行号
   * **col_offset** ([int](https://docs.python.org/3/library/functions.html#int))  – 位置的列偏移量

## *class* tvm.ir.SequentialSpan(*spans*)

源跨度序列。


此跨度特定于一个表达式，它来自 IR 变换后的多个表达式。
* **参数：spans** ([Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany))：跨度数组。 



## **tvm.ir.assert_structural_equal(*lhs*,*rhs*,*map_free_vars=False*)**

断言 lhs 和 rhs 在结构上彼此相等。
* **参数：**
   * **lhs** (*Object*)  – 左操作数
   * **rhs** (*Object*)  – 左操作数
   * **map_free_vars** ([bool](https://docs.python.org/3/library/functions.html#bool))  – 我们是否应该将不受任何定义约束的自由变量映射为彼此相等



:raises ValueError: 如果断言条件不成立，则抛出 ValueError 异常。


:::info 另见

`structural_equal`

:::

## **tvm.ir.load_json(*json_str*)→ Object**

从 json_str 加载 tvm 对象。
* **参数：json_str** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：json 字符串
* **返回：node** ：已加载的 tvm 节点
* **返回类型：** 对象


## tvm.ir.save_json(*node*) → [str](https://docs.python.org/3/library/stdtypes.html#str)

将 tvm 对象保存为 json 字符串。
* **参数：** 节点 (对象)  – 要保存的 TVM 对象
* **返回：json_str**  – 已保存的 json 字符串
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)


## **tvm.ir.structural_equal(*lhs*,*rhs*,*map_free_vars=False*)**

检查 lhs 和 rhs 的结构相等性。


结构相等性在 IRNodes 的 DAG 中以递归方式定义。有两种类型的节点：
* 图节点：lhs 中的图节点只能映射为 rhs 中的一个且仅一个图节点
* 普通节点：相等性是递归定义的，不受图节点的限制


Vars(tir::Var,relax::Var) 是图节点


如果满足以下条件之一，则 var 类型节点（例如 tir::Var）可以映射为等于具有相同类型的另一个 var：
* 它们出现在同一个定义点（例如函数参数）
* 它们通过 same_as 关系指向同一个 VarNode
* 它们出现在同一个使用点，并且 map_free_vars 设置为True


var 的规则用于重新映射函数参数和 let 绑定中出现的变量
* **参数：**
   * **lhs** (*Object*)  – 左操作数
   * **rhs** (*Object*)  – 左操作数
   * **map_free_vars** ([bool](https://docs.python.org/3/library/functions.html#bool))  – 自由变量（即没有定义站点的变量）是否应映射为彼此相等
* **返回：result**  – 比较结果
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)


:::info 另见

`structural_hash`, `assert_strucural_equal`

:::


## **tvm.ir.structural_hash(*node*,*map_free_vars=False*)**

计算节点的结构哈希。


结构哈希值在 IRNodes 的 DAG 中递归定义。节点有两种：
* 普通节点：哈希值仅由其内容和类型定义
* 图节点：每个图节点将被分配一个唯一的索引，该索引按访问期间首次出现的顺序排序。图节点的哈希值由其内容的哈希值和索引的哈希值组合而成


structural_hash 与 structural_equal 保持一致。如果两个节点在结构上相等，则它们的结构哈希（具有相同的 map_free_vars 选项）也应该相等


如果两个节点的结构哈希彼此相等，那么这两个节点在结构上彼此相等的可能性很高（除了罕见的哈希值冲突情况）
* **参数：**
   * **node** (对象)  – 要进行散列的输入
   * **map_free_vars** ([bool](https://docs.python.org/3/library/functions.html#bool))  – 如果 map_free_vars 设置为 true，我们将按照自由变量出现的顺序对其进行哈希处理。否则，我们将根据它们在内存中的指针地址对其进行哈希处理
* **返回：result**  – 哈希结果
* **返回类型：**[int](https://docs.python.org/3/library/functions.html#int)


:::info 另见

`structrual_equal`

:::

## *class* tvm.ir.Array(*input_list:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*]*)

数组容器。

## *class* tvm.ir.Map(*input_dict:*[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)*[*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*,*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*]*)

映射容器。


**方法：**

|[keys](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#keys--a-set-like-object-providing-a-view-on-ds-keys)()|[values](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#values--an-object-providing-a-view-on-ds-values)（）|
|:----|:----|
|[values](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#values--an-object-providing-a-view-on-ds-values)()||
|[items](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#items-1)()|从映射中获取项目|
|[get](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#getkey-defaultnone-1)(key[, default])|获取具有默认值的元素|

### keys() → a set-like object providing a view on D's keys
### values() → an object providing a view on D's values

### items()
从映射中获取项目。

### get(*key*, *default=None*)
获取具有默认值的元素。
* **参数：**
   * **key** ([object](https://docs.python.org/3/library/functions.html#object))  – 属性键
   * **默认** ([object](https://docs.python.org/3/library/functions.html#object))  – 默认对象
* **返回：value**  – 结果值
* **返回类型：**[object](https://docs.python.org/3/library/functions.html#object)

## *class* tvm.ir.BaseExpr
所有表达式的基类。

## *class* tvm.ir.GlobalVar(*name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str), *type_annot:*[Type](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimtypedtype)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)

IR 中的全局变量。


GlobalVar 用于引用存储在 IRModule 中的全局函数。
* **参数：name_hint** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 变量的名称

## *class* tvm.ir.PrimExpr

所有原始表达式的基类。


PrimExpr 用于低级代码优化和整数分析。

## *class* tvm.ir.Range(*begin:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *end:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)

表示 TVM 中的范围。


您无需显式创建 Range。Python 列表和元组将在 API 函数中自动转换为 Range。
* **参数：**
   * **begin** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))  – 当 end 为 None 时，为范围的起始值。否则，为范围的长度
   * **end** (*Optional[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)  – 范围的结束值
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)  – 此节点在源代码中的位置



:::Note

如果 `end` 参数不为 `None`，构造函数会创建范围 `[begin, end)`。否则，它会创建范围 `[0, begin)`

:::


方法：

|[from_min_extent](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#static-from_min_extentmin_valueprimexpr-extentprimexpr-spanspannone-none--range)(min_value, extent[, span])|通过最小值和范围构造一个 Range|
|:----|:----|

### *static* from_min_extent(*min_value:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *extent:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)

通过最小值和范围构建范围。


这构建了一个范围 [min_value, min_value + extent)。
* **参数：**
   * **min_value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))  – 范围的最小值
   * **extent** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))  – 范围的范围
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)  – 此节点在源代码中的位置
* **返回：rng**  – 构建的范围
* **返回类型：**[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)

## *class* tvm.ir.RelaxExpr
所有非原始表达式的基类。


**属性：**

|[struct_info](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#propertystruct_infostructinfonone)|Get the struct info field获取结构信息字段|
|:----|:----|

### ***property*struct_info*:***[StructInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)***|***[None](https://docs.python.org/3/library/constants.html#None)

获取结构体信息字段
* **返回：** struct_info – 如果可用，则为结构信息。
* **返回类型：**[tvm.relax.StructInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)


## ***class*tvm.ir.BaseFunc**
所有函数的基类。


属性：

|[attrs](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#propertyattrs)|返回函数的 attrs 成员。|
|:----|:----|


**方法：**

|[with_attr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#with_attrattr_key_or_dictattr_valuenonebasefunc)(attr_key_or_dict[, attr_value])|创建函数的新副本并更新属性。|
|:----|:----|
|[with_attrs](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#with_attrsattr_mapdictattrsdictstrobjectbasefunc)(attr_map)|复制 IRModule 并将给定的属性映射添加到其中。|
|[without_attr]()(attr_key)|创建一个带有未提供键的属性的新函数副本。|


### ***property*attrs**

返回函数的 attrs 成员。


### **with_attr(*attr_key_or_dict*,*attr_value=None*)→**[BaseFunc](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.BaseFunc)

创建函数的新副本并更新属性。
* **参数：**
   * attr_key_or_dict (Union[str, dict]) – 用于指定属性键或包含多个键值对字典。
   * attr_value (对象) –- 新的属性值。
* **返回：** func –  函数的新副本
* **返回类型:** [BaseFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#classtvmirbasefunc)


### **with_attrs(*attr_map:***[DictAttrs](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.DictAttrs)***|***[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***, Object]*)→**[BaseFunc](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.BaseFunc)

复制 IRModule 并添加给定的属性映射到它。 :param attr_map: 属性映射 :type attr_map: Union[DictAttrs, Dict[str, Object]]
* **返回：** func – 函数的新副本
* **返回类型：** [BaseFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#classtvmirbasefunc)

### **without_attr(*attr_key:***[str](https://docs.python.org/3/library/stdtypes.html#str)***)→**[BaseFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#classtvmirbasefunc)

创建函数的新副本，其中包含一个没有提供键的属性。
* **参数：** attr_key (str) – 要从属性对中删除的属性键。
* **返回：** func – 函数的新副本
* **返回类型：** [BaseFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#classtvmirbasefunc)


## *class* tvm.ir.CallingConv(*value*)

可能的调用约定种类。

## *class* tvm.ir.GlobalInfo

所有可出现在 IR 中的全局信息的基础节点。

**方法：**

|[same_as](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#same_asother)（其他）|使用结构等式重载。|
|:----|:----|

### **same_as(*other*)**

用结构相等性进行重载。



## *class* tvm.ir.DummyGlobalInfo

## *class* tvm.ir.VDevice(*target=None*, *vdevice_id:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *memory_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'global'*)

## *class* tvm.ir.IRModule(*functions=None*, *attrs=None*, *global_infos=None*)

包含函数和类型定义的 IRModule。


IRModule 是整个堆栈中所有 IR 转换的基本单元。
* **参数：functions** (*Optional**[***[dict](https://docs.python.org/3/library/stdtypes.html#dict)***]****.*)  – 全局变量到 BaseFunc 的映射


**方法:**

|[functions_items](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#functions_items)()|按字母顺序获取 self.functions.items() 中的项。|
|:----|:----|
|[update](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#updateother)(other)   **update** (其他)|将其他模块中的函数插入当前模块。|
|[update_func](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#update_funcvar-func)(var, func)   **update_func** (变量, 函数)|更新模块中对应全局变量的函数。|
|[update_global_info](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#update_global_infoname-global_info)(name, global_info)|更新模块中的全局信息。|
|[get_global_var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#get_global_varname)(name)   **get_global_var** (名称)|通过名称在函数中获取一个全局变量。|
|[get_global_vars](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#get_global_vars)()|收集在此模块中定义的所有全局变量。|
|[replace_global_vars](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#replace_global_varsreplacementsdictstrglobalvarstrglobalvar--irmodule)(replacements)   **replace_global_vars** (替换)|替换模块中的 GlobalVar 实例。|
|[from_expr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#static-from_exprexpr-functionsnone)(expr[, functions])|从一个独立的表达式中构建一个模块。|
|[get_attr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#get_attrattr_key)(attr_key)|获取 IRModule 属性。|
|[with_attr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#with_attrattr_key-attr_value)(attr_key, attr_value)|复制 IRModule 并为其添加一个属性。|
|[without_attr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#without_attrattr_keystr--irmodule)(attr_key)|复制 IRModule 并删除一个属性键及其关联的值。|
|[with_attrs](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#with_attrsattr_mapdictattrsdictstr-object--irmodule)(attr_map)|复制 IRModule 并将给定的属性映射添加到其中。|


### **functions_items()**

按字母顺序获取 self.functions.items() 中的项。
* **返回：** items – items 函数。
* **返回类型：** List[[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)[[GlobalVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none), [Function](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxfunctionparamslistvarbodyrelaxexprret_struct_infostructinfononenoneis_pureboolnonetrueattrsdictattrsnonenonespanspannonenone)]]

### update(*other*)
将另一个模块中的函数插入当前模块。
* **参数：other** ([IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone))  – 要合并到当前模块的模块

### update_func(*var*, *func*)

更新模块中某个全局变量对应的函数。
* **参数：**
   * **var** ([GlobalVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none))  – 全局变量
   * **func** ([tvm.ir.BaseFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#classtvmirbasefunc))  – 要插入的函数

### update_global_info(*name*, *global_info*)

更新模块中的全局信息。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 全局信息的名称
   * **global_info** (*List[*[GlobalInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirglobalinfo)*]*)  – 需要更新的全局信息

### get_global_var(*name*)

通过名称获取函数中的全局变量。
* **参数：name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 全局变量的名称
* **返回：global_var**  – 映射到的全局变量`name`
* **返回类型：**[GlobalVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none)
* **抛出：tvm.error.TVMError 找不到相应的全局变量时抛出。**

### get_global_vars()

收集此模块中定义的所有全局变量。
* **返回：global_vars**  – 全局变量数组
* **返回类型：**[Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany)[[GlobalVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none)]

## replace_global_vars(*replacements:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[GlobalVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[GlobalVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none)*]*) → [IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)

替换模块内的 GlobalVar 实例。


在 IRModule 中替换 GlobalVar。由于 IRModule 可能包含对 GlobalVar 的内部引用（无论是在 TIR 中还是在 Relax 中），因此在替换或重命名 GlobalVar 时都应使用此方法。
* **参数：**
   * **replacements** (*Dict[Union[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,_ expr.GlobalVar],_ *Union[[str](https://docs.python.org/3/library/stdtypes.html#str),_ expr.GlobalVar]_*]*)  – 一个字典，其中每个键都是一个要替换的 GlobalVar，相应的值是用来替换它的 GlobalVar
* **返回：** 更新后的模块
* **返回类型：**[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)

### *static* from_expr(*expr*, *functions=None*)

从独立表达式构建模块。
* **参数：**
   * **expr** ([RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr))  – 起始表达式
   * **global_funcs** (*Optional[*[dict](https://docs.python.org/3/library/stdtypes.html#dict)*]*)  – 全局变量到函数定义的映射
* **返回：mod**  – 包含传递的定义的模块，其中 expr 被设置为入口点（如果需要，可以包装在函数中）
* **返回类型：** Module

### get_attr(*attr_key*)

获取 IRModule 属性。
* **参数：attr_key** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 属性键
* **返回：attr_value**  – 属性值
* **返回类型：** Any

### with_attr(*attr_key*, *attr_value*)

复制 IRModule 并向其添加属性。
* **参数：**
   * **attr_key** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 属性键
   * **attr_value** (*Object*)  – 新的属性值
* **返回：mod**  – 具有属性的 IRModule 的新拷贝
* **返回类型：**[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)

### without_attr(*attr_key:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnonee)

复制 IRModule 并删除属性键及其关联值。:param attr_key: 属性键。:type attr_key: str
* **返回：mod**  – 没有属性的 IRModule 的新拷贝
* **返回类型：**[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)

### with_attrs(*attr_map:*[DictAttrs](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.DictAttrs)*|*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object]*) → [IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)

复制 IRModule 并将给定的属性映射添加到其中。：param attr_map：属性映射：type attr_map：Union[DictAttrs, Dict[str, Object]]
* **返回：mod**  – 具有属性的 IRModule 的新拷贝
* **返回类型：**[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)

## *class* tvm.ir.Op

IR 中的原始运算符。


**方法：**

|[get](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Op.get)(op_name)|获取给定名称的 Op。|
|:----|:----|
|[get_attr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#get_attrattr_name)(attr_name)|获取有关操作员的附加属性。|
|[has_attr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#has_attrattr_name)(attr_name)|检查操作员是否具有附加属性。|
|[set_attr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#set_attrattr_name-value-plevel10)(attr_name, value[, plevel])|设置有关操作员的属性。|
|[reset_attr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#reset_attrattr_name)(attr_name)|重置有关操作员的属性。|
|[add_argument](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#add_argumentname-type-description)(name, type, description)|向函数添加参数信息。|
|[set_support_level](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#set_support_levellevel)(level)|设置op的支持级别。|
|[set_num_inputs](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#set_num_inputsn)(n)|设置op的支持级别。|
|[set_attrs_type_key](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#set_attrs_type_keykey)(key)|设置op的属性类型键。|
|[list_op_names](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#static-list_op_names)()|列出 op 注册表中的所有 op 名称。|

### *static* get(*op_name*)
获取给定名称的 Op。
* **参数：op_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 操作名称
* **返回：op**  – 对应名称的 op
* **返回类型：**[Op](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirop)

### get_attr(*attr_name*)

获取有关操作员的附加属性。
* **参数：attr_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 属性名称
* **返回：value**  – 属性值
* **返回类型：**[object](https://docs.python.org/3/library/functions.html#object)

### has_attr(*attr_name*)

检查操作员是否具有附加属性。
* **参数：attr_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 属性名称
* **返回：value**  – 运算符是否具有附加属性
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)

### set_attr(*attr_name*, *value*, *plevel=10*)

设置有关操作员的属性。
* **参数：**
   * **attr_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 属性名称
   * **value** ([object](https://docs.python.org/3/library/functions.html#object))  – 属性值
   * **plevel** ([int](https://docs.python.org/3/library/functions.html#int))  – 优先级

### reset_attr(*attr_name*)

重置有关操作员的属性。
* **参数：attr_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 属性名称

### add_argument(*name*, *type*, *description*)

向函数添加参数信息。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 参数名称
   * **type** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 参数类型
   * **description** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 参数描述

### set_support_level(*level*)

设置op的支持级别。
* **参数：level** ([int](https://docs.python.org/3/library/functions.html#int))  – 支持级别

### set_num_inputs(*n*)
设置op的支持级别。
* **参数：n** ([int](https://docs.python.org/3/library/functions.html#int))  – 输入数字

### set_attrs_type_key(*key*)

设置op的属性类型键。
* **参数：key** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 类型键

### *static* list_op_names()

列出 op 注册表中的所有 op 名称。
* **返回：value**  – 注册的操作名称
* **返回类型：** List[[str](https://docs.python.org/3/library/stdtypes.html#str)]

## **tvm.ir.register_intrin_lowering(*op_name*,*target*,***,*f=None*,*level=10*)**

注册 Op 降低函数。
* **参数：**
   * **op_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 操作名称
   * **target** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 给定内在降低函数的目标字符串
   * **f** (*function*,_ 可选)  – 要注册的函数
   * **level** ([int](https://docs.python.org/3/library/functions.html#int))  – 优先级
* **返回：fregister**  – 如果未指定 f，则注册操作降低函数
* **返回类型：** function

## **tvm.ir.register_op_attr(*op_name*,*attr_key*,*value=None*,*level=10*)**

通过名称注册操作数的属性。
* **参数：**
   * **op_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 运算符的名称
   * **attr_key** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 属性名称。
   * **value** ([object](https://docs.python.org/3/library/functions.html#object),_ 可选)  – 要设置的值
   * **level** ([int](https://docs.python.org/3/library/functions.html#int),_ 可选)  – 优先级
* **返回：fregister**  – 如果未指定值，则注册函数
* **返回类型：** function

## *class* tvm.ir.FuncType(*arg_types*, *ret_type*)

函数类型。


一个函数类型由以下部分组成：一组类型参数，用于定义泛型函数；一组类型约束（暂时省略）；一系列参数类型；以及返回类型。
* **参数：**
   * **arg_types** (*List[*[tvm.ir.Type](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Type)*]*)  – 参数类型
   * **ret_type** ([tvm.ir.Type](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Type))  – 返回类型

## *class* tvm.ir.PointerType(*element_type*, *storage_scope=''*)

低级 TIR 中使用的 PointerType。
* **参数：**
   * **element_type** ([tvm.ir.Type](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Type))  – 指针元素的类型
   * **storage_scope** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 指针寻址的存储范围

## *class* tvm.ir.PrimType(*dtype*)

低级 IR 中的原始数据类型。
* **参数：dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 运行时数据类型与 primtype 相关

## *class* tvm.ir.TupleType(*fields*)

元组值的类型。
* **参数：fields** (*List[*[Type](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Type)*]*)  – 元组中的字段

## *class* tvm.ir.Type

所有类型的基类。


**方法：**

|[same_as](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#same_asother-1)(other)|通过引用相等性比较两种 TVM 类型|
|:----|:----|

### same_as(*other*)

通过引用相等性比较两种 TVM 类型。

