---

title: tvm.tir

---


张量级 IR 的命名空间。 


## *class* tvm.tir.Buffer

TVM 中的符号数据缓冲区。


缓冲区提供了一种在 TVM 中表示数据结构的数据布局特化的方法。


不要直接构造，而是使用 `decl_buffer()`。有关更多详细信息，请参阅文档`decl_buffer()`。


:::info 另见


`decl_buffer`

声明缓冲区。

:::


### access_ptr(*access_mask*, *ptr_type='handle'*, *content_lanes=1*, *offset=0*, *extent=None*)


获取缓冲区头部的访问指针。


这是与外部函数交互时获取缓冲区数据 ptress 的推荐方法。
* **参数：**
   * **access_mask** ([int](https://docs.python.org/3/library/functions.html#int))：访问模式 MASK。指示访问是读取还是写入数据内容。
   * **ptr_type** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：结果指针的数据类型。除非我们要将指针转换为特定类型，否则无需指定。
   * **content_lanes** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)：数据类型的通道数。对于矢量类型，此值大于 1。
   * **offset** (*Expr,optional*)：指针的偏移量。我们可以使用它来偏移 ptr 地址的元素数量。
   * **extent** (*Expr,optional*)：指针的范围。


**示例**

```python
# 获取用于读取的访问指针
buffer.access_ptr("r")
# 获取带位掩码的读/写访问指针
buffer.access_ptr(Buffer.READ | Buffer.WRITE)
# 获取带字符串标志的读/写访问指针
buffer.access_ptr("rw")
# 获取带偏移量的读取访问指针
buffer.access_ptr("r", offset = 100)
# 获取带范围（extent）的读取访问指针
buffer.access_ptr("r", extent = 100)
```
### vload(*begin*, *dtype=None*, *predicate=None*)


生成一个从开始索引加载 dtype 的 Expr。
* **参数：**
   * **begin** ([Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany)*ofExpr*)：Buffer.dtype 单位的起始索引。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：要加载的数据类型，可以是具有多个 Buffer.dtype 的通道的矢量类型。
   * **predicate** (*Optional[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：一个布尔值向量掩码，指示要加载向量的哪些通道。掩码中的通道数必须等于正在加载的通道数。
* **返回：load**：相应的负载表达式。
* **返回类型：** Expr。

### vstore(*begin*, *value*, *predicate=None*) 


生成一个将值存储到开始索引的 Stmt。
* **参数：**
   * **begin** ([Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany)*ofExpr*)：Buffer.dtype 单位的起始索引。
   * **value** (*Expr*)**：** 要存储的值。
   * **predicate** (*Optional[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：一个布尔值的矢量掩码，指示要存储矢量的哪些通道。掩码中的通道数必须等于值中的通道数。
* **返回：store**：相应的存储语句。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### 

### scope()


返回与此缓冲区关联的存储作用域。:returns: **scope**：与此缓冲区关联的存储范围。:rtype: str

### 

### get_flattened_buffer()


生成一个该缓冲区的扁平化版本的 Buffer。
* **返回：flattened**：相应的平面缓冲区。
* **返回类型：**[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)。

### offset_of(*indices*)


确定扁平缓冲区中提供的索引的偏移量。
* **参数：indices** (*Union**[***[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)***,*** ***List****[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]*)**：** 原始缓冲区中元素的索引。
* **返回：flattened_indices** - 扁平缓冲区中元素的偏移索引。
* **返回类型：** List[[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)]。

## tvm.tir.decl_buffer(*shape*, *dtype=None*, *name='buffer'*, *data=None*, *strides=None*, *elem_offset=None*, *scope=''*, *data_alignment=-1*, *offset_factor=0*, *buffer_type=''*, *axis_separators=None*, *span=None*)


声明一个新的符号缓冲区。


通常，缓冲区在下拉和构建过程中会自动创建。只有当用户想要指定自己的缓冲区布局时才需要这样做。


有关缓冲区使用的详细讨论，请参阅下面的注释。
* **参数：**
   * **shape** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*ofExpr*)：缓冲区的形状。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：缓冲区的数据类型。
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：缓冲区的名称。
   * **data** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,optional*)：缓冲区中的数据指针。
   * **strides** (*arrayofExpr*)：缓冲区*的步幅。
   * **elem_offset** (*Expr,optional*)*：* 数组到数据的起始偏移量。以 dtype 元素的数量表示。
   * **scope** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)**：** 缓冲区的存储范围（如果不是全局的）。如果 scope 等于空字符串，则表示它是全局内存。
   * **data_alignment** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)：1，则对齐方式将设置为 TVM 的内部默认值。
   * **offset_factor** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)：elem_offset 字段的因子。设置后，elem_offset 必须是 offset_factor 的倍数。如果传入 0，则对齐方式将设置为 1。如果传入非零值，且 elem_offset 不为 None，我们将为 elem_offset 创建一个 tir.Var。
   * **buffer_type** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional**,*** ***{""****,"auto_broadcast"}*)：> buffer[i][0][k]。
   * **axis_separators** ([list](https://docs.python.org/3/library/stdtypes.html#list)*of*[int](https://docs.python.org/3/library/functions.html#int)*,optional*)：如果传递，则为轴组之间的分隔符列表，每个轴都会展平为一个输出轴。对于平坦的内存空间，应为 None 或空列表。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：decl_buffer 在源中创建的位置。
* **返回：buffer**：创建的缓冲区。
* **返回类型：**[tvm.tir.Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)。


:::note

缓冲区数据结构反映了 dlpack 中的 DLTensor 结构。虽然 DLTensor 数据结构非常通用，但创建仅处理特定数据结构情况的函数，并使编译后的函数从中受益通常很有帮助。


如果用户在构造函数时传入 strides 并将 elem_offset 设置为 None，则该函数将针对紧凑且对齐的 DLTensor 进行特化。如果用户将完全通用的符号数组传递给 strides，则生成的函数将变为完全通用的函数。

:::

## *class* tvm.tir.DataProducer

## *class* tvm.tir.Layout

布局由大写字母、小写字母和数字组成，其中大写字母表示主轴，对应的小写字母及其因子大小表示从轴。例如，NCHW16c 可以描述一个 5 维张量，其大小为 [batch_size, channel, height, width, channel_block]。其中，从轴 channel_block=16 表示主轴 C（通道）的因子大小。


:::info 另见

`layout`

声明布局。

:::


### index_of(*axis*)


获取轴的索引。
* **参数：axis** ([str](https://docs.python.org/3/library/stdtypes.html#str))：轴名称，需要为 [az,AZ]。
* **返回：index**：1。
* **返回类型：**[int](https://docs.python.org/3/library/functions.html#int)

### factor_of(*axis*)

获取从属轴的因子大小。
* **参数：axis** ([str](https://docs.python.org/3/library/stdtypes.html#str))：轴名称，需要为 [az,AZ]
* **返回：factor**：轴的从属轴的大小（如果轴是主轴），或轴本身的大小（如果轴是从属轴）。如果轴不在布局中，则返回 -1。
* **返回类型：**[int](https://docs.python.org/3/library/functions.html#int)。

## *class* tvm.tir.BijectiveLayout

两种布局（源布局和目标布局）的双射映射。它提供彼此之间的形状和索引转换。


不要直接构造，而是使用 [bijective_layout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#tvmtirbijective_layoutsrc_layoutstrlayout-dst_layoutstrlayout--bijectivelayout)。有关更多详细信息，请参阅文档 [bijective_layout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#tvmtirbijective_layoutsrc_layoutstrlayout-dst_layoutstrlayout--bijectivelayout)。
* **参数：**
   * **src_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)*or*[Layout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#tvmtirlayoutlayout_strstr-dtypestr-int32--layout))：源布局。
   * **dst_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)*or*[Layout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#tvmtirlayoutlayout_strstr-dtypestr-int32--layout))：目标布局。


:::info 另见

`bijective_layout`

声明布局。

:::


### forward_index(*index*)


给定 src-layout 的索引，推断 dst 索引。
* **参数：index** ([Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany)*ofExpr*)：src 布局中的索引。
* **返回：dst_index**：layout 中推断的索引。
* **返回类型：**[Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany) of Expr。

### backward_index(*index*)


给定 dst-layout 的索引，推断 src 索引。
* **参数：index** ([Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany)*ofExpr*)：layout 中的索引。
* **返回：src_index：** layout 中推断的索引。
* **返回类型：**[Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany) of Expr。

### forward_shape(*shape*)


给定 src-layout 的形状，推断 dst 的形状。
* **参数：shape** ([Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany)*ofExpr*)：layout 中的形状。
* **返回：dst_shape**：layout 中推断的形状。
* **返回类型：**[Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany) of Expr。

### backward_shape(*shape*)


给定 dst-layout 的形状，推断 src 的形状。
* **参数：shape** ([Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany)*ofExpr*)*：*layout 中的形状。
* **返回：src_shape**：中推断的形状。
* **返回类型：**[Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany) of Expr。

## tvm.tir.bijective_layout(*src_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Layout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#tvmtirlayoutlayout_strstr-dtypestr-int32--layout), *dst_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Layout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#tvmtirlayoutlayout_strstr-dtypestr-int32--layout)) → [BijectiveLayout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbijectivelayout)


创建双射布局映射。
* **参数：**
   * **src_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)*or*[Layout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#tvmtirlayoutlayout_strstr-dtypestr-int32--layout))：源布局。
   * **dst_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)*or*[Layout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#tvmtirlayoutlayout_strstr-dtypestr-int32--layout))：目标布局。
* **返回：bijective_layout：** 创建的双射布局。
* **返回类型：**[BijectiveLayout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbijectivelayout)。

## tvm.tir.layout(*layout_str:*[str](https://docs.python.org/3/library/stdtypes.html#str), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'int32'*) → [Layout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#tvmtirlayoutlayout_strstr-dtypestr-int32--layout)


从字符串创建布局节点。
* **参数：**
   * **layout_str** ([str](https://docs.python.org/3/library/stdtypes.html#str))：布局表示由大写字母、小写字母和数字组成，其中大写字母表示主轴，对应的小写字母及其因子大小表示从轴。例如，NCHW16c 可以描述一个 5 维张量，其参数为 [batch_size, channel, height, width, channel_block]。其中，从轴 channel_block=16 表示主轴 C（通道）的因子大小。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 返回布局中生成的轴变量的数据类型。必须为整数类型。
* **返回：layout**：创建的布局。
* **返回类型：**[Layout](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#tvmtirlayoutlayout_strstr-dtypestr-int32--layout)。

## *class* tvm.tir.Var(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Type](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirtype), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)】


符号变量。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) - 名称。
   * **dtype** (*Union[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[ir.Type](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirtype)*]*)：数据类型。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.SizeVar(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Type](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirtype), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


表示张量索引大小的符号变量。


大于或等于零。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：名称。
   * **dtype** (*Union[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[ir.Type](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirtype)*]*)*：* 数据类型。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此表达式在源代码中的位置。

## *class* tvm.tir.Reduce(*combiner:*[CommReducer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtircommreducerlhslistvar-rhslistvar-resultlistprimexpr-identity_elementlistprimexpr-spanspannone-none), *src:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *rdom:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[IterVar](range-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)*]*, *condition:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *value_index:*[int](https://docs.python.org/3/library/functions.html#int), *init:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)

归约节点。
* **参数：**
   * **combiner** ([CommReducer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtircommreducerlhslistvar-rhslistvar-resultlistprimexpr-identity_elementlistprimexpr-spanspannone-none))：合并器。
   * **src** ([list](https://docs.python.org/3/library/stdtypes.html#list)*ofExpr*)**：** 源表达式。
   * **rdom** ([list](https://docs.python.org/3/library/stdtypes.html#list)*of*[IterVar](range-varvarstr-iter_typeint-thread_tagstr--spanspannone-none))*：](*[https://docs.python.org/3/library/stdtypes.html#list](https://docs.python.org/3/library/stdtypes.html#list)*"（在 Python v3.13 中）")-*[迭代](range-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)域。
   * **condition** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：归约条件。
   * **value_index** ([int](https://docs.python.org/3/library/functions.html#int))*：* 值索引。
   * **init** ([list](https://docs.python.org/3/library/stdtypes.html#list)*ofExpr*)：输出的初始值。可以是 int、float 或 ProducerLoad 。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.FloatImm(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *value:*[float](https://docs.python.org/3/library/functions.html#float), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


浮点常数。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：数据类型。
   * **value** ([float](https://docs.python.org/3/library/functions.html#float))：常量值。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.IntImm(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *value:*[int](https://docs.python.org/3/library/functions.html#int), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


整数常量。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 数据类型。
   * **value** ([int](https://docs.python.org/3/library/functions.html#int))：常量值。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.StringImm(*value:*[str](https://docs.python.org/3/library/stdtypes.html#str), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


字符串常量。
* **参数：**
   * **value** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 函数的值。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Cast(*dtype*, *value*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)

转换表达式。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：数据类型。
   * **value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：函数的值。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Add(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)

Add 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此表达式在源代码中的位置。

## *class* tvm.tir.Sub(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Sub 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Mul(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Mul 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Div(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Div 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Mod(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Mod 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.FloorDiv(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


FloorDiv 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此表达式在源代码中的位置。

## *class* tvm.tir.FloorMod(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


FloorMod 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Min(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Min 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：*右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Max(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Max 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.EQ(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


EQ 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.NE(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


NE 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.LT(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)

LT 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.LE(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


LE 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.GT(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


GT 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此表达式在源代码中的位置。

## *class* tvm.tir.GE(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


GE 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.And(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


And 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此表达式在源代码中的位置。

## *class* tvm.tir.Or(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Or 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左边的操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Not(*a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Not 节点。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入值。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：*此表达式在源代码中的位置。

## *class* tvm.tir.Select(*condition:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *true_value:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *false_value:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Select 节点。


:::Note

Select 可能会同时计算 true_value 和 false_value。如果您只想获取仅评估正确分支的条件表达式，请使用 `tvm.tir.if_then_else`。

:::
* **参数：**
   * **condition** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：条件表达式。
   * **true_value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：条件为真时要取的值。
   * **false_value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：条件为假时取的值。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.BufferLoad(*buffer:*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *indices:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *predicate:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


BufferLoad 节点。
* **参数：**
   * **buffer** ([Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer))*：* 要加载的缓冲区。
   * **indices** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：从中加载值的缓冲区索引。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此表达式在源代码中的位置。
   * **predicate** (*Optional[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)**：** 一个布尔值向量掩码，指示要加载向量的哪些通道。掩码中的通道数必须等于正在加载的通道数。

## *class* tvm.tir.ProducerLoad(*producer:*[DataProducer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirdataproducer), *indices:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


ProducerLoad 节点。
* **参数：**
   * **producer** ([DataProducer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirdataproducer))*：* 要加载的缓冲区。
   * **indices** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：缓冲区索引。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Ramp(*base:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *stride:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *lanes:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Ramp 节点。
* **参数：**
   * **base** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：基本表达式。
   * **stride** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 斜坡的步幅。
   * **lanes** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 表达式的车道。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此表达式在源代码中的位置。

## *class* tvm.tir.Broadcast(*value:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *lanes:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Broadcast 节点。
* **参数：**
   * **value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 表达式的值。
   * **lanes** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：表达式的车道。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Shuffle(*vectors:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *indices:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Shuffle 节点。
* **参数：**
   * **vectors** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)*：* 向量。
   * **indices** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)*：* 索引。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Call(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *op:*[Op](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirop)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *args:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


tir.Call 节点。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：返回数据类型。
   * **op** (*Union[*[Op](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirop)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：要调用的函数，或全局 tvm.Op 的名称。
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list)*ofExpr*)：调用的输入*参数。*
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)**：** 此表达式在源代码中的位置。

## *class* tvm.tir.CallEffectKind

可能的 tir.Call 效果种类。

## *class* tvm.tir.Let(*var:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *value:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *body:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Let 节点。
* **参数：**
   * **var** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：绑定中的变量。
   * **value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：要绑定的值。
   * **body** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：主体表达式。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.IterVar(*dom:*[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none), *var:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *iter_type:*[int](https://docs.python.org/3/library/functions.html#int), *thread_tag:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= ''*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


表示迭代变量。


IterVar 表示计算中的轴迭代。
* **参数：**
   * **dom** ([Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none))- 迭代的范围。
   * **var** (*Union[*[tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)*：* 用于迭代的内部变量。
   * **iter_type** ([int](https://docs.python.org/3/library/functions.html#int))*：*迭代类型。
   * **thread_tag** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：*线程类型标签。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。


:::info 另见

`te.thread_axis`


创建线程轴 IterVar。

`te.reduce_axis`


创建归约轴 IterVar。

:::

## *class* tvm.tir.CommReducer(*lhs:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*]*, *rhs:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*]*, *result:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *identity_element:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


交换约简运算符。
* **参数：**
   * **lhs** (*List[*[tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*]*)*：* 减速器的左参数。
   * **rhs** (*List[*[tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*]*)*：* 减速器的正确参数。
   * **result** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)**：** 约简结果。
   * **identity_element** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：身份元素。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此表达式在源代码中的位置。

## *class* tvm.tir.Stmt


所有语句的基类。

## *class* tvm.tir.LetStmt(*var:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *value:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *body:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


LetStmt 节点。
* **参数：**
   * **var** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：绑定中的变量。
   * **value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：要绑定的值。
   * **body** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))*：* 正文语句。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：源代码中 stmt 的位置。

## *class* tvm.tir.AssertStmt(*condition:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *message:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *body:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


AssertStmt 节点。
* **参数：**
   * **condition** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：断言条件。
   * **message** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：错误消息。
   * **body** ([tvm.tir.Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))**：** 主体语句。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：源代码中 stmt 的位置。

## *class* tvm.tir.ForKind(*value*)


for 循环的种类。


:::Note

ForKind 可以改变循环的控制流语义，需要在所有 TIR 传递中考虑它。

:::

## *class* tvm.tir.For(*loop_var:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *min:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *extent:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *kind:*[ForKind](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirforkindvalue), *body:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *thread_binding:*[IterVar](range-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *annotations:*[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


 For 节点。
* **参数：**
   * **loop_var** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))*：* 循环变量。
   * **min** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：起始值。
   * **extent** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 循环的长度。
   * **kind** ([ForKind](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirforkindvalue))：for 的类型。
   * **body** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：正文语句。
   * **thread_binding** (*Optional[*[tir.IterVar](range-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)*]*)：此循环绑定到的线程。仅当 kind 为 ThreadBinding 时有效/
   * **annotations** (*Optional**[****Mapping**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Object****]]*)：额外的注解提示。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：源代码中 stmt 的位置。

## *class* tvm.tir.While(*condition:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *body:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


While 节点。
* **参数：**
   * **condition** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：终止条件。
   * **body** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))*：* 正文语句。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)**：** 源代码中 stmt 的位置。

## *class* tvm.tir.BufferStore(*buffer:*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *value:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *indices:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *predicate:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


缓冲存储节点。
* **参数：**
   * **buffer** ([Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer))：缓冲区。
   * **value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 我们要存储的值。
   * **indices** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)*：* 要存储的索引位置。
   * **predicate** (*Optional[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)*：* 一个布尔值的矢量掩码，指示要存储矢量的哪些通道。掩码中的通道数必须等于值中的通道数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：源代码中 stmt 的位置。

## *class* tvm.tir.BufferRealize(*buffer:*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *bounds:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)*]*, *condition:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *body:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


BufferRealize 节点。
* **参数：**
   * **buffer** ([Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer))：缓冲区。
   * **bounds** (*List[*[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)*]*)：我们要存储的值。
   * **condition** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：实现条件。
   * **body** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：语句的主体。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 源代码中 stmt 的位置。

## *class* tvm.tir.Allocate(*buffer_var:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *extents:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *condition:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *body:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *annotations:*[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Allocate 节点。
* **参数：**
   * **buffer_var** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：缓冲区变量。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：缓冲区的数据类型。
   * **extents** ([list](https://docs.python.org/3/library/stdtypes.html#list)*ofExpr*)：分配的范围。
   * **condition** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 条件。
   * **body** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))*：* 正文语句。
   * **annotations** (*Optional**[****Mapping**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Object****]]*)：附加注解提。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：源代码中 stmt 的位置。

## *class* tvm.tir.AllocateConst(*buffer_var:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *extents:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *data_or_idx:*NDArray*|*[int](https://docs.python.org/3/library/functions.html#int), *body:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *annotations:*[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


AllocateConst 节点。
* **参数：**
   * **buffer_var** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：缓冲区变量。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：缓冲区的数据类型。
   * **extents** ([list](https://docs.python.org/3/library/stdtypes.html#list)*ofExpr*)*：* 分配的范围。
   * **data_or_idx** (*Union[*NDArray*,*[int](https://docs.python.org/3/library/functions.html#int)*]*)：如果是 NDArray，则这是与常量关联的 const 数据。如果是整数，则这是包含 AllocateConst 的 IRModule 的“constants”属性的索引。
   * **body** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：正文语句。
   * **annotations** (*Optional**[****Mapping**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Object****]]*)**：** 关于分配的附加注解。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 源代码中 stmt 的位置。

## *class* tvm.tir.AttrStmt(*node: Object*, *attr_key:*[str](https://docs.python.org/3/library/stdtypes.html#str), *value:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *body:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


AttrStmt 节点。
* **参数：**
   * **node** (*Object*)**：** 注解属性的节点。
   * **attr_key** ([str](https://docs.python.org/3/library/stdtypes.html#str))**：** 属性类型键。
   * **value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：属性的值。
   * **body** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))*：* 正文语句。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 源代码中 stmt 的位置。

## *class* tvm.tir.DeclBuffer(*buffer:*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *body:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


DeclBuffer 节点。
* **参数：**
   * **buffer** ([Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer))**：**正在声明的缓冲区。
   * **body** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：要执行的主体语句。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此 DeclBuffer 在源代码中的位置。

## *class* tvm.tir.SeqStmt(*seq:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*]*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


语句序列。
* **参数：**
   * **seq** (*List[*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*]*)：语句。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：源代码中 stmt 的位置。

## *class* tvm.tir.IfThenElse(*condition:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *then_case:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *else_case:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*|*[None](https://docs.python.org/3/library/constants.html#None), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


IfThenElse 节点。
* **参数：**
   * **condition** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：表达式。
   * **then_case** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：如果条件为真则执行的语句。
   * **else_case** (*Optional[*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*]*)：如果条件为假则执行的语句。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：源代码中 stmt 的位置。

## *class* tvm.tir.Evaluate(*value:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Evaluate 节点。
* **参数：**
   * **value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：要评估的表达式。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：源代码中 stmt 的位置。

## tvm.tir.stmt_seq(args:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)) → [SeqStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirseqstmtseqliststmt-spanspannone-none)


制定语句序列。
* **参数：*args***(*Union[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*,*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*]*)*：* 要组合为序列的语句列表。
* **返回：stmt**：组合语句。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

## tvm.tir.stmt_list(*stmt:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)]


从块中创建 stmt 列表。
* **参数：stmt** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))*：* 输入语句。
* **返回：stmt_list**：解压后的语句列表。
* **返回类型：** List[[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)]。

## *class* tvm.tir.BufferRegion(*buffer:*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *region:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)*]*)

BufferRegion 节点。
* **参数：**
   * **buffer** ([Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer))**：** 缓冲区的缓冲区。
   * **region** (*List[*[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)*]*)：缓冲区的区域数组。

## *class* tvm.tir.MatchBufferRegion(*buffer:*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *source:*[BufferRegion](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferRegion))


MatchBufferRegion 节点。
* **参数：**
   * **buffer** ([Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer))***：*** 目标缓冲区。
   * **source** ([BufferRegion](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferRegion))**：** 源缓冲区的区域。

## *class* tvm.tir.Block(*iter_vars:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[IterVar](range-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)*]*, *reads:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[BufferRegion](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferRegion)*]*, *writes:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[BufferRegion](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferRegion)*]*, *name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str), *body:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt), *init:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *alloc_buffers:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *match_buffers:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[MatchBufferRegion](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmatchbufferregionbufferbuffer-sourcebufferregion)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *annotations:*[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


Block 节点。
* **参数：**
   * **iter_vars** (*List[*[IterVar](range-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)*]*)：块变量。
   * **reads** (*List[*[BufferRegion](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferRegion)*]*)：块的读取缓冲区区域。
   * **writes** (*List[*[BufferRegion](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferRegion)*]*)：块的写入缓冲区区域。
   * **name_hint** ([str](https://docs.python.org/3/library/stdtypes.html#str))：块的 name_hint。
   * **body** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：块的主体。
   * **init** (*Optional[*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*]*)：缩减块的 init 块。
   * **alloc_buffers** (*Optional**[***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)***]*]*)：缓冲区分配。
   * **match_buffers** (*Optional**[****List**[***[MatchBufferRegion](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmatchbufferregionbufferbuffer-sourcebufferregion)***]****]*)：子区域缓冲区匹配。
   * **annotations** (*Optional**[****Mapping**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Object****]]*)：额外的注解提示。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此块在源代码中的位置。

## *class* tvm.tir.BlockRealize(*iter_values:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *predicate:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[bool](https://docs.python.org/3/library/functions.html#bool), *block:*[Block](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)

BlockRealize 节点。
* **参数：**
   * **iter_values** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)*：* 块变量的绑定值。
   * **predicate** (*Union[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*,*[bool](https://docs.python.org/3/library/functions.html#bool)*]*)：块的谓词。
   * **block** ([Block](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none))*：* 要实现的块。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此 block_realize 在源代码中的位置。

## *class* tvm.tir.PrimFunc(*params*, *body*, *ret_type=None*, *buffer_map=None*, *attrs=None*, *span=None*)


函数声明表达式。
* **参数：**
   * **params** (*List*[****Union**[***[tvm.tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)***,** [tvm.tir.Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)***]****]*)：函数的输入参数列表。
   * **body** ([tvm.tir.Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：函数主体。
   * **ret_type** ([tvm.ir.Type](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirtype))：函数的返回类型注解。
   * **buffer_map** ([Map](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirmapinput_dictmappinganyany)*[*[tvm.tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,*[tvm.tir.Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*]*)：缓冲区绑定图。
   * **attrs** (*Optional*[*tvm.Attrs]*)：函数的属性，可以为 None。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此 itervar 在源代码中的位置。

### 

### with_body(*new_body*, *span=None*)

创建具有相同集合签名但具有新主体的新 PrimFunc。
* **参数：**
   * **new_body** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：新主体。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此 itervar 在源代码中的位置。
* **返回：new_func**：创建的新函数。
* **返回类型：**[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)。

### specialize(*param_map:*[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)*[*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*]*)


PrimFunc 的专门参数。
* **参数：param_map** (*Mapping*[***[tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)***,***Union***[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*,*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*]])：从函数参数到实例的映射。


**示例**


我们可以定义一个具有符号形状的 Meta TIR 函数：

```python
@T.prim_func
def mem_copy(a: T.handle, b: T.handle, m: T.int32, n: T.int32) -> None:
    A = T.match_buffer(a, (m, n), "float32")
    B = T.match_buffer(b, (m, n), "float32")

    for i, j in T.grid(m, n):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]
```

然后我们可以利用给定的形状或缓冲区使其特化。

```python
a, _, m, n = mem_copy.params
func = mem_copy.specialize({a: tir.decl_buffer((16, 16))})
# 或者
func = mem_copy.specialize({n: 16, m: 16})
```


专门的函数：

```python
@T.prim_func
def mem_copy_16_16(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    B = T.match_buffer(b, (16, 16), "float32")

    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]
```
* **返回：func：** 带有特殊参数的新函数。
* **返回类型：**[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)。

## *class* tvm.tir.TensorIntrin(*desc*, *impl*)


张量的内在函数。
* **参数：**
   * **desc** ([PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))**：** 描述计算的函数。
   * **impl** ([PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))**：** 执行的实现函数。



### *static* register(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *desc:*[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone), *impl:*[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone), *override:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)


使用其名称注册张量内在函数。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 要注册的 TensorIntrin 的名称。
   * **desc** ([PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))*：* 描述计算的函数。
   * **impl** ([PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)) - 执行的实现函数。
   * **override** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否覆盖现有的内在函数。

### *static* get(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *allow_missing:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [TensorIntrin](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirtensorintrindesc-impl) | [None](https://docs.python.org/3/library/constants.html#None)


通过名称查找张量内在函数。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：要查找的 TensorIntrin 的名称。
   * **allow_missing** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否允许缺失张量内部参数。若为 False，则在张量内部参数。
   * **exist.** 不存在。
* **返回：result**：具有指定名称的 TensorIntrin，如果未找到则为 None。
* **返回类型：** Optional[[TensorIntrin](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirtensorintrindesc-impl)]。

## *class* tvm.tir.IndexMap(*initial_indices*, *final_indices*, *inverse_index_map*)


从多维索引到另一组多维索引的映射。
* **参数：**
   * **initial_indices** (*List[*[tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*]*)：表示重新映射之前的索引的变量。
   * **final_indices** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：定义重新映射后的索引的表达式。
   * **inverse_index_map** (*Union*[****Callable**,** ***Optional***[*[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*]]*)*：可选的预定义逆索引图。定义此方法后，IndexMap::Inverse 将返回预定义的逆索引图。否则，逆索引图将即时计算。用户有责任确保预定义逆索引图的正确性。



### *static* from_func(*mapping_function:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable), *ndim:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *inverse_index_map:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*|*[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*,*, *index_dtype: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'int64'*)

从函数创建索引图。
* **参数：**
   * **mapping_function** (*Callable*)：从源索引映射到目标索引的函数。该函数应接受 tir.Var 参数并返回 tir.PrimExpr 或 tir.PrimExpr 列表。返回 tir.PrimExpr 相当于返回包含该 tir.PrimExpr 的长度为 1 的列表。
   * **ndim** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*)：此转换应应用到的缓冲区的维数。如果 mapping_function 使用可变参数*args，则必须指定 ndim。如果 mapping_function 不使用可变参数，则 ndim 为可选。
   * **inverse_index_map** (*Union**[****Callable**,*** ***Optional****[*[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*]]*)：可选的预定义逆索引图。定义此方法后，IndexMap::Inverse 将返回预定义的逆索引图。否则，逆索引图将即时计算。用户有责任确保预定义逆索引图的正确性。
* **返回：index_map**：返回表示 mapping_function 的 IndexMap 。
* **返回类型：**[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)。

### *static* from_func_with_separators(*mapping_function:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable), *ndim:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *inverse_index_map:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*|*[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, ***, *index_dtype: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'int64'*)


从函数创建索引图。
* **参数：**
   * **mapping_function** (*Callable*)：用于从源索引映射到目标索引的函数。该函数应接受 tir.Var 参数并返回 tir.PrimExpr 或列表。返回列表的每个元素应为 tir.PrimExpr 或 IndexMap.AXIS_SEPARATOR 对象。返回 tir.PrimExpr 相当于返回包含该 tir.PrimExpr 的长度为 1 的列表。
   * **ndim** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*)：此转换应应用到的缓冲区的维数。如果 mapping_function 使用可变参数*args，则必须指定 ndim 。如果 mapping_function 不使用可变参数，则 ndim 为可选。
   * **inverse_index_map** (*Union**[****Callable**,*** ***Optional****[*[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*]]*)**：** 可选的预定义逆索引图。定义此方法后，IndexMap::Inverse 将返回预定义的逆索引图。否则，逆索引图将即时计算。用户有责任确保预定义逆索引图的正确性。
   * **index_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 映射函数中输入迭代器使用的默认索引 dtype。
* **返回：ret**：返回一个元组，其第一个元素是表示**mapping_function 的 IndexMap ，其第二个索引是 IndexMap.AXIS_SEPARATOR 发生的索引列表 。
* **返回类型：**[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)[[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map), List[[int](https://docs.python.org/3/library/functions.html#int)]]。

### is_equivalent_to(*other_map:*[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)) → [bool](https://docs.python.org/3/library/functions.html#bool)


如果索引图等效，则返回。
* **参数：other_map** ([IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map))：应该进行比较的 IndexMap。
* **返回：is_equivalent**：如果两个映射表示相同的转换，则为 True，否则为 False。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

### map_indices(*indices:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)]


将索引图应用于一组索引。
* **参数：indices** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：要映射的索引。
* **返回：result**：映射的索引。
* **返回类型：** List[[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)]。

### map_shape(*shape:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)]

将索引图应用于缓冲区形状。
* **参数：shape** (*List[*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：要映射的缓冲区形状。
* **返回：result**：映射的形状。
* **返回类型：** List[[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)]。

### map_ndarray(*arr_src:*NDArray) → NDArray


将此索引映射应用于输入 NDArray，以转换其布局。
* **参数：arr_src** (*runtime.NDArray*)：要转换的 NDArray。
* **返回：arr_dst**：转换后的 NDArray。
* **返回类型：** runtime.NDArray。

### inverse(*shape:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*) → [IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)


返回该映射的逆映射。  


如果该函数不是双射（bijective），则会抛出错误。
* **参数：shape** (*List**[****Union**[***[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)***,***[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)***]****]*)：需要确定逆的区域。用于验证映射在此范围内是否为双射。
* **返回：inverse**：逆。
* **返回类型：**[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)。

### non_surjective_inverse(*shape:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map), [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)]


返回该映射的逆映射。  


可用于引入填充（padding）的变换。
* **参数：shape** (*List**[****Union**[***[Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)***,***[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)***]****]*)：需要确定逆的区域。用于确定谓词。
* **返回：result**：逆，以及逆映射到输入范围中的有效索引的谓词。
* **返回类型：**[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)[[IndexMap](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map), [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)]。


**示例**

```python
index_map = IndexMap.from_func(lambda i: [i//4, i%4])
inverse_map, predicate = index_map.non_surjective_inverse([14])
assert inverse_map.is_equivalent_to(IndexMap.from_func(lambda j,k: [4*j + k])
print(predicate) # 打印 "(axis0==3) && (axis2 >= 2)"
```
## tvm.tir.call_packed_lowered(args*, *span=None*)

调用 packed 的低版本。packed 函数的参数可以是 Expr 或 Buffer。当传入 Expr 时，参数为对应的 POD 类型。当参数为 Buffer 时，对应的 PackedFunc 将收到一个 TVMArrayHandle，其内容在回调期间有效。如果 PackedFunc 是 Python 回调，则对应的参数为 NDArray。
* **参数：**
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list)*ofExprorBuffer.*)：位置参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::info 另见

`te.extern`


使用外部函数调用创建张量。

:::

## tvm.tir.call_cpacked_lowered(args*, *span=None*)

call c-packed 的低版本。与 call_packed 相同，但第一个参数是函数名（与 call_extern 类似），最后一个参数是资源句柄。
* **参数：**
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list)*ofExprorBuffer.*)：位置参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此运算符在源代码中的位置。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


:::info 另见

`te.extern`


使用外部函数调用创建张量。

:::

## tvm.tir.call_tir(*global_var:*[GlobalVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none), args*)


调用同一 IRModule 中的另一个 PrimFunc。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.call_packed(args*, *span=None*)

通过调用外部打包函数来构建表达式。


打包函数的参数可以是 Expr 或 Buffer。Expr 为参数时，参数为对应的 POD 类型。


当参数为 Buffer 时，对应的 PackedFunc 会收到一个 TVMArrayHandle，其内容在回调期间有效。如果 PackedFunc 是 python 回调，则对应的参数为 NDArray。
* **参数：**
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list)*ofExprorBuffer.*)：位置参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此运算符在源代码中的位置。
* **返回：call：** 调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


:::info 另见

`te.extern`


使用外部函数调用创建张量。

:::

## tvm.tir.call_cpacked(args*, *span=None*)


通过调用外部打包函数来构建表达式。


与 call_packed 相同，但第一个参数是函数名（如 call_extern 中一样），最后一个参数是资源句柄。
* **参数：**
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list)*ofExprorBuffer.*)：位置参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


:::info 另见

`te.extern`


使用外部函数调用创建张量。

:::

## tvm.tir.call_intrin(*dtype*, *func_name*, args*, *span=None*)


通过调用内部函数来构建表达式。


内部函数可以通过内部转换规则使用多种数据类型进行重载。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：内部函数名称。
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list))：位置参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.call_pure_extern(*dtype*, *func_name*, args*, *span=None*)

通过调用纯外部函数来构建表达式。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 外部函数名称。
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list))**：** 位置参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.call_extern(*dtype*, *func_name*, args*, *span=None*)


通过调用外部函数来构建表达式。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：外部函数名称。
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list))**：** 位置参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此运算符在源代码中的位置。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.call_llvm_intrin(*dtype*, *name*, args*, *span=None*)


通过调用 llvm 内部函数来构建表达式
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 结果的数据类型。
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))**：** llvm 内部函数的名称。
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list))：位置参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.call_llvm_pure_intrin(*dtype*, *name*, args*, *span=None*)


通过调用纯 llvm 内部函数来构建表达式
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* llvm 内部函数的名称。
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list))：位置参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

## tvm.tir.ret(*val*, *span=None*)


创建 tir 返回表达式
* **参数：**
   * **val** (*Expr*)：返回的 tir 表达式，其数据类型为 int、float 或 void 指针。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：ret**：返回表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.all(args*, *span=None*)


创建一个新的表达式，该表达式表示所有参数条件的交集。
* **参数：**
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list))：符号布尔表达式列表。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：expr**：表达式。
* **返回类型：** Expr。

## tvm.tir.any(args*, *span=None*)


创建一个新的表达式，表示所有参数条件的并集。
* **参数：**
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list))：符号布尔表达式列表。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：expr** *：* 表达式。
* **返回类型：** Expr。

## tvm.tir.min_value(*dtype*, *span=None*)


dtype 的最小值。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))**：** 数据类型。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：value**：dtype 的最小值。
* **返回类型：** tvm.Expr。


## tvm.tir.max_value(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

dtype 的最大值
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：数据类型。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)**：** 此运算符在源代码中的位置。
* **返回：value**：dtype 的最大值。
* **返回类型：** tvm.Expr

## tvm.tir.trace(*args*, *trace_action='tvm.default_trace_action'*)

在运行时跟踪张量数据。


trace 函数允许在运行时跟踪特定的张量。跟踪值应作为最后一个参数。应指定跟踪操作，默认情况下使用 tvm.default_trace_action。
* **参数：**
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list)*ofExprorBuffers.*)：位置参数。
   * **trace_action** (*str.*)：跟踪操作的名称。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


:::info 另见

`tvm.tir.call_packed`


创建打包函数。

:::


## tvm.tir.tvm_stack_alloca(*dtype_str*, *num*)

返回堆栈上的新 dtype[num]。
* **参数：**
   * **dtype_str** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 数组的数据类型。
   * **num** ([int](https://docs.python.org/3/library/functions.html#int))：数组的大小。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_stack_make_shape(args*)


在堆栈上分配一个形状元组，返回句柄。
* **参数：args** ([int](https://docs.python.org/3/library/functions.html#int))：元组形状。
* **返回：call** *：* 调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_stack_make_array(*data*, *shape*, *strides*, *ndim*, *arr_dtype*, *elem_offset*)


在堆栈上分配一个 NDArray（DLTensor），返回句柄。
* **参数：**
   * **data** (*Expr*)**：** 数组的数据。
   * **shape** (*Expr*)：数组的形状。
   * **strides** (*Expr*)：数组的步幅。
   * **ndim** (*Expr*)**：** 数组的维度。
   * **arr_dtype** (*Expr*)*：* 数组的数据类型。
   * **elem_offse** (*Expr*)**：** 数组的元素偏移量。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_tuple(value*)

在 AttrStmt 的值字段中创建一个元组结构。
* **参数：value** (*Expr*)：元组中的值。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.handle_add_byte_offset(*handle*, *offset*)


为句柄添加偏移量。
* **参数：**
   * **handle** (*Expr*)：句柄。
   * **offset** ([int](https://docs.python.org/3/library/functions.html#int))**：** 偏移量。
* **返回：call：** 调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_struct_get(*arr*, *index*, *field*, *dtype*)


获取数组中的结构字段值。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的日期类型。
   * **arr** (*StructType)：结构数组。
   * **index** ([int](https://docs.python.org/3/library/functions.html#int))：结构的索引。
   * **field** ([int](https://docs.python.org/3/library/functions.html#int))：结构体的字段。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_struct_set(*arr*, *index*, *field*, *value*)


设置数组中结构字段的值。
* **参数：**
   * **arr** (*StructType)：结构数组。
   * **index** ([int](https://docs.python.org/3/library/functions.html#int))：结构的索引。
   * **field** ([int](https://docs.python.org/3/library/functions.html#int))：结构体的字段。
   * **value** (*Expr*)：要在字段中设置的值。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.address_of(*obj:*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*|*[BufferLoad](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferLoad), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


返回缓冲区中元素的地址。
* **参数：**
   * **obj** (*Union[*[Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*,*[BufferLoad](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferLoad)*]*)：缓冲区或缓冲区负载。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.lookup_param(*param_name*, *span=None*)


按名称返回参数。
* **参数：**
   * **param_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))**：** 参数的名称。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)***：*** 此运算符在源代码中的位置。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.assume(*cond=None*)


提供可用于简化的真实陈述。
* **参数：cond** (*Expr*)：约束条件。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

## tvm.tir.undef()

返回已初始化的任意值。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_thread_allreduce(freduce_args*)


在线程块内执行 allreduce。
* **参数：freduce_args** (*Expr*)：参数。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

## tvm.tir.type_annotation(*dtype*)


创建类型注解表达式。
* **参数：dtype** (*Expr*)：数据类型。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_access_ptr(*ptype*, *data*, *offset*, *extent*, *rw_mask*)

通过内存访问模式信息获取头部访问地址。
* **参数：**
   * **ptype** (*Expr*)**：** 指针的数据类型。
   * **data** (*DType)：指针的数据。
   * **offset** ([int](https://docs.python.org/3/library/functions.html#int))：指针的偏移量。
   * **extent** ([int](https://docs.python.org/3/library/functions.html#int))**：** 指针的范围。
   * **rw_mask** ([int](https://docs.python.org/3/library/functions.html#int))：读写掩码。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_throw_last_error()


抛出 TVMGetLastError()。
* **返回：ret**：返回表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_load_matrix_sync(*fragment*, *m*, *n*, *k*, *index*, *buffer_ptr*, *stride*, *layout*)


TVM 张量核心负载运算符的内在函数。
* **参数：**
   * **fragment** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))*：* wmma 片段。
   * **m** (*UIntImm*)：wmma 片段的形状。
   * **n** (*UIntImm*)：wmma 片段的形状。
   * **k** (*UIntImm*)：wmma 片段的形状。
   * **index** (*Expr*)：片段索引。
   * **buffer_ptr** (*Expr*)：片段缓冲区指针。
   * **stride** (*Expr*)：片段步幅。
   * **layout** (*Literal**[****"row_major"**,*** ***"column_major"****]*)：片段布局。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_store_matrix_sync(*fragment*, *m*, *n*, *k*, *index*, *buffer_ptr*, *stride*, *layout*)

TVM 张量核心存储运算符的内在函数。
* **参数：**
   * **fragment** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))*：* wmma 片段。
   * **m** (*UIntImm*)：wmma 片段的形状。
   * **n** (*UIntImm*)：wmma 片段的形状。
   * **k** (*UIntImm*)：wmma 片段的形状。
   * **index** (*Expr*)：片段索引。
   * **buffer_ptr** (*Expr*)：片段缓冲区指针。
   * **stride** (*Expr*)：片段步幅。
   * **layout** (*Literal**[****"row_major"**,*** ***"column_major"****]*)：片段布局。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_mma_sync(*fragment_d*, *index_d*, *fragment_a*, *index_a*, *fragment_b*, *index_b*, *fragment_c*, *index_c*)


TVM 张量核心 mma_sync 运算符的内在函数。
* **参数：**
   * **fragment_d** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：wmma fragment_d。
   * **index_d** (*Expr*)：fragment_d 索引。
   * **fragment_a** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：wmma fragment_a。
   * **index_a** (*Expr*)：fragment_a 索引。
   * **fragment_b** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：wmma fragment_b。
   * **index_b** (*Expr*)*：* fragment_b 索引。
   * **fragment_c** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))*：* wmma fragment_c。
   * **index_c** (*Expr*)**：** fragment_c 索引。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_bmma_sync(*fragment_d*, *index_d*, *fragment_a*, *index_a*, *fragment_b*, *index_b*, *fragment_c*, *index_c*)

张量核心 bmma_sync 运算符的 TVM 内在函数。
* **参数：**
   * fragment_d (tir.Var) – bwmma 片段_d。
   * index_d (Expr) – 片段 d 的索引。
   * fragment_a (tir.Var) – bwmma 片段_a。
   * **index_a** (*Expr*)：fragment_a 索引。
   * fragment_b (tir.Var) – bwmma 片段_b。
   * index_b (Expr) – 片段_b 的索引。
   * fragment_c (tir.Var) – bwmma 片段_c。
   * index_c (Expr) – 片段_c 的索引。
* **返回：call** *：* 调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tvm_fill_fragment(*fragment*, *m*, *n*, *k*, *index*, *value*)


TVM 张量核心 fill_fragment 运算符的内在函数。
* **参数：**
   * **fragment** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：wmma 片段。
   * **m** (*UIntImm*) - wmma 片段的形状。
   * **n** (*UIntImm*) - wmma 片段的形状。
   * **k** (*UIntImm*) - wmma 片段的形状。
   * **index** (*Expr*) - 片段索引。
   * **value** (*Expr*) - 片段中要填充的值。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_mma(*dtype*, *shape*, *A_layout*, *B_layout*, *A_dtype*, *B_dtype*, *C_dtype*, *multiplicand_a*, *a_index*, *multiplicand_b*, *b_index*, *accumulator*, *c_index*, *saturate*, *operator=None*)

TVM 内在的 ptx 张量核心 mma 指令 [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma)。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 结果的数据类型。
   * **shape** ([str](https://docs.python.org/3/library/stdtypes.html#str))**：** mma 片段的形状。
   * **A_layout** (*Literal**[****"row"**,*** ***"col"****]*)：被乘数片段 A 的布局。
   * **B_layout** (*Literal**[****"row"**,*** ***"col"****]*)：被乘数片段 B 的布局。
   * **A_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：被乘数片段 A 的数据类型。
   * **B_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))**：** 被乘数片段 B 的数据类型。
   * **C_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：累加器片段 C 的数据类型。
   * **multiplicand_a** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：被乘数片段 A 变量。
   * **a_index** (*Expr*)*：* 被乘数片段 A 的索引。
   * **multiplicand_b** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：被乘数片段 B 变量。
   * **b_index** (*Expr*)：被乘数片段 A 的索引。
   * **accumulator** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：累加器片段 C 变量。
   * **c_index** (*Expr*)：累加器片段 C 的索引。
   * **saturate** ([bool](https://docs.python.org/3/library/functions.html#bool))：输出处的可选饱和度。
   * **operator** (*Optional**[****Literal**[****"xor"**,*** ***"and"****]]*)：1 位运算符。
* **返回：call：** 调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_mma_sp(*dtype*, *shape*, *A_layout*, *B_layout*, *A_dtype*, *B_dtype*, *C_dtype*, *multiplicand_a*, *a_index*, *multiplicand_b*, *b_index*, *accumulator*, *c_index*, *metadata*, *meta_index*, *sparse_selector*, *saturate*)


TVM 稀疏张量核心 ptx 指令的内在函数 [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma)。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **shape** ([str](https://docs.python.org/3/library/stdtypes.html#str))：mma 片段的形状。
   * **A_layout** (*Literal**[****"row"**,*** ***"col"****]*)：被乘数片段 A 的布局。
   * **B_layout** (*Literal**[****"row"**,*** ***"col"****]*)：被乘数片段 B 的布局。
   * **A_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：被乘数片段 A 的数据类型。
   * **B_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：被乘数片段 B 的数据类型。
   * **C_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：被乘数片段 C 的数据类型。
   * **multiplicand_a** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：被乘数片段 A 变量。
   * **a_index** (*Expr*)**：** 被乘数片段 A 的索引。
   * **multiplicand_b** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：被乘数片段 B 变量。
   * **b_index** (*Expr*)：被乘数片段 B 的索引。
   * **accumulator** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))*：* 累加器片段 C 变量。
   * **c_index** (*Expr*)：累加器片段 C 的索引。
   * **metadata** (*Expr*)：操作数的元数据。
   * **meta_index** (*Expr*)：操作数的元数据索引。
   * **sparse_selector** (*Expr*)：指示存储元数据的线程的稀疏选择器。
   * **saturate** ([bool](https://docs.python.org/3/library/functions.html#bool))：输出处的可选饱和度。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.mma_store(*dtype*, *m*, *n*, *dst_ptr*, *src_ptr*, *src_offset*, *dst_stride*)

TVM 内部函数，用于将 PTX MMA 的结果存储到目标指针中。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))**：** 结果的数据类型。
   * **m** ([IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none))**：** mma 片段的形状。
   * **n** ([IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none))：mma 片段的形状。
   * **dst_ptr** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))*：* 目标指针变量。
   * **src_ptr** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))*：* 源指针变量。
   * **src_offset** (*Expr*)*：* 源偏移量。
   * **dst_stride** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：目标步幅。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.mma_fill(*dtype*, *local_size*, *local_ptr*, *offset*)


TVM 内在函数，用于对 MMA 累积寄存器进行零初始化。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))**：** 结果的数据类型。
   * **local_size** ([IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none))：元素的数量。
   * **local_ptr** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))**：** 目标指针变量。
   * **offset** (*Expr*)**：** 目标偏移量。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_ldmatrix(*dtype*, *trans*, *num*, *type*, *local_ptr*, *local_offset*, *smem_ptr*, *smem_offset*)


TVM 内部函数，用于从共享内存中加载 ptx 矩阵 [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix)。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **trans** ([bool](https://docs.python.org/3/library/functions.html#bool))：矩阵以列主格式加载。
   * **num** ([IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none))：矩阵的数量。
   * **type** (*Literal**[****".b16"]*)：矩阵的数据类型。
   * **local_ptr** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：本地指针变量。
   * **local_offset** (*Expr*)：本地指针的偏移量。
   * **smem_ptr** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：共享内存指针变量。
   * **smem_offset** (*Expr*)：共享内存指针的偏移量。
* **返回：call：** 调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_cp_async(*dtype*, *shared_ptr*, *shared_offset*, *global_ptr*, *global_offset*, *bytes*)


TVM 内部使用 cp.async 将 ptx 异步复制到从全局到共享内存 [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **shared_ptr** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))**：** 共享内存指针变量。
   * **shared_offset** (*Expr*)：共享内存指针的偏移量。
   * **global_ptr** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：全局内存指针变量。
   * **global_offset** (*Expr*)：全局内存指针的偏移量。
   * **bytes** ([int](https://docs.python.org/3/library/functions.html#int))：要复制的数据大小。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

## tvm.tir.ptx_cp_async_bulk(*dtype*, *shared_ptr*, *shared_offset*, *global_ptr*, *global_offset*, *bytes*, *barrier_id*)

TVM 使用 cp.async.bulk 将 ptx 异步复制到从全局到共享内存的内在 [函数 https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **shared_ptr** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：共享内存指针变量。
   * **shared_offset** (*Expr*)：共享内存指针的偏移量。
   * **global_ptr** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：全局内存指针变量。
   * **global_offset** (*Expr*)**：** 全局内存指针的偏移量。
   * **bytes** ([int](https://docs.python.org/3/library/functions.html#int))**：** 要复制的数据大小。
   * **barrier_id** ([int](https://docs.python.org/3/library/functions.html#int))：屏障共享内存指针的 ID。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_commit_group()


TVM ptx 异步复制提交内在函数 [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group)。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_wait_group(*num*)

TVM 内部用于 ptx 异步复制等待 [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group)。
* **参数：num** ([int](https://docs.python.org/3/library/functions.html#int))：要等待的最近未提交的待处理 cp.async 组的数量。
* **返回：call：** 调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_cp_async_barrier(*barrier_id*)


TVM 使用 cp.async.mbarrier.arrive 实现 ptx 异步复制屏障的内在机制 [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)。
* **参数：barrier_id** ([int](https://docs.python.org/3/library/functions.html#int))：屏障共享内存指针的 ID。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_init_barrier_thread_count(*barrier_id*, *thread_count*)


TVM 使用 mbarrier.init 来初始化线程数的 ptx 屏障 [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init)。
* **参数：**
   * **barrier_id** ([int](https://docs.python.org/3/library/functions.html#int))：屏障共享内存指针的 ID。
   * **thread_count** ([int](https://docs.python.org/3/library/functions.html#int))：预计到达屏障的线程数。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_arrive_barrier(*barrier_id*)


TVM 使用 mbarrier.arrive 实现 ptx 屏障到达的内在机制 [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)。
* **参数：barrier_id** ([int](https://docs.python.org/3/library/functions.html#int))：屏障共享内存指针的 ID。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_arrive_barrier_expect_tx(*barrier_id*, *byte_count*)


TVM 内在函数，用于使用 mbarrier.arrive.expect_tx 实现 ptx 屏障到达并期望 tx [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive) [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)。
* **参数：**
   * **barrier_id** ([int](https://docs.python.org/3/library/functions.html#int))：屏障共享内存指针的 ID。
   * **byte_count** ([int](https://docs.python.org/3/library/functions.html#int))：增加 mbarrier 对象的 tx 计数以跟踪附加异步事务的完成情况。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ptx_wait_barrier(*barrier_id*)


TVM 使用 mbarrier.try_wait 等待 ptx 屏障的内在机制 [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)。
* **参数：barrier_id** ([int](https://docs.python.org/3/library/functions.html#int))：屏障共享内存指针的 ID。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.create_barriers(*barrier_count*)


TVM 固有创建 N 个屏障。
* **参数：barrier_count** ([int](https://docs.python.org/3/library/functions.html#int))：要创建的障碍数量。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.make_filled_simdgroup_matrix(*d:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *index:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *value:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *col:*[int](https://docs.python.org/3/library/functions.html#int)*= 8*, *row:*[int](https://docs.python.org/3/library/functions.html#int)*= 8*)

创建填充的 SIMDGroup 矩阵。
* **参数：**
   * **d** (*var*)：simdgroup var。
   * **index** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 矩阵的索引。
   * **value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：要填充的值。
   * **col** ([int](https://docs.python.org/3/library/functions.html#int))*：* 列数。
   * **row** ([int](https://docs.python.org/3/library/functions.html#int))：行数。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.simdgroup_load(*d:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *index:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *ptr:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *stride:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *col:*[int](https://docs.python.org/3/library/functions.html#int)*= 8*, *row:*[int](https://docs.python.org/3/library/functions.html#int)*= 8*, *transpose_matrix:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)

将数据从设备内存或线程组内存加载到 simdgroup。
* **参数：**
   * **d** (*var*)**：** simdgroup var。
   * **index** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：矩阵的索引。
   * **ptr** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 指针。
   * **stride** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：步幅。
   * **col** ([int](https://docs.python.org/3/library/functions.html#int))：列数。
   * **row** ([int](https://docs.python.org/3/library/functions.html#int))：行数。
   * **transpose_matrix** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置矩阵。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.simdgroup_multiply_accumulate(*d:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *index_d:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *a:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *index_a:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *b:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *index_b:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *c:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *index_c:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))

在 simdgroup 中对两个矩阵进行乘法和累加，即 d = a * b + c。
* **参数：**
   * **d** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：目标矩阵。
   * **index_d** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：目标矩阵的索引。
   * **a** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))**：** 第一个矩阵。
   * **index_a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：第一个矩阵的索引。
   * **b** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：第二个矩阵。
   * **index_b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：第二个矩阵的索引。
   * **c** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：第三个矩阵。
   * **index_c** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：第三个矩阵的索引。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.simdgroup_store(*d:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *index:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *ptr:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *stride:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *col:*[int](https://docs.python.org/3/library/functions.html#int)*= 8*, *row:*[int](https://docs.python.org/3/library/functions.html#int)*= 8*, *transpose_matrix:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)

将数据从 simdgroup 存储到设备内存或线程组内存。
* **参数：**
   * **d** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：SIMDGroup。
   * **index** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：矩阵的索引。
   * **ptr** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：指针。
   * **stride** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：步幅。
   * **col** ([int](https://docs.python.org/3/library/functions.html#int))：列数。
   * **row** ([int](https://docs.python.org/3/library/functions.html#int))：行数。
* **transpose_matrix：bool**

     是否转置矩阵。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.vectorlow(*dtype*, *vec*)


获取向量的低位一半。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **vec** ([list](https://docs.python.org/3/library/stdtypes.html#list))：输入向量。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.vectorhigh(*dtype*, *vec*)


获取向量的高位一半。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **vec** ([list](https://docs.python.org/3/library/stdtypes.html#list))：输入向量。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.vectorcombine(*dtype*, *vec1*, *vec2*)
连接两个向量。
* **参数：**
   * **vec1** ([list](https://docs.python.org/3/library/stdtypes.html#list))：输入向量。
   * **vec2** ([list](https://docs.python.org/3/library/stdtypes.html#list))：输入向量。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.infinity(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)


数据类型的无穷大值。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：数据类型。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：value**：dtype 的无穷大值。
* **返回类型：** tvm.Expr。

## tvm.tir.reinterpret(*dtype*, *value*, *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

数据类型的无穷大值。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：数据类型。
   * **value** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入值。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：value**：重新解释 dtype 的转换值。
* **返回类型：** tvm.Expr。

## tvm.tir.exp(*x*)


取输入 x 的指数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.exp2(*x*)

计算 2**x。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

## tvm.tir.exp10(*x*)


计算 10**x。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.log(*x*)


对输入 x 取对数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** *：* 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

## tvm.tir.log2(*x*)


对输入 x 取 log2。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.log10(*x*)


对输入 x 取 log10。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.log1p(*x*)


对输入 x 取 log(x + 1)。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ldexp(*x1*, *x2*)


返回 x1 * (2 ** x2)。
* **参数：**
   * **x1** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **x2** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.clz(*x*)


计算整数 x 的前导零位。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入 32 位或 64 位整数。如果输入为 0，则结果未定义。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.sin(*x*)

对输入 x 取正弦值。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.sinh(*x*)


对输入 x 取 sinh。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

## tvm.tir.asin(*x*)


取输入 x 的 asin。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.asinh(*x*)


取输入 x 的正弦值。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.cos(*x*)


取输入 x 的 cos。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** *：* 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.cosh(*x*)


对输入 x 取余弦值。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.acos(*x*)


对输入 x 取余数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.acosh(*x*)


对输入 x 取余数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tan(*x*)

对输入 x 取 tan。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.tanh(*x*)

对输入 x 取双曲 tanh。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.atan(*x*)


对输入 x 取正切值。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.atan2(*x1*, *x2*)


取 arctan2(x1, x2)。
* **参数：**
   * **x1** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **x2** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.atanh(*x*)


对输入 x 进行 atanh 处理。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.bitwise_and(*x*, *y*, *span=None*)


对两个值进行按位与运算。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左操作数。
   * **y** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：res**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.bitwise_not(*x*, *span=None*)


对输入值进行按位非。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：res**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.bitwise_or(*x*, *y*, *span=None*)

对两个值进行按位或操作。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左操作数。
   * **y** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：res**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.bitwise_xor(*x*, *y*, *span=None*)


对两个值进行按位异或。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左操作数。
   * **y** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 右操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：res**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.erf(*x*)


取输入 x 的高斯误差函数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** *：* 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.sigmoid(*x*)

快速获取 S 形函数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.sqrt(*x*)


对输入 x 取平方根。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.rsqrt(*x*)


取输入 x 的平方根的倒数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.floor(*x: PrimExprWithOp*, *span=None*)

取浮点输入 x 的下限。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ceil(*x*, *span=None*)

对浮点输入 x 取上限。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)**：** 此运算符在源代码中的位置。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.hypot(*x1*, *x2*)


相当于 sqrt(x1**2 + x2**2)，逐个元素。
* **参数：**
   * **x1** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **x2** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y：** 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.trunc(*x*, *span=None*)


获取输入的截断值。


标量 x 的截断值是最接近的整数 i，它比 x 更接近零。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.abs(*x*, *span=None*)


逐个获取输入元素的绝对值。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)**：** 此运算符在源代码中的位置。
* **返回：y：** 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.round(*x*, *span=None*)


将数组元素四舍五入为最接近的整数。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：y** *：* 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.nextafter(*x1*, *x2*)


返回在 x1 和 x2 之间，比 x1 更接近 x2 的下一个浮点数。
* **参数：**
   * **x1** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **x2** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：*输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.nearbyint(*x*, *span=None*)


将数组元素四舍五入为最接近的整数。此内在函数使用 llvm.nearbyint 而不是 llvm.round，后者速度更快，但结果与 te.round 不同。值得注意的是，nearbyint 根据舍入模式进行舍入，而 te.round (llvm.round) 则忽略该模式。有关两者之间的差异，请参阅： https: [//en.cppreference.com/w/cpp/numeric/math/round](https://en.cppreference.com/w/cpp/numeric/math/round) [https://en.cppreference.com/w/cpp/numeric/math/nearbyint](https://en.cppreference.com/w/cpp/numeric/math/nearbyint)。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.power(*x*, *y*, *span=None*)


x 次方 y。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **y** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：指数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：z**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.pow(*x*, *y*, *span=None*)


x 次方 y。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 输入参数。
   * **y** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：指数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：z**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.popcount(*x*)

计算输入 x 中设置位的数量。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.fmod(*x*, *y*)

返回 x 除以 y 后的余数，其符号与 x 相同。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **y** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：z：** 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.if_then_else(*cond*, *t*, *f*, *span=None*)

条件选择表达式。
* **参数：**
   * **cond** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：条件。
   * **t** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 如果 cond 为真，则结果表达式。
   * **f** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：如果 cond 为假，则结果表达式。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：result**：条件表达式的结果。
* **返回类型：**[Node](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirnode)

:::Note

与 Select 不同，if_then_else 不会执行不满足条件的分支。您可以使用它来防止越界访问。与 Select 不同，如果向量中某些通道的条件不同，则 if_then_else 无法进行向量化。

:::

## tvm.tir.likely(*cond*, *span=None*)


将情况标记为可能。
* **参数：**
   * **cond** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：y** *：* 标记的表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.isnan(*x*, *span=None*)


检查输入值是否为 Nan。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.isnullptr(*x*, *span=None*)


检查输入值是否为 nullptr。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.isfinite(*x*, *span=None*)


检查输入值是否有限。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.isinf(*x*, *span=None*)


检查输入值是否无限。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源代码中的位置。
* **返回：y：** 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.copysign(*x1*, *x2*)


逐个元素地将 x1 的符号更改为 x2 的符号。
* **参数：**
   * **x1** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：*输入参数。
   * **x2** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.div(*a*, *b*, *span=None*)


按照 C/C++ 语义计算 a / b。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左侧操作数，已知为非负数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数，已知为非负。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

当操作数为整数时，返回 truncdiv(a, b, span)。

:::

## tvm.tir.indexdiv(*a*, *b*, *span=None*)

计算 floor(a / b)，其中 a 和 b 为非负数。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左侧操作数，已知为非负数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数，已知为非负。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)*：* 此运算符在源中的位置。
* **返回：res** *：* 结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

使用此函数拆分非负索引。此函数可以利用操作数的非负性。

:::

## tvm.tir.indexmod(*a*, *b*, *span=None*)


计算 indexdiv 的余数。a 和 b 非负。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 左侧操作数，已知为非负数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数，已知为非负。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

使用此函数拆分非负索引。此函数可以利用操作数的非负性。

:::

## tvm.tir.truncdiv(*a*, *b*, *span=None*)


计算两个表达式的 truncdiv。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左侧操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

这是 C 语言中的默认整数除法行为。

:::

## tvm.tir.truncmod(*a*, *b*, *span=None*)

计算两个表达式的 truncmod。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 左侧操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

这是 C 语言中的默认整数除法行为。

:::

## tvm.tir.floordiv(*a*, *b*, *span=None*)


计算两个表达式的 floordiv。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左侧操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.floormod(*a*, *b*, *span=None*)


计算两个表达式的 floormod。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左侧操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ceildiv(*lhs*, *rhs*, *span=None*)


通用 ceildiv 运算符。
* **参数：**
   * **lhs** ([object](https://docs.python.org/3/library/functions.html#object))：左操作数。
   * **rhs** ([object](https://docs.python.org/3/library/functions.html#object))*：* 右操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：op**：ceildiv 运算的结果 Expr。
* **返回类型：** tvm.Expr。

## tvm.tir.logaddexp(*a*, *b*, *span=None*)


计算两个表达式的 logaddexp。
* **参数：**
   * **a** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：左侧操作数。
   * **b** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：右侧操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.comm_reducer(*fcombine*, *fidentity*, *name='reduce'*)


创建一个交换减速器用于减速。
* **参数：**
   * **fcombine** (*function**(****Expr -> Expr -> Expr)*)：一个二元函数，以两个 Expr 作为输入并返回一个 Expr。
   * **fidentity** (*function**(****str -> Expr)*)：以字符串类型作为输入并返回 const Expr 的函数。

**返回：reducer**：在 axis 上创建 Reduce 表达式的函数。有两种使用方法：
   * accept (expr, axis, where) 来在指定轴上生成一个 Reduce Expr。
   * 直接使用多个 Exprs。
* **返回类型：** function。


**示例**

```python
n = te.var("n")
m = te.var("m")
mysum = te.comm_reducer(lambda x, y: x+y,
    lambda t: tvm.tir.const(0, dtype=t), name="mysum")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), name="k")
B = te.compute((n,), lambda i: mysum(A[i, k], axis=k), name="B")
```
## tvm.tir.min(*expr*, *axis*, *where=None*, *init=None*, args*)


在轴上创建最小表达式。
* **参数：**
   * **expr** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：源表达式。
   * **axis** ([IterVar](range-varvarstr-iter_typeint-thread_tagstr--spanspannone-none))*：* 缩减 IterVar 轴。
   * **where** (*optional,Expr*)：归约操作的过滤条件。
* **返回：value**：结果值。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**示例**

```python
m = te.var("m")
n = te.var("n")
A = te.placeholder((m, n), name="A")
k = te.reduce_axis((0, n), name="k")

# 使用这个最小值归约器（min reducer）有两种方式：
# 方式 1：接受 (expr, axis, where) 参数来生成一个归约表达式（Reduce Expr）
# tvm.min 表示 tvm.te.min 或 tvm.tir.min。

B = te.compute((m,), lambda i: tvm.min(A[i, k], axis=k), name="B")

# 方式 2：直接用于多个表达式：
min_res = tvm.min(m, n)
```
## tvm.tir.max(*expr*, *axis*, *where=None*, *init=None*, args*)


在轴上创建最大表达式。
* **参数：**
   * **expr** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：源表达式。
   * **axis** ([IterVar](range-varvarstr-iter_typeint-thread_tagstr--spanspannone-none))：缩减 IterVar 轴。
   * **where** (*optional,Expr*)*：*归约操作的过滤谓词。
* **返回：value**：结果值。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**示例**

```python
m = te.var("m")
n = te.var("n")
A = te.placeholder((m, n), name="A")
k = te.reduce_axis((0, n), name="k")

# 使用这个最大值归约器（max reducer）有两种方式：
# 方式 1：接受 (expr, axis, where) 参数来生成一个归约表达式（Reduce Expr）
# tvm.max 表示 tvm.te.max 或 tvm.tir.max。

B = te.compute((m,), lambda i: tvm.max(A[i, k], axis=k), name="B")

# 方式 2：直接用于多个表达式：
max_res = tvm.max(m, n)
```
## tvm.tir.sum(*expr*, *axis*, *where=None*, *init=None*, args*)


在轴上创建一个求和表达式。
* **参数：**
   * **expr** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：源表达式。
   * **axis** ([IterVar](range-varvarstr-iter_typeint-thread_tagstr--spanspannone-none))：缩减 IterVar 轴。
   * **where** (*optional,Expr*)：归约操作的过滤条件。
* **返回：value**：结果值。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**示例**

```python
m = te.var("m")
n = te.var("n")
A = te.placeholder((m, n), name="A")
k = te.reduce_axis((0, n), name="k")

# 使用这个求和归约器（sum reducer）有两种方式：
# 方式 1：接受 (expr, axis, where) 参数来生成一个归约表达式（Reduce Expr）
# tvm.sum 表示 tvm.te.sum 或 tvm.tir.sum。

B = te.compute((m,), lambda i: tvm.sum(A[i, k], axis=k), name="B")

# 方式 2：直接用于多个表达式：
sum_res = tvm.sum(m, n)
```
## tvm.tir.q_multiply_shift(*x*, *y*, *q*, *s*)


对两个 Q 数 x 和 y 执行乘法，然后右移 s。数学表达式为：

out = round(x*y*2^-s) 输出 = 舍入（x*y*2^-s） 。


有关 Q 数的更多信息，请参见：[https://en.wikipedia.org/wiki/Q_(number_format](https://en.wikipedia.org/wiki/Q*(number_format)) 舍入规则是舍入到最接近的值，将一半向上舍入（即，round(x.1) = x 和 round (x.5) = x+1）。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 第一个 Q 数。
   * **y** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：第二个 Q 数。
   * **q** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：x 和 y 中的小数位数。必须大于 0。
   * **s** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 数移位。
* **返回：y**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.q_multiply_shift_per_axis(*x:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *y:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *ls:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *rs:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *q:*[IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none), *is_lshift_required:*[IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none), *is_rshift_required:*[IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none))


执行两个 Q 数字 x 和 y 之间的乘法。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 第一个 Q 数。
   * **y** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：第二个 Q 号。
   * **ls** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：整数左移。
   * **rs** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 整数右移。
   * **q** ([IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none))**：** x 和 y 的小数位数。必须大于 0。
   * **is_lshift_required** ([IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none))：我们是否需要进行左移。
   * **is_rshift_required** ([IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none))：我们是否需要进行右移。
* **返回：z**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.shift_left(*x*, *y*, *span=None*)


返回 x 左移 y 位的结果。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **y** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))*：* 输入参数。
* **返回：z：** 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.shift_right(*x*, *y*, *span=None*)


返回 x 右移 y 位的结果。
* **参数：**
   * **x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
   * **y** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：z**：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.TVMBackendAllocWorkspace(*device_type*, *device_id*, *nbytes*, *dtype_code_hint*, *dtype_bits_hint*)


分配临时工作空间的后端函数。
* **参数：**
   * **device_type** ([int](https://docs.python.org/3/library/functions.html#int))：将分配空间的设备类型。
   * **device_id** ([int](https://docs.python.org/3/library/functions.html#int))：将分配空间的设备 ID。
   * **nbytes** ([int](https://docs.python.org/3/library/functions.html#int))：请求的空间大小。
   * **dtype_code_hint** ([int](https://docs.python.org/3/library/functions.html#int))：数组元素的类型代码。仅在某些后端（例如 OpenGL）中使用。
   * **dtype_bits_hint** ([int](https://docs.python.org/3/library/functions.html#int))**：** 数组元素的类型位。仅在某些后端（例如 OpenGL）中使用。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.TVMBackendFreeWorkspace(*device_type*, *device_id*, *ptr*)


后端函数用于释放临时工作空间。
* **参数：**
   * **device_type** ([int](https://docs.python.org/3/library/functions.html#int))：将分配空间的设备类型。
   * **device_id** ([int](https://docs.python.org/3/library/functions.html#int))：将分配空间的设备 ID。
   * **ptr** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：结果分配的空间指针。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.start_profile_intrinsic(*id*)


启动配置文件内在。：param id：内在 id。：type id：int。
* **返回：call：** 调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.end_profile_intrinsic(*id*)


结束配置文件内在：param id：内在 id：type id：int。
* **返回：call***：*调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.vscale()


获取目标的 vscale 值。它将被降低到 llvm.vscale 内部函数 ( [https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic](https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic) ) :returns: **call** – tir.Call 到 vscale 内部函数 :rtype: PrimExpr。

## tvm.tir.get_active_lane_mask(*dtype*, *base*, *limit*)


给定上限（限制）和当前值（基数）计算谓词掩码。


它将被降低到 llvm.get.active.lane.mask 内在函数。（[https://llvm.org/docs/LangRef.html#llvm-get-active-lane-mask-intrinsics](https://llvm.org/docs/LangRef.html#llvm-get-active-lane-mask-intrinsics)）。
* **参数：**
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：结果的数据类型。
   * **base** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：表示基数的表达式。
   * **limit** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：表示限制的表达式。

## tvm.tir.get_vscale_expr(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype*, *min_size:*[int](https://docs.python.org/3/library/functions.html#int)*= 128*) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


创建依赖于数据类型的可扩展表达式。
* **参数：**
   * **dtype** (*Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***tvm.DataType****]*)：元素数据类型。
   * **min_size** ([int](https://docs.python.org/3/library/functions.html#int))：可缩放矢量的最小尺寸（以位为单位）。

## tvm.tir.dp4a(*vec1*, *vec2*, *acc=0*)


两个 int8x4 向量的点积并添加一个可选累加器。
* **参数：**
   * **vec1** (*int8x4*)：输入向量。
   * **vec2** (*int8x4*)：输入向量。
   * **acc** (*int32*)：累加器。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.tir.ignore_loop_partition(*predicate*) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


注解谓词不被视为循环分区的目标条件。
* **参数：predicate** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：带注解的谓词表达式。

## tvm.tir.add(*lhs*, *rhs*, *span=None*)


通用加法运算符。
* **参数：**
   * **lhs** ([object](https://docs.python.org/3/library/functions.html#object))：左操作数。
   * **rhs** ([object](https://docs.python.org/3/library/functions.html#object))：右操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：op**：加法运算的结果 Expr。
* **返回类型：** tvm.Expr。

## tvm.tir.subtract(*lhs*, *rhs*, *span=None*)


通用减法运算符。
* **参数：**
   * **lhs** ([object](https://docs.python.org/3/library/functions.html#object))：左操作数。
   * **rhs** ([object](https://docs.python.org/3/library/functions.html#object))：右操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：op**：减法运算的结果 Expr。
* **返回类型：** tvm.Expr。

## tvm.tir.multiply(*lhs*, *rhs*, *span=None*)


通用乘法运算符。
* **参数：**
   * **lhs** ([object](https://docs.python.org/3/library/functions.html#object))：左操作数。
   * **rhs** ([object](https://docs.python.org/3/library/functions.html#object))：右操作数。
   * **span** (*Optional[*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*)：此运算符在源中的位置。
* **返回：op**：乘法运算的结果 Expr。
* **返回类型：** tvm.Expr。

## *class* tvm.tir.BlockDependenceInfo(*mod:*[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)*|*[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))

使用两个核心对象 BlockScope 和 StmtSRef 帮助构建和查询块级依赖关系的对象。


公开的数据结构包括：1）sref2scope：从 srefs 映射到其对应的 BlockScope 2）stmt2ref：从块映射到对应的 StmtSRefs。


请注意，此对象不存储循环的 SRef，因为其目的仅用于公开块级依赖关系。这带来了一个优势：给定块 sref 的作用域块（父块）可以直接通过 sref->parent 进行访问。

### get_sref(*block:*[Block](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)) → [StmtSRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref) | [None](https://docs.python.org/3/library/constants.html#None)


返回指向该块的相应 sref
* **参数：stmt** ([Block](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none))：要检索 sref 的块。
* **返回：sref**：对应的 sref/
* **返回类型：**[StmtSRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)/

### get_block_scope(*block_sref:*[StmtSRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)) → [BlockScope](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockscope)


获取与块 sref 对应的 BlockScope/。
* **参数：block_sref** ([StmtSRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref))：要检索的块 sref。
* **返回：scope**：对应的 BlockScope。
* **返回类型：**[StmtSRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)。

## tvm.tir.build(*mod:*[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*|*[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone), *target:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Target](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *pipeline:*[None](https://docs.python.org/3/library/constants.html#None)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Pass](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)*= 'default'*)


构建一个带有签名的函数，为结合目标信息的设备生成代码。
* **参数：**
   * **mod** (*Union[*[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*,*[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)*]*)**：** 要构建的输入。
   * **target** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** [Target](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone)***]****]*)：编译的目标。
   * **pipeline** (*Union**[****None,*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[tvm.transform.Pass](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)*]*)：用于编译的管道。
* **返回：** 结合主机和设备代码的模块。
* **返回类型：** tvm.runtime.Module。

## tvm.tir.get_tir_pipeline(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'default'*, ***kwargs*) → [Pass](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


按名称获取预构建管道。
* **参数：name** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：管道的名称。

## tvm.tir.get_default_tir_pipeline(*target:*[Target](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone)) → [Pass](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


获取给定目标的默认 TIR 管道。

## *class* tvm.tir.PyStmtExprVisitor


Python StmtExprVisitor 用于为 Stmt 和 PrimExpr 定义自定义访问者。


用户可以自定义任意的访问函数。

### visit_stmt(*stmt:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)) → [None](https://docs.python.org/3/library/constants.html#None)


访问AttrStmt。
* **参数：stmt** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：要访问的 Stmt。

### visit_expr(*expr:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)) → [None](https://docs.python.org/3/library/constants.html#None)
访问 PrimExpr。
* **参数：expr** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：要访问的 PrimExpr。

### visit_attr_stmt_(*op:*[AttrStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirattrstmtnode-object-attr_keystr-valueprimexpr-bodystmt-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 AttrStmt，用户可以自定义该函数，在 C++ 端覆盖 VisitStmt_(const AttrStmtNode* op)。
* **参数：op** ([AttrStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirattrstmtnode-object-attr_keystr-valueprimexpr-bodystmt-spanspannone-none))：要访问的 AttrStmt。

### visit_if_then_else_(*op:*[IfThenElse](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirifthenelseconditionprimexpr-then_casestmt-else_casestmtnone-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 IfThenElse，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const IfThenElseNode* op)。
* **参数：op** ([IfThenElse](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirifthenelseconditionprimexpr-then_casestmt-else_casestmtnone-spanspannone-none))：要访问的 IfThenElse。

### visit_let_stmt_(*op:*[LetStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirletstmtvarvar-valueprimexpr-bodystmt-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 LetStmt。用户可以自定义此函数，在 C++ 端覆盖 VisitStmt_(const LetStmtNode* op)。
* **参数：op** ([LetStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirletstmtvarvar-valueprimexpr-bodystmt-spanspannone-none))：要访问的 LetStmt。

### visit_for_(*op:*[For](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 For，用户可以自定义该函数，在 C++ 端覆盖 VisitStmt_(const ForNode* op)。
* **参数：op** ([For](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none))：要访问的 For。

### visit_while_(*op:*[While](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirwhileconditionprimexpr-bodystmt-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 While。用户可以自定义此函数，在 C++ 端覆盖 VisitStmt_(const WhileNode* op)。
* **参数：op** ([While](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirwhileconditionprimexpr-bodystmt-spanspannone-none))：需要访问的 While 部分。

### visit_allocate_(*op:*[Allocate](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirallocatebuffer_varvar-dtypestr-extentslistprimexpr-conditionprimexpr-bodystmt-annotationsmappingstr-object-none-none-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)

Visit Allocate，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const AllocateNode* op)。
* **参数：op** ([Allocate](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirallocatebuffer_varvar-dtypestr-extentslistprimexpr-conditionprimexpr-bodystmt-annotationsmappingstr-object-none-none-spanspannone-none))：要访问的分配。

### visit_allocate_const_(*op:*[AllocateConst](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirallocateconstbuffer_varvar-dtypestr-extentslistprimexpr-data_or_idxndarrayint-bodystmt-annotationsmappingstr-object-none-none-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None) 


访问 AllocateConst，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const AllocateConstNode* op)。
* **参数：op** ([AllocateConst](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirallocateconstbuffer_varvar-dtypestr-extentslistprimexpr-data_or_idxndarrayint-bodystmt-annotationsmappingstr-object-none-none-spanspannone-none))：要访问的 AllocateConst。

### visit_decl_buffer_(*op:*[DeclBuffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirdeclbufferbufferbuffer-bodystmt-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 DeclBuffer，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const DeclBufferNode* op)。
* **参数：op** ([DeclBuffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirdeclbufferbufferbuffer-bodystmt-spanspannone-none))：要访问的 DeclBuffer。

### visit_buffer_store_(*op:*[BufferStore](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferStore)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 BufferStore，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const BufferStoreNode* op)。
* **参数：op** ([BufferStore](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferStore))：要访问的 BufferStore。

### visit_buffer_realize_(*op:*[BufferRealize](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferRealize)) → [None](https://docs.python.org/3/library/constants.html#None) 


访问 BufferRealize，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const BufferRealizeNode* op)。
* **参数：op** ([BufferRealize](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferRealize))：要访问的 BufferRealize。

### visit_assert_stmt_(*op:*[AssertStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirattrstmtnode-object-attr_keystr-valueprimexpr-bodystmt-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 AssertStmt，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const AssertStmtNode* op)。
* **参数：op** ([AssertStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirattrstmtnode-object-attr_keystr-valueprimexpr-bodystmt-spanspannone-none))：要访问的 AssertStmt。

### visit_seq_stmt_(*op:*[SeqStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirseqstmtseqliststmt-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 SeqStmt，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const SeqStmtNode* op)。
* **参数：op** ([SeqStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirseqstmtseqliststmt-spanspannone-none))：要访问的 SeqStmt。

### visit_evaluate_(*op:*[Evaluate](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirevaluatevalueprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


Visit Evaluate，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const EvaluateNode* op)。
* **参数：op** ([Evaluate](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirevaluatevalueprimexpr-spanspannone-none))：要访问的评估。

### visit_block_(*op:*[Block](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问区块。用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const BlockNode* op)。
* **参数：op** ([Block](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none))：要访问的区块。

### visit_block_realize_(*op:*[BlockRealize](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-noneRealize)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 BlockRealize。用户可以自定义此函数，在 C++ 端覆盖 VisitStmt_(const BlockRealizeNode* op)。
* **参数：op** ([BlockRealize](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-noneRealize))：要访问的 BlockRealize。

### visit_var_(*op:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


参观 Var。


用户可以自定义该函数，在 C++端覆盖 VisitVar_(const VarNode* op)。
* **参数：op** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))：要访问的 tir.Var。

### visit_size_var_(*op:*[SizeVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirsizevarnamestr-dtypestrtype-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 SizeVar。


用户可以自定义该函数，在 C++端覆盖 VisitSizeVar_(const SizeVarNode* op)。
* **参数：op** ([SizeVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirsizevarnamestr-dtypestrtype-spanspannone-none))：要访问的 SizeVar。

### visit_buffer_load_(*op:*[BufferLoad](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferLoad)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 BufferLoad。


用户可以自定义该函数，在 C++端覆盖 VisitBufferLoad_(const BufferLoadNode* op)。
* **参数：op** ([BufferLoad](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferLoad))：要访问的 BufferLoad。

### visit_producer_load_(*op:*[ProducerLoad](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirproducerloadproducerdataproducer-indiceslistprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None) 


访问 ProducerLoad。


用户可以自定义该函数，在 C++端覆盖 VisitProducerLoad_(const ProducerLoadNode* op)。
* **参数：op** ([ProducerLoad](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirproducerloadproducerdataproducer-indiceslistprimexpr-spanspannone-none))：要访问的 ProducerLoad。

### visit_let_(*op:*[Let](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirletvarvar-valueprimexpr-bodyprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 Let。


用户可以自定义该函数，在 C++端覆盖 VisitLet_(const LetNode* op)。
* **参数：op** ([Let](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirletvarvar-valueprimexpr-bodyprimexpr-spanspannone-none))：要访问的 Let。

### visit_call_(*op:*[Call](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtircalldtypestr-opopstr-argslistprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)
访问调用。


用户可以自定义该函数，在 C++端覆盖 VisitCall_(const CallNode* op)。
* **参数：op** ([tir.Call](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtircalldtypestr-opopstr-argslistprimexpr-spanspannone-none))：要访问的 tir.Call。

### visit_add_(*op:*[Add](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiraddaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问添加。


用户可以自定义该函数，在 C++端覆盖 VisitAdd_(const AddNode* op)。
* **参数：op** ([Add](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiraddaprimexpr-bprimexpr-spanspannone-none))：要访问的 Add。

### visit_sub_(*op:*[Sub](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirsubaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 Sub。


用户可以自定义该函数，在 C++端覆盖 VisitSub_(const SubNode* op)。
* **参数：op** ([Sub](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirsubaprimexpr-bprimexpr-spanspannone-none))：要访问的 Sub。

### visit_mul_(*op:*[Mul](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmulaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)

参观 Mul。


用户可以自定义该函数，在 C++端覆盖 VisitMul_(const MulNode* op)。
* **参数：op** ([Mul](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmulaprimexpr-bprimexpr-spanspannone-none))：要访问的 Mul。

### visit_div_(*op:*[Div](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirdivaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 Div。


用户可以自定义该函数，在 C++端覆盖 VisitDiv_(const DivNode* op)。
* **参数：op** ([Div](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirdivaprimexpr-bprimexpr-spanspannone-none))：要访问的 Div。

### visit_mod_(*op:*[Mod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmodaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 Mod。


用户可以自定义该函数，在 C++端覆盖 VisitMod_(const ModNode* op)。
* **参数：op** ([Mod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmodaprimexpr-bprimexpr-spanspannone-none))：要访问的 Mod。

### visit_floor_div_(*op:*[FloorDiv](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloordivaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 FloorDiv。


用户可以自定义该函数，在 C++端覆盖 VisitFloorDiv_(const FloorDivNode* op)。
* **参数：op** ([FloorDiv](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloordivaprimexpr-bprimexpr-spanspannone-none))：要访问的 FloorDiv。

### visit_floor_mod_(*op:*[FloorMod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloormodaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 FloorMod。


用户可以自定义该函数，在 C++端覆盖 VisitFloorMod_(const FloorModNode* op)。
* **参数：op** ([FloorMod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloormodaprimexpr-bprimexpr-spanspannone-none))：要访问的 FloorMod。

### visit_min_(*op:*[Min](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirminaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 Min。


用户可以自定义该函数，在 C++端覆盖 VisitMin_(const MinNode* op)。
* **参数：op** ([Min](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirminaprimexpr-bprimexpr-spanspannone-none))：要访问的 Min。

### visit_max_(*op:*[Max](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmaxaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 Max。


用户可以自定义该函数，在 C++端覆盖 VisitMax_(const MaxNode* op)。
* **参数：op** ([Max](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmaxaprimexpr-bprimexpr-spanspannone-none))：要访问的最大值。

### visit_eq_(*op:*[EQ](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtireqaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 EQ。


用户可以自定义该函数，在 C++端覆盖 VisitEQ_(const EQNode* op)。
* **参数：op** ([EQ](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtireqaprimexpr-bprimexpr-spanspannone-none))：要访问的 EQ。

### visit_ne_(*op:*[NE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirneaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 NE。


用户可以自定义该函数，在 C++端覆盖 VisitNE_(const NENode* op)。
* **参数：op** ([NE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirneaprimexpr-bprimexpr-spanspannone-none))：要访问的 NE。

### visit_lt_(*op:*[LT](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirltaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 LT。


用户可以自定义该函数，在 C++端覆盖 VisitLT_(const LTNode* op)。
* **参数：op** ([LT](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirltaprimexpr-bprimexpr-spanspannone-none))**：** 要访问的 LT。

### visit_le_(*op:*[LE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirleaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 LE。


用户可以自定义该函数，在 C++端覆盖 VisitLE_(const LENode* op)。
* **参数：op** ([LE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirleaprimexpr-bprimexpr-spanspannone-none))**：** 要访问的 LE。

### visit_gt_(*op:*[GT](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirgtaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 GT。


用户可以自定义该函数，在 C++端覆盖 VisitGT_(const GTNode* op)。
* **参数：op** ([GT](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirgtaprimexpr-bprimexpr-spanspannone-none))：要访问的 GT。

### visit_ge_(*op:*[GE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirgeaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 GE。


用户可以自定义该函数，在 C++端覆盖 VisitGE_(const GENode* op)。
* **参数：op** ([GE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirgeaprimexpr-bprimexpr-spanspannone-none))：要访问的 GE。

### visit_and_(*op:*[And](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirandaprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 And。


用户可以自定义该函数，在 C++端覆盖 VisitAnd_(const AndNode* op)。
* **参数：op** ([And](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirandaprimexpr-bprimexpr-spanspannone-none))：要访问的 And。

### visit_or_(*op:*[Or](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiroraprimexpr-bprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 Or。


用户可以自定义该函数，在 C++端覆盖 VisitOr_(const OrNode* op)。
* **参数：op** ([Or](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiroraprimexpr-bprimexpr-spanspannone-none))：要访问的 Or。

### visit_reduce_(*op:*[Reduce](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirreducecombinercommreducer-srclistprimexpr-rdomlistitervar-conditionprimexpr-value_indexint-initlistprimexpr-none-none-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 Reduce。


用户可以自定义该函数，在 C++端覆盖 VisitReduce_(const ReduceNode* op)。
* **参数：op** ([Reduce](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirreducecombinercommreducer-srclistprimexpr-rdomlistitervar-conditionprimexpr-value_indexint-initlistprimexpr-none-none-spanspannone-none))：要访问的 Reduce。

### visit_cast_(*op:*[Cast](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtircastdtype-value-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 Cast。


用户可以自定义该函数，在 C++端覆盖 VisitCast_(const CastNode* op)。
* **参数：op** ([Cast](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtircastdtype-value-spanspannone-none))：要访问的 Cast。

### visit_not_(*op:*[Not](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirnotaprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


不访问。


用户可以自定义该函数，在 C++端覆盖 VisitNot_(const NotNode* op)。
* **参数：op** ([Not](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirnotaprimexpr-spanspannone-none))：不可访问。

### visit_select_(*op:*[Select](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirselectconditionprimexpr-true_valueprimexpr-false_valueprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问选择。


用户可以自定义该函数，在 C++端覆盖 VisitSelect_(const SelectNode* op)。
* **参数：op** ([Select](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirselectconditionprimexpr-true_valueprimexpr-false_valueprimexpr-spanspannone-none))：要访问的选择。

### visit_ramp_(*op:*[Ramp](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirrampbaseprimexpr-strideprimexpr-lanesprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)

参观 Ramp。


用户可以自定义该函数，在 C++端覆盖 VisitRamp_(const RampNode* op)。
* **参数：op** ([Ramp](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirrampbaseprimexpr-strideprimexpr-lanesprimexpr-spanspannone-none))：要访问的坡道。

### visit_broadcast_(*op:*[Broadcast](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbroadcastvalueprimexpr-lanesprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问广播。


用户可以自定义该函数，在 C++端覆盖 VisitBroadcast_(const BroadcastNode* op)。
* **参数：op** ([Broadcast](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbroadcastvalueprimexpr-lanesprimexpr-spanspannone-none))：要访问的广播。

### visit_shuffle_(*op:*[Shuffle](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirshufflevectorslistprimexpr-indiceslistprimexpr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 Shuffle。


用户可以自定义该函数，在 C++端覆盖 VisitShuffle_(const ShuffleNode* op)。
* **参数：op** ([Shuffle](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirshufflevectorslistprimexpr-indiceslistprimexpr-spanspannone-none))：要访问的 Shuffle。

### visit_int_imm_(*op:*[IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None) 


访问 IntImm。


用户可以自定义该函数，在 C++端覆盖 VisitIntImm_(const IntImmNode* op)。
* **参数：op** ([IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none))：要访问的 IntImm。

### visit_float_imm_(*op:*[FloatImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloatimmdtypestr-valuefloat-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)

访问 FloatImm。


用户可以自定义该函数，在 C++端覆盖 VisitFloatImm_(const FloatImmNode* op)。
* **参数：op** ([FloatImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloatimmdtypestr-valuefloat-spanspannone-none))**：** 要访问的 FloatImm。

### visit_string_imm_(*op:*[StringImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstringimmvaluestr-spanspannone-none)) → [None](https://docs.python.org/3/library/constants.html#None)


访问 StringImm。


用户可以自定义该函数，在 C++端覆盖 VisitStringImm_(const StringImmNode* op)。
* **参数：op** ([StringImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstringimmvaluestr-spanspannone-none))：要访问的 StringImm。

## *class* tvm.tir.PyStmtExprMutator

Python StmtExprMutator 用于为 Stmt 和 PrimExpr 定义自定义变量。


用户可以自定义任意的访问函数。

### visit_expr(*expr:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 PrimExpr。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr(const PrimExpr& expr)。
* **参数：expr** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：要访问的 PrimExpr。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_stmt(*stmt:*[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


Visit Stmt，用户可以自定义该函数，在 C++端覆盖 VisitStmt(const Stmt& stmt)。
* **参数：stmt** ([Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：要访问的 Stmt。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_attr_stmt_(*op:*[AttrStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirattrstmtnode-object-attr_keystr-valueprimexpr-bodystmt-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问 AttrStmt，用户可以自定义该函数，在 C++ 端覆盖 VisitStmt_(const AttrStmtNode* op)。
* **参数：op** ([AttrStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirattrstmtnode-object-attr_keystr-valueprimexpr-bodystmt-spanspannone-none))：要访问的 AttrStmt。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_if_then_else_(*op:*[IfThenElse](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirifthenelseconditionprimexpr-then_casestmt-else_casestmtnone-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)

访问 IfThenElse，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const IfThenElseNode* op)。
* **参数：op** ([IfThenElse](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirifthenelseconditionprimexpr-then_casestmt-else_casestmtnone-spanspannone-none))**：** 要访问的 IfThenElse。
* **返回：result：** 变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_let_stmt_(*op:*[LetStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirletstmtvarvar-valueprimexpr-bodystmt-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问 LetStmt。用户可以自定义此函数，在 C++ 端覆盖 VisitStmt_(const LetStmtNode* op)。
* **参数：op** ([LetStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirletstmtvarvar-valueprimexpr-bodystmt-spanspannone-none))**：** 要访问的 LetStmt。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_for_(*op:*[For](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问 For。用户可以自定义该函数，在 C++ 端覆盖 VisitStmt_(const ForNode* op)。
* **参数：op** ([For](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none))：要访问的 For。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_while_(*op:*[While](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirwhileconditionprimexpr-bodystmt-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问 While。用户可以自定义此函数，在 C++ 端覆盖 VisitStmt_(const WhileNode* op)。
* **参数：op** ([While](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirwhileconditionprimexpr-bodystmt-spanspannone-none))：需要访问的 While 部分。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_allocate_(*op:*[Allocate](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirallocatebuffer_varvar-dtypestr-extentslistprimexpr-conditionprimexpr-bodystmt-annotationsmappingstr-object-none-none-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


Visit Allocate，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const AllocateNode* op)。
* **参数：op** ([Allocate](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirallocatebuffer_varvar-dtypestr-extentslistprimexpr-conditionprimexpr-bodystmt-annotationsmappingstr-object-none-none-spanspannone-none))：要访问的分配。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)

### visit_allocate_const_(*op:*[AllocateConst](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirallocateconstbuffer_varvar-dtypestr-extentslistprimexpr-data_or_idxndarrayint-bodystmt-annotationsmappingstr-object-none-none-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问 AllocateConst，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const AllocateConstNode* op)。
* **参数：op** ([AllocateConst](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirallocateconstbuffer_varvar-dtypestr-extentslistprimexpr-data_or_idxndarrayint-bodystmt-annotationsmappingstr-object-none-none-spanspannone-none))：要访问的 AllocateConst。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_decl_buffer_(*op:*[DeclBuffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirdeclbufferbufferbuffer-bodystmt-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问 DeclBuffer，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const DeclBufferNode* op)。
* **参数：op** ([DeclBuffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirdeclbufferbufferbuffer-bodystmt-spanspannone-none))：要访问的 DeclBuffer。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_buffer_store_(*op:*[BufferStore](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferStore)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问 BufferStore，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const BufferStoreNode* op)。
* **参数：op** ([BufferStore](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferStore))：要访问的 BufferStore。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_buffer_realize_(*op:*[BufferRealize](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferRealize)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问 BufferRealize，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const BufferRealizeNode* op)。
* **参数：op** ([BufferRealize](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferRealize))：要访问的 BufferRealize。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_assert_stmt_(*op:*[AssertStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirattrstmtnode-object-attr_keystr-valueprimexpr-bodystmt-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt) 


访问 AssertStmt，用户可以自定义该函数，在 C++ 端覆盖 VisitStmt_(const AssertStmtNode* op)。
* **参数：op** ([AssertStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirattrstmtnode-object-attr_keystr-valueprimexpr-bodystmt-spanspannone-none))**：** 要访问的 AssertStmt。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_seq_stmt_(*op:*[SeqStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirseqstmtseqliststmt-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问 SeqStmt，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const SeqStmtNode* op)。
* **参数：op** ([SeqStmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirseqstmtseqliststmt-spanspannone-none))：要访问的 SeqStmt。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_evaluate_(*op:*[Evaluate](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirevaluatevalueprimexpr-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt) 

Visit Evaluate，用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const EvaluateNode* op)。
* **参数：op** ([Evaluate](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirevaluatevalueprimexpr-spanspannone-none))：要访问的评估。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_block_(*op:*[Block](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问区块。用户可以自定义该函数，在 C++端覆盖 VisitStmt_(const BlockNode* op)。
* **参数：op** ([Block](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none))：要访问的区块。
* **返回：result**：变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_block_realize_(*op:*[BlockRealize](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-noneRealize)) → [Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)


访问 BlockRealize。用户可以自定义此函数，在 C++ 端覆盖 VisitStmt_(const BlockRealizeNode* op)。
* **参数：op** ([BlockRealize](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-noneRealize))：要访问的 BlockRealize。
* **返回：result：** 变异的 Stmt。
* **返回类型：**[Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

### visit_var_(*op:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


参观 Var。


用户可以自定义该函数，在 C++端覆盖 VisitVar_(const VarNode* op)。
* **参数：op** ([tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))**：** 要访问的 tir.Var。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_size_var_(*op:*[SizeVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirsizevarnamestr-dtypestrtype-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 SizeVar。


用户可以自定义该函数，在 C++端覆盖 VisitSizeVar_(const SizeVarNode* op)。
* **参数：op** ([SizeVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirsizevarnamestr-dtypestrtype-spanspannone-none))：要访问的 SizeVar。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_buffer_load_(*op:*[BufferLoad](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferLoad)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 BufferLoad。


用户可以自定义该函数，在 C++端覆盖 VisitBufferLoad_(const BufferLoadNode* op)。
* **参数：op** ([BufferLoad](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferLoad))：要访问的 BufferLoad。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_producer_load_(*op:*[ProducerLoad](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirproducerloadproducerdataproducer-indiceslistprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 ProducerLoad。


用户可以自定义该函数，在 C++端覆盖 VisitProducerLoad_(const ProducerLoadNode* op)。
* **参数：op** ([ProducerLoad](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirproducerloadproducerdataproducer-indiceslistprimexpr-spanspannone-none))：要访问的 ProducerLoad。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_let_(*op:*[Let](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirletvarvar-valueprimexpr-bodyprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 Let.


用户可以自定义该函数，在 C++端覆盖 VisitLet_(const LetNode* op)。
* **参数：op** ([Let](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirletvarvar-valueprimexpr-bodyprimexpr-spanspannone-none))**：** 要访问的 Let。
* **返回：result** *：* 变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_call_(*op:*[Call](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtircalldtypestr-opopstr-argslistprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问调用。


用户可以自定义该函数，在 C++端覆盖 VisitCall_(const CallNode* op)。
* **参数：op** ([tir.Call](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtircalldtypestr-opopstr-argslistprimexpr-spanspannone-none))：要访问的 tir.Call。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_add_(*op:*[Add](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiraddaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问添加。


用户可以自定义该函数，在 C++端覆盖 VisitAdd_(const AddNode* op)。
* **参数：op** ([Add](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiraddaprimexpr-bprimexpr-spanspannone-none))：要访问的 Add。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_sub_(*op:*[Sub](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirsubaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 Sub。


用户可以自定义该函数，在 C++端覆盖 VisitSub_(const SubNode* op)。
* **参数：op** ([Sub](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirsubaprimexpr-bprimexpr-spanspannone-none))：要访问的 Sub。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_mul_(*op:*[Mul](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmulaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


参观 Mul。


用户可以自定义该函数，在 C++端覆盖 VisitMul_(const MulNode* op)。
* **参数：op** ([Mul](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmulaprimexpr-bprimexpr-spanspannone-none))：要访问的 Mul。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_div_(*op:*[Div](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirdivaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 Div。


用户可以自定义该函数，在 C++端覆盖 VisitDiv_(const DivNode* op)。
* **参数：op** ([Div](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirdivaprimexpr-bprimexpr-spanspannone-none))：要访问的 Div。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_mod_(*op:*[Mod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmodaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 Mod。


用户可以自定义该函数，在 C++端覆盖 VisitMod_(const ModNode* op)。
* **参数：op** ([Mod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirmodaprimexpr-bprimexpr-spanspannone-none))：要访问的 Mod。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_floor_div_(*op:*[FloorDiv](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloordivaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 FloorDiv。


用户可以自定义该函数，在 C++端覆盖 VisitFloorDiv_(const FloorDivNode* op)。
* **参数：op** ([FloorDiv](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloordivaprimexpr-bprimexpr-spanspannone-none))：要访问的 FloorDiv。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_floor_mod_(*op:*[FloorMod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloormodaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 FloorMod。


用户可以自定义该函数，在 C++端覆盖 VisitFloorMod_(const FloorModNode* op)。
* **参数：op** ([FloorMod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloormodaprimexpr-bprimexpr-spanspannone-none))：要访问的 FloorMod。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_min_(*op:*[Min](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirminaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

访问 Min。


用户可以自定义该函数，在 C++端覆盖 VisitMin_(const MinNode* op)。
* **参数：op** ([Min](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirminaprimexpr-bprimexpr-spanspannone-none))：要访问的 Min。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_max_(*op:*[Max](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirminaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


拜访 Max。


用户可以自定义该函数，在 C++端覆盖 VisitMax_(const MaxNode* op)。
* **参数：op** ([Max](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirminaprimexpr-bprimexpr-spanspannone-none))：要访问的最大值。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_eq_(*op:*[EQ](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtireqaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

访问 EQ。


用户可以自定义该函数，在 C++端覆盖 VisitEQ_(const EQNode* op)。
* **参数：op** ([EQ](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtireqaprimexpr-bprimexpr-spanspannone-none))：要访问的 EQ。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_ne_(*op:*[NE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirneaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问NE。


用户可以自定义该函数，在 C++端覆盖 VisitNE_(const NENode* op)。
* **参数：op** ([NE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirneaprimexpr-bprimexpr-spanspannone-none))：要访问的 NE。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_lt_(*op:*[LT](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirltaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

访问 LT。


用户可以自定义该函数，在 C++端覆盖 VisitLT_(const LTNode* op)。
* **参数：op** ([LT](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirltaprimexpr-bprimexpr-spanspannone-none))：要访问的 LT。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

### visit_le_(*op:*[LE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirleaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 LE。


用户可以自定义该函数，在 C++端覆盖 VisitLE_(const LENode* op)。
* **参数：op** ([LE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirleaprimexpr-bprimexpr-spanspannone-none))：要访问的 LE。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_gt_(*op:*[GT](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirgtaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

访问 GT。


用户可以自定义该函数，在 C++端覆盖 VisitGT_(const GTNode* op)。
* **参数：op** ([GT](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirgtaprimexpr-bprimexpr-spanspannone-none))：要访问的 GT。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_ge_(*op:*[GE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirgeaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


参观 GE。


用户可以自定义该函数，在 C++端覆盖 VisitGE_(const GENode* op)。
* **参数：op** ([GE](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirgeaprimexpr-bprimexpr-spanspannone-none))：要访问的 GE。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_and_(*op:*[And](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirandaprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 And。


用户可以自定义该函数，在 C++端覆盖 VisitAnd_(const AndNode* op)。
* **参数：op** ([And](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirandaprimexpr-bprimexpr-spanspannone-none))**：** 要访问的 And。
* **返回：result：** 变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_or_(*op:*[Or](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiroraprimexpr-bprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 Or。


用户可以自定义该函数，在 C++端覆盖 VisitOr_(const OrNode* op)。
* **参数：op** ([Or](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiroraprimexpr-bprimexpr-spanspannone-none))：要访问的 Or。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_reduce_(*op:*[Reduce](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirreducecombinercommreducer-srclistprimexpr-rdomlistitervar-conditionprimexpr-value_indexint-initlistprimexpr-none-none-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 Reduce。


用户可以自定义该函数，在 C++端覆盖 VisitReduce_(const ReduceNode* op)。
* **参数：op** ([Reduce](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirreducecombinercommreducer-srclistprimexpr-rdomlistitervar-conditionprimexpr-value_indexint-initlistprimexpr-none-none-spanspannone-none))**：** 要访问的 Reduce。
* **返回：result：** 变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_cast_(*op:*[Cast](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtircastdtype-value-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 Cast。


用户可以自定义该函数，在 C++端覆盖 VisitCast_(const CastNode* op)。
* **参数：op** ([Cast](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtircastdtype-value-spanspannone-none))：要访问的演员表。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_not_(*op:*[Not](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirnotaprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

不访问。


用户可以自定义该函数，在 C++端覆盖 VisitNot_(const NotNode* op)。
* **参数：op** ([Not](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirnotaprimexpr-spanspannone-none))：不可访问。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_select_(*op:*[Select](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirselectconditionprimexpr-true_valueprimexpr-false_valueprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问选择。


用户可以自定义该函数，在 C++端覆盖 VisitSelect_(const SelectNode* op)。
* **参数：op** ([Select](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirselectconditionprimexpr-true_valueprimexpr-false_valueprimexpr-spanspannone-none))：要访问的选择。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_ramp_(*op:*[Ramp](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirrampbaseprimexpr-strideprimexpr-lanesprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)

参观 Ramp。


用户可以自定义该函数，在 C++端覆盖 VisitRamp_(const RampNode* op)。
* **参数：op** ([Ramp](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirrampbaseprimexpr-strideprimexpr-lanesprimexpr-spanspannone-none))：要访问的坡道。
* **返回：result：** 变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_broadcast_(*op:*[Broadcast](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbroadcastvalueprimexpr-lanesprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问广播。


用户可以自定义该函数，在 C++端覆盖 VisitBroadcast_(const BroadcastNode* op)。
* **参数：op** ([Broadcast](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbroadcastvalueprimexpr-lanesprimexpr-spanspannone-none))：要访问的广播。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_shuffle_(*op:*[Shuffle](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirshufflevectorslistprimexpr-indiceslistprimexpr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 Shuffle。


用户可以自定义该函数，在 C++端覆盖 VisitShuffle_(const ShuffleNode* op)。
* **参数：op** ([Shuffle](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirshufflevectorslistprimexpr-indiceslistprimexpr-spanspannone-none))：要访问的 Shuffle。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_int_imm_(*op:*[IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 IntImm。


用户可以自定义该函数，在 C++端覆盖 VisitIntImm_(const IntImmNode* op)。
* **参数：op** ([IntImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none))：要访问的 IntImm。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_float_imm_(*op:*[FloatImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloatimmdtypestr-valuefloat-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 FloatImm。


用户可以自定义该函数，在 C++端覆盖 VisitFloatImm_(const FloatImmNode* op)。
* **参数：op** ([FloatImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirfloatimmdtypestr-valuefloat-spanspannone-none))：要访问的 FloatImm。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

### visit_string_imm_(*op:*[StringImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstringimmvaluestr-spanspannone-none)) → [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


访问 StringImm。


用户可以自定义该函数，在 C++端覆盖 VisitStringImm_(const StringImmNode* op)。
* **参数：op** ([StringImm](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstringimmvaluestr-spanspannone-none))：要访问的 StringImm。
* **返回：result**：变异的 PrimExpr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


