---

title: tvm.relax.op

---


Relax 核心运算符。

## tvm.relax.op.assert_op(*condition:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *format_args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *format:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*= ''*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


创建一个对 Relax 的 assert_op 操作的调用（在 Python 中，assert 是保留字，因此名称必须不同）。
* **参数：**
   * **condition** (*Union*[***Expr,*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：断言条件。
   * **format_args** (*Optional*[****Union**[**Expr,*** ***List****[**Expr**]]*]*)：如果条件失败，则为错误消息提供格式化参数。
   * **format** (*Union*[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***Expr***])：错误消息的格式字符串或 StringImm。
* **返回：result**：一个调用 Relax 断言操作的 relax.Call。
* **返回类型：** Expr。

## tvm.relax.op.call_builtin_with_ctx(*func:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *, *sinfo_args: [StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)] | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)

relax.Call 内置函数。
* **参数：**
   * **func** (*Expr*)：要调用的内置函数。
   * **args** (*Expr*)**：** 输入参数。
   * **sinfo_args** (*Optional*[****Union**[***[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)***,** **List**[*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*]**]** *]*)：调用节点中的结构信息参数。
* **返回：ret**：创建的调用节点。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.call_dps_packed(*func:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *out_sinfo:*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*]*) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)


relax.Call 一个目标传递风格的打包函数并返回输出。


注意：被调用的函数被认为是纯函数（除了修改指定的输出参数）。如果函数确实产生了其他副作用，那么编译器可能会最终移除、重新排序或重复这些效果——不提供任何保证。
* **参数：**
   * **func** (*Union[***[str](https://docs.python.org/3/library/stdtypes.html#str)**,****Expr****])：目标传递风格的函数，可以是 ExternFunc。
   * **args** (*Expr*)：输入参数。
   * **out_sinfo** (*Union*[***[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)***,***List***[*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*]]*)：call_dps_packed 输出的结构信息。它应该是一个 TensorStructInfo 或 TensorStructInfo 列表。每个元素表示一个返回张量的结构信息。
* **返回：ret：** 一个用于 call_dps_packed 操作的调用节点。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)

## tvm.relax.op.call_inplace_packed(*func:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[ExternFunc](/docs/api-reference/python-api/tvm-relax#classtvmrelaxexternfuncglobal_symbolstringstruct_infostructinfononenonespanspannonenone)*|*[GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none), args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *inplace_indices:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *sinfo_args:*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*]*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


构造一个调用打包函数的调用，该函数会消耗其部分参数「就地」并返回被修改的参数（别名），但应被视为其他方面是纯函数。inplace_indices 参数指示哪些输出是被修改的参数。


生成的调用将具有与直接调用打包函数相同的语义。


注意：这应该用于用户知道使用这些参数调用打包函数实际上不会引起任何其他副作用的情况。如果它用于一个确实会导致其他副作用的调用，那么编译器可能会最终删除、重新排序或重复该调用，并且不保证调用方产生的任何副作用。


警告：此运算符在类型系统中被视为纯操作，即使它执行了副作用（修改某些参数）。因此，用户必须确保它被安全使用（即，修改后的参数在修改后不应处于活动状态，它们不应在修改后别名化值）。
* **参数：**
   * **func** ( *Union [*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[ExternFunc](/docs/api-reference/python-api/tvm-relax#classtvmrelaxexternfuncglobal_symbolstringstruct_infostructinfononenonespanspannonenone)*]* )：PackedFunc 或 ExternFunc 节点的名称（全局符号）。
   * **args** ( *Expr* )：PackedFunc 的参数。
   * **inplace_indices** ( *Union [*[int](https://docs.python.org/3/library/functions.html#int)*, List [*[int](https://docs.python.org/3/library/functions.html#int)*] ]* )：指定哪些参数应用于就地计算。如果inplace_indices是单个整数，它将被转换为单例列表。假设inplace_indices[i] = j，其中j >= 0。则第 i 个输出将是 `args[j] 的别名。如果inplace_indices[i] = -1，则第 i 个输出将是新分配的张量。`inplace_indices 中至少有一个成员不能为 -1。
   * **sinfo_args** ( *Union [*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*, List [*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*] ]* )：结构信息参数列表（提供返回值的结构信息）。
* **返回：result**：Relax 调用，对应于 call_pure_packed(ExternFunc(func), args, DictAttrs(kwargs), sinfo_args)。
* **返回类型：** Expr。

## tvm.relax.op.call_pure_packed(*func:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[ExternFunc](/docs/api-reference/python-api/tvm-relax#classtvmrelaxexternfuncglobal_symbolstringstruct_infostructinfononenonespanspannonenone)*|*[GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none), args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *sinfo_args:*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*]*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


构建一个调用打包函数的调用，即使打包调用通常不被视为纯函数。


生成的调用将具有与直接调用打包函数相同的语义。


注意：这应该用于用户知道使用这些参数调用打包函数实际上不会产生任何副作用的情况。如果用于一个确实会导致副作用的调用，那么编译器可能会最终移除、重新排序或重复该调用，并且对被调用方的任何副作用不做任何保证。
* **参数：**
   * **func** (*Union[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[ExternFunc](/docs/api-reference/python-api/tvm-relax#classtvmrelaxexternfuncglobal_symbolstringstruct_infostructinfononenonespanspannonenone)*]*)：PackedFunc 或 ExternFunc 节点的名称（全局符号）。
   * **args** (*Expr*)：PackedFunc 的参数。
   * **sinfo_args** (*Union*[***[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)***,******List****[*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*]])：返回值结构信息的参数列表。
* **返回：result**：一个 Relax 调用，对应于 call_pure_packed(ExternFunc(func), args, DictAttrs(kwargs), sinfo_args)。
* **返回类型：** Expr。

## tvm.relax.op.call_tir(*gvar:*[GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none), *args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *out_sinfo:*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*]*, *tir_vars:*[ShapeExpr](/docs/api-reference/python-api/tvm-relax#classtvmrelaxshapeexprvalueslistprimexprtupleprimexprarrayspanspannonenone)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)


relax.Call 一个 tir.prim_func 并返回输出。
* **参数：**
   * **gvar** ([GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none))：指向 tir PrimFunc 的 GlobalVar。
   * **args** (*Expr*)：输入参数。
   * **out_sinfo** (*Union*[***[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)***,***List***[*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*]])：调用_tir 的输出结构信息。它应该是一个 TensorStructInfo 或 TensorStructInfo 的列表。每个元素表示一个返回张量的结构信息。
   * **tir_vars** (*Optional*[****Union**[***[ShapeExpr](/docs/api-reference/python-api/tvm-relax#classtvmrelaxshapeexprvalueslistprimexprtupleprimexprarrayspanspannonenone)***,** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)***]***,List[***[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)***]*]]*)：表示调用 func 时解包的整数元组的 ShapeExpr。如果未使用则为 null。
* **返回：ret**：call_tir 运算符的调用节点。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.call_tir_inplace(*gvar:*[GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none), *args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *inplace_indices:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *out_sinfo:*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*]*, *tir_vars:*[ShapeExpr](/docs/api-reference/python-api/tvm-relax#classtvmrelaxshapeexprvalueslistprimexprtupleprimexprarrayspanspannonenone)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)


relax 调用 TIR PrimFunc 并返回结果，在指定位置执行计算（基于 inplace_indices 参数；输出将与通过 in-place 索引选择的输入别名）。


警告：此运算符在类型系统中被视为纯操作，但实际上会修改由 inplace_indices 指定的参数。此运算符不应直接使用，而应由检查是否安全进行原地操作（即，作为输出的参数没有被别名化或调用 call_tir_inplace 后仍然活跃）的 passes 插入。


仅应出于测试目的对此运算符进行直接调用。
* **参数：**
   * **gvar** ( [GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none) )：GlobalVar 引用 TIR PrimFunc。
   * **args** ( *Expr* )**：** 输入参数。
   * **inplace_indices** ( *Union [*[int](https://docs.python.org/3/library/functions.html#int)*, List [*[int](https://docs.python.org/3/library/functions.html#int)*] ]* )：指定哪些参数应用于就地计算。如果inplace_indices是单个整数，它将被转换为单例列表。假设inplace_indices[i] = j，其中j >= 0。则第 i 个输出将是 `args[j] 的别名。如果inplace_indices[i] = -1，则第 i 个输出将是新分配的张量。`inplace_indices 中至少有一个成员不能为 -1。
   * **out_sinfo** ( *Union [*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*, List [*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*] ]* )：call_tir_inplace 输出的结构信息。它应该是一个TensorStructInfo或一个TensorStructInfo列表。每个列表表示返回张量的结构信息。如果给出一个TensorStructInfo列表，则结果将是一个TensorStructInfo元组。
   * **tir_vars** ( *Optional [ Union [*[ShapeExpr](/docs/api-reference/python-api/tvm-relax#classtvmrelaxshapeexprvalueslistprimexprtupleprimexprarrayspanspannonenone)*,*[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] , List [*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] ] ]* )：ShapeExpr 表示调用 func 时需要解包的整数元组。若未使用则为 null。
* **返回：ret**：call_tir 运算符的调用节点。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.call_tir_with_grad(*gvar:*[GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none), *args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *out_sinfo:*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*]*, *te_grad_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *te_grad_kwargs:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object] = None*, *tir_vars:*[ShapeExpr](/docs/api-reference/python-api/tvm-relax#classtvmrelaxshapeexprvalueslistprimexprtupleprimexprarrayspanspannonenone)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)


relax 调用 tir.prim_func 并返回输出。这个内建函数会将 te 梯度函数（通过 te_grad_name 指向）绑定到 call_tir_with_grad 节点。梯度传递过程会调用这个 te 梯度函数。
* **参数：**
   * **gvar** ([GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none))**：** 指向 tir PrimFunc 的 GlobalVar。
   * **args** (*Expr*)：输入参数。
   * **out_sinfo** (*Union*[***[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)***,******List****[*[TensorStructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone)*]])：call_tir_with_grad 输出的结构信息。它应该是一个 TensorStructInfo 或 TensorStructInfo 列表。每个元素表示一个返回张量的结构信息。
   * **te_grad_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：与 call_tir_with_grad 节点相关联的 te 梯度函数的注册名称。必须作为关键字参数提供。
   * **te_grad_kwargs** (*Dict*[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,******Object****],optional*)*： 传递给 te 梯度函数的关键字参数。可选作为关键字参数提供。默认：{}。
   * **tir_vars** (*Optional**[****Union**[***[ShapeExpr](/docs/api-reference/python-api/tvm-relax#classtvmrelaxshapeexprvalueslistprimexprtupleprimexprarrayspanspannonenone)***,** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)***]****,List[***[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)***]*]])：表示调用 func 时要解包的整数元组的 ShapeExpr。如果未使用则为 null
* **返回：ret**：调用 call_tir_with_grad 运算符的调用节点。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)

## tvm.relax.op.hint_on_device(*data*, *dst_vdevice*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

它提供了一个提示，指定输入数据应该在哪个设备上执行。这个提示被 RealizeVDevice 用来传播虚拟设备。
* **参数：**
   * **data** (*Expr*)：要复制的张量。
   * **dst_device** ([VDevice](/docs/api-reference/python-api/tvm-ir#class-tvmirvdevicetargetnone-vdevice_idint-0-memory_scopestr-global))：数据预期执行的目標设备。
* **返回：result**：结果。
* **返回类型：** Expr。

## tvm.relax.op.invoke_closure(*closure:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *sinfo_args:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*] |*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)

调用闭包。
* **参数：**
   * **closure** (*Expr*)：VMClosure 对象。
   * **args** (*Expr*)：输入参数。
   * **type_args** (*Union*[****List**[***[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)***]****,*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*]*)：CallNode 的结构信息参数。
* **返回：ret**-invoke_closure 的调用。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.invoke_pure_closure(*closure:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *sinfo_args:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*] |*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)


调用闭包并指示编译器该闭包是纯函数。


注意：这应该用于用户知道使用这些参数调用闭包实际上不会产生任何副作用的情况。如果用于一个确实会产生副作用的调用，那么编译器可能会最终移除、重新排序或重复该调用，并且对调用方的任何副作用不做任何保证。
* **参数：**
   * **closure** (*Expr*)：VMClosure 对象。
   * **args** (*Expr*)：输入参数。
   * **type_args** (*Union*[****List**[***[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)***]****,*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*])：CallNode 的结构信息参数。
* **返回：ret**：调用 invoke_pure_closure。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.make_closure(*func:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → Object

创建一个带有自由变量的闭包并返回该闭包。
* **参数：**
   * **func** (*Expr*)：该闭包，可以是 ExternFunc 或 PrimFunc。
   * **args** (*Expr*)-输入参数。
* **返回：ret**：VMClosure。
* **返回类型：** Object。

## tvm.relax.op.null_value() → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)


创建一个表示空值对象的调用节点。
* **返回：ret：** 创建的调用节点。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.print(values:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]*, *format:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*= '') → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

打印操作以打印值。
* **参数：**
   * **values** (*List*[Expr])：要打印的值。
   * **format** (*Union*[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***Expr***]*)*：格式字符串或 StringImm。
* **返回：result**：一个 relax Call，在运行时将打印值。
* **返回类型：** Expr。

## tvm.relax.op.register_gradient(*op_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *fgradient:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*[[*[Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenon)*,*[Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)*,*[Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenon)*,*[BlockBuilder](/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_builderblockbuildermodirmodulenone-none)*],*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]] = None*, *level:*[int](https://docs.python.org/3/library/functions.html#int)*= 10*)

为 relax 运算符注册运算符梯度函数。
* **参数：**
   * **op_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：运算符的名称。
   * **fgradient** (*function(**orig_var: relax.Var****,orig_call: relax.Call,***output_grad: relax.Var****,ctx: BlockBuilder))：> partials: List[Expr] 使用的梯度函数。
   * **level** ([int](https://docs.python.org/3/library/functions.html#int))：优先级级别

## tvm.relax.op.shape_of(*expr:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


获取张量的形状。
* **参数：**
   * **expr** (*Expr*)：输入的 Expr。
* **返回：result**：一个 relax.Call，它获取输入的形状。
* **返回类型：** Expr。

## tvm.relax.op.shape_to_tensor(*expr:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

将形状转换为张量 expr。 :param expr: 输入的 Expr :type expr: Expr。
* **返回：result**：一个 relax relax.Call，将形状值转换为张量。
* **返回类型：** Expr。

## tvm.relax.op.tensor_to_shape(*expr:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

将张量转换为形状表达式。 :param expr: 输入的 Expr :type expr: Expr。
* **返回：result**：一个 relax relax.Call，将张量值转换为形状。
* **返回类型：** Expr。

## tvm.relax.op.to_vdevice(*data*, *dst_vdevice*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


将数据复制到目标设备。此运算符有助于在不同设备之间进行数据传输，以支持异构执行。
* **参数：**
   * **data** (*Expr*)：要复制的张量。
   * **dst_device** ([VDevice](/docs/api-reference/python-api/tvm-ir#class-tvmirvdevicetargetnone-vdevice_idint-0-memory_scopestr-global))：数据被复制到的目标设备。
* **返回：result**：复制的结果。
* **返回类型：** Expr。

## tvm.relax.op.add(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


使用 numpy 风格的广播进行加法。
* **参数：**
   * **x1** (*Expr*)：第一个输入张量。
   * **x2** (*Expr*)：第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** Expr。


**示例**

```python
bb = relax.BlockBuilder()
a = relax.Var("a", relax.TensorStructInfo(shape=(2, 3), dtype="float32"))
b = relax.Var("b", relax.TensorStructInfo(shape=(2, 1), dtype="float32"))
c = bb.normalize(relax.op.add(a, b))  # c has TensorStructInfo(shape=(2, 3), dtype="float32")
```
## tvm.relax.op.bitwise_and(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


按位 AND :param x1: 第一个输入张量。 :type x1: relax.Expr :param x2: 第二个输入张量。 :type x2: relax.Expr。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.bitwise_or(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


按位 OR :param x1: 第一个输入张量。 :type x1: relax.Expr :param x2: 第二个输入张量。 :type x2: relax.Expr。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.bitwise_xor(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


按位异或 :param x1: 第一个输入张量。 :type x1: relax.Expr :param x2: 第二个输入张量。 :type x2: relax.Expr
* **返回：result：** 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.divide(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


使用 numpy 风格的广播进行除法。
* **参数：**
   * **x1** (*relax.Expr*)：第一个输入张量。
   * **x2** (*relax.Expr*)：第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.equal(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

逐元素广播测试（lhs == rhs）。
* **参数：**
   * **x1** (*relax.Expr*)：第一个输入张量。
   * **x2** (*relax.Expr*)：第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.floor_divide(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


使用 numpy 风格的广播进行地板除。
* **参数：**
   * **x1** (*relax.Expr*)：第一个输入张量。
   * **x2** (*relax.Expr*)：第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.log_add_exp(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入的指数和的对数，逐元素进行。
* **参数：**
   * **x1** (*Expr*)：第一个输入张量。
   * **x2** (*Expr*)：第二个输入张量。
* **返回：** The element-wise log-sum-exp of x1 and x2。
* **返回类型：** Expr。

## tvm.relax.op.floor_mod(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

使用 numpy 风格的广播进行地板取模。
* **参数：**
   * **x1** (*Expr*)：第一个输入张量。
   * **x2** (*Expr*)：第二个输入张量。

## tvm.relax.op.greater(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


广播的逐元素测试 (lhs > rhs)。
* **参数：**
   * **x1** (*relax.Expr*)：第一个输入张量。
   * **x2** (*relax.Expr*)：第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.greater_equal(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

广播的逐元素测试 (lhs >= rhs)。
* **参数：**
   * **x1** (*relax.Expr*)**：** 第一个输入张量。
   * **x2** (*relax.Expr*)：第二个输入张量。
* **返回：result**-计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.left_shift(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


位左移 :param x1: 要移位的输入张量。 :type x1: relax.Expr :param x2: 移位的位数。 :type x2: relax.Expr。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.less(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

广播的逐元素测试 (lhs < rhs)。
* **参数：**
   * **x1** (*relax.Expr*)：第一个输入张量。
   * **x2** (*relax.Expr*)：第二个输入张量。
* **返回：result**-计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.less_equal(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


广播的逐元素测试 (lhs <= rhs)。
* **参数：**
   * **x1** (*relax.Expr*)**：** 第一个输入张量。
   * **x2** (*relax.Expr*)：第二个输入张量。
* **返回：result***：*计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.logical_and(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


逻辑与 :param x1: 第一个输入张量。 :type x1: relax.Expr :param x2: 第二个输入张量。 :type x2: relax.Expr
* **返回：result**-计算结果。
* **返回类型：** relax.Expr，

## tvm.relax.op.logical_or(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


逻辑或 :param x1: 第一个输入张量。 :type x1: relax.Expr :param x2: 第二个输入张量。 :type x2: relax.Expr。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.logical_xor(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

逻辑异或 :param x1: 第一个输入张量。 :type x1: relax.Expr :param x2: 第二个输入张量。 :type x2: relax.Expr。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.maximum(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


逐元素最大值。
* **参数：**
   * **x1** (*relax.Expr*)：第一个输入张量。
   * **x2** (*relax.Expr*)-第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.minimum(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)
逐元素最小值。
* **参数：**
   * **x1** (*relax.Expr*)：第一个输入张量。
   * **x2** (*relax.Expr*)：第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.mod(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


使用 numpy 风格的广播进行取模运算。
* **参数：**
   * **x1** (*Expr*)：第一个输入张量。
   * **x2** (*Expr*)：第二个输入张量。

## tvm.relax.op.multiply(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


使用 numpy 风格的广播进行乘法。
* **参数：**
   * **x1** (*Expr*)-第一个输入张量。
   * **x2** (*Expr*)：第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** Expr。

## tvm.relax.op.not_equal(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

对广播后的元素进行不等性测试 (lhs != rhs)。
* **参数：**
   * **x1** (*relax.Expr*)-第一个输入张量。
   * **x2** (*relax.Expr*)：第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.power(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr))

使用 numpy 风格的广播进行幂运算。
* **参数：**
   * **x1** (*relax.Expr*)-第一个输入张量。
   * **x2** (*relax.Expr*)：第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.right_shift(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

位右移 :param x1: 要移位的输入张量。 :type x1: relax.Expr :param x2: 移位的位数。 :type x2: relax.Expr。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.subtract(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


numpy 风格的广播减法。
* **参数：**
   * **x1** (*relax.Expr*)：第一个输入张量。
   * **x2** (*relax.Expr*)*：*第二个输入张量。
* **返回：result**：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.arange(*start:*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone), *end:*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *step:*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*= 1*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


构造一个元素均匀分布的张量。
* **参数：**
   * **start** (*Union*[****PrimExprLike,*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*]*)*：*区间的起始值。
   * **end** (*Optional*[****Union**[****PrimExprLike**,***[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*]])：区间的结束。如果未提供，它将被设置为 start，而 start 将被设置为 0。
   * **step** (*Union*[***PrimExprLike,*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*]*)：步长。
   * **dtype** (*Optional*[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***DataType****]]*)：创建的张量的数据类型。
* **返回：result**：结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.full(*shape:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *fill_value:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

用标量值填充数组。
* **参数：**
   * **shape** (*Union*[***[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[****PrimExprLike**]***,Expr]*)：创建的张量的形状。
   * **fill_value** (*relax.Expr*)：填充的值。必须是一个标量张量。
   * **dtype** (*Optional*[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,** ***DataType***]])：创建的张量的数据类型。如果未给出 dtype，默认将使用 fill_value 的数据类型。
* **返回：result**：结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.full_like(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *fill_value:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


构造一个张量，其特征为：- 形状与输入数据张量的形状相同，- 其值被输入标量 fill_value 填充。
* **参数：**
   * **x** (*relax.Expr*)：输入张量，提供形状，当 dtype 字段未指定时提供 dtype。
   * **fill_value** (*relax.Expr*)：填充的值。必须是一个标量张量。
   * **dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,** ***DataType***]]*)*：创建的张量的数据类型。如果未给出 dtype，默认将使用输入张量的 dtype。
* **返回：result**：结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.hamming_window(window_size, periodic, alpha, beta, dtype)

Hamming 窗函数。
* **参数：**
   * **window_size** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：返回窗口的大小。
   * **periodic** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 如果为 True，返回一个周期函数使用的窗口。如果为 False，返回一个对称窗口。
   * **alpha** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：系数 alpha。
   * **beta** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：系数 beta。
* **返回：ret**：结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.ones(*shape:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

构建一个所有元素都为 1 的张量，具有输入的形状和数据类型。
* **参数：**
   * **shape** (*Union*[***[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[****PrimExprLike**]***,Expr]*)：创建的张量的形状。
   * **dtype** (*Union*[***[str](https://docs.python.org/3/library/stdtypes.html#str)***, ***DataType***])：创建的张量的数据类型。
* **返回：result**-结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.ones_like(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


构造一个全为 1 的张量，其形状与输入张量的形状相同。
* **参数：**
   * **x** (*relax.Expr*)-输入张量，当 dtype 字段未指定时，提供形状和 dtype。
   * **dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,** ***DataType***]])：创建的张量的数据类型。如果 dtype 未给出，将默认使用输入张量的 dtype。
* **返回：result**：结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.eye(*n:*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone), *m:*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *k:*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*= 0*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype = 'float32'*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


构造一个对角线上为 1，其余位置为 0 的 2-D 张量。
* **参数：**
   * **n** (*Union*[***PrimExprLike,*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*]*)**：** 输出张量的行数。
   * **m** (*Optional*[****Union**[****PrimExprLike**,*** [PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)***]***]*)：输出中的列数。如果为 None，则默认为 n。
   * **k** (*Union*[***PrimExprLike,*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*]*)：对角线的索引：0（默认值）表示主对角线，正值表示上对角线，负值表示下对角线。
   * **dtype** (*Union* *[***[str](https://docs.python.org/3/library/stdtypes.html#str)**, ***DataType***])：创建的张量的数据类型。
* **返回：result**：结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.eye_like(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *k:*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*= 0*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


返回一个二维张量，其对角线元素为 1，其余元素为 0，形状与输入张量相同。
* **参数：**
   * **x** (*relax.Expr*)：输入张量，当 dtype 字段未指定时，提供形状和 dtype。
   * **k** (*Union*[***PrimExprLike,*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*]*)：对角线的索引：0（默认值）表示主对角线，正数表示上对角线，负数表示下对角线。
   * **dtype** (*Optional*[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,** ***DataType***]])：创建的张量的数据类型。如果未给出 dtype，默认将使用输入张量的数据类型。
* **返回：result** ：结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.tril(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *k:*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*= 0*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


返回矩阵或矩阵批次的下三角部分。
* **参数：**
   * **x** (*relax.Expr*) ：tril 操作将应用到的张量。它至少需要有两个维度。
   * **k** ([int](https://docs.python.org/3/library/functions.html#int)) ：指示要置零元素的上方对角线的索引。如果 k = 0，对角线是主对角线。如果 k < 0，对角线在主对角线下方。如果 k > 0，对角线在主对角线上方。
* **返回：ret** ：结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.triu(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *k:*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*= 0*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

返回矩阵或矩阵批次的上三角部分。
* **参数：**
   * **x** (*relax.Expr*) ：将对 triu 应用的张量。它至少需要有两个维度。
   * **k** ([int](https://docs.python.org/3/library/functions.html#int)) ：指示要置零元素的 diagonals 以下索引。如果 k = 0，对角线是主对角线。如果 k < 0，对角线在主对角线下方。如果 k > 0，对角线在主对角线上方。
* **返回：ret** ：结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.zeros(*shape:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


构建一个全为零的张量，其输入形状和 dtype。
* **参数：**
   * **shape** (*Union*[***[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[****PrimExprLike**]***,Expr]*) ***：*** 创建的张量的形状。
   * **dtype** (*Union*[***[str](https://docs.python.org/3/library/stdtypes.html#str)** *,** ***DataType***]) **：** 创建的张量的数据类型。
* **返回：result** ：结果张量。
* **返回类型：** relax.Expr

## tvm.relax.op.zeros_like(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


构造一个全为零的张量，其形状与输入张量的形状相同。
* **参数：**
   * **x** (*relax.Expr*) ：输入张量，当 dtype 字段未指定时，提供形状和 dtype。
   * **dtype** (*Optional*[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)*,***DataType***]]) **：** 创建的张量的数据类型。如果未给出 dtype，默认将使用输入张量的数据类型。
* **返回：result** ：结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.astype(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


将输入张量转换为指定的数据类型。
* **参数：**
   * **x** (*relax.Expr*) ：运算符的输入数据。
   * **dtype** (*Union*[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***DataType***]) ：目标数据类型。
* **返回：result** ：转换结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.wrap_param(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype = 'float32'*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


如果输入数据的类型与给定的类型不同，则将模型参数输入张量转换为数据类型。 :param data: 运算符的输入数据。 :type data: relax.Expr :param dtype: 目标数据类型 :type dtype: Union[str, DataType]。
* **返回：result** ：转换后的结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.dynamic_strided_slice(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *begin:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *end:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *strides:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

动态步长切片张量。begin、end、strides 可以在运行时计算。
* **参数：**
   * **x** (*Expr*) ：要切片的源张量。
   * **begin** (*Expr*) ：切片开始的索引，包含在内。
   * **end** (*Expr*) ：指示切片结束的索引，不包含在内。
   * **strides** (*Expr*) **：**指定步长值，可以是负数，此时输入张量将在该特定轴上反转。如果未指定，默认为与轴数相同长度的全为 1 的列表。
* **返回：ret** ：切片结果。
* **返回类型：** relax.Expr。

:::note

dyn_strided_slice 要求输入的 begin、end 和 strides 的长度与数据张量的 rank 相同。

:::

## tvm.relax.op.strided_slice(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axes:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *begin:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *end:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *strides:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *assume_inbound:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


对张量进行带步长的切片。
* **参数：**
   * **x** (*relax.Expr*) *：* 要进行切片的源张量。
   * **axes** (*List[*[int](https://docs.python.org/3/library/functions.html#int)*]*) *：* 应用切片的轴。
   * **begin** (*List*[*PrimExprLike]*) ：切片开始的索引，包含在内。
   * **end** (*List*[*PrimExprLike]*) ：指示切片结束的索引，不包含在内。
   * **strides** (*Optional**[****List**[****PrimExprLike**]****]) ：指定步长值，可以是负数，此时输入张量将在该特定轴上反转。如果未指定，默认为与轴数相同长度的全为 1 的列表。
   * **assume_inbound** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：是否假设索引在范围内。如果设置为 false，超出范围的索引将被裁剪到边界。
* **返回：ret** ：切片结果。
* **返回类型：** relax.Expr。


strided_slice 要求输入的 begin、end 和 strides 的长度与 axes 相同。

:::

## tvm.relax.op.take(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *indices:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

从一个张量沿轴取出元素。它的语义与 numpy.take（[https://numpy.org/doc/stable/reference/generated/numpy.take.html）基本相同，可以涵盖](https://numpy.org/doc/stable/reference/generated/numpy.take.html）基本相同，可以涵盖) torch.take（[https://pytorch.org/docs/stable/generated/torch.take.html）和](https://pytorch.org/docs/stable/generated/torch.take.html）和) onnx.gather（[https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-13）。](https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-13）。)
* **参数：**
   * **x** (*relax.Expr*) ：源张量。
   * **indices** (*relax.Expr*) **：** 要提取的值的索引。
   * **axis** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*) ：选择值的轴。如果为 none，输入张量必须是一维的。
* **返回：ret** ：提取的结果。
* **返回类型：** relax.Expr

## tvm.relax.op.einsum(*operands*, *subscripts*)

对数据进行爱因斯坦求和约定求值。
* **参数：**
   * **operands** (*Union*(****List**[****relax.Expr**]****,*[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*[**relax.Expr**])) ：表达式列表。
   * **subscripts** ([str](https://docs.python.org/3/library/stdtypes.html#str)) -爱因斯坦求和表达式字符串。
* **返回：result** ：einsum op 的输出。
* **返回类型：** relax.Expr。

## tvm.relax.op.linear(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *weight:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *bias:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


对输入数据应用线性变换：y = xA^T + b。
* **参数：**
   * **data** (*relax.Expr*) *：* 输入数据。
   * **weight** (*relax.Expr*) ：权重张量。
   * **bias** (*Optional*[*Expr]*) ：偏置张量。
   * **out_dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)*,****DataType****]]) ：乘法结果的数值类型。如果未指定，输出类型将与输入类型相同。


注意


Relax 不将线性运算符视为原始运算符，而是通过组合转置、矩阵乘法和加法运算符来实现它。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.matmul(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


对两个张量进行一般矩阵乘法，并在批处理维度上进行广播。


语义和输出形状推导规则指定为 [https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html。](https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html。)
* **参数：**
   * **x1** (*relax.Expr*) ：第一个输入张量。
   * **x2** (*relax.Expr*) **：** 第二个输入张量。
   * **out_dtype** (*Optional* *[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,** ***DataType****]]) *：* matmul 结果的 dtype。如果未指定，输出 dtype 将与输入 dtype 相同。
* **返回：result** **：** 计算结果。
* **返回类型：** relax.Expr

## tvm.relax.op.outer(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

计算两个输入表达式的外积。
* **参数：**
   * **x1** (*relax.Expr*) ：第一个输入表达式。
   * **x2** (*relax.Expr*) ：第二个输入表达式。


**注意**


该操作计算两个表达式之间的外积，结果是一个张量，其中每个元素是来自 x1 和 x2 的元素的乘积。它在张量和矩阵运算中常用，用于将低维输入扩展为高维表示。
* **返回：result** ：表示外积的结果表达式。
* **返回类型：** relax.Expr。

## tvm.relax.op.broadcast_to(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *shape:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

将张量广播到指定形状。
* **参数：**
   * **x** (*relax.Expr*) ：运算符的输入数据。
   * **shape** (*Union*[***[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[****PrimExprLike**]***,Expr]*) -目标形状。
* **返回：result** ：广播后的张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.collapse_sum_like(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *collapse_target:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


返回数据到 collapse_target 的形状的求和。


详情请参阅 relax.op.collapse_sum_to。
* **参数：**
   * **data** (*relax.Expr*) ：输入张量。
   * **collapse_target** (*relax.Expr*) -要折叠到的张量的形状。
* **返回：result** -求和后的结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.collapse_sum_to(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *shape:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

返回数据折叠到给定形状的求和结果。


collapse_sum_to 是 tvm.relax.op.broadcast_to 和其他广播运算符在自动微分过程中的反向运算符。


我们预期 data 是某个广播操作中广播某个给定形状张量的结果。因此，给定形状和 data.shape 必须遵循广播规则。


在计算过程中，会从右到左检查 data.shape 和 shape 的所有轴。对于一个轴，如果它满足以下规则之一，data 将在该轴上求和：- 该轴存在于 data.shape 中但不存在于 shape 中，或- 该轴存在于 data.shape 中，并且在 shape 中等于 1。
* **参数：**
   * **data** (*relax.Expr*) ：输入张量。
   * **shape** (*Union*[***[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[****PrimExprLike**]****,relax.Expr]*) *：* 要折叠成的形状。
* **返回：result** ：按给定形状求和后的结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.concat(*tensors:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 0*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

沿给定轴连接输入张量。
* **参数：**
   * **tensors** (*Union*[****relax.Expr**,** ***List***[**relax.Expr**]]) ：一个元组类型的 Expr，包含要连接的张量，或一个张量列表。
   * **axis** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*) - 沿此轴将张量连接起来。如果 axis 为 None，则在连接之前需要将输入张量展平。
* **返回：result** *：* 连接的张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.expand_dims(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


在 axis 指定的位置插入新的轴。
* **参数：**
   * **x** (*relax.Expr*) -运算符的输入数据。
   * **axis** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,***List***[*[int](https://docs.python.org/3/library/functions.html#int)*]]) ：1, data.ndim] 内，遵循负索引的约定。
* **返回：result** **：** 转换后的结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.flatten(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


将所有张量维度展平为一个。
* **参数：x** (*relax.Expr*) - 运算符的输入数据。
* **返回：result** ：展平的结果。
* **返回类型：** relax.Expr

## tvm.relax.op.flip(*data*, *axis*)


翻转指定轴上的元素顺序，同时保持数组形状。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：翻转的轴。
* **返回：ret** ：计算结果。
* **返回类型：** relax.Expr。


**示例**

```python
x = [[1., 2.], [3., 4.]]
relax.flip(x, axis=0) = [[3., 4.], [1., 2.]]

relax.flip(x, axis=1) = [[2., 1.], [4., 3.]]
```
## tvm.relax.op.gather_elements(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *indices:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


根据指定轴上的索引从数据中收集元素。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **indices** (*relax.Expr*) ：索引张量，必须为整数类型。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：指定的索引轴。默认为 0。
* **返回：ret** ：计算结果。
* **返回类型：** relax.Expr。


**示例**

```python
data = [[1, 2], [3, 4]]
indices = [[0, 0], [1, 0]]
axis = 1
output = [[1, 1], [4, 3]]

data = [[1, 2, 3], [4, 5, 6]]
indices = [[1, 1, 1]]
axis = 0
output = [[4, 5, 6]]
```
## tvm.relax.op.gather_nd(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *indices:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *batch_dims:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


使用 updates 中的值更新由 indices 定义的位置处的数据。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **indices** (*relax.Expr*) ：索引张量，必须为整数类型。
   * **batch_dims** ([int](https://docs.python.org/3/library/functions.html#int)) ：批量维度的数量。默认为 0。
* **返回：ret** ：计算结果。
* **返回类型：** relax.Expr。


**示例**

```python
batch_dims = 0
data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
output  = [0,3]           # output_shape  = [2]

batch_dims = 1
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[1],[0]]                     # indices_shape = [2, 1]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```
## tvm.relax.op.index_put(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *indices:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]*, *values:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *accumulate:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


此操作使用来自 values 的对应值更新 data 中由 indices 指定的位置。indices 是一个张量元组，其中每个张量对应 data 中的一个维度。当 accumulate 为 True 时，操作执行累积（加法）而不是替换。reduction 参数允许指定不同的缩减操作。 :param data: 要修改的输入张量 :type data: relax.Expr :param indices: 指定更新位置的索引张量元组（每个维度一个） :type indices: Union[Expr, Tuple[Expr]] :param values: 要放置在指定索引处的值 :type values: relax.Expr :param accumulate: 是否累积（加）值而不是替换（默认：False） :type accumulate: bool
* **返回：result** ：一个与 data 形状相同但指定位置已更新的新张量。
* **返回类型：** relax.Expr。


**示例**

```python
# inputs
data = torch.zeros(3, 3)
indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
values = torch.tensor([1.0, 2.0])
# output
output = [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
]
# with accumulate=True
output = [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0],
]

```
## tvm.relax.op.index_tensor(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *indices:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

高级张量索引（NumPy/PyTorch 风格）。


给定 k 个索引张量 `indices = (I0, I1, …, Ik‑1)` ，此运算符从 `data` 中选择元素，就像在 NumPy/PyTorch 中编写了 `data[I0, I1, …, Ik‑1]` 一样：


所有索引张量必须具有整数 dtype。


它们的形状按照常规的 NumPy 方式广播到一个公共形状 `B` 。


结果形状是 `B + data.shape[k:]` （即广播形状后跟 `data` 中未索引的剩余轴）。


在编译时，Relax 检查索引张量的数量 `k` 是否不超过 `data.ndim` ，dtype 是否为整数，以及形状是否一致（广播兼容）。
* **参数：**
   * **data** (*relax.Expr*) `：`要索引的输入张量。
   * **indices** (*Union**[****relax.Expr**,*** ***List****[**relax.Expr****]]*) ：一个包含索引张量的元组表达式，或一个将被自动提升为元组表达式的 Python `list` / `tuple` 。每个张量必须具有整数 dtype。
* **返回：result** ：高级索引后得到的张量。其 dtype 等于 `data.dtype`。
* **返回类型：** relax.Expr。


**示例**

```python
import numpy as np
import tvm.relax as R

x   = R.const(np.arange(9).reshape(3, 3).astype("float32"))
row = R.const(np.array([0, 2]))        # shape (2,)
col = R.const(np.array([1, 0]))        # shape (2,)

y = R.index_tensor(x, [row, col])
# y.shape == (2,) ;  y == [1., 6.]

# Broadcasting: row : (2,1), col : (1,3)  →  B = (2,3)
row = R.const(np.array([[0],[1]]))
col = R.const(np.array([[0,1,2]]))
z = R.index_tensor(x, [row, col])
# z.shape == (2,3)
```
## tvm.relax.op.meshgrid(*tensors:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]*, *indexing:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 'ij'*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

从输入张量生成坐标网格。
* **参数：**
   * **tensors** (*Union*[****relax.Expr**,** ***List***[**relax.Expr**]]) ：一个元组类型的 Expr，包含 1D 张量（或提升为 1D 的标量），用于生成坐标网格，或此类张量的列表。
   * **indexing** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) *：* 索引模式，为“ij”（矩阵索引）或“xy”（笛卡尔索引）。默认为“ij”。
* **返回：result** *：* 一个表示坐标网格的张量元组。
* **返回类型：** relax.Expr。

## tvm.relax.op.layout_transform(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *index_map:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*|*[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map), *pad_value:*[int](https://docs.python.org/3/library/functions.html#int)*|*[float](https://docs.python.org/3/library/functions.html#float)*|*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *axis_separators:*[int](https://docs.python.org/3/library/functions.html#int)*| axis_separator |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *input_axis_separators:*[int](https://docs.python.org/3/library/functions.html#int)*| axis_separator |*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


修改张量的布局。
* **参数：**
   * **x** (*relax.Expr*) ***：*** 运算符的输入张量。
   * **index_map** (*Union**[****Callable,*[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*]*) ：应用转换。
   * **pad_value** (*Optional*[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,** [float](https://docs.python.org/3/library/functions.html#float)***,*** [PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)***]****]*) ：如果转换结果导致隐式填充，则用于填充的值。如果未指定，可以使用任何值。
   * **axis_separators** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,** ***IndexMap.AXIS_SEPARATOR****]]*) ：用于 index_map 创建非扁平化缓冲区的 axis_separators。
* **返回：result** *：* 转换后的张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.one_hot(*indices:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *on_value:*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone), *off_value:*[PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone), *depth:*[int](https://docs.python.org/3/library/functions.html#int), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


返回一个 one-hot 张量。
* **参数：**
   * **indices** (*relax.Expr*) **：** 要设置为 on_value 的索引。
   * **on_value** ([relax.PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)) ：要在 indices 处填充的值。
   * **off_value** ([relax.PrimValue](/docs/api-reference/python-api/tvm-relax#classtvmrelaxprimvaluevalueprimexprintspanspannonenone)) ：要在其他位置填充的值。
   * **depth** ([int](https://docs.python.org/3/library/functions.html#int)) ：hot 维度的深度。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)  -1，即在末尾添加一个新维度。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。


**示例**

```python
indices = [0, 1, 2]
depth = 3
on_value = 1
off_value = 0

one_hot(indices, on_value, off_value, depth) =
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
```
>tvm.relax.op.permute_dims(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axes:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr) 

对数组的维度进行重排。
* **参数：**
   * **x** (*relax.Expr*) ：运算符的输入数据。
   * **axes** (*Optional*[****List**[***[int](https://docs.python.org/3/library/functions.html#int)***]***]*) **：** 目标轴顺序。如果未指定，permute_dims 将反转所有轴的顺序。
* **返回：result** ：转置结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.repeat(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *repeats:*[int](https://docs.python.org/3/library/functions.html#int), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


重复数组中的元素。
* **参数：**
   * **data** (*relax.Expr*) ：输入张量。
   * **repeats** ([int](https://docs.python.org/3/library/functions.html#int)) ：重复次数。
   * **axis** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*) ：重复值的轴。负数从后向前计数。默认情况下，使用展平的输入数组，并返回一个展平的输出数组。
* **返回：ret** ：计算结果。
* **返回类型：** relax.Expr。


**示例**

```python
x = R.const([[1, 2], [3, 4]])
lv1 = R.repeat(x, repeats=2) # lv1 == [1, 1, 2, 2, 3, 3, 4, 4]
lv2 = R.repeat(x, repeats=2, axis=1) # lv2 == [[1., 1., 2., 2.],
                                     #         [3., 3., 4., 4.]]
```
## tvm.relax.op.reshape(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *shape:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


重塑输入数组。


`-1` 通过使用输入维度的余数来推断输出形状的维度，同时保持新数组的大小与输入数组相同。shape 最多可以有一个维度为-1。

```python
x.shape = (2, 3, 4), shape = (6, 1, -1), result.shape = (6, 1, 4)
x.shape = (2, 3, 4), shape = (3, -1, 8), result.shape = (3, 1, 8)
x.shape = (2, 3, 4), shape = (-1,), result.shape = (24,)
```
* **参数：**
   * **x** (*relax.Expr*) ：运算符的输入数据。
   * **shape** (*Union*[***[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[****PrimExprLike**]***,Expr]*) ：新的形状。应与原始形状兼容。
* **返回：result** ：重塑后的结果。
* **返回类型：** relax.Expr。


 :::note

`-1` 推理仅在编译时执行。也就是说，在任何情况下，如果 `-1` 的维度长度不能在编译时推断，将会抛出错误。

:::

## tvm.relax.op.scatter_elements(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *indices:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *updates:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *reduction:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'update'*)

ONNX 风格的散布元素。此操作根据 indices 指定的特定索引位置，将 data 中的值更新为 updates 中指定的值。例如，在 2D 张量中，对应于[i][j]条目的更新操作如下：

```python
output[indices[i][j]][j] = updates[i][j] if axis = 0
output[i][indices[i][j]] = updates[i][j] if axis = 1
```


当 reduction 设置为某种归约函数 f 时，对应于[i][j]条目的更新操作如下：

```python
output[indices[i][j]][j] += f(output[indices[i][j]][j], updates[i][j]) if axis = 0
output[i][indices[i][j]] += f(output[i][indices[i][j]], updates[i][j]) if axis = 1
```


其中 f 是 update、add、mul、mean、max、min。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **indices** (*relax.Expr*) ：在 data 中需要更新的索引位置。
   * **updates** (*relax.Expr*) ：用于替换的值。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：散布的轴。
   * **reduction** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：应用的归约类型：update（更新）、add（加法）、mul（乘法）、mean（平均值）、max（最大值）、min（最小值）。默认为“update”。
* **返回：result** ：结果与 data 具有相同的大小，形状也与 data 相同。
* **返回类型：** relax.Expr。


**示例**

```python
# inputs
data = [
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
 ]
 indices = [
     [1, 0, 2],
     [0, 2, 1],
 ]
 updates = [
     [1.0, 1.1, 1.2],
     [2.0, 2.1, 2.2],
 ]
 axis = 0
 reduction = "update"

 # output P
 output = [
     [2.0, 1.1, 0.0]
     [1.0, 0.0, 2.2]
     [0.0, 2.1, 1.2]
 ]
```
## tvm.relax.op.scatter_nd(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *indices:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *updates:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *reduction:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'update'*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

将更新分散到数组中，根据索引。
* **参数：**
   * **data** (*relax.Expr*) ：要更新的输入数据。
   * **indices** (*relax.Expr*) ：data 中要更新的索引位置。
   * **updates** (*relax.Expr*) ：要替换的值。
   * **reduction** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：应用约简的类型：update、add、mul、max、min。默认为「update」。
* **返回：result** -结果与 data 具有相同的形状。
* **返回类型：** relax.Expr。


**示例**

```python
# inputs
data = [1, 2, 3, 4, 5, 6, 7, 8]
indices = [[4], [3], [1], [7]]
updates = [9, 10, 11, 12]

# output
output = [1, 11, 3, 10, 9, 6, 7, 12]
```
## tvm.relax.op.slice_scatter(*input_tensor:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *src:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *start*, *end*, *step*, *axis=0*)

将 src 张量的值嵌入到 input 的指定维度中。
* **参数：**
   * **input_tensor** (*relax.Expr*) -要更新的输入张量。
   * **src** (*relax.Expr*) ：要嵌入到输入中的张量。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：插入切片的维度。
   * **start** ：插入切片的起始索引。
   * **end** ：插入切片的结束索引。
   * **step** ：跳过的元素数量。
* **返回：result** ：与 data 相同形状的计算结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.split(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *indices_or_sections:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


沿轴通过分段或索引分割输入张量。


如果 indices_or_sections 是一个整数，输入将沿给定轴均匀分割（如果可能）。如果沿给定维度的张量大小不能被整数整除，最后一个分段将更小。


如果 indices_or_sections 是一个由整数或 PrimExpr 混合组成的元组，则其中的条目指示沿轴分割数组的索引。
* **参数：**
   * **x** (*relax.Expr*) **：** 要分割的张量。
   * **indices_or_sections** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,***List***[**PrimExprLike**]]) ：分割成的索引或部分。接受一个整数或一个列表。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：沿此轴分割。
* **返回：ret** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.squeeze(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


在数组中压缩轴。
* **参数：**
   * **x** (*relax.Expr*) ：运算符的输入数据。
   * **axis** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***List****[*[int](https://docs.python.org/3/library/functions.html#int)*]]*) ：要移除的轴集。如果 axis = None，则移除所有维度为 1 的轴。如果任何指定的轴的维度不等于 1，则会产生错误。
* **返回：result** ：压缩后的结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.stack(*tensors:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

沿着新轴堆叠输入张量。
* **参数：**
   * **tensors** (*Union**[****relax.Expr**,*** ***List****[**relax.Expr****]]*) ：一个包含待堆叠张量的元组类型的 Expr，或张量列表。所有输入张量必须具有相同的形状。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：结果张量中输入张量将被堆叠的轴。负值会循环。默认为 0。
* **返回：result** ：与输入张量相比，堆叠的张量多了一个维度。
* **返回类型：** relax.Expr。

## tvm.relax.op.tile(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *repeats:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


通过 repeats 指定的次数重复 data 来构造一个数组。


如果 repeats 的长度为 l，data 的维度为 d，则结果将具有 max(l, d)的维度。


如果 d < l，数据将通过在前面添加新的轴来提升为 l 维。因此，形状为 (3,) 的张量在 2 维复制时会提升为 (1, 3)，在 3 维复制时会提升为 (1, 1, 3)。如果这不是期望的行为，请在调用此函数之前手动将数据提升为 d 维。


如果 d > l，reps 将通过在其前面添加 1 来提升为长度 d。因此，对于形状为 (2, 3, 4, 5) 的数据，一个 reps 为 (2, 2) 将被视为 (1, 1, 2, 2)。
* **参数：**
   * **data** (*relax.Expr*) **：** 运算符的输入数据。
   * **repeats** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***]****,List[***[int](https://docs.python.org/3/library/functions.html#int)*]*]) **：** 沿每个轴的数据重复次数。
* **返回：ret** ：计算结果。
* **返回类型：** relax.Expr。


**示例**

```python
x = R.const([[1, 2], [3, 4]])
lv1 = R.tile(x, reps=(2, 3)) # lv1 = [[1., 2., 1., 2., 1., 2.],
                             #        [3., 4., 3., 4., 3., 4.],
                             #        [1., 2., 1., 2., 1., 2.],
                             #        [3., 4., 3., 4., 3., 4.]]
lv2 = R.tile(x, reps=2) # lv2 = [[1., 2., 1., 2.],
                        #        [3., 4., 3., 4.]]
```
## tvm.relax.op.masked_fill(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *mask:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *value:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr))


用指定的值填充由掩码定义位置的张量。 :param x: 运算符的输入数据。 :type x: relax.Expr :param mask: 掩码。 :type mask: relax.Expr :param value: 要在输入张量中设置的值。 :type value: relax.Expr
* **返回：result** ：填充的张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.dequantize(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *scale:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *zero_point:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'float32'*)


去量化运算符 该运算符接收输入并产生去量化输出。输入张量可以是任何形状。输出形状与输入形状相同。


output = clamp(scale 运算符* (input_tensor - zero_point), out_dtype::min, out_dtype::max)。
* **参数：**
   * **data** (*tvm.relax.Expr*) ：待解量的输入张量。
   * **scale** (*tvm.relax.Expr*) ：输入缩放值。
   * **zero_point** (*tvm.relax.Expr*) ：输入零点。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：1，对应最后一个轴。
   * **out_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*) ：输出张量的数据类型。
* **返回：result** ：计算结果。
* **返回类型：** tvm.relax.Expr。

## tvm.relax.op.quantize(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *scale:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *zero_point:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'int8'*)


量化操作 该操作接收输入并产生量化输出。输入张量可以是任意形状。输出形状与输入形状相同。


Q_output = clamp((round(input_tensor/scale) + zero_point), out_dtype::min, out_dtype::max)。
* **参数：**
   * **data** (*tvm.relax.Expr*) **：** 待量化的输入张量。
   * **scale** (*tvm.relax.Expr*) ：输出缩放比例。
   * **zero_point** (*tvm.relax.Expr*) - 输出 zero_point。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) **：** 1，对应最后一个轴。
   * **out_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*) ：输出张量的数据类型。
* **返回：result** ：计算结果。
* **返回类型：** tvm.relax.Expr。

## tvm.relax.op.multinomial_from_uniform(*prob:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *uniform_sample:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *sample_indices:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'int64'*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr) 


返回一个张量，其中每一行包含从位于张量 prob 相应行的多项式概率分布中采样的索引。


**注意**


为了更好的 CPU 性能，请使用‘vm.builtin.multinomial_from_uniform’。为了获得准确的结果，请确保概率在 0 和 1 之间且总和为 1。
* **参数：**
   * **prob** (*relax.Expr*) ：D 张量，表示概率分布。每一行代表一个批次在词汇表上的分布，其中：值范围在 [0, 1] 之间，表示每个词汇项的概率。每行的值之和为 1，形成一个有效的分布。
   * **uniform_sample** (*relax.Expr*) ：D 张量。值范围在 0 到 1 之间，表示均匀采样的概率。
   * **sample_indices** (*relax.Expr*) ：D 张量，指示要采样的具体概率分布。sample_indices[i] 的值决定了第 i 个词应该从第 sample_indices[i] 个概率分布中采样。例如，如果有 3 个不同的概率分布，并且要求从每个分布中采样 2、3 和 4 个词，那么 sample_indices 将是 [0, 0, 1, 1, 1, 2, 2, 2, 2]。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：输出张量的数据类型。
* **返回：result** ：计算得到的形状为 (n, 1) 的张量。
* **返回类型：** relax.Expr。


**示例**

```python
prob = [[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]]
usample = [[0.4], [0.9]]
sample_indices = [[0], [1]]

multinomial_from_uniform(prob, usample)
-> [[1], [2]]
multinomial_from_uniform(prob, usample, sample_indices)
-> [[1], [2]]
```
## tvm.relax.op.argmax(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算给定轴上张量元素的 argmax。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据张量。
   * **axis** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*) ：沿着该轴执行 argmax 操作。默认值 axis=None 将计算输入张量中所有元素的 argmax。支持负索引。
   * **keepdims** ([bool](https://docs.python.org/3/library/functions.html#bool)) **：** 如果设置为 True，被缩减的轴将保留在结果中作为大小为一的维度。使用此选项，结果将正确地广播到输入张量。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.argmin(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算给定轴上张量元素的最小值。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据张量。
   * **axis** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*) ：执行 argmin 操作的轴。默认值 axis=None 将计算输入张量中所有元素的最小值。支持负索引。
   * **keepdims** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：如果设置为 True，被缩减的轴将保留在结果中作为大小为一的维度。使用此选项，结果将正确地广播到输入张量。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.where(*condition:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

根据条件值从输入张量中选择元素。


对于给定位置，如果条件为 True，返回 x1 中的对应值；否则返回 x2 中的对应值。
* **参数：**
   * **condition** (*relax.Expr*) ：当为 True 时，返回 x1；否则返回 x2。必须与 x1 和 x2 广播兼容。必须具有布尔数据类型。
   * **x1** (*relax.Expr*) **：** 第一个输入张量。必须与 condition 和 x2 兼容广播。
   * **x2** (*relax.Expr*) ：第二个输入张量。必须与 condition 和 x1 兼容广播。
* **返回：result** **：** 结果张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.nonzero(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


找到张量中非零元素的索引。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据张量。
* **返回：result** ：一个包含非零元素索引的二维张量。
* **返回类型：** relax.Expr。

:::note

该函数等价于 onnx.nonzero。

:::


**示例**

```python
x = [[0, 1],
     [2, 0]]
nonzero(x) = [[0, 1],
              [1, 0]]
```
## tvm.relax.op.unique(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *sorted:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*= True*, *return_index:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*= False*, *return_inverse:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*= False*, *return_counts:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*= False*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


查找给定张量中的唯一元素。此外，它还可以选择性地返回 - 输入张量中给出唯一值的索引； - 重建输入张量的唯一张量的索引； - 每个唯一值在输入张量中出现的次数。
* **参数：**
   * **x** (*relax.Expr*) ：输入张量。
   * **sorted** (*Union*[***[bool](https://docs.python.org/3/library/functions.html#bool)***,***Expr***]) **：** 是否在返回输出前对唯一元素进行升序排序。
   * **return_index** (*Union*[***[bool](https://docs.python.org/3/library/functions.html#bool)***,***Expr***]) ：是否返回一个额外的张量，包含唯一张量中元素在原始输入中的索引。
   * **return_inverse** (*Union*[***[bool](https://docs.python.org/3/library/functions.html#bool)***,***Expr***]) ：是否返回一个额外的张量，包含原始输入中的元素在返回的唯一列表中的索引。
   * **return_counts** (*Union*[***[bool](https://docs.python.org/3/library/functions.html#bool)***,***Expr***]) ：是否返回一个额外的张量，其中包含每个唯一元素的数量。
   * **axis** (*Optional*) ：应用于唯一值的维度。如果未指定，则返回扁平化输入的唯一值。
* **返回：ret** ：创建的 relax 调用，
* **返回类型：** relax.Expr

## tvm.relax.op.argsort(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*, *descending:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'int32'*)

沿给定轴进行排序，并返回一个与输入数组形状相同的索引数组，该数组按排序顺序索引数据。
* **参数：**
   * **data** (*relax.Expr*) ：输入数据张量。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：沿此轴对输入张量进行排序。
   * **descending** ([bool](https://docs.python.org/3/library/functions.html#bool)) -是否按降序排序，默认为 False，
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str)) *：* 输出索引的数据类型。
* **返回：out** ：与 data 形状相同的张量。
* **返回类型：** relax.Expr，

## tvm.relax.op.sort(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*, *descending:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)


沿给定轴进行排序，并返回排序后的数组。
* **参数：**
   * **x** (*relax.Expr*) ：输入张量。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) *：* 沿着哪个轴对输入张量进行排序。默认使用输入的最后一个轴。
   * **descending** ([bool](https://docs.python.org/3/library/functions.html#bool)) *：* 是否按降序排序，默认为 False。
* **返回：out** *：* 排序后的张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.topk(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *k:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*, *ret_type:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'both'*, *largest:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'int32'*)


获取输入张量沿给定轴的 top k 元素。


ret_type 指定返回类型，可以是（"both"，"values"，"indices"）之一。
* **参数：**
   * **data** (*relax.Expr*) ：输入数据张量。
   * **k** ([int](https://docs.python.org/3/library/functions.html#int)) ：选择顶部元素的数量。如果 k < 1，则返回所有元素。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) **：** 按指定轴对输入张量进行排序的轴。
   * **ret_type** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：返回类型 [both, values, indices]。“both”：返回 top k 数据和索引。“values”：仅返回 top k 数据。“indices”：仅返回 top k 索引。
   * **largest** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：是否返回最大或最小的元素。如果 largest 为 False，则返回 k 个最小的元素。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str)) *：* 索引输出的数据类型。
* **返回：out** *：* 计算结果。
* **返回类型：** relax.Expr or List[relax.Expr]。

## tvm.relax.op.cumprod(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *exclusive:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)

Numpy 风格的累积乘积操作。返回沿给定轴的元素累积乘积。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **axis** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*) ：计算累积乘积的轴。默认值（None）是对展平数组进行 cumprod 计算。
   * **dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***DataType****]]*) ：返回数组的类型以及计算元素时使用的累加器的类型。如果未指定 dtype，则默认为 data 的类型。
   * **exclusive** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：如果为 false（默认），则所有元素都包含在乘积中。如果为 true，则第一个元素被排除在乘积之外。
* **返回：result** ：结果与 data 的大小相同，如果 axis 不为 None，则形状也与 data 相同。如果 axis 为 None，结果是一个 1 维数组。
* **返回类型：** relax.Expr。


**示例**

```python
a = [[1, 2, 3], [4, 5, 6]]

cumprod(a)  # if axis is not provided, cumprod is done over the flattened input.
-> [ 1,  2,  6, 24, 120, 720]

cumprod(a, dtype="float32")
-> [  1.,  2.,  6., 24., 120., 720.]

cumprod(a, axis=0)  # multiply over rows for each of the 3 columns
-> [[1, 2, 3],
    [4, 10, 18]]

cumprod(a, axis=1)
-> [[ 1,  2,  6],
    [ 4,  20, 120]]

a = [1, 1, 1, 0, 1, 1, 0]  # a is a boolean array
cumprod(a, dtype=int32)  # dtype should be provided to get the expected results
-> [1, 1, 1, 0, 0, 0, 0]
```
## tvm.relax.op.cumsum(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *exclusive:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)

Numpy 风格的累积和操作。返回沿给定轴的元素的累积包含和。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **axis** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*) ：沿其计算累积和的轴。默认值（None）是计算扁平化数组的 cumsum。
   * **dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***DataType****]]*) ：返回数组的类型以及在其中求和的累加器的类型。如果未指定 dtype，则默认为 data 的 dtype。
   * **exclusive** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：如果为 false（默认值），则所有元素都包含在和中。如果为 true，则第一个元素被排除在和中。
* **返回：result** ：结果与 data 的大小相同，如果 axis 不为 None，则形状也与 data 相同。如果 axis 为 None，结果是一个 1 维数组。
* **返回类型：** relax.Expr。


**示例**

```python
a = [[1, 2, 3], [4, 5, 6]]

cumsum(a)  # if axis is not provided, cumsum is done over the flattened input.
-> [ 1,  3,  6, 10, 15, 21]

cumsum(a, dtype="float32")
-> [  1.,   3.,   6.,  10.,  15.,  21.]

cumsum(a, axis=0)  # sum over rows for each of the 3 columns
-> [[1, 2, 3],
    [5, 7, 9]]

cumsum(a, axis=1)
-> [[ 1,  3,  6],
    [ 4,  9, 15]]

a = [1, 0, 1, 0, 1, 1, 0]  # a is a boolean array
cumsum(a, dtype=int32)  # dtype should be provided to get the expected results
-> [1, 1, 2, 2, 3, 4, 4]
```
## tvm.relax.op.max(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

计算给定轴上张量元素的最大值。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据张量。
   * **axis** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***List****[*[int](https://docs.python.org/3/library/functions.html#int)*]]*]*) ：沿其执行最大值操作的轴或轴。默认值 axis=None 将计算输入张量中所有元素的最大值。支持负索引。
   * **keepdims** ([bool](https://docs.python.org/3/library/functions.html#bool)) *：* 如果设置为 True，则被缩减的轴将保留为结果中的尺寸为一的维度。使用此选项，结果将正确广播与输入张量。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.mean(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算给定轴上张量元素的平均值。
* **参数：**
   * **x** (*relax.Expr*) **：** 输入数据张量。
   * **axis** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***List****[*[int](https://docs.python.org/3/library/functions.html#int)*]*]*]) ：指定执行平均操作的轴或轴。默认值 axis=None 将计算输入张量中所有元素的平均值。支持负索引。
   * **keepdims** ([bool](https://docs.python.org/3/library/functions.html#bool)) *：* 如果设置为 True，被缩减的轴将保留为结果中的尺寸为一的维度。使用此选项，结果将正确广播与输入张量。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.min(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算在给定轴上张量元素的最小值。
* **参数：**
   * **x** (*relax.Expr*) *：* 输入数据张量。
   * **axis** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***List****[*[int](https://docs.python.org/3/library/functions.html#int)*]**]***]*) ：沿着其执行最小值操作的轴或轴。默认值 axis=None 将计算输入张量中所有元素的最小值。支持负索引。
   * **keepdims** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：如果设置为 True，则被缩减的轴将保留在结果中作为大小为 1 的维度。使用此选项，结果将正确地广播到输入张量。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.prod(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


对指定轴上的张量元素进行乘积计算。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据张量。
   * **axis** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***List****[*[int](https://docs.python.org/3/library/functions.html#int)*]**]**]) ：执行乘积计算的轴或轴列表。默认值 axis=None 将计算输入张量所有元素的乘积。支持负索引。
   * **keepdims** ([bool](https://docs.python.org/3/library/functions.html#bool)) -如果设置为 True，则被缩减的轴将保留为结果中的维度，其大小为 1。使用此选项，结果将正确地广播到输入张量。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.std(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算张量元素在给定轴上的标准差。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据张量。
   * **axis** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***List****[*[int](https://docs.python.org/3/library/functions.html#int)*]]*]*) *：* 指定执行标准差计算的轴或轴。默认值 axis=None 将计算输入张量所有元素的标准差。支持负索引。
   * **keepdims** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：如果设置为 True，被缩减的轴将保留为结果中的尺寸为一的维度。使用此选项，结果将正确广播到输入张量。
* **返回：result** **：** 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.sum(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


对指定轴上的张量元素求和。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据张量。
   * **axis** (*Optional*[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,***List****[*[int](https://docs.python.org/3/library/functions.html#int)]**]**]) - 求和的轴或轴列表。默认值 axis=None 将求和输入张量的所有元素。支持负索引。
   * **keepdims** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：如果设置为 True，则被缩减的轴将保留为结果中的尺寸为一的维度。使用此选项，结果将正确广播与输入张量。
* **返回：result** ***：*** 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.variance(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算张量元素在给定轴上的方差。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据张量。
   * **axis** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***List*** *[*[int](https://docs.python.org/3/library/functions.html#int)*]* *]]*) *：* 执行方差操作的轴或轴。默认值 axis=None 将计算输入张量中所有元素的方差。支持负索引。
   * **keepdims** ([bool](https://docs.python.org/3/library/functions.html#bool)) *：* 如果设置为 True，则被缩减的轴将保留为结果中的尺寸为一的维度。使用此选项，结果将正确广播与输入张量。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.ewise_fma(*x1:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x2:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x3:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


元素级融合乘加运算符 返回 x1∗x2+x3 的元素级结果。
* **参数：**
   * **x1** (*relax.Expr*) ：乘法的左操作数。
   * **x2** (*relax.Expr*) -乘法的右操作数。
   * **x3** (*relax.Expr*) ：加法的操作数。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.abs(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素绝对值。
* **参数：**
   * **x** (*relax.Expr*) -输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.acos(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

计算输入数据的逐元素反余弦。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.acosh(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

计算输入数据的逐元素反双曲余弦。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.asin(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素反正弦值。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.asinh(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

计算输入数据的逐元素反双曲正弦。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** -计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.atan(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素反正切。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.atanh(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素反双曲正切。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.bitwise_not(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐位取反。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.ceil(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


对输入数据取上整。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** **：** 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.clip(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *min:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *max:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


将张量值裁剪到指定的最小值和最大值。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
   * **min** (*relax.Expr*) ：最小值。
   * **max** (*relax.Expr*) **：** 最大值。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.cos(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素余弦值。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.cosh(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素双曲余弦。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.erf(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入的错误函数。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算每个元素的错误函数。
* **返回类型：** relax.Expr。

## tvm.relax.op.exp(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算数据的逐元素指数。
* **参数：**
   * **x** (*relax.Expr*) **：** 输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.floor(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


取输入数据的下界。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.isfinite(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


检查输入值是否有限。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.isinf(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


检查输入值是否为无穷大。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.isnan(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


检查输入值是否为 NaN。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** -计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.log(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素自然对数。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.logical_not(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逻辑非。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** -计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.negative(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素负值。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.round(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


将输入数据的每个元素四舍五入到最近的整数。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** -计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.rsqrt(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

计算输入数据的逐元素倒数平方根。

$$1/sqrt(x)$$
* **参数：**
   * **x** (*relax.Expr*) ：输入数据
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.sigmoid(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素 Sigmoid 函数。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.sign(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


为输入数据的每个元素返回一个指示该数字符号的标志。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.sin(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素正弦值。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.sinh(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素双曲正弦值。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据
* **返回：result** -计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.sqrt(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

计算输入数据的逐元素平方根。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据
* **返回：result** -计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.square(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


对输入数据的每个元素进行平方。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.tan(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算输入数据的逐元素正切值。
* **参数：**
   * **x** (*relax.Expr*) *：* 输入数据
* **返回：result** -计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.tanh(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

计算输入数据的逐元素 tanh。
* **参数：**
   * **x** (*relax.Expr*) -输入数据
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.trunc(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


取输入数据的截断值。:param x: 输入数据 :type x: relax.Expr。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## 

与神经网络相关的运算符。

## tvm.relax.op.nn.adaptive_avg_pool1d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *output_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


1D 自适应平均池化运算符。该运算符是实验性的。


该运算符以数据为输入，并在由 W 表示的每个窗口上进行 1D 平均值计算。


在默认情况下，当 data_layout 为 NCW 时，一个形状为(batch_size, in_channels, width)的数据张量将产生一个形状为(batch_size, in_channels, output_width)的输出张量。


池化核大小和步长大小会根据期望的输出大小自动选择。


### 对于 output_size： 


如果未提供此参数，输入的高度和宽度将用作输出宽度。


如果为 output_size 提供一个整数，则对于任何输入（NCW），输出大小为（N x C x output_size）。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **output_size** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]]*) ：输出高度和宽度。如果未指定，则与输入的高度和宽度相同。如果指定，则长度必须为 1 或 2。
   * **layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：输入的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：输出的布局。如果未指定，则与 data_layout 相同
* **返回：result** **：** 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.adaptive_avg_pool2d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *output_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


2D 自适应平均池化运算符。此运算符为实验性功能。


这个运算符将数据作为输入，并在每个由 WxH 表示的窗口上进行 2D 平均值计算。

在默认情况下，当 data_layout 为 NCHW 时，一个形状为(batch_size, in_channels, height, width)的数据张量，将产生一个形状为(batch_size, in_channels, output_height, output_width)的输出张量。


池化核和步长大小会根据期望的输出大小自动选择。


### 对于 output_size： 


如果未提供此参数，输入的高度和宽度将用作输出高度和宽度。


如果为 output_size 提供了一个整数，则对于任何输入（NCHW），输出大小为 (N x C x output_size x output_size)。


如果为 output_size 提供了一个整数的元组（高度，宽度），则对于任何输入（NCHW），输出大小为 (N x C x 高度 x 宽度)。
* **参数：**
   * **data** (*relax.Expr*) **：**运算符的输入数据。
   * **output_size** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]]*) ：输出高度和宽度。如果未指定，则与输入高度和宽度相同。如果指定，其长度必须是 1 或 2。
   * **layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) **：** 输入的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：输出的布局。如果未指定，则与 data_layout 相同
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.adaptive_avg_pool3d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *output_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCDHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


3D 自适应平均池化运算符。此运算符为实验性功能。


这个运算符以数据为输入，并在每个由 WxH 表示的窗口上进行 3D 平均值计算。


在默认情况下，当 data_layout 为 NCDHW 时，一个形状为(batch_size, in_channels, depth, height, width)的数据张量，将产生一个形状为(batch_size, in_channels, output_depth, output_height, output_width)的输出张量。


池化核大小和步长大小会根据期望的输出大小自动选择。


### 对于 output_size： 


如果这个参数没有提供，输入的深度、高度和宽度将作为输出的深度、高度和宽度使用。


如果为 output_size 提供一个整数，则输出大小为(N x C x output_size x output_size x output_size)，适用于任何输入(NCDHW)。

如果为 output_size 提供一个整数元组(depth, height, width)，则输出大小为(N x C x depth x height x width)，适用于任何输入(NCDHW)。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **output_size** (*Optional**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]]*) ：输出高度和宽度。如果未指定，则与输入高度和宽度相同。如果指定，则长度必须是 1 或 3。
   * **layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) *：* 输入的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：输出的布局。如果未指定，则与 data_layout 相同
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.attention(*query:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *key:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *value:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *bias:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *scale:*[FloatImm](/docs/api-reference/python-api/tvm-tir#class-tvmtirfloatimmdtypestr-valuefloat-spanspannone-none)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *causal_mask:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *window_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算融合多头注意力。


所有输入张量都是 BSNH 布局的 4 维张量。

$$FMA(Q, K, V) = \text{Softmax}(Q @ K^T) @ V$$
:::note

输入张量必须具有 float16 数据类型。

:::
* **参数：**
   * **query** (*relax.Expr*) ：运算符的输入查询。输入查询的布局应为 (batch_size, seq_len, num_head, head_dim)。
   * **key** (*relax.Expr*) ：运算符的输入键。输入键的布局应为 (batch_size, seq_len_kv, num_head, head_dim)。
   * **value** (*relax.Expr*) ：运算符的输入值。输入值的布局应为 (batch_size, seq_len_kv, num_head, head_dim_v)。
   * **bias** (*Optional*[*Expr]*) ：运算符的可选注意力偏置。注意力偏置的布局应为以 seq_len_kv 结尾的 4 维张量，且可广播到 (batch_size, num_head, seq_len, seq_len_kv)。
   * **scale** (*Optional[*[float](https://docs.python.org/3/library/functions.html#float)*]*) **：** 应用于注意力分数的缩放值，默认为 1 / sqrt(head_dim)。
   * **causal_mask** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) **：可选的因果掩码**，即“TopLeft”和“BottomRight”。对于“TopLeft”，掩码矩阵为 `np.tril( , k=0)`，而对于“BottomRight”，掩码矩阵为 `np.tril( , k=abs(seq_len - seq_len_kv))`。例如，若 `seq_len = 4`，`seq_len_kv = 2`，则“TopLeft”的掩码为：


```plain
[[1, 0],
[1, 1],
[1, 1],
[1, 1]]
```
“BottomRight”的掩码：
```plain
[[1, 1],
[1, 1],
[1, 1],
[1, 1]]
```
以 seq_len = 2, seq_len_kv = 4 为例，'TopLeft'的掩码为：
```plain
[[1, 0, 0, 0],
[1, 1, 0, 0]]
```
“BottomRight”的掩码：
```plain
[[1, 1, 1, 0],
[1, 1, 1, 1]]
```
* window_size (可选[int])：滑动窗口注意力的窗口大小。
* **返回：result** ：计算结果。输出布局应为(batch_size, seq_len, num_head, head_dim_v)。
* **返回类型：** relax.Expr

## tvm.relax.op.nn.attention_bias(*query:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *key:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *value:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *bias:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *scale:*[FloatImm](/docs/api-reference/python-api/tvm-tir#class-tvmtirfloatimmdtypestr-valuefloat-spanspannone-none)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *causal_mask:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *window_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算融合多头注意力。


IRModule.script() 将注意力操作转换为注意力偏置，这与 TVMScript 解析器不兼容。该函数使 TVMScript 的打印与 TVMScript 的解析器兼容。


所有输入张量都是 BSNH 布局的 4 维张量。

$$FMA(Q, K, V) = \text{Softmax}(Q @ K^T) @ V$$

:::note

输入张量必须具有 float16 数据类型。

:::
* **参数：**
   * **query** (*relax.Expr*) ：运算符的输入查询。输入查询的布局应为 (batch_size, seq_len, num_head, head_dim)。
   * **key** (*relax.Expr*) ：运算符的输入键。输入键的布局应为 (batch_size, seq_len_kv, num_head, head_dim)。
   * **value** (*relax.Expr*) ：运算符的输入值。输入值的布局应为 (batch_size, seq_len_kv, num_head, head_dim_v)。
   * **bias** (*Optional*[*Expr]*) ：运算符的可选注意力偏置。注意力偏置的布局应为以 seq_len_kv 结尾的 4 维张量，且可广播到 (batch_size, num_head, seq_len, seq_len_kv)。
   * **scale** (*Optional[*[float](https://docs.python.org/3/library/functions.html#float)*]*) *：* 应用于注意力分数的缩放值，默认为 1 / sqrt(head_dim)。
   * **causal_mask** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) `：`
   * **可选的因果掩码**，即“TopLeft”和“BottomRight”。对于“TopLeft”，掩码矩阵为 `np.tril( , k=0)`，而对于“BottomRight”，掩码矩阵为 `np.tril( , k=abs(seq_len - seq_len_kv))`。例如，当 `seq_len = 4`，`seq_len_kv = 2` 时，“TopLeft”的掩码为：

```plain
 [[1, 0],
[1, 1],
[1, 1],
[1, 1]]
```
“BottomRight”的掩码：
```plain
[[1, 1],
[1, 1],
[1, 1],
[1, 1]]
```
with seq_len = 2, seq_len_kv = 4, mask for ‘TopLeft’:
```plain
[[1, 0, 0, 0],
[1, 1, 0, 0]]
```
“BottomRight”的掩码：
```plain
[[1, 1, 1, 0],
[1, 1, 1, 1]]
```
* **window_size** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*) ：滑动窗口注意力的窗口大小。
* **返回：result** ：计算结果。输出布局应为(batch_size, seq_len, num_head, head_dim_v)。
* **返回类型：** relax.Expr

## tvm.relax.op.nn.attention_var_len(*queries:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *keys:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *values:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *seqstart_q:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *max_seqlen_q:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *seqstart_k:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *max_seqlen_k:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *scale:*[FloatImm](/docs/api-reference/python-api/tvm-tir#class-tvmtirfloatimmdtypestr-valuefloat-spanspannone-none)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *causal_mask:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *window_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

计算变长批量序列的融合多头注意力。


给定连接输入和序列长度信息，该运算符比单独为每个序列调用普通注意力运算符更高效地计算所有序列的注意力。
* **参数：**
   * **queries** (*relax.Expr*) **：** 沿第二个轴连接的输入查询。其形状必须为 (1, total_seq_len, num_head, head_dim)。
   * **keys** (*relax.Expr*) **：** 沿着第二轴连接的输入键。其形状必须为 (1, total_seq_len_kv, num_head, head_dim)。
   * **values** (*relax.Expr*) **：** 沿着第二轴连接的输入值。其形状必须为 (1, total_seq_len_kv, num_head, head_dim_v)。
   * **seqstart_q** (*Optional*[*Expr]*) ：查询序列长度的累积和，以 0 开头。其 dtype 必须为 int32。例如，如果批量处理的序列长度为[2, 5, 3]，则该张量的值为[0, 2, 7, 10]。
   * **seqstart_k** (*Optional*[*Expr]*) ：键序列长度的累积和，以 0 开头。默认情况下与 seqstart_q 相同。
   * **max_seqlen_q** (*Optional*[*Expr]*) ：批次中查询序列的最大长度。必须是 int32。
   * **max_seqlen_k** (*Optional*[*Expr]*) ：批次中键序列的最大长度。必须是 int32。默认情况下与 max_seqlen_q 相同。
   * **scale** (*Optional[*[float](https://docs.python.org/3/library/functions.html#float)*]*) ：应用于注意力分数的缩放值，默认为 1 / sqrt(head_dim)。
   * **causal_mask** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：
   * **可选的因果掩码**，即“TopLeft”和“BottomRight”。对于“TopLeft”，掩码矩阵为 `np.tril(` `, k=0)`，而对于“BottomRight”，掩码矩阵为 `np.tril(` `, k=abs(seq_len - seq_len_kv))`。例如，若 `seq_len = 4`，`seq_len_kv = 2`，则“TopLeft”的掩码为：

```plain
[[1, 0],
[1, 1],
[1, 1],
[1, 1]]
```
“BottomRight”的掩码：
```plain
[[1, 1],
[1, 1],
[1, 1],
[1, 1]]
```
with seq_len = 2, seq_len_kv = 4, mask for ‘TopLeft’:
```plain
[[1, 0, 0, 0],
[1, 1, 0, 0]]
```
“BottomRight”的掩码：
```plain
[[1, 1, 1, 0],
[1, 1, 1, 1]]
```
* **window_size** (*Optional[*[int](https://docs.python.org/3/library/functions.html#int)*]*) **：** 滑动窗口注意力的窗口大小。
* **返回：result** ：计算结果，形状为 (1, total_seq_len, num_head, head_dim_v)。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.avg_pool1d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *pool_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1,)*, *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1,)*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = (0, 0)*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1,)*, *ceil_mode:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *count_include_pad:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


1D 平均池化运算符。


此运算符接收数据作为输入，并通过 stride 定义的步长，使用 pool_size 大小的窗口进行 1D 平均值计算。


在默认情况下，当 data_layout 为 NCW 时，一个数据张量 shape 为(batch_size, channels, width)，用于生成输出张量。


ceil_mode 用于在计算输出形状时取上整或下整。count_include_pad 指示是否在计算中包含或排除填充的输入值。该运算符接受数据布局规范。
* **参数：**
   * **data** (*relax.Expr*) ***：*** 运算符的输入数据。
   * **pool_size** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***]****]*) *：* 池化窗口的大小。必须具有长度为 1。
   * **strides** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：池化的步长。必须具有长度为 1。
   * **padding** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：池化的填充。必须具有长度为 1 或 2。
   * **dilation** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：池化的膨胀。必须具有长度为 1。
   * **ceil_mode** ([bool](https://docs.python.org/3/library/functions.html#bool)) *：* 一个布尔值，指示是否使用 ceil 或 floor 来计算输出形状。使用 ceil 时，输入张量的每个元素都将被滑动窗口覆盖。
   * **count_include_pad** ([bool](https://docs.python.org/3/library/functions.html#bool)*,optional*) ：是否将填充包括在内以计算平均值。
   * **layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：输入的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) -输出的布局。如果未指定，则与 data_layout 相同
* **返回：result** ：计算结果。
* **返回类型：** Expr。

## tvm.relax.op.nn.avg_pool2d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *pool_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = (0, 0)*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *ceil_mode:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *count_include_pad:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


2D 平均池化运算符。


该运算符以数据为输入，通过指定步长 stride，在 pool_size 大小的窗口内进行 2D 平均值计算。


在默认情况下，当数据布局为 NCHW 时，一个形状为(batch_size, in_channels, height, width)的数据张量，将按照以下规则产生输出张量：



$$ 
\operatorname{out}(b,c,y,x) = \frac{1}{kh \cdot kw} 
\sum_{m=0}^{kh-1}\sum_{n=0}^{kw-1} 
\operatorname{data}(b,c,\text{stride}_{0}\cdot y + m,\ \text{stride}_{1}\cdot x + n) 
$$


在计算之前对数据进行填充。ceil_mode 用于在计算输出形状时取上整或下整。该运算符接受数据布局规范。
* **参数：**
   * **data** (*relax.Expr*) ***：*** 运算符的输入数据。
   * **pool_size** (*Union*[[int](https://docs.python.org/3/library/functions.html#int)**,** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) - 池化窗口的大小。必须具有长度为 1 或 2。
   * **strides** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) - 池化的步长。必须具有长度为 1 或 2。
   * **padding** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]]*) ：池化操作的填充。必须具有长度为 1、2 或 4。
   * **dilation** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) **：** 池化操作的膨胀率。必须具有长度为 1 或 2。
   * **ceil_mode** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：一个布尔值，指示是否使用 ceil 或 floor 来计算输出形状。使用 ceil 时，输入张量的每个元素都将被滑动窗口覆盖。
   * **count_include_pad** ([bool](https://docs.python.org/3/library/functions.html#bool)*,optional*) ：是否将填充包括在内以计算平均值。
   * **layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：输入的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) *：* 输出的布局。如果未指定，则与 data_layout 相同
* **返回：result** *：* 计算结果。
* **返回类型：** Expr。

## tvm.relax.op.nn.avg_pool3d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *pool_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1, 1)*, *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1, 1)*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = (0, 0, 0)*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1, 1)*, *ceil_mode:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *count_include_pad:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCDHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


2D 平均池化运算符。


这个运算符以数据为输入，通过 stride 定义的步长，使用 pool_size 大小的窗口进行 3D 平均值计算。


在默认情况下，当 data_layout 为 NCDHW 时，一个形状为(batch_size, channels, depth, height, width)的数据张量，将产生一个输出张量。


ceil_mode 用于在计算输出形状时取上整或下整。count_include_pad 指示是否在计算中包含或排除填充的输入值。该运算符接受数据布局规范。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **pool_size** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：池化窗口的大小。它必须具有 1 或 3 的长度。
   * **strides** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：池化的步长。必须具有长度为 1 或 3。
   * **padding** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]]*) ：池化操作的填充。长度必须是 1、3 或 6。
   * **dilation** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) *：* 池化操作的膨胀率。必须具有长度为 1 或 3。
   * **ceil_mode** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：一个布尔值，指示是否使用 ceil 或 floor 来计算输出形状。使用 ceil 时，输入张量的每个元素都将被滑动窗口覆盖。
   * **count_include_pad** ([bool](https://docs.python.org/3/library/functions.html#bool)*,optional*) ：是否将填充包括在内以计算平均值。
   * **layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) -输入的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：输出的布局。如果未指定，则与 data_layout 相同。
* **返回：result** ：计算结果。
* **返回类型：** Expr。

## tvm.relax.op.nn.batch_norm(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *gamma:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *beta:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *moving_mean:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *moving_var:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int), *epsilon:*[float](https://docs.python.org/3/library/functions.html#float)*= 1e-05*, *center:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *scale:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *momentum:*[float](https://docs.python.org/3/library/functions.html#float)*= 0.1*, *training:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

Batch normalization 层（Ioffe 和 Szegedy，2014）。


对每个批次进行输入规范化，即应用一个保持激活均值接近 0 和激活标准差接近 1 的变换。

$$
\begin{split}
\operatorname{data\_mean}[i] &= \operatorname{mean}(\operatorname{data}[:, i, :, \dots]) \\
\operatorname{data\_var}[i] &= \operatorname{var}(\operatorname{data}[:, i, :, \dots])
\end{split}
$$

均值和方差都通过将输入视为向量来返回标量。

然后计算归一化输出，其形状与输入相同，如下所示：

$$
\operatorname{out}[:, i, :, \dots] = 
\frac{\operatorname{data}[:, i, :, \dots] - \operatorname{data\_mean}[i]}{\sqrt{\operatorname{data\_var}[i] + \epsilon}} \cdot \gamma[i] + \beta[i]
$$


假设输入在轴 1 上的大小为 k，那么 `gamma` 和 `beta` 的形状都是(k,)。


除了输入和输出之外，此运算符还接受两个辅助状态， `moving_mean` 和 `moving_var` ，它们是长度为 k 的向量。它们是整个数据集的全局统计信息，通过


```plain
moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
moving_var = moving_var * momentum + data_var * (1 - momentum)
```


参数 `axis` 指定输入形状中哪个轴表示“通道”（单独归一化的组）。默认值为 1。指定 -1 将通道轴设置为输入形状中的最后一项。


:::note

这个运算符有两种模式：
* 训练模式。
   * 使用从当前批次计算出的均值和方差进行归一化。
   * 更新并返回运行均值和运行方差。
* 推理模式。
   * 使用 running_mean 和 running_var 参数进行归一化。
   * 不要更新运行均值和运行方差。直接返回原始值。

在合法化阶段，该运算符默认会被合法化为训练模式。

您可以使用 tvm.relax.transform.DecomposeOpsForInference 来分解运算符，使其执行推理模式计算。类似地，使用 tvm.relax.transform.DecomposeOpsForTraining 来执行训练模式计算。

:::
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **gamma** (*relax.Expr*) **：** gamma 缩放因子。
   * **beta** (*relax.Expr*) **：** beta 偏移因子。
   * **moving_mean** (*relax.Expr*) **：** 输入的运行均值。
   * **moving_var** (*relax.Expr*) ：输入的运行方差。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：沿此轴应用归一化。
   * **epsilon** ([float](https://docs.python.org/3/library/functions.html#float)) - 添加到方差中的小浮点数，以避免除以零。
   * **center** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：指示是否将 beta 偏移添加到归一化张量中。
   * **scale** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：指示是否将伽马缩放相乘。
   * **momentum** ([float](https://docs.python.org/3/library/functions.html#float)) ：用于 moving_mean 和 moving_var 更新的值。
   * **training** ([bool](https://docs.python.org/3/library/functions.html#bool)) *：* relax batch_norm 处于训练模式。要将其转换为推理模式，可以使用 DecomposeOpsForInference。
* **返回：result** - 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.conv1d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *weight:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*] = 1*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = 0*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*] = 1*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCW'*, *kernel_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'OIW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


1D 卷积。


该运算符将权重作为 1D 卷积核，并将其与数据卷积以产生输出。


在默认情况下，其中数据布局为 NCW，核布局为 OIW，conv1d 接收一个数据张量，其形状为(batch_size, in_channels, width)，以及一个权重张量，其形状为(channels, in_channels, kernel_w)，其中 kernel_w 是 W 核维度的长度，以产生一个输出张量，其规则如下：
$$
\operatorname{out}[b, c, x] = 
\sum_{dx, k} 
\operatorname{data}[b, k, \operatorname{strides} \cdot x + dx] \cdot \operatorname{weight}[c, k, dx]
$$

在计算之前，数据张量和权重张量分别应用填充和膨胀。该运算符接受数据布局规范。从语义上讲，该运算符将布局转换为规范布局（数据为 NCW，权重为 OIW），执行计算，然后转换为 out_layout。
* **参数：**
   * **data** (*relax.Expr*) *：* 运算符的输入数据。
   * **weight** (*relax.Expr*) ：权重表达式。
   * **strides** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：卷积的步长。必须具有长度 1。
   * **padding** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]]*) **：** 卷积前输入两侧的填充。必须具有长度为 1 或 2。
   * **dilation** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：指定用于扩张卷积的扩张率。必须具有长度 1。
   * **groups** ([int](https://docs.python.org/3/library/functions.html#int)) **：** 分组卷积将输入分成多少组。输入和输出通道数应该能被组数整除。
   * **data_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：输入的布局。
   * **kernel_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：权重的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：输出的布局。如果未指定，则与 data_layout 相同
   * **out_dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***DataType****]]*) *：* 指定混合精度 conv1d 的输出数据类型。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.conv1d_transpose(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *weight:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*] = 1*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = 0*, *output_padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*] = 0*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*] = 1*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCW'*, *kernel_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'IOW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


1D 转置卷积运算符。


这个运算符可以看作是 conv1d 的梯度运算符。


输出形状可以在 data_layout == “NCW”和 kernel_layout == “IOW”的简单情况下解释。假设数据形状为(N, in_channel, in_w)，权重形状为(in_channel, out_channel, weight_w)，需要保证 in_channel % groups == 0。输出形状将是(N, out_channel 运算符* groups, out_w)，其中
* out_w = ((in_w - 1) 运算符* strides[0] + weight_w - 2 运算符* padding[0] + output_padding[0])
* **参数：**
   * **data** (*relax.Expr*) ***：*** 运算符的输入数据。
   * **weight** (*relax.Expr*) **：** 权重表达式。
   * **strides** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：卷积的步长。必须具有长度 1。
   * **padding** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]]*) ：卷积前输入两侧的填充。必须具有长度为 1 或 2。
   * **output_padding** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…***]**]**,optional) - 用于区分输出形状。
   * **dilation** (*Union*[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：指定用于扩张卷积的扩张率。必须具有长度为 1。
   * **groups** ([int](https://docs.python.org/3/library/functions.html#int)) ：分组卷积将输入分成多少组。输入和输出通道数应该能被组数整除。
   * **data_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：输入的布局。
   * **kernel_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) -权重的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) *：* 输出的布局。如果未指定，则与 data_layout 相同
   * **out_dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***DataType****]]*) - 指定混合精度 conv2d 的输出数据类型。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.conv2d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *weight:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = (0, 0)*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCHW'*, *kernel_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'OIHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

二维卷积。


该运算符将权重作为卷积核，并将其与数据卷积以产生输出。


在默认情况下，其中数据布局为 NCHW，核布局为 OIHW，conv2d 接收一个形状为（batch_size，in_channels，height，width）的数据张量，以及一个形状为（channels，in_channels，kernel_h，kernel_w）的权重张量，其中 kernel_h 和 kernel_w 是 H 和 W 核维度的长度，以产生一个输出张量，其规则如下：

$$
\operatorname{out}[b, c, y, x] =
\sum_{dy, dx, k} 
\operatorname{data}[b, k, \operatorname{strides}[0] \cdot y + dy, \operatorname{strides}[1] \cdot x + dx] 
\cdot \operatorname{weight}[c, k, dy, dx]
$$


在计算之前，对数据和权重分别应用填充和膨胀。该运算符接受数据布局规范。从语义上讲，运算符将布局转换为规范布局（数据为 NCHW，权重为 OIHW），执行计算，然后转换为 out_layout。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **weight** (*relax.Expr*) ：权重表达式。
   * **strides** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：卷积的步长。必须具有长度为 1 或 2。
   * **padding** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]]*) *：* 卷积前输入两侧的填充。必须具有长度为 1、2 或 4。
   * **dilation** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]***]) **：** 指定用于扩张卷积的扩张率。必须具有长度为 1 或 2。
   * **groups** ([int](https://docs.python.org/3/library/functions.html#int)) **：** 分组卷积将输入分成多少组。输入和输出通道数应该能被组数整除。
   * **data_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：输入的布局。
   * **kernel_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) *：* 权重的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) *：* 输出的布局。如果未指定，则与 data_layout 相同
   * **out_dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***DataType****]]*) - 指定混合精度 conv2d 的输出数据类型。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.conv2d_transpose(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *weight:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = (0, 0)*, *output_padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (0, 0)*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCHW'*, *kernel_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'IOHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


二维转置卷积运算符。


这个运算符设计为 conv2d 的梯度运算符。这意味着，如果


out = conv2d(data, weight, strides, padding, dilation),


那么相对于 data 的梯度可以按如下方式计算：


data_grad = conv2d_transpose(out_grad, weight, strides, padding, output_padding, dilation)，output_padding 是一个用于确定输出形状的参数。


当 data_layout == “NCHW”且 kernel_layout == “IOHW”时，输出形状可以简单解释。假设数据形状为(N, in_channel, in_h, in_w)，权重形状为(in_channel, out_channel, weight_h, weight_w)，我们需要确保 in_channel % groups == 0。输出的形状将是(N, out_channel 运算符* groups, out_h, out_w)，其中
* out_h = ((in_h - 1) 运算符* strides[0] + weight_h - 2 运算符* padding[0] + output_padding[0])
* out_w = ((in_w - 1) 运算符* strides[1] + weight_w - 2 运算符* padding[1] + output_padding[1])
* **参数：**
   * **data** (*relax.Expr*) **：** 运算符的输入数据。
   * **weight** (*relax.Expr*) ：权重表达式。
   * **strides** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：卷积的步长。必须具有长度为 1 或 2。
   * **padding** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]]*) **：** 卷积前输入两边的填充。必须具有长度为 1、2 或 4。
   * **output_padding** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]**]****,optional*) ：用于区分输出形状。
   * **dilation** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：指定用于扩张卷积的扩张率。必须具有长度为 1 或 2。
   * **groups** ([int](https://docs.python.org/3/library/functions.html#int)) ：分组卷积将输入分成多少组。输入和输出通道数应该能被组数整除。
   * **data_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) *：* 输入的布局。
   * **kernel_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) *：* 权重的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) *：* 输出的布局。如果未指定，则与 data_layout 相同
   * **out_dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***DataType****]]*) *：* 指定混合精度 conv2d 的输出数据类型。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.conv3d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *weight:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1, 1)*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = (0, 0, 0)*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1, 1)*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCDHW'*, *kernel_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'OIDHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


3D 卷积。


该运算符将权重作为卷积核，将其与数据卷积以产生输出。


在默认情况下，当 data_layout 为 NCDHW 且 kernel_layout 为 OIDHW 时，conv3d 接收一个形状为(batch_size, in_channels, depth, height, width)的数据张量和一个形状为(channels, in_channels, kernel_d, kernel_h, kernel_w)的权重张量，其中 kernel_d、kernel_h 和 kernel_w 是 D、H 和 W 卷积核维度的长度，以产生一个输出张量，其规则如下：

$$
\operatorname{out}[b, c, z, y, x] =
\sum_{dz, dy, dx, k} 
\operatorname{data}[b, k, 
\operatorname{strides}[0] \cdot z + dz, 
\operatorname{strides}[1] \cdot y + dy, 
\operatorname{strides}[2] \cdot x + dx] 
\cdot \operatorname{weight}[c, k, dz, dy, dx]
$$


在计算之前，数据会应用填充，权重会应用膨胀。该运算符接受数据布局规范。从语义上讲，运算符会将布局转换为规范布局（数据为 NCDHW，权重为 OIDHW），执行计算，然后转换为 out_layout。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **weight** (*relax.Expr*) ：权重表达式。
   * **strides** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：卷积的步长。必须具有长度为 1 或 3。
   * **padding** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]]*) ：卷积前输入两边的填充。必须具有长度为 1、3 或 6。
   * **dilation** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) **：** 指定用于扩张卷积的扩张率。必须具有长度为 1 或 3。
   * **groups** ([int](https://docs.python.org/3/library/functions.html#int)) ：分组卷积将输入分成多少组。输入和输出通道数应该能被组数整除。
   * **data_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) *：* 输入的布局。
   * **kernel_layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：权重的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：输出的布局。如果未指定，则与 data_layout 相同
   * **out_dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***DataType****]]*) ：指定混合精度 conv2d 的输出数据类型。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.cross_entropy_with_logits(*predictions:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *labels:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


predictions 和 labels 之间的 logits 交叉熵。


predictions 和 labels 的形状必须相同。当 ndim >= 2 时，第一个维度被视为批次大小 N。在这种情况下，计算结果将除以 N 以进行均值归约。

$$\text{cross\_entropy\_with\_logits}(x_i, y_i) = \frac{\sum_i -x_i \cdot y_i}{N}$$
* **参数：**
   * **predictions** (*relax.Expr*) ：预测值。
   * **labels** (*relax.Expr*) ：标签（真实值）。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.dropout(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *rate:*[float](https://docs.python.org/3/library/functions.html#float)*= 0.5*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


对输入张量应用 dropout 操作。


在训练期间，输入的每个元素以概率 `p` 被设置为 0。整个数组被缩放为 `1/(1-p)` 以保持输入的期望和不变。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **rate** ([float](https://docs.python.org/3/library/functions.html#float)) ：元素重置为 0 的概率。
* **返回：result** ：dropout 的结果，是一个包含两个张量的元组。第一个是原始张量，第二个是掩码张量（未丢弃元素处为 1.0，丢弃处为 0.0）。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.gelu(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


高斯误差线性单元函数

$$\text{GeLU}(x) = 0.5 * x * (1 + \text{erf}(x * 0.5**0.5))$$

其中 erf 是高斯误差函数。
* **参数：**
   * **data** (*relax.Expr*) *：*输入数据。
* **返回：result** -计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.nn.gelu_tanh(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

高斯误差线性单元函数，使用 tanh 近似

$$\text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))$$
* **参数：**
   * **data** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.nn.group_norm(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *gamma:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *beta:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *num_groups:*[int](https://docs.python.org/3/library/functions.html#int), *channel_axis:*[int](https://docs.python.org/3/library/functions.html#int), *axes:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *epsilon:*[float](https://docs.python.org/3/library/functions.html#float)*= 1e-05*, *center:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *scale:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


分组归一化（Yuxin Wu 等人，2016 年）。将分组归一化应用于 n 维输入数组。该运算符接收一个 n 维输入数组。首先沿通道轴将输入数组分成多个组。然后对每个组应用层归一化。
* **参数：**
   * **data** (*relax.Expr*) ：将 group_norm 应用的输入。
   * **gamma** (*relax.Expr*) ：gamma 缩放因子。
   * **beta** (*relax.Expr*) *：* beta 偏移因子。
   * **num_groups** ([int](https://docs.python.org/3/library/functions.html#int)) *：* 将通道分离成的组数。
   * **channel_axis** ([int](https://docs.python.org/3/library/functions.html#int)) *：* 输入数据中通道轴的索引。
   * **axes** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,***List***[*[int](https://docs.python.org/3/library/functions.html#int)*]]) *：* 正规化应用的轴（不包括组轴）。
   * **epsilon** ([float](https://docs.python.org/3/library/functions.html#float)) -添加到方差中的小浮点数，以避免除以零。
   * **center** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：指示是否将 beta 偏移添加到归一化张量中。
   * **scale** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：指示是否将伽马缩放相乘。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.instance_norm(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *gamma:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *beta:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *channel_axis:*[int](https://docs.python.org/3/library/functions.html#int), *axes:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *epsilon:*[float](https://docs.python.org/3/library/functions.html#float)*= 1e-05*, *center:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *scale:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


**参数：**
   * **data** (*relax.Expr*) *：* 应用 instance_norm 的输入。
   * **gamma** (*relax.Expr*) *：* gamma 缩放因子。
   * **beta** (*relax.Expr*) ：beta 偏移因子。
   * **axes** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***List****[*[int](https://docs.python.org/3/library/functions.html#int)*]]*) -正在应用归一化的轴。
   * **epsilon** ([float](https://docs.python.org/3/library/functions.html#float)) ：添加到方差中的小浮点数，以避免除以零。
   * **center** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：指示是否将 beta 偏移添加到归一化张量中。
   * **scale** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：指示是否将伽马缩放相乘。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.layer_norm(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *gamma:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *beta:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axes:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *epsilon:*[float](https://docs.python.org/3/library/functions.html#float)*= 1e-05*, *center:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *scale:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

层归一化（Lei Ba 等人，2016 年）。将层归一化应用于 n 维输入数组。该运算符接收一个 n 维输入数组，并使用指定的轴对输入进行归一化：

$$out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis)+\epsilon}} * gamma + beta$$

与批量归一化不同，均值和方差是沿着通道维度计算的。


假设输入在轴 1 上的大小为 k，那么 gamma 和 beta 的形状都是(k,)。

:::note

这个运算符可以在推理时被优化掉。

:::
* **参数：**
   * **data** (*relax.Expr*) ：将 layer_norm 应用于的输入。
   * **gamma** (*relax.Expr*) *：* gamma 缩放因子。
   * **beta** (*relax.Expr*) ：beta 偏移因子。
   * **axes** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***List****[*[int](https://docs.python.org/3/library/functions.html#int)*]]*) ：沿着这些轴应用归一化。
   * **epsilon** ([float](https://docs.python.org/3/library/functions.html#float)) ：添加到方差中的小浮点数，以避免除以零。
   * **center** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：指示是否将 beta 偏移添加到归一化张量中。
   * **scale** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：指示是否将伽马缩放相乘。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.leakyrelu(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *alpha:*[float](https://docs.python.org/3/library/functions.html#float)*= 0.01*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


修正线性单元。

$$text{LeakyReLU, negative_slope}(x) = max(x, 0) + negative_slope * min(x, 0)$$
* **参数：**
   * **data** (*relax.Expr*) ：输入数据。
   * **alpha** ([float](https://docs.python.org/3/library/functions.html#float)) ：控制负斜率的角，用于负输入。默认值为 0.01
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.log_softmax(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


计算 log softmax。

$$\text{log\_softmax}(x_i) = \log\left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}\right)$$

这个运算符可以在推理时被优化掉。
* **参数：**
   * **data** (*relax.Expr*) *：* 运算符的输入数据。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) *：* 计算 log softmax 时沿其进行求和的轴。如果未指定，默认为输入张量的最后一个轴。支持负索引。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.max_pool1d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *pool_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1,)*, *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1,)*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = (0, 0)*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1,)*, *ceil_mode:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *count_include_pad:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

1D 最大池化运算符。


此运算符以数据为输入，通过 stride 定义的步长，在 pool_size 大小的窗口内进行 1D 最大值计算。


在默认情况下，当 data_layout 为 NCW 时，一个形状为(batch_size, channels, width)的数据张量将产生一个输出张量。


ceil_mode 用于在计算输出形状时取上整或下整。count_include_pad 指示是否在计算中包含或排除填充的输入值。该运算符接受数据布局规范。
* **参数：**
   * **data** (*relax.Expr*) *：* 运算符的输入数据。
   * **pool_size** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) **：** 池化窗口的大小。它必须具有长度为 1。
   * **strides** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：池化的步长。它必须具有长度为 1。
   * **padding** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]]*) -池化的填充。它必须具有长度为 1 或 2。
   * **dilation** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) - 池化的膨胀率。必须为长度为 1。
   * **ceil_mode** ([bool](https://docs.python.org/3/library/functions.html#bool)) - 一个布尔值，指示是否使用 ceil 或 floor 来计算输出形状。使用 ceil 时，输入张量的每个元素都将被滑动窗口覆盖。
   * **count_include_pad** ([bool](https://docs.python.org/3/library/functions.html#bool)*,optional*) *：* 是否将填充包括在内以计算平均值。
   * **layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) *：*输入的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：输出的布局。如果未指定，则与 data_layout 相同
* **返回：result** ：计算结果。
* **返回类型：** Expr。

## tvm.relax.op.nn.max_pool2d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *pool_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = (0, 0)*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *ceil_mode:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *count_include_pad:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


2D 最大池化运算符。


该运算符接收数据作为输入，并通过 stride 定义的步长，在 pool_size 大小的窗口内进行 2D 最大值计算。


在默认情况下，当数据布局为 NCHW 时，一个形状为(batch_size, in_channels, height, width)的数据张量，将按照以下规则产生输出张量：


形状为(b, c, h, w)的数据和池化大小(kh, kw)
$$
\operatorname{out}(b, c, y, x) =
\max_{m=0, \dots, kh-1} \max_{n=0, \dots, kw-1} 
\operatorname{data}(b, c, \operatorname{stride}[0] \cdot y + m, \operatorname{stride}[1] \cdot x + n)
$$


在计算之前对数据进行填充。ceil_mode 用于在计算输出形状时取上整或下整。该运算符接受数据布局规范。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **pool_size** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：池化窗口的大小。必须具有长度为 1 或 2。
   * **strides** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：池化的步长。必须具有长度为 1 或 2。
   * **padding** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]]*) *：* 池化操作的填充。必须具有长度为 1、2 或 4。
   * **dilation** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) *：* 池化操作的膨胀率。必须具有长度为 1 或 2。
   * **ceil_mode** ([bool](https://docs.python.org/3/library/functions.html#bool)) *：* 一个布尔值，指示是否使用 ceil 或 floor 来计算输出形状。使用 ceil 时，输入张量的每个元素都将被滑动窗口覆盖。
   * **count_include_pad** ([bool](https://docs.python.org/3/library/functions.html#bool)*,optional*) - 是否将填充包括在内以计算平均值。
   * **layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：输入的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) -输出的布局。如果未指定，则与 data_layout 相同。
* **返回：result** ：计算结果。
* **返回类型：** Expr。

## tvm.relax.op.nn.max_pool3d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *pool_size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1, 1)*, *strides:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1, 1)*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] = (0, 0, 0)*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1, 1)*, *ceil_mode:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *count_include_pad:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCDHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


3D 最大池化运算符。


该运算符接收数据作为输入，并通过 stride 定义的步长，使用 pool_size 大小的窗口进行 3D 最大值计算。


在默认情况下，当 data_layout 为 NCDHW 时，一个形状为(batch_size, channels, depth, height, width)的数据张量，将产生一个输出张量。


ceil_mode 用于在计算输出形状时取上整或下整。count_include_pad 指示是否在计算中包含或排除填充的输入值。该运算符接受数据布局规范。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **pool_size** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：池化窗口的大小。必须具有 1 或 3 的长度。
   * **strides** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) ：池化的步长。必须具有长度为 1 或 3。
   * **padding** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***…****]]*) ：池化操作的填充。长度必须是 1、3 或 6。
   * **dilation** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[int](https://docs.python.org/3/library/functions.html#int)***,*** [int](https://docs.python.org/3/library/functions.html#int)***]****]*) *：* 池化操作的膨胀率。必须为长度为 1 或 3 的值。
   * **ceil_mode** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：一个布尔值，指示是否使用 ceil 或 floor 来计算输出形状。使用 ceil 时，输入张量的每个元素都将被滑动窗口覆盖。
   * **count_include_pad** ([bool](https://docs.python.org/3/library/functions.html#bool)*,optional*) ：是否将填充包括在内以计算平均值。
   * **layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：输入的布局。
   * **out_layout** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：输出的布局。如果未指定，则与 data_layout 相同
* **返回：result** ：计算结果。
* **返回类型：** Expr。

## tvm.relax.op.nn.nll_loss(*predictions:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *targets:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *weights:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *reduction:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'mean'*, *ignore_index:*[int](https://docs.python.org/3/library/functions.html#int)*= -100*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

负对数似然损失。

output[n, i*1, i*2, …, i_k] = -p * w, where - p = predictions[n, t, i*1, i*2, i_k], - t = targets[n, i*1, i*2, …, i_k], - w = weights[t] if t != ignore_index else 0。



result = reduction(output)。
* **参数：**
   * **predictions** (*relax.Expr*) *：* 维张量，形状为 (N, C, d_1, d_2, …, d_k)，其中 C 是目标类别的数量。
   * **targets** (*relax.Expr*) *：* 维张量，形状为 (N, d_1, d_2, …, d_k)。必须是 int 数据类型。
   * **weights** (*Optional**[****relax.Expr]*) ：D 张量，形状为 (C,)。如果未指定，则被视为所有元素都为 1。
   * **reduction** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：应用于输出的归约方法。可能的值有“mean”、“sum”和“none”。
   * **ignore_index** ([int](https://docs.python.org/3/library/functions.html#int)) ：要忽略的目标值。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.pad(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *pad_width:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …]*, *pad_mode:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 'constant'*, *pad_value:*[float](https://docs.python.org/3/library/functions.html#float)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 0.0*)


这个运算符接收一个张量，并使用指定的值按照指定的宽度对每个轴进行填充。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据
   * **pad_width** (*Union**[****List**[***[int](https://docs.python.org/3/library/functions.html#int)***]****,*[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*[*[int](https://docs.python.org/3/library/functions.html#int)*,…**]****],required*) ：每个轴边缘填充的值数量，格式为 ((before_1, after_1), …, (before_N, after_N))
   * **pad_mode** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：‘constant’, ‘reflect’, ‘replicate’, ‘circular’ ‘constant’ 使用常数值填充 pad_value ‘reflect’ 通过镜像值（不包括边缘）进行填充 ‘replicate’ 通过重复边缘值进行填充。 ‘circular’ 通过从另一侧循环值进行填充。默认为 ‘constant’
   * **pad_value** (*Optional**[****Union**[***[float](https://docs.python.org/3/library/functions.html#float)***,*** ***Expr****]]*) ：用于填充的值。默认为 0。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.pixel_shuffle(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *upscale_factor:*[int](https://docs.python.org/3/library/functions.html#int))


像素重排运算符。


该运算符对输入张量执行像素重排操作，常用于图像超分辨率任务中的高效亚像素卷积。它将形状为 (N, C × r^2, H, W) 的张量中的元素重新排列为形状为 (N, C, H × r, W × r) 的张量，其中 r 是放大因子。
* **参数：**
   * **data** (*relax.Expr*) ：像素重排运算符的输入张量。它必须有 4 个维度，格式为 (N, C 运算符* r^2, H, W)，其中 r 是放大因子。
   * **upscale_factor** ([int](https://docs.python.org/3/library/functions.html#int)) ：放大因子 r。它决定了输入张量的空间分辨率（高度和宽度）增加的量。
* **返回：result** **：** 变换后的张量，形状为 (N, C, H 运算符* r, W 运算符* r)。
* **返回类型：** relax.Expr。


**示例**


如果输入张量的形状为 (1, 8, 10, 15) 且上采样因子为 2，则结果张量的形状将为 (1, 2, 20, 30)。

## tvm.relax.op.nn.prelu(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *alpha:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


参数化整流线性单元（PReLU）。

$$PReLU(x) = x \text{ if } x > 0 \text{ else } \alpha * x$$
* **参数：**
   * **data** (*relax.Expr*) ：输入张量。
   * **alpha** (*relax.Expr*) ：可学习的斜率张量，按通道应用。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) -alpha 值应用的轴。默认为 1（假设为 NCHW 格式）。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.relu(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


修正线性单元。

$$\text{ReLU}(x) = \max(x, 0)$$
* **参数：**
   * **data** (*relax.Expr*) **：** 输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.relu6(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


ReLU6 激活函数。

$$\text{ReLU6}(x) = \min(\max(x, 0), 6)$$
* **参数：**
   * **data** (*relax.Expr*) ：输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.rms_norm(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *weight:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axes:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] = -1*, *epsilon:*[float](https://docs.python.org/3/library/functions.html#float)*= 1e-05*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


均方根归一化（Biao Zhang 等，2019）。对 n 维输入数组应用均方根归一化。该运算符接收一个 n 维输入数组，并使用指定轴对输入进行归一化：

$$out = \frac{data}{\sqrt{mean(data, axis)+\epsilon}} * weight + bias$$
* **参数：**
   * **data** (*relax.Expr*) **：** rms_norm 将要应用到的输入。
   * **weight** (*relax.Expr*) ：缩放因子。
   * **bias** (*relax.Expr*) ：偏移因子。
   * **axes** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***List****[*[int](https://docs.python.org/3/library/functions.html#int)*]]*) - 沿着这些轴应用归一化。
   * **epsilon** ([float](https://docs.python.org/3/library/functions.html#float)) ：添加到平方均值的小浮点数，以避免除以零。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.nn.selu(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


缩放指数线性单元（SELU）。

$$\begin{split}\text{SELU}(x) = \lambda \begin{cases}     x & \text{if } x > 0 \     \alpha (e^x - 1) & \text{if } x \leq 0 \end{cases}\end{split}$$

其中 λ≈1.0507 和 α≈1.6733 。
* **参数：**
   * **data** (*relax.Expr*) ：输入数据。
* **返回：result** *：* 计算结果。
* **返回类型：** relax.Expr

## tvm.relax.op.nn.silu(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


Sigmoid 线性单元函数。

$$\text{SiLU}(x) = x * \text{sigmoid}(x)$$
* **参数：**
   * **data** (*relax.Expr*) *：* 输入数据。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。

:::note

输入张量必须具有 float dtype。

:::

## tvm.relax.op.nn.softmax(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

计算 softmax。

$$\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) *：* 计算 softmax 时沿其求和的轴。如果未指定，默认为输入张量的最后一个轴。支持负索引。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。


输入张量必须具有 float dtype。


## tvm.relax.op.nn.softplus(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *beta:*[float](https://docs.python.org/3/library/functions.html#float)*= 1.0*, *threshold:*[float](https://docs.python.org/3/library/functions.html#float)*= 20.0*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


Softplus 激活函数。

$$\text{Softplus}(x) = \frac{1}{\beta} \log(1 + e^{\beta x})$$
* **参数：**
   * **data** (*relax.Expr*) **：** 输入数据。
   * **beta** ([float](https://docs.python.org/3/library/functions.html#float)*,optional*) ：控制过渡的平滑度。默认值为 1.0。
   * **threshold** ([float](https://docs.python.org/3/library/functions.html#float)*,optional*) ：超过该值后，函数近似为线性以避免数值不稳定性。默认值为 20.0。
* **返回：result** ：计算结果。
* **返回类型：** relax.Expr。


# tvm.relax.op.builtin


Relax 内置运算符。

## tvm.relax.op.builtin.alloc_tensor(*shape:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *runtime_device_index:*[int](https://docs.python.org/3/library/functions.html#int)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *storage_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*= 'global'*) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)


构建一个 relax.Call 来分配具有特定形状、dtype 和 runtime_device_index 的张量。
* **参数：**
   * **shape** (*Expr*) **：** 要分配的张量的形状。
   * **dtype** (*Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Expr****]*) ：要分配的张量的数据类型。
   * **runtime_device_index** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***Expr****]*) ：1 保留用于主机设备。
   * **storage_scope** (*Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Expr****]*) ：指示分配存储的范围。
* **返回：result** ：一个 relax relax.Call，它获取已分配的张量。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.builtin.stop_lift_params(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


一个指示输入张量的消费者不应被提升到 transform_params 函数的标志。
* **参数：**
   * **x** (*relax.Expr*) ：输入数据。
* **返回：result** **：** 结果张量，与输入张量相同。
* **返回类型：** relax.Expr。

## 

与 CCL 相关的运算符。

## tvm.relax.op.ccl.allgather(*x*, *num_workers:*[int](https://docs.python.org/3/library/functions.html#int), *in_group:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*)

AllGather 运算符。
* **参数：**
   * **x** (*relax.Expr*) *：*输入张量。
   * **num_worker** ([int](https://docs.python.org/3/library/functions.html#int)) ：用于收集数据的工人数。
   * **in_group** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：指示 gather 操作是全局执行还是默认在组内执行。
* **返回：result** ：allgather 的结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.ccl.allreduce(*x*, *op_type:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'sum'*, *in_group:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*)

Allreduce 运算符。
* **参数：**
   * **x** (*relax.Expr*) ：输入张量。
   * **op_type** ([str](https://docs.python.org/3/library/stdtypes.html#str)) **：** 应用于输入数据的归约操作类型。目前支持「sum」、「prod」、「min」、「max」和「avg」。
   * **in_group** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：归约操作是否默认全局执行或在组内执行。
* **返回：result** ：allreduce 的所有结果。
* **返回类型：** relax.Expr。

## tvm.relax.op.ccl.broadcast_from_worker0(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


从 worker-0 广播数据到所有其他 worker。
* **参数：x** (*relax.Expr*) *：* 要广播的张量。
* **返回：result** *：* 已广播到所有其他 worker 的相同张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.ccl.scatter_from_worker0(*x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *num_workers:*[int](https://docs.python.org/3/library/functions.html#int), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


执行从 worker-0 开始的 scatter 操作，将给定的 buffer 分成相等的部分。
* **参数：**
   * **x** (*relax.Expr*) ：需要分成相等部分并相应发送给每个 worker 的 buffer。
   * **num_worker** ([int](https://docs.python.org/3/library/functions.html#int)) *：* 工作人员数量，即给定的 buffer 应被分成多少部分。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：要 scatter 的张量的维度。默认为 0。
* **返回：result** *：* 由不同工作进程接收的 Chunked Tensor。
* **返回类型：** relax.Expr。

## 

# tvm.relax.op.distributed


用于分布式 Relax 的运算符。

## tvm.relax.op.distributed.annotate_sharding(*input:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *device_mesh: DeviceMesh*, *placement: Placement*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


为张量标注分片计划。
* **参数：**
   * **input** (*relax.Expr*) *：* 输入张量。
   * **device_mesh** (*DeviceMesh*) - 分片计划的设备网格。
   * **placement** (*Placement*) ：分片计划的放置。
* **返回：result** ：未修改的张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.distributed.redistribute(*input:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *device_mesh: DeviceMesh*, *placement: Placement*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


重分布张量。
* **参数：**
   * **input** (*relax.Expr*) *：* 输入张量。
   * **device_mesh** (*DeviceMesh*) ：重分布后的设备网格。
   * **placement** (*Placement*) ：重分布后的放置。
* **返回：result** ：重分布后的张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.distributed.call_tir_local_view(*gvar:*[GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none), *args:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *out_sinfo: DTensorStructInfo |*[List](https://docs.python.org/3/library/typing.html#typing.List)*[DTensorStructInfo]*, *tir_vars:*[ShapeExpr](/docs/api-reference/python-api/tvm-relax#classtvmrelaxshapeexprvalueslistprimexprtupleprimexprarrayspanspannonenone)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)

relax.Call 一个 tir.prim_func 并返回输出。该 prim_func 应该是一个工作本地函数，实际上在每个工作节点上执行，而不是未分割的函数。这个运算符的输出是 DTensor 或 DTensor 的元组。
* **参数：**
   * **gvar** ([GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none)) ：指向 tir PrimFunc 的 GlobalVar。
   * **args** (*Expr*) ：输入参数。
   * **out_sinfo** (*Union**[****DTensorStructInfo**,*** ***List****[**DTensorStructInfo****]]*) ：调用_tir 的输出结构信息。它应该是一个 DTensorStructInfo 或一个 DTensorStructInfo 的列表。每一个表示一个返回张量的结构信息。
   * **tir_vars** (*Optional**[****Union**[***[ShapeExpr](/docs/api-reference/python-api/tvm-relax#classtvmrelaxshapeexprvalueslistprimexprtupleprimexprarrayspanspannonenone)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)***]****,List**[***[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)***]****]]*) ：表示调用 func 时需要解包的整数元组的 ShapeExpr。如果未使用则为 null。
* **返回：ret** *：* call_tir_local_view 运算符的调用节点。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.distributed.redistribute_replica_to_shard(*input:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *num_workers:*[int](https://docs.python.org/3/library/functions.html#int), *axis:*[int](https://docs.python.org/3/library/functions.html#int)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

沿一个轴将张量切片成多个部分，


每个工作节点获取一个部分。input.struct_info.shape[axis] % num_workers == 0 是必需的。每个工作节点必须拥有输入的完全相同副本。这是 redistribute 操作的专用版本。
* **参数：**
   * **input** (*relax.Expr*) ：要被分割成等份的缓冲区。
   * **num_worker** ([int](https://docs.python.org/3/library/functions.html#int)) ：工作线程的数量，即给定缓冲区应被分割成的份数。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：要分割的张量的轴。
* **返回：result** **：** 每个设备上保留的分割张量。
* **返回类型：** relax.Expr。

# tvm.relax.op.grad


用于查找 relax 运算符梯度的运算符。

## tvm.relax.op.grad.Expr

`RelaxExpr` 的别名。

## tvm.relax.op.grad.avg_pool2d_backward(*output_grad:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *pool_size:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *strides:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *padding:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (0, 0, 0, 0)*, *dilation:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *ceil_mode:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *count_include_pad:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

relax.nn.avg_pool2d 的反向运算符。除了 output_grad 之外的所有参数与relax.nn.avg_pool2d 相同。返回关于 data 的梯度。
* **参数：output_grad** (*relax.Expr*) *：* 对 avg_pool2d 结果的梯度。
* **返回：result** ：对数据的梯度。
* **返回类型：** relax.Expr。

## tvm.relax.op.grad.end_checkpoint(*input:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


标记检查点阶段的结束。参见 tvm.relax.op.grad.start_checkpoint。
* **参数：input** (*relax.Expr*) ：检查点阶段的输出。
* **返回：result** ：与输入相同的张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.grad.max_pool2d_backward(*output_grad:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *pool_size:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *strides:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *padding:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (0, 0, 0, 0)*, *dilation:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*,*[int](https://docs.python.org/3/library/functions.html#int)*] = (1, 1)*, *ceil_mode:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *count_include_pad:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCHW'*, *out_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


relax.nn.max_pool2d 的反向运算符。除了 output_grad 之外的所有参数与 。relax.nn.max_pool2d 相同。返回相对于数据的梯度。
* **参数：output_grad** (*relax.Expr*) ：对 max_pool2d 结果的梯度。
* **返回：result** *：* 对数据的梯度。
* **返回类型：** relax.Expr。

## tvm.relax.op.grad.nll_loss_backward(*output_grad:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *predictions:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *targets:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *weights:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *reduction:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'mean'*, *ignore_index:*[int](https://docs.python.org/3/library/functions.html#int)*= -100*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

relax.nn.nll_loss 的反向运算符。除了 output_grad 外，所有参数与 relax.nn.nll_loss 相同。返回对预测的梯度。
* **参数：output_grad** (*relax.Expr*) ：对 nll_loss 结果的梯度。
* **返回：result**：对预测的梯度。
* **返回类型：** relax.Expr。

## tvm.relax.op.grad.no_grad(*input:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

对输入无梯度的虚拟运算符。
* **参数：input** (*relax.Expr*) ：对应的输入张量。
* **返回：result** ：相对于输入的无梯度表示。
* **返回类型：** relax.Expr。

## tvm.relax.op.grad.start_checkpoint(*input:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)

标记检查点阶段的开始。start_checkpoint 和 end_checkpoint 之间的计算将被标记为检查点阶段。


与其在反向传播时存储整个计算图的所有中间激活值，检查点阶段不会保存中间激活值，而是在反向过程中重新计算它们。


例如， `` a = relax.Var("a", relax.TensorStructInfo((2, 2), "float32")) b = relax.Var("b", relax.TensorStructInfo((2, 2), "float32")) c = a * 2 d = b * 2 c_cp = start_checkpoint(c) d_cp = start_checkpoint(d) e = c_cp + d_cp e_out = end_checkpoint(e) `` 然后 e 将在反向阶段重新计算。


参见 tvm.relax.transform.Gradient、tvm.relax.testing.nn.checkpoint、tvm.relax.op.grad.end_checkpoint 获取更多信息。
* **参数：input** (*relax.Expr*) *：* 标记检查点阶段输入的张量。
* **返回：result** *：* 与输入相同的张量。
* **返回类型：** relax.Expr。

## tvm.relax.op.grad.take_backward(*output_grad:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *x:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *indices:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


relax.take 的反向运算符。除了 output_grad 之外的所有参数与 relax.take 相同。返回关于 x 的梯度。
* **参数：output_grad** (*relax.Expr*) ：对 take 结果的梯度。
* **返回：result** ：对 x 的梯度。
* **返回类型：** relax.Expr。


##

# tvm.relax.op.image


图像运算符。

## tvm.relax.op.image.resize2d(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *size:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *roi:*[float](https://docs.python.org/3/library/functions.html#float)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[float](https://docs.python.org/3/library/functions.html#float)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCHW'*, *method:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'linear'*, *coordinate_transformation_mode:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'half_pixel'*, *rounding_method:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'round'*, *cubic_alpha:*[float](https://docs.python.org/3/library/functions.html#float)*= -0.75*, *cubic_exclude:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *extrapolation_value:*[float](https://docs.python.org/3/library/functions.html#float)*= 0.0*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*| dtype |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


图像 resize2d 运算符。


该运算符接收数据作为输入，并按照给定的缩放因子进行 2D 缩放。在默认情况下，如果 data_layout 为 NCHW 且数据形状为(n, c, h, w)，则输出将具有形状(n, c, size[0], size[1])。


method 指示计算输出值时使用的算法，method 可以是("linear", "nearest_neighbor", "cubic")之一。
* **参数：**
   * **data** (*relax.Expr*) ：运算符的输入数据。
   * **size** (*Union**[****Expr**,*** ***PrimExprLike****,*[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*[**PrimExprLike****]]*) ：图像将被调整到的输出尺寸。如果指定为列表，其长度必须是 1 或 2。如果指定为 Expr，其必须具有维度 2。
   * **roi** (*Optional**[****Union**[***[float](https://docs.python.org/3/library/functions.html#float)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***[***[float](https://docs.python.org/3/library/functions.html#float)***]****]]*) ：用于裁剪输入图像的区域。预期大小为 4，格式为[start_h, start_w, end_h, end_w]。仅在使用 coordinate_transformation_mode 为 tf_crop_and_resize 时使用。
   * **layout** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：输入的布局。
   * **method** ([str](https://docs.python.org/3/library/stdtypes.html#str)) *：* 使用的缩放方法 [最近邻, 线性, 三次插值]。
   * **coordinate_transformation_mode** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：描述如何将调整大小后的张量中的坐标转换为原始张量中的坐标。定义可以在 topi/image/resize.py 中找到。[半像素, 对齐角点, 非对称, pytorch_half_pixel, tf_half_pixel_for_nn, 和 tf_crop_and_resize]。
   * **rounding_method** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：指示在最近邻方法中如何找到“最近”的像素 [四舍五入, 向下取整, 向上取整
   * **cubic_alpha** ([float](https://docs.python.org/3/library/functions.html#float)) ***：*** 双三次插值的样条系数。
   * **cubic_exclude** ([int](https://docs.python.org/3/library/functions.html#int)) ：标志位，用于在双三次插值时排除图像外部区域。
   * **extrapolation_value** ([float](https://docs.python.org/3/library/functions.html#float)) ：当 roi 位于图像外部时使用的填充值。
   * **out_dtype** (*Optional**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***DataType****]]*) -输出张量的数据类型。如果未指定，输出将具有与输入相同的数据类型（如果未指定）。
* **返回：result** ：调整大小后的结果。
* **返回类型：** relax.Expr。

## 

# tvm.relax.op.memory

Relax 内存原语。

## tvm.relax.op.memory.alloc_storage(*size:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *virtual_device_index:*[int](https://docs.python.org/3/library/functions.html#int)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *storage_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)


构造一个 relax.Call 来分配具有特定大小、虚拟设备索引、存储范围和 dtype 的存储。
* **参数：**
   * **size** (*Expr*) ：要分配的存储的大小。
   * **virtual_device_index** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***Expr****]*) ：1 保留用于主机设备。
   * **storage_scope** (*Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Expr****]*) ：指示分配存储的范围。
   * **dtype** (*Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Expr****]*) ：要分配的存储的数据类型。
* **返回：result** ：一个 relax relax.Call，它获取已分配的存储。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.memory.alloc_tensor(*storage:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *offset:*[int](https://docs.python.org/3/library/functions.html#int)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *shape:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone) 


构建一个 relax.Call 来在指定的存储上从给定偏移量开始分配一个张量。
* **参数：**
   * **storage** (*Expr*) -分配张量的存储。
   * **offset** (*Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,*** ***Expr****]*) *：* 分配张量的存储偏移量。
   * **shape** (*Expr*) -要分配的张量的形状。
   * **dtype** (*Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Expr****]*) ：要分配的张量的数据类型。
* **返回：result** ：一个 relax relax.Call，它获取已分配的张量。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.memory.kill_storage(*storage:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)


构建一个 relax.Call 来销毁一个存储。
* **参数：storage** (*Expr*) ：要杀死的存储。
* **返回：result** ：一个用于杀死存储的 relax relax.Call。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.memory.kill_tensor(*tensor:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)


构建一个 relax.Call 来销毁一个张量。
* **参数：tensor** (*Expr*) ：要销毁的张量。
* **返回：result** ：一个用于销毁张量的 relax relax.Call。
* **返回类型：**[relax.Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)。

## tvm.relax.op.memory.view(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *dtype:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *relative_byte_offset:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


提供一个现有张量的视图。


视图的形状可能不同，数据类型可能不同，并且可能相对于源数组从偏移量开始。


无论使用这些选项的哪种组合，视图都不会访问通过输入数据数组无法访问的内存。即使数据数组本身是共享后备数组的视图，此限制也适用。
* **参数：**
   * **data** (*relax.Expr*) **：**运算符的输入数据。
   * **shape** (*Optional**[****Union**[****Sequence**[****PrimExprLike**]****,Expr**]****]*) ：目标形状。应为 relax.ShapeExpr，或可转换为 relax.ShapeExpr 的集合。
   * **dtype** (*Optional**[****Expr]*) ：目标数据类型。应为 relax.ShapeExpr，或可转换为 relax.ShapeExpr 的集合。
   * **relative_byte_offset** (*Optional**[****Expr]*) ：输出 NDArray 的偏移量，相对于数据的字节偏移量。如果为 None，视图的偏移量与数据的偏移量相同。
* **返回：result** ：张量视图。
* **返回类型：** relax.Expr。

## tvm.relax.op.memory.ensure_zero_offset(*data:*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


确保张量具有 elem_offset == 0。如有必要，将进行复制。
* **参数：**
   * **data** (*relax.Expr*) ***：***输入张量。
   * **Results。**
   * **-------**
   * **result** (*relax.Expr*) *：*元素偏移量等于 0 的张量。

## 

# tvm.relax.op.op_attrs


用于 Relax 操作的属性节点。

## *class* tvm.relax.op.op_attrs.CallTIRWithGradAttrs 


call_tir_with_grad 操作使用的属性。

### ***property*te_grad_kwargs**

传递给 te gradient 函数的关键字参数。

### ***property*te_grad_name**

与这个 call_tir_with_grad 节点关联的 te 梯度函数的名称。

## *class* tvm.relax.op.op_attrs.InitAttrs


用于 full/full_like、ones/ones_like 和 zeros/zeros_like 运算符的属性。

### ***property*dtype**

创建的张量的数据类型。

## *class* tvm.relax.op.op_attrs.TriluAttrs


tril 和 triu 运算符中使用的属性。

### ***property*k**

主对角线上方或下方要排除或包含的对角线数量。

## *class* tvm.relax.op.op_attrs.AstypeAttrs


astype 运算符中使用的属性。

### ***property*dtype**

  目标数据类型。

## *class* tvm.relax.op.op_attrs.TakeAttrs


take 运算符中使用的属性。

### ***property*axis**

选择值的轴。

### ***property*mode**

处理越界索引的模式。

## *class* tvm.relax.op.op_attrs.StridedSliceAttrs


strided_slice 运算符使用的属性。

### ***property*assume_inbound**

是否假设索引在有效范围内。如果设置为 false，超出范围的索引将被裁剪到边界。

## *class* tvm.relax.op.op_attrs.MatmulAttrs


matmul 运算符的属性。

### ***property*out_dtype**

输出张量的数据类型。

## *class* tvm.relax.op.op_attrs.Conv2DAttrs


nn.conv2d 的属性

### ***property*data_layout**

输入数据的维度顺序。可以是‘NCHW’、‘NHWC’等。‘N’、‘C’、‘H’、‘W’分别代表批次、通道、高度和宽度维度。卷积应用于‘H’和‘W’维度。

### ***property*dilation**

指定用于扩张卷积的扩张率。

### ***property*groups**

分组卷积将输入分成多少组。输入和输出通道的数量应该能被组数整除。

### ***property*kernel_layout**

权重的维度排序。可以是‘OIHW’、‘OIHW16o16i’等。‘O’、‘I’、‘H’、‘W’分别代表 num_filter、input_channel、height 和 width 维度。

### ***property*out_dtype**

输出数据类型，在混合精度设置下设为显式类型。

### ***property*out_layout**

输出维度的排序。可以是‘NCHW’、‘NHWC’等。'N'、'C'、'H'、'W'分别代表批次、通道、高度和宽度维度。默认与输入布局相同。

### ***property*padding**

底部和右侧将使用与顶部和左侧相同的填充值四个 int : 填充宽度按 (顶部, 左侧, 底部, 右侧) 的顺序排列。

**Type :** 所有边使用相同的填充两个整数。

### ***property*strides**
指定卷积的步长。



## *class* tvm.relax.op.op_attrs.Conv3DAttrs

nn.conv3d 的属性。 

### ***property*data_layout**

输入数据的维度顺序。可以是 ‘NCDHW’、‘NDHWC’ 等。'N'、'C'、'D'、'H'、'W' 分别代表批次、通道、深度、高度和宽度维度。卷积在 ‘D’、‘H’ 和 ‘W’ 维度上应用。

### ***property*dilation**

指定用于扩张卷积的扩张率。

### ***property*groups**

分组卷积将输入分成多少组。输入和输出通道的数量应该能被组数整除。

### ***property*kernel_layout**

权重的维度排序。可以是“OIDHW”、“OIDHW16o16i”等。“O”、“I”、“D”、“H”、“W”分别代表 num_filter、input_channel、depth、height 和 width 维度。

### ***property*out_dtype**

输出数据类型，在混合精度设置下设为显式类型。

### ***property*out_layout**

输出维度的排序。可以是‘NCDHW’、‘NDHWC’等。'N'、'C'、'D'、'H'、'W'分别代表批次、通道、深度、高度和宽度维度。默认与输入布局相同。

### ***property*padding**

底部和右侧将使用与顶部和左侧相同的填充值六整数：填充宽度按顺序为（前，后，上，左，下，右）。

**Type :**如果填充非零，则输入隐式零填充 Padding 支持对称和非对称作为一整数。

**Type :**所有边使用相同的填充两个整数。


### ***property*strides**

指定卷积的步长。


## *class* tvm.relax.op.op_attrs.Conv2DTransposeAttrs

nn.conv2d_transpose 的属性。

### ***property*data_layout**

输入数据的维度顺序。可以是‘NCHW’、‘NHWC’等。‘N’、‘C’、‘H’、‘W’分别代表批次、通道、高度和宽度维度。卷积应用于‘H’和‘W’维度。

### ***property*dilation**

指定用于扩张卷积的扩张率。

### ***property*groups**

分组卷积将输入分成多少组。输入和输出通道的数量应该能被组数整除。

### ***property*kernel_layout**

权重的维度排序。可以是‘OIHW’、‘OIHW16o16i’等。‘O’、‘I’、‘H’、‘W’分别代表 num_filter、input_channel、height 和 width 维度。

### ***property*out_dtype**

输出数据类型，在混合精度设置下设为显式类型。

### ***property*out_layout**

输出维度的排序。可以是‘NCHW’、‘NHWC’等。'N'、'C'、'H'、'W'分别代表批次、通道、高度和宽度维度。默认与输入布局相同。

### ***property*output_padding**

用于消除输出形状的歧义。

### ***property*padding**

底部和右侧将使用与顶部和左侧相同的填充值四个 int : 填充宽度按 (顶部, 左侧, 底部, 右侧) 的顺序排列。

**Type :**如果填充非零，则输入隐式零填充 Padding 支持对称和非对称作为一整数。

**Type :**所有边使用相同的填充两个整数。

### ***property*strides**

指定卷积的步长。

## *class* tvm.relax.op.op_attrs.Pool2DAttrs


nn.max_pool2d 的属性。

### ***property*ceil_mode**

一个布尔值，表示是否使用向上取整或向下取整来计算输出形状。使用向上取整时，输入张量的每个元素都将被滑动窗口覆盖。

### ***property*count_include_pad**

当为真时，将包含填充以计算平均值。

### ***property*dilation**

指定卷积的膨胀率。

### ***property*layout**

输入数据的维度顺序。可以是‘NCHW’、‘NHWC’等。’N’、‘C’、‘H’、‘W’分别代表批次、通道、高度和宽度维度。池化操作应用于‘H’和‘W’维度。

### ***property*out_layout**

输出数据的维度顺序。可以是‘NCHW’、‘NHWC’等。’N’、‘C’、‘H’、‘W’分别代表批次、通道、高度和宽度维度。池化操作应用于‘H’和‘W’维度。

### ***property*padding**

底部和右侧将使用与顶部和左侧相同的填充值四个 int : 填充宽度按 (顶部, 左侧, 底部, 右侧) 的顺序排列。

**Type :** 如果填充非零，则输入隐式零填充 Padding 支持对称和非对称作为一整数。

**Type :** 所有边使用相同的填充两个整数。

### ***property*pool_size**
池化窗口的大小。

### ***property*strides**

指定卷积的步长。




## *class* tvm.relax.op.op_attrs.AdaptivePool2DAttrs


2d 自适应池化运算符的属性。

### ***property*layout**

输入数据的维度顺序。可以是‘NCHW’、‘NHWC’等。’N’、‘C’、‘H’、‘W’分别代表批次、通道、高度和宽度维度。池化操作应用于‘H’和‘W’维度。

### ***property*out_layout**

输出数据的维度顺序。可以是‘NCHW’、‘NHWC’等。’N’、‘C’、‘H’、‘W’分别代表批次、通道、高度和宽度维度。池化操作应用于‘H’和‘W’维度。

### ***property*output_size**

输出高度和宽度。

## *class* tvm.relax.op.op_attrs.SoftmaxAttrs


nn.softmax 运算符的属性。

### ***property*axis**

计算 softmax 时要对其求和的轴。

### *class* tvm.relax.op.op_attrs.BatchNormAttrs


批量归一化运算符使用的属性

### ***property*axis**
归一化应用的轴。

### ***property*center**

指示是否将 beta 偏移量添加到归一化张量。

### ***property*epsilon**

为方差添加的小浮点数以避免除以零。

### ***property*momentum**

用于移动平均均值和移动方差更新的值。

### ***property*scale**

指示是否将 gamma 尺度相乘。

### ***property*training**

我们是否在训练（即不在评估模式下）。

## *class* tvm.relax.op.op_attrs.LayerNormAttrs


层归一化运算符使用的属性。

### ***property*axes**
归一化应用的轴。

### ***property*center**

指示是否将 beta 偏移量加到归一化张量上。

### ***property*epsilon**

一个小的浮点数加到方差上以避免除以零。

### ***property*scale**

指示是否将伽马尺度相乘。

## *class* tvm.relax.op.op_attrs.InstanceNormAttrs


instance_norm 操作使用的属性

### ***property*axes**

归一化应用的轴。

### ***property*center**

指示是否将 beta 偏移量添加到归一化张量。

### ***property*channel_axis**

表示通道的轴。

### ***property*epsilon**

为方差添加的小浮点数以避免除以零。

### ***property*scale**

指示是否将 gamma 缩放相乘。

## *class* tvm.relax.op.op_attrs.DropoutAttrs


dropout 运算符的属性。

### ***property*rate**

训练期间被丢弃的输入比例。

## *class* tvm.relax.op.op_attrs.StatisticalAttrs


统计运算符中使用的属性。

### ***property*axis**

执行归约的轴或轴。

### ***property*keepdims**

如果设置为 True，缩减的轴会以大小为 1 的维度保留在结果中。

## *class* tvm.relax.op.op_attrs.ConcatAttrs 


concat 运算符的属性。

### ***property*axis**

输入数组连接的轴。应在 [-ndim, ndim) 范围内。

## *class* tvm.relax.op.op_attrs.ExpandDimsAttrs


expand_dims 运算符的属性。

### ***property*axis**

输入数组扩展的轴。所有值都必须位于范围 [-data.ndim - 1, data.ndim] 内，遵循负索引的约定。

## *class* tvm.relax.op.op_attrs.PermuteDimsAttrs 


permute_dims 运算符的属性。

### ***property*axes**

目标轴顺序，如果未指定则逆序。

## *class* tvm.relax.op.op_attrs.SortAttrs


排序运算符的属性

### ***property*axis**

计算排序的轴。默认使用最后一个轴。

### ***property*descending**

是否按降序排序。如果未指定，则默认为升序。

## *class* tvm.relax.op.op_attrs.ArgsortAttrs


argsort 运算符的属性。

### ***property*axis**

argsort 计算的轴。默认使用最后一个轴。

### ***property*descending**

是否以降序 argsort。如果未指定，默认为升序。

### ***property*dtype**

输出索引的数据类型。

## *class* tvm.relax.op.op_attrs.SplitAttrs


split 运算符使用的属性。

### ***property*axis**

要拆分的轴。

### ***property*indices_or_sections**

索引输入数组或分割部分的数目。

## *class* tvm.relax.op.op_attrs.SqueezeAttrs


squeeze 运算符的属性。

### ***property*axis**

在输入张量中要压缩的轴。如果 axis = None，则压缩维度 1 的所有轴；否则，压缩 axes 中指定的维度。如果轴的维度不是 1，则会出错。

## *class* tvm.relax.op.op_attrs.StackAttrs


concat 运算符的属性。

### ***property*axis**

用于堆叠输入张量的轴。该轴将被插入到输出的这个位置，因此它必须在范围 [-ndim-1, ndim] 内，其中 ndim 是输入张量的维度数量。

## *class* tvm.relax.op.op_attrs.IndexPutAttrs


index_put 运算符的属性。

### ***property*accumulate**

是否累积（相加）值而不是替换。如果为真，执行 tensor[indices] += values，否则执行 tensor[indices] = values。

## *class* tvm.relax.op.op_attrs.LayoutTransformAttrs


布局转换操作中使用的属性。

### ***property*axis_separators**

生成扁平输出轴时输入轴之间的分隔符。

### ***property*index_map**

要应用的空间变换。

### ***property*input_axis_separators**

重新生成输出时的轴之间的分隔符。

### ***property*pad_value**

如果布局转换会导致隐式填充，则用于填充的特定值。如果未指定，编译器可以自由选择任何值。

## *class* tvm.relax.op.op_attrs.Resize2DAttrs


图像 resize2d 操作中使用的属性。

### ***property*coordinate_transformation_mode**

描述了如何将调整大小后的张量的坐标转换为原始张量的坐标。详细内容请参阅 ONNX Resize 算子规范。可选的选项有 half_pixel、align_corners 和 asymmetric。

### ***property*cubic_alpha**

双三次插值的三次样条系数。

###  ***property*cubic_exclude**

用于在双三次插值过程中排除图像外部的标志。

###  ***property*extrapolation_value**

当 roi 在图像外部时返回的值。

###  ***property*layout**

输入数据的维度顺序。可以是‘NCHW’、‘NHWC’等。‘N’、‘C’、‘H’、‘W’分别代表批次、通道、高度和宽度维度。Resize 操作应用于‘H’和‘W’维度。

###  ***property*method**

指定用于缩放的模式。nearest_neighbor - 最近邻 linear - 双线性插值 cubic - 双三次插值。

###  ***property*out_dtype**

输出张量的 dtype。如果未指定，输出将具有与输入相同的 dtype（如果未指定）。

###  ***property*roi**

感兴趣区域（Region of Interest）用于坐标变换模式‘tf_crop_and_resize’。

###  ***property*rounding_method**

指示在最近邻方法中如何找到“最近”的像素。可用选项有 round、floor 和 ceil。


## *class* tvm.relax.op.op_attrs.ArgmaxArgminAttrs

argmax/argmin 运算符的属性。

### ***property*axis**

执行 argmin/argmax 的轴。

###  ***property*keepdims**

如果设置为 True，则缩减的轴会以大小为 1 的维度保留在结果中。


## *class* tvm.relax.op.op_attrs.RepeatAttrs


repeat 运算符的属性。

### ***property*axis**


沿哪个轴重复值。负数从后向前计数。默认情况下，使用展平的输入数组，并返回一个展平的输出数组。

###  ***property*repeats**


重复次数。


## *class* tvm.relax.op.op_attrs.TileAttrs


分块运算符的属性。

### ***property*repeats**

每个轴上数据重复的次数。

## *class* tvm.relax.op.op_attrs.ScanopAttrs


扫描运算符的属性。

### ***property*axis**


执行扫描计算沿的轴。默认值（None）是对展平数组进行计算。

###  ***property*dtype**


输出数据类型。如果未指定 dtype，则默认为输入数据的 dtype。

###  ***property*exclusive**


第一个元素不包括。


## *class* tvm.relax.op.op_attrs.TopKAttrs


topk 运算符的属性。

### ***property*axis**

沿此轴对输入张量进行排序。

###  ***property*dtype**

输出索引的数据类型。

###  ***property*k**


选择顶部元素的数量。

###  ***property*largest**


是否返回最大或最小的元素。默认情况下，返回最大的 k 个元素。

###  ***property*ret_type**

返回类型 [both, values, indices]。both - 返回 top k 数据和索引.values - 仅返回 top k 数据.indices - 仅返回 top k 索引。


## *class* tvm.relax.op.op_attrs.EinsumAttrs


einsum 运算符的属性。

### ***property*subscripts**

einsum 表达式字符串。

## *class* tvm.relax.op.op_attrs.FlipAttrs


 翻转运算符的属性。

### ***property*axis**


翻转的轴。



## ***class*tvm.relax.op.op_attrs.PadAttrs**


pad 算子中使用的属性。

### ***property*pad_mode**

使用的填充类型。“constant”用 constant_value 进行填充，“edge”使用输入数组的边缘值进行填充，“reflect”通过相对于边缘反射值进行填充。

###  ***property*pad_value**


用于填充填充区域的值。

###  ***property*pad_width**


每个轴边缘填充的值的数量，格式为（before_1, after_1, …, before_N, after_N）。

## ***class*tvm.relax.op.op_attrs.MultinomialFromUniformAttrs**


multinomial_from_uniform 算子的属性。

### ***property*dtype**

输出索引的数据类型。

## ***class*tvm.relax.op.op_attrs.CallInplacePackedAttrs**


call_inplace_packed 操作符使用的属性。

## ***class*tvm.relax.op.op_attrs.CallTIRInplaceAttrs**


call_tir_inplace 操作符使用的属性。

## ***class*tvm.relax.op.op_attrs.ToVDeviceAttrs**


to_vdevice 操作符使用的属性。

### ***property*dst_vdevice**


数据被复制到的目标设备。

## ***class*tvm.relax.op.op_attrs.HintOnDeviceAttrs**


用于 hint_on_device 操作符的属性。

### ***property*dev_id**

设备 ID。

###  ***property*dev_type**


数据预期执行设备类型。

## ***class*tvm.relax.op.op_attrs.ScatterCollectiveAttrs**


scatter 集体操作符中使用的属性。

### ***property*axis**


要散布的张量的轴。张量将沿此轴分块。

###  ***property*num_workers**


工作线程的数量，也是给定缓冲区应分块成部分的数量。

## ***class*tvm.relax.op.op_attrs.AttentionAttrs**


用于注意力算子的属性。

### ***property*causal_mask**


因果掩码的类型，即“TopLeft”和“BottomRight”。

###  ***property*scale**


在 softmax 之前应用的定制缩放。默认值为 1 / sqrt(head_dim)。

###  ***property*window_size**


滑动窗口注意力的窗口大小。

## ***class*tvm.relax.op.op_attrs.Conv1DAttrs**


nn.conv1d 的属性。

### ***property*data_layout**

输入数据的维度排序。可以是 ‘NCW’、‘NWC’ 等。'N'、'C'、'W' 分别代表批次、通道、宽度。卷积应用于 ‘W’ 维度。

###  ***property*dilation**


指定用于扩张卷积的扩张率。

###  ***property*groups**


分组卷积将输入分成多少组。输入和输出通道的数量应该能被组数整除。

###  ***property*kernel_layout**

权重的维度排序。可以是“OIW”、“IOW”等。“O”、“I”、“W”分别代表 num_filter、input_channel 和 width 维度。

###  ***property*out_dtype**


输出数据类型，在混合精度设置下设为显式类型。

###  ***property*out_layout**


输出维度的排序。可以是‘NCW’、‘NWC’等。'N'、'C'、'W'分别代表批次、通道和宽度维度。默认与输入布局相同。

###  ***property*padding**


padding 宽度顺序为（左，右）

**Type :** 如果填充非零，则输入隐式零填充 Padding 支持对称和非对称作为一整数。

**Type :** 相同填充应用于两侧两个整数。

###  ***property*strides**


指定卷积的步长。

## ***class*tvm.relax.op.op_attrs.Conv1DTransposeAttrs**


nn.conv1d_transpose 的属性。

### ***property*data_layout**

channel, widthdimensions respectively. Convolution is applied on the ‘W’ dimensions.

输入数据的维度排序。可以是 ‘NCW’、‘NWC’ 等。'N'、'C'、'W' 分别代表批次、通道、宽度。卷积应用于 ‘W’ 维度。

###  ***property*dilation**


指定用于扩张卷积的扩张率。

###  ***property*groups**


分组卷积将输入分成多少组。输入和输出通道的数量应该能被组数整除。

###  ***property*kernel_layout**


权重的维度排序。可以是“OIW”、“IOW”等。“O”、“I”、“W”分别代表 num_filter、input_channel 和 width 维度。

###  ***property*out_dtype**


输出数据类型，在混合精度设置下设为显式类型。

###  ***property*out_layout**


输出维度的排序。可以是‘NCW’、‘NWC’等。'N'、'C'、'W'分别代表批次、通道和宽度维度。默认与输入布局相同。

###  ***property*output_padding**


用于消除输出形状的歧义。

###  ***property*padding**


padding 宽度顺序为（左，右）

**Type :** 如果填充非零，则输入隐式零填充 Padding 支持对称和非对称作为一整数。

**Type :** 相同填充应用于两侧两个整数。

###  ***property*strides**


指定卷积的步长。

## ***class*tvm.relax.op.op_attrs.Pool1DAttrs**


nn.max_pool1d 和 nn.avg_pool1d 的属性。

### ***property*ceil_mode**

一个布尔值，表示是否使用向上取整或向下取整来计算输出形状。使用向上取整时，输入张量的每个元素都将被滑动窗口覆盖。

###  ***property*count_include_pad**


当为真时，将包含填充以计算平均值。

###  ***property*dilation**


指定卷积的膨胀率。

###  ***property*layout**


输入数据的维度顺序。可以是“NCW”、“NWC”等。“N”、“C”、“W”分别代表批次、通道和宽度维度。池化操作应用于“W”维度。

###  ***property*out_layout**


输出数据的维度顺序。可以是“NCW”、“NWC”等。“N”、“C”、“W”分别代表批次、通道和宽度维度。池化操作应用于“W”维度。

###  ***property*padding**


padding 宽度顺序为（左，右）

**Type :** 如果填充非零，则输入隐式零填充 Padding 支持对称和非对称作为一整数。

**Type :** 所有边使用相同的填充两个整数。

###  ***property*pool_size**


池化窗口的大小。

###  ***property*strides**


指定卷积的步长。

## ***class*tvm.relax.op.op_attrs.Pool3DAttrs**


nn.max_pool3d 和 nn.avg_pool3d 的属性。

### ***property*ceil_mode**


一个布尔值，表示是否使用向上取整或向下取整来计算输出形状。使用向上取整时，输入张量的每个元素都将被滑动窗口覆盖。

###  ***property*count_include_pad**


当为真时，将包含填充以计算平均值。

###  ***property*dilation**


指定卷积的膨胀率。

###  ***property*layout**


输入数据的维度顺序。可以是‘NCDHW’、‘NDHWC’等。‘N’、‘C’、‘D’、‘H’、‘W’分别代表批次、通道、深度、高度和宽度维度。池化操作应用于‘D’、‘H’和‘W’维度。

###  ***property*out_layout**


输出数据的维度顺序。可以是‘NCDHW’、‘NDHWC’等。‘N’、‘C’、‘D’、‘H’、‘W’分别代表批次、通道、深度、高度和宽度维度。池化操作应用于‘D’、‘H’和‘W’维度。

###  ***property*padding**


back, bottom, right 将使用与 front, top, left 相同的填充 four int : 填充宽度按顺序为（front, top, left, back, bottom, right）。

**Type :** 如果填充非零，则输入隐式零填充 Padding 支持对称和非对称作为一整数。

**Type :** 相同填充用于所有侧边三个整数。

###  ***property*pool_size**


池化窗口的大小。

###  ***property*strides**


指定卷积的步长。

## ***class*tvm.relax.op.op_attrs.AdaptivePool1DAttrs**


1d 自适应池算子的属性。

### ***property*layout**


输入数据的维度顺序。可以是‘NCW’、‘NWC’等。’N’、‘C’、‘W’分别代表批次、通道和宽度维度。池化操作应用于‘W’维度。

###  ***property*out_layout**


输出数据的维度顺序。可以是‘NCW’、‘NWC’等。’N’、‘C’、‘W’分别代表批次、通道和宽度维度。池化操作应用于‘W’维度。

###  ***property*output_size**

 输出宽度。

## ***class*tvm.relax.op.op_attrs.AdaptivePool3DAttrs**


三维自适应池化算子的属性。

### ***property*layout**


输入数据的维度顺序。可以是‘NCDHW’、‘NDHWC’等。‘N’、‘C’、‘D’、‘H’、‘W’分别代表批次、通道、深度、高度和宽度维度。池化操作应用于‘D’、‘H’和‘W’维度。

###  ***property*out_layout**


输出数据的维度顺序。可以是‘NCDHW’、‘NDHWC’等。‘N’、‘C’、‘D’、‘H’、‘W’分别代表批次、通道、深度、高度和宽度维度。池化操作应用于‘D’、‘H’和‘W’维度。

###  ***property*output_size**


输出深度、高度和宽度。

## ***class*tvm.relax.op.op_attrs.LeakyReluAttrs**


leaky_relu 操作符使用的属性。

### ***property*alpha**


负部分的斜率。

## ***class*tvm.relax.op.op_attrs.SoftplusAttrs**


softplus 算子中使用的属性。

### ***property*beta**


控制 Softplus 转换锐度的缩放因子。

###  ***property*threshold**


确定何时使用线性近似以保证数值稳定性的值。

## ***class*tvm.relax.op.op_attrs.PReluAttrs**


在 prelu 算子中使用的属性。

### ***property*axis**


应用 alpha 值的轴。

## ***class*tvm.relax.op.op_attrs.PixelShuffleAttrs**


pixel_shuffle 算子中使用的属性。

### ***property*upscale_factor**


空间上采样用的缩放因子。

## ***class*tvm.relax.op.op_attrs.GroupNormAttrs**

group_norm 操作符使用的属性。

### ***property*axes**


沿其应用归一化（不包括通道轴）的轴。

###  ***property*center**


指示是否将 beta 偏移量添加到归一化张量。

###  ***property*channel_axis**


表示通道的轴。

###  ***property*epsilon**


为方差添加的小浮点数以避免除以零。

###  ***property*num_groups**


将通道分开的组数。

###  ***property*scale**


指示是否将 gamma 缩放相乘。

## ***class*tvm.relax.op.op_attrs.RMSNormAttrs**


rms_norm 操作符使用的属性。

### ***property*axes**


归一化应用的轴。

###  ***property*epsilon**


为方差添加的小浮点数以避免除以零。

## ***class*tvm.relax.op.op_attrs.NLLLossAttrs**


nll_loss 操作符使用的属性。

### ***property*ignore_index**

要忽略的目标值。

###  ***property*reduction**


应用于输出的归约方法。可以是'none'、'mean'或'sum'。

## ***class*tvm.relax.op.op_attrs.AllReduceAttrs**


用于 allreduce 操作符的属性。

### ***property*in_group**


该归约操作是在分组内执行、全局执行还是默认在分组内执行。

###  ***property*op_type**


应用于输入数据的归约操作类型。目前仅支持求和。

## ***class*tvm.relax.op.op_attrs.AllGatherAttrs**


allgather 操作使用的属性。

### ***property*in_group**


allgather 操作是在组内执行还是全局执行，或者默认在组内执行。

###  ***property*num_workers**


工作线程的数量，也是给定缓冲区应分块成部分的数量。

## ***class*tvm.relax.op.op_attrs.WrapParamAttrs**


用于 wrap_param 操作的属性。

### ***property*dtype**

  目标数据类型。

## ***class*tvm.relax.op.op_attrs.QuantizeAttrs**


量化/反量化算子中使用的属性。

### ***property*axis**


通道逐个量化/反量化的输出通道轴。默认值是-1，对应于最后一个轴。

###  ***property*out_dtype**

 输出数据类型。

## ***class*tvm.relax.op.op_attrs.GatherElementsAttrs**


 操作符的属性。

### ***property*axis**

用于索引的轴。

## ***class*tvm.relax.op.op_attrs.GatherNDAttrs**

gather_nd 算子的属性。

### ***property*batch_dims**


批处理维度的数量。

## ***class*tvm.relax.op.op_attrs.MeshgridAttrs**


网格运算符的属性

### ***property*indexing**


指定网格维度如何排序。

## ***class*tvm.relax.op.op_attrs.ScatterElementsAttrs**


scatter_elements 操作符的属性。

### ***property*axis**


选择值的轴。

###  ***property*reduction**


scatter_elements 元素的归约模式，可以是“update”、“add”、“mul”、“mean”、“min”或“max”。

## ***class*tvm.relax.op.op_attrs.ScatterNDAttrs**


scatter_nd 操作符的属性。

### ***property*reduction**


ScatterND 的累积模式，可以是“update”、“add”、“mul”、“min”或“max”。

## ***class*tvm.relax.op.op_attrs.SliceScatterAttrs**


slice_scatter 操作符的属性。

### ***property*axis**


插入切片的维度。

## ***class*tvm.relax.op.op_attrs.OneHotAttrs**


one_hot 操作符的属性。

### ***property*axis**

Axis to fill.  填充轴。

###  ***property*depth**


One-hot 维度的深度。


