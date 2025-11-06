---

title: tvm.relax.transform

---



TVM Relax 中用于图优化和程序变换的模块

## tvm.relax.transform.AdjustMatmulOrder()


将 x*(A*B)重新排序为(x*A)*B。


有助于优化 LoRA 计算，其中 matmul(x, LoraA*LoraB)可以计算为 matmul(matmul(x, LoraA), LoraB)，从而减少总内存使用量。
* **返回：ret**：相应的 pass。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.AllocateWorkspace() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


分配一个工作区，用一个足够大的张量来表示，用于所有需要临时存储的外部函数，并将其附加到外部函数的参数中。


外部函数可以通过 kWorkspaceSize 属性指定其工作空间要求。
* **返回：ret**：用于分配工作空间的注册通道。
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.AlterOpImpl(*op_impl_map:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*]*, *op_buffer_transforms:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[IndexMap](/docs/api-reference/python-api/tvm-tir/#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*|*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*]]*, *op_buffer_axis_separators:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[List](https://docs.python.org/3/library/typing.html#typing.List)*[axis_separator |*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*]]*, *op_buffer_input_axis_separators:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[List](https://docs.python.org/3/library/typing.html#typing.List)*[axis_separator |*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*]]*)


将所有具有匹配“operator_name”属性的 PrimFunc 替换为可能在 I/O 缓冲区上具有不同布局的替代 PrimFunc。I/O 缓冲区的布局转换存在于 op_buffer_transforms 映射中。将布局转换插入到被替换的 PrimFunc 的调用点中，以便新的 PrimFunc 将 I/O 张量转换为预期的布局。
* **参数：**
   * **op_impl_map** (*Dict[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*]*)：op_kind 到 PrimFunc 的映射。
   * **op_buffer_transforms** (*Dict**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***List****[**Union****[*[IndexMap](/docs/api-reference/python-api/tvm-tir/#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*,Callable]])：op_kind 用于为每个缓冲区布局转换图。
   * **op_buffer_axis_separators** (*Dict**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***List****[**Union****[**IndexMap.AXIS_SEPARATOR****,Callable**]****]]*)：每个 index_map 的 op_kind 到 axis_separator。
   * **op_buffer_input_axis_separators** (*Dict**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***List****[**Union****[**IndexMap.AXIS_SEPARATOR****,Callable**]****]]*)：输入 index_map 的 op_kind 到 axis_separator。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.AnnotateTIROpPattern() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)

注释 TIR 函数的操作模式类型。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.AttachAttrLayoutFreeBuffers() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将布局空闲缓冲区附加到 tir::PrimFunc。


此过程用于根据 relax 函数中的函数用法，将布局空闲缓冲区附加到 tir::PrimFunc。目前，布局空闲缓冲区是模型权重和 relax 常量。


请注意，我们建议在此过程之前应用 CanonicalizeBindings。
* **返回：ret**：用于附加布局空闲缓冲区的注册过程。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.AttachGlobalSymbol() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将 global_symbol 附加到 Relax 函数和 TIR Primfuncs 以进行代码生成。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.BindParams(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *params:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*,*Tensor*| ndarray]*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将模块函数的参数绑定到常量张量。
* **参数：**
   * **func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：要绑定的函数名称。
   * **params** (*Dict**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***]****,Union**[****tvm.runtime.NDArray**,*** ***np.ndarray****]]*)**：** 从参数或参数名称到常量张量的映射。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.BindSymbolicVars(*binding_map:*[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将模块函数的参数绑定到常量张量。
* **参数：**
   * **binding_map** (*Mapping**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** [tvm.tir.Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)***]****,tvm.tir.PrimExpr]*)：从符号 varname 到整数的映射。
   * **func_name** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：要绑定的函数名称。如果为 None （默认），则模块内的所有函数都将更新。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.BundleModelParams(*param_tuple_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将多个模型参数捆绑成一个元组参数


对于每个函数，如果该函数具有属性“num_input”，则将其运行时参数和编译时权重分开。运行时参数（例如激活函数）是第一个 num_input 参数，其余参数是编译时权重。
* **参数：param_tuple_name** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：元组参数的名称。如果未指定，则默认为“model_params”。
* **返回：ret**- 用于捆绑模型参数的注册过程。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.CallTIRRewrite() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


为 call_tir 和 call_dps_packed 执行显式张量分配。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.CanonicalizeBindings() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


规范化变量定义（例如，如果 y = x 且 z = y，则将 y 和 z 替换为 x）。同时简化匹配强制类型转换节点（消除冗余检查）和元组索引。


最好与常量折叠和消除未使用的定义相结合。


注意：如果数据流变量仅用于与数据流块输出变量（即非数据流变量）的绑定，则此过程还将删除数据流变量并用数据流变量的直接定义替换输出变量的绑定。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.CombineParallelMatmul(*check=None*)


将共享同一 LHS 矩阵的多个 matmul 操作符合并为一个，然后进行切片。当树中的所有 matmul 分支都具有相同的融合操作符集时，这些融合操作符将应用于合并后的 matmul 输出，然后再进行切片。


目前仅支持有限的融合操作，包括 bias add、relu、gelu、gelu_tanh 和 silu 激活。
* **参数：**
   * **check** (*Callable**[****[*[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*,List**[***[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***]****,List**[***[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***]****,Dict**[***[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***,*** ***Expr**]**]****,*[bool](https://docs.python.org/3/library/functions.html#bool)*]*)– 用于筛选不需要分支的函数，其函数签名为 (input, [rhs], [bias], binding)-> bool。
  
* **返回：ret**：相应的 pass。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.ComputePrimValue() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


计算所有 R.prim_value 实例。


虽然高级的 Relax 函数可以包含符号变量形式的表达式，但这些表达式无法在 Relax 函数中原生计算。为了给符号表达式提供值（例如 R.prim_value(N*N)，其中 N 是符号变量），此过程会生成一个 PrimFunc 函数，用于计算该表达式。之后，Relax 函数图会更新，包含对该 PrimFunc 函数的调用，以替代原始的 R.prim_value(expr)。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.ConvertLayout(*desired_layouts:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]]*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)

自动布局转换过程。
* **参数：desired_layouts** (*Dict**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***List****[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]]*)：conv2d 操作所需的布局是从操作名称到所需特征图、权重和输出的所需布局的映射。例如，如果我们想将 conv2d 的布局从 NCHW 转换为 NHWC，我们可以将 conv2d 的所需布局设置为 。`{"relax.nn.conv2d": ["NHWC", "OHWI"]}`
* **返回：ret**：布局转换的注册过程。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。
将绑定块内的连续数据流操作转换为数据流块的过程。

## tvm.relax.transform.ConvertToDataflow(*min_size:*[int](https://docs.python.org/3/library/functions.html#int)*= 2*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)

将绑定块内的连续数据流操作转换为数据流块的过程。

注意：可能需要先调用 ConvertToDataflow。
* **参数：min_size** ([int](https://docs.python.org/3/library/functions.html#int))：提取新块所需的连续数据流绑定的最小数量。
* **返回：ret**：传递。
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## *class* tvm.relax.transform.DataflowBlockPass


对模块中每个 tvm.relax.DataflowBlock 进行操作的通道。

## tvm.relax.transform.DataflowUseInplaceCalls() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将对可就地执行的操作符（通常是逐元素运算）的调用更改为就地实现。支持的操作符将被替换为对 call_tir_inplace 的调用，该调用会调用这些操作符的就地 PrimFunc 实现（这些实现基于这些操作符的合法化）。


注意：可能需要先调用 ConvertToDataflow 来提供数据流块。
* **返回：ret**：该传递。
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.DeadCodeElimination(*entry_functions:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


删除 IRModule 中的死代码。目前，它会删除：


1. 未使用的本地 VarBindings（未使用绑定变量且未使用不纯操作的 VarBinding）。

2. 模块中未使用的 Relax 函数。我们从入口函数开始检测调用链，并删除所有未使用的函数。


任何留空的绑定块都将被规范化器删除。


**备注**


对于功能性的 DCE，使用 py:func: tvm.relax.analysis.remove_all_unused。
* **参数：entry_functions** (*Optional**[****List**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***]****]*)：开始的入口函数集。
* **返回：ret**：已注册的通行证。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.DecomposeOpsForInference(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass) 


在推理过程中，分解由其他算子组成的复合算子。例如，批量归一化的结果（一个三元组）将被简化。注意力机制、tensor_to_shape 等也可以分解成多个简化的算子。
* **参数：func_name** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：指定函数的名称。如果未指定，则该过程将在所有函数中运行。
* **返回：ret**：注册通行证。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.DecomposeOpsForTraining(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


在训练过程中，分解由其他算子组合而成的复合算子。例如，批量归一化的结果（一个三元组）将被简化。注意力机制、tensor_to_shape 等也可以分解成多个简化的算子。
* **参数：**
   * **func_name** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：指定函数的名称。如果未指定，则该过程将在所有函数中运行。
* **返回：ret**：注册通行证
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)

## tvm.relax.transform.EliminateCommonSubexpr(*call_only=False*) → [FunctionPass](/docs/api-reference/python-api/tvm-relax-transform#class-tvmrelaxtransformfunctionpass)


消除函数内的公共子表达式。


注意：对于嵌套函数，此过程在这些函数*内执行 CSE。*
* **参数：call_only** ([bool](https://docs.python.org/3/library/functions.html#bool))：如果为 True，则启用仅消除呼叫节点。
* **返回：ret**：消除公共子表达式的注册过程。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.ExpandMatmulOfSum()


扩展 matmul(x, A+B)为 matmul(x,A) + matmul(x,B)。


如果任一操作数可以在编译时完全计算（仅取决于 kNumInput 之后的函数参数），则抑制此扩展。


对于优化 LoRA 计算很有用，其中 matmul(x, Base + LoraA*LoraB)可以扩展为 matmul(x, Base) + matmul(x, LoraA*LoraB)，从而允许使用 CombineParallelMatmul 进行优化。
* **返回：ret**：相应的 pass。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.ExpandTupleArguments() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将元组参数扩展为内部函数。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.FewShotTuning(*valid_count:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *benchmark:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


该通道专为静态形状 PrimFuncs 的少量调优而设计。它会检查 PrimFunc 中的所有块，并根据 MetaSchedule 调度规则进行循环融合、拆分和其他转换，但会直接从搜索空间采样，而不是使用调优算法。用户可以指定要尝试的有效计数数量，以及是否使用运行器进行基准测试。
* **参数：**
   * **valid_count** ([int](https://docs.python.org/3/library/functions.html#int))：要尝试的有效计数的数量。
   * **benchmark** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否使用 runner 进行基准测试。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.FoldConstant() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


折叠数据流块内的常量表达式。


注意：可能需要先调用 ConvertToDataflow 来提供数据流块。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## *class* tvm.relax.transform.FunctionPass


一个作用于模块中每个 tvm.relax.Function 的 Pass。一个函数 Pass 类应通过 function_pass 创建。

## tvm.relax.transform.FuseOps(*fuse_opt_level=-1*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


此过程将绑定分组到 Relax 函数的数据流块中，并根据过程实现中描述的融合算法为每个组生成一个新的 Relax 函数分组。通过将绑定分组到新的 Relax 函数中，我们将被操作函数中的绑定替换为对新分组函数的函数调用。


一个名为“FuseTIR”的后续过程将为每个分组函数生成一个 TIR PrimFunc。


注意：可能需要先调用 ConvertToDataflow 来提供数据流块。
* **参数：fuse_opt_level** ([int](https://docs.python.org/3/library/functions.html#int))：1 表示将从 pass 上下文中推断级别。
* **返回：ret**：操作符融合的注册通道。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.FuseOpsByPattern(*patterns:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[FusionPattern](/docs/api-reference/python-api/tvm-relax-transform#class-tvmrelaxtransformfusionpatternnamestr-pattern-dfpattern-annotation_patternsmappingstr-dfpattern-none-none-checkcallablepatterncheckcontextbool-none-none-attrs_gettercallabledictstrrelaxexprdictstrstr-none-none)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*]*, *bind_constants:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *annotate_codegen:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *entry_functions:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将模式匹配应用于给定模块中的每个函数，并将匹配的表达式分组为一个新函数。


最终结果与 FuseOps 类似，但融合完全由提供的模式驱动。


注意：仅在数据流块内操作。可能需要先调用 ConvertToDataflow 。
* **参数：**
   * **patterns** (*List**[****Union**[***[FusionPattern](/docs/api-reference/python-api/tvm-relax-transform#class-tvmrelaxtransformfusionpatternnamestr-pattern-dfpattern-annotation_patternsmappingstr-dfpattern-none-none-checkcallablepatterncheckcontextbool-none-none-attrs_gettercallabledictstrrelaxexprdictstrstr-none-none)***,*** [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)***]****]*)：要匹配的模式列表。模式的顺序决定了匹配的优先级顺序。高优先级模式应出现在列表的前面。除了 FusionPattern 之外，还可以将元组作为此列表的元素传递。模式将通过 `FusionPattern(*item)` 构建。
   * **bind_constants** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否在分组函数中保留绑定常量。
   * **annotate_codegen** ([bool](https://docs.python.org/3/library/functions.html#bool))：如果为 True，将每个创建的复合函数用另一个函数包装，该函数的体仅包含对复合函数的调用，并给外层函数添加“Codegen”和“global_symbol”属性。 “Codegen”属性设置为对应模式名称的前缀。例如，如果模式名称为“dnnl.conv2d_relu”，则设置为“dnnl”。如果创建的复合函数打算在不使用 MergeCompositeFunctions 遍历的情况下卸载到外部后端，则此值必须为 True。
   * **entry_functions** (*Optional**[****List**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***]****]*)：开始的入口函数集。
* **返回：ret**：基于模式的融合的注册过程。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.FuseTIR() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


如果可能的话，将原始 Relax 函数融合到更大的 TIR 函数中。
* **返回：ret***：*tir 融合的注册通道。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## *class* tvm.relax.transform.FusionPattern(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *pattern: DFPattern*, *annotation_patterns:*[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, DFPattern] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *check:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*[[*[PatternCheckContext](/docs/api-reference/python-api/tvm-relax-transform#class-tvmrelaxtransformpatterncheckcontext)*],*[bool](https://docs.python.org/3/library/functions.html#bool)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *attrs_getter:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*[[*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]],*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


FuseOpsByPattern 使用的模式。它主要是 DFPattern，但也包含其他信息以帮助进行融合过程。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：*模式名称。通常以后端名称开头，例如“cutlass.matmul”。
   * **pattern** (*DFPattern*)：用于匹配可由外部后端处理的表达式的数据流模式。
   * **annotation_patterns** (*Mapping**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***DFPattern****]*)：用于从模式匹配结果中提取重要表达式的映射。此映射中的所有 DFPattern 都应为模式的一部分。
   * **check** (*Callable**[****[*[PatternCheckContext](/docs/api-reference/python-api/tvm-relax-transform#class-tvmrelaxtransformpatterncheckcontext)*],*[bool](https://docs.python.org/3/library/functions.html#bool)*]*)*：*检查匹配结果是否被接受的函数。

## tvm.relax.transform.Gradient(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *require_grads:*[Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *target_index:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


逆向模式自动微分。


此过程将区分 IRModule 中的一个函数。现在输入函数必须只有一个数据流块（可能需要先调用 ConvertToDataflow）。


对于由 func_name 指定的给定函数，它会生成一个名为 func_name + “_adjoint”的新函数。新函数会计算**微分目标函数**相对于原函数 require_grads 指定的参数的梯度。


如果函数只有一个返回值，则该返回值将被指定为 target。如果函数有多个返回值，则该目标将被指定为第 target_index 个返回值。target 必须是一个标量（零维张量）。


新功能如下：

```python
@R.function
def main_adjoint(original_parameters):
    with R.dataflow():
        # 原始函数的绑定
        ...
        # 正在计算梯度
        ...
        R.output(original_outputs, grad*1, grad*2, ...)
    return (original_return_value, (grad*1, grad*2, ...))
```


此 AD 通道还支持检查点，具体功能请参阅“以亚线性内存成本训练深度网络” - Chen, Tianqi, et al. (2016)。更多详情，请参阅 tvm.relax.testing.nn.checkpoint。
* **参数：**
   * **func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：特定函数的名称。
   * **require_grads** (*Optional**[****Union**[***[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***,*** ***List****[*[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*]**]****]*)：需要伴随函数的松弛变量。必须是给定函数的参数，且不能重复。如果未指定，则计算所有参数的伴随函数。
   * **target_index** ([int](https://docs.python.org/3/library/functions.html#int))：如果指定的函数有多个返回值，则指定返回值的索引作为目标。如果未指定，则第一个返回值将作为目标。
* **返回：ret**：通行证。
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


**示例**


以下代码展示了如何使用此过程：

```python
@I.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
            # 使用 R.sum 将张量归约为一个标量
            lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
            R.output(lv2)
        return lv2

After = relax.transform.Gradient("main")(Module)
```


梯度 pass 之后的模块将是：

```python
@I.ir_module
class After:
    @R.function
    def main(
        x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
            lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
            R.output(lv2)
        return lv2

    @R.function
    def main_adjoint(
        x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
    ) -> R.Tuple(
        R.Tensor((), dtype="float32"),
        R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
    ):
        with R.dataflow():
            # 原始绑定
            lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
            lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
            # 关于中间变量的绑定
            lv2*adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
            lv1*adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(
                lv2*adjoint, (3, 3)
            )
            # 关于参数的绑定
            x_adjoint: R.Tensor((3, 3), dtype="float32") = lv1*adjoint
            y_adjoint: R.Tensor((3, 3), dtype="float32") = lv1*adjoint
            R.output(lv2, x_adjoint, y_adjoint)
        # 返回值：(orig_return_values, tuple(adjoints))
        return (lv2, (x_adjoint, y_adjoint))
```


第二个示例是返回多个值并使用 target_index 指定目标：

```python
@I.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
    ) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")):
        with R.dataflow():
            lv1: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            lv2: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
            R.output(lv1, lv2)
        return (lv1, lv2)

After = relax.transform.Gradient("main", target_index=1)(Module)
```


梯度 pass 之后的模块将是：

```python
@I.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
    ) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")):
        with R.dataflow():
            lv1: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            lv2: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
            R.output(lv1, lv2)
        return (lv1, lv2)

    @R.function
    def main_adjoint(
        x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
    ) -> R.Tuple(
        R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")),
        R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
    ):
        with R.dataflow():
            # 原始绑定
            lv1: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            lv2: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
            # 关于中间变量的绑定，与目标无关的中间变量梯度不会被计算
            # 被计算
            lv2*adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
            # 关于参数的绑定
            x_adjoint: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
            y_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(
                lv2*adjoint, (3, 3)
            )
            R.output(lv1, lv2, x_adjoint, y_adjoint)
        # 返回值: (orig_return_values, tuple(adjoints))
        return ((lv1, lv2), (x_adjoint, y_adjoint))
```
## tvm.relax.transform.InlinePrivateFunctions() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


内联所有私有放松函数。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.KillAfterLastUse() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)

最后一次使用后删除所有张量/存储对象。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.LambdaLift() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将本地函数提升到全局。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.LazyGetInput() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


延迟请求输入的通道。


很多情况下，模型权重的大小超出了 GPU 的可用内存。在这种情况下，接受所有模型权重作为参数的函数将无法调用。在这种情况下，函数必须根据需要加载参数，并在不再需要时卸载。


此过程会修改一个函数，使其所有模型权重（第一个 func.attrs[“num_input”]参数之后的参数）按需加载。该函数并非直接接受权重作为函数参数，而是接受一个回调函数参数，该回调函数可以根据需要加载每个参数。回调函数接受两个参数：第一个是模型权重的索引，第二个是参数的名称。回调函数应返回指定的参数。

```python
@R.function
def before(A: R.Tensor([16,32],"float32")):
    ...

@R.function
def after(fget_param: R.Callable([R.Prim('int64'), R.Object], R.Object)):
    A_untyped = fget_param(0, R.str('A'))
    A = R.match_cast(A_untyped, R.Tensor([16,32], "float32")
    ...
```
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.LazySetOutput() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


设置可用函数输出的通道。


很多情况下，模型权重的大小超出了 GPU 的可用内存。在这种情况下，将所有模型权重作为单个返回值的函数将无法被调用。在这种情况下，参数必须在生成时返回，并从 GPU 卸载（或保存到磁盘），然后再生成其他输出。


此过程会改变一个函数，使其所有输出在可用时都返回。该函数接受一个额外的回调参数，该回调会在函数每次输出时调用。回调接受两个参数：第一个是生成的输出元组的索引（如果输出不是元组，则为零）；第二个是值本身。

```python
@R.function
def before(args):
    ...
    return (A, B)

@R.function
def after(args, fset_param: R.Callable([R.Prim('int64'), R.Object])):
    ...
    fset_param(0, A)
    ...
    fset_param(1, B)
    ...
    return ()
```
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.LegalizeOps(*customize_legalize_map:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*[[*[BlockBuilder](/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_builderblockbuildermodirmodulenone-none)*,*[Call](/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)*],*[RelaxExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *enable_warning:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)


将 Relax 函数中的高级操作符调用合法化为 call_tir 和相应的低级 TIR PrimFuncs。


对于每个高级操作符，我们将合法化的方式注册为一个函数，该函数接受一个上下文 BlockBuilder 和被合法化的 Relax.Call 作为输入，并返回合法化的调用。这里输入的 BlockBuilder 主要用于将 call_te 创建的 PrimFunc 添加到上下文 IRModule 中。


每个操作符的合法函数都被注册为操作符的一个属性（属性键为 FLegalize ）。


此 pass 为用户提供可定制性，以便用户为操作符使用自己的合法化函数。该 pass 接受一个可选的自定义 map，其键为操作符名称 ( str )，值是函数 ( LegalizeFunc )。默认的合法化函数将被自定义函数覆盖。
* **参数：**
   * **customize_legalize_map** (*Optional**[****Dict**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***LegalizeFunc****]]*)：自定义操作符合法化函数映射。自定义函数将覆盖默认函数。
   * **enable_warning** ([bool](https://docs.python.org/3/library/functions.html#bool))：一个布尔值，指示是否针对未注册操作合法化函数的 CallNode 打印警告。默认情况下，我们不打印警告。
* **返回：ret**：注册通行证。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


**示例**


以下代码展示了如何使用此过程：

```python
# 定义传入的 IRModule
@tvm.script.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        z: R.Tensor((2, 3), "float32") = R.add(x, y)
        r: R.Tensor((2, 3), "float32") = R.multiply(y, z)
        return r

# 为 'relax.add' 定义自定义的合法化函数
def customize_legalize_add(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    from tvm import topi
    return bb.call_te(topi.add, call.args[1], call.args[0])

# 将自定义函数应用于模块的 pass。
mod = LegalizeOps({"relax.add": customize_legalize_add})(Module)
```


通过 mod.show()打印出结果，我们可以看到合法化后的 IRModule 变成了。

```python
@tvm.script.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        z = R.call_tir(add, (y, x), (2, 3), dtype="float32")
        r = R.call_tir(multiply, (y, z), (2, 3), dtype="float32")
        return r

    @T.prim_func
    def add(
        A: T.Buffer((2, 3), "float32"),
        B: T.Buffer((2, 3), "float32"),
        T_add: T.Buffer((2, 3), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        for ax0, ax1 in T.grid(2, 3):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

    @T.prim_func
    def multiply(
        A: T.Buffer((2, 3), "float32"),
        B: T.Buffer((2, 3), "float32"),
        T_multiply: T.Buffer((2, 3), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        for ax0, ax1 in T.grid(2, 3):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = A[v_ax0, v_ax1] * B[v_ax0, v_ax1]
```
## tvm.relax.transform.LiftTransformParams(*shared_transform:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] = False*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


函数参数的提升变换。


当函数的某些输入被标记为“参数”（模型权重）时，此过程会识别参数的变换，并将其提升到名为 transform_params 的单独函数。transform_params 接受原始参数的元组作为输入，并返回变换后参数的元组。原始函数将被重写，以接受变换后参数的元组作为输入。


用户需要在运行时调用 transform_params 函数，将转换后的参数作为输入 pass 给原始函数。
* **参数：**
   * **shared_transform** (*Union**[***[bool](https://docs.python.org/3/library/functions.html#bool)***,*** ***List****[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]]*)：指示参数转换函数将如何生成。
   * False（默认）：为每个具有“num_input”属性的函数生成单独的参数转换函数。
   * True：生成单个参数转换函数，包含所有具有“num_input”属性的函数中共同的预处理步骤。
   * List[str]: 将生成一个单一参数转换函数，其中包含每个函数名称在列表中的预处理步骤。传递具有“num_input”属性的所有函数的列表或空列表等同于传递 True。
* **返回：ret**：用于提升参数变换的注册过程。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.LowerAllocTensor() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


降低 R.builtin.alloc_tensor 的剩余实例。


静态内存规划器移除了 R.builtin.alloc_tensor 的静态实例，并用 R.memory.alloc_storage 和 R.memory.alloc_tensor 替代。不过，R.builtin.alloc_tensor 仍然保留，可用于任何动态分配。


此转换会将所有剩余的 R.builtin.alloc_tensor 实例替换为 R.memory.alloc_storage 和 R.memory.alloc_tensor。如果不存在 R.builtin.alloc_tensor，则此过程无效。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.LowerRuntimeBuiltin() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将通用内在函数降低为 VM 内在函数。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.MergeCompositeFunctions() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将 FuseOpsByPattern 创建的一个或多个复合函数组合成一个新函数。新函数将使用“Codegen”和“global_symbol”属性进行注释，并计划将其卸载到外部后端。
* **返回：ret**：合并复合函数的注册过程。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.MetaScheduleApplyDatabase(*work_dir:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *enable_warning:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


从调整数据库中应用最佳计划。
* **参数：**
   * **work_dir** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：如果未提供数据库，则推断默认数据库的工作目录（当用户 pass 数据库时它将被忽略）。
   * **enable_warning** ([bool](https://docs.python.org/3/library/functions.html#bool))*：* 一个布尔值，指示是否打印数据库中未显示的 TIR 函数的警告。默认情况下，我们不打印警告。
* **返回：ret**：注册通行证。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.MetaScheduleTuneIRMod(*params:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[Tensor]*]*, *work_dir:*[str](https://docs.python.org/3/library/stdtypes.html#str), *max_trials_global:*[int](https://docs.python.org/3/library/functions.html#int), *max_trials_per_task:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *op_names:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Pass](tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


使用 MetaSchedule 调整 Relax IRModule。
* **参数：**
   * **params** (*Dict[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*]*)：模型参数。
   * **work_dir** ([str](https://docs.python.org/3/library/stdtypes.html#str))：工作目录。
   * **max_trials_gloabl** ([int](https://docs.python.org/3/library/functions.html#int))：允许调整的最大试验次数。
   * **max_trials_per_task** ([int](https://docs.python.org/3/library/functions.html#int))*：*每个任务的最大试验次数。
   * **op_names** (*Optional**[****List**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***]****]*)：用于指定需要调整的操作符的操作符名称列表。当值为 None 时，表示所有操作符均已调整。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.MetaScheduleTuneTIR(*work_dir:*[str](https://docs.python.org/3/library/stdtypes.html#str), *max_trials_global:*[int](https://docs.python.org/3/library/functions.html#int)) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


使用 MetaSchedule 调整 TIR。：param work_dir：工作目录：type work_dir：str：param max_trials_gloabl：允许调整的最大试验次数：type max_trials_gloabl：int。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.Normalize() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将 Relax IR 转换为范式，即表达式被规范化（没有嵌套，因此 AST 在 ANF 中），并且表达式的所有 struct_info_都可用。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.NormalizeGlobalVar() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


可能重命名 IRModule 中的 GlobalVar 以确保这些属性：


1.（不变）首先确保每个公共函数与其“global_symbol”属性同名；2.为确保1.，我们可能需要重命名名称冲突的私有函数；3.最后，每个 GlobalVar 的名称在 IRModule 中都是唯一的。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## *class* tvm.relax.transform.PatternCheckContext


检查函数 FusionPattern.check 的输入。
* **参数：**
   * **matched_expr** (*Expr*)：与 FusionPattern.pattern 匹配的表达式。
   * **annotated_expr** (*Mapping**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Expr****]*)***：** 包含 FusionPattern.annotation_patterns 中的子模式匹配的所有表达式的映射。
   * **matched_bindings** (*Mapping**[***[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***,*** ***Expr****]*)：从变量映射到其值。它包含由 FuseOpsByPattern 融合的绑定变量。
   * **var_usages** (*Mapping**[***[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***,*** ***Sequence****[*[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*]]*)：将变量定义映射到一组用途的映射。它包含函数中使用的所有变量。
   * **value_to_bound_var** (*Mapping**[****Expr,*[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*]*)：将值映射到其绑定变量。匹配的表达式后没有变量。

## tvm.relax.transform.RealizeVDevice() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


传播虚拟设备信息。
* **返回：ret**：注册通行证。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.RemovePurityChecking() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


在模块中的所有纯函数上激活 relax.force_pure，并将所有纯覆盖操作解包为正常版本。


这实际上意味着将不再有纯度跟踪，这对于低级代码生成有用。
* **返回：ret**：通行证。
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

:::note

应在 ToNonDataflow() 之后使用。

:::

## tvm.relax.transform.RemoveUnusedOutputs() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


从内部函数中删除未使用的输出。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.RemoveUnusedParameters() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


删除内部函数未使用的参数。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.ReorderPermuteDimsAfterConcat()


将 concat(permute_dims(A), permute_dims(B))重新排序为 permute_dims(concat(A,B))。


用于优化 CombineParallelMatmul 之后的计算。优化后的 nn.Linear 实现的模式会查找 matmul(activations, permute_dims(weights)) 。 CombineParallelMatmul 之后 ，matmul(activations, concat(permute_dims(A), permute_dims(B)))不再符合该模式。将其重新排列为 matmul(activations, permute_dims(concat(A,B)))即可恢复模式匹配。
* **返回：ret**：相应的 pass。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpasss)。

## tvm.relax.transform.ReorderTakeAfterMatmul()


将 matmul(x, take(weights, indices))重新排序为 take(matmul(x,weights),indices)。


对于优化 LoRA 计算很有用，其中可以将多个 LoRA 批处理在一起。
* **返回：ret**：相应的 pass。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)

## tvm.relax.transform.RewriteCUDAGraph() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


重写一个 Relax 模块，用于使用 CUDA 计算图执行。此过程识别出可以使用 CUDA 计算图执行的区域，并将其提升到新函数中，以便在运行时捕获计算图。
* **返回：ret**：重写 cuda graph 的注册通道。
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.RewriteDataflowReshape() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将所有类似 reshape 的 call_tir 转换为虚拟机 reshape 操作符调用。虚拟机 reshape 操作符调用将在运行时进一步简化为 CreateView 操作，而不是执行真正的数据复制。这里的“类似 reshape”包括 reshape、expand_dims、flatten 等。


注意：仅在数据流块中操作。可能需要先调用 ConvertToDataflow 。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.RunCodegen(*target_options:*[dict](https://docs.python.org/3/library/stdtypes.html#dict)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *entry_functions:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


使用带注释的代码生成和全局符号生成运行时::模块。
* **参数：**
   * **target_options** (*Optional[*[dict](https://docs.python.org/3/library/stdtypes.html#dict)*]*)：目标名称和编译选项对。
   * **entry_functions** (*Optional**[****List**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***]****]*)：开始的入口函数集。
* **返回：ret**：用于删除未使用函数的注册过程。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.SplitCallTIRByPattern(*patterns:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*]*, *fcodegen:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将 PrimFunc 拆分为两部分：第一部分是 TIR PrimFunc，它是与某个模式匹配，第二部分是原始 PrimFunc 的其余部分。它将调用 fcodegen 生成匹配模式的代码，并将其替换为 ExternFunc 调用。
* **参数：**
   * **patterns** (*List[*[PrimFunc](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)*]*)：要匹配的模式列表。
   * **fcodegen** (*Callable*[****[**List****[**MatchResult****]**],List**[****Object**]****]*)：用于生成匹配模式的代码的函数。
* **返回：ret**：用于拆分 call_tir 的注册通道。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.SplitLayoutRewritePreproc() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将 TIR 布局重写拆分为多个 TIR 函数。此过程用于 meta_schedule 调整后的预打包权重。
* **返回：ret**：用于分割 TIR 布局重写的注册过程。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)

## tvm.relax.transform.StaticPlanBlockMemory() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


静态内存规划在 BindingBlock 级别进行。该过程将尽力重用已分配的内存，以减少已分配内存的总量。


该阶段通过 TIR 变量上限注释的方式“支持”动态形状。我们可以选择为 Relax 函数添加“tir_var_upper_bound”属性注释。该属性值是一个从字符串到整数的字典，用于将 TIR 变量的名称表示为 TIR 变量的上限值。注意：为清晰起见，注释的上限属性仅适用于函数签名中的 TIR 变量。


例如，我们可以用 来注释一个 Relax 函数 。这意味着函数签名中名为“n”的变量的最大值将有上界 1024。在内存规划时，我们将使用 1024 作为它的值。`R.func_attr({"tir_var_upper_bound": {"n": 1024}})`
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.ToMixedPrecision(*out_dtype='float32'*, *fp16*input_names: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


自动混合精度通道。目前，该通道假设输入模块仅为 fp32，并会自动将某些操作的 fp32 转换为 fp16。


注意：主要在数据流块内操作。可能需要先调用 ConvertToDataflow 。
* **参数：**
   * **out_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：gemm/conv 的输出数据类型，即累加器的数据类型。
   * **fp16*input_names****(*List[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：函数参数的名称，其数据类型应为 fp16。函数签名将相应更改。
* **返回：ret**：混合精度的注册通道。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.ToNonDataflow() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将所有数据流结构转换为非数据流版本。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.TopologicalSort(*order='depth-first'*, *direction='from-inputs'*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


按指定顺序对 Relax.Dataflow 块中的绑定进行排序。
* **参数：**
   * **order** ([str](https://docs.python.org/3/library/stdtypes.html#str))：绑定的发出顺序。允许的值为“深度优先”和“广度优先”。
   * **direciton** ([str](https://docs.python.org/3/library/stdtypes.html#str))：排序应该执行的方向。允许的值是“from-input”和“from-outputs”。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.UpdateParamStructInfo(*sinfo_func:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*[[*[Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*],*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*|*[None](https://docs.python.org/3/library/constants.html#None)*]*)


更新参数的结构信息。


更新参数的结构体信息。内部绑定和函数返回类型将使用 Relax 的结构体推断规则进行更新。结构体推断产生的错误将传递给用户。
* **参数：sinfo_func** (*Callable**[****[*[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*]**,*** ***Optional****[*[StructInfo](/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)*]]*)：一个函数，它会为每个函数参数调用一次，并返回更新后的结构体信息。如果函数返回 None，则表示参数未被修改。
* **返回：ret**：相应的 pass。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.UpdateVDevice(*new_vdevice:*[VDevice](/docs/api-reference/python-api/tvm-ir#class-tvmirvdevicetargetnone-vdevice_idint-0-memory_scopestr-global), *index:*[int](https://docs.python.org/3/library/functions.html#int)) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


更新虚拟设备。
* **参数：**
   * **new_vdevice** ([tvm.ir.VDevice](/docs/api-reference/python-api/tvm-ir#class-tvmirvdevicetargetnone-vdevice_idint-0-memory_scopestr-global))：新的虚拟设备。
   * **index** ([int](https://docs.python.org/3/library/functions.html#int))：设备索引指示将执行更新的设备。
* **返回：ret**：修改虚拟设备的注册通行证。
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.VMBuiltinLower() → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpasss)


将通用内在函数降低为 VM 内在函数。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.VMShapeLower(***, *emit_err_ctx: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


降低符号形状和参数并匹配结构信息匹配。
* **参数：emit_err_ctx** (*Optional[*[bool](https://docs.python.org/3/library/functions.html#bool)*]*)：是否发出错误上下文字符串，可出于测试目的关闭。
* **返回：ret。**
* **返回类型：**[tvm.ir.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.relax.transform.dataflowblock_pass(*pass_func=None*, *opt_level=None*, *name=None*, *required=None*, *traceable=False*) → [Callable](https://docs.python.org/3/library/typing.html#typing.Callable) | [DataflowBlockPass](/docs/api-reference/python-api/tvm-relax-transform#class-tvmrelaxtransformdataflowblockpass)


装饰数据流块传递。


当提供 pass_func 时，此函数返回回调。否则，它返回使用给定优化函数创建的数据流块 pass。
* **参数：**
   * **pass_func** (*Optional**[****Callable**[****(*[DataflowBlock](/docs/api-reference/python-api/tvm-relax#classtvmrelaxdataflowblockbindingslistbindingspanspannonenone)*,Module,*[PassContext](/docs/api-reference/python-api/tvm-transform#classtvmtransformpasscontextopt_level2required_passnonedisabled_passnoneinstrumentsnoneconfignone)***)*** ***-> DataflowBlock****]]*)：转换函数或类。
   * **opt_level** ([int](https://docs.python.org/3/library/functions.html#int))：此数据流块 pass 的优化级别。
   * **name** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)**：** 数据流块 pass 的名称。名称可以为空。在这种情况下，优化函数的名称将用作 pass 名称。
   * **required** (*Optional**[****List**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***]****]*)：数据流块 pass 所依赖的 pass 列表。
   * **traceable** (*Boolean*)：布尔变量，表示数据流块 pass 是否可追踪。
* **返回：create_dataflowblock_pass**：如果未提供 pass_func，则返回一个装饰器；否则返回装饰后的结果。返回的装饰器根据输入有两种行为：装饰 pass 函数时，将返回一个新的 DataflowBlockPass。装饰类类型时，将返回一个新的 DataflowBlockPass 类。
* **返回类型：** Union[Callable, [DataflowBlockPass](/docs/api-reference/python-api/tvm-relax-transform#class-tvmrelaxtransformdataflowblockpass)]。


**示例**


下面的代码块装饰了一个数据流块传递类。

```python
@relax.transform.dataflowblock_pass(opt_level=1)
class TestReplaceBinding:
    # 一个简单的测试函数，用于将第一个 VarBinding 替换为另一个。
    
    def **init**(self):
        # 创建一个新的 VarBinding
        m, n = tir.Var("m", "int64"), tir.Var("n", "int64")
        lv0 = relax.Var("lv1", relax.TensorStructInfo([m, n], "float32"))
        val = relax.const(np.random.rand(24, 56))
        self.new_binding = relax.VarBinding(lv0, val)

    def transform_dataflowblock(self, block, mod, ctx):
        # 仅用于演示
        # Replace the first binding in the DataflowBlock
        new_bindings = [self.new_binding, block.bindings[1]]
        new_block = relax.expr.DataflowBlock(new_bindings, block.span)
        return new_block

@tvm.script.ir_module
class InputMod:
    @R.function
    def f1(x: Tensor[(m, n), "float32"]):
        with relax.dataflow():
            lv0 = relax.multiply(x, x)
            gv0 = relax.add(x, x)
            relax.output(gv0)
        return gv0
# block_pass 现在是一个特殊的 pass，它会将每个块的第一个绑定替换为常量值绑定

block_pass = TestReplaceBinding()
# 现在 InputMod 的每个 DataflowBlock 中的第一个绑定
# 都被 new_binding 替换了
updated_mod = block_pass(InputMod)
```


以下代码通过装饰用户定义的转换函数来创建数据流块传递。

```python
@relax.transform.dataflowblock_pass(opt_level=2)
def transform(block, mod, ctx):
    # 我在这里的转换
    return block

block_pass = transform
assert isinstance(block_pass, relax.transform.DataflowBlockPass)
assert block_pass.info.opt_level == 2

# 给定一个模块 m，可以按如下方式调用优化：
updated_mod = block_pass(m)
# 现在 transform 应该已经应用到提供的模块 m 中的每个 DataflowBlock
# 并且更新后的模块将被返回
```
## tvm.relax.transform.function_pass(*pass_func=None*, *opt_level=None*, *name=None*, *required=None*, *traceable=False*) → [Callable](https://docs.python.org/3/library/typing.html#typing.Callable) | [FunctionPass](/docs/api-reference/python-api/tvm-relax-transform#class-tvmrelaxtransformfunctionpass)

装饰一个函数 pass。


当提供 pass_func 时，此函数返回一个回调。否则，它返回使用给定优化函数创建的函数传递。
* **参数：**
   * **pass_func** (*Optional**[****Callable**[****(*[Function](/docs/api-reference/python-api/tvm-relax#classtvmrelaxfunctionparamslistvarbodyrelaxexprret_struct_infostructinfononenoneis_pureboolnonetrueattrsdictattrsnonenonespanspannonenone)*,Module**,*** [PassContext](h/docs/api-reference/python-api/tvm-transform#classtvmtransformpasscontextopt_level2required_passnonedisabled_passnoneinstrumentsnoneconfignone)***)*** ***-> Function****]]*)：转换函数或类。
   * **opt_level** ([int](https://docs.python.org/3/library/functions.html#int))：此函数 pass 的优化级别。
   * **name** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)*：* 函数 pass 的名称。名称可以为空。在这种情况下，优化函数的名称将用作 pass 名称。
   * **required** (*Optional**[****List**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***]****]*)**：** 函数 pass 所依赖的 pass 列表。
   * **traceable** (*Boolean*)：布尔变量，表示函数 pass 是否可追踪。
* **返回：create_function_pass**：如果未提供 pass_func，则返回一个装饰器；否则返回装饰后的结果。返回的装饰器根据输入有两种行为：装饰一个 pass 函数时，将返回一个新的 FunctionPass。装饰一个类类型时，将返回一个新的 FunctionPass 类。
* **返回类型：** Union[Callable, [FunctionPass](/docs/api-reference/python-api/tvm-relax-transform#class-tvmrelaxtransformfunctionpass)]。


**示例**


下面的代码块装饰了一个函数 pass 类。

```python
@relax.transform.function_pass(opt_level=1)
class TestReplaceFunc:
    def **init**(self, new_func):
        self.new_func = new_func

    def transform_function(self, func, mod, ctx):
# 仅用于演示
# 将 transform 函数改为 new_func

        return self.new_func

@R.function
def f1(x: Tensor[(m, n), "float32"]):
    return x

@tvm.script.ir_module
class InputMod:
    @R.function
    def f2(x: Tensor[(m, n), "float32"]):
        gv0 = relax.add(x, x)
        return gv0
# fpass 现在是一个特殊的 pass，它会将每个函数都替换为 f1
fpass = TestReplaceFunc(f1)
# 现在 InputMod 中的每个函数都被 f1 替换了
updated_mod = fpass(InputMod)
```


以下代码通过装饰用户定义的转换函数来创建函数 pass。

```python
@relax.transform.function_pass(opt_level=2)
def transform(func, mod, ctx):
    # 我在这里的转换操作
    return func

function_pass = transform
assert isinstance(function_pass, relax.transform.FunctionPass)
assert function_pass.info.opt_level == 2

# 给定一个模块 m，可以按如下方式调用优化：
updated_mod = function_pass(m)
# 现在 transform 应该已经应用到提供的模块 m 中的每个函数
# 并且更新后的模块将被返回
```
## *class* tvm.relax.transform.AttachExternModules(*extern_modules:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[ExternModule]*)


将变量边界附加到每个 Relax 函数，这主要有助于内存规划。

## *class* tvm.relax.transform.FastMathTransform(args*, ***kwargs*)


通过将昂贵的非线性函数转换为快速但近似的函数。

## *class* tvm.relax.transform.FuseTransposeMatmul(args*, ***kwargs*)


融合转置 + matmul 的编译器过程。

## *class* tvm.relax.transform.IPCAllReduceRewrite(*allreduce_strategy:*[int](https://docs.python.org/3/library/functions.html#int))

将 all-reduce 操作重写为具有 IPC 内存的定制 all-reduce 实现。

## *class* tvm.relax.transform.LazyTransformParams(*fget_item='get_item'*, *fset_item='set_item'*, *extra_get_item_params=None*, *extra_set_item_params=None*)


将 transform_params 函数转换为惰性版本。（按需将输入加载到内存中，并在最后一次使用后立即释放。）


注意：在此过程之前应调用 ToNonDataflow() 和 RemovePurityTracking()。
* **参数：**
   * **fget_item** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* get_item 函数的名称。
   * **fset_item** ([str](https://docs.python.org/3/library/stdtypes.html#str))：set_item 函数的名称。
   * **extra_get_item_params** ([list](https://docs.python.org/3/library/stdtypes.html#list)*of*[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone))*：*get_item 函数除 index 之外的参数。[给定](https://docs.python.org/3/library/stdtypes.html#list)*（在 Python v3.13 中）的参数将放在 index 之前。例如，如果 extra_get_item_params 为 [param1, param2]，则 pass 将生成 call_packed(fget_item, [param1, param2, index])。
   * **extra_set_item_params** ([list](https://docs.python.org/3/library/stdtypes.html#list)*of*[relax.Var](/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone))：set_item 函数除 index 和 value 之外的参数。给定的参数将放置在 index 和 value 之前。例如，如果 extra_set_item_params 为 [param1, param2]，则 pass 将生成 call_packed(fset_item, [param1, param2, index, value])。

## *class* tvm.relax.transform.LowerGPUIPCAllocStorage(args*, ***kwargs*)


降低 IPC 内存上的存储/张量分配。

## *class* tvm.relax.transform.OptimizeLayoutTransform


通过删除由 AlterOpImpl 过程引入的冗余变换布局操作符。

## *class* tvm.relax.transform.FoldBatchnormToConv2D


将 Batchnorm 与其之前的 Conv2D 融合：此优化是 FoldScaleAxis 的一个特例，它将缩放折叠到 conv2d 权重中。当 FoldScaleAcis 增强以支持这种情况时，可以移除此步骤。

## *class* tvm.relax.transform.RemoveRedundantReshape


 转换过程删除多余的重塑操作符。

