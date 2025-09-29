---

title: tvm.relax.分析

---


Relax IR 分析。

## **class*tvm.relax.analysis.BaseCheckResult(*value*)**

返回细粒度基础检查的结果。


:::Note

基本检查带有细粒度的失败级别。
* FAIL_L0：lhs 和 rhs 根本没有交集。
* FAIL_L1：我们通过查看静态信息来了解失败情况。
* FAIL_L2：由于未知的符号变量关系导致失败。

:::

## **tvm.relax.analysis.all_global_vars(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)**]**


返回表达式 expr 中的所有全局变量。:param expr: 表达式。:type expr: Expr
* **返回：** **ret**：expr 中的全局变量列表，按 DFS 后顺序排列。
* **返回类型：** List[[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)]

## **tvm.relax.analysis.all_vars(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**]**


返回表达式 expr 中的所有（局部）变量。:param expr: 表达式。:type expr: Expr
* **返回：** **ret**：expr 中的变量列表，按 DFS 后顺序排列。
* **返回类型：** List[[relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)]。

## **tvm.relax.analysis.bound_vars(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**]**


返回表达式 expr 中的所有绑定变量。绑定变量是指所有在 expr 中声明的变量。它们仅在该 expr 内部有意义，并且只能在该 expr 中使用。:param expr: 表达式。:type expr: Expr
* **返回：** **ret**：expr 中绑定变量的列表，按 DFS 后顺序排列。
* **返回类型：** List[[relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)]。

## **tvm.relax.analysis.collect_non_negative_expressions(*sinfo:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)**]**


收集非负面语境中使用的 TIR 表达式。


获取在使用结构体信息的上下文中为非负的 TIR 变量。例如，任何用作张量形状的表达式。


返回的列表已去重：每个 TIR 表达式最多出现一次。列表的顺序与结构体 info 中的出现顺序一致。
* **参数：sinfo** ()：要分析的结构信息对象。
* **返回：** **ret**：可以从 StructInfo 定义的 TIR 变量列表。
* **返回类型：** List[[tir.Var](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Var)]。

## **tvm.relax.analysis.computable_at_compile_time(*func:***[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**]**


收集可以在编译时计算值的变量。


如果函数具有 kNumInput 属性，则前 kNumInput 个参数在运行时提供，而所有剩余参数可能在编译时已知。此实用程序会收集所有仅直接或间接依赖于编译时已知参数的变量绑定。
* **参数：func** ()：要分析的 Relax.Function。
* **返回：** **ret**：可以在编译时计算的变量集，按照它们在函数中出现的顺序排列。
* **返回类型：** List[[relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)]

## **tvm.relax.analysis.contains_impure_call(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*own_name:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)***|***[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)→**[bool](https://docs.python.org/3/library/functions.html#bool)

检查给定的表达式（可能是函数体）是否包含任何不纯调用。
* **参数：**
   * **expr** (*Expr*)：要检查的表达式。如果 expr 是一个函数，则检查其函数体。
   * **own_name** ( *or*  *(*optional**))：对于递归函数，分析可以忽略自调用来检查纯度。
* **返回：** **ret**：如果存在不纯调用（调用可能具有可见副作用的函数），则为真。
* **返回类型：** [bool](https://docs.python.org/3/library/functions.html#bool)


依赖于 StructInfo 注解，因此请确保模块已先进行规范化。此外，嵌套*函数中的非纯调用并不意味着*外部表达式包含非纯调用——只有当嵌套函数*稍后被调用时*，才意味着外部表达式包含非纯调用。


## **tvm.relax.analysis.definable_tir_vars_in_struct_info(*sinfo:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Var)**]**


从输入结构体信息中获取可能定义的 TIR 变量。返回的列表已去重 - 每个 TIR 变量最多出现一次。
* **参数：sinfo** ()：要分析的结构信息对象。
* **返回：** **ret**：可以从 StructInfo 定义的 TIR 变量列表。
* **返回类型：** List[[tir.Var](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Var)]。

## **tvm.relax.analysis.defined_symbolic_vars(*func:***[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**]**

获取输入函数中定义的 TIR 变量。返回的列表已去重 - 每个 TIR 变量最多出现一次。
* **参数：func** ()：要分析的函数对象。
* **返回：** **ret**：输入函数中定义的符号变量列表。
* **返回类型：** List[[relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)]。

## **tvm.relax.analysis.derive_call_ret_struct_info(*func_sinfo:***[FuncStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.FuncStructInfo)**,*call:***[Call](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Call)**,*ctx:***[BlockBuilder](https://tvm.apache.org/docs/reference/api/python/relax/block_builder.html#tvm.relax.block_builder.BlockBuilder)**)→**[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)


从输入中获取调用的 ret 值结构信息。
* **参数：**
   * **func_sinfo** ()：调用的函数签名。
   * **call** ()：调用表达式。
   * **ctx** (tvm.relax.BlockBuilder)：上下文块构建器。
* **返回：** **ret**：派生的返回值结构信息。
* **返回类型：** [StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)。

:::Note

这是一个内部派生函数，在这种情况下，call.op 字段被忽略，并且派生仅依赖于 func_sinfo。

:::

## **tvm.relax.analysis.detect_recursion(*mod:***[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)**]]**


查找模块中所有递归或相互递归函数集。


如果两个或多个函数之间存在循环引用，则它们相互递归。例如，如果有两个函数 A 和 B，则如果 A 调用 B，B 调用 A，则它们相互递归。另一种情况是，有三个函数 A、B 和 C，其中 A 调用 B，B 调用 C，C 调用 A。


（请注意，函数不必互相调用即可互相引用。例如，如果一个函数返回另一个函数，那么即使没有调用，它仍然是一个可能递归的引用。）


如果一个函数只是递归的，而不是与任何其他函数相互递归的，那么它将被报告为一个单独的组。
* **参数：mod** (*The module*)。
* **返回：** **ret**：列表中的每个成员都是一个全局函数列表，这些函数相互递归引用。如果一个函数只是递归函数，且不与其他函数互相递归，那么它将是此列表中的单例函数。
* **返回类型：** List[List[[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)]]。

## **tvm.relax.analysis.erase_to_well_defined(*sinfo:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**,*shape_var_map:***[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)***[***[Var](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Var)***,***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*var_map:***[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)***[***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)***,***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)***] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*)→**[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)


将 sinfo 擦除为明确定义的形式。


此函数删除了 StructInfo 对给定映射中未定义的形状和变量的依赖。
* **参数：**
   * **sinfo** ()：输入结构信息。
   * **shape_var_map** (Dict[, tir.PrimExpr])：指定定义的形状变量及其应映射到的值。
   * **var_map** (Dict[, Expr])：指定定义的变量及其应映射到的值。
* **返回：** **ret**：相应的已擦除结构信息。
* **返回类型：** [StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)。

## **tvm.relax.analysis.free_symbolic_vars(*func:***[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**]**


获取输入函数中使用但未定义的 TIR 变量。返回的列表已去重：每个 TIR 变量最多出现一次。
* **参数：func** ()：要分析的函数对象。
* **返回：** **ret**：在输入函数中使用但未定义的符号变量列表。
* **返回类型：** List[[relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)]。

## **tvm.relax.analysis.free_vars(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**]**


返回表达式 expr 中的所有自由变量。自由变量是指表达式中未受 VarBinding 或函数参数绑定的变量。:param expr: 表达式。:type expr: Expr
* **返回：** **ret**：expr 中的自由变量列表，按 DFS 后顺序排列。
* **返回类型：** List[[relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)]。

## **tvm.relax.analysis.get_static_type(*sinfo:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**)→**[Type](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Type)


从 StructInfo 中获取相应的静态类型。
* **参数：sinfo** ()：输入结构信息。
* **返回：** **ret：**相应的静态类型。
* **返回类型：** [Type](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Type)。

## **tvm.relax.analysis.get_var2val(*func:***[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)**)→**[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)**[**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**,**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**]**[](https://tvm.apache.org/docs/reference/api/python/relax/analysis.html#tvm.relax.analysis.get_var2val)


为函数中的每个变量获取从 Relax.Var 到 Expr 的映射。
* **参数：func** ()：要分析的输入函数。
* **返回：** A mapping from relax.Var to Expr。
* **返回类型：** Dict[[relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var), Expr]。

## **tvm.relax.analysis.has_reshape_pattern(*func:***[PrimFunc](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.PrimFunc)**)→**[bool](https://docs.python.org/3/library/functions.html#bool)


检查给定的 PrimFunc 是否本质上执行了重塑操作。重塑操作还包括 expand_dims、squeeze、flatten 等。


这里允许的重塑模式是：例如，假设操作是 B[l_0, l_1, ..., l_b] = A[r_0, r_1, ..., r_a]，我们检查是否可以证明缓冲区 B 下的 l_0, ..., l_b 的扁平化索引等于缓冲区 A 下的 r_0, ..., r_a 的扁平化索引。
* **参数：func** ()：要检查的函数。
* **返回：** **ret：** 一个布尔值，指示给定的 PrimFunc 是否正在进行重塑。
* **返回类型：** [bool](https://docs.python.org/3/library/functions.html#bool)。


**注意**


根据上面的描述，返回的结果只能是假阴性，而不能是假阳性，因为只要我们无法证明相等性，我们就会返回 false。此属性保证了该函数的安全性。

## **tvm.relax.analysis.name_to_binding(*func:***[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)**)→**[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Binding](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Binding)**]]**


返回从变量名到其绑定的映射。

## tvm.relax.analysis.post_order_visit(expr, fvisit)


按照 DFS 后排序的顺序递归访问 ir 节点，并应用 fvisit。保证每个节点只被访问一次。
* **参数：**
   * **expr** (*tvm.relax.Expr*)：输入表达式。
   * **fvisit** (*function*)**：** 要应用的访问者函数。

## **tvm.relax.analysis.remove_all_unused(*func:***[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)**)→**[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)


它会删除：1. DataflowBlock 中未使用的本地 VarBindings。2. 函数中未使用的 DataflowBlocks。
* **参数：func** ()：要分析的输入函数。


注意


对于 IRModule-wise DCE，使用 py:func: tvm.relax.transform.DeadCodeElimination。
* **返回：** 移除了未使用变量的函数。
* **返回类型：** [Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)

## **tvm.relax.analysis.struct_info_base_check(*base:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**,*derived:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**)→**[BaseCheckResult](https://tvm.apache.org/docs/reference/api/python/relax/analysis.html#tvm.relax.analysis.BaseCheckResult)


运行基础检查来查看基础是否包含派生。
* **参数：**
   * **base** ()：基本结构信息。
   * **derived** ()：派生的结构信息。
* **返回：** **ret：** 派生的返回值结构信息。
* **返回类型：** [StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)。


## **tvm.relax.analysis.struct_info_lca(*lhs:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**,*rhs:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**)→**[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)

将两个结构信息统一到它们最近的祖先。
* **参数：**
   * **lhs** ()[：](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)左操作数。
   * **rhs** ()**：** 右操作数。
* **返回：** **ret：** 相应的 lca 结果。
* **返回类型：** [StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)。

## **tvm.relax.analysis.suggest_layout_transforms(*func:***[PrimFunc](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.PrimFunc)**,*write_buffer_transforms:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[IndexMap](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.IndexMap)***|***[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)***]*)→**[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)**[**[Block](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Block)**,**[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)**[**[Block](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Block)**|**[Buffer](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Buffer)**,**[IndexMap](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.IndexMap)**]]**


建议在 PrimFunc 中对块和缓冲区进行布局转换。
* **参数：**
   * **func** ()：将对其执行分析并建议转换的 PrimFunc。
   * **write_buffer_transforms** (List[**Union**[, Callable])：输出缓冲区的布局转换列表。布局转换的数量必须与 PrimFunc 的输出数量匹配。
* **返回：** **ret**： func 中每个块的建议转换。对于每个块，返回的值是从对象（块或缓冲区）到其索引映射转换的映射。
* **返回类型：** Dict[[Block](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Block), Dict[Union[[Block](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Block), [Buffer](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Buffer)], [IndexMap](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.IndexMap)]]

## **tvm.relax.analysis.tir_vars_in_struct_info(*sinfo:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Var)**]**


获取输入结构体信息中出现的 TIR 变量。返回的列表已去重 - 每个 TIR 变量最多出现一次。
* **参数：sinfo** ()**：** 要分析的结构信息对象。
* **返回：** **ret：** 输入结构信息中出现的 TIR 变量列表。
* **返回类型：** List[[tir.Var](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Var)]

## **tvm.relax.analysis.tir_vars_in_struct_info(*sinfo:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Var)**]**


分析数据流块中的变量 use-def 链。
* **参数：dfb** ()[：](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)要分析的数据流块。
* **返回：** A mapping from variable definition to its uses.
* **返回类型：** Dict[[relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var), List[[relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)]]

## **tvm.relax.analysis.udchain(*dfb:***[DataflowBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataflowBlock)**)→**[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)**[**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**,**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**]]**

检查 IRModule 是否格式正确。
* **参数：**
   * **obj** (*Union[**tvm.IRModule**,*]*)：输入 IRModule 或 Relax.Function。
   * **check_struct_info** ()**：**一个布尔标志，指示是否检查属性“每个 Expr 必须具有定义的结构信息”。
* **返回：** **ret**：如果 IRModule 格式正确则为 True，否则为 False。
* **返回类型：** [bool](https://docs.python.org/3/library/functions.html#bool)


默认情况下，始终会检查结构信息。只有在测试用例中，check_struct_info 才可能为 false，这样其他格式良好的需求才能得到良好的测试，而不会因为缺少结构信息而受阻。


## **tvm.relax.analysis.well_formed(*obj:***[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)***|***[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)**,*check_struct_info:***[bool](https://docs.python.org/3/library/functions.html#bool)***= True*)→**[bool](https://docs.python.org/3/library/functions.html#bool)


用于估算 IRModule 中 Relax 函数内存使用情况的分析函数。估算内容包括内存规划前后需要分配的总内存大小。


结果可能会被高估，因为估算是静态的，不考虑控制流（例如“if”和跨函数调用）。它只是简单地累加每个 alloc_tensor 和 alloc_storage 的大小。


该分析函数用于演示内存规划的效果。
* **参数：mod** (*Union*[*, ]*)：需要分析其内部函数的输入 IRModule。如果输入是函数，我们将用 IRModule 将其包装起来，并将函数命名为“main”。
* **返回：** **est**：估计信息，以字符串的形式。
* **返回类型：** [str](https://docs.python.org/3/library/stdtypes.html#str)


**注意**



我们将「relax.memory.alloc_tensor/storage」视为内存规划产生的结果。

