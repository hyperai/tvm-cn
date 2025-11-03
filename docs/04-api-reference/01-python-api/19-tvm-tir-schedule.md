---

title: tvm.tir.schedule

---


TensorIR 调度 API 的命名空间。

## *class* tvm.tir.schedule.BlockScope


每个块（block）的 sref 树中都有一个对应的对象，它用于跟踪块之间的生产者-消费者依赖关系。


术语表：
* 块作用域：sref 树的一个连续子树，以每个块 sref 为根，其组成部分包括：
   * **作用域根（scope root）**：一个块 sref。
   * **内部 sref（internal srefs）**：循环 sref。
   * **作用域叶子（scope leaves）**：块 sref。
* 子块：作用域根或特定内部 sref 下的作用域叶块。

### get_deps_by_src(*block:*[StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Dependency](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduledependency)]


获取所有 src 为目标“block”的依赖项。
* **参数：block** ([StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)) ：查询的区块。
* **返回：blocks**：依赖项。
* **返回类型：** List[[Dependency](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduledependency)]。

### get_deps_by_dst(*block:*[StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Dependency](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduledependency)]


获取所有 dst 为目标块的依赖项。
* **参数：block** ([StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref))：查询的区块。
* **返回：blocks**：依赖项。
* **返回类型：** List[[Dependency](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduledependency)]。

## *class* tvm.tir.schedule.Dependency


一个元组 (src, dst, kind)，表示特定类型的依赖关系。例如，(A, B, kRAW) 表示区块 B 依赖于区块 A，依赖关系类型为「写后读」，即区块 B 读取区块 A 写入的结果。
* **参数：**
   * **src** ([StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref))：依赖关系的来源。
   * **dst** ([StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref))：依赖关系的目标。
   * **kind** ([DepKind](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduledepkindvalue))：依赖类型。

## *class* tvm.tir.schedule.DepKind(*value*)

依赖类型。

### RAW

写后读依赖性。
* **类型：** int = 0。

### WAW


写后写依赖性。
* **类型：** int = 1。

### WAR


读后写依赖项。TensorIR 目前不支持。
* **类型：** int = 2。

### OPAQUE


不透明依赖。
* **类型：** int = 3

## *class* tvm.tir.schedule.StmtSRef


引用 TensorIR 中可调度元素的对象，又名“sref”。


**术语表：**
* **块 sref**：指向一个 TensorIR 块的 `StmtSref`。 
* **循环 sref**：指向一个 TensorIR for 循环的 `StmtSRef`。 
* **父 sref**：某个 sref 的父节点是指向其在 TensorIR AST 中最近可调度语句的祖先块/循环 sref。 
* **根 sref**：指向根块的 sref。每个 sref 都恰好有一个父 sref，根 sref 除外。 
* **sref 树**：由 sref 的父子关系形成的树，由 TensorIR AST 唯一确定。 



### *property* stmt*:*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)*|*[For](/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none)*|*[None](https://docs.python.org/3/library/constants.html#None)


对象引用的块/语句。

### *property* parent*:*[StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)*|*[None](https://docs.python.org/3/library/constants.html#None)


父 sref。

### *static* inline_mark() → [StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)


一个特殊的 StmtSRef，它不指向 AST 中的任何 stmt，仅作为「标记」来提示 compute-at 执行 compute-inline 的工作。

### *static* root_mark() → [StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)


一个特殊的 StmtSRef，它不指向 AST 中的任何 stmt，仅作为「标记」来提示 compute-at 不执行任何操作。

## *class* tvm.tir.schedule.Instruction(*kind:*[InstructionKind](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkind), *inputs:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*]*, *attrs:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*]*, *outputs:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*]*)


每个调度指令对应一个调度原语。

### kind

指令类型。
* **类型：**[InstructionKind](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkind)

### inputs 


指令的输入随机变量，每个元素的类型可以是以下之一： - BlockRV - LoopRV - ExprRV - float - int - str - None。
* **类型：** List[INPUT_RV_TYPE]。

### attrs


指令的属性。与运算符的属性类似，指令的属性是指令所需的任意常量元数据。例如，在 GetBlock 中要检索的块的名称。
* **类型：** List[ATTR_TYPE]。

### outputs


指令的输出随机变量，每个元素的类型可以是以下之一： - BlockRV - LoopRV - ExprRV，仅限原子变量，不会是常量或复合 PrimExpr。
* **类型：** List[OUTPUT_RV_TYPE]。

## *class* tvm.tir.schedule.InstructionKind

一种指令类型（InstructionKind），例如 Split、Reorder 等。除了名称之外，每种指令类型都有自己的属性，包括：

1. 一个布尔值，表示该指令是否为纯指令（pure），即不会改变调度状态。 

2. 一个函数对象（functor），用于将该指令应用到 TensorIR 调度中。 

3. 一个函数对象，用于将该指令转换为 Python 语法的语句。 

4. 一个函数对象，用于将指令的属性序列化为 JSON。 

5. 一个函数对象，用于从 JSON 反序列化指令的属性。 


与 `tvm.ir.op` 不同，`InstructionKind` 不支持非结构化属性，主要是因为目前没有需要添加其他属性的使用场景。

### name


指令种类名称。
* **类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)。

:::Note

目前函子属性尚未在 Python 端公开。

:::


### *property* is_pure *:* [bool](https://docs.python.org/3/library/functions.html#bool)

指示指令是否为纯指令，即单独删除该指令不会改变调度状态。例如，指令 GetBlock 是纯指令，因为它不会改变任何内容；而 ComputeInline 则不是纯指令，因为删除该指令会导致不同的调度结果。
* **返回：pure**：布尔标志，指示指令是否为纯指令。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

### *static* get(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [InstructionKind](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkind)


使用其名称检索 InstructionKind。
* **参数：name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：InstructionKind 的注册名称。
* **返回：kind**：检索到的 InstructionKind。
* **返回类型：**[InstructionKind](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkind)。

## *class* tvm.tir.schedule.BlockRV


引用一个块的随机变量。

## tvm.tir.schedule.ExprRV


别名`PrimExpr`。

## *class* tvm.tir.schedule.LoopRV

引用循环的随机变量。

## *class* tvm.tir.schedule.Schedule(*mod:*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*|*[IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone), *, *seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *debug_mask: [str](https://docs.python.org/3/library/stdtypes.html#str) | [int](https://docs.python.org/3/library/functions.html#int) = 'none'*, *error_render_level: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'detail'*, *enable_check: [bool](https://docs.python.org/3/library/functions.html#bool) = True*)


面向用户的调度类。


调度是一组转换操作，用于改变计算顺序，但保留计算语义。以下是一些调度示例：1) 将一个循环拆分为两个；2) 重新排序两个循环；3) 将特定缓冲区的计算内联到其消费者中。


调度类存储了辅助信息，以便正确、有效地进行调度。


### *property* mod*:*[IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)


返回正在调度的模块的 AST。

### *property* state:[ScheduleState](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulestatemodprimfuncirmodule--debug_mask-str--int--none-enable_check-bool--true)


返回当前调度类中的 ScheduleState。

### *property* trace:[Trace](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduletraceinstslistinstruction-decisionsdictinstructionany)*|*[None](https://docs.python.org/3/library/constants.html#None)

返回内部维护的调度程序执行跟踪。

### *property* func_working_on*:*[GlobalVar](/docs/api-reference/python-api/tvm-ir#class-tvmirglobalvarname_hintstr-type_annottypenone-none)*|*[None](https://docs.python.org/3/library/constants.html#None)


返回当前正在执行的调度函数的 GlobalVar。

### work_on(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)


指示调度执行 IRModule 中的一项函数。


默认情况下，调度会处理名为“main”的函数；如果 IRModule 中只有一个函数，则只会处理该函数。如果 IRModule 中有多个函数，且它们的名称都不为“main”，则用户必须调用此方法明确指定要处理的函数。


如果未指定其 func_name，此糖函数将指导 GetBlock 方法。
* **参数：func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：要处理的函数的名称。

### copy() → [Schedule](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true)


返回调度的副本，包括状态和符号表，* 保证 * 1) SRef 树完全重建；* 2) 正在调度的 IRModule 未受影响；* 3) 所有随机变量在副本中有效，指向相应的 sref * 重建。
* **返回：copy**：调度的新副本。
* **返回类型：**[Schedule](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true)。

### seed(*seed:*[int](https://docs.python.org/3/library/functions.html#int)) → [None](https://docs.python.org/3/library/constants.html#None)


播下随机数。
* **参数：seed** ([int](https://docs.python.org/3/library/functions.html#int))：新的随机种子，如果使用设备随机则为 –1，否则为非负数。

### fork_seed() → [int](https://docs.python.org/3/library/functions.html#int)

返回分叉的随机状态作为新调度的种子。
* **返回：seed**：分叉的随机状态，与当前随机状态不同。
* **返回类型：**[int](https://docs.python.org/3/library/functions.html#int)。

### show(args, **kwargs*) → [None](https://docs.python.org/3/library/constants.html#None)


打印高亮 TVM 脚本的简写。


所有参数都转发给底层的 Module.show 和 Trace.show 方法。

### get(*rand_var_or_sref:*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*|*[StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref) → [int](https://docs.python.org/3/library/functions.html#int) | [Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none) | [For](/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none) | [None](https://docs.python.org/3/library/constants.html#None)


返回： - BlockRV 计算结果对应的 Block； - LoopRV 计算结果对应的 For； - ExprRV 计算结果对应的整数； - 块 sref 指向的对应 Block； - 循环 sref 指向的对应 For；
* **参数：rand_var_or_sref** (*Union**[****ExprRV,*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*,*[StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)*]*)：需要评估的随机变量 / sref。
* **返回：result**：相应的结果。
* **返回类型：** Optional[Union[[int](https://docs.python.org/3/library/functions.html#int), [Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none), [For](/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none)]]。

### get_sref(*rand_var_or_stmt:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*|*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)*|*[For](/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none)) → [StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref) | [None](https://docs.python.org/3/library/constants.html#None)


返回与给定对象对应的 sref，支持对象类型：1) LoopRV  2) BlockRV  3) Block  4) For。
* **参数：rand_var_or_stmt** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*,*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)*,*[For](/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none)*]*)：需要评估的随机变量 / sref。
* **返回：result**：相应的结果。
* **返回类型：** Optional[[StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)]。

### remove_rv(*rand_var:*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) → [None](https://docs.python.org/3/library/constants.html#None)


从符号表中删除一个随机变量。
* **参数：rand_var** (*Union**[***[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)***,[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)***,*** ***ExprRV***])：要删除的随机变量。

### sample_categorical(*candidates:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *probs:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[float](https://docs.python.org/3/library/functions.html#float)*]*, *decision:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


给定概率分布，对整数进行采样。
* **参数：**
   * **candidates** (*List[*[int](https://docs.python.org/3/library/functions.html#int)*]*)：需要抽样的候选人。
   * **probs** (*List[*[float](https://docs.python.org/3/library/functions.html#float)*]*)： 每个候选人的概率。
   * *decision**（*可选**[ [int](https://docs.python.org/3/library/functions.html#int)*]*）：采样决策（如果有）。
* **返回：result**：从候选人中抽取的随机变量。
* **返回类型：** ExprRV。

### sample_perfect_tile(*loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *n:*[int](https://docs.python.org/3/library/functions.html#int), *max_innermost_factor:*[int](https://docs.python.org/3/library/functions.html#int)*= 16*, *decision:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)] 


对因素进行采样以完美平铺特定循环。
* **参数：**
   * **loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv))： 要平铺的循环。
   * **n** ([int](https://docs.python.org/3/library/functions.html#int)) ：要采样的图块数量。
   * **max_innermost_factor** ([int](https://docs.python.org/3/library/functions.html#int))：最内层循环中允许采样的最大图块大小。
   * **decision**（*可选[List[*[int](https://docs.python.org/3/library/functions.html#int)*]]*）：抽样决策（如果有）。
* **返回：result**：长度为n 的列表，即随机采样的完美图块尺寸。
* **返回类型：** List[ExprRV]。

### sample_partitioned_tile(*loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *n:*[int](https://docs.python.org/3/library/functions.html#int), *partition_pos:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *innerpart_factor:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *decision:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)]


将因子采样到特定循环的分区块中。
* **参数：**
   * **loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) ：要平铺的循环。
   * **n** ([int](https://docs.python.org/3/library/functions.html#int)) ：要采样的图块数量。
   * **partition_pos** ([int](https://docs.python.org/3/library/functions.html#int)) ：将 tile 分成两部分的位置。
   * **innerpart_factor** ([int](https://docs.python.org/3/library/functions.html#int)) ：第二部分的因子。
   * **decision**（*可选[List[*[int](https://docs.python.org/3/library/functions.html#int)*]]*）：抽样决策（如果有）。
* **返回：result**：长度为n 的列表，即采样的随机分区图块大小。
* **返回类型：** List[ExprRV]。

### sample_compute_location(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *decision:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)


对给定块的计算位置进行采样。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：需要采样计算位置的块。
   * *decision**（*可选**[ [int](https://docs.python.org/3/library/functions.html#int)*]*）：采样决策。
* **返回：result**：计算输入块的采样循环。
* **返回类型：**[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)。

### get_block(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)


通过名称检索特定函数中的块。


默认情况下，如果未指定 func_name，则调度程序将在当前正在“处理”的函数中搜索块。要切换要处理的函数，请在调用此方法之前使用 work_on。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：块的名称。
   *  func_name (可选[[str](https://docs.python.org/3/library/stdtypes.html#str)]= None ) ：函数的名称。
* **返回：block**：如果存在 0 个或多个具有特定名称的块，则会引发检索到的块 IndexError。
* **返回类型：**[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)。

### get_loops(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)]


获取块在其范围内的父循环，从外到内。
* **参数：block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：查询块。
* **返回：loops**：给定块范围内的循环列表，从外到内。
* **返回类型：** List[[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)]。

### get_child_blocks(*block_or_loop:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]


获取特定块/循环的叶块。
* **参数：block_or_loop** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*]*)：查询块/循环。
* **返回：blocks**：特定块/循环内的叶块列表。
* **返回类型：** List[[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)]。

### get_producers(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]


获取特定区块的生产者。
* **参数：block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：查询中的块。
* **返回：producers**：给定区块的生产者列表。
* **返回类型：** List[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]。

### get_consumers(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]


获取特定区块的消费者。
* **参数：block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：查询中的块。
* **返回：consumers**：给定区块的消费者列表。
* **返回类型：** List[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]。

### get_output_blocks(*scope_block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]


获取给定范围内的输出块列表输出块是至少有一个缓冲区被写入但未在 PrimFunc 内分配的块
* **参数：scope_block** (*Union**[***[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)***,[str](https://docs.python.org/3/library/stdtypes.html#str)***]****,*)：收集输出块的范围块。
* **返回：output_blocks**：写入某个输出缓冲区的所有块的列表。
* **返回类型：** List[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]。

### merge(loops:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*]*) → [LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)


将循环列表合并为一个。其 LCA 下的循环要求：1) 作用域相同。2) 不能包含注释或线程绑定。3) 从 0 开始，且具有相同的范围和相同的嵌套深度。4) 从目标循环到其 LCA，内层循环必须是外层循环的唯一子循环。
* **参数：*loops****(*List[*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*]*)：要合并的循环。
* **返回：fused_loop**：合并后的新循环。
* **返回类型：**[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)。


**示例**


在应用合并之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_merge(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0
```


创建调度并执行融合：

```python
sch = tir.Schedule(before_fuse)
i1, _ = sch.get_loops(sch.get_block("B"))
i2, _ = sch.get_loops(sch.get_block("C"))
sch.merge(i1, i2)
print(sch.mod["main"].script())
```


应用融合后，IR 变为：

```python
@T.prim_func
def after_fuse(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))
    # 这两个循环被合并为一个。
    for i_m in range(128):
        for j in range(128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i_m, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * T.float32(2)
        for j in range(128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i_m, j])
                T.reads(A[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = A[vi, vj] * T.float32(2)
```
### fuse(loops:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*]*, *preserve_unit_iters:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)[](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true.fuse) 


将一系列连续的循环融合为一个。它要求：1) 循环不能包含注释或线程绑定。2) 第 (i+1) 个循环必须是第 i 个循环的唯一子循环。3) 所有循环必须以 0 开头。4) 待融合循环的域不能依赖于另一个待融合循环。
* **参数：*loops****(*List*[*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*]*)：要融合的循环。
* **返回：fused_loop**：融合后的新循环。
* **返回类型：**[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)。


**示例**


在应用 fuse 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_fuse(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度并执行融合：

```python
sch = tir.Schedule(before_fuse)
i, j = sch.get_loops(sch.get_block("B"))
sch.fuse(i, j)
print(sch.mod["main"].script())
```


应用融合后，IR 变为：

```python
@T.prim_func
def after_fuse(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    # 这两个循环被融合为一个。
    for i_j_fused in T.serial(0, 16384):
        with T.block("B"):
            vi = T.axis.S(128, T.floordiv(i_j_fused, 128))
            vj = T.axis.S(128, T.floormod(i_j_fused, 128))
            B[vi, vj] = A[vi, vj] * 2.0
```
### split(*loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *factors:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*]*, *preserve_unit_iters:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *disable_predication:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)]


将循环拆分为一系列连续的循环。它要求：1）循环不能包含注解或线程绑定。2）循环必须从 0 开始。可以添加谓词以确保循环总数保持不变。在因子中，最多只有一个因子可以为 None，这将被自动推断出来。
   * **参数：loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) ：要拆分的循环。
   * **factors** (*List**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,******ExprRV****,None**]***]*)：分裂因子潜在输入为： – None – ExprRV – 正常整数。
   * **preserve_unit_iters** ([bool](https://docs.python.org/3/library/functions.html#bool))： 是否在块绑定中保留单元迭代器。
   * **disable_predication** ([bool](https://docs.python.org/3/library/functions.html#bool))：如果启用，则不创建用于保护循环的谓词。当使用可扩展因子进行拆分时，如果调度编写器知道这些因子可以被循环界限整除，则此功能非常有用。


警告：如果不小心使用，启用此功能可能会导致错误的代码生成。
* **返回：split_loops**：分割后的新循环。
* **返回类型：** List[[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)]。


**示例**


在分割之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_split(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```
创建计划并执行拆分：
```python
sch = tir.Schedule(before_split)
i, j = sch.get_loops(sch.get_block("B"))
sch.split(i, factors=[2, 64])
print(sch.mod["main"].script())
```


应用拆分后，IR 变为：

```python
@T.prim_func
def after_split(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    # the original loop is split into 2 loops
    for i0, i1, j in T.grid(2, 64, 128):
        with T.block("B"):
            vi = T.axis.S(128, i0 * 64 + i1)
            vj = T.axis.S(128, j)
            B[vi, vj] = A[vi, vj] * 2.0
```
### loop_partition(*loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *factors:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[None](https://docs.python.org/3/library/constants.html#None)*]*, *preserve_unit_iters:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)]

将循环划分为一系列连续的循环。它要求：1）循环不能包含注解或线程绑定。可以添加谓词以确保循环总数保持不变。在因子中，最多只有一个因子可以为 None，这将被自动推断出来。
* **参数：**
   * **loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) ：要分割的循环。
   * **factors** (*List**[****Union**[***[int](https://docs.python.org/3/library/functions.html#int)***,***ExprRV****,None*]*]*) ：分区因子潜在输入包括： – None – ExprRV – 正常整数。
   * **preserve_unit_iters** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：是否在块绑定中保留单元迭代器。
* **返回：partition_loops**：分区后的新循环。
* **返回类型：** List[[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)]。


**示例**


在分区之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_partition(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度并进行分区：

```python
sch = tir.Schedule(before_partition)
i, j = sch.get_loops(sch.get_block("B"))
sch.partition(i, factors=[2, 64])
print(sch.mod["main"].script())
```


应用分区后，IR 变为：

```python
def after_partition(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    # 原来的循环被划分为 3 个循环。
    with T.block("root"):
        T.reads()
        T.writes()
        with T.block("B_i_common"):
            T.reads()
            T.writes()
            with T.block("B_i0_partition"):
                T.reads()
                T.writes()
                for i0, j in T.grid(2, 128):
                    with T.block("B_i0"):
                        vi, vj = T.axis.remap("SS", [i0, j])
                        T.reads(A[0:2, 0:128])
                        T.writes(B[0:2, 0:128])
                        B[vi, vj] = A[vi, vj] * T.float32(2)
            with T.block("B_i1_partition"):
                T.reads()
                T.writes()
                for i1 in range(2, 66):
                    for j in range(128):
                        with T.block("B_i1"):
                            vi, vj = T.axis.remap("SS", [i1, j])
                            T.reads(A[2:66, 0:128])
                            T.writes(B[2:66, 0:128])
                            B[vi, vj] = A[vi, vj] * T.float32(2)
            with T.block("B_partition_2"):
                T.reads()
                T.writes()
                for i2 in range(66, 128):
                    for j in range(128):
                        with T.block("B_i2"):
                            vi, vj = T.axis.remap("SS", [i2, j])
                            T.reads(A[66:128, 0:128])
                            T.writes(B[66:128, 0:128])
                            B[vi, vj] = A[vi, vj] * T.float32(2)
```
### reorder(ordered_loops:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*]*) → [None](https://docs.python.org/3/library/constants.html#None)


重新排序循环列表。它不要求循环是连续的。它要求：1）循环位于同一链中。这意味着：循环可以排序为 [l_1, l_2, … , l_n]，其中 l_i 是 l_{i+1} 的祖先，并且 l_1 和 l_n 之间只有单分支循环（这也表明它们在同一范围内）。2）重新排序后，外层循环的定义域不能依赖于任何内层循环。3）对于循环嵌套下的每个块，其块绑定必须是仿射的，并且块变量必须是数据并行的或缩减的。4）参数中不允许有重复的循环。
* **参数：*ordered_loops****(*List[*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*]*)：新顺序的循环。


**示例**


重新排序之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_reorder(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度并重新排序：

```python
sch = tir.Schedule(before_reorder)
i, j = sch.get_loops(sch.get_block("B"))
sch.reorder(j, i)
print(sch.mod["main"].script())
```


应用重新排序后，IR 变为：

```python
@T.prim_func
def after_reorder(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    # Here j and i are reordered
    for j, i in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```
### reorder_block_iter_var(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv), *new_order:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*) → [None](https://docs.python.org/3/library/constants.html#None)


对给定块内的 itervars 进行重新排序。
* **参数：**
   * **block** ([BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv))：要转换的块。
   * **new_order** (*List[*[int](https://docs.python.org/3/library/functions.html#int)*]*) ：新的块 itervar 顺序。


**示例**


在 reorder_block_iter_var 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def matmul(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
```


创建调度并执行 reorder_block_iter_var：

```python
sch = tir.Schedule(matmul)
C = sch.get_block("C")
sch.reorder_block_iter_var(C, [2, 1, 0])
```


应用 reorder_block_iter_var 后，IR 变为：

```python
@T.prim_func
def matmul_after_reorder_block_iter_var(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    for i, j, k in T.grid(128, 128, 128):
        with T.block("C"):
            vk, vj, vi = T.axis.remap("RSS", [k, j, i])
            T.reads(A[vi, vk], B[vj, vk])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
```


:::info 另见

`reorder`

:::


### add_unit_loop(*block_or_loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*|*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)) → [LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)

在特定块或循环之上创建一个新的单元循环。
* **参数：block_or_loop** (*Union[*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*,*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*]*)：在其上创建新循环的区块。
* **返回：new_loop**：新的单元循环。
* **返回类型：**[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)。


**示例**


在 add_unit_loop 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_add_unit_loop(
    A: T.Buffer((), "int32"),
    B: T.Buffer((), "int32"),
    C: T.Buffer((), "int32"),
) -> None:
    with T.block("C"):
        vi = T.axis.spatial(1, 0)
        C[()] = A[()] + B[()]
```


创建调度并执行添加单元循环：

```python
sch = tir.Schedule(before_add_unit_loop)
sch.add_unit_loop(sch.get_block("C"))
print(sch.mod["main"].script())
```


应用添加单位循环后，IR 变为：

```python
@T.prim_func
def after_add_unit_loop(
    A: T.Buffer((), "int32"),
    B: T.Buffer((), "int32"),
    C: T.Buffer((), "int32"),
) -> None:
    for u in T.serial(1):
        with T.block("C"):
            vi = T.axis.spatial(1, 0)
            C[()] = A[()] + B[()]
```
### parallel(*loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) → [None](https://docs.python.org/3/library/constants.html#None)


并行化输入循环。它要求：1）循环所在的作用域块应具有阶段流水线属性；2）循环下的所有块都是完整块或归约块，并且具有仿射绑定；3）对于循环下的每个块，循环只能包含在数据并行块迭代器的绑定中。
* **参数：loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv))：要并行化的循环。


**示例**


并行之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_parallel(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度并并行执行：

```python
sch = tir.Schedule(before_parallel)
i, j = sch.get_loops(sch.get_block("B"))
sch.parallel(i)
```


应用并行后，IR 变为：

```python
@T.prim_func
def after_parallel(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i in T.parallel(0, 128):
        for j in T.serial(0, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
```
### vectorize(*loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) → [None](https://docs.python.org/3/library/constants.html#None)


对输入循环进行矢量化。它要求：1）循环所在的作用域块应具有阶段流水线属性；2）循环下的所有块均为完整块或归约块，且具有仿射绑定；3）对于循环下的每个块，该循环只能包含在数据并行块迭代器的绑定中。
* **参数：loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv))：要矢量化的循环。


**示例**


在矢量化之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_vectorize(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度并进行矢量化：

```python
sch = tir.Schedule(before_vectorize)
i, j = sch.get_loops(sch.get_block("B"))
sch.vectorize(j)
```


应用矢量化后，IR 变为：

```python
@T.prim_func
def after_vectorize(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i in T.serial(0, 128):
        for j in T.vectorized(0, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
```
### bind(*loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *thread_axis:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)


将输入循环绑定到给定的线程轴。它要求：1) 循环所在的作用域块应具有 stage-pipeline 属性；2) 循环下的所有块均为完整块或归约块，且具有仿射绑定；3) 对于循环下的每个块，如果线程轴以“threadIdx”开头，则该循环只能包含在数据并行块迭代器和归约块迭代器的绑定中。否则，该循环只能包含在数据并行块迭代器的绑定中。
* **参数：**
   * **loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) ：要绑定到线程轴的循环。
   * **thread_axis** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：要绑定到循环的线程轴。可能的候选：– blockIdx.x/y/z – threadIdx.x/y/z – vthread.x/y/z – vthread（这是一个将被弃用的遗留行为。请改用 vthread.x/y/z 。）。


**示例**


绑定之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_bind(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度并进行绑定：

```python
sch = tir.Schedule(before_bind)
i, j = sch.get_loops(sch.get_block("B"))
sch.bind(i, "blockIdx.x")
sch.bind(j, "threadIdx.x")
```


应用绑定后，IR 变为：

```python
@T.prim_func
def after_bind(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i in T.thread_binding(0, 128, thread = "blockIdx.x"):
        for j in T.thread_binding(0, 128, thread = "threadIdx.x"):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
```
### unroll(*loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) → [None](https://docs.python.org/3/library/constants.html#None)


展开输入循环。它不需要。
* **参数：loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv))：要展开的循环。


**示例**


展开之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_unroll(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度并展开：

```python
sch = tir.Schedule(before_unroll)
i, j = sch.get_loops(sch.get_block("B"))
sch.unroll(i)
```


应用展开后，IR 变为：

```python
@T.prim_func
def after_unroll(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i in T.unroll(0, 128):
        for j in T.serial(0, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
```
### cache_read(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *read_buffer_index:*[int](https://docs.python.org/3/library/functions.html#int)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *storage_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str), *consumer_blocks:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)


创建一个块，用于将缓冲区读入读缓存。它需要：


1. 最多有一个块在范围内写入缓冲区。


2. 范围块具有阶段管道属性。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：目标缓冲区的消费者块。
   * **buffer** (*Union[*[int](https://docs.python.org/3/library/functions.html#int)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*]*) ：块读取区域中缓冲区的索引、块中读取缓冲区的唯一名称或块读取区域内的 Buffer 对象。
   * **storage_scope** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：目标存储范围。
   * **consumer_blocks** (*Optional**[****List**[****Union**[***[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)***,*** [str](https://docs.python.org/3/library/stdtypes.html#str)***]****]]*) ：可选，包含需要从缓存读取数据的消费者列表。若未指定，则所有消费者都将使用缓存。
* **返回：cached_block**：缓存阶段的块。
* **返回类型：**[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)。


**示例**


在 cache_read 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_cache_read(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度和 cache_read：

```python
sch = tir.Schedule(before_cache_read)
block_b = sch.get_block("B")
sch.cache_read(block_b, 0, "local")
print(sch.mod["main"].script())
```


应用 cache_read 后，IR 变为：

```python
@T.prim_func
def after_cache_read(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    A_local = T.alloc_buffer((128, 128), scope="local")
    for i, j in T.grid(128, 128):
        with T.block("A_local"):
            vi, vj = T.axis.remap("SS", [i, j])
            A_local[vi, vj] = A[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A_local[vi, vj] * 2.0
```
### cache_write(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *write_buffer_index:*[int](https://docs.python.org/3/library/functions.html#int)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *storage_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str), *consumer_blocks:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)


创建一个块，用于将缓冲区读入写缓存。它需要：


1. 只有一个块在范围内写入缓冲区。


2. 范围块具有阶段管道属性。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：目标缓冲区的生产者块。
   * **write_buffer_index** ([int](https://docs.python.org/3/library/functions.html#int)) ：块的写入区域中缓冲区的索引、块中写入缓冲区的唯一名称或块写入区域内的缓冲区对象。
   * **storage_scope** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：目标存储范围。
   * **consumer_blocks** (*Optional**[****List**[****Union**[***[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)***,*** [str](https://docs.python.org/3/library/stdtypes.html#str)***]****]]*) ：可选列表，列出应直接从缓存读取数据的消费者。如果未指定，则所有消费者都将从原始缓冲区读取数据。
* **返回：cached_block**：缓存阶段的块。
* **返回类型：**[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)。


**示例**


在 cache_write 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_cache_write(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度和 cache_write：

```python
sch = tir.Schedule(before_cache_write)
block_b = sch.get_block("B")
sch.cache_write(block_b, 0, "local")
print(sch.mod["main"].script())
```


应用 cache_write 后，IR 变为：

```python
@T.prim_func
def after_cache_write(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    B_local = T.alloc_buffer((128, 128), scope="local")
    for i, j in T.grid(128, 128):
        with T.block("A_local"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_local[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = B_local[vi, vj]
```
### reindex_cache_read(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *read_buffer_index:*[int](https://docs.python.org/3/library/functions.html#int), *storage_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str), *index_map:*[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*|*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)) → [BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)


创建一个块，使用索引图指定的自定义索引将缓冲区读取到读取缓存中。缓冲区的读取区域必须是单个点。


缓存阶段块遵循块中循环和块迭代变量的原始顺序。如果某个块迭代变量未出现在缓冲区访问区域中，则该块迭代变量及其对应的循环变量将被省略。然后，用户可以使用 transform_block_layout 原语来重新排序缓存读/写块的块迭代变量及其周围的循环。


与 cache_read 不同，reindex_cache_read 仅支持单个消费者， 当有多个消费者时请使用 cache_read 。
* **参数：**
   * **block** ([BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv))：目标缓冲区的消费者块。
   * **read_buffer_index** ([int](https://docs.python.org/3/library/functions.html#int)) ：块读取区域中缓冲区的索引。
   * **storage_scope** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：目标存储范围。
   * **index_map** (*Union*[***[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)***,****Callable****]) ：用户定义索引来访问分配的缓存缓冲区，从块迭代变量映射。
* **返回：cached_block**：缓存阶段的块。
* **返回类型：**[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)。


**示例**


在 reindex_cache_read 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_reindex_cache_read(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度和 reindex_cache_read：

```python
sch = tir.Schedule(before_cache_read)
block_b = sch.get_block("B")
sch.reindex_cache_read(block_b, 0, "local", lambda vi, vj: (vj, vi))
print(sch.mod["main"].script())
```


应用 reindex_cache_read 后，IR 变为：

```python
@T.prim_func
def after_reindex_cache_read(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    A_local = T.alloc_buffer((128, 128), scope="local")
    for i, j in T.grid(128, 128):
        with T.block("A_local"):
            vi, vj = T.axis.remap("SS", [i, j])
            A_local[vj, vi] = A[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A_local[vj, vi] * 2.0
```


:::info 另见

`reindex_cache_write`, `transform_block_layout`, `transform_layout`, `cache_read`, `reindex`

:::


### reindex_cache_write(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *write_buffer_index:*[int](https://docs.python.org/3/library/functions.html#int), *storage_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str), *index_map:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*|*[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)) → [BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)


创建一个块，使用索引图指定的自定义索引将缓冲区读入写缓存。缓冲区的写入区域必须是单个点。


缓存阶段块遵循块中循环和块迭代变量的原始顺序。如果某个块迭代变量未出现在缓冲区访问区域中，则该块迭代变量及其对应的循环变量将被省略。然后，用户可以使用 transform_block_layout 原语来重新排序缓存读/写块的块迭代变量及其周围的循环。


与 cache_write 不同，reindex_cache_write 仅支持单个消费者， 当有多个消费者时请使用 cache_write 。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：目标缓冲区的消费者块。
   * **write_buffer_index** ([int](https://docs.python.org/3/library/functions.html#int)) ：块写入区域中缓冲区的索引。
   * **storage_scope** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：目标存储范围。
   * **index_map** (*Union**[****Callable,*[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*]*) ：用户定义索引来访问分配的缓存缓冲区，从块迭代变量映射。
   * **consumer_blocks** (*Optional**[****List**[****Union**[***[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)***,*** [str](https://docs.python.org/3/library/stdtypes.html#str)***]****]]*)：可选列表，列出应直接从缓存读取数据的消费者。如果未指定，则所有消费者都将从原始缓冲区读取数据。
* **返回：cached_block**：缓存阶段的块。
* **返回类型：**[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)。


**示例**


在 reindex_cache_write 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_reindex_cache_write(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度和 reindex_cache_write：

```python
sch = tir.Schedule(before_cache_write)
block_b = sch.get_block("B")
sch.reindex_cache_write(block_b, 0, "local", lambda vi, vj: (vi // 2, vi % 2, vj))
print(sch.mod["main"].script())
```


应用 reindex_cache_write 后，IR 变为：

```python
@T.prim_func
def after_cache_write(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (64, 2, 128))
    B_local = T.alloc_buffer((128, 128), scope="local")
    for i, j in T.grid(128, 128):
        with T.block("A_local"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_local[vi % 2, vi // 2, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = B_local[vi % 2, vi // 2, vj]
```


:::info 另见

`reindex_cache_read`, `transform_block_layout`, `transform_layout`, `cache_write`, `reindex`

:::

### 

### cache_inplace(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *read_buffer_index:*[int](https://docs.python.org/3/library/functions.html#int)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *storage_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]


创建块，将缓冲区读写到缓存块中。它要求目标块同时读写目标缓冲区。主要用于就地操作。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：目标块对目标缓冲区进行操作。
   * **read_buffer_index** ([int](https://docs.python.org/3/library/functions.html#int)) ：块读取区域中缓冲区的索引、块中读取缓冲区的唯一名称或块读取区域内的缓冲区对象。
   * **storage_scope** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：目标存储范围。
* **返回：cached_blocks**：缓存阶段的块，首先读取缓存，然后写入缓存。
* **返回类型：** List[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]。


**示例**


在 cache_inplace 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_cache_inplace(data_io: T.Buffer((64), "int32")):
    for i0 in T.serial(1):
        with T.block("A"):
            T.reads(data_io[:64])
            T.writes(data_io[:64])
            T.evaluate(T.call_extern("call_impl", data_io.data, dtype=""))
```


创建调度和 cache_inplace：

```python
sch = tir.Schedule(before_cache_inplace)
block_a = sch.get_block("A")
sch.cache_inplace(block_a, 0, "local")
print(sch.mod["main"].script())
```


应用 cache_inplace 后，IR 变为：

```python
@T.prim_func
def cache_inplace(data_io: T.Buffer(64, "int32")) -> None:
    data_io_local = T.alloc_buffer([64], dtype="int32", scope="local")
    for i0 in T.serial(1):
        for ax0 in T.serial(64):
            with T.block("data_io_local"):
                v0 = T.axis.spatial(64, ax0)
                T.reads(data_io[v0])
                T.writes(data_io_local[v0])
                data_io_local[v0] = data_io[v0]
        with T.block("A"):
            T.reads(data_io_local[0 : 64])
            T.writes(data_io_local[0 : 64])
            T.evaluate(T.call_extern("call_impl", data_io_local.data, dtype=""))
        for ax0 in T.serial(64):
            with T.block("data_io_local"):
                v0 = T.axis.spatial(64, ax0)
                T.reads(data_io_local[v0])
                T.writes(data_io[v0])
                data_io[v0] = data_io_local[v0]
```
### cache_index(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *storage_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str), *cse_thresh:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]


创建一个块来缓存预先计算的索引以供以后使用。如果没有索引计算，则保持不变。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：目标块对目标缓冲区进行操作。
   * **storage_scope** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：缓存块的存储范围。
   * **cse_thresh** ([int](https://docs.python.org/3/library/functions.html#int)) ：确定公共子表达式的重复阈值，默认值 0 表示缓存所有索引计算。
* **返回：cached_blocks**：写入缓存缓冲区的阶段的块。
* **返回类型：** List[[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)]。


**示例**


在 cache_inplace 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def resize(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (1, 3, 40, 40))
    B = T.match_buffer(b, (1, 3, 80, 80))
    for i0, i1, i2, i3 in T.grid(1, 3, 80, 80):
        with T.block("A"):
            n, c, vi, vj = T.axis.remap("SSSS", [i0, i1, i2, i3])
            B[n, c, vi, vj] = A[n, c, vi//4 + vj//4, vj//2]
```


创建调度和 cache_index：

```python
sch = tir.Schedule(resize)
block_a = sch.get_block("A")
sch.cache_index(block_a, "global", 1)
print(sch.mod["main"].script())
```


应用 cache_index 之后，IR 变为：

```python
@T.prim_func
def resize_cache_index(
    A: T.Buffer((1, 3, 40, 40), "float32"), B: T.Buffer((1, 3, 80, 80), "float32")
) -> None:
    index_var_0 = T.alloc_buffer([80, 80], dtype="int32", strides=[1])
    index_var_1 = T.alloc_buffer([80], dtype="int32", strides=[1])
    for ax0, ax1 in T.grid(80, 80):
        with T.block("index_0"):
            v0 = T.axis.spatial(80, ax0)
            v1 = T.axis.spatial(80, ax1)
            T.reads()
            T.writes(index_var_0[v0, v1])
            index_var_0[v0, v1] = v0 // 4 + v1 // 4
    for ax0 in T.serial(80):
        with T.block("index_1"):
            v0 = T.axis.spatial(80, ax0)
            T.reads()
            T.writes(index_var_1[v0])
            index_var_1[v0] = v0 // 2
    for i0, i1, i2, i3 in T.grid(1, 3, 80, 80):
        with T.block("A"):
            n, c, vi, vj = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[n, c, vi // 4 + vj // 4, vj // 2])
            T.writes(B[n, c, vi, vj])
            B[n, c, vi, vj] = A[n, c, index_var_0[vi, vj], index_var_1[vj]]
```
### reindex(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *buffer:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[int](https://docs.python.org/3/library/functions.html#int)*] |*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)) → [BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)


创建一个块，用于将缓冲区读/写到读/写缓存中，并进行重建索引。缓存的布局将与读/写缓冲区的块的迭代器相同。它要求：1) 只有一个块读取/写入目标缓冲区；2) 块中只有一个缓冲区加载/存储该缓冲区；
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：访问目标缓冲区的块。如果是字符串，则必须唯一地标识一个块。
   *  **缓冲区**（*联合[*[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*，*[int](https://docs.python.org/3/library/functions.html#int)*]，*[缓冲区](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*，*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*）：需要转换的缓冲区，或者如何识别需要转换的缓冲区的规范。如果缓冲区是的元组`(str,int)`，则第一项应该是“读”或“写”，第二项是块的读或写区域的索引。如果 buffer 是一个字符串，则它是缓冲区的名称，该缓冲区必须存在于块的读/写操作中。此外，块的读/写操作中不能包含多个同名缓冲区。 如果缓冲区是缓冲区对象，它必须存在于块的读/写范围内。
* **返回：reindex_block**：重新索引阶段的块。
* **返回类型：**[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)。


**示例**


在重新索引之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_reindex(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vj, vi] * 2.0
```


创建调度并重新索引：

```python
sch = tir.Schedule(before_reindex)
block = sch.get_block("B")
sch.reindex(block, ("read", 0))
```


应用重新索引后，IR 变为：

```python
@T.prim_func
def after_reindex(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32")
) -> None:
    A_reindex = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("A_reindex"):
            vi, vj = T.axis.remap("SS", [i, j])
            A_reindex[vi, vj] = A[vj, vi]
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A_reindex[vi, vj] * 2.0
```
### compute_at(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *preserve_unit_loops:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *index:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*) → [None](https://docs.python.org/3/library/constants.html#None)


Compute-At。将生产者块移动到特定循环下，并重新生成由该块引发的循环，以使生产者块生成的缓冲区能够覆盖其消费者块在给定循环下消耗的区域。它需要：


1. block 和 loop 处于同一作用域内，loop 不是 block 的祖先

2. 范围块具有 stage-pipeline 属性。

3. 给定块所在的作用域块的子树满足紧凑数据流条件。即，作用域块的子树中的所有块必须是完整块或缩减块。

4. 该块不是相对于作用域块的输出块，即该块写入的缓冲区是在作用域块下分配的。

5. 该块的所有消费者都在给定的循环下。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：要移动的块。
   * **loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) ：移动方块的循环。
   * **preserve_unit_loops** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：是否保留范围为 1 的简单循环。
   * **index** ([int](https://docs.python.org/3/library/functions.html#int)) ：循环体子树块的块索引： – index = –1表示插入到最后一个可能的插入点； – index = –2表示插入到第一个可能的插入点； – 否则，index 是一个非负数，表示插入点。


**示例**


在 compute-at 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


创建调度并执行计算：

```python
sch = tir.Schedule(before_compute_at)
block = sch.get_block("B")
loop, _ = sch.get_loops(sch.get_block("C"))
sch.compute_at(block, loop, preserve_unit_loops=False)
print(sch.mod["main"].script())
```


应用 compute-at 之后，IR 变为：

```python
@T.prim_func
def after_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i in T.serial(0, 128):
        for j in T.serial(0, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for j in T.serial(0, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0
```
### reverse_compute_at(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *preserve_unit_loops:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *index:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*) → [None](https://docs.python.org/3/library/constants.html#None)


反向计算。将消费者块移动到特定循环下，并重新生成由该块引发的循环，以使消费者块消耗的缓冲区能够覆盖其生产者块在给定循环下生成的缓冲区。它需要：

1. block 和 loop 处于同一作用域内，loop 不是 block 的祖先。

2. 范围块具有 stage-pipeline 属性。

3. 给定块所在的作用域块的子树满足紧凑数据流条。件。即，作用域块的子树中的所有块必须是完整块或缩减块。

4. 该区块的所有生产者都在给定的循环下。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：要移动的块。
   * **loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) ：移动方块的循环。
   * **preserve_unit_loops** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：是否保留范围为 1 的简单循环。
   * **index** ([int](https://docs.python.org/3/library/functions.html#int)) ：循环体子树块的块索引： – index = –1表示插入到最后一个可能的插入点； – index = –2表示插入到第一个可能的插入点； – 否则，index 是一个非负数，表示插入点。


**示例**


在反向计算之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_reverse_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


创建调度并进行反向计算：

```python
sch = tir.Schedule(before_reverse_compute_at)
block = sch.get_block("C")
loop, _ = sch.get_loops(sch.get_block("B"))
sch.reverse_compute_at(block, loop, preserve_unit_loops=False)
print(sch.mod["main"].script())
```


应用反向计算后，IR 变为：

```python
@T.prim_func
def after_reverse_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i in T.serial(0, 128):
        for j in T.serial(0, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for j in T.serial(0, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0
```
### compute_inline(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)

将块内联到其消费者中。它需要：


1. 该块是一个完整的非根块，只产生一个缓冲区。


2. 该块一定不能是范围内的唯一叶。


3. 块的主体必须是以下形式的 BufferStore 语句，其中 LHS 的索引都是不同的原子变量，并且语句中不允许使用除这些索引变量之外的任何其他变量。`A[i, j, k, ...] = ...`。
* **参数：block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：要内联到其消费者的块。


**示例**


在 compute-inline 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_inline(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


创建调度并执行内联计算：

```python
sch = tir.Schedule(before_inline)
sch.compute_inline(sch.get_block("B"))
print(sch.mod["main"].script())
```


应用 compute-inline 后，IR 变为：

```python
@T.prim_func
def after_inline(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0
```
### reverse_compute_inline(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)


将块内联到其唯一的生产者中。它需要：


1. 该块是一个完整的非根块，它只产生和消耗一个缓冲区。

2. 该块一定不能是范围内的唯一叶。

3. 该块的唯一生产者是先读后写生产者，并且是完整的非根块。

4. 块的主体必须是以下形式的 BufferStore 语句， 其中 RHS 上每个 BufferLoad 的索引都是不同的原子变量，并且语句中不允许使用除这些索引变量之外的任何其他变量。`B[f(i, j, k, ...)] = g(i, j, k, A[i, j, k, ...] ...)`。
* **参数：block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：要内联到其生产者的块。


**示例**


在 reverse-compute-inline 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_inline(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


创建调度并执行反向内联计算：

```python
sch = tir.Schedule(before_inline)
sch.reverse_compute_inline(sch.get_block("C"))
print(sch.mod["main"].script())
```


应用反向计算内联后，IR 变为：

```python
@T.prim_func
def after_inline(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0
```
### decompose_reduction(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) → [BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)


将缩减块分解为两个独立的块。


a. 初始化块，由归约块的初始化语句翻译而来；


b. 更新块，即没有 init 语句的原始块。

初始化块插入到给定循环之前。


调度原语需要：


1. 输入块是一个缩减块。


2. 输入循环是块的祖先。


3. 输入循环不低于与 reduce block var 相关的所有循环。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：要分解的缩减块。
   * **loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) ：在其上方插入 init 块的循环。
* **返回：init_block**：初始化块。
* **返回类型：**[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)。


**示例**


在分解-还原之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_decompose(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    for i, j, k in tir.grid(128, 128, 128):
        with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
```


创建调度并使用指定的循环进行分解-缩减：

```python
sch = tir.Schedule(before_decompose)
C = sch.get_block("C")
i, j, k = sch.get_loops(C)
sch.decompose_reduction(C, i)
print(sch.mod["main"].script())
```


应用分解-还原后，IR 变为：

```python
@T.prim_func
def after_decompose(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    for i in tir.serial(128):
        for j in tir.serial(128):
            with tir.block([128, 128]) as [vi, vj]:
                C[vi, vj] = 0.0
    for i, j, k in tir.grid(128, 128, 128):
        with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
```
### rfactor(*loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *factor_axis:*[int](https://docs.python.org/3/library/functions.html#int)) → [BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)


通过指定的循环对关联约简块进行因式分解。


关联约简无法直接并行化，因为它在累加过程中可能导致竞争条件。或者，可以通过以下步骤在循环中分解约简：- 步骤 1：将约简均匀地切分为n 个独立的块，其中 n 是循环长度；- 步骤 2：分别计算每个块并将结果写入n 个中间缓冲区；- 步骤 3：将第n 个独立的缓冲区累加到结果缓冲区中。请注意，上述步骤 2 引入了并行化的机会。


RFactor 是一个调度原语，它实现了上面描述的转换：给定一个写入缓冲区 B 的块，它将分解一个范围为 n 的循环。


例如，下面的伪代码累积 B[i] = sum(A[i, : , : ])：

```python
for i in range(128):                    # 循环 i 是一个数据并行循环。
    for j in range(128):                # 循环 j 是一个归约循环。
        for k in range(128):            # 循环 k 是一个归约循环。
            B[i] = B[i] + A[i, j, k]
```


假设 RFactor 应用于最内层循环 k 且 factor_axis = 1。然后 RFactor 创建一个中间缓冲区和两个块。


1. 中间缓冲区，或称“rf-buffer”，其秩为 ndim(B) + 1，大小为 size(B) * n，其形状由 shape(B)扩展而来，具体方法是在 factor_axis 指定的位置添加一个 n 轴。例如：
* shape(B) = [1, 2, 3], factor_axis = 0 => shape(B_rf) = [n, 1, 2, 3]。
* shape(B) = [1, 2, 3], factor_axis = 1 => shape(B_rf) = [1, n, 2, 3]。
* shape(B) = [1, 2, 3], factor_axis = 2 => shape(B_rf) = [1, 2, n, 3]。
* shape(B) = [1, 2, 3], factor_axis = 3 => shape(B_rf) = [1, 2, 3, n]。


1. rfactor 块（或称“rf-block”）是指在循环 k 中写入 rf-buffer 而不进行累积的块，也就是说，循环 k 从归约循环转换为数据并行循环。在我们的示例中，rf-block 为：

```python
B_rf = np.zeros((128, 128))     # 寄存器文件（rf）缓冲区
for k in range(128):            # 循环 k 被转换为数据并行循环
    for i in range(128):        # 循环 i 是数据并行循环（保持不变）
        for j in range(128):    # 循环 j 是归约循环（保持不变）
            B_rf[i, k] = B_rf[i, k] + A[i, j, k]
```


1. 写回块（wb-block）用于将 rf 缓冲区的内容累加到结果缓冲区中。除用于累加的循环 k 外，所有归约循环均被移除。在我们的示例中，wb-block 为：

```python
for i in range(128):            # 循环 i 是数据并行循环（保持不变）
                                # 循环 j 被移除，因为它是归约循环
    for k in range(128):        # 循环 k 是归约循环（保持不变）
        B[i] = B[i] + B_rf[i, k]
```
* **参数：**
   * **loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) ：我们要执行 rfactor 的循环外部块。
   * **factor_axis** ([int](https://docs.python.org/3/library/functions.html#int)) ：新维度在新引入的 rfactor 缓冲区中的位置。
* **返回：rf_block**：计算每个切片的部分结果的块（即上图所描述的第一个块）。
* **返回类型：**[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)。


**示例**


在 rfactor 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_rfactor(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128,))
    for ii, i, j in T.grid(128, 128, 128):
    with T.block("B"):
        vii, vi, vj = T.axis.remap("SRR", [ii, i, j])
        with T.init():
            B[vii] = 0.0
        B[vii] = B[vii] + A[vii, vi, vj]
```


创建调度并执行 rfactor：

```python
sch = tir.Schedule(before_rfactor)
_, _, k = sch.get_loops(sch.get_block("B"))
sch.rfactor(k, 0)
print(sch.mod["main"].script())
```


应用 rfactor 后，IR 变为：

```python
@T.prim_func
def after_rfactor(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128, 128])
    B = T.match_buffer(b, [128])
    B_rf = T.alloc_buffer([128, 128])
    for i2, ii, i in T.grid(128, 128, 128):
        with T.block("B_rf"):
            vi2, vii, vi = T.axis.remap("SSR", [i2, ii, i])
            with T.init():
                B_rf[vi2, vii] = 0.0
            B_rf[vi2, vii] = (B_rf[vi2, vii] + A[vii, vi, vi2])
    for ii, i2 in T.grid(128, 128):
        with T.block("B"):
            vii, vi2 = T.axis.remap("SR", [ii, i2])
            with T.init():
                B[vii] = 0.0
            B[vii] = B[vii] + B_rf[vi2, vii]
```
:::Note

Rfactor 要求：1）循环只有一个子块，并且它是一个缩减块；2）循环是一个缩减循环，即循环变量只绑定到块绑定中的缩减变量；3）循环未并行化、矢量化、展开或绑定到任何线程轴；4）循环所在的块范围是分阶段管道；5）缩减块外部的最外层循环应将缩减块作为其第一个子块；6）最外层缩减循环应该只有一个子块；7）未绑定到块绑定中的任何缩减或数据并行变量的一元范围循环不应出现在某些缩减循环下；8）缩减块应该只写入一个缓冲区，并且它的 init 和 body 都是简单的 BufferStore，并且该模式注册为关联缩减器。预定义模式包括：加法、乘法、最小值和最大值；9）块顶部的每个循环不能同时绑定到数据并行和缩减块绑定； 10) `factor_axis 应在[-ndim(B) - 1, ndim(B)]范围内，其中 B 是归约块写入的缓冲区。负索引根据 numpy 约定进行规范化。

:::


### storage_align(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *buffer_index:*[int](https://docs.python.org/3/library/functions.html#int), *axis:*[int](https://docs.python.org/3/library/functions.html#int), *factor:*[int](https://docs.python.org/3/library/functions.html#int), *offset:*[int](https://docs.python.org/3/library/functions.html#int)) → [None](https://docs.python.org/3/library/constants.html#None)


设置特定维度的对齐要求，使得 stride[axis] == k * factor + offset（其中 k 为某个值）。这有助于设置内存布局，从而实现更友好的内存访问模式。例如，我们可以将对齐设置为 factor=2，offset=1，以避免在 GPU 共享内存中线程访问更高维度时发生内存库冲突。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：缓冲区的生产者块。
   * **buffer_index** ([int](https://docs.python.org/3/library/functions.html#int)) ：块写入区域中缓冲区的索引。
   * **轴**（[int](https://docs.python.org/3/library/functions.html#int)）：指定对齐的尺寸。
   * **factor** ([int](https://docs.python.org/3/library/functions.html#int)) ：对齐的因子倍数。
   * **offset** ([int](https://docs.python.org/3/library/functions.html#int)) ：所需的偏移因子。


**示例**


在 storage_align 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_storage_align(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


创建调度并执行 storage_align：

```python
sch = tir.Schedule(before_storage_align)
sch.storage_align(sch.get_block("B"), buffer_index=0, axis=0, factor=128, offset=1)
print(sch.mod["main"].script())
```


应用 storage_align 之后，IR 变为：

```python
@T.prim_func
def after_storage_align(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            T.block_attr({"buffer_dim_align": [[[0, 128, 1]]]})
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


降低传递次数后，缓冲区 B 的步幅将为 [129, 1]。

:::Note

Storage_align 要求缓冲区是通过 alloc_buffer 定义的中间缓冲区。

:::

### set_scope(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *buffer_index:*[int](https://docs.python.org/3/library/functions.html#int)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *storage_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)


设置一个缓冲区的存储范围，其中缓冲区由一个块和一个写入索引指定。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：缓冲区的生产者块。
   * **buffer_index** ([int](https://docs.python.org/3/library/functions.html#int)) ：块写入区域中缓冲区的索引。
   * **storage_scope** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：要设置的存储范围。


**示例**


在 set_scope 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_set_scope(
    A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")
) -> None:
    B = T.alloc_buffer((128, 128), dtype="float32")

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


创建调度并执行 set_scope：

```python
sch = tir.Schedule(before_set_scope)
sch.set_scope(sch.get_block("B"), buffer_index=0, storage_scope="shared")
print(sch.mod["main"].script())
```


应用 set_scope 后，IR 变为：

```python
@T.prim_func
def after_set_scope(
    A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")
) -> None:
    B_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_shared[vi, vj] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B_shared[vi, vj] + T.float32(1)
```
:::Note
set_scope 要求缓冲区是通过 alloc_buffer 定义的中间缓冲区。

:::

### unsafe_set_dtype(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *buffer_index:*[int](https://docs.python.org/3/library/functions.html#int), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)


设置缓冲区的数据类型，其中缓冲区由块和写入索引指定。


该调度原语不安全，可能会因为类型转换而改变程序的正确性，请谨慎使用。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：缓冲区的生产者块。
   * **buffer_index** ([int](https://docs.python.org/3/library/functions.html#int)) ：块写入区域中缓冲区的索引。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：要设置的数据类型。


**示例**


在 unsafe_set_dtype 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_set_dtype(
    A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")
) -> None:
    B = T.alloc_buffer((128, 128), dtype="float32")

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j]
            C[vi, vj] = B[vi, vj] + 1.0
```


创建调度并执行 unsafe_set_dtype：

```python
sch = tir.Schedule(before_set_dtype)
sch.unsafe_set_dtype("B", buffer_index=0, dtype="float16")
print(sch.mod["main"].script())
```


应用 set_dtype 后，IR 变为：

```python
@T.prim_func
def after_set_dtype(
    A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")
) -> None:
    B = T.alloc_buffer((128, 128), dtype="float16")

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = T.cast(A[vi, vj] * 2.0, "float16")
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j]
            C[vi, vj] = T.cast(B[vi, vj], "float32") + 1.0
```
:::Note

unsafe_set_dtype 要求缓冲区是通过 alloc_buffer 定义的中间缓冲区。

:::

### blockize(*target:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*]*, *preserve_unit_iters:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)


将多个块或以特定循环为根的子树转换为一个块。
* **参数：**
   * **target** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*orList[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*]*)：子树的根或指定块。
   * **preserve_unit_iters** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：是否在块绑定中保留单元迭代器。
* **返回：result**：新的区块。
* **返回类型：**[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)。


**示例**


在分块之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_blockize(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32")
) -> None:
    for i_0, j_0, i_1, j_1 in T.grid(8, 8, 16, 16):
        with T.block("B"):
            vi = T.axis.spatial(128, i_0 * 16 + i_1)
            vj = T.axis.spatial(128, j_0 * 16 + j_1)
            T.reads(A[vi, vj])
            T.writes(B[vi, vj])
            B[vi, vj] = A[vi, vj] * T.float32(2)
```


创建调度并执行 set_scope：

```python
sch = tir.Schedule(before_blockize)
B = sch.get_block("B")
_, _, i1, _ = sch.get_loops(B)
sch.blockize(i1)
print(sch.mod["main"].script())
```


应用分块化后，IR 变为：

```python
@T.prim_func
def after_blockize(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32")
)-> None:
    for i_0, j_0 in T.grid(8, 8):
        with T.block("B_o"):
            vio, vjo = T.axis.remap("SS", [i_0, j_0])
            T.reads(A[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
            T.writes(B[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
            for i_1, j_1 in T.grid(16, 16):
                with T.block("B"):
                    vi, vj = T.axis.remap("SS", [i_1, j_1])
                    T.reads(A[vio * 16 + vi, vjo * 16 + vj])
                    T.writes(B[vio * 16 + vi, vjo * 16 + vj])
                    B[vio * 16 + vi, vjo * 16 + vj] = A[vio * 16 + vi, vjo * 16 + vj]                                                                   * T.float32(2)
```
:::Note

blockize 要求给定循环下恰好有一个块，并且该块的绑定可以被从给定循环开始的循环所表示的子空间整除。

:::

### tensorize(*block_or_loop:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *tensor_intrin:*[str](https://docs.python.org/3/library/stdtypes.html#str), *preserve_unit_iters:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [None](https://docs.python.org/3/library/constants.html#None)


使用张量内在函数对循环所包含的计算进行张量化。
* **参数：**
   * **block_or_loop** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*]*) ：要张量化的循环。
   * **tensor_intrin** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：张量内部函数或张量内部函数的名称。
   * **preserve_unit_iters** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：是否在块绑定中保留单元迭代器。


**示例**


在张量化之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_tensorize(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    # 主体
    # 使用 T.block("root")
    for i_0, j_0, k_0, i_1, j_1, k_1 in T.grid(8, 8, 8, 16, 16, 16):
        with T.block("update"):
            vi = T.axis.spatial(128, i_0 * 16 + i_1)
            vj = T.axis.spatial(128, j_0 * 16 + j_1)
            vk = T.axis.reduce(128, k_0 * 16 + k_1)
            T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            T.writes(C[vi, vj])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
```
声明并注册张量内在函数：
```python
@T.prim_func
def mma_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

@T.prim_func
def mma_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(
            T.tvm_mma_sync(
                C.data,
                C.elem_offset // 256,
                A.data,
                A.elem_offset // 256,
                B.data,
                B.elem_offset // 256,
                C.data,
                C.elem_offset // 256,
                dtype="handle",
            )
        )

tir.TensorIntrin.register("test_mma_intrin", mma_desc, mma_intrin)
```


创建调度并进行张量化：

```python
sch = tir.Schedule(before_tensorize)
update = sch.get_block("update")
_, _, _, i1, _, _ = sch.get_loops(update)
sch.tensorize(i1, "test_mma_intrin")
print(sch.mod["main"].script())
```


应用张量化后，IR 变为：

```python
@T.prim_func
def after_tensorize(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    # 主体
    # 使用 T.block("root")
    for i_0, j_0, k_0 in T.grid(8, 8, 8):
        with T.block("update_o"):
            vio, vjo, vko = T.axis.remap("SSR", [i_0, j_0, k_0])
            T.reads(
                C[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16],
                A[vio * 16 : vio * 16 + 16, vko * 16 : vko * 16 + 16],
                B[vjo * 16 : vjo * 16 + 16, vko * 16 : vko * 16 + 16],
            )
            T.writes(C[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
            A_1 = T.match_buffer(
                A[vio * 16 : vio * 16 + 16, vko * 16 : vko * 16 + 16],
                [16, 16],
                dtype="float32",
                offset_factor=1,
            )
            B_1 = T.match_buffer(
                B[vjo * 16 : vjo * 16 + 16, vko * 16 : vko * 16 + 16],
                [16, 16],
                dtype="float32",
                offset_factor=1,
            )
            C_1 = T.match_buffer(
                C[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16],
                [16, 16],
                dtype="float32",
                offset_factor=1,
            )
            T.evaluate(
                T.tvm_mma_sync(
                    C_1.data,
                    C_1.elem_offset // 256,
                    A_1.data,
                    A_1.elem_offset // 256,
                    B_1.data,
                    B_1.elem_offset // 256,
                    C_1.data,
                    C_1.elem_offset // 256,
                    dtype="handle",
                )
            )
```
### annotate(*block_or_loop:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *ann_key:*[str](https://docs.python.org/3/library/stdtypes.html#str), *ann_val:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[int](https://docs.python.org/3/library/functions.html#int)*|*[float](https://docs.python.org/3/library/functions.html#float)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[int](https://docs.python.org/3/library/functions.html#int)*|*[float](https://docs.python.org/3/library/functions.html#float)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*] |*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[int](https://docs.python.org/3/library/functions.html#int)*|*[float](https://docs.python.org/3/library/functions.html#float)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[int](https://docs.python.org/3/library/functions.html#int)*|*[float](https://docs.python.org/3/library/functions.html#float)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]*) → [None](https://docs.python.org/3/library/constants.html#None)


使用键值对注释块/循环。
* **参数：**
   * **block_or_loop** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*]*) ：要注释的块/循环。
   * **ann_key** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：注释键。
   * **ann_val** (*AnnotationValueT*) ：注释值。


**示例**


在注释之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_annotate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度并做注释：

```python
sch = tir.Schedule(before_annotate)
sch.annotate(sch.get_block("B"), "ann_key", "ann_value")
print(sch.mod["main"].script())
```


应用注释后，IR 变为：

```python
@T.prim_func
def after_annotate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.block_attr({"ann_key", "ann_value"})
            B[vi, vj] = A[vi, vj] * 2.0
```
### unannotate(*block_or_loop:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv), *ann_key:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)


使用键 ann_key 取消注释块/循环的注释。
* **参数：**
   * **block_or_loop** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)*]*) ：要取消注释的块/循环。
   * **ann_key** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：注释键。


**示例**


在取消注释之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_unannotate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.block_attr({"ann_key", "ann_value"})
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度并做注释：

```python
sch = tir.Schedule(before_unannotate)
sch.unannotate(sch.get_block("B"), "ann_key")
print(sch.mod["main"].script())
```


应用 unannotate 后，IR 变为：

```python
@T.prim_func
def after_unannotate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```
### transform_layout(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *buffer:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[int](https://docs.python.org/3/library/functions.html#int)*] |*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *index_map:*[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*|*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable), *pad_value:*[int](https://docs.python.org/3/library/functions.html#int)*|*[float](https://docs.python.org/3/library/functions.html#float)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*|*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *, *assume_injective_transform: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [None](https://docs.python.org/3/library/constants.html#None)


将 IndexMap 表示的转换应用于缓冲区。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：访问目标缓冲区的块。如果是字符串，则必须唯一地标识一个块。
   * **缓冲区**（*联合[*[元组](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*，*[int](https://docs.python.org/3/library/functions.html#int)*]，*[缓冲区](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*，*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*）：需要转换的缓冲区，或者如何识别需要转换的缓冲区的规范。如果缓冲区是的元组`(str,int)`，则第一项应该是「读」或「写」，第二项是块的读或写区域的索引。如果 buffer 是一个字符串，则它是缓冲区的名称，该缓冲区必须存在于块的读/写操作中。此外，块的读/写操作中不能包含多个同名缓冲区。如果缓冲区是缓冲区对象，它必须存在于块的读/写范围内。
   * **index_map**（Union[[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map),Callable]）：要应用的转换。如果 index_map 是可调用的，并且返回的列表包含IndexMap.AXIS_SEPARATOR，则除了 TransformLayout 原语之外，还会调用 SetAxisSeparators 原语。
   * pad_value **（可选** [ Union[[int](https://docs.python.org/3/library/functions.html#int)，[float](https://docs.python.org/3/library/functions.html#float)，[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)，[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)，Callable]]）：用于转换引入的任何填充的值。如果调度包含指定缓冲区的生产者块，则填充值将尽可能作为生产者块的一部分写入，否则写入生产者块之后。否则，如果缓冲区是输入，则将插入一个注释块以声明填充值包含已知值。填充值可能不包含 BufferLoad 的实例，除非它从被转换的缓冲区加载一个值（例如，创建一个由重复元素组成的填充循环缓冲区）。

 

注意：如果应用于输入缓冲区，调用范围负责确保 pad_value 存在。代数简化、分支消除和其他优化可能会假定满足此先决条件，并可能导致返回不正确的结果。

 如果为 None，则转换可能不会引入填充。如果是 int、float 或 PrimExpr，则转换是填充中要呈现的特定值。如果是 IndexMap 或 Callable，则转换是根据转换后的索引在填充中呈现的值。
   * **假设_injective_transform** ( [bool](https://docs.python.org/3/library/functions.html#bool) ) ：如果设置为 true，调度原语将假定 index_map 是可注入的，并跳过检查映射索引的重叠部分。这对于分析未涵盖的复杂 index_map 非常有用。调用者有责任确保索引图是可注入的，否则，调度的正确性将无法保证。


**示例**


在 transform_layout 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_transform_layout(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


创建调度并执行 transform_layout：

```python
sch = tir.Schedule(before_storage_align)
sch.transform_layout(sch.get_block("B"), buffer=("write",0),
                     index_map=lambda m, n: (m // 16, n // 16, m % 16, n % 16))
print(sch.mod["main"].script())
```


应用 transform_layout 后，IR 变为：

```python
@T.prim_func
def two_elementwise_transformed_intermediate_buffer(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((8, 8, 16, 16), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi // 16, vj // 16, vi % 16, vj % 16] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi // 16, vj // 16, vi % 16, vj % 16] + 1.0
```
### transform_block_layout(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *index_map:*[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)*|*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)) → [None](https://docs.python.org/3/library/constants.html#None)


将 IndexMap 表示的转换应用于块。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：要转换的块。
   * **index_map** (*Union**[***[IndexMap](/docs/api-reference/python-api/tvm-tir#class-tvmtirindexmapinitial_indices-final_indices-inverse_index_map)***,*** ***Callable****]*) ：要应用的转换。


**示例**


在 transform_block_layout 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_transform_block_layout(
    A: T.Buffer((16, 16), "float32"),
    B: T.Buffer((16, 16), "float32")
) -> None:
    for i, j in T.grid(16, 16):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
```


创建调度并执行 transform_block_layout：

```python
sch = tir.Schedule(before_transform_block_layout)
sch.transform_block_layout(sch.get_block("B"), lambda i, j: (i * 16 + j,))
print(sch.mod["main"].script())
```


应用 transform_block_layout 后，IR 变为：

```python
@T.prim_func
def after_transform_block_layout(
    A: T.Buffer((16, 16), "float32"),
    B: T.Buffer((16, 16), "float32")
) -> None:
    for i in range(256):
        with T.block("B"):
            vi, = T.axis.remap("S", [i])
            B[vi // 16, vi % 16] = A[vi // 16, vi % 16] * 2.0
```
### set_axis_separator(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *buffer:*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[int](https://docs.python.org/3/library/functions.html#int)*] |*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), *axis_separators:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)) → [None](https://docs.python.org/3/library/constants.html#None)


设置缓冲区的轴分隔符，其中缓冲区由块和读取或写入索引指定。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：访问目标缓冲区的块。如果是字符串，则必须唯一地标识一个块。
   * **缓冲区**（*联合[*[元组](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*，*[int](https://docs.python.org/3/library/functions.html#int)*]，*[缓冲区](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*，*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*）：需要转换的缓冲区，或者如何识别需要转换的缓冲区的规范。如果缓冲区是的元组`(str,int)`，则第一项应该是「读」或「写」，第二项是块的读或写区域的索引。如果 buffer 是一个字符串，则它是缓冲区的名称，该缓冲区必须存在于块的读/写操作中。此外，块的读/写操作中不能包含多个同名缓冲区。如果缓冲区是缓冲区对象，它必须存在于块的读/写范围内。
   * **axis_separators**（*可选*[***列表[*[int](https://docs.python.org/3/library/functions.html#int)*]]*）：轴分隔符。


**示例**


在 set_axis_separator 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_set_axis_separator(
    A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")
) -> None:
    B = T.alloc_buffer((128, 128), dtype="float32")

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


创建调度并执行 set_axis_separator：

```python
sch = tir.Schedule(before_set_axis_separator)
sch.set_axis_separators(sch.get_block("B"), buffer=("write", 0),
                        axis_separators=[1])
print(sch.mod["main"].script())
```


应用 set_axis_separator 后，IR 变为：

```python
@T.prim_func
def after_set_axis_separators(
    A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")
) -> None:
    B = T.alloc_buffer([128, 128], dtype="float32", axis_separators=[1])

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + T.float32(1)
```
### decompose_padding(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) → [BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)

将填充计算模式块分解为两个独立的块。


a. 将 const pad 值填充到满写区域的块；


b. 将边界值填充到填充谓词为真的区域中的块。


填充值填充块插入到给定循环之前。


调度原语需要：


1. 输入块是一个完整的块。


2. 输入循环是块的祖先。


3. 输入块是与填充模式匹配的块。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：要分解的填充块。
   * **loop** ([LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) ：在其上方插入 pad 值填充块的循环。
* **返回：pad_value_block**：填充 const pad 值的块。
* **返回类型：**[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)。


**示例**


在分解填充之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_decompose(x: T.Buffer(128, "int32"), y: T.Buffer(140, "int32")):
    for i in range(140):
        with T.block("block"):
            vi = T.axis.remap("S", [i])
            y[vi] = T.if_then_else(vi >= 6 and vi < 134, x[vi - 6], 0, dtype="int32")
```


创建调度并使用指定的循环进行分解填充：

```python
sch = tir.Schedule(before_decompose, debug_mask="all")
block = sch.get_block("block")
sch.decompose_padding(block, sch.get_loops(block)[0])
print(sch.mod["main].script())
```


应用分解填充后，IR 变为：

```python
@T.prim_func
def after_decompose(x: T.Buffer(128, "int32"), y: T.Buffer(140, "int32")):
    for i in T.serial(140):
        with T.block("block_pad_const"):
            vi = T.axis.spatial(140, i)
            y[vi] = 0
    for i in T.serial(128):
        with T.block("block"):
            vi = T.axis.spatial(128, i)
            y[vi + 6] = x[vi]
```
### can_decompose_padding(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *loop:*[LoopRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv)) → [bool](https://docs.python.org/3/library/functions.html#bool)


检查块是否匹配填充模式并且可以分解。

### pad_einsum(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *padding:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*) → [None](https://docs.python.org/3/library/constants.html#None)


填充 Einsum 的计算。


在具有简单绑定的块上，此原语通过给定的填充因子填充块的迭代域，例如，当填充因子为 16 时，127 -> 128,132 -> 144。将生成额外的生产者和消费者填充块，以避免越界缓冲区访问。


Einsum 模式意味着缓冲区访问上的所有索引要么是常量（例如 B[0]），要么是变量（例如 B[i]），但不是由复合表达式（例如 B[i + 1]）组成。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：与 Einsum 模式匹配的块。
   * **padding** (*List[*[int](https://docs.python.org/3/library/functions.html#int)*]*) ：每个块迭代器的填充。


**示例**


在应用 pad-einsum 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_pad_einsum(
    A: T.Buffer((127, 127), "float32"),
    B: T.Buffer((127, 127), "float32"),
    C: T.Buffer((127, 127), "float32"),
) -> None:
    for i0, i1, i2 in T.grid(127, 127, 127):
        with T.block("C_shared"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]
```


创建调度并使用指定的块执行 pad-einsum：

```python
sch = tir.Schedule(before_pad_einsum, debug_mask="all")
block = sch.get_block("C_shared")
sch.pad_einsum(block, [32, 32, 32])
print(sch.mod["main"].script())
```


应用分解填充后，IR 变为：

```python
@T.prim_func
def main(
    A: T.Buffer((127, 127), "float32"),
    B: T.Buffer((127, 127), "float32"),
    C: T.Buffer((127, 127), "float32"),
):
    # 使用 T.block("root") 块
    A_pad = T.alloc_buffer((128, 128))
    B_pad = T.alloc_buffer((128, 128))
    C_pad = T.alloc_buffer((128, 128))
    for i0, i1 in T.grid(128, 128):
        with T.block("A_pad"):
            v0, v1 = T.axis.remap("SS", [i0, i1])
            A_pad[v0, v1] = T.if_then_else(
                v0 < 127 and v1 < 127,
                A[v0, v1],
                T.float32(0),
            )
    for i0, i1 in T.grid(128, 128):
        with T.block("B_pad"):
            v0, v1 = T.axis.remap("SS", [i0, i1])
            B_pad[v0, v1] = T.if_then_else(
                v0 < 127 and v1 < 127,
                B[v0, v1],
                T.float32(0),
            )
    for i0, i1, i2 in T.grid(128, 128, 128):
        with T.block("C_shared"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            with T.init():
                C_pad[i, j] = T.float32(0)
            C_pad[i, j] = C_pad[i, j] + A_pad[i, k] * B_pad[k, j]
    for i0, i1 in T.grid(127, 127):
        with T.block("C_pad"):
            v0, v1 = T.axis.remap("SS", [i0, i1])
            C[v0, v1] = C_pad[v0, v1]
```
### rolling_buffer(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), *write_buffer_index:*[int](https://docs.python.org/3/library/functions.html#int)) → [None](https://docs.python.org/3/library/constants.html#None)


通过滚动缓冲计算目标缓冲区，选择块祖先循环中出现的具有正边界重叠的最外层可滚动轴作为滚动轴，沿滚动维度折叠并循环缓冲区，并附加块谓词以避免重新计算重叠元素。它需要：


1. 该块不是输出块并且仅具有 RAW 依赖项。


2. 该缓冲区是通过 alloc_buffer 定义的中间缓冲区。


1. 缓冲区的生产者和消费者的 LCA 是一个 for 循环，通常缓冲区的生产者和消费者通过 compute_at 进行级联。


1. 缓冲区的访问区域至少有一个维度包含正边界重叠。
* **参数：**
   * **block** (*Union[*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：缓冲区的生产者块。
   * **write_buffer_index** ([int](https://docs.python.org/3/library/functions.html#int)) ：块写入区域中缓冲区的索引。


**示例**

在 rolling_buffer 之前，在 TensorIR 中，IR 是：

```python
@T.prim_func
def before_rolling_buffer(
    A: T.Buffer((12, 12), "int8"), C: T.Buffer((8, 8), "int8")
) -> None:
    # 主体
    # 使用 T.block("root") 块
    B = T.alloc_buffer([10, 10], dtype="int8")
    for i0, i1 in T.grid(2, 2):
        for ax0, ax1, ax2, ax3 in T.grid(6, 6, 3, 3):
            with T.block("B"):
                ax0_1 = T.axis.spatial(10, i0 * 4 + ax0)
                ax1_1 = T.axis.spatial(10, i1 * 4 + ax1)
                rv0, rv1 = T.axis.remap("RR", [ax2, ax3])
                B[ax0_1, ax1_1] = T.max(
                    B[ax0_1, ax1_1], A[ax0_1 + rv0, ax1_1 + rv1]
                )
        for ax0, ax1, ax2, ax3 in T.grid(4, 4, 3, 3):
            with T.block("C"):
                ax0_1 = T.axis.spatial(8, i0 * 4 + ax0)
                ax1_1 = T.axis.spatial(8, i1 * 4 + ax1)
                rv0, rv1 = T.axis.remap("RR", [ax2, ax3])
                C[ax0_1, ax1_1] = T.max(
                    C[ax0_1, ax1_1], B[ax0_1 + rv0, ax1_1 + rv1]
                )
```


创建调度并执行 rolling_buffer：

```python
sch = tir.Schedule(before_rolling_buffer)
sch.rolling_buffer(sch.get_block("B"), write_buffer_index=0)
print(sch.mod["main"].script())
```


应用 rolling_buffer 之后，IR 变为：

```python
@T.prim_func
def after_rolling_buffer(
    A: T.Buffer((12, 12), "int8"),
    C: T.Buffer((8, 8), "int8")
) -> None:
    # 主体
    # 使用 T.block("root") 块
    B = T.alloc_buffer([6, 10], dtype="int8")
    for i0, i1 in T.grid(2, 2):
        for ax0, ax1, ax2, ax3 in T.grid(6, 6, 3, 3):
            with T.block("B"):
                T.where((i0 < 1 or 2 <= ax0) and (i1 < 1 or 2 <= ax1))
                ax0_1 = T.axis.spatial(10, i0 * 4 + ax0)
                ax1_1 = T.axis.spatial(10, i1 * 4 + ax1)
                rv0, rv1 = T.axis.remap("RR", [ax2, ax3])
                B[ax0_1 % 6, ax1_1] = T.max(
                    B[ax0_1 % 6, ax1_1], A[ax0_1 + rv0, ax1_1 + rv1]
                )
        for ax0, ax1, ax2, ax3 in T.grid(4, 4, 3, 3):
            with T.block("C"):
                ax0_1 = T.axis.spatial(8, i0 * 4 + ax0)
                ax1_1 = T.axis.spatial(8, i1 * 4 + ax1)
                rv0, rv1 = T.axis.remap("RR", [ax2, ax3])
                C[ax0_1, ax1_1] = T.max(
                    C[ax0_1, ax1_1], B[ax0_1 % 6 + rv0, ax1_1 + rv1]
                )
```
:::Note

目标缓冲区的消费者块的 region_cover 属性将变为 false。

:::

### enter_postproc() → [None](https://docs.python.org/3/library/constants.html#None)


标志着调度后处理阶段开始的无操作。

### unsafe_hide_buffer_access(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv), *buf_type:*[str](https://docs.python.org/3/library/stdtypes.html#str), *buf_index_array:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*) → [None](https://docs.python.org/3/library/constants.html#None)


隐藏给定块中的某些缓冲区访问。这是一个不安全的调度原语。
* **参数：**
   * **块**（[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)）：我们隐藏读取访问权限的块。
   * **buf_type** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：缓冲区类型：「读」/「写」。
   * **buf_index_array** (*List[*[int](https://docs.python.org/3/library/functions.html#int)*]*) ：我们隐藏访问的缓冲区索引数组。

:::Note

此调度原语不安全，并且可能无法通过依赖性分析。unsafe_hide_buffer_access 的一个用例是隐藏对索引缓冲区的访问（例如在稀疏计算中），以便我们可以进一步对块进行张量化（出现在读/写区域中的索引缓冲区可能无法通过张量化原语中的模式匹配，而隐藏对这些缓冲区的访问可以解决这个问题）。

:::

### annotate_buffer_access(*block:*[BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv), *buffer_index:*[int](https://docs.python.org/3/library/functions.html#int), *buf_type:*[str](https://docs.python.org/3/library/stdtypes.html#str), *gen_new_ranges:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)) → [None](https://docs.python.org/3/library/constants.html#None)


注释块的读取或写入区域。
* **参数：**
   * **block** ([BlockRV](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv)) ：需要注释的块。
   * **buffer_index** ([int](https://docs.python.org/3/library/functions.html#int)) ：块的读取或写入区域中缓冲区的索引。
   * **buf_type** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：缓冲区类型：「读」或「写」。
   * **gen_new_ranges** (*Callable*) ：一个函数，它接受块的 iter_vars 并返回一个 Tuple[Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], …] 类型的值，该值定义了缓冲区新的读写区域。该 Tuple 中的每个元素可以是：一个表示 iter_var 本身的 PrimExpr；一个包含两个 PrimExpr 的 Tuple，表示范围 (begin, end)。


**示例**


为缓冲区注释二维读取区域。在 annotate_buffer_access 之前，在 TensorIR 中，IR 为：

```python
@T.prim_func
def before_annotate_buffer_access(
    A: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32")
) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


创建调度并执行 annotate_buffer_access：

```python
sch = tir.Schedule(before_annotate_buffer_access)
block = sch.get_block("B")
sch.annotate_buffer_access(block, 0, "read",
lambda vi, vj: ((vi - 1, vi + 1), (vj - 1, vj + 1)))
print(sch.mod["main"].script())
```


应用 annotate_buffer_access 后，IR 变为：

```python
@T.prim_func
def after_annotate_buffer_access(
    A: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32")
) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi - 1:vi + 1, vj - 1:vj + 1])
            T.writes(B[vi, vj])
            T.block_attr({"explicit_read_region": 0})
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
```


这将块“B”中缓冲区 A（索引 0）的读取区域注释为块迭代域中每个 (vi, vj) 的 [vi-1:vi+1, vj-1:vj+1]。

:::Note

此函数允许手动指定读取或写入区域，这在编译器无法准确推断访问模式（例如复杂的数据依赖型访问）的情况下非常有用。它会覆盖指定缓冲区的自动推断区域。该函数会向块添加一个注释，指示已为给定索引处的缓冲区提供了显式区域。此注释在 CompactBufferAllocation 过程中用于遵循手动指定的区域，而不是依赖自动推断。

使用此函数时应谨慎，因为错误的注释可能会导致错误的代码生成或运行时错误。务必确保指定的区域涵盖该块对给定缓冲区执行的所有实际读取或写入操作。

:::

## *exception* tvm.tir.schedule.ScheduleError


TensorIR 调度期间发生的错误。

## *class* tvm.tir.schedule.ScheduleDebugMask(*value*) 


ScheduleState 类中 debug_mask 标志的位掩码。


如果 debug_mask 标志的某个位被置位，则会执行相应的验证过程。例如，如果(debug_mask & VERIFY_SREF_TREE) != 0，则在每次调度指令后都会验证 sref 树的正确性。

### VERIFY_SREF_TREE


验证 sref 树的正确性
* **类型：** int = 1

### VERIFY_CACHED_FLAGS


验证 affine_binding、region_cover 和 stage_pipeline 的正确性
* **类型：** int = 2

### *class* tvm.tir.schedule.ScheduleState(*mod:*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*|*[IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone), ***, *debug_mask: [str](https://docs.python.org/3/library/stdtypes.html#str) | [int](https://docs.python.org/3/library/functions.html#int) = 'none'*, *enable_check: [bool](https://docs.python.org/3/library/functions.html#bool) = True*)

调度状态，它公开了一个 Replace 方法作为所有调度原语操作 TensorIR 的主要手段。


数据结构包含以下信息 1）正在调度的 AST（mod）2）可调度语句的 sref 树（由 srefs 指示）3）每个块范围的依赖信息（block_info）4）从 AST 节点到 sref 树中的反向映射（get_sref）5）调试标志，如果设置，则启用额外检查（debug_mask）6）启用检查标志，如果为 False，则禁用某些先决条件检查。
* **参数：**
   * **mod** ([IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)) ：正在调度的模块的 AST。
   * **debug_mask** ([int](https://docs.python.org/3/library/functions.html#int)) ：在对象构造之后以及每次调用 Replace 方法后进行额外的正确性检查。
   * **enable_check** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：指示是否为某些调度原语启用先决条件检查，默认为 True。

### get_sref(*stmt:*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)*|*[For](/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none)) → [StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref) | [None](https://docs.python.org/3/library/constants.html#None)


返回指向 stmt 的相应 sref。
* **参数：stmt** (*Union[*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)*,*[For](/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none)*]*)：TensorIR 中可调度的语句，用于检索其 sref。
* **返回：sref**：对应的 sref。
* **返回类型：**[StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)。

### get_block_scope(*block_sref:*[StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)) → [BlockScope](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockscope)


获取与块 sref 对应的 BlockScope。
* **参数：block_sref** ([StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref))：要检索的块 sref。
* **返回：sref**：对应的 sref。
* **返回类型：**[StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)。

### replace(*src_sref:*[StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)), *tgt_stmt:*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)*|*[For](/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none)*|*[BlockRealize](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockrealizeiter_valueslistprimexpr-predicateprimexprbool-blockblock-spanspannone-none), *block_sref_reuse:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)*,*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None)


将 src_sref 指向的 AST 部分替换为具体的语句 tgt_stmt，并相应地维护 sref 树。当 ScheduleState 持有 IRModule 和 IR 节点的唯一副本时，Replace 将尽可能执行写时复制。

Only 3 types of replacements are allowed: from src_sref->stmt to tgt_stmt. 1) Block -> Block 2) Loop -> Loop 3) Loop -> BlockRealize。

只允许 3 种类型的替换：从 src_sref->stmt 到 tgt_stmt。1) Block -> Block 2) Loop -> Loop 3) Loop -> BlockRealize。
* **参数：**
   * **src_sref** ([StmtSRef](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulestmtsref)) ：TensorIR AST 中要替换的语句的 sref。
   * **tgt_stmt** (*Union[*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)*,*[For](/docs/api-reference/python-api/tvm-tir#class-tvmtirforloop_varvar-minprimexpr-extentprimexpr-kindforkind-bodystmt-thread_bindingitervarnone-none-annotationsmappingstr-object-none-none-spanspannone-none)*,*[BlockRealize](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockrealizeiter_valueslistprimexpr-predicateprimexprbool-blockblock-spanspannone-none)*]*) ：要替换为的语句。
   * **block_sref_reuse** (*Optional**[****Dict**[***[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)***,*** [Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)***]****]= None*) ：将旧块（在 src_sref–>stmt 下的子树中被替换）映射到新块（在 tgt_stmt 下的子树中被替换），并强制在它们之间重用 sref（而不是创建新的 sref），即在被替换之后，指向旧块的 sref 将指向新块。

:::Note

根据循环变量的重用自动检测循环引用的重用。

:::

### *class* tvm.tir.schedule.Trace(*insts:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany)*]*, *decisions:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany)*,*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*]*)


调度程序的执行轨迹。


跟踪包含两个部分：1）迄今为止调用的指令 2）根据这些指令做出的随机决策（如果有）。


跟踪可以序列化为：1）可往返的 JSON 格式：可以保存到文件并重新加载 2）Python 语法：允许用户复制粘贴跟踪以重现调度过程。


可以通过重新应用所有指令（可能包含相应的决策）将迹线应用于 TensorIR 调度。如果采样指令没有相应的决策，则会调用重新采样；否则，将相应地重用现有决策。

### insts


程序执行过程中调用的指令。
* **类型：** List[[Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany)]

### decisions


根据这些指示做出的随机决定。
* **类型：** Dict[[Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany), DECISION_TYPE]。

### get_decision(*inst:*[Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany)) → [Any](https://docs.python.org/3/library/typing.html#typing.Any) | [None](https://docs.python.org/3/library/constants.html#None)


检索针对特定指令做出的决定。
* **参数：insts** ([Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany))：要检索其决策的指令。
* **返回：decision**：相应的决定；如果没有对指令做出决定，则为无。
* **返回类型：** Optional[DECISION_TYPE]。

### append(*inst:*[Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany), *decision:*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None)


将新指令附加到跟踪中。
* **参数：**
   * **insts**（[指令](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany)）：要附加的新指令。
   * *decision*（*可选*[ *DECISION_TYPE]= None*）：对此指令做出的随机决定。

### pop() → [Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany) | [None](https://docs.python.org/3/library/constants.html#None)


删除最后一条指令，以及根据该指令做出的决定（如果有）。
* **返回：popped_inst**：返回被移除的指令；如果跟踪为空，则返回 std::nullopt。
* **返回类型：**[Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany)。

### apply_to_schedule(*sch:*[Schedule](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true), *remove_postproc:*[bool](https://docs.python.org/3/library/functions.html#bool), *decision_provider:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*[[*[Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany)*,*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*],*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*],*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*],*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None)


将跟踪应用于 TensorIR 调度。
* **参数：**
   * **sch**（[调度](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true)）：要应用的调度。
   * **remove_postproc** ( [bool](https://docs.python.org/3/library/functions.html#bool) ) ：如果删除后处理指令。
   * *decision_provider*（*可选*[ *Callable]= None*）：一个回调函数，允许用户在应用指令时动态修改决策。回调函数的签名如下：第一个参数：指令；第二个参数：输入随机变量；第三个参数：属性；第四个参数：决策。返回：新的决策。

### as_json(*remove_postproc:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)


将跟踪序列化为 JSON 样式的对象。
* **参数：remove_postproc** (*bool = False*)：如果删除后处理指令。
* **返回：json**：JSON 样式的对象。
* **返回类型：** JSON_TYPE。

### as_python(*remove_postproc:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]


将跟踪序列化为 Python 语句序列。
* **参数：remove_postproc** (*bool = False*)：如果删除后处理指令。
* **返回：py_stmts**：python 语句序列。
* **返回类型：** List[[str](https://docs.python.org/3/library/stdtypes.html#str)]。

### with_decision(*inst:*[Instruction](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany), *decision:*[Any](https://docs.python.org/3/library/typing.html#typing.Any), *remove_postproc:*[bool](https://docs.python.org/3/library/functions.html#bool)) → [Trace](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduletraceinstslistinstruction-decisionsdictinstructionany)


创建一条带有已更改决策的指令的新轨迹，假设该指令存在于结果轨迹中。
* **参数：**
   * **inst**（[指令](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleinstructionkindinstructionkind-inputslistany-attrslistany-outputslistany)）：需要更改其决策的指令。
   * **决策**（*DECISION_TYPE*）：需要更改为的决策。
   * **remove_postproc** ( [bool](https://docs.python.org/3/library/functions.html#bool) ) ：如果删除后处理指令。
* **返回：trace**：决策发生改变的新跟踪。
* **返回类型：**[Trace](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduletraceinstslistinstruction-decisionsdictinstructionany)。

### simplified(*remove_postproc:*[bool](https://docs.python.org/3/library/functions.html#bool)) → [Trace](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduletraceinstslistinstruction-decisionsdictinstructionany)

通过消除死代码来简化跟踪。
* **参数：remove_postproc** ([bool](https://docs.python.org/3/library/functions.html#bool))：如果删除后处理指令。
* **返回：trace**：简化的跟踪。
* **返回类型：**[Trace](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduletraceinstslistinstruction-decisionsdictinstructionany)。

### *static* apply_json_to_schedule(*json_obj:*[Any](https://docs.python.org/3/library/typing.html#typing.Any), *sch:*[Schedule](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true)) → [None](https://docs.python.org/3/library/constants.html#None)

将 JSON 序列化的跟踪应用于 TensorIR 调度。
* **参数：**
   * **json_obj** ( *JSON_TYPE* ) ：JSON 序列化的跟踪。
   * **sch**（[调度](/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true)）：TensorIR 调度。

### show(*style:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *black_format:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [None](https://docs.python.org/3/library/constants.html#None)


用于打印突出显示的 TVM 脚本。
* **参数：**
   * **style**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：Pygmentize 打印样式，如果为 None 则自动检测。 更多详情请参阅 tvm.script.highlight.cprint 。
   * **black_format** ( [bool](https://docs.python.org/3/library/functions.html#bool) ) ：如果为 true，则使用格式化程序 Black 格式化 TVMScript。如果为 None，则根据“TVM_BLACK_FORMAT”环境变量确定。


