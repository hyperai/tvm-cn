---

title: tvm.dlight

---


DLight 包为深度学习工作负载提供了开箱即用的高效调度。

## *class* tvm.dlight.BlockInfo(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *iters:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[IterInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightiterinfokind-typing_extensionsliterals-r-o-varvar-domprimexpr-loop_rvlooprv)*]*, *block_rv:*[BlockRV](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleblockrv), *reduction_block:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)

有关 TIR 区块的信息。

### dom() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int) | [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)]


块的迭代域。

### dom_kind() → [str](https://docs.python.org/3/library/stdtypes.html#str)


块的迭代域类型，例如 SSSS、SSSR。

### is_injective() → [bool](https://docs.python.org/3/library/functions.html#bool)


该块是否是可注入的，即其所有迭代域都是可注入的。

### is_elementwise(*sch:*[Schedule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true)) → [bool](https://docs.python.org/3/library/functions.html#bool)


该块是否是元素级的，即读/写区域之间的简单映射.

### is_reduction() → [bool](https://docs.python.org/3/library/functions.html#bool)

该块是否为减少工作负载。

### is_gemv() → [bool](https://docs.python.org/3/library/functions.html#bool)


该块是否为 GEMV 工作负载。

### is_gemm() → [bool](https://docs.python.org/3/library/functions.html#bool)


该块是否为 GEMM 工作负载。

## *class* tvm.dlight.IterInfo(*kind: typing_extensions.Literal[S, R, O]*, *var:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none), *dom:*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *loop_rv:*[LoopRV](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirschedulelooprv))


有关循环/iter var 的信息。

### *property* dom*:*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


循环的迭代域。

## tvm.dlight.normalize_prim_func(*sch:*[Schedule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[BlockInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightblockinfonamestr-iterslistiterinfo-block_rvblockrv-reduction_blockbool-false)] | [None](https://docs.python.org/3/library/constants.html#None)


将 primfunc 规范化为范式.

## *class* tvm.dlight.ApplyDefaultSchedule(rules:*[ScheduleRule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightschedulerule))


IRModule 过程将 ScheduleRules 列表应用于模块中的所有 PrimFuncs。

## *class* tvm.dlight.ScheduleRule


任意函数的薄包装器，可用于调度 TIR PrimFunc。


给定一个 PrimFunc、一个目标和一个可调标志，ScheduleRule 的 apply 方法将返回一个 Schedule、一个 Schedule 列表或 None，其中 None 表示该规则不适用于给定的 PrimFunc。如果可调标志为 True，则 ScheduleRule 可以返回一个 Schedule 或一个 Schedule 列表，并且 Schedule 可以包含可调指令。如果可调标志为 False，则 ScheduleRule 只能返回一个 Schedule，并且 Schedule 不允许包含可调指令。

### apply(*func:*[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone), *target:*[Target](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone), *tunable:*[bool](https://docs.python.org/3/library/functions.html#bool)) → [None](https://docs.python.org/3/library/constants.html#None) | [Schedule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Schedule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true)]


将 ScheduleRule 应用于给定的 PrimFunc。
* **参数：**
   * **func** ([tir.PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))：应用 ScheduleRule 的 PrimFunc。
   * **target** ([Target](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone))：调度所要构建的编译目标。
   * **tunable** ([bool](https://docs.python.org/3/library/functions.html#bool))：调度是否允许包含可调指令。
* **返回：results**：可以是 Schedule、Schedule 列表或 None，其中 None 表示该规则不适用于给定的 PrimFunc。
* **返回类型：** Union[None, tir.Schedule, List[tir.Schedule]].

### *static* from_callable(*name*) → [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone), [Target](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone), [bool](https://docs.python.org/3/library/functions.html#bool)], [None](https://docs.python.org/3/library/constants.html#None) | [Schedule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Schedule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true)]]], [ScheduleRule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightschedulerule)]


从可调用对象创建 ScheduleRule。
* **参数：name** ([str](https://docs.python.org/3/library/stdtypes.html#str))。
* **返回：decorator**：接受可调用函数并返回 ScheduleRule 的装饰器。
* **返回类型：** Callable。


**示例**

```plain
@ScheduleRule.from_callable("MyRule")
def my_rule(func: tir.PrimFunc, target: Target, tunable: bool) -> Union[None, Schedule]
   # 使用 func 和 target 做一些操作
```
### is_target_available(*target:*[Target](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone)) → [bool](https://docs.python.org/3/library/functions.html#bool)


检查该规则是否适用于给定的目标。
* **参数：target** ([Target](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-target#class-tvmtargettargettarget-hostnone))：调度所要构建的编译目标。
* **返回：available：** 该规则是否适用于给定的目标。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

## tvm.dlight.try_inline(*sch:*[Schedule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true), *blocks:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[BlockInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightblockinfonamestr-iterslistiterinfo-block_rvblockrv-reduction_blockbool-false)*]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[BlockInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightblockinfonamestr-iterslistiterinfo-block_rvblockrv-reduction_blockbool-false)]


尝试内联尽可能多的块，并返回剩余的块。
* **参数：**
   * **sch** (*tir.Schedule*)**：** 用于内联块的 TIR 调度。
   * **blocks** (*List[*[BlockInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightblockinfonamestr-iterslistiterinfo-block_rvblockrv-reduction_blockbool-false)*]*)：要内联的块。
* **返回：remaining**[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightblockinfonamestr-iterslistiterinfo-block_rvblockrv-reduction_blockbool-false)无法内联的剩余块。
* **返回类型：** List[[BlockInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightblockinfonamestr-iterslistiterinfo-block_rvblockrv-reduction_blockbool-false)]。

## tvm.dlight.try_inline_contiguous_spatial(*sch:*[Schedule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir-schedule#class-tvmtirscheduleschedulemodprimfuncirmodule--seed-int--none--none-debug_mask-str--int--none-error_render_level-str--detail-enable_check-bool--true), *block_infos:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[BlockInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightblockinfonamestr-iterslistiterinfo-block_rvblockrv-reduction_blockbool-false)*]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[BlockInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightblockinfonamestr-iterslistiterinfo-block_rvblockrv-reduction_blockbool-false)]


尝试在时间表中内联连续的空间块。
* **参数：**
   * **sch** (*tir.Schedule*)：用于内联块的 TIR 调度。
   * **block_infos** (*List[*[BlockInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightblockinfonamestr-iterslistiterinfo-block_rvblockrv-reduction_blockbool-false)*]*)：要尝试的区块。
* **返回：remaining**：无法内联的剩余块。
* **返回类型：** List[[BlockInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-dlight#class-tvmdlightblockinfonamestr-iterslistiterinfo-block_rvblockrv-reduction_blockbool-false)]。


