---

title: tvm.te

---


张量表达语言的命名空间。



**函数:**

|[any](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteanyargs-spannone)(*args[,span])|创建参数中所有条件的并集的新表达式。|
|:----|:----|
|[all](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteallargs-spannone)(*args[,span])|创建所有条件交集的新表达式。|
|[min_value](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtemin_valuedtype-spannone)(dtype[,span])|dtype 的最小值。|
|[max_value](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtemax_valuedtypestr-spanspannone-none--any)(dtype[,span])|dtype 的最大值。|
|[trace](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtetraceargs-trace_actiontvmdefault_trace_action)(args[,trace_action])|在运行时跟踪张量数据。|
|[exp](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteexpx)(x)|取输入 x 的指数。|
|[erf](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteerfx)(x)|取输入 x 的高斯误差函数。|
|[tanh](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtetanhx)(x)|对输入 x 取双曲 tanh。|
|[sigmoid](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtesigmoidx)(x)|快速获取 S 形函数。|
|[log](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtelogx)(x)|对输入 x 取对数。|
|[tan](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtetanx)(x)|对输入 x 取 tan。|
|[cos](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtecosx)(x)|取输入 x 的 cos。|
|[sin](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtesinx)(x)|对输入 x 取正弦值。|
|[sqrt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtesqrtx)(x)|对输入 x 取平方根。|
|[rsqrt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtersqrtx)(x)|取输入 x 的平方根的倒数。|
|[floor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtefloorx-primexprwithop-spannone)(x[,span])|取浮点输入 x 的下限。|
|[ceil](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteceilx-spannone)(x[,span])|对浮点输入 x 取上限。|
|[sinh](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtesinhx)(x)|对输入 x 取 sinh。|
|[cosh](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtecoshx)(x)|对输入 x 取余弦值。|
|[log2](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtelog2x)(x)|对输入 x 取 log2。|
|[log10](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtelog10x)(x)|对输入 x 取 log10。|
|[asin](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteasinx)(x)|取输入 x 的 asin。|
|[asinh](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteasinhx)(x)|取输入 x 的正弦值。|
|[acos](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteacosx)(x)|对输入 x 取余数。|
|[acosh](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteacoshx)(x)|对输入 x 取余数。|
|[atan](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteatanx)(x)|对输入 x 取正切值。|
|[atanh](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteatanhx)(x)|对输入 x 进行 atanh 处理。|
|[trunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtetruncx-spannone)(x[,span])|获取输入的截断值。|
|[abs](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteabsx-spannone)(x[,span])|逐个获取输入元素的绝对值。|
|[round](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteroundx-spannone)(x[,span])|将数组元素四舍五入为最接近的整数。|
|[nearbyint](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtenearbyintx-spannone)(x[,span])|将数组元素四舍五入为最接近的整数。|
|[power](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtepowerx-y-spannone)(x,y[,span])|x 次方 y。|
|[popcount](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtepopcountx)(x)|计算输入 x 中设置位的数量。|
|[fmod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtefmodx-y)(x,y)|返回 x 除以 y 后的余数，其符号与 x 相同。|
|[if_then_else](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteif_then_elsecond-t-f-spannone)(cond,t,f[,span])|条件选择表达式。|
|[isnan](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteisnanx-spannone)(x[,span])|检查输入值是否为 Nan。|
|[isfinite](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteisfinitex-spannone)(x[,span])|检查输入值是否有限。|
|[isinf](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteisinfx-spannone)(x[,span])|检查输入值是否无限。|
|[div](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtediva-b-spannone)(a,b[,span])|按照 C/C++ 语义计算 a / b。|
|[indexdiv](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteindexdiva-b-spannone)(a,b[,span])|计算 floor(a / b)，其中 a 和 b 为非负数。|
|[indexmod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteindexmoda-b-spannone)(a,b[,span])|计算 indexdiv 的余数。a 和 b 非负。|
|[truncdiv](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtetruncdiva-b-spannone)(a,b[,span])|计算两个表达式的 truncdiv。|
|[truncmod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtetruncmoda-b-spannone)(a,b[,span])|计算两个表达式的 truncmod。|
|[floordiv](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtefloordiva-b-spannone)(a,b[,span])|计算两个表达式的 floordiv。|
|[floormod](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtefloormoda-b-spannone)(a,b[,span])|计算两个表达式的 floormod。|
|[logaddexp](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtelogaddexpa-b-spannone)(a,b[,span])|计算两个表达式的 logaddexp。|
|[comm_reducer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtecomm_reducerfcombine-fidentity-namereduce)(fcombine,fidentity[,name])|创建一个交换减速器用于减速。|
|[min](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteminexpr-axis-wherenone-initnone-args)(expr,axis[,where,init])|在轴上创建最小表达式。|
|[max](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtemaxexpr-axis-wherenone-initnone-args)(expr,axis[,where,init])|在轴上创建最大表达式。|
|[sum](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtesumexpr-axis-wherenone-initnone-args)(expr,axis[,where,init])|在轴上创建一个求和表达式。|
|[add](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteaddlhs-rhs-spannone)(lhs,rhs[,span])|通用加法运算符。|
|[subtract](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtesubtractlhs-rhs-spannone)(lhs,rhs[,span])|通用减法运算符。|
|[multiply](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtemultiplylhs-rhs-spannone)(lhs,rhs[,span])|通用乘法运算符。|
|[tag_scope](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtetag_scopetag)(tag)|运算符标签范围。|
|[placeholder](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteplaceholdershape-dtypenone-nameplaceholder)(shape[,dtype,name])|构造一个空的张量对象。|
|[compute](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtecomputeshape-fcompute-namecompute-tag-attrsnone-varargs_namesnone)(shape,fcompute[,name,tag,attrs,…])|通过计算形状域来构建一个新的张量。|
|[scan](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtescaninit-update-state_placeholder-inputsnone-namescan-tag-attrsnone)(init,update,state_placeholder[,…])|通过扫描轴来构建新的张量。|
|[extern](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteexternshape-inputs-fcompute-nameextern-dtypenone-in_buffersnone-out_buffersnone-tag-attrsnone)(shape,inputs,fcompute[,name,…])|通过外部函数计算多个张量。|
|[var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtevarnametindex-dtypeint32-spannone)([name,dtype,span])|创建具有指定名称和数据类型的新变量。|
|[size_var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtesize_varnamesize-dtypeint32-spannone)([name,dtype,span])|创建一个新变量表示张量形状的大小，它是非负的。|
|[const](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteconstvalue-dtypeint32-spannone)(value[,dtype,span])|创建具有指定值和数据类型的新常量。|
|[thread_axis](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtethread_axisdomnone-tag-name-spannone)([dom,tag,name,span])|创建一个新的 IterVar 来表示线程索引。|
|[reduce_axis](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtereduce_axisdom-namerv-thread_tag-spannone)(dom[,name,thread_tag,span])|创建一个新的 IterVar 进行缩减。|
|[create_prim_func](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmtecreate_prim_funcopslisttensorvar-index_dtype_overridestrnone-none--primfunc)(ops[,index_dtype_override])|从张量表达式创建 TensorIR PrimFunc。|
|[extern_primfunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#tvmteextern_primfuncinput_tensorslisttensor-primfuncprimfunc-kwargs)(input_tensors,primfunc,…)|通过可调度的 TIR PrimFunc 计算张量。|

**类:**

|[TensorSlice](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensorslicetensor-indices)(tensor,indices)|用于从张量启用切片语法的辅助数据结构。|
|:----|:----|
|[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)|Tensor 对象，构造方法参见 function.Tensor|
|[PlaceholderOp](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmteplaceholderop)|占位符操作。|
|[ComputeOp](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtecomputeop)|标量运算。|
|[ScanOp](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtescanop)|扫描操作。|
|[ExternOp](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmteexternop)|外部操作。|

## tvm.te.any(args*, *span=None*)


创建参数中所有条件的并集的新表达式。
* **参数：**
   * **args**（[列表](https://docs.python.org/3/library/stdtypes.html#list)）：符号布尔表达式列表。
   * *span*（*可选*[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：expr**：表达式。
* **返回类型：** Expr。


**别名**`tvm.tir.any()`

## tvm.te.all(args*, *span=None*)


创建一个新的表达式，该表达式表示所有参数条件的交集。
* **参数：**
   * **args**（[列表](https://docs.python.org/3/library/stdtypes.html#list)）：符号布尔表达式列表。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：expr**：表达式
* **返回类型：** Expr。


**别名**`tvm.tir.all()`

## tvm.te.min_value(*dtype*, *span=None*)

dtype 的最小值。
* **参数：**
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：数据类型。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：value**：dtype 的最小值。
* **返回类型：** tvm.Expr。


**别名**`tvm.tir.min_value()`

## tvm.te.max_value(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *span:*[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)


dtype 的最大值。
* **参数：**
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)）*：*数据类型。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：value**：dtype 的最大值。
* **返回类型：** tvm.Expr。


**别名**`tvm.tir.max_value()`

## tvm.te.trace(*args*, *trace_action='tvm.default_trace_action'*)


在运行时跟踪张量数据。


trace 函数允许在运行时跟踪特定的张量。跟踪值应作为最后一个参数。应指定跟踪操作，默认情况下使用 tvm.default_trace_action。
* **参数：**
   * **args**（*Expr**或*** *Buffers*[列表](https://docs.python.org/3/library/stdtypes.html#list)*。*）**：** 位置参数。
   * **trace_action**（*str.*）：跟踪操作的名称。
* **返回：call**：调用表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


:::info 另见

`tvm.tir.call_packed`

创建打包函数。

:::

别名 `tvm.tir.trace()`

## tvm.te.exp(*x*)


取输入 x 的指数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** [：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.exp()`

## tvm.te.erf(*x*)


取输入 x 的高斯误差函数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** [：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.erf()`

## tvm.te.tanh(*x*)

对输入 x 取双曲 tanh。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** [：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.tanh()`

## tvm.te.sigmoid(*x*)


快速获取 S 形函数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.sigmoid()`

## tvm.te.log(*x*)


对输入 x 取对数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.log()`

## tvm.te.tan(*x*)


对输入 x 取 tan。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** **：** 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.tan()`

## tvm.te.cos(*x*)

取输入 x 的 cos。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.cos()`

## tvm.te.sin(*x*)


对输入 x 取正弦值。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.sin()`

## tvm.te.sqrt(*x*)


对输入 x 取平方根。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.sqrt()`

## tvm.te.rsqrt(*x*)


取输入 x 的平方根的倒数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.rsqrt()`

## tvm.te.floor(*x: PrimExprWithOp*, *span=None*)


取浮点输入 x 的下限。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：输入参数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)此运算符在源代码中的位置。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.floor()`

## tvm.te.ceil(*x*, *span=None*)


对浮点输入 x 取上限。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )**：** 输入参数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.ceil()`

## tvm.te.sinh(*x*)


对输入 x 取 sinh。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.sinh()`

## tvm.te.cosh(*x*)


对输入 x 取余弦值。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.cosh()`

## tvm.te.log2(*x*)


对输入 x 取 log2。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.log2()`

## tvm.te.log10(*x*)


对输入 x 取 log10。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.log10()`

## tvm.te.asin(*x*)


取输入 x 的 asin。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.asin()`

## tvm.te.asinh(*x*)


取输入 x 的正弦值。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.asinh()`

## tvm.te.acos(*x*)


对输入 x 取余数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.acos()`

## tvm.te.acosh(*x*)


对输入 x 取余数。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.acosh()`

## tvm.te.atan(*x*)


对输入 x 取正切值。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.atan()`

## tvm.te.atanh(*x*)


对输入 x 进行 atanh 处理。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** *：* 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.atanh()`

## tvm.te.trunc(*x*, *span=None*)


获取输入的截断值。


标量 x 的截断值是最接近的整数 i，它比 x 更接近零。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：输入参数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.trunc()`

## tvm.te.abs(*x*, *span=None*)


逐个获取输入元素的绝对值。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：输入参数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.abs()`

## tvm.te.round(*x*, *span=None*)


将数组元素四舍五入为最接近的整数。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：输入参数。
   * span（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.round()`

## tvm.te.nearbyint(*x*, *span=None*)


将数组元素四舍五入为最接近的整数。此内在函数使用 llvm.nearbyint 而不是 llvm.round，后者速度更快，但结果与 te.round 不同。值得注意的是，nearbyint 根据舍入模式进行舍入，而 te.round (llvm.round) 则忽略该模式。有关两者之间的差异，请参阅： https: [//en.cppreference.com/w/cpp/numeric/math/round](https://en.cppreference.com/w/cpp/numeric/math/round) [https://en.cppreference.com/w/cpp/numeric/math/nearbyint](https://en.cppreference.com/w/cpp/numeric/math/nearbyint)。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：输入参数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：y** **：** 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.nearbyint()`

## tvm.te.power(*x*, *y*, *span=None*)


x 次方 y。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：输入参数。
   * **y** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：指数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：z** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.power()`

## tvm.te.popcount(*x*)

计算输入 x 中设置位的数量。
* **参数：x** ([PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.popcount()`

## tvm.te.fmod(*x*, *y*)


返回 x 除以 y 后的余数，其符号与 x 相同。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：输入参数。
   * **y** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：输入参数。
* **返回：z** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.fmod()`

## tvm.te.if_then_else(*cond*, *t*, *f*, *span=None*)


条件选择表达式。
* **参数：**
   * **cond** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：条件。
   * **t** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )**：** 如果 cond 为真，则结果表达式。
   * **f** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：如果 cond 为假，则结果表达式。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：result**：条件表达式的结果。
* **返回类型：**[Node](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirnode)。

:::Note

与 Select 不同，if_then_else 不会执行不满足条件的分支。您可以使用它来防止越界访问。与 Select 不同，如果向量中某些通道的条件不同，则 if_then_else 无法进行向量化。


别名`tvm.tir.if_then_else()`

:::

## tvm.te.isnan(*x*, *span=None*)


检查输入值是否为 Nan。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )*：* 输入参数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.isnan()`

## tvm.te.isfinite(*x*, *span=None*)


检查输入值是否有限。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：输入参数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源代码中的位置。
* **返回：y** *：* 结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.isfinite()`

## tvm.te.isinf(*x*, *span=None*)


检查输入值是否无限。
* **参数：**
   * **x** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：输入参数。
   * *span*（*可选*[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)此运算符在源代码中的位置。
* **返回：y** ：结果。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.isinf()`

## tvm.te.div(*a*, *b*, *span=None*)


按照 C/C++ 语义计算 a / b。
* **参数：**
   * **a** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：左侧操作数，已知为非负数。
   * **b** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：右侧操作数，已知为非负。
   * span（*可选*[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

当操作数为整数时，返回 truncdiv(a, b, span)。

:::

别名`tvm.tir.div()`


## tvm.te.indexdiv(*a*, *b*, *span=None*)


计算 floor(a / b)，其中 a 和 b 为非负数。
* **参数：**
   * **a** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：左侧操作数，已知为非负数。
   * **b** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：右侧操作数，已知为非负。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

使用此函数拆分非负索引。此函数可以利用操作数的非负性。

:::

别名`tvm.tir.indexdiv()`


## tvm.te.indexmod(*a*, *b*, *span=None*)


计算 indexdiv 的余数。a 和 b 非负。
* **参数：**
   * **a** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )`：`左侧操作数，已知为非负数。
   * **b** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：右侧操作数，已知为非负。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

使用此函数拆分非负索引。此函数可以利用操作数的非负性。

:::

别名`tvm.tir.indexmod()`


## tvm.te.truncdiv(*a*, *b*, *span=None*)


计算两个表达式的 truncdiv。
* **参数：**
   * **a** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：左侧操作数。
   * **b** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：右侧操作数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

这是 C 中的默认整数除法行为。

:::

别名`tvm.tir.truncdiv()`


## tvm.te.truncmod(*a*, *b*, *span=None*)


计算两个表达式的 truncmod。
* **参数：**
   * **a** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：左侧操作数。
   * **b** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：右侧操作数。
   * *span*（*可选*[[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

这是 C 中的默认整数除法行为。

:::

别名`tvm.tir.truncmod()`


## tvm.te.floordiv(*a*, *b*, *span=None*)


计算两个表达式的 floordiv。
* **参数：**
   * **a** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：左侧操作数。
   * **b** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：右侧操作数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.floordiv()`

## tvm.te.floormod(*a*, *b*, *span=None*)


计算两个表达式的 floormod。
* **参数：**
   * **a** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：左侧操作数。
   * **b** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：右侧操作数。
   * *span*（*可选*[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.floormod()`

## tvm.te.logaddexp(*a*, *b*, *span=None*)


计算两个表达式的 logaddexp。
* **参数：**
   * **a** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：左侧操作数。
   * **b** ( [PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )**：** 右侧操作数。
   * span（*可选*[[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。


**别名**`tvm.tir.logaddexp()`

## tvm.te.comm_reducer(*fcombine*, *fidentity*, *name='reduce'*) 


创建一个交换减速器用于减速。
* **参数：**
   * **fcombine** (function **(** Expr -> Expr -> Expr))  ：一个二元函数，以两个 Expr 作为输入并返回一个 Expr。
   * **fidentity** (function **(** str -> Expr))  ：以字符串类型作为输入并返回 const Expr 的函数。
* **返回：reducer**：在 axis 上创建 Reduce 表达式的函数。有两种使用方法：
   * accept (expr, axis, where) 来在指定轴上生成一个 Reduce Expr；
   * 直接使用多个 Exprs。
* **返回类型：函数**


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


**别名**`tvm.tir.comm_reducer()`

## tvm.te.min(*expr*, *axis*, *where=None*, *init=None*, args*)


在轴上创建最小表达式。
* **参数：**
   * **expr**（[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)）：源表达式。
   * **轴**（[IterVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiritervardomrange-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)）：缩减 IterVar 轴
   * *where*（*可选*，*Expr*）：减少的过滤谓词。
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


**别名**`tvm.tir.min()`

## tvm.te.max(*expr*, *axis*, *where=None*, *init=None*, args*)


在轴上创建最大表达式。
* **参数：**
   * **expr**（[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)）：源表达式。
   * **轴**（[IterVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiritervardomrange-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)）：缩减 IterVar 轴。
   * *where**（*可选**，*Expr*）：减少的过滤谓词。
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


**别名**`tvm.tir.max()`

## tvm.te.sum(*expr*, *axis*, *where=None*, *init=None*, args*)


在轴上创建一个求和表达式。
* **参数：**
   * **expr**（[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)）：源表达式。
   * **轴**（[IterVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiritervardomrange-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)）：缩减 IterVar 轴。
   * *where**（*可选**，*Expr*）：减少的过滤谓词。
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


**别名**`tvm.tir.sum()`

## tvm.te.add(*lhs*, *rhs*, *span=None*)


通用加法运算符。
* **参数：**
   * **lhs**（[对象](https://docs.python.org/3/library/functions.html#object)）：左操作数。
   * **rhs**（[对象](https://docs.python.org/3/library/functions.html#object)）：右操作数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：op**：加法运算的结果 Expr。
* **返回类型：** tvm.Expr。


**别名**`tvm.tir.add()`

## tvm.te.subtract(*lhs*, *rhs*, *span=None*)


通用减法运算符。
* **参数：**
   * **lhs**（[对象](https://docs.python.org/3/library/functions.html#object)）：左操作数。
   * **rhs**（[对象](https://docs.python.org/3/library/functions.html#object)）[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)右操作数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：op**：减法运算的结果 Expr。
* **返回类型：** tvm.Expr。


**别名**`tvm.tir.subtract()`

## tvm.te.multiply(*lhs*, *rhs*, *span=None*)


通用乘法运算符。
* **参数：**
   * **lhs**（[对象](https://docs.python.org/3/library/functions.html#object)）：左操作数。
   * **rhs**（[对象](https://docs.python.org/3/library/functions.html#object)）*：* 右操作数。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）*：* 此运算符在源中的位置。
* **返回：op**：乘法运算的结果 Expr。
* **返回类型：** tvm.Expr。


**别名**`tvm.tir.multiply()`

## *class* tvm.te.TensorSlice(*tensor*, *indices*)


用于从张量启用切片语法的辅助数据结构。


**方法：**

|[asobject](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#asobject)()|将切片转换为对象。|
|:----|:----|

**属性：**

|[dtype](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#property-dtype)|张量的数据内容。|
|:----|:----|

### asobject()


将切片转换为对象。

### *property* dtype

张量的数据内容。

## *class* tvm.te.Tensor

Tensor 对象，构造方法参见 function.Tensor


**属性：**

|[ndim](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#property-ndim)|张量的维度。|
|:----|:----|

## *property* ndim

张量的维度。




## tvm.te.tag_scope(*tag*)

操作符标签范围。
* **参数：tag** ([str](https://docs.python.org/3/library/stdtypes.html#str))：标签名称。
* **返回：tag_scope**：标签范围对象，可用作装饰器或上下文管理器。
* **返回类型：** TagScope。


**示例**

```python
n = te.var('n')
m = te.var('m')
l = te.var('l')
A = te.placeholder((n, l), name='A')
B = te.placeholder((m, l), name='B')
k = te.reduce_axis((0, l), name='k')

with tvm.te.tag_scope(tag='matmul'):
    C = te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k))

# 或者使用 tag_scope 作为装饰器 @tvm.te.tag_scope(tag="conv")
def compute_relu(data):
    return te.compute(data.shape, lambda *i: tvm.tir.Select(data(*i) < 0, 0.0, data(*i)))
```
## tvm.te.placeholder(*shape*, *dtype=None*, *name='placeholder'*)


构造一个空的张量对象。
* **参数：**
   * **shape**（*Expr的*[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)）：张量的形状。
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：张量的数据类型。
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：张量的名称提示。
* **返回：tensor**：创建的张量。
* **返回类型：**[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.te.compute(*shape*, *fcompute*, *name='compute'*, *tag=''*, *attrs=None*, *varargs_names=None*)


通过计算形状域来构建一个新的张量。


计算规则是 result[axis] = fcompute(axis)
* **参数：**
   * **shape**（*Expr的*[元组](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)）：张量的形状。
   * **fcompute**（*indices–>value 的lambda 函数*）：指定输入源表达式。
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）*：* 张量的名称提示。
   * **tag**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：有关计算的附加标签信息。
   * **attrs**（[dict](https://docs.python.org/3/library/stdtypes.html#dict)*，可选*）**：** 有关计算的附加辅助属性。
   * **varargs_names**（[list](https://docs.python.org/3/library/stdtypes.html#list)*，可选*）[：](https://docs.python.org/3/library/stdtypes.html#list)每个可变参数使用的名称。如果未提供，可变参数将被称为 i1、i2……
* **返回：tensor**：创建的张量。
* **返回类型：**[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.te.scan(*init*, *update*, *state_placeholder*, *inputs=None*, *name='scan'*, *tag=''*, *attrs=None*)

通过扫描轴来构建新的张量。
* **参数：**
   * **init** ( [Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Tensor*[列表)](https://docs.python.org/3/library/stdtypes.html#list)[：](https://docs.python.org/3/library/stdtypes.html#list)[第一个 init.shape[0] 时间戳](https://docs.python.org/3/library/stdtypes.html#list)[的](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)初始条件。
   * **更新**（[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或*[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)[列表](https://docs.python.org/3/library/stdtypes.html#list)*）：由符号*张量给出的扫描更新规则。
   * **state_placeholder**（[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Tensor*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：更新使用的占位符变量[。](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)
   * **输入**( [Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor) 或 *Tensor* [列表](https://docs.python.org/3/library/stdtypes.html#list)*，可选*)：扫描的输入列表。这不是必需的，但有助于编译器更快地检测扫描主体[。](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：张量的名称提示。
   * **tag**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：有关计算的附加标签信息。
   * **attrs**（[dict](https://docs.python.org/3/library/stdtypes.html#dict)*，可选*）：有关计算的附加辅助属性。
* **返回：tensor**：创建的张量或张量元组包含多个输出。
* **返回类型：**[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or [list](https://docs.python.org/3/library/stdtypes.html#list) of Tensors。


**示例**

```python
# 以下代码等价于 numpy.cumsum
m = te.var("m")
n = te.var("n")
X = te.placeholder((m, n), name="X")
s_state = te.placeholder((m, n))
s_init = te.compute((1, n), lambda _, i: X[0, i])
s_update = te.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
res = tvm.te.scan(s_init, s_update, s_state, X)
```
## tvm.te.extern(*shape*, *inputs*, *fcompute*, *name='extern'*, *dtype=None*, *in_buffers=None*, *out_buffers=None*, *tag=''*, *attrs=None*)


通过外部函数计算多个张量。
   * **参数：**
   * **shape**（[元组](https://docs.python.org/3/library/stdtypes.html#tuple)*或元组*[列表](https://docs.python.org/3/library/stdtypes.html#list)*。*）：输出的形状。
   * **输入**（*Tensor*[列表）](https://docs.python.org/3/library/stdtypes.html#list)[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)输入
   * *fcompute**（*输入**的**lambda 函数*，*输出–> stmt*）***：*** 指定用于执行计算的 IR 语句。请参阅以下注释以了解 fcompute 的函数签名。

:::Note
* **参数**
   * *ins* (list of [tvm.tir.Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)) ：每个输入的占位符。
   * *outs* (list of [tvm.tir.Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)) ：每个输出的占位符。
   * *返回：stmt(*[tvm.tir.Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*)* ：执行数组计算的语句。

:::
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）**：** 张量的名称提示。
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)[列表](https://docs.python.org/3/library/stdtypes.html#list)*，可选）：输出的数据类型，默认情况*下 dtype 与输入相同。
   * **in_buffers**（[tvm.tir.Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*或**tvm.tir.Buffer***[列表](https://docs.python.org/3/library/stdtypes.html#list)*，可选*）：输入缓冲区[。](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)
   * **out_buffers**（[tvm.tir.Buffer](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*或**tvm.tir.Buffer***[列表](https://docs.python.org/3/library/stdtypes.html#list)，*可选*）：输出缓冲区[。](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)

**tag: str, optional**

有关计算的附加标签信息。

**attrs: dict, optional**

有关计算的附加辅助属性。
* **返回：tensor**：创建的张量或张量元组包含多个输出。
* **返回类型：**[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or [list](https://docs.python.org/3/library/stdtypes.html#list) of Tensors。


**示例**


在下面的代码中，C 是通过调用外部 PackedFunc tvm.contrib.cblas.matmul 生成的。

```python
A = te.placeholder((n, l), name="A")
B = te.placeholder((l, m), name="B")
C = te.extern((n, m), [A, B],
               lambda ins, outs: tvm.tir.call_packed(
                  "tvm.contrib.cblas.matmul",
                    ins[0], ins[1], outs[0], 0, 0), name="C")
```
## tvm.te.var(*name='tindex'*, *dtype='int32'*, *span=None*)

创建具有指定名称和数据类型的新变量。
* **参数：**
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：名称。
   * **dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：数据类型。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此变量在源中的位置。
* **返回：var**：结果符号变量。
* **返回类型：**[tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)。

## tvm.te.size_var(*name='size'*, *dtype='int32'*, *span=None*)


创建一个新变量表示张量形状的大小，它是非负的。
* **参数：**
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：名称。
   * **dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )*：* 数据类型。
   * *span**（*可选**[[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此变量在源中的位置。
* **返回：var**：结果符号形状变量。
* **返回类型：**[SizeVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirsizevarnamestr-dtypestrtype-spanspannone-none)。

## tvm.te.const(*value*, *dtype='int32'*, *span=None*)


创建具有指定值和数据类型的新常量。
* **参数：**
   * *value* ( *Union[*[bool](https://docs.python.org/3/library/functions.html#bool)*,*[int](https://docs.python.org/3/library/functions.html#int)*,*[float](https://docs.python.org/3/library/functions.html#float)*,numpy.ndarray,tvm.nd.NDArray]* )*：* 常量值。
   * **dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：数据类型。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）*：* 此变量在源中的位置。
* **返回：const**：结果常量 expr。
* **返回类型：**[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.te.thread_axis(*dom=None*, *tag=''*, *name=''*, *span=None*)


创建一个新的 IterVar 来表示线程索引。
* **参数：**
   * **dom** ( [Range](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)*或*[str](https://docs.python.org/3/library/stdtypes.html#str) )**：** 迭代的范围，当传入 str 时，dom 设置为 None，str 作为标签。
   * **tag**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：线程标签。
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：变量的名称。
   * span（*可选*[[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）**：** 此变量在源中的位置。
* **返回：axis**：线程迭代变量。
* **返回类型：**[IterVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiritervardomrange-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)。

## tvm.te.reduce_axis(*dom*, *name='rv'*, *thread_tag=''*, *span=None*)


创建一个新的 IterVar 进行缩减。
* **参数：**
   * **dom** ([范围](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none))**：** 迭代的范围。
   * **name** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )*：* 变量的名称。
   * *thread_tag**（*可选**[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]*）：thread_tag 的名称。
   * *span**（*可选**[ [Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此变量在源中的位置。
* **返回：axis**：表示值的迭代变量。
* **返回类型：**[IterVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtiritervardomrange-varvarstr-iter_typeint-thread_tagstr--spanspannone-none)。

## tvm.te.create_prim_func(*ops:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*|*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*]*, *index_dtype_override:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)


从张量表达式创建 TensorIR PrimFunc。
* **参数：ops** (*List**[****Union**[****_tensor.Tensor**,*** [tvm.tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)***]****]*)：源表达式。


**示例**

我们使用以下代码定义一个 matmul 核：

```python
import tvm
from tvm import te
from tvm.te import create_prim_func
import tvm.script

A = te.placeholder((128, 128), name="A")
B = te.placeholder((128, 128), name="B")
k = te.reduce_axis((0, 128), "k")
C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
func = create_prim_func([A, B, C])
print(func.script())
```


如果我们想使用 TensorIR 调度对此类核进行转换，我们需要使用 create_prim_func([A, B, C])来创建一个可调度的 PrimFunc。生成的函数如下所示：

```python
@T.prim_func
def tir_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    for i, j, k in T.grid(128, 128, 128):
        with T.block():
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] += A[vi, vk] * B[vj, vk]
```
* **返回：func**：创建的函数。
* **返回类型：**[tir.PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)

## tvm.te.extern_primfunc(*input_tensors:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*]*, *primfunc:*[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone), ***kwargs*)


通过可调度的 TIR PrimFunc 计算张量。
* **参数：**
   * **input_tensors**（[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)[列表](https://docs.python.org/3/library/stdtypes.html#list)）：映射到相应 primfunc 输入参数的输入张量。
   * **primfunc** ( [PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone) )：TIR PrimFunc。
* **返回：tensor**：如果包含多个输出，则创建的张量或张量元组。
* **返回类型：**[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or [list](https://docs.python.org/3/library/stdtypes.html#list) of Tensors。


**示例**


在下面的代码中，TVMScript 定义的 TIR PrimFunc 被内联到 TE ExternOp 中。在此代码中应用 te.create_prim_func。

```python
A = te.placeholder((128, 128), name="A")
B = te.placeholder((128, 128), name="B")

@T.prim_func
def before_split(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0

C = te.extern_primfunc([A, B], func)
```
## *class* tvm.te.PlaceholderOp

占位符操作。

## *class* tvm.te.ComputeOp

标量运算。

## *class* tvm.te.ScanOp

扫描操作。


**属性：**

|[scan_axis](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#property-scan_axis)|表示扫描轴，仅当它是 ScanOp 时定义。|
|:----|:----|

### *property* scan_axis


表示扫描轴，仅当它是 ScanOp 时定义。

### *class* tvm.te.ExternOp

 外部操作。

