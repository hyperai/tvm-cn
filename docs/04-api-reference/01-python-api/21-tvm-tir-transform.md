---

title: tvm.tir.transform

---


所有 TIR 变换的命名空间

### tvm.tir.transform.prim_func_pass(*pass_func=None*, *opt_level:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *required:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *traceable=False*) → [Callable](https://docs.python.org/3/library/typing.html#typing.Callable) | [PrimFuncPass](/docs/api-reference/python-api/tvm-tir-transform#class-tvmtirtransformprimfuncpass)


装饰一个函数传递。


当提供 pass_func 时，此函数返回一个回调。否则，它返回使用给定优化函数创建的函数传递。
* **参数：**
   * **pass_func** (*可选**[****可调用[(*[tvm.tir.PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*,*[IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)*,*[PassContext](/docs/api-reference/python-api/tvm-transform#classtvmtransformpasscontextopt_level2required_passnonedisabled_passnoneinstrumentsnoneconfignone)*)–> tvm.tir.PrimFunc]]* )：转换函数或类。
   * **opt_level**（[int](https://docs.python.org/3/library/functions.html#int)）：此模块传递的优化级别。
   * *name**（*可选**[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]*）：函数传递的名称。名称可以为空。在这种情况下，优化函数的名称将用作传递名称。
   * **required**（可选[List[[str](https://docs.python.org/3/library/stdtypes.html#str)]]）：函数传递所依赖的传递列表。
* **返回：create_function_pass**：如果未提供 pass_func，则返回一个装饰器；否则返回装饰后的结果。返回的装饰器根据输入有两种行为：装饰一个 pass 函数时，将返回一个新的 FunctionPass。装饰一个类类型时，将返回一个新的 FunctionPass 类。
* **返回类型：** Union[Callable, [FunctionPass](/docs/api-reference/python-api/tvm-relax-transform#class-tvmrelaxtransformfunctionpass)]。


**示例**


下面的代码块装饰了一个函数传递类。

```python
@tvm.tir.transform.prim_func_pass(opt_level=1)
class TestReplaceFunc:
    def **init**(self, new_func):
        self.new_func = new_func

    def transform_function(self, func, mod, ctx):
        # 仅用于演示
        # 将 func 转换为 new_func
        return self.new_func
```


以下代码通过装饰用户定义的转换函数来创建函数传递。

```python
@tvm.tir.transform.prim_func_pass(opt_level=2)
def transform(func, mod, ctx):
    # 我的转换操作写在这里。
    return func

function_pass = transform
assert isinstance(function_pass, transform.FunctionPass)
assert function_pass.info.opt_level == 2

# 给定一个模块 m，可以如下调用优化：
updated_mod = function_pass(m)
# 现在常量折叠（constant folding）应该已经应用到提供的模块 m 中的每个函数。
# 并且返回更新后的模块。
```
## *class* tvm.tir.transform.PrimFuncPass


`tvm.tir.PrimFunc()`对模块中的每个模块进行操作的传递。应通过 py:func: tvm.tir.transform.function_pass 创建函数传递类。

## tvm.tir.transform.AnnotateDeviceRegions()


注释应在设备上运行的位置。


插入 AttrStmt 节点，指定 PrimFunc 中应执行区域的目标。仅修改具有 tvm::attr::kTarget 属性且该目标定义了主机的函数。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.AnnotateEntryFunc()
如果它是 IRModule 中唯一的函数，则将 PrimFunc 设置为入口点。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.Apply(*ftransform*)


将 ftransform 应用于模块中的每个函数。


此函数是 tvm.tir.transform.prim_func_pass 的薄包装器。
* **参数：ftransform** (*tvm.tir.PrimFunc -> tvm.tir.PrimFunc*)：转换过程。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.BF16ComputeLegalize()


使 bf16 计算操作合法化。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.BF16StorageLegalize()


将 bf16 存储类型合法化为 u16。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.BindTarget(*target*)


使用给定的目标注释 PrimFunc:param target: target:type target: tvm.target.Target。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.CombineContextCall()


在宿主函数中组合上下文调用。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.CommonSubexprElimTIR(*enable_cse_tir:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *identify_equiv_terms:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)


用新变量代替冗余计算。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.CompactBufferAllocation(*is_strict:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*)


压缩缓冲区访问区域。通过删除未访问的缓冲区，即缩小缓冲区形状并在必要时调整访问区域。


**示例**


在窄化（narrowing）之前，B 是一个 [16, 16] 的缓冲区，但只访问了一个窄向量 B[i, 0:16]。

```python
for i in range(0, 16):
    with T.block():
        B = T.alloc_buffer(16, 16)
        for j in range(0, 16):
            B[i, j] = A[i, j] + 1
        for j in range(0, 16):
            C[i, j] = B[i, j] + 1
```
这个 pass 会缩小缓冲区的形状，并相应调整其访问区域。在这个特定例子中，由于只访问了 B 的一个 1 × 16 向量，该 pass 会将 B 缩小为形状 [1, 16]，并将访问 B[i, j] 改为 B[0, j]。
```python
for i in range(0, 16):
    with T.block():
        B = T.alloc_buffer(1, 16)
        for j in range(0, 16):
            B[0, j] = A[i, j] + 1
        for j in range(0, 16):
            C[i, j] = B[0, j] + 1
```
* **参数：is_strict** ([bool](https://docs.python.org/3/library/functions.html#bool))：确保压缩后的形状始终小于原始形状。否则，允许增大形状以匹配实际访问的缓冲区。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.ConvertBlocksToOpaque()

将所有块变量替换为它们绑定的 PrimExprs，由 BlockRealize 中的相应 iter_values 指示，然后通过删除 BlockRealize 中的所有 iter_values 和 Block 中的 iter_vars 将块转换为不透明的块。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.ConvertForLoopsToSerial()


将并行 For 循环转换为串行 For 循环。
* **返回：fpass**：结果通过
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)

## tvm.tir.transform.ConvertSSA()

将 IRModule 转换为 SSA 形式。


此过程处理同一 tir.Var 出现在同一模块内多个函数中的情况。例如，将一个函数中的片段提取到另一个函数后，同一个 tir.Var 可能既在原始函数的主函数中定义，又在提升函数中作为参数定义。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.DecorateDeviceScope()


将所有函数体装饰为设备函数。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.DefaultGPUSchedule() 


该过程为 PrimFuncs 设置默认线程绑定，包括符号形状函数，允许它们在 GPU 设备上构建和执行。它检查 PrimFunc 内的所有块，并根据循环范围和目标信息（例如最大线程块数和每个块的最大线程数）进行循环融合、拆分和重新排序操作。


此过程的主要目标并非优化性能，而是为非调度或符号形状的 PrimFuncs 生成有效的 GPU 内核。此过程目前仅适用于 CUDA 目标平台。
* **返回：ret。**
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.ExtractPrimFuncConstants()


收集并统一非标量常量到模块的属性「常量」数组。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.FP8ComputeLegalize(*promote_dtype_str:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'float32'*)


使 fp8 计算操作合法化。
* **参数：promote_dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))：我们将 fp8 提升到的数据类型，选项：float16/float32。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.FP8StorageLegalize()


将 fp8 存储类型合法化为 u8。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.Filter(*fcond:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable))


过滤掉不满足给定条件的 PrimFuncs。fcond 应该是一个接受 primfunc 并返回布尔值的函数。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.FlattenBuffer()


对于不包含不透明块的 TIR，将多维 BufferLoad 和 BufferStore 展平为单维 BufferLoad/BufferStore。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.ForceNarrowIndexToInt32()

强制将索引表达式和整数缓冲区缩小为 int32 dtype。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


在默认情况下不应使用此通行证。


## tvm.tir.transform.HoistExpression()


HoistIfThenElse 的通用版本。


将循环不变表达式提升到符合条件的循环之外。搜索以下类型的表达式：
* LetStmt 绑定。
* IfThenElse 条件。
* 布尔运算符。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


## ***class*tvm.tir.transform.HoistExpressionConfig**

提升表达式传递的配置。

### ***property*hoisted_conditionals**

要提升的布尔表达式类型的位标志。

### ***property*hoisted_let_bindings**
要提升的 let 绑定类型的位标志。

## tvm.tir.transform.HoistIfThenElse(*variant:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


将循环不变的 IfThenElse 节点提升到符合条件的循环之外。
* **参数：variant** (*Optional**[****String]*)：pass 的变体。variant 可以具有以下任意一个值 [“basic”, None(Default)]。基本变体支持基本提升场景，其中它期望 For 和 If 节点连续到位，并且不涉及全局范围变量或更高级的场景。默认变体支持所有提升场景，即 {“Basic” + “Advanced”} 通过 PassContext 配置控制支持，如下所示：

config={“tir.HoistIfThenElse”: {“support_block_scope_hoisting”: True}}。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


## ***class*tvm.tir.transform.HoistIfThenElseConfig**
提升机配置 if then else pass。

### ***property*support_block_scope_hoisting**

提升 if cond 与块作用域变量。

## *class* tvm.tir.transform.HoistedConditionals(*value*)

HoistExpressionConfig.conditional_types 中使用的标志。


每个位标志代表一种应提升到尽可能最外层循环的表达式类型。

### Never *= 0*


不提升条件。

### IfElseStmt *= 1*


如果设置，则在 IfElseStmt 中寻找提升候选。

### IfElseExpr *= 2*


如果设置，则在 tir.if_then_else 中寻找提升候选。

### BooleanExpression *= 4*


如果设置，则在所有布尔表达式中寻找提升候选。

### UsingBlockVar *= 8*


如果设置，则允许提升使用块变量的条件（例如 threadIdx.x）。

### All *= 15*


启用所有条件提升。

## *class* tvm.tir.transform.HoistedLetBindings(*value*)


HoistExpressionConfig.let_binding_types 中使用的标志。


每个位标志代表一种 let 绑定表达式，应将其提升到尽可能最外层的循环。

### Never *= 0*


不提升 let 绑定。

### RequiredByConditional *= 1*


提升条件使用的绑定。

### LetStmt *= 2*

LetStmt 中发生的绑定。

### LetExpr *= 4*


Let 表达式中发生的绑定。

### All *= 7*

启用所有 let 绑定的提升。

## tvm.tir.transform.InferFragment()


使用张量内在函数推断 TensorCore 片段信息。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.InjectDoubleBuffer()


注入双缓冲区语句。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


## ***class*tvm.tir.transform.InjectDoubleBufferConfig**

注入双缓冲区通道的配置。

### ***property*split_loop**

Split loop factors  分割环系数。

## tvm.tir.transform.InjectPTXAsyncCopy()


使用异步复制将全局到 CUDA 上的共享内存复制重写。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.InjectPTXLDG32(*enable_inject_ptx_intrin:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*)


注入 ptx.ldg.32 内部函数。
* **参数：enable_inject_ptx_intrin** ([bool](https://docs.python.org/3/library/functions.html#bool))：如果为 True，则注入 ptx.ldg.32 内部函数。

## tvm.tir.transform.InjectPermutedLayout()


在 mma 中注入排列布局。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.InjectRollingBuffer()


注入滚动缓冲区语句。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.InjectSoftwarePipeline()


将带注释的循环转换为并行生产者和消费者的流水线循环。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.InjectVirtualThread()

注入虚拟线程循环。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.InlinePrivateFunctions()


内联调用私有函数。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.InstrumentBoundCheckers()


instrument 绑定检查器。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.InstrumentProfileIntrinsics()


插入对 instrument 函数和循环级分析的内部调用。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LiftThreadBinding()


将相同的线程绑定提升到其 LCA 环。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LoopPartition()


注入虚拟线程循环。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


## ***class*tvm.tir.transform.LoopPartitionConfig**

配置循环分区传递。

### ***property*no_unroll_loop_with_extent_one**

不要展开范围为 1 的循环。

### ***property*partition_const_loop**

Split constant loop  拆分常量循环。

### ***property*unroll_loop_with_partition_hint_no_interval**

展开循环，有 pragma_loop_partition_hint 且无间隔。


## tvm.tir.transform.LowerAsyncDMA()


降低异步 DMA 至 DMA。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LowerAutoCopy()


自动对自动复制块进行内存优化。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LowerCrossThreadReduction()


降低从线程绑定到内部函数调用的跨线程减少。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

### tvm.tir.transform.LowerCustomDatatypes()


降低自定义数据类型。


有关添加自定义数据类型的更多信息，请参阅 tvm::datatypes::Registry。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LowerDeviceKernelLaunch()


降低跨设备函数调用。


在此过程之前，主机到设备的调用表示为子程序调用，并在内部指定环境参数（例如 env_thread）。设备函数是内部函数，没有 tvm::attr::kGlobalSymbol 属性。


经过此过程后，主机到设备的调用将表示为内置的 tvm_call_packed 函数。设备函数是一个对外暴露的函数，具有非空的 tvm::attr::kGlobalSymbol 属性。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

### tvm.tir.transform.LowerDeviceStorageAccessInfo()


降低设备上附加存储的访问信息。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

:::note

所有存储访问分析完成后运行此过程。

:::

## tvm.tir.transform.LowerInitBlock()


将块初始化语句转换为 IfThenElse 语句 。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LowerIntrin()


降低目标特定的内在调用。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LowerMatchBuffer()


删除块内的匹配缓冲区。此外，它将验证绑定。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LowerOpaqueBlock()


解除封锁以确保 TIR 不会再次被安排。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LowerTVMBuiltin()


降低 TVM 内置内在函数。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LowerThreadAllreduce()


降低跨线程 alleduce。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LowerVtcmAlloc()


降低 vtcm 分配。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.LowerWarpMemory()


降低 warp 内存访问到低级设备相关的函数调用。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.MakePackedAPI()


将模块中的 PrimFuncs 转换为打包的 func API。


在此过程之前，PrimFunc 可能已在 PrimFuncNode::buffer_map 中定义了 Buffer 参数。此过程将使用 buffer_map，并使用它生成实现基于打包的 TVM FFI API 的参数。


对于静态形状，BufferNode::shape、BufferNode::strides 和 BufferNode::elem_offset 成员变量用于对用户提供的 DLTensor*或 tvm.nd.array 参数中相应成员变量生成运行时检查。（例如，接受形状为[16,32]的缓冲区的 PrimFunc 验证 DLTensor::shape 数组是否为[16,32]。）。


对于动态缓冲区，其中一个或多个 BufferNode 成员变量使用未由其他 PrimFunc 参数定义的 tir.Var ，这些变量将用于根据相应的 DLTensor 成员定义变量。（例如，接受形状为[tir.Var(“n”), tir.Var(“m”)]的缓冲区的 PrimFunc ，在传递 形状为[16,32]的 DLTensor 时，将根据参数的形状定义n = 16和 n=32 。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.MakeUnpackedAPI()


将模块中的 PrimFuncs 转换为与内部调用兼容的C API。


在此过程之前，PrimFunc 可能已在 PrimFuncNode::buffer_map 中定义了 Buffer 参数。此过程将使用 buffer_map，并使用它生成可由 C API 直接调用的 T*参数（例如 float32* ）。


对于静态形状，无需执行运行时验证来确认参数缓冲区的形状是否与预期形状匹配。对于动态形状，MakeUnpackedAPI 要求将动态参数作为单独的 tir.Var 参数传递。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.ManifestSharedMemoryLocalStage()


为 GPU 上的共享内存访问添加显式本地阶段。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.MergeSharedMemoryAllocations()


此过程将多个 TIR 级共享内存分配合并为一个分配。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.NarrowDataType(*target_bits:*[int](https://docs.python.org/3/library/functions.html#int))


将 stmt 中的 PrimExpr 数据类型缩小到 target_bits。
* **参数：**
   * **target_bits** ([int](https://docs.python.org/3/library/functions.html#int))：目标位配置。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

:::note

在 FlattenBuffer 之后运行此过程。

:::

## tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()


将缓冲区分配定位到精确位置（通常是缓冲区访问的 lca）。此过程将在分配位置注入带有 alloc_buffers 的不透明块。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.PointerValueTypeRewrite()


重写参数的指针内容类型，以及函数内部的 Alloc，以使用最常访问的类型进行加载/存储，从而尽可能避免在后端进行指针转换。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.ReduceBranchingThroughOvercompute()


通过引入过度计算来减少分支。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


## ***class*tvm.tir.transform.ReduceBranchingThroughOvercomputeConfig**

通过超额计算传递减少分支的配置。

### ***property*use_dataflow_analysis**

如果为 true，则传播已知缓冲区值并用于静态证明超额计算是有效的。

## tvm.tir.transform.RemoveAssume()


删除所有 builtin::assume 实例。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.RemoveNoOp()

从 Stmt 中删除 No Op。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


## ***class*tvm.tir.transform.RemoveNoOpConfig**

删除无作通道的配置。

### ***property*max_simplification_steps**

如果不为零，则 RewriteSimplifier 将在指定的步骤数后引发错误。用于调试和测试目的。

###  ***property*use_dataflow_analysis**

如果为 true，则传播已知的缓冲区值并用于静态证明语句为 no-ops。


## tvm.tir.transform.RemoveStoreUndef()

从 Stmt 中删除未定义值的存储。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.RemoveWeightLayoutRewriteBlock(*skip_ndarray_rewrite=False*)


在调整阶段进行基准测试之前删除权重布局重写块。
* **参数：**
   * **skip_ndarray_rewrite** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：

如果为 True，则将跳过根据给定索引图精确重写 NDArray 的操作。仅正确转换 NDArray 的形状，并用随机值填充目标数组的内容。在 MetaSchedule 调优过程中，如果多次调用此过程，则重写前后 NDArray 的原始数据无关紧要。由于使用 IndexMap 的 MapNDArray 进行 NDArray 布局重写目前速度较慢，因此有时需要跳过精确的重写。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.RenormalizeSplitPattern()


将分割模式从 floordiv(floormod()) 重新规范化为 floormod(floordiv())。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.RewriteUnsafeSelect()

检测并重写包含内存访问的不安全选择。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.Simplify()


对语句和表达式进行算术简化。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


## ***class*tvm.tir.transform.SimplifyConfig**

简化传递的配置。

### ***property*apply_constraints_to_boolean_branches**

如果为 true，则在另一个分支提供的约束下简化 AND/OR 的每个分支。

###  ***property*convert_boolean_to_and_of_ors**

如果为 true，则将条件简化为 OR 的 AND。

###  ***property*propagate_knowns_to_prove_conditional**

如果为 true，则传播已知缓冲区值并用于静态证明条件。

###  ***property*propagate_knowns_to_simplify_expressions**


如果为 true，则传播已知缓冲区值，并尽可能用于替换 BufferLoad。

###  ***property*transitively_prove_inequalities**


如果为 true，则使用作用域约束的传递组合简化条件。


## tvm.tir.transform.SkipAssert()


跳过断言语句。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.SplitHostDevice()


将函数分为主机函数和设备函数。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.StorageRewrite()


重写存储分配模式。


将分配移至尽可能最外层的作用域。尝试在分配之间共享空间，以便在可能的情况下制定静态分配计划。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.ThreadSync(*storage_scope:*[str](https://docs.python.org/3/library/stdtypes.html#str))


在共享缓冲区的并行读/写之间插入同步。
* **参数：storage_scope** ([str](https://docs.python.org/3/library/stdtypes.html#str))：目标存储范围。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.TransformMmaBufferLayout()


变换 mma 缓冲区布局。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.UnifyThreadBinding()


统一所有“blockIdx.x/y/z”、“threadIdx.x/y/z”和“vthread.x/y/z”的线程绑定。统一之前，绑定到同一线程轴的两个变量（例如“threadIdx.x”）在其 AttrStmts 中使用不同的 IterVar 和变量。统一之后，我们将为它们使用合并后的 IterVar 和变量。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

:::note

vthread 是一种将被弃用的遗留行为，但 vthread 的线程绑定在本阶段仍然统一。请使用 vthread.x、vthread.y 和 vthread.z 代替。

:::

## tvm.tir.transform.UnrollLoop()


展开以 unroll 标记的常量循环。


此过程还自动将 pragma unroll 标签附加到符合标准的循环。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


## ***class*tvm.tir.transform.UnrollLoopConfig**


展开循环通道的配置。

### ***property*auto_max_depth**

可以自动展开的循环的最大嵌套级别。

###  ***property*auto_max_extent**

将展开的循环的最大范围。

###  ***property*auto_max_step**

循环中要自动展开的步数阈值。

###  ***property*explicit_unroll**

是否显式展开循环而不是设置编译指示。

###  ***property*unroll_local_access**

是否始终展开本地访问。


## tvm.tir.transform.UseAssumeToReduceBranches()


此过程尝试通过过度计算填充区域的值来消除布局特定的填充分支。消除分支将有助于矢量化代码，并提升元素级操作的性能。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.VectorizeLoop(*enable_vectorize:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*)


降低矢量化循环。
* **参数：enable_vectorize** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否启用矢量化。关闭时将降低到标量循环。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.VerifyMemory()


验证 func 是否包含非法主机端直接内存访问。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.transform.VerifyVTCMLimit(*limit=None*) 


验证分配的 vtcm 内存的大小是否满足限制。
* 
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。



