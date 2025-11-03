---

title: tvm.tir.analysis

---

包装现有的分析工具。

## tvm.tir.analysis.analysis.expr_deep_equal(*lhs:*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *rhs:*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)) → [bool](https://docs.python.org/3/library/functions.html#bool)


深度比较两个嵌套表达式。
* **参数：**
   * **lhs** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)) ：左操作数。
   * **rhs** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)) ：右操作数。
* **返回：result**：比较结果。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。


:::Note

此函数不会重新映射变量绑定，除非 x.same_as(y)，否则它不会对 (let x = 1 in x + 1) 和 (let y = 1 in y + 1) 返回 true。使用 py:func: tvm.ir.structural_equal 来处理结构变量重新映射。


由于不重新映射变量的限制，该函数运行速度比 StructuralEqual 更快，并且可以在算术简化期间用作实用函数。


始终首先考虑 py:func: tvm.ir.structural_equal，它处理结构重映射。

:::

:::info 另见

`tvm.ir.structural_equal`

:::

## 

## tvm.tir.analysis.analysis.verify_ssa(*func:*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)) → [bool](https://docs.python.org/3/library/functions.html#bool)


验证函数是否为 SSA 形式。
* **参数：func** ([tvm.tir.PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))：要验证的模块。
* **返回：result**：验证的结果。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

## tvm.tir.analysis.analysis.verify_memory(*func:*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)) → [bool](https://docs.python.org/3/library/functions.html#bool) 


验证 func 是否包含非法主机端直接内存访问。
* **参数：func** ([tvm.tir.PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))：要验证的模块。
* **返回：result**：验证的结果。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

## tvm.tir.analysis.analysis.verify_gpu_code(*func:*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone), *constraints:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[int](https://docs.python.org/3/library/functions.html#int)*]*) → [None](https://docs.python.org/3/library/constants.html#None)


验证模块是否包含非法主机端直接内存访问。
* **参数：**
   * **func** ([tvm.tir.PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))：要验证的模块。
   * **constraints** (*Dict[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[int](https://docs.python.org/3/library/functions.html#int)*]*)：属性约束。
* **返回：result**：验证的结果。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

## tvm.tir.analysis.analysis.get_block_access_region(*block:*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none), *buffer_var_map:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[BufferRegion](/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferregionbufferbuffer-regionlistrange)]]


检测此块中的张量区域被读取或写入。


区域按照 AST 中出现的顺序排序。
* **参数：**
   * **block** ([tvm.tir.Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none))：我们正在检测读/写区域的块。
   * **buffer_var_map** (*Dict[*[tir.Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*]*)：可以访问该块的外部缓冲区。从缓冲区 var 映射到缓冲区。
* **返回：result**：访问区域的数组。包括三种类型的 `BufferRegion` 数组：
   * 第一: 读区域。
   * 第二: 写区域。
   * 第三: 不透明区域。
* **返回类型：** List[List[[BufferRegion](/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferregionbufferbuffer-regionlistrange)]]。

## tvm.tir.analysis.analysis.get_block_read_write_region(*block:*[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none), *buffer_var_map:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[BufferRegion](/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferregionbufferbuffer-regionlistrange)]]


根据块主体语句自动检测块读/写区域。


不透明访问将被视为读访问和写访问。
* **参数：**
   * **block** ([tvm.tir.Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none))：我们正在检测读/写区域的块。
   * **buffer_var_map** (*Dict[*[tir.Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,*[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*]*)：可以访问该块的外部缓冲区。从缓冲区 var 映射到缓冲区。
* **返回：result**：仅由输入块的读取区域和写入区域组成的数组。
* **返回类型：** List[List[[BufferRegion](/docs/api-reference/python-api/tvm-tir#class-tvmtirbufferregionbufferbuffer-regionlistrange)]]。

## tvm.tir.analysis.analysis.calculate_allocated_bytes(*func_or_mod:*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*|*[IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)) → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] | [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)]]


计算 TIR PrimFuncs 所需的每个内存范围分配的内存。
* **参数：func_or_mod** (*Union[*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*,*[IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)*]*)：待检测的函数或模块。如果传入的是模块，则会计算模块内所有 PrimFunc 的分配内存。
* **返回：result**：IRModule 中每个函数在每个作用域内分配的内存大小（以字节为单位），以字典形式返回，字典的键为函数名，值是分配的大小。如果传入单个 PrimFunc，则返回函数名称“main”。
* **返回类型：** Union[Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)], Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)]]]。

## tvm.tir.analysis.analysis.detect_buffer_access_lca(*func:*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)) → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), [Stmt](/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)]


检测缓冲区访问的最低公共祖先 (LCA)，包括高级访问（BufferLoad、BufferStore）和低级访问（BufferLoad、BufferStore 和不透明访问）。LCA 可以是 For 循环，也可以是 Block。
* **参数：func** ([tvm.tir.PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))：要检测的函数。
* **返回：result**：从缓冲区映射到对其的所有访问的 LCA。
* **返回类型：** Dict[[Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer), [Stmt](/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)]。

## tvm.tir.analysis.analysis.estimate_tir_flops(*stmt_or_mod:*[Stmt](/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*|*[IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)) → [float](https://docs.python.org/3/library/functions.html#float)


估计 TIR 片段的 FLOP。
* **参数：stmt_or_mod** (*Union[*[Stmt](/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*,*[IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)*]*)：需要估计的 TIR 片段或 IRModule。
* **返回：flops**：估计的 FLOP。
* **返回类型：**[float](https://docs.python.org/3/library/functions.html#float)。

## tvm.tir.analysis.analysis.undefined_vars(*node:*[Stmt](/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*|*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *defs:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)]


在 TIR 语句或表达式中查找未定义的变量。
* **参数：**
   * **node** (*Union[*[Stmt](/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)*,*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：要检查的 TIR 语句或表达式。
   * **defs** (*Optional**[****List**[***[tir.Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)***]****]*)：定义的变量。
* **返回：result**：未定义的变量。
* **返回类型：** List[[tir.Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)]。

## tvm.tir.analysis.analysis.verify_well_formed(*obj:*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*|*[IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone), *assert_mode:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [bool](https://docs.python.org/3/library/functions.html#bool)


验证给定的 TIR 是否格式正确。验证包括：
* 检查表达式是否不包含在块外定义的变量。
* **参数：**
   * **obj** (*Union[*[tvm.tir.PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)*,*[tvm.ir.IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)*]*)**：** 要验证的功能或模块。
   * **assert_mode** ([bool](https://docs.python.org/3/library/functions.html#bool))**：** 当函数格式不正确时，指示是否会引发错误。
* **返回：result**：它是否是一个格式良好的 TIR 函数。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

## tvm.tir.analysis.analysis.OOBChecker()

检测数组中的越界内存访问。
* **返回：fpass**：结果通过。
* **返回类型：**[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。

## tvm.tir.analysis.analysis.find_anchor_block(*mod:*[IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)) → [Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)


找到给定模块的「锚块」。


我们将锚块（anchor block）定义为满足以下条件的块：

1.  拥有一个初始化语句（init statement）； 

2.  拥有最大的浮点运算量（flops count）。 


注： 当存在多个带初始化语句的块时，第二个条件才会被使用。


例如，如果输入模块由 conv2d 和融合空间块组成，则 conv2d 为锚块。输入模块不得包含多个这样的块。例如，包含两个 conv2d 的模块不允许作为输入。


然而，由 Winograd 卷积创建的模块包含多个带有 init 语句的块（输入变换、分批 GEMM 和输出变换）。我们使用第二个条件（即 flops count）来确定分批 GEMM 块是锚块。
* **参数：mod** ([tvm.ir.IRModule](/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone))：输入 TIR 模块。
* **返回：anchor_block**：如果找到则为锚块，否则为无。
* **返回类型：**[Block](/docs/api-reference/python-api/tvm-tir#class-tvmtirblockiter_varslistitervar-readslistbufferregion-writeslistbufferregion-name_hintstr-bodystmt-initstmtnone-none-alloc_bufferslistbuffer-none-none-match_bufferslistmatchbufferregion-none-none-annotationsmappingstr-object-none-none-spanspannone-none)。

## tvm.tir.analysis.analysis.get_vtcm_compaction_passes() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)]


用于获取降低通道列表的实用函数，用于计算压缩的 VTCM 分配大小。
* **返回：result**：返回通行证列表。
* **返回类型：** List[[tvm.transform.Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)]。

## tvm.tir.analysis.analysis.is_pure_function(*func:*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)) → [bool](https://docs.python.org/3/library/functions.html#bool)


检查函数是否为纯函数。

## tvm.tir.analysis.analysis.assert_pure_function(*func:*[PrimFunc](/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)) → [bool](https://docs.python.org/3/library/functions.html#bool)

 断言该函数是纯函数。

