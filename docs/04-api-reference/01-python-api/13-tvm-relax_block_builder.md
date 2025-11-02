---

title: tvm.relax_block_builder

---



构建 Relax AST 的开发人员 API。

## *class* tvm.relax.block_builder.FunctionScope(*block_builder*, *name*, *params*, *attrs*, *is_pure*)


函数的辅助作用范围。

## *class* tvm.relax.block_builder.DataflowScope(*block_builder*)


数据流块的辅助范围。

## *class* tvm.relax.block_builder.TestingScope(*block_builder*, *def_vars*)


用于测试目的的辅助范围。

## *class* tvm.relax.block_builder.BlockBuilder(*mod:*[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)


用于构建 Relax IR 以进行测试和开发的构建器。


**示例**

```python
m = tir.Var("m", "int32")
n = tir.Var("n", "int32")
x = rx.Var("x", rx.TensorStructInfo([m, n], "float16"))
y = rx.Var("y", rx.TensorStructInfo([n], "float16")
bb = rx.BlockBuilder()
with bb.function([x, y], "func"):
    with bb.dataflow() as df:
        lv0 = bb.emit(rx.add(x, y))
        lv1 = bb.emit(rx.multiply(lv0, y))
        gv0 = bb.emit_output(lv1)
    bb.emit_func_output(gv0)
mod = bb.get()
```


BlockBuilder 还可以通过 nn.Module API 构建神经网络。

```python
from tvm.relax.testing import nn

n = tir.Var("n", "int64")
input_size = 784
hidden_sizes = [128, 32]
output_size = 10
bb = rx.BlockBuilder()

with bb.function("main"):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(),
    )
    data = nn.Placeholder((n, input_size), name="data")
    output = model(data)
    params = [data] + model.parameters()
    builder.emit_func_output(output, params=params)
mod = bb.get()
```
### *static* current() → [BlockBuilder](https://tvm.hyper.ai/docs/reference/api/python/relax/block_builder.html#tvm.relax.block_builder.BlockBuilder) | [None](https://docs.python.org/3/library/constants.html#None)


返回当前的 BlockBuilder。

### function(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *params:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*|*[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *attrs:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *pure:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *private:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [FunctionScope](hhttps://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_builderfunctionscopeblock_builder-name-params-attrs-is_pure)

注释一个 Relax 函数。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：函数的名称。
   * **params** ([tvm.relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*|*[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*|List **[***[tvm.relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***]****,optional*)：函数的参数。如果 params 为 None，则表示将函数参数的初始化推迟到 emit_func_output。
   * **attrs** (*Dict*[***[str](https://docs.python.org/3/library/stdtypes.html#str),*** ***Object***],optional)：函数 attrs。
   * **pure** ([bool](https://docs.python.org/3/library/functions.html#bool)*,optional*)：函数是否被注释为纯函数。
   * **private** ([bool](https://docs.python.org/3/library/functions.html#bool)*,optional*)：函数是否被注释为私有函数。如果函数是私有的，则它没有全局符号属性。如果它不是私有的且不是内部函数，则它将具有全局符号属性（映射到函数名称）。
* **返回：ret**：用于构建 Relax 函数节点的 FunctionScope。
* **返回类型：**[FunctionScope](hhttps://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_builderfunctionscopeblock_builder-name-params-attrs-is_pure)

## testing_scope(*def_vars:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*]*) → [TestingScope](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_buildertestingscopeblock_builder-def_vars)

启动用于单元测试的范围。
* **参数：def_vars** (*List[*[tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*]*)：标记为范围内定义的符号变量列表。
* **返回：ret**：用于设置用于发射和其他目的的构建器的 TestingScope。
* **返回类型：**[TestingScope](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_buildertestingscopeblock_builder-def_vars)

## dataflow() → [DataflowScope](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_builderdataflowscopeblock_builder)


注释 Relax 数据流块。
* **返回：ret**：用于构建 Relax 数据流块的 DataflowScope。
* **返回类型：**[DataflowScope](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_builderdataflowscopeblock_builder)。

## emit(*expr:*[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= ''*) → [Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)


发出一个 expr。这将推断 expr 的形状和类型，创建一个变量，并将 expr 绑定到该变量。
* **参数：**  
   * **expr** (*tvm.relax.Expr*)：要发出的 Expr。
   * **name_hint** ([str](https://docs.python.org/3/library/stdtypes.html#str))：绑定变量的名称提示。
* **返回：ret**：与输入 expr 绑定的新创建的变量。
* **返回类型：**[tvm.relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)

### call_te(*func:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable), **args:*[Any](https://docs.python.org/3/library/typing.html#typing.Any), ***kwargs:*[Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


根据 te 函数生成一个调用节点。该函数将参数从 Relax 表达式转换为 te 张量。回调函数应返回一个 te 张量或一个 te 张量列表。请参阅 emit_te 中的详细示例。
* **参数：**   
   * **func** (*Callable*)*：*返回 te 张量或 te 张量列表的函数。
   * **args** (*Any,optional*)：传递给函数的参数。
   * **kwargs** (*Any,optional*)：
      * 传递给函数的关键字参数。请注意，以下关键字参数是保留的：
      * ’primfunc_name_hint’ 用语将名称提示传递给生成的PrimFunc。
      * ’primfunc_attrs’保留用于传递要添加到创造的 PrimFunc 的函数属性。
* **返回：ret**：新创建的调用节点。
* **返回类型：**[tvm.relax.Call](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)

### call_te_with_grad(*func:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable), **args:*[Any](https://docs.python.org/3/library/typing.html#typing.Any), *te_grad_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *te_grad_kwargs:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, ***kwargs:*[Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


根据 te 函数生成 call 节点。该方法会生成一个 call_tir_with_grad 节点，即绑定了 te 梯度函数（以 te_grad_name 为参数）的 call_tir 节点。
* **参数：**   
   * **func** (*Callable*)：返回 te 张量或 te 张量列表的函数。
   * **args** (*Any,optional*)：传递给函数的参数。
   * **te_grad_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：与 call_tir_with_grad 节点关联的 te 梯度函数的注册名称。必须作为关键字参数提供。
   * **te_grad_kwargs** (*Dict*[[str](https://docs.python.org/3/library/stdtypes.html#str),***Object***],optional)：传递给 te 梯度函数的关键字参数。可选地，以关键字参数的形式提供。默认值：{}。
   * **kwargs** (*Any,optional*)：传递给函数的关键字参数。请注意，以下关键字参数是保留的：
      * ’primfunc_name_hint’ 用于将名称提示传递给生成的 PrimFunc。
      * ’primfunc_attrs’ 保留用于传递要添加到创建的 PrimFunc 的函数属性。
* **返回：ret**：新创建的调用节点。
* **返回类型：**[tvm.relax.Call](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxcalloprelaxexpropargslistrelaxexprtuplerelaxexprattrsattrsnonenonesinfo_argsliststructinfotuplestructinfononenonespanspannonenone)

### emit_te(*func:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable), *args:[Any](https://docs.python.org/3/library/typing.html#typing.Any), ***kwargs:*[Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)


根据 te 函数发出一个调用节点。该函数将参数从松弛表达式转换为 te 张量。回调函数应返回一个 te 张量或一个 te 张量列表。
* **参数：**   
   *    **func(*Callable*)：** 返回 te 张量或 te 张量列表的函数。
   *    **args** (*Any,optional*)：传递给函数的参数。
   *    **kwargs** (*Any,optional*)：传递给函数的关键字参数。请注意，“primfunc_name_hint”键保留用于将名称提示传递给生成的 PrimFunc。
* **返回：ret**：与调用代码绑定的新创建的变量。
* **返回类型：**[tvm.relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)


**示例**

```python
bb = rx.BlockBuilder()
n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
x = rx.Var("x", rx.TensorStructInfo([n, m], "float32"))
y = rx.Var("y", rx.TensorStructInfo([n, m], "float32"))

def te_func(args, args_dict, msg):
    A = args[0]
    B = args_dict["B"]
    return te.compute((128, 128), lambda i, j: A[i, j] + B[i, j])

with bb.function([x, y], "rx_func"):
    out = bb.emit_te(te_func, [x], {"B": y}, msg="hello")
    bb.emit_func_output(out)
```


将导致 TVMScript

```python
@tvm.script.ir_module
class Module:
    @T.prim_func
    def te_func(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle,
                var_compute: T.handle) -> None:
        # function attr dict
        T.func_attr({"tir.noalias": True})
        m = T.int64()
        n = T.int64()
        rxplaceholder = T.match_buffer(var_rxplaceholder, [n, m], dtype="float32")
        rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [n, m], dtype="float32")
        compute = T.match_buffer(var_compute, [128, 128], dtype="float32")
        # body
        # with T.block("root")
        for i0, i1 in T.grid(128, 128):
            with T.block("compute"):
                i, j = T.axis.remap("SS", [i0, i1])
                T.reads([rxplaceholder[i, j], rxplaceholder_1[i, j]])
                T.writes([compute[i, j]])
                compute[i, j] = rxplaceholder[i, j] + rxplaceholder_1[i, j]

    @R.function
    def rx_func(x: Tensor((n, m), "float32"), y: Tensor((n, m), "float32")) -> Tensor:
        # block 0
        gv = relax.call_tir("te_func", (x, y), R.Tensor((128, 128), "float32"))
        return gv
```


示例


```plain
bb = relax.BlockBuilder()
n = tir.Var("n", "int64")
x = relax.Var("x", relax.TensorStructInfo([n], "float32"))
y = relax.Var("y", relax.TensorStructInfo([n + 1], "float32"))

def te_func(A):
    C = te.compute((n + 1), lambda i: A[i])
    return C

with bb.function("rx_func", [x, y]):
    x1 = bb.emit_te(te_func, y)
    bb.emit_func_output(x1)
```


将导致 TVMScript

```python
@tvm.script.ir_module
class Module:
    @T.prim_func
    def te_func(var_rxplaceholder: T.handle, var_compute: T.handle, n: T.int64) -> None:
        rxplaceholder = T.match_buffer(var_rxplaceholder, [n + T.int64(1)],
                                       dtype="float32")
        compute = T.match_buffer(var_compute, [n + T.int64(1)], dtype="float32")
        # body
        # with T.block("root")
        for i0 in T.serial(0, n + T.int64(1)):
            with T.block("compute"):
                i = T.axis.spatial(n + T.int64(1), i0)
                T.reads([rxplaceholder[i]])
                T.writes([compute[i]])
                compute[i] = rxplaceholder[i]

    @R.function
    def rx_func(x: Tensor((n,), "float32"), y: Tensor(((n + 1),), "float32"))
        -> Tensor(None, "float32", ndim=-1):
        # block 0
        gv = relax.call_tir(te_func, (y,), R.Tensor((n + 1,), "float32"), (n,))
        return gv
```
### match_cast(*value:*[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *struct_info:*[StructInfo](https://tvm..org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo), *name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= ''*) → [Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)


发出 MatchCast。
* **参数：**
   * **value** (*tvm.relax.Expr*)：要发出的 MatchCast 的值。
   * **struct_info** ([StructInfo](https://tvm..org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo))[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)要匹配的结构信息。
   * **name_hint** ([str](https://docs.python.org/3/library/stdtypes.html#str))[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)匹配转换的名称。
* **返回：ret**[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)一个新创建的变量，其界限是转换结果。
* **返回类型：**[tvm.relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)

### emit_output(*output:*[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]*, *name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= ''*) → [Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)


发出当前数据流块或函数的输出。
* **参数：**
   * **output** (*Expr|*[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)|List[Expr])：当前块/函数的输出。
   * **name_hint** ([str](https://docs.python.org/3/library/stdtypes.html#str))[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)绑定变量的名称提示。
* **返回：ret**：与输出绑定的返回变量。
* **返回类型：**[tvm.relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)

### emit_func_output(*output:*[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*|*[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)*]*, *params:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*|*[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [GlobalVar](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)


为函数发出输出。
* **参数：**
   * **output** (*Expr|*[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)|List[*Expr]*)：当前块/函数的输出。
   * **params** ([tvm.relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*|*[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*|List[***[tvm.relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***]*,optional)[：](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)要构建的函数的参数。如果 params 为 None，则表示 params 已在函数中初始化并具有作用域。
* **返回：gvar：** 代表函数的 GlobalVar。
* **返回类型：**[tvm.ir.GlobalVar](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)

### normalize(*expr:*[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) → [RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)


规范化 Expr 以完成其形状和类型。
* **参数：expr** (*Expr*)：输入表达式。
* **返回：ret**：具有规范化形状和类型的 expr。
* **返回类型：** Expr

### get() → [IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)


返回中间 IRModule。适用于在构建过程中需要 IRModule 的情况。
* **返回：ret**：正在构建具有 Relax 和 TIR 功能的 IRModule。
* **返回类型：** tvm.IRModule。

### finalize() → [IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)


完成构建过程并返回结果 IRModule。


可能在 IRModule 中重命名 GlobalVars 以确保名称的唯一性和不变性：每个公共函数都与其「global_symbol」属性具有相同的名称。


请注意，此方法应仅在构建过程结束时调用一次，因为它可能会使此构建器先前返回的全局变量无效。另请参阅 tvm.relax.transform.NormalizeGlobalVar。
* **返回：ret：** 正在构建具有 Relax 和 TIR 功能的 IRModule。
* **返回类型：** tvm.IRModule

### get_unique_name(*name_prefix:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

生成具有指定前缀的唯一名称。
* **参数：name_hint** ([str](https://docs.python.org/3/library/stdtypes.html#str))：名称前缀。
* **返回：ret：** 生成的名称。
* **返回类型：** [str](https://docs.python.org/3/library/stdtypes.html#str)

### add_func(*func:*[BaseFunc](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.BaseFunc), *func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [GlobalVar](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)


向正在构建的 IRModule 添加 Relax 函数或 TIR PrimFunc。
* **参数：**
   * **func** ([BaseFunc](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.BaseFunc))[：](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.BaseFunc)要添加的函数。
   * **func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：要添加的函数的名称。
* **返回：gvar**：与添加的函数绑定的全局变量。
* **返回类型：**[GlobalVar](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)

### update_func(*gv:*[GlobalVar](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar), *updated_func:*[BaseFunc](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.BaseFunc)) → [None](https://docs.python.org/3/library/constants.html#None)


向正在构建的 IRModule 添加 Relax 函数或 TIR PrimFunc。
* **参数：**
   * **gv** ([GlobalVar](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar))：引用要更新的函数的全局变量。
   * **updated_func** ([BaseFunc](https://tvm..org/docs/reference/api/python/ir.html#tvm.ir.BaseFunc))：更新后的函数。

### current_block_is_dataflow() → [bool](https://docs.python.org/3/library/functions.html#bool)

检查正在构建的块是否是 DataflowBlock。
* **返回：ret**：一个布尔值，指示正在构建的块是否为 DataflowBlock。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)

### emit_normalized(*binding:*[Binding](https://tvm..org/docs/reference/api/python/relax/relax.html#tvm.relax.Binding)) → [None](https://docs.python.org/3/library/constants.html#None)


发出已经规范化的绑定。
* **参数：binding** ([Binding](https://tvm..org/docs/reference/api/python/relax/relax.html#tvm.relax.Binding))：要发出的绑定。

### lookup_binding(*var:*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)) → [RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr) | [None](https://docs.python.org/3/library/constants.html#None)


在绑定表中查找变量。
* **参数：var** ([relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone))：输入变量。
* **返回：expr**：与输入变量绑定的 Expr。
* **返回类型：** Expr。

### begin_scope(*params:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None) 


开始一个新的范围，带有范围内可见的可选参数。
* **参数：params** (*Optional*[***List**[***[relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***]***]*)：范围内可见的参数。


:::Note

当引入新范围（函数、序列）时应调用此函数，以正确跟踪变量可用性并帮助尽力推断。

:::

### end_scope() → [None](https://docs.python.org/3/library/constants.html#None)

 结束当前作用域。请参阅 begin_scope 了解详情

