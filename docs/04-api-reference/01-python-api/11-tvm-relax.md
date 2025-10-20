---

title: tvm.relax

---



Relax IR 命名空间包含 IR, type, operator, builder, vm, etc等。


## ***class*tvm.relax.VirtualMachine(*rt_mod: Module | Executable*,*device: Device |***[List](https://docs.python.org/3/library/typing.html#typing.List)***[Device]*,*memory_cfg:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)***[Device,***[str](https://docs.python.org/3/library/stdtypes.html#str)***] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*profile:***[bool](https://docs.python.org/3/library/functions.html#bool)***= False*)**

Relax VM 运行时。


### **invoke_closure(*closure: Object*,**args:***[Any](https://docs.python.org/3/library/typing.html#typing.Any)**)→ Object**

调用闭包。
* **参数：**
   * **closure** (*Object*) ：VMClosure 对象。
   * **args** (*[*tvm.runtime.NDArray**] or *[*np.ndarray**]) ：闭包的参数。
* **返回：** **result** ：输出。
* **返回类型：** Object。


### **save_function(*func_name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*saved_name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,**args:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***]*,*include_return:***[bool](https://docs.python.org/3/library/functions.html#bool)***= True*,***kwargs:***[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***]*)→**[None](https://docs.python.org/3/library/constants.html#None)



便利函数。从模块中获取一个函数并保存为 PackedFunc 对象，调用该对象时将使用给定的参数调用该函数。可以使用 saved_name 从模块访问 PackedFunc 对象。包含此函数是为了方便计时试验：调用返回的 PackedFunc 对象进行字典查找的开销比通常通过虚拟机运行的开销要小。


如果已保存的名称被使用，则可以覆盖它，但它不能覆盖 Relax 源中定义的函数的名称。


这实际上是在创建一个闭包，但该函数具有不同的名称，以避免与 invoke_closure 混淆（它们不应该一起使用）。
* **参数：**
   * **func_name** () ：需要打包的函数。
   * **saved_name** () ：应保存结果闭包的名称。
   * **include_return** () ：已保存的 PackedFunc 是否应返回其输出。如果通过 RPC 进行计时，则可能不希望在机器之间发送输出。
   * **args** (List[**Any**]) ：与函数打包的参数。
   * **kwargs** (Dict[str,Any]) ：与函数打包的任何命名参数


### **set_input(*func_name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,**args:***[Any](https://docs.python.org/3/library/typing.html#typing.Any)**,***kwargs:***[Any](https://docs.python.org/3/library/typing.html#typing.Any)**)→**[None](https://docs.python.org/3/library/constants.html#None)

将输入设置为函数。此接口在使用 VM over RPC 时有效，其内部将参数中的 NDArray 转换为 DLTensor。在 RPC 中，远程只能使用最小 C 语言运行时，支持 DLTensor。


注意：如果使用 set_input ，*则必须使用* invoke_stateful 调用该函数，并且必须使用 get_outputs 获取结果。
* **参数：**
   * **func_name** () ：函数的名称。
   * **args** (*List*[**tvm.runtime.NDArray**] or *List*[**np.ndarray**]) ：函数的参数。
   * **kwargs** ( *of* *str to tvm.runtime.NDArray* *or* *np.ndarray*) ：函数的命名参数。

### **invoke_stateful(*func_name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**)→**[None](https://docs.python.org/3/library/constants.html#None)

relax 从 VM 模块使用 set_input 设置的参数调用命名函数。如果没有先使用 set_input 调用 invoke_stateful（即使是为了设置 0 个输入），调用 invoke_stateful 是错误的；反之，如果已经调用过 set_input，不使用 invoke_stateful 调用函数也是错误的。


可以通过调用 get_outputs 获取调用结果。
* **参数：func_name** () ：要调用的函数的名称。

### **get_outputs(*func_name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**)→ Object |**[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)**[**[Any](https://docs.python.org/3/library/typing.html#typing.Any)**]**

调用 invoke_stateful 后，根据给定名称获取函数输出的值。


如果没有先调用 invoke_stateful 则调用此函数是错误的。
* **参数：func_name** () ：应获取其输出的函数的名称。
* **返回：** **ret** ：先前通过 invoke_stateful 调用该函数的结果。如果结果是一个元组，则返回一个字段列表。这些字段也可能是元组，因此可以任意嵌套。
* **返回类型：** Union[tvm.Object, [Tuple](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Tuple)[Any]]


### **set_instrument(*instrument: Function*)→**[None](https://docs.python.org/3/library/constants.html#None)

设置 instrument 函数。


如果存在 instrument 函数，该函数将在每次 Relax.Call 指令之前/之后被调用。该函数具有以下签名：

```python
def instrument(
    func: Union[VMClosure, PackedFunc],
    func_symbol: str,
    before_run: bool,
    ret_value: any,
    *args) -> bool:
    pass
```



该工具采用以下参数： - func：要调用的函数对象。- func_symbol：函数的符号名称。 - before_run：是在调用之前还是之后。 - ret_value：调用的返回值，仅在运行后有效。 - args：传递给调用的参数。


检测函数可以选择一个整数，该整数对应后续运行的操作方向。更多详情，请参阅 VMInstrumentReturnKind。
* **参数：instrument** (*tvm.runtime.PackedFunc*) ：每次 VM 调用 instr 时调用的检测函数。


:::info 另见

`VMInstrumentReturnKind`

VM 中可能的返回值。

:::

### **time_evaluator(*func_name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*dev: Device*,*number:***[int](https://docs.python.org/3/library/functions.html#int)***= 10*,*repeat:***[int](https://docs.python.org/3/library/functions.html#int)***= 1*,*min_repeat_ms:***[int](https://docs.python.org/3/library/functions.html#int)***= 0*,*cooldown_interval_ms:***[int](https://docs.python.org/3/library/functions.html#int)***= 0*,*repeats_to_cooldown:***[int](https://docs.python.org/3/library/functions.html#int)***= 1*,*f_preproc:***[str](https://docs.python.org/3/library/stdtypes.html#str)***= ''*)→**[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)**[[...], BenchmarkResult]**

返回一个用于对模块中的函数进行计时的求值器。它遵循与 tvm.runtime.module 中的 time_evaluator 相同的约定。它可以与 save_function() 结合使用，从而避免额外的字典查找。
* **参数：**
   * **func_name** () ：模块中函数的名称。
   * **dev** (*Device*) ：我们应该在其上运行此函数的设备。
   * **number** () ：运行此函数取平均值的次数。我们将这些运行称为一次重复测量。
   * **repeat** (*,* *optional*) ：重复测量的次数。该函数总共会被调用 (1 + number x repeat) 次，其中第一次调用是预热，之后会被丢弃。返回的结果包含重复成本，每次重复成本是 number 成本的平均值。
   * **min_repeat_ms** (*,* *optional*) ：一次重复的最短持续时间（以毫秒为单位）。默认情况下，一次重复包含 number 个运行。如果设置了此参数，则参数 number 将动态调整以满足一次重复的最短持续时间要求。即，当一次重复的运行时间低于此时间时，number 参数将自动增加。
   * **cooldown_interval_ms** (*,* *optional*) ： repeats_to_cooldown 定义的重复次数之间的冷却间隔（以毫秒为单位）。
   * **repeats_to_cooldown** (*,* *optional*) ：冷却激活前的重复次数。
   * **f_preproc** (*,* *optional*) ：在执行时间评估器之前我们要执行的预处理函数名称。


:::Note

该函数将被调用（1 + 数字 x 重复）次，如果存在延迟初始化，则第一次调用将被丢弃。

:::


**示例**


与 VM 函数正常使用（如果函数返回元组，则可能无法通过 RPC 工作）：

```python
target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(TestTimeEvaluator, target)
vm = relax.VirtualMachine(mod, tvm.cpu())
timing_res = vm.time_evaluator("func_name", tvm.cpu())(arg0, arg1, ..., argn)
```



与有状态 API 一起使用：

```python
target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(TestTimeEvaluator, target)
vm = relax.VirtualMachine(mod, tvm.cpu())
vm.set_input("func_name", arg0, arg1, ..., argn)
timing_res = vm.time_evaluator("invoke_stateful", tvm.cpu())("func_name")
```



通过 save_function 保存闭包（这会减少定时部分中的字典查找次数）：

```python
target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(TestTimeEvaluator, target)
vm = relax.VirtualMachine(mod, tvm.cpu())
vm.set_input("func_name", arg0, arg1, ..., argn)
timing_res = vm.time_evaluator("invoke_stateful", tvm.cpu())("func_name")
```
* **返回：** **ftimer**：该函数接受与 func 相同的参数，并返回 BenchmarkResult。ProfileResult 报告重复时间成本（以秒为单位）。
* **返回类型：** 函数。


### profile(func_name: str, *args)

分析函数调用。
* **参数：**
   * **func_name** () ：函数的名称。
   * **args** (*List* *of*  *or* *other objects supported by PackedFunc.*) ：函数的参数。
* **返回：** **report**：格式化的分析结果，显示每个操作的时间测量。
* **返回类型：** [tvm.runtime.profiling.Report](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-profiling#class-tvmruntimeprofilingreportcallssequencedictstr-object-device_metricsdictstrdictstr-object-configurationdictstr-object)。


## ***class*tvm.relax.VMInstrumentReturnKind(*value*)**

一个枚举。


## **tvm.relax.Expr**

RelaxExpr 的别名。


## ***class*tvm.relax.Id**

Var 中使用的唯一标识符（名称）。保证在所有过程中保持稳定。


## ***class*tvm.relax.Var(*name_hint:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[Id](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxid)**,*struct_info:***[StructInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*span:***[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

所有 Relax 绑定的变量类。
* **参数：**
   * **name_hint** (*Union*[[str](https://docs.python.org/3/library/stdtypes.html#str), Id]) ：变量的名称提示。
   * **struct_info** (*Optional*[[StructInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)])：变量的结构信息注释。
   * **span** (*Optional*[[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)])：指向原始源代码的 Span。


### ***property*name_hint*:* [str](https://docs.python.org/3/library/stdtypes.html#str)

获取当前变量的名称提示。


## ***class*tvm.relax.DataflowVar(*name_hint:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[Id](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxid)**,*struct_info:***[StructInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*span:***[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

变量节点的子类型，用于标记来自正常可见的“function local”绑定的数据流变量。
* **参数：**
   * **name_hint** (Union[[str](https://docs.python.org/3/library/stdtypes.html#str), Id]) ：变量的名称提示。
   * **struct_info** (Optional[[StructInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)]) *：变量的结构信息注释。*
   * **span** (Optional[[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)]) *：指向原始源代码的 Span。*


## ***class*tvm.relax.Binding**

Relax 中绑定的基类。


## ***class*tvm.relax.MatchCast(*var:***[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)**,*value:***[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)**,*struct_info:***[StructInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxstructinfo)**,*span:***[Span](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**
运行时将值与结构信息匹配。


此操作进行运行时检查，在第一次出现时填充未定义的符号形状 vars 和 struct_info 中的 vars，并在其他情况下插入相等断言。
* **参数：**
   * **var** () ：匹配转换绑定到的返回变量。
   * **value** (*Expr*) ：输入值表达式。
   * **struct_info** () ：要匹配的结构信息。


## ***class*tvm.relax.VarBinding(*var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**,*value:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

变量绑定，将 lhs 的变量与 rhs 的变量绑定。
* **参数：**
   * **var** () ：匹配转换绑定到的返回变量。
   * **value** (*Expr*) ：输入值表达式。


## ***class*tvm.relax.BindingBlock(*bindings:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[Binding](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Binding)***]*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

绑定块的基类，内部绑定可以是不纯的（具有副作用或控制流）。


## ***class*tvm.relax.DataflowBlock(*bindings:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[Binding](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Binding)***]*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**
数据流块，内部绑定是纯粹的（没有副作用，也没有控制流）。


## ***class*tvm.relax.SeqExpr(*blocks:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)***]*,*body:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

一系列绑定块后跟一个表达式。


## ***class*tvm.relax.ShapeExpr(*values:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***] |***[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)***[***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***, ...] |***[Array](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Array)**,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

形状表达式允许用户构建包含 PrimExpr 的形状。
* **参数：**
   * **values** (Union[**List**[]**,**[, ...], ]) ：形状表达式的值。
   * **span** (Optional[]) *：*指向原始源代码的 Span。


## ***class*tvm.relax.Tuple(*fields:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)***] |***[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)***[***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)***, ...]*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

将多个字段组合在一起的元组表达式。
* **参数：**
   * **fields** (*Union*[**List**[**Expr**]**,***[*Expr**, ...]]) ：元组中的字段。
   * **span** (Optional[]) ：指向原始源代码的 Span。


## ***class*tvm.relax.TupleGetItem(*tuple_value:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*index:***[int](https://docs.python.org/3/library/functions.html#int)**,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

从元组中获取第 index 项。
* **参数：**
   * **tuple_value** (*Expr*) ：输入元组表达式。
   * **index** () ：索引。
   * **span** (Optional[]) ：指向原始源代码的 Span。

# 

## ***class*tvm.relax.Function(*params:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)***]*,*body:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*ret_struct_info:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*is_pure:***[bool](https://docs.python.org/3/library/functions.html#bool)***|***[None](https://docs.python.org/3/library/constants.html#None)***= True*,*attrs:***[DictAttrs](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.DictAttrs)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

Relax 函数。

### ***static*create_empty(*params:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)***]*,*ret_struct_info:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**,*is_pure:***[bool](https://docs.python.org/3/library/functions.html#bool)***|***[None](https://docs.python.org/3/library/constants.html#None)***= True*,*attrs:***[DictAttrs](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.DictAttrs)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

构建一个不带主体的 Relax.Function。

### bind_symbolic_vars(binding_map: Mapping[str | Var, PrimExpr])→Function

返回具有更新的符号变量的新函数。
* **参数：**
   * **binding_map** (Mapping[**Union**[, ], ]) ：待替换值的映射。键可以是 tir.Var 或变量的字符串名称。如果通过名称引用变量，则该名称必须唯一地标识函数中的符号变量。
   * **返回：** **func**：更新后的函数
   * **返回类型：** [Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)


### **bind_symbolic_vars(*binding_map:***[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[Var](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Var)***,***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***]*)→**[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)


返回具有更新的符号变量的新函数。
* **参数：**
   * **binding_map** (*Mapping*[) ：Union[str，relax.Var]，Union[int，float，PrimExpr，tvm.runtime.NDArray，_np.ndarray，Expr]，
   * 要替换的映射关系的值。
   * 键（Key）可以是类型，也可以是 Relax 变量的字符串名称。如果通过名称来指定变量，该名称必须唯一标识函数中的一个参数。
   * 值（Value）必须是一个 Relax 表达式，或者转换可以为 Relax 表达式的值。该值必须与被替换的变量类型兼容。
   * **返回：** **func** ：更新后的函数。
   * **返回类型：** [Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)。

## ***class*tvm.relax.ExternFunc(*global_symbol: String*,*struct_info:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

外部函数，代表一个 PackedFunc。

## ***class*tvm.relax.Call(*op:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)***|***[Op](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Op)**,*args:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)***] |***[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)***[***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)***, ...]*,*attrs:***[Attrs](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Attrs)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*sinfo_args:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***] |***[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)***[***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***, ...] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

Relax 中的函数调用节点。


Relax.Call 节点对应于计算图术语中的运算符应用节点。
* **参数：**
   * **op** ( *or* *any tvm.relax.Expr with function type.*) ：要调用的操作。
   * **args** (*Union*[**List**[**Expr**]**,***[*Expr**, ...]]) ：调用的参数。
   * **attrs** (*Optional*[*])：调用的属性，可以为 None。*
   * **sinfo_args** (*Optional[**Union**[**List**[*]*,***[,* ...]**]**]) ：CallNode 的结构信息参数。sinfo_args 设计为仅对内部操作（例如，call_tir、call_builtin_with_ctx 等）和对 ExternFuncs 的调用为非空，主要用于结构信息推断。
   * **span** (*Optional*[*])：指向原始源代码的 Span。*

## ***class*tvm.relax.If(*cond:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*true_branch:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*false_branch:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

Relax 中的条件表达式。
* **参数：**
   * **cond** (*Expr*) ：条件。
   * **true_branch** (*Expr*) ：条件为真时计算的表达式。
   * **false_branch** (*Expr*) ：当条件为假时计算的表达式。
   * **span** (Optional[]) *：指向原始源代码的 Span。*

## ***class*tvm.relax.Constant(*data:***[NDArray](https://tvm.apache.org/docs/reference/api/python/runtime/ndarray.html#tvm.runtime.ndarray.NDArray)**,*struct_info:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**
常数张量
* **参数：**
   * **data** (*tvm.nd.NDArray*) ：常数张量的数据。
   * **struct_info** (Optional[]) *：*常量张量的结构信息。若未指定，则根据数据推断。
   * **span** (Optional[]*)：*指向原始源代码的 Span。

:::Note

标量常数由 ndim-0 常数张量表示。

:::

## ***class*tvm.relax.PrimValue(*value:***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***|***[int](https://docs.python.org/3/library/functions.html#int)**,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**
prim expr 表示值。

## ***class*tvm.relax.DataTypeImm(*value: dtype |***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**
表示数据类型常量。

## ***class*tvm.relax.StringImm(*value:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

表示字符串文字常量。

## **tvm.relax.const(*value:***[bool](https://docs.python.org/3/library/functions.html#bool)***|***[int](https://docs.python.org/3/library/functions.html#int)***|***[float](https://docs.python.org/3/library/functions.html#float)***| ndarray |***[NDArray](https://tvm.apache.org/docs/reference/api/python/runtime/ndarray.html#tvm.runtime.ndarray.NDArray)**,*dtype:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)→**[Constant](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Constant)

创建一个常数值。
* **参数：**
   * **value** (*Union*[*, ,* *,* *numpy.ndarray*, *tvm.nd.NDArray*]) ：常量值。
   * **dtype** (*Optional*[*])：结果常量的数据类型。*


:::Note

当 dtype 为 None 时，我们使用以下规则：
* int 映射到“int32”。
* float 映射到“float32”。
* bool 映射到“bool”。
* 其他使用与 numpy 相同的默认规则。

:::

## **tvm.relax.extern(*name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*struct_info:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

创建外部函数。

## **tvm.relax.get_shape_of(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)

获取 expr 的形状。
* **参数：expr** (*Expr*) ：输入表达式。
* **返回：** **shape** ：形状表达式。
* **返回类型：** Expr


:::Note

此函数要求对 expr 进行归一化。如果 expr 的 StructInfo 不是 TensorStructInfo，函数将报错。它会尽可能尝试返回符号函数。如果张量没有编译时符号形状，函数将选择返回 Relax.Call(relax.op.shape_of, [expr])。

:::

## ***class*tvm.relax.ObjectType(*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

与 tvm::runtime::Object 对应的类型是 TVM 中所有可能的对象值的基础。

## ***class*tvm.relax.ShapeType(*ndim:***[int](https://docs.python.org/3/library/functions.html#int)***= -1*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

Relax 中的形状类型。
* **参数：ndim** (*Optional*[*])：形状的大小。

## ***class*tvm.relax.TensorType(*ndim=-1*,*dtype='float32'*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

Relax 中的动态张量类型。


这是分配给具有已知 dtype 和未知形状的张量的类型。
* **参数：ndim** (Optional[]) ：张量的 ndim。
* **dtype** (Optional[]) ：内容数据类型。

## ***class*tvm.relax.PackedFuncType(*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**
Relax 中的 ExternFunc 的类型。

## ***class*tvm.relax.ExecBuilder**[](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ExecBuilder)

用于发出指令并为虚拟机构建可执行文件的构建器。

### **r(*idx:***[int](https://docs.python.org/3/library/functions.html#int)**)→**[int](https://docs.python.org/3/library/functions.html#int)[](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ExecBuilder.r)

将指令的参数设置为寄存器。

### **imm(*value:***[int](https://docs.python.org/3/library/functions.html#int)**)→**[int](https://docs.python.org/3/library/functions.html#int)

将指令的参数设置为立即数。

### **c(*idx:***[int](https://docs.python.org/3/library/functions.html#int)**)→**[int](https://docs.python.org/3/library/functions.html#int)


将指令的参数设置为常数。

### **f(*name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**)→**[int](https://docs.python.org/3/library/functions.html#int)


将指令的参数设置为函数。

### **declare_function(*func_name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*kind: VMFuncKind = VMFuncKind.PACKED_FUNC*)→**[None](https://docs.python.org/3/library/constants.html#None)


声明一个函数。

### **function(*func_name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*num_inputs:***[int](https://docs.python.org/3/library/functions.html#int)***|***[None](https://docs.python.org/3/library/constants.html#None)***= 0*,*param_names:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*)→ VMFuncScope**

注释 VM 函数。

### **emit_call(*name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*args:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[NDArray](https://tvm.apache.org/docs/reference/api/python/runtime/ndarray.html#tvm.runtime.ndarray.NDArray)***| dtype] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*dst:***[int](https://docs.python.org/3/library/functions.html#int)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)→**[None](https://docs.python.org/3/library/constants.html#None)

发出调用打包函数的调用指令。

### **emit_ret(*result:***[int](https://docs.python.org/3/library/functions.html#int)**)→**[None](https://docs.python.org/3/library/constants.html#None)


发出返回指令。

### **emit_goto(*pc_offset*)**[](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ExecBuilder.emit_goto)


发出 goto 指令。

### emit_if(cond, false_offset)[](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ExecBuilder.emit_if)


发出 if 指令。

### **get()→**[VMExecutable](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.VMExecutable)[](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ExecBuilder.get)


返回可执行文件。

## **tvm.relax.call_tir(*gvar:***[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)**,*args:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*out_sinfo:***[TensorStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.TensorStructInfo)***|***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[TensorStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.TensorStructInfo)***]*,*tir_vars:***[ShapeExpr](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ShapeExpr)***|***[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)***[***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***] |***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*)→**[Call](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Call)


relax.Call 调用一个 tir.prim_func 并返回输出。
* **参数：**
   * **gvar** () ：GlobalVar 引用 tir PrimFunc。
   * **args** (*Expr*) ：输入参数。
   * **out_sinfo** (Union[, List[]]) ：call_tir 输出的结构信息。它应该是一个 TensorStructInfo 或一个 TensorStructInfo 列表。每个 TensorStructInfo 表示返回张量的结构信息。
   * **tir_vars** (Optional[**Union**[, [], List[]**]**]) ：ShapeExpr 表示调用 func 时需要解包的整数元组。若未使用则为 null。
* **返回：** **ret**：call_tir 运算符的调用节点。
* **返回类型：** [relax.Call](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Call)。

## **tvm.relax.call_tir_inplace(*gvar:***[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)**,*args:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*inplace_indices:***[int](https://docs.python.org/3/library/functions.html#int)***|***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*,*out_sinfo:***[TensorStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.TensorStructInfo)***|***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[TensorStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.TensorStructInfo)***]*,*tir_vars:***[ShapeExpr](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ShapeExpr)***|***[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)***[***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***] |***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*)→**[Call](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Call)

relax.Call 调用 TIR PrimFunc 并返回结果，就地执行指定的计算（基于 inplace_indices 参数；输出将为就地索引选择的输入设置别名）。


警告：此运算符被类型系统视为纯运算符，但实际上会改变 inplace_indices 指定的参数。此运算符不应直接使用，而应在已检查是否可安全地就地执行操作（即，所有指定为输出的参数均未使用别名或在调用 call_tir_inplace 后保持活动状态）的遍历过程中插入。


直接呼叫该操作员仅应出于测试目的。
* **参数：**
   * **gvar** () ：引用 TIR 原始函数的 GlobalVar。
   * **args** (*Expr*) ：输入参数。
   * **inplace_indices** (*Union*[*, List*[*]])：1。
   * **out_sinfo** (Union[, List[]]) ：call_tir_inplace 输出的结构信息。它应该是一个 TensorStructInfo 或一个 TensorStructInfo 列表。每个列表表示返回张量的结构信息。如果给出一个 TensorStructInfo 列表，则结果将是一个 TensorStructInfo 元组。
   * **tir_vars** (Optional[**Union**[,[], List[]**]**]) ：ShapeExpr 表示调用 func 时需要解包的整数元组。若未使用则为 null。
* **返回：** **ret**：call_tir 运算符的调用节点。
* **返回类型：** [relax.Call](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Call)

## **tvm.relax.call_pure_packed(*func:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[ExternFunc](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ExternFunc)***|***[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)**,**args:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*sinfo_args:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***|***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***]*)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


构造一个对打包函数的调用，该调用应被视为纯粹的，即使打包调用通常不被视为纯粹的。


生成的调用将具有与直接调用打包函数相同的语义。


注意：这应该用于用户知道使用这些参数调用打包函数**实际上**不会引起任何副作用的情况。如果将其用于**确实**会产生副作用的调用，编译器最终可能会删除、重新排序或重复该调用，并且不保证被调用方不会产生任何副作用。
* **参数：**
   * **func** (*Union*[*,]*) ：PackedFunc 或 ExternFunc 节点的名称（全局符号）。
   * **args** (*Expr*) ：PackedFunc 的参数。
   * **sinfo_args** (*Union*[*, List*[*]])：结构信息参数列表（提供返回值的结构信息）。*
* **返回：** **result**：Relax 调用，对应于 call_pure_packed(ExternFunc(func), args, DictAttrs(kwargs), sinfo_args)。
* **返回类型：** Expr。

## **tvm.relax.call_dps_packed(*func:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*args:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*out_sinfo:***[TensorStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.TensorStructInfo)***|***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[TensorStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.TensorStructInfo)***]*)→**[Call](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Call)



Relax。调用目的地传递式打包函数并返回输出。


注意：被调用函数被假定为*纯函数*（除了修改指定的输出参数之外）。如果该函数确实产生了其他副作用，编译器可能会移除、重新排序或重复这些副作用——对此我们无法保证。
* **参数：**
   * **func** (Union[, Expr]) ：目标传递样式函数，可以是 ExternFunc。
   * **args** (Expr) ：输入参数。
   * **out_sinfo** (Union[, List[]]) ：call_dps_packed 输出的结构信息。它应该是一个 TensorStructInfo 或一个 TensorStructInfo 列表。每个 TensorStructInfo 表示返回张量的结构信息。
* **返回：** **ret**：call_dps_packed 运算符的调用节点。
* **返回类型：** [relax.Call](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Call)

## **tvm.relax.call_tir_with_grad(*gvar:***[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)**,*args:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*out_sinfo:***[TensorStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.TensorStructInfo)***|***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[TensorStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.TensorStructInfo)***]*,*te_grad_name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*te_grad_kwargs:***[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***, Object] = None*,*tir_vars:***[ShapeExpr](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ShapeExpr)***|***[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)***[***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***] |***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*)→**[Call](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Call)

relax.Call 调用 tir.prim_func 并返回输出。此内在函数会将 te 梯度函数（由 te_grad_name 引用）绑定到 call_tir_with_grad 节点。该 te 梯度函数将被梯度传递调用。
* **参数：**
   * **gvar** () ：GlobalVar 引用 tir PrimFunc。
   * **args** (Expr) ：输入参数。
   * **out_sinfo** (Union[, List[]]) ：call_tir_with_grad 输出的结构信息。它应该是一个 TensorStructInfo 或一个 TensorStructInfo 列表。每个 TensorStructInfo 表示返回张量的结构信息。
   * **te_grad_name** () ：与 call_tir_with_grad 节点关联的 te 梯度函数的注册名称。必须作为关键字参数提供。
   * **te_grad_kwargs** (Dict[, Object], optional) ：传递给 te 梯度函数的关键字参数。可选地，以关键字参数的形式提供。默认值：{}
   * **tir_vars** (Optional[**Union**[,[], List[]**]**]) ：ShapeExpr 表示调用 func 时需要解包的整数元组。若未使用则为 null。
* **返回：** **ret：** call_tir_with_grad 运算符的调用节点。
* **返回类型：** [relax.Call](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Call)

## ***class*tvm.relax.ExprFunctor**

在 Expr 上定义的抽象 visitor 。定义表达式的默认分派，并实现存储。

### **visit_expr(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)
将 visitor 应用到表达式。

## ***class*tvm.relax.PyExprVisitor**


一个抽象的 ExprVisitor，在 Python 端具有自定义方法。这是面向用户的类，用于方法覆盖继承。*tvm_metadata* 描述了要继承的类（“cls”）以及用户可以覆盖的方法（“methods”）。

Note: @relax.expr_functor.visitor is required for proper usage of any inherited class。

注意：任何继承类的正确使用都需要@relax.expr_functor.visitor。


另请参阅：visitor、_PyExprVisitor


示例：

```python
@relax.expr_functor.visitor
def MyExprVisitor(PyExprVisitor):
    ...
```


### visit_expr(expr: RelaxExpr)→ None


Expr 的通用调度程序。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr(const Expr& expr)。
* **参数：expr** (*Expr*) ：要访问的 expr。

### **visit_expr(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**)→**[None](https://docs.python.org/3/library/constants.html#None)


通用的 Binding 调度器。用户可以自定义此函数，在 C++ 端覆盖 VisitBinding(const Binding& binding)。
* **参数：binding** () ：要访问的绑定。

### **visit_binding(*binding:***[Binding](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Binding)**)→**[None](https://docs.python.org/3/library/constants.html#None)

BindingBlock 的通用调度器。用户可以自定义此函数，在 C++ 端覆盖 VisitBindingBlock(const BindingBlock& block)。
* **参数：block** () ：要访问的块。

### **visit_binding_block(*block:***[BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)**)→**[None](https://docs.python.org/3/library/constants.html#None)


用于访问 var 定义点的通用调度器。用户可以自定义此函数，在 C++ 端覆盖 VisitVarDef(const Relax.Var& var)。需要注意的是，visit_var_() 只会访问 Var 的使用点。
* **参数：var** () ：要访问的 var。

### **visit_var_def(*var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**)→**[None](https://docs.python.org/3/library/constants.html#None)

访问常量。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const ConstantNode op)。
* **参数：op** () ：要访问的常量。

### **visit_constant_(*op:***[Constant](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Constant)**)→**[None](https://docs.python.org/3/library/constants.html#None)

访问 Tuple。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const TupleNode op)。
* **参数：op** () ：要访问的元组。

### **visit_tuple_(*op:***[Tuple](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Tuple)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问变量。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const VarNode op)。
* **参数：op** () ：要访问的 relax.Var。

### **visit_var_(*op:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**)→**[None](https://docs.python.org/3/library/constants.html#None)

访问 DataflowVar。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const DataflowVarNode op)。
* **参数：op** () ：要访问的 DataflowVar。

### **visit_dataflow_var_(*op:***[DataflowVar](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataflowVar)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 ShapeExpr。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const ShapeExprNode op)。
* **参数：op** () ：要访问的 ShapeExpr。

### **visit_shape_expr_(*op:***[ShapeExpr](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ShapeExpr)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 ExternFunc。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const ExternFuncNode* op)。
* **参数：op** () ：要访问的 ExternFunc。

### **visit_extern_func_(*op:***[ExternFunc](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ExternFunc)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 GlobalVar。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const GlobalVarNode op)。
* **参数：op** () ：要访问的 GlobalVar。

### **visit_global_var_(*op:***[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)**)→**[None](https://docs.python.org/3/library/constants.html#None)

访问函数。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const FunctionNode op)。
* **参数：op** () ：要访问的函数。

### **visit_function_(*op:***[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问调用。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const CallNode op)。
* **参数：op** () ：要访问的 relax.Call。

### **visit_seq_expr_(*op:***[SeqExpr](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.SeqExpr)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 SeqExpr。用户可以自定义此函数，在 C++端覆盖 VisitExpr_(const SeqExprNode op)。
* **参数：op** () ：要访问的 SeqExpr。

### **visit_if_(*op:***[If](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.If)**)→**[None](https://docs.python.org/3/library/constants.html#None)

访问 If。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const IfNode op)。
* **参数：op** () ：要访问的 If。

### **visit_op_(*op:***[Op](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Op)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 Op。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const OpNode op)。
* **参数：op** () ：要访问的 Op。

### **visit_tuple_getitem_(*op:***[TupleGetItem](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.TupleGetItem)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 TupleGetItem。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const TupleGetItemNode op)。
* **参数：op** () ：要访问的 TupleGetItem。

### **visit_prim_value_(*op:***[PrimValue](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.PrimValue)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 PrimValue。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const PrimValueNode op)。
* **参数：op** () ：要访问的 PrimValue。

### **visit_string_imm_(*op:***[StringImm](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StringImm)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 StringImm。用户可以自定义此函数，在 C++端覆盖 VisitExpr_(const StringImmNode op)。
* **参数：op** () ：要访问的 StringImm。


### **visit_data_type_imm_(*op:***[DataTypeImm](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataTypeImm)**)→**[None](https://docs.python.org/3/library/constants.html#None)


 访问 DataTypeImm。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const 。
* **参数：op** () ：要访问的 DataTypeImm。


### **visit_var_binding_(*binding:***[VarBinding](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.VarBinding)**)→**[None](https://docs.python.org/3/library/constants.html#None)



访问 VarBinding。用户可以自定义此函数，在 C++ 端覆盖 VisitBinding_(const VarBindingNode* binding)。
* **参数：binding** () ：要访问的 VarBinding。

### **visit_match_cast_(*binding:***[MatchCast](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.MatchCast)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 MatchCast。用户可以自定义此函数，在 C++ 端覆盖 VisitBinding_(const MatchCastNode binding)。
* **参数：binding** () ：要访问的 MatchCast。

### **visit_binding_block_(*block:***[BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 BindingBlock。用户可以自定义此函数，在 C++端覆盖 VisitBlock_(const BindingBlockNode block)。
* **参数：block** () ：要访问的 BindingBlock。

### **visit_dataflow_block_(*block:***[DataflowBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataflowBlock)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 DataflowBlock。用户可以自定义此函数，在 C++ 端覆盖 VisitBindingBlock_(const DataflowBlockNode block)。
* **参数：block** () ：要访问的 DataflowBlock。

### **visit_var_def_(*var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 relax.Var 定义站点。用户可以自定义此函数，在 C++端覆盖 VisitVarDef_(const VarNode var)。
* **参数：var** () ：要访问的 relax.Var。

### **visit_dataflow_var_def_(*var:***[DataflowVar](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataflowVar)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 DataflowVar 定义站点。用户可以自定义此函数，在 C++ 端覆盖 VisitVarDef_(const DataflowVarNode var)。
* **参数：var** () ：要访问的 DataflowVar。

### **visit_span(*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)**)→**[None](https://docs.python.org/3/library/constants.html#None)

访问 Span。用户可以自定义此函数，在 C++ 端覆盖 VisitSpan(const Span& span)。
* **参数：span** () ：要访问的 Span。

## ***class*tvm.relax.PyExprMutator(*mod:***[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**


一个抽象的 ExprMutator，在 Python 端具有自定义方法。这是面向用户的用于方法覆盖继承的类。*tvm_metadata* 描述了要继承的类（“cls”）、用户可以覆盖的方法（“methods”）以及构造函数的参数（“fields”）。


注意：任何继承类的正确使用都需要@relax.expr_functor.mutator。


另请参阅：visitor、_PyExprVisitor。


示例：

```python
@relax.expr_functor.mutator
def MyExprMutator(PyExprMutator):
    ...
```


### **visit_expr(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


Expr 的通用调度程序。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr(const Expr& expr)。
* **参数：expr** (*Expr*) ：要访问的 expr。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_binding(*binding:***[Binding](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Binding)**)→**[None](https://docs.python.org/3/library/constants.html#None)


通用的 Binding 调度器。用户可以自定义此函数，在 C++ 端覆盖 VisitBinding(const Binding& binding)。
* **参数：binding** () ：要访问的绑定。

### **visit_binding_block(*block:***[BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)**)→**[BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)


BindingBlock 的通用调度器。用户可以自定义此函数，在 C++ 端覆盖 VisitBindingBlock(const BindingBlock& block)。
* **参数：block** () ：要访问的块。
* **返回：** **result**：转换后的绑定块。
* **返回类型：** [BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)。

### **visit_var_def(*var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**)→**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)


用于访问 var 定义点的通用调度器。用户可以自定义此函数，在 C++ 端覆盖 VisitVarDef(const Relax.Var& var)。需要注意的是，visit_var_() 只会访问 Var 的使用点。
* **参数：var** () ：要访问的 var。
* **返回：** **result**：后序重写后的 var。
* **返回类型：** [relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)。

### **visit_constant_(*op:***[Constant](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Constant)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问常量。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const ConstantNode op)。
* **参数：op** () ：要访问的常量。
* **返回：** **result** ：转换后的 Expr。
* **返回类型：** Expr。

### **visit_tuple_(*op:***[Tuple](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Tuple)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问 Tuple。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const TupleNode op)。
* **参数：op** () ：要访问的元组。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_var_(*op:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问变量。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const VarNode op)。
* **参数：op** () ：要访问的 relax.Var。
* **返回：** **result：**转换后的 Expr。
* **返回类型：** Expr。

### **visit_dataflow_var_(*op:***[DataflowVar](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataflowVar)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问 DataflowVar。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const DataflowVarNode op)。
* **参数：op** () ：要访问的 DataflowVar。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_shape_expr_(*op:***[ShapeExpr](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ShapeExpr)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)

访问 ShapeExpr。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const ShapeExprNode op)。
* **参数：op** () ：要访问的 ShapeExpr。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_extern_func_(*op:***[ExternFunc](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.ExternFunc)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问 ExternFunc。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const ExternFuncNode op)。
* **参数：op** () ：要访问的 ExternFunc。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_global_var_(*op:***[GlobalVar](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.GlobalVar)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问 GlobalVar。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const GlobalVarNode op)。
* **参数：op** () ：要访问的 GlobalVar。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_function_(*op:***[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)

访问函数。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const FunctionNode op)。
* **参数：op** () ：要访问的函数。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_call_(*op:***[Call](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Call)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问调用。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const CallNode op)。
* **参数：op** () ：要访问的 relax.Call。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_seq_expr_(*op:***[SeqExpr](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.SeqExpr)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问 SeqExpr。用户可以自定义此函数，在 C++端覆盖 VisitExpr_(const SeqExprNode op)。
* **参数：op** () ：要访问的 SeqExpr。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_if_(*op:***[If](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.If)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问 If。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const IfNode op)。
* **参数：op** () ：要访问的 If。
* **返回：** **result**– 转换后的 Expr。
* **返回类型：** Expr。

### **visit_op_(*op:***[Op](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Op)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问 Op。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const OpNode op)。
* **参数：op** () ：要访问的 Op。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_tuple_getitem_(*op:***[TupleGetItem](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.TupleGetItem)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问 TupleGetItem。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const TupleGetItemNode op)。
* **参数：op** () ：要访问的 TupleGetItem。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_prim_value_(*op:***[PrimValue](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.PrimValue)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问 PrimValue。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const PrimValueNode op)。
* **参数：op** () ：要访问的 PrimValue。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_string_imm_(*op:***[StringImm](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StringImm)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


访问 StringImm。用户可以自定义此函数，在 C++端覆盖 VisitExpr_(const StringImmNode op)。
* **参数：op** () ：要访问的 StringImm。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_data_type_imm_(*op:***[DataTypeImm](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataTypeImm)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)

访问 DataTypeImm。用户可以自定义此函数，在 C++ 端覆盖 VisitExpr_(const DataTypeImmNode op)。
* **参数：op** () ：要访问的 DataTypeImm。
* **返回：** **result**：转换后的 Expr。
* **返回类型：** Expr。

### **visit_var_binding_(*binding:***[VarBinding](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.VarBinding)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 VarBinding。用户可以自定义此函数，在 C++ 端覆盖 VisitBinding_(const VarBindingNode binding)。
* **参数：binding** () ：要访问的 VarBinding。

### **visit_match_cast_(*binding:***[MatchCast](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.MatchCast)**)→**[None](https://docs.python.org/3/library/constants.html#None)


访问 MatchCast。用户可以自定义此函数，在 C++ 端覆盖 VisitBinding_(const MatchCastNode* binding)。
* **参数：binding** () ：要访问的 MatchCast。

### **visit_binding_block_(*block:***[BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)**)→**[BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)


访问 BindingBlock。用户可以自定义此函数，在 C++端覆盖 VisitBlock_(const BindingBlockNode block)。
* **参数：block** () ：要访问的 BindingBlock。
* **返回：** **result**：转换后的绑定块。
* **返回类型：** [BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)。

### **visit_dataflow_block_(*block:***[DataflowBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataflowBlock)**)→**[BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)


访问 DataflowBlock。用户可以自定义此函数，在 C++ 端覆盖 VisitBindingBlock_(const DataflowBlockNode block)。
* **参数：block** () ：要访问的 DataflowBlock。
* **返回：** **result**：转换后的绑定块。
* **返回类型：** [BindingBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.BindingBlock)。

### **visit_var_def_(*var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**)→**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)


访问 relax.Var 定义站点。用户可以自定义此函数，在 C++端覆盖 VisitVarDef_(const VarNode var)。
* **参数：var** () ：要访问的 relax.Var。
* **返回：** **result**：后序重写后的 var。
* **返回类型：** [relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)。

### **visit_dataflow_var_def_(*var:***[DataflowVar](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataflowVar)**)→**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)


访问 DataflowVar 定义站点。用户可以自定义此函数，在 C++ 端覆盖 VisitVarDef_(const DataflowVarNode var)。
* **参数：var** () ：要访问的 DataflowVar。
* **返回：** **result**：后序重写后的 var。
* **返回类型：** [relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)。

### **visit_span(*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)**)→**[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)


访问 Span。用户可以自定义此函数，在 C++ 端覆盖 VisitSpan(const Span& span)。
* **参数：span** ()：要访问的 Span。
* **返回：** **result**：转换后的跨度。
* **返回类型：** [Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)。

### **visit_expr_post_order(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


按后序重写 Expr 并进行规范化。
* **参数：expr** (*Expr*)：要重写的 Expr。
* **返回：** **result**：后序重写后的 Expr。
* **返回类型：** Expr。

### **set_var_remap(*vid:***[Id](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Id)**,*var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**)→**[None](https://docs.python.org/3/library/constants.html#None)


在使用站点中将 var 重新映射到新的 var。
* **参数：vid** ()：旧变量的 vid。
* **var** ()：新的 var。

### **get_var_remap(*vid:***[Id](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Id)**)→**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)


在使用站点中将 var 重新映射到新的 var。
* **参数：vid** ()：旧 var 的 vid。
* **返回：** **var**：重新映射的 var。
* **返回类型：** [relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)。

### **visit_with_new_scope(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


用新的作用域重写 expr，用于函数主体和 If 的分支。
* **参数：expr** (*Expr*)：要访问的 expr。
* **返回：** **var**：访问后的表达式。
* **返回类型：** [relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)。

### **lookup_binding(*var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**|**[None](https://docs.python.org/3/library/constants.html#None)


查找绑定到变量的值。注意：对于函数参数，此函数返回 std::nullopt。
* **参数：var** ()：要查找的变量。
* **返回：** **var**：绑定到输入变量的值。
* **返回类型：** [relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)

### **with_struct_info(*var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**,*struct_info:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**)→**[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)


如果原始变量的形状或类型与指定的形状或类型不匹配，则创建一个具有指定形状和类型的新变量。
* **参数：var** ()：要更新的变量。
* **struct_info** ()：结构信息。
* **返回：** **var**：填充有形状和类型的 var。
* **返回类型：** [relax.Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)。

## ***class*tvm.relax.StructInfo**


所有 StructInfo 的基类。


StructInfo 包含静态类型和运行时结构信息。

### **same_as(*other*)**


结构平等导致超载。

### **is_base_of(*derived:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**)→**[bool](https://docs.python.org/3/library/functions.html#bool)


检查自身是否是另一个派生结构信息的基础。
* **参数：derived** ()：要检查的派生结构信息。
* **返回：** **result**：检查结果。
* **返回类型：** [bool](https://docs.python.org/3/library/functions.html#bool)。

## ***class*tvm.relax.ObjectStructInfo(*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

对象的 StructInfo。

## ***class*tvm.relax.PrimStructInfo(*dtype:***[str](https://docs.python.org/3/library/stdtypes.html#str)***| dtype |***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*value:***[int](https://docs.python.org/3/library/functions.html#int)***|***[float](https://docs.python.org/3/library/functions.html#float)***|***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

原始 POD 值的结构信息。
* **参数：dtype_or_expr** (Union[, DataType, ])：原始值的数据类型，或原始值的已知表达式。

## ***class*tvm.relax.ShapeStructInfo(*values:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*ndim:***[int](https://docs.python.org/3/library/functions.html#int)***= -1*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

形状值的结构信息。
* **参数：values** (*O*ptional[**List**[]])：如果已知，则为符号形状值。
* **ndim** (Optional[])：形状的大小。


:::Note

不要同时指定值和 ndim。

:::

## ***class*tvm.relax.TensorStructInfo(*shape:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)***|***[None](https://docs.python.org/3/library/constants.html#None)***|***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[PrimExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.PrimExpr)***] = None*,*dtype:***[str](https://docs.python.org/3/library/stdtypes.html#str)***= 'float32'*,*vdevice:***[VDevice](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.VDevice)***|***[None](https://docs.python.org/3/library/constants.html#None)***|***[str](https://docs.python.org/3/library/stdtypes.html#str)***= None*,*ndim:***[int](https://docs.python.org/3/library/functions.html#int)***= -1*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**


张量值的结构信息。
* **参数：shape** (Optional[**Expr**])：形状表达式。
* **dtype** (Optional[])：内容数据类型。
* **vdevice** (Optional[**Vdevice**])：虚拟设备。
* **ndim** (Optional[])：张量的维数。

:::Note

不要同时指定 shape 和 ndim。

:::

## ***class*tvm.relax.TupleStructInfo(*fields:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***]*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**

Tuple 值的 StructInfo。
* **参数：fields** (List[])：字段的结构信息。

## ***class*tvm.relax.FuncStructInfo(*params:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***]*,*ret:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)**,*purity:***[bool](https://docs.python.org/3/library/functions.html#bool)***= True*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**


函数值的结构信息。
* **参数：**
   * **params** (*Lis*t[])*：字段的结构信息。*
   * **ret** ()：返回值的结构信息。
   * **purity** ()：函数是否纯（没有可见的副作用）。注意：只有当函数对所有输入都为纯函数时，我们才认为它是纯函数。如果函数仅在某些情况下才有可见的副作用，我们仍然认为它是非纯函数。

### ***static*opaque_func(***,*ret:***[StructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.StructInfo)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*derive_func:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[EnvFunc](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.EnvFunc)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*purity:***[bool](https://docs.python.org/3/library/functions.html#bool)***= False*,*span:***[Span](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Span)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)→**[FuncStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.FuncStructInfo)

创建一个不透明的 FuncStructInfo。


不透明函数要么采用指定返回值的结构信息的 ret，要么采用提供自定义派生规则的 derive_func。
* **参数：ret** (Optional[])：函数返回值的结构信息。
* **derive_func** (Optional[**Union**[,]])：用于推导的环境函数。
* **purity** ()：函数是否纯（默认为 false，因为大多数不透明函数都不是纯函数）。
* **span** (Optional[])：ast 的可选跨度信息。
* **返回：** **info。**
* **返回类型：** [FuncStructInfo](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.FuncStructInfo)。

:::note


我们不能同时指定 ret 和 derive_func。

:::

## **tvm.relax.get_default_pipeline(*target:***[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)**)**


获取给定目标的默认 Relax 编译管道。

## **tvm.relax.get_pipeline(*name:***[str](https://docs.python.org/3/library/stdtypes.html#str)***= 'zero'*,***kwargs*)→**[Pass](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.Pass)


按名称获取预构建管道。
* **参数：**
   * **name** (Optional[])：管道的名称。
   * **kwargs** (Dict[, ])**：** 用于配置管道的关键字参数。
* **返回：** **pipeline**[：](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.Pass)转换管道。
* **返回类型：** [tvm.transform.Pass](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.Pass)。

## **tvm.relax.register_pipeline(*name:***[str](https://docs.python.org/3/library/stdtypes.html#str)**)**


注册新管道。
* **参数：name** ()**：**管道的名称

## **tvm.relax.convert_to_expr(*value:***[Any](https://docs.python.org/3/library/typing.html#typing.Any)**)→**[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)


辅助函数将输入转换为 Expr，遵循以下规则：1. 如果输入已经是 Relax.Expr ，则返回输入本身；2.如果输入是 PrimExpr ，则返回 Relax.PrimValue；3.如果输入是 tvm.String 或 str ，则返回 Relax.StringImm；4.如果输入是 Expr 的元组/列表，则返回 Relax.Tuple。

**注意**

1. tvm.tir.StringImm 因歧义而不被允许，它可以是 Relax.StringImm 或 Relax.PrimValue。

## **tvm.relax.build(*mod:***[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)**,*target:***[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)***|***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*params:***[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[list](https://docs.python.org/3/library/stdtypes.html#list)***] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*relax_pipeline:***[None](https://docs.python.org/3/library/constants.html#None)***|***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[Pass](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.Pass)***= 'default'*,*tir_pipeline:***[None](https://docs.python.org/3/library/constants.html#None)***|***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[Pass](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.Pass)***= 'default'*,*exec_mode:***[str](https://docs.python.org/3/library/stdtypes.html#str)***= 'bytecode'*,***,*system_lib:***[bool](https://docs.python.org/3/library/functions.html#bool)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*)→ Executable**


构建一个 IRModule 到 VM 可执行文件。
* **参数：**
   * **mod** ()：要构建的输入 IRModule。
   * **target** (Optional*[**Union**[, ]])：「一个构建目标，可以包含可选的主机端编译目标。」
   * 当 TVM 编译设备特定程序（如 CUDA）时，还需要主机（CPU）端代码与驱动程序交互，以正确设置维度和参数*。*`host` 用于指定主机端代码生成目标。默认情况下，如果启用了 `llvm`*，*则使用它，否则使用 `stackvm` 解释器。
   * **params** (Optional[**Dict**[,]])：将绑定的输入 IRModule 的参数。
   * **relax_pipeline** (str = "default")：要使用的 Relax 编译管道。
   * **tir_pipelinie** (str = "default")**：** 要使用的 TIR 编译管道。
   * **exec_mode** ()：执行模式。
   * **system_lib** (Optional[])：是否构建正在静态打包的系统库，并自动将生成的函数注册到系统中。默认情况下，会根据目标自动检测。
* **返回：** **ex**：可由虚拟机加载的可执行文件。
* **返回类型：** tvm.relax.Executable。


**示例**

```python
class InputModule:
    @R.function
    def foo(x: Tensor((3, 4), "float32"), y: Tensor((3, 4), "float32")):
        z = R.add(x, y)
        return z

mod = InputModule
target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(mod, target)
```


## ***class*tvm.relax.VMExecutable(*mod: Module*)**


VM 编译器或 ExecBuilder 发出的虚拟机可执行对象。

### **stats()→**[str](https://docs.python.org/3/library/stdtypes.html#str)


打印可执行文件的详细统计信息。

### **as_text()→**[str](https://docs.python.org/3/library/stdtypes.html#str)


将说明打印为文本格式。

### **as_python()→**[str](https://docs.python.org/3/library/stdtypes.html#str)

将指令打印为 python 程序。

## ***class*tvm.relax.DataflowBlockRewrite(*dfb:***[DataflowBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataflowBlock)**,*root_fn:***[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)**)**

一个绑定/语句级数据流块重写器。


:::Note


由于 TVM AST 节点的不可变性和写时复制特性，重写并非就地完成。相反，会创建一个新的 DataflowBlock，并通过 mutated_dfb 返回。同样，其新的根函数也会由 mutated_root_fn 创建并返回。要将此更改应用于 IRModule，请使用 mutate_irmodule 重写构造函数中注册的旧函数。

### **replace_all_uses(*old_var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**,*new_var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**)→**[None](https://docs.python.org/3/library/constants.html#None)


将所有 old_var 替换为 new_var。
* **参数：old_var** ()**：** 要替换的旧变量。
* **new_var** ()**：** 要替换的新变量。

### **add(*expr:***[RelaxExpr](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.RelaxExpr)**,*name:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,*is_dfvar:***[bool](https://docs.python.org/3/library/functions.html#bool)***= False*)→**[None](https://docs.python.org/3/library/constants.html#None)


使用自动生成的变量名向 DataflowBlock 添加新语句。
* **参数：**
   * **expr** (Expr)**：** 要添加的表达式。
   * **name** (Optional[], optional)：变量名称，默认为 None。
   * **is_dfvar** (, optional)：变量类型，默认为 False。


:::Note


如果未指定变量名，则会自动生成“tmp$”形式的变量。如果 is_dfvar 为 True，则变量类型为 DataflowVar，否则为 Var。relax.Var 表示变量是 DataflowBlock 的输出变量，而 DataflowVar 表示变量是 DataflowBlock 的内部变量。

### **remove_unused(*var:***[Var](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Var)**,*allow_undef=False*)→**[None](https://docs.python.org/3/library/constants.html#None)


当且仅当语句未使用时，才通过其变量定义删除该语句。
* **参数：**
   * **var** ()：未使用的变量定义。
   * **allow_undef** ([bool](https://docs.python.org/3/library/functions.html#bool),*optional*)*：是否允许 var 为未定义变量，默认为 False。
* **抛出:** **如果变量已被使用或未定义时** （allow_undef=False** **） ，则引发 TVMError** **。** 

### **remove_all_unused()→**[None](https://docs.python.org/3/library/constants.html#None)


删除所有未使用的变量。


**注意**


这也可以删除其他 DataflowBlocks 中未使用的变量。

### **mutated_dfb()→**[DataflowBlock](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.DataflowBlock)


返回转换的 DataflowBlock。

### **mutated_root_fn()→**[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)


返回转换的根函数。

### **mutate_irmodule(*irmodule:***[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)**)→**[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)


通过用转换的根函数替换旧函数来返回更新的 IRModule。
* **参数：irmodule** (*tvm.IRModule*)：要更新的基本 IRModule。
* **返回：** 返回更新后的 IRModule.
* **返回类型：** tvm.IRModule


