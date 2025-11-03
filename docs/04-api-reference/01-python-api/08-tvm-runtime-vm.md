---

title: tvm.runtime.vm

---


Relax 虚拟机。


## *class* tvm.runtime.vm.VMInstrumentReturnKind(*value*)

一个枚举。


## *class* tvm.runtime.vm.VirtualMachine(*rt_mod: Module | Executable*, *device: Device |*[List](https://docs.python.org/3/library/typing.html#typing.List)*[Device]*, *memory_cfg:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[Device,*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *profile:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)

Relax VM 运行时。


### invoke_closure(*closure: Object*, args:*[Any](https://docs.python.org/3/library/typing.html#typing.Any)) → Object
调用一个闭包。
* **参数：** 
   * **closure** (*Object*) ：VMClosure 对象。
   * **args** ([list](https://docs.python.org/3/library/stdtypes.html#list)*[**tvm.runtime.Tensor****] or*[list](https://docs.python.org/3/library/stdtypes.html#list)*[**np.ndarray****]*) ：闭包的参数。
* **返回：** result：输出。
* **返回类型：** Object。


### save_function(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *saved_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), args:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*]*, *include_return:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, ***kwargs:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[Any](https://docs.python.org/3/library/typing.html#typing.Any)*]*) → [None](https://docs.python.org/3/library/constants.html#None)

一个便捷函数。它从模块中获取一个函数并保存为 PackedFunc 对象，调用该对象时将使用给定的参数调用该函数。可以使用 saved_name 从模块访问 PackedFunc 对象。包含此功能是为了方便计时试验：调用返回的 PackedFunc 对象进行字典查找的开销比通常通过虚拟机运行的开销要小。


如果已保存的名称被使用，则会覆盖它，但它不会覆盖 Relax 源中定义的函数的名称。


这实际上是在创建一个闭包，但该函数具有不同的名称，以避免与 invoke_closure 混淆（它们不应该一起使用）。
* **参数：**
   * **func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：需要打包的函数。
   * **saved_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：应保存结果闭包的名称。
   * **include_return** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：已保存的 PackedFunc 是否应返回其输出。如果通过 RPC 进行计时，则可能不希望在机器之间发送输出。
   * **args** (*List**[****Any]*) ：与函数打包的参数。
   * **kwargs** (*Dict**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Any****]*) ：与函数打包的任何命名参数。


### set_input(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), args:*[Any](https://docs.python.org/3/library/typing.html#typing.Any), ***kwargs:*[Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [None](https://docs.python.org/3/library/constants.html#None)

设置函数输入。此接口在使用 VM over RPC 时有效，其内部将参数中的 NDArray 转换为 DLTensor。在 RPC 中，远程只能使用最小的 C 语言运行时，而 DLTensor 是受支持的。


注意：如果使用 set_input ，必须使用 invoke_stateful 调用该函数，并且必须使用 get_outputs 获取结果。
* **参数：**   
   * **func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：函数的名称。
   * **args** (*List**[****tvm.runtime.Tensor**] or*** ***List****[**np.ndarray****]*) ：函数的参数。
   * **kwargs** ([dict](https://docs.python.org/3/library/stdtypes.html#dict)*ofstr to tvm.runtime.NDArrayornp.ndarray*) ：函数的命名参数。


### invoke_stateful(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)

使用通过 set_input 设置的参数，从 VM 模块调用指定的函数。如果未先使用 set_input 就调用 invoke_stateful 函数（即使是为了设置 0 个输入），则会出错；反之，如果已调用 set_input ，则在未使用 invoke_stateful 的情况下调用该函数也会出错。


可以通过调用 get_outputs 获取调用结果。
* **参数：func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：要调用的函数的名称。


### get_outputs(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → Object | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]

调用 invoke_stateful 后，获取给定名称的函数输出的值。


如果没有先调用 invoke_stateful 就调用此函数是错误的。
* **参数：func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：应获取其输出的函数的名称。
* **返回：ret** ：先前通过 invoke_stateful 调用该函数的结果。若结果是一个元组，则返回一个字段列表。这些字段也可能是元组，因此可以任意嵌套。
* **返回类型：** Union[tvm.Object, [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)[Any]]。


### set_instrument(*instrument: Function*) → [None](https://docs.python.org/3/library/constants.html#None)

设置 instrumentation 函数。


如果存在 instrument 函数，该函数将在每次调用指令之前/之后被调用。该函数具有以下签名：

```python
def instrument(
    func: Union[VMClosure, PackedFunc],
    func_symbol: str,
    before_run: bool,
    ret_value: any,
    *args) -> bool:
    pass
```


该工具采用以下参数：func：要调用的函数对象。func_symbol：函数的符号名称。before_run：是在调用之前还是之后。ret_value：调用的返回值，仅在运行后有效。args：传递给调用的参数。


检测函数可以选择一个整数，该整数对应后续运行的操作方向。更多详情，请参阅 VMInstrumentReturnKind。
* **参数：instrument** (*tvm.runtime.PackedFunc*)  – 每次 VM 调用 instr 时都会调用的检测函数。



:::note

`VMInstrumentReturnKind` 

VM 中可能的返回值。

:::


### time_evaluator(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *dev: Device*, *number:*[int](https://docs.python.org/3/library/functions.html#int)*= 10*, *repeat:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *min_repeat_ms:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *cooldown_interval_ms:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *repeats_to_cooldown:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *f_preproc:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= ''*) → [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[…], BenchmarkResult]

返回一个评估器，用于对模块中的函数进行计时。它遵循与 `tvm.runtime.module` 中的 `time_evaluator` 相同的约定。可以与 `save_function()` 一起使用，以避免额外的字典查找开销，从而使计时更加高效。
* **参数：**
   *    **func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))  – 模块中函数的名称。
   *    **dev** (*Device*)  – 我们应该在其上运行此功能的设备。
   *    **number** ([int](https://docs.python.org/3/library/functions.html#int))  – 运行此函数取平均值的次数。我们将这些运行称为一次重复测量。
   *    **repeat** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)  – 重复测量的次数。该函数总共会被调用 (1 + number x repeat) 次，其中第一次调用是预热，之后会被丢弃。返回的结果包含重复成本，每次重复成本是number成本的平均值。
   *    **min_repeat_ms** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)  – 一次重复的最短持续时间（以毫秒为单位）。默认情况下，一次重复包含 number 个运行。如果设置了此参数，则参数 number 将动态调整以满足一次重复的最短持续时间要求。即，当一次重复的运行时间低于此时间时，number 参数将自动增加。
   *    **cooldown_interval_ms** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)  –  repeats_to_cooldown 定义的重复次数之间的冷却间隔（以毫秒为单位）。
   *    **repeats_to_cooldown** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)  – 冷却激活前的重复次数。
   *    **f_preproc** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)  – 在执行时间评估器之前我们要执行的预处理函数名称。


:::Note

该函数将被调用 (1 + 数字 x 重复) 次，如果存在延迟初始化，则第一次调用将被丢弃。

:::


**示例**


与 VM 函数正常使用（如果函数返回元组，则可能无法通过 RPC 工作）：

```python
target = tvm.target.Target("llvm", host="llvm")
ex = tvm.compile(TestTimeEvaluator, target)
vm = relax.VirtualMachine(mod, tvm.cpu())
timing_res = vm.time_evaluator("func_name", tvm.cpu())(arg0, arg1, ..., argn)
```


与有状态的 API 一起使用：

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
vm.save_function("func_name", "func_name_saved", arg0, arg1, ..., argn)
timing_res = vm.time_evaluator("func_name_saved", tvm.cpu())()
```
* **返回：ftimer：** 该函数接受与 func 相同的参数，并返回 BenchmarkResult。ProfileResult 报告重复时间成本（以秒为单位）。
* **返回类型：** function。


### profile(*func_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), args*)

分析函数调用。
* **参数：**   
   * **func_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：函数的名称。
   * **args** (*List of [Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or other objects supported by PackedFunc.*) ：函数的参数。
* **返回：report** ：格式化的分析结果，显示每个操作的时间测量。
* **返回类型：**[tvm.runtime.profiling.Report](/docs/api-reference/python-api/tvm-runtime-profiling#class-tvmruntimeprofilingreportcallssequencedictstr-object-device_metricsdictstrdictstr-object-configurationdictstr-object)。


