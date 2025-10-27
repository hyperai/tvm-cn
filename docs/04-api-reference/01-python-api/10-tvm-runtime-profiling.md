---

title: tvm 运行时分析

---

在 python 中注册分析对象。

## *class* tvm.runtime.profiling.Report(*calls:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object]]*, *device_metrics:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object]]*, *configuration:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Object]*)

在分析运行期间收集的信息的容器。


### calls

每次调用的分析指标（函数名称、运行时、设备等）。
* 类型：[Array](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirarrayinput_listsequenceany)[Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), Object]]


### device_metrics

整个运行过程中收集的每个设备指标。
* 类型：Dict[Device, Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), Object]]


### csv()

将此分析报告转换为 CSV 格式。


这仅包括调用，而不包括整体指标。
* **返回：csv**：以 CSV 格式调用。
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)


### table(*sort=True*, *aggregate=True*, *col_sums=True*)

生成人类可读的表格。
* **参数：** 
   * **sort** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：如果aggregate为true，则是否按时长降序对调用帧进行排序。如果 aggregate为False，则是否按程序中出现的顺序对帧进行排序。
   * **aggregate** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：是否将对同一操作的多个调用合并为一行。
   * **col_sums** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：是否包含每列的总和。
* **返回：table**：一个人类可读的表格。
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)。


### json()

将此分析报告转换为 JSON 格式。


示例输出：
* **返回：json**：格式化的 JSON。
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)


### *classmethod* from_json(*s*)

从 JSON 反序列化报告。
* **参数：s** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：通过 序列化报告`json()`。
* **返回：report**：反序列化的报告。
* **返回类型：**[Report](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-profiling#class-tvmruntimeprofilingreportcallssequencedictstr-object-device_metricsdictstrdictstr-object-configurationdictstr-object)


## *class* tvm.runtime.profiling.Count(*count:*[int](https://docs.python.org/3/library/functions.html#int))

某物的整型计数。


## *class* tvm.runtime.profiling.Duration(*duration:*[float](https://docs.python.org/3/library/functions.html#float))

某事物的持续时间。


## *class* tvm.runtime.profiling.Percent(*percent:*[float](https://docs.python.org/3/library/functions.html#float))

某物的百分比。


## *class* tvm.runtime.profiling.Ratio(*ratio:*[float](https://docs.python.org/3/library/functions.html#float))

两个对象的比率。


## *class* tvm.runtime.profiling.MetricCollector

用户定义的分析指标收集接口。


## *class* tvm.runtime.profiling.DeviceWrapper(*dev: Device*)

包装 tvm.runtime.Device。


## tvm.runtime.profiling.profile_function(*mod*, *dev*, *collectors*, *func_name=None*, *warmup_iters=10*)

收集函数执行的性能信息。通常与已编译的 PrimFunc 一起使用。


这些信息可能包括性能计数器，例如缓存命中率和 FLOP，它们有助于调试单个 PrimFuncs 的性能问题。根据所使用的 MetricCollector，可以收集不同的指标。


**示例**
* **参数：**
   * **mod** (*Module*) ：包含要分析的功能的模块。
   * **dev** (*Device*) ：运行该功能的设备。
   * **collectors** (*List[*[MetricCollector](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-profiling#class-tvmruntimeprofilingmetriccollector)*]*) ：`MetricCollector`将收集性能信息。
   * **func_name** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) ：要配置的mod中的函数名称。默认为mod的entry_name。
   * **warmup_iters** ([int](https://docs.python.org/3/library/functions.html#int)) ：收集性能信息之前运行函数的迭代次数。建议将其设置为大于 0，以获得一致的缓存效果。默认为 10。
* **返回：prof**：PackedFunc 采用与**mod[func_name]相同的参数，并以Dict[str, ObjectRef] 的形式返回性能指标，其中值可以是CountNode、DurationNode、PercentNode。
* **返回类型：** PackedFunc[args, Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), ObjectRef]]。


