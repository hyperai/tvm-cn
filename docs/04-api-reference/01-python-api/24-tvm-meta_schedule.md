---

title: tvm.meta_schedule

---


tvm.meta_schedule 软件包。元调度基础设施。



**类:**

|[Builder](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Builder)|抽象构建器接口。|
|:----|:----|
|[CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)|成本模型。|
|[Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)|抽象数据库接口。|
|[ExtractedTask](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ExtractedTask)(task_name,mod,target,…)|从高级 IR 中提取的调优任务。|
|[FeatureExtractor](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.FeatureExtractor)|从测量候选中提取特征以用于成本模型。|
|[MeasureCallback](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCallback)|测量结果出来后适用的规则可用。|
|[Mutator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Mutator)|Mutator 旨在改变轨迹以探索设计空间。|
|[Postproc](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Postproc)|将后处理器应用于调度的规则。|
|[Profiler](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Profiler)()|调整时间分析器。|
|[Runner](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Runner)|抽象运行器接口。|
|[ScheduleRule](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule)|修改调度中的块的规则。|
|[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)(sch,args_info)|衡量候选类别。|
|[SearchStrategy](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy)|搜索策略是生成度量候选的类。|
|[SpaceGenerator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SpaceGenerator)|抽象设计空间生成器接口。|
|[TaskScheduler](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler)|抽象任务调度程序接口。|
|[TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)([mod,target,space_generator,…])|调整上下文类旨在包含调整任务的所有资源。|

**函数:**

|[tune_tir](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.tune_tir)(mod,target,work_dir,…[,…])|调整 TIR 函数或 TIR 函数的 IRModule。|
|:----|:----|
|[tune_tasks](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.tune_tasks)(*,tasks,task_weights,work_dir,…)|调整任务列表。使用任务调度程序。|
|[derived_object](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.derived_object)(cls)|用于为 TVM 对象注册派生子类的装饰器。|

## *class* tvm.meta_schedule.Builder


抽象构建器接口。


**方法：**

|[build](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Builder.build)(build_inputs)|构建给定的输入。|
|:----|:----|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Builder.create)([kind])|创建一个构建器。|

### build(*build_inputs:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[BuilderInput]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[BuilderResult]


构建给定的输入。
* **参数：build_inputs** (*List*[*BuilderInput]*)：要构建的输入。
* **返回：build_results**：构建给定输入的结果。
* **返回类型：** List[BuilderResult]。

### *static* create(*kind: typing_extensions.Literal[local] = 'local'*, args*, ***kwargs*) → [Builder](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Builder)


创建一个构建器。
* **参数：kind** (*Literal*[*"local"]*)：构建器的类型。目前仅支持「local」。
* **返回：builder**：构建器创建。
* **返回类型：**[Builder](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Builder)。

## *class* tvm.meta_schedule.CostModel


成本模型。


**方法：**

|[load](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel.load)(path)|从给定的文件位置加载成本模型。|
|:----|:----|
|[save](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel.save)(path)|将成本模型保存到给定的文件位置。|
|[update](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel.update)(context, candidates, results)|根据运行结果更新成本模型。|
|[predict](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel.predict)(context, candidates)|使用成本模型预测标准化分数。|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel.create)(kind, *args, **kwargs)|创建一个 CostModel。|

### load(*path:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)


从给定的文件位置加载成本模型。
* **参数：path** ([str](https://docs.python.org/3/library/stdtypes.html#str))：文件路径。

### save(*path:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)


将成本模型保存到给定的文件位置。
* **参数：path** ([str](https://docs.python.org/3/library/stdtypes.html#str))：文件路径。

### update(*context:*[TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext), *candidates:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]*, *results:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[RunnerResult]*) → [None](https://docs.python.org/3/library/constants.html#None)


根据运行结果更新成本模型。
* **参数：**
   * **context** ( [TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)*,* ) ：调整上下文。
   * candidates（*列表*[ [MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]*）：衡量 candidates。
   * *results*( *List[RunnerResult]* ) ：度量候选的运行结果。

### predict(*context:*[TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext), *candidates:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]*) → ndarray

使用成本模型预测标准化分数。
* **参数：**
   * **context** ( [TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)*,* ) ：调整上下文。
   * candidates（*列表*[ [MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]*）：衡量 candidates。
* **返回：result**：预测的标准分数。
* **返回类型：** np.ndarray。

### *static* create(*kind: typing_extensions.Literal[xgb, mlp, random, none]*, args*, ***kwargs*) → [CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)


创建一个 CostModel。
* **参数：** 成本模型的种类。可以是“xgb”、“mlp”、“random”或“none”。
* **返回：cost_model**[：](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)创建的成本模型。
* **返回类型：**[CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)

## *class* tvm.meta_schedule.Database

抽象数据库接口。


**方法：**

|[has_workload](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.has_workload)(mod)|检查数据库是否具有给定的工作负载。|
|:----|:----|
|[commit_workload](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.commit_workload)(mod)|如果缺失，则将工作负载提交到数据库。|
|[commit_tuning_record](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.commit_tuning_record)(record)|将调整记录提交到数据库。|
|[get_top_k](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.get_top_k)(workload, top_k)|从数据库中获取给定工作负载的前 K 条有效调优记录。|
|[get_all_tuning_records](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.get_all_tuning_records)()|从数据库中获取所有调优记录。|
|[query_tuning_record](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.query_tuning_record)(mod, target, workload_name)|从数据库中查询给定工作量的最佳记录。|
|[query_schedule](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.query_schedule)(mod, target, workload_name)|从数据库中查询给定工作负载的最佳调度。|
|[query_ir_module](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.query_ir_module)(mod, target, workload_name)|从数据库中查询给定工作负载的最佳 IRModule。|
|[dump_pruned](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.dump_pruned)(destination)|将修剪后的数据库转储为 JSONDatabase 格式的文件。|
|[query](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.query)(mod, target, *[, workload_name, kind])|查询数据库以检索给定工作负载的最佳优化结果。|
|[current](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.current)()|获取范围内的当前数据库。|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database.create)([kind])|创建数据库。|

### has_workload(*mod:*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)) → [bool](https://docs.python.org/3/library/functions.html#bool)


检查数据库是否具有给定的工作负载。:param mod: 要搜索的 IRModule。:type mod: IRModule。
* **返回：result**：数据库是否具有给定的工作负载。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

### commit_workload(*mod:*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)) → Workload


如果缺失，则将工作负载提交到数据库。
* **参数：mod** ([IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule))：要搜索或添加的 IRModule。
* **返回：workload**[：](https://tvm.apache.org/docs/reference/api/python/topi.html#tvm.topi.nn.Workload)与给定 IRModule 对应的工作负载。
* **返回类型：**[Workload](https://tvm.apache.org/docs/reference/api/python/topi.html#tvm.topi.nn.Workload)。

### commit_tuning_record(*record: TuningRecord*) → [None](https://docs.python.org/3/library/constants.html#None)


将调整记录提交到数据库。
* **参数：record** (*TuningRecord*)：要添加的调整记录。

### get_top_k(*workload: Workload*, *top_k:*[int](https://docs.python.org/3/library/functions.html#int)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[TuningRecord]


从数据库中获取给定工作负载的前 K 条有效调优记录。
* **参数：**
   * **工作负载**（[Workload](https://tvm.apache.org/docs/reference/api/python/topi.html#tvm.topi.nn.Workload)） ：要搜索的工作负载。
   * **top_k** ( [int](https://docs.python.org/3/library/functions.html#int) ) ：要获取的顶级记录的数量。
* **返回：top_k_records**：前 K 条记录。
* **返回类型：** List[TuningRecord]。

### get_all_tuning_records() → [List](https://docs.python.org/3/library/typing.html#typing.List)[TuningRecord]


从数据库中获取所有调优记录。
* **返回：tuning_records**：来自数据库的所有调整记录。
* **返回类型：** List[TuningRecord]。

### query_tuning_record(*mod:*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule), *target:*[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target), *workload_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → TuningRecord | [None](https://docs.python.org/3/library/constants.html#None)


从数据库中查询给定工作量的最佳记录。
* **参数：**
   * **mod**（[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)） ：要搜索的 IRModule。
   * **目标**（[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)） ：要搜索的目标。
   * **workload_name** ( [str](https://docs.python.org/3/library/stdtypes.html#str) ) ：要搜索的工作负载的名称。
* **返回：tuning_record**：给定工作负载的最佳记录；如果未找到，则返回 None。
* **返回类型：** Optional[TuningRecord]。

### query_schedule(*mod:*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule), *target:*[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target), *workload_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule) | [None](https://docs.python.org/3/library/constants.html#None)


从数据库中查询给定工作负载的最佳调度。
* **参数：**
   * **mod**（[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)） ：要搜索的 IRModule。
   * **目标**（[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)） **：** 要搜索的目标。
   * **workload_name** ( [str](https://docs.python.org/3/library/stdtypes.html#str) ) ：要搜索的工作负载的名称。
* **返回：schedule**：给定工作负载的最佳调度；如果未找到，则为 None。
* **返回类型：** Optional[tvm.tir.Schedule]。

### query_ir_module(*mod:*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule), *target:*[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target), *workload_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule) | [None](https://docs.python.org/3/library/constants.html#None)


从数据库中查询给定工作负载的最佳 IRModule。
* **参数：**
   * **mod**（[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)） ：要搜索的 IRModule。
   * **目标**（[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)） **：** 要搜索的目标。
   * **workload_name** ( [str](https://docs.python.org/3/library/stdtypes.html#str) ) ：要搜索的工作负载的名称。
* **返回：ir_module**：给定工作负载的最佳 IRModule；如果未找到，则为 None。
* **返回类型：** Optional[[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)]。

### dump_pruned(*destination:*[Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)) → [None](https://docs.python.org/3/library/constants.html#None)


将修剪后的数据库转储为 JSONDatabase 格式的文件。
* **参数：destination** ([Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database))：要转储到的目标数据库。

### query(*mod:*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule), *target:*[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target), *, *workload_name: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'main'*, *kind: typing_extensions.Literal[schedule] | typing_extensions.Literal[record] | typing_extensions.Literal[ir_module] = 'schedule'*) → [Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule) | [IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule) | TuningRecord


查询数据库以检索给定工作负载的最佳优化结果。
* **参数：**
   * **mod**（[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)） ：要搜索的 IRModule。
   * **目标**（[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)） *：* 要搜索的目标。
   * **kind**（*str =“schedule”|“record”|“ir_module”*）**：** 要返回的优化结果的类型。
* **返回：result** *：* 给定工作负载的最佳优化结果。
* **返回类型：** Union[tvm.tir.Schedule, [IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule), TuningRecord]。

### *static* current() → [Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database) | [None](https://docs.python.org/3/library/constants.html#None)


获取范围内的当前数据库。

### *static* create(*kind: typing_extensions.Literal[json, memory, union, ordered_union] |*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*[[*[Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule)*],*[bool](https://docs.python.org/3/library/functions.html#bool)*] = 'json'*, args*, ***kwargs*) → [Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)


创建数据库。
* **参数：**
   * **kind** (*str = "json"|"memory"|"union"|"ordered_union"|Callable**[****[**tvm.tir.Schedule****],*)。
   * **bool** ：要创建的数据库类型。支持以下类型：“json”、“memory”、“union”、“ordered_union”和自定义调度函数。
* **返回：database**：创建的数据库。
* **返回类型：**[Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)。

## *class* tvm.meta_schedule.ExtractedTask(*task_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *mod:*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule), *target:*[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target), *dispatched:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)*]*, *weight:*[int](https://docs.python.org/3/library/functions.html#int))


从高级 IR 中提取的调优任务。
* **参数：**
   * **task_name** ( [str](https://docs.python.org/3/library/stdtypes.html#str) ) [：](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)提取的任务名称。
   * **mod** ( [IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule) ) ：高级 IR。
   * **目标**（[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)） *：* 目标信息。
   * *dispatched*( *List[*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)*]* ) ：高级 IR 可能分派到的低级 IR 列表。
   * **weight**（[int](https://docs.python.org/3/library/functions.html#int)） ：任务的权重。

## *class* tvm.meta_schedule.FeatureExtractor


从测量候选中提取特征以用于成本模型。


**方法：**

|[extract_from](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.FeatureExtractor.extract_from)(context, candidates)|从给定的测量候选中提取特征。|
|:----|:----|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.FeatureExtractor.create)(kind, *args, **kwargs)|创建一个 CostModel。|

### extract_from(*context:*[TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext), *candidates:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[NDArray](https://tvm.apache.org/docs/reference/api/python/runtime/ndarray.html#tvm.runtime.ndarray.NDArray)]


从给定的测量候选中提取特征。
* **参数：**
   * **context**（[TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)）[：](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)特征提取的调整上下文。
   * **candidates*（*列表**[ [MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]*）：从中提取特征的测量 candidates。
* **返回：features** *：* tvm ndarray 提取的特征。
* **返回类型：** List[[NDArray](https://tvm.apache.org/docs/reference/api/python/runtime/ndarray.html#tvm.runtime.ndarray.NDArray)]。

### *static* create(*kind: typing_extensions.Literal[per - store - feature]*, args*, ***kwargs*) → [FeatureExtractor](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.FeatureExtractor)


创建一个 CostModel。

## *class* tvm.meta_schedule.MeasureCallback 


测量结果出来后适用的规则可用。


**方法：**

|[apply](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCallback.apply)(task_scheduler, task_id, …)|将测量回调应用于给定的调度。|
|:----|:----|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCallback.create)(kind)|创建测量回调列表。|

### apply(*task_scheduler:*[TaskScheduler](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler), *task_id:*[int](https://docs.python.org/3/library/functions.html#int), *measure_candidates:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]*, *builder_results:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[BuilderResult]*, *runner_results:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[RunnerResult]*) → [None](https://docs.python.org/3/library/constants.html#None)


将测量回调应用于给定的调度。
* **参数：**
   * **task_scheduler**（[TaskScheduler](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler)） ：任务调度程序。
   * **task_id** ( [int](https://docs.python.org/3/library/functions.html#int) ) *：* 任务 ID。
   * *measure_candidates* ( *List[*[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]* ) ：测量候选对象。
   * *builder_results* ( *List[BuilderResult]* ) *：* 通过构建度量候选来获得构建器结果。
   * *runner_results* ( *List[RunnerResult]* ) *：* 通过运行构建的测量候选结果来获得运行器的结果。

### *static* create(*kind: typing_extensions.Literal[default]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasureCallback](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCallback)]


创建测量回调列表。

## *class* tvm.meta_schedule.Mutator


Mutator 旨在改变轨迹以探索设计空间。


**方法：**

|[apply](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Mutator.apply)(trace)|将变异函数应用于给定的跟踪。|
|:----|:----|
|[clone](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Mutator.clone)()|克隆变异器。|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Mutator.create)(kind)|创建默认变量列表。|

### apply(*trace:*[Trace](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Trace)) → [Trace](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Trace) | [None](https://docs.python.org/3/library/constants.html#None)


将变异函数应用于给定的跟踪。
* **参数：trace** ([Trace](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Trace))[：](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Mutator)给定的突变轨迹。
* **返回：trace**：如果变异器失败则返回 None，否则返回变异的跟踪。
* **返回类型：** Optional[[Trace](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Trace)]。

### clone() → [Mutator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Mutator)


克隆变异器。
* **返回：mutator**：–已克隆的 mutator。
* **返回类型：**[Mutator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Mutator)。

### *static* create(*kind: typing_extensions.Literal[llvm, cuda, cuda - tensorcore, hexagon]*) → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Mutator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Mutator), [float](https://docs.python.org/3/library/functions.html#float)]


创建默认变量列表。
* **参数：kind** (Literal[**"llvm"**, "cuda","cuda-tensorcore"*, "hexagon"])：tensorcore”* ，“hexagon”） **：** 变量的种类。
* **返回：mutators**：修改器列表。
* **返回类型：** List[[Mutator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Mutator)]。

## *class* tvm.meta_schedule.Postproc


将后处理器应用于调度的规则。


**方法：**

|[apply](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Postproc.apply)(sch)|将后处理器应用于给定的调度。|
|:----|:----|
|[clone](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Postproc.clone)()|克隆后处理器。|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Postproc.create)(kind)|创建默认后处理器列表。|

### apply(*sch:*[Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule)) → [bool](https://docs.python.org/3/library/functions.html#bool)


将后处理器应用于给定的调度。
* **参数：sch** (*tvm.tir.Schedule*)**：** 需要进行后期处理的调度。
* **返回：result**：后处理器是否成功应用。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

### clone() → [Postproc](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Postproc)


克隆后处理器。
* **返回：cloned_postproc**：克隆的后处理器。
* **返回类型：**[Postproc](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Postproc)。

### *static* create(*kind: typing_extensions.Literal[llvm, cuda, cuda - tensorcore, hexagon]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Postproc](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Postproc)]


创建默认后处理器列表。
* **参数：kind** (*Literal**[****"llvm"**,*** ***"cuda"****,"cuda-tensorcore"**,*** ***"hexagon"****]*)*：*tensorcore”* *，“hexagon”*） ：后处理器的种类。
* **返回：postprocs**：后处理器列表。
* **返回类型：** List[[Mutator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Mutator)]。

## *class* tvm.meta_schedule.Profiler


调整时间分析器。


**方法：**

|[get](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Profiler.get)()|几秒钟内即可获得分析结果。|
|:----|:----|
|[table](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Profiler.table)()|以表格形式获取分析结果。|
|[current](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Profiler.current)()|获取当前分析器。|
|[timeit](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Profiler.timeit)(name)|Timeit 代码块。|

### get() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]


几秒钟内即可获得分析结果。

### table() → [str](https://docs.python.org/3/library/stdtypes.html#str)

以表格形式获取分析结果。

### *static* current() → [Profiler](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Profiler) | [None](https://docs.python.org/3/library/constants.html#None)


获取当前分析器。

### *static* timeit(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str))

Timeit 代码块。

## *class* tvm.meta_schedule.Runner

抽象运行器接口。


**方法：**

|[run](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Runner.run)(runner_inputs)|运行构建的工件并获取运行器未来。|
|:----|:----|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Runner.create)([kind])|创建一个 Runner。|

### run(*runner_inputs:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[RunnerInput]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[RunnerFuture]


运行构建的工件并获取运行器未来。
* **参数：runner_inputs** (*List*[*RunnerInput]*)*：* 运行器的输入。
* **返回：runner_futures**：运行器的未来。
* **返回类型：** List[RunnerFuture]

### *static* create(*kind: typing_extensions.Literal[local, rpc] = 'local'*, args*, ***kwargs*) → [Runner](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Runner) 


创建一个 Runner。

## *class* tvm.meta_schedule.ScheduleRule


修改调度中的块的规则。


**方法：**

|[apply](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule.apply)(sch, block)|将调度规则应用于给定调度中的特定块。|
|:----|:----|
|[clone](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule.clone)()|深度克隆调度规则。|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule.create)(kind)|为给定类型创建调度规则列表。|

### apply(*sch:*[Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule), *block:*[BlockRV](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.BlockRV)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule)]


将调度规则应用于给定调度中的特定块。
* **参数：**
   * **sch**（*tvm.tir.Schedule*）：要修改的调度。
   * **块**（[BlockRV](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.BlockRV)）：应用调度规则的特定块。
* **返回：design_spaces**[：](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule)应用调度规则生成的调度列表。
* **返回类型：** List[tvm.tir.Schedule]。

### clone() → [ScheduleRule](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule)


深度克隆调度规则。
* **返回：cloned_rule**[：](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule)克隆的调度规则。
* **返回类型：**[ScheduleRule](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule)。

### *static* create(*kind: typing_extensions.Literal[llvm, cuda, cuda - tensorcore, hexagon]*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[ScheduleRule](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule)]


为给定类型创建调度规则列表。
* **参数：kind** (*Literal**[****"llvm"**,*** ***"cuda"****,"cuda-tensorcore"**,*** ***"hexagon"****]*)：tensorcore"* *,"hexagon"]*） ：调度规则的种类。
* **返回：rules**[：](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule)调度规则列表。
* **返回类型：** List[[ScheduleRule](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.ScheduleRule)]。

## *class* tvm.meta_schedule.MeasureCandidate(*sch:*[Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule), *args_info:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[ArgInfo]*)

衡量候选类别。
* **参数：**
   * **sch**（*tvm.tir.Schedule*）：要测量的调度。
   * *args_info* ( *List[ArgInfo]* ) ：参数信息。

## *class* tvm.meta_schedule.SearchStrategy


搜索策略是生成度量候选的类。


**方法：**

|[pre_tuning](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy.pre_tuning)(max_trials, num_trials_per_iter, …)|预先调整搜索策略。|
|:----|:----|
|[post_tuning](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy.post_tuning)()|对搜索策略进行后期调整。|
|[generate_measure_candidates](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy.generate_measure_candidates)()|从设计空间生成测量候选以进行测量。|
|[notify_runner_results](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy.notify_runner_results)(measure_candidates, …)|使用分析结果更新搜索策略。|
|[clone](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy.clone)()|克隆搜索策略。|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy.create)([kind])|创建搜索策略。|

### pre_tuning(*max_trials:*[int](https://docs.python.org/3/library/functions.html#int), *num_trials_per_iter:*[int](https://docs.python.org/3/library/functions.html#int), *design_spaces:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule)*]*, *database:*[Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *cost_model:*[CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None)

预先调整搜索策略。
* **参数：**
   * **max_trials** ( [int](https://docs.python.org/3/library/functions.html#int) ) ：最大试验次数。
   * **num_trials_per_iter**（[int](https://docs.python.org/3/library/functions.html#int)）*：* 每次迭代的试验次数。
   * design_spaces（List *[tvm.tir.Schedule]*）：调整过程中使用的设计空间。
   * *数据库*（*可选**[[数据库](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)] *= 无*） *：* 调整过程中使用的数据库。
   * *cost_model**（*可选**[* [CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)*]= None*）：调整过程中使用的成本模型。

### post_tuning() → [None](https://docs.python.org/3/library/constants.html#None)


对搜索策略进行后期调整。

### generate_measure_candidates() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)] | [None](https://docs.python.org/3/library/constants.html#None)


从设计空间生成测量候选以进行测量。
* **返回：measure_candidates** *：* 生成的测量候选，如果完成则为 None。
* **返回类型：** Optional[List[[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)]]。

### notify_runner_results(*measure_candidates:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]*, *results:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[RunnerResult]*) → [None](https://docs.python.org/3/library/constants.html#None)

使用分析结果更新搜索策略。
* **参数：**
   * *measure_candidates* ( *List[*[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]* ) ：需要更新的测量候选。
   * *results* ( *List[RunnerResult]* ) ：来自运行器的分析结果。

### clone() → [SearchStrategy](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy)


克隆搜索策略。
* **返回：cloned**：克隆的搜索策略。
* **返回类型：**[SearchStrategy](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy)。

### *static* create(*kind: typing_extensions.Literal[evolutionary, replay - trace, replay - func] = 'evolutionary'*, args*, ***kwargs*) → [SearchStrategy](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy)


创建搜索策略。

## *class* tvm.meta_schedule.SpaceGenerator


抽象设计空间生成器接口。


**方法：**

|[generate_design_space](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SpaceGenerator.generate_design_space)(mod)|给定一个模块生成设计空间。|
|:----|:----|
|[clone](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SpaceGenerator.clone)()|克隆设计空间生成器。|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SpaceGenerator.create)([kind])|创建一个设计空间生成器。|

### generate_design_space(*mod:*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule)]


给定一个模块生成设计空间。
* **参数：mod** ([IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule))：用于设计空间生成的模块。
* **返回：design_spaces**：生成的设计空间，即调度。
* **返回类型：** List[tvm.tir.Schedule]。

### clone() → [SpaceGenerator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SpaceGenerator)


克隆设计空间生成器。
* **返回：cloned_sg** *：* 克隆的设计空间生成器。
* **返回类型：**[SpaceGenerator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SpaceGenerator)。

### *static* create(*kind: typing_extensions.Literal[post - order - apply, union] | ~typing.Callable[[~tvm.tir.schedule.schedule.Schedule], None] | ~typing.Callable[[~tvm.tir.schedule.schedule.Schedule], ~tvm.tir.schedule.schedule.Schedule] | ~typing.Callable[[~tvm.tir.schedule.schedule.Schedule], ~typing.List[~tvm.tir.schedule.schedule.Schedule]] = 'post-order-apply'*, args*, ***kwargs*) → [SpaceGenerator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SpaceGenerator)


创建一个设计空间生成器。

## *class* tvm.meta_schedule.TaskScheduler


抽象任务调度程序接口。


**方法：**

|[next_task_id](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler.next_task_id)()|获取下一个任务 ID。|
|:----|:----|
|[join_running_task](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler.join_running_task)(task_id)|等待任务完成。|
|[tune](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler.tune)(tasks, task_weights, max_trials_global, …)|自动调节。|
|[terminate_task](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler.terminate_task)(task_id)|终止任务。|
|[touch_task](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler.touch_task)(task_id)|触摸任务并更新其状态。|
|[print_tuning_statistics](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler.print_tuning_statistics)()|打印出人类可读的调整统计数据格式。|
|[create](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler.create)([kind])|创建任务调度程序。|

### next_task_id() → [int](https://docs.python.org/3/library/functions.html#int)


获取下一个任务 ID。
* **返回：next_task_id**：下一个任务 ID。
* **返回类型：**[int](https://docs.python.org/3/library/functions.html#int)。

### join_running_task(*task_id:*[int](https://docs.python.org/3/library/functions.html#int)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[RunnerResult]

等待任务完成。
* **参数：task_id** ([int](https://docs.python.org/3/library/functions.html#int))*：* 要加入的任务 ID。
* **返回：results**：结果列表。
* **返回类型：** List[RunnerResult]。

### tune(*tasks:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)*]*, *task_weights:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[float](https://docs.python.org/3/library/functions.html#float)*]*, *max_trials_global:*[int](https://docs.python.org/3/library/functions.html#int), *max_trials_per_task:*[int](https://docs.python.org/3/library/functions.html#int), *num_trials_per_iter:*[int](https://docs.python.org/3/library/functions.html#int), *builder:*[Builder](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Builder), *runner:*[Runner](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Runner), *measure_callbacks:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[MeasureCallback](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCallback)*]*, *database:*[Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)*|*[None](https://docs.python.org/3/library/constants.html#None), *cost_model:*[CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)*|*[None](https://docs.python.org/3/library/constants.html#None)) → [None](https://docs.python.org/3/library/constants.html#None)

自动调节。
* **参数：**
   * *任务( List *[*[TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)*]* ) ：作为任务的调整上下文列表。
   * *task_weights（*List *[*[float](https://docs.python.org/3/library/functions.html#float)*]*） ：任务权重列表。
   * **max_trials_global** ( [int](https://docs.python.org/3/library/functions.html#int) ) ：全局最大试验次数。
   * **max_trials_per_task** ( [int](https://docs.python.org/3/library/functions.html#int) ) ：每个任务的最大试验次数。
   * **num_trials_per_iter**（[int](https://docs.python.org/3/library/functions.html#int)）：每次迭代的试验次数。
   * **builder**（[Builder](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Builder)）：构建器。
   * **runner**（[Runner](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Runner)） ：运行器。
   * *measure_callbacks* ( *List[*[MeasureCallback](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCallback)*]* ) [：](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)测量回调列表。
   * *数据库**（*可选**[[数据库](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)]） ：数据库。
   * *cost_model**（*可选**[ [CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)*]*）：成本模型。

### terminate_task(*task_id:*[int](https://docs.python.org/3/library/functions.html#int)) → [None](https://docs.python.org/3/library/constants.html#None)


终止任务。
* **参数：task_id** ([int](https://docs.python.org/3/library/functions.html#int))*：*要终止的任务 ID。

### touch_task(*task_id:*[int](https://docs.python.org/3/library/functions.html#int)) → [None](https://docs.python.org/3/library/constants.html#None)

触摸任务并更新其状态。
* **参数：task_id** ([int](https://docs.python.org/3/library/functions.html#int))[：](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler)要检查的任务 ID。

### print_tuning_statistics() → [None](https://docs.python.org/3/library/constants.html#None)print_tuning_statistics) 


打印出人类可读的调整统计数据格式。

### *static* create(*kind: typing_extensions.Literal[round - robin, gradient] = 'gradient'*, args*, ***kwargs*) → [TaskScheduler](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TaskScheduler) 


创建任务调度程序。

## tvm.meta_schedule.tune_tir(*mod:*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)*|*[PrimFunc](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.PrimFunc), *target:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target), *work_dir:*[str](https://docs.python.org/3/library/stdtypes.html#str), *max_trials_global:*[int](https://docs.python.org/3/library/functions.html#int), ***, *max_trials_per_task: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *num_trials_per_iter: [int](https://docs.python.org/3/library/functions.html#int) = 64*, *builder: [Builder](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Builder) | typing_extensions.Literal[local] = 'local'*, *runner: [Runner](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Runner) | typing_extensions.Literal[local, rpc] = 'local'*, *database: [Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database) | typing_extensions.Literal[json, memory] = 'json'*, *cost_model: [CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel) | typing_extensions.Literal[xgb, mlp, random] = 'xgb'*, *measure_callbacks: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasureCallback](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCallback)] | [MeasureCallback](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCallback) | typing_extensions.Literal[default] = 'default'*, *task_scheduler: Literal[gradient, round - robin] = 'gradient'*, *space: Literal[post - order - apply, union] = 'post-order-apply'*, *strategy: Literal[replay - func, replay - trace, evolutionary] = 'evolutionary'*, *num_tuning_cores: typing_extensions.Literal[physical, logical] | [int](https://docs.python.org/3/library/functions.html#int) = 'physical'*, *seed: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *module_equality: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'structural'*, *special_space: Literal[post - order - apply, union]] | None = None*, *post_optimization: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = False*) → [Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)


调整 TIR 函数或 TIR 函数的 IRModule。
* **参数：**
   * *mod* ( *Union[*[ir.IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)*,*[tir.PrimFunc](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.PrimFunc)*]* ) ：要调整的 TIR IRModule。
   * *target* ( *Union[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)*]* ) ：需要调整的目标。
   * **work_dir** ( [str](https://docs.python.org/3/library/stdtypes.html#str) ) *：* 工作目录。
   * **max_trials_global** ( [int](https://docs.python.org/3/library/functions.html#int) ) ：全局运行的最大试验次数。
   * *max_trials_per_task**（*可选**[ [int](https://docs.python.org/3/library/functions.html#int)*]*）：每个任务运行的最大试验次数。
   * **num_trials_per_iter** ( [int](https://docs.python.org/3/library/functions.html#int) ) *：* 每次迭代运行的试验次数
   * **builder**（*Builder.BuilderType*） *：* 构建骑。
   * **runner**（*Runner.RunnerType*） *：* 运行器。
   * **数据库**（*Database.DatabaseType*） **：** 数据库。
   * **cost_model**（*CostModel.CostModelType*）：成本模型。
   * **measure_callbacks** ( *MeasureCallback.CallbackListType* ) ：测量回调。
   * **task_scheduler** ( *TaskScheduler.TaskSchedulerType* ) *：* 任务调度程序。
   * **空间**（*SpaceGenerator.SpaceGeneratorType*） **：** 空间生成器。
   * **策略**（*SearchStrategy.SearchStrategyType*） *：* 搜索策略。
   * **num_tuning_cores** ( *Union[Literal["physical","logical"],int]* ) *：* 调整期间要使用的 CPU 核心数。
   * *seed**（*可选**[ [int](https://docs.python.org/3/library/functions.html#int)*]*）*：* 随机数生成器的种子。
   * *module_equality* (*可选[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]* ) *：* 用于指定模块相等性测试和散列方法的字符串。
   * *special_space* (*可选[Mapping[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,SpaceGenerator.SpaceGeneratorType]]* ) ：从任务名称到该任务的特殊空间生成器的映射。
* **返回：database**：包含所有调优记录的数据库。
* **返回类型：**[Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)。

## tvm.meta_schedule.tune_tasks(***, *tasks: [List](https://docs.python.org/3/library/typing.html#typing.List)[[TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)]*, *task_weights: [List](https://docs.python.org/3/library/typing.html#typing.List)[[float](https://docs.python.org/3/library/functions.html#float)]*, *work_dir: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *max_trials_global: [int](https://docs.python.org/3/library/functions.html#int)*, *max_trials_per_task: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *num_trials_per_iter: [int](https://docs.python.org/3/library/functions.html#int) = 64*, *builder: [Builder](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Builder) | typing_extensions.Literal[local] = 'local'*, *runner: [Runner](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Runner) | typing_extensions.Literal[local, rpc] = 'local'*, *database: [Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database) | typing_extensions.Literal[json, memory] = 'json'*, *cost_model: [CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel) | typing_extensions.Literal[xgb, mlp, random] = 'xgb'*, *measure_callbacks: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasureCallback](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCallback)] | [MeasureCallback](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCallback) | typing_extensions.Literal[default] = 'default'*, *task_scheduler: Literal[gradient, round - robin] = 'gradient'*, *module_equality: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'structural'*, *post_optimization: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = False*) → [Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)

调整任务列表。使用任务调度程序。
   * **参数：**
   * *任务**（*列表**[ [TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)*]*） ：要调整的任务列表。
   * *task_weights（*List *[*[float](https://docs.python.org/3/library/functions.html#float)*]*）：每个任务的权重。
   * **work_dir** ( [str](https://docs.python.org/3/library/stdtypes.html#str) ) ：工作目录。
   * **max_trials_global** ( [int](https://docs.python.org/3/library/functions.html#int) ) **：** 全局运行的最大试验次数。
   * max_trials_per_task**（*可选**[ [int](https://docs.python.org/3/library/functions.html#int)*]*）*：* 每个任务运行的最大试验次数。
   * **num_trials_per_iter** ( [int](https://docs.python.org/3/library/functions.html#int) ) ：每次迭代运行的试验次数
   * **builder**（*Builder.BuilderType*） **：** 构建器。
   * **runner**（*Runner.RunnerType*） ：运行器。
   * **数据库**（*Database.DatabaseType*） *：* 数据库。
   * **cost_model**（*CostModel.CostModelType*）**：** 成本模型。
   * **measure_callbacks** ( *MeasureCallback.CallbackListType* ) *：* 测量回调。
   * **task_scheduler** ( *TaskScheduler.TaskSchedulerType* ) ：任务调度程序。
   * module_equality（*可选*[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]*）：用于指定模块相等性测试和哈希方法的字符串。它必须是以下之一：
   * 结构化：使用 StructuralEqual/Hash。
   * “ignore–ndarray”：与“structural”相同，但在相等时忽略 ndarray 原始数据。
   
   测试和散列。
   * “anchor–block”：对从中提取的锚块进行相等性测试和哈希处理

给定模块。“ignore–ndarray”变量用于提取的块或未找到锚块的情况。有关锚块的定义，请参阅 tir/analysis/analysis.py。
   * **post_optimization**（*可选*[*布尔值]*）：使用 Droplet Search 作为利用空间生成后优化。
* **返回：database**：包含所有调优记录的数据库。
* **返回类型：**[Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)。

## *class* tvm.meta_schedule.TuneContext(*mod:*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *, *target: [Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None*, *space_generator: SpaceGenerator.SpaceGeneratorType | [None](https://docs.python.org/3/library/constants.html#None) = None*, *search_strategy: SearchStrategy.SearchStrategyType | [None](https://docs.python.org/3/library/constants.html#None) = None*, *task_name: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'main'*, *rand_state: [int](https://docs.python.org/3/library/functions.html#int) = -1*, *num_threads: [int](https://docs.python.org/3/library/functions.html#int) | typing_extensions.Literal[physical, logical] = 'physical'*, *logger: [Logger](https://docs.python.org/3/library/logging.html#logging.Logger) | [None](https://docs.python.org/3/library/constants.html#None) = None*)


调整上下文类旨在包含调整任务的所有资源。
* **参数：**
   * *mod**（*可选**[ [IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)*]= None*） [：](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SpaceGenerator)要优化的工作负载。
   * *目标**（*可选**[[目标](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)] *= 无*） *：* 要优化的目标。
   * **space_generator** ( *Union[None,ScheduleFnType,*[SpaceGenerator](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SpaceGenerator)*]= None* ) *：* 设计空间生成器。
   * *search_strategy* ( *Union[None,*[SearchStrategy](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.SearchStrategy)*]= None* ) ：搜索策略。如果为 None ，则策略留空。
   * *task_name**（*可选**[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]= None*）：调整任务的名称。
   * **logger**（[logging.Logger](https://docs.python.org/3/library/logging.html#logging.Logger)） *：* 用于调整任务的记录器。
   * **rand_state** ( *int = –1* ) ：随机状态。需为 [1, 2^31–1] 范围内的整数，–1 表示使用随机数。
   * **num_threads**（*int = None*）：要使用的线程数，None 表示使用逻辑 CPU 数量。


**方法：**

|[generate_design_space](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext.generate_design_space)()|给定一个模块生成设计空间。|
|:----|:----|
|[pre_tuning](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext.pre_tuning)(max_trials[, …])|在调整之前，SearchStrategy 需要调用的方法来做必要的准备。|
|[post_tuning](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext.post_tuning)()|调用 SearchStrategy 进行调整后必要清理的方法。|
|[generate_measure_candidates](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext.generate_measure_candidates)()|从设计空间中生成一批测量候选对象以供测量。|
|[notify_runner_results](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext.notify_runner_results)(measure_candidates, …)|使用分析结果更新 SearchStrategy 中的状态。|
|[clone](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext.clone)()|克隆 TuneContext。|

### generate_design_space() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule)]


给定一个模块生成设计空间。


使用 self.mod 委托给 self.space_generator.generate_design_space
* **返回：design_spaces** *：* 生成的设计空间，即调度。
* **返回类型：** List[tvm.tir.Schedule]。

### pre_tuning(*max_trials:*[int](https://docs.python.org/3/library/functions.html#int), *num_trials_per_iter:*[int](https://docs.python.org/3/library/functions.html#int)*= 64*, *design_spaces:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Schedule](https://tvm.apache.org/docs/reference/api/python/tir/schedule.html#tvm.tir.schedule.Schedule)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *database:*[Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *cost_model:*[CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None)

在调整之前，SearchStrategy 需要调用的方法来做必要的准备。


委托给 self.search_strategy.pre_tuning。
* **参数：**
   * **max_trials** ( [int](https://docs.python.org/3/library/functions.html#int) ) *：* 要执行的最大试验次数。
   * **num_trials_per_iter**（*int = 64*）：每次迭代要执行的试验次数。
   * **design_spaces** ( *Optional[List[tvm.tir.Schedule]] )：调优过程中使用的设计空间。若为 None，则使用*self.generate_design_space()的结果。
   * *database**（*可选**[ [Database](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.Database)*]= None*）：调优过程中使用的数据库。如果为 None 且搜索策略为 EvolutionarySearch，则使用 tvm.meta_schedule.database.MemoryDatabase。
   * *cost_model* (*可选[*[CostModel](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.CostModel)*]= None* ) ：调优过程中使用的成本模型。如果为 None，且搜索策略为 EvolutionarySearch，则使用tvm.meta_schedule.cost_model.RandomModel。

### post_tuning() → [None](https://docs.python.org/3/library/constants.html#None)


调用 SearchStrategy 进行调整后必要清理的方法。


委托给 self.search_strategy.post_tuning。

### generate_measure_candidates() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)] | [None](https://docs.python.org/3/library/constants.html#None)


从设计空间中生成一批测量候选对象以供测量。


委托给 self.search_strategy.generate_measure_candidates。
* **返回：measure_candidates**：生成的测量候选，如果搜索完成则为 None。
* **返回类型：** Optional[List[[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)]]。

### notify_runner_results(*measure_candidates:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]*, *results:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[RunnerResult]*) → [None](https://docs.python.org/3/library/constants.html#None)


使用分析结果更新 SearchStrategy 中的状态。


委托给 self.search_strategy.notify_runner_results。
* **参数：**
   * *measure_candidates* ( *List[*[MeasureCandidate](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.MeasureCandidate)*]* ) **：** 需要更新的测量候选。
   * *results* ( *List[RunnerResult]* ) [：](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)来自运行器的分析结果。

### clone() → [TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)


克隆 TuneContext。
* **返回：cloned_context**：克隆的 TuneContext。
* **返回类型：**[TuneContext](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html#tvm.meta_schedule.TuneContext)。

## tvm.meta_schedule.derived_object(*cls:*[type](https://docs.python.org/3/library/functions.html#type)) → [type](https://docs.python.org/3/library/functions.html#type)


用于为 TVM 对象注册派生子类的装饰器。
* **参数：cls** ([type](https://docs.python.org/3/library/functions.html#type))：要注册的派生类。
* **返回：cls**：装饰的 TVM 对象。
* **返回类型：**[type](https://docs.python.org/3/library/functions.html#type)。


**示例**

```python
@register_object("meta_schedule.PyRunner")
class _PyRunner(meta_schedule.Runner):
    def __init__(self, f_run: Callable = None):
        self.__init_handle_by_constructor__(_ffi_api.RunnerPyRunner, f_run)

class PyRunner:
    _tvm_metadata = {
        "cls": _PyRunner,
        "methods": ["run"]
    }
    def run(self, runner_inputs):
        raise NotImplementedError

@derived_object
class LocalRunner(PyRunner):
    def run(self, runner_inputs):
        ...
```


