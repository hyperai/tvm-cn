---

title: tvm.runtime.disco

---

TVM 分布式运行时 API。


## *class* tvm.runtime.disco.DModule(*dref:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref), *session:*[Session](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscosession))

Disco 会话中的一个模块。


## *class* tvm.runtime.disco.DPackedFunc(*dref:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref), *session:*[Session](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscosession))
一个 Disco 会话中的 PackedFunc。


## *class* tvm.runtime.disco.DRef

一个在所有工作进程上都存在的对象。控制进程为每个对象分配一个唯一的「register ID」，工作进程使用这个 ID 来引用驻留在自身进程上的对象。


### debug_get_from_remote(*worker_id:*[int](https://docs.python.org/3/library/functions.html#int)) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

从远程 worker 获取 DRef 的值。它仅用于调试目的。
* **参数：worker_id** ([int](https://docs.python.org/3/library/functions.html#int)) ：需要获取的 worker 的 ID。
* **返回：value**：寄存器的值。
* **返回类型：**[object](https://docs.python.org/3/library/functions.html#object)。


### debug_copy_from(*worker_id:*[int](https://docs.python.org/3/library/functions.html#int),*value: ndarray |Tensor*)) → [None](https://docs.python.org/3/library/constants.html#None)

将 NDArray 值复制到远程以用于调试目的。
* **参数：**   
   * **worker_id** ([int](https://docs.python.org/3/library/functions.html#int)) ：要复制到的 worker 的 ID。
   * **value** (*Union**[****numpy.ndarray,*[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*]*) ：要复制的值。


## *class* tvm.runtime.disco.ProcessSession(*num_workers:*[int](https://docs.python.org/3/library/functions.html#int), *num_groups:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *entrypoint:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'tvm.exec.disco_worker'*)

一个由基于管道的多进程支持的 Disco 会话。


## *class* tvm.runtime.disco.Session

一个 Disco 交互式会话。它允许用户使用各种 PackedFunc 调用约定与 Disco 命令队列进行交互。


### empty(*shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *device: Device |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *worker0_only:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *in_group:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)

在所有 worker 上创建一个空的 NDArray 并将它们附加到 DRef。
* **参数：**
   * **shape** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*of*[int](https://docs.python.org/3/library/functions.html#int)) ：NDArray 的形状。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：NDArray 的数据类型。
   * **device** (*Optional**[****Device]= None*) ：NDArray 的设备。
   * **worker0_only** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：如果为 False（默认值），则为每个 worker 分配一个数组。如果为 True，则仅在 worker0 上分配一个数组。
   * **in_group** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：当worker0_only为 True时生效。如果为 True（默认），则在每个组的第一个 Worker 上分配一个数组。如果为 False，则仅在全局范围内为 worker0 分配一个数组。
* **返回：array** ：创建的 NDArray。
* **返回类型：** [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)。


### shutdown()

关闭 Disco 会话。


### *property* num_workers*:*[int](https://docs.python.org/3/library/functions.html#int)

返回会话中的 worker 数量。


### get_global_func(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)

获取 worker 的全局函数。
* **参数：name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：全局函数的名称。
* **返回：func** ：全局打包函数。
* **返回类型：**[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)。

### import_python_module(*module_name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None)

在每个 worker 中导入一个 python 模块。


通话前可能需要执行此操作。
* **参数：module_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：Python 模块名称，它将用于 Python 导入语句中。


### call_packed(*func:*[DRef](https://tvm.apache.org/docs/reference/api/python/runtime/disco.html#tvm.runtime.disco.DRef), *args) → [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)

在提供可变参数的 worker 上调用 PackedFunc。
* **参数：**
   * **func** (*PackedFunc*) ：要调用的函数。
   * ***args** (*various types*) ：DRef。
* **返回：return_value**：函数调用的返回值。
* **返回类型：** 各种类型。


:::Note

不支持的类型的示例： - NDArray、DLTensor； - TVM 对象，包括 PackedFunc、Module 和 String。

:::


### sync_worker_0() → [None](https://docs.python.org/3/library/constants.html#None)

将控制器与 worker-0 同步，它会等待直到 worker-0 执行完所有现有的指令。


### copy_from_worker_0(*host_array:Tensor, remote_array:[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)*) → [None](https://docs.python.org/3/library/constants.html#None)

将 NDArray 从 worker-0 复制到控制器端 NDArray。
* **参数：**
   * **host_array** (*numpy.ndarray*) ：0的数组。
   * **remote_array** (*[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)* ：0 上的 NDArray。


### copy_to_worker_0(*host_array:*Tensor, *remote_array:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)

将控制器端 NDArray 复制到 worker-0。
* **参数：**
   * **host_array** (*[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)* ) ：要复制到 worker-0 的数组
   * **remote_array** (*Optiona[*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)*]*) ：0 上的目标 NDArray。
* **返回：output_array：** 包含 worker0 上复制数据的 DRef，以及所有其他 Worker 上的 std::nullopt。如果提供了remote_array ，则此返回值与remote_array相同。否则，它是新分配的空间。
* **返回类型：** [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)。


### load_vm_module(*path:*[str](https://docs.python.org/3/library/stdtypes.html#str), *device: Device |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [DModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodmoduledrefdref-sessionsession)

从文件加载 VM 模块。
* **参数：**
   * **path** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：VM 模块文件的路径。
   * **device** (*Optional**[****Device]= None*) ：加载 VM 模块的设备。默认为每个 Worker 的默认设备。
* **返回：module** ：已加载的 VM 模块。
* **返回类型：**[DModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodmoduledrefdref-sessionsession)


### init_ccl(*ccl:*[str](https://docs.python.org/3/library/stdtypes.html#str), *device_ids)

初始化底层通信集合库。
* **参数：**
   * **ccl** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：通信集体库的名称。目前支持的库包括：-nccl - rccl - mpi。
   * ****device_ids** (*[int*](https://docs.python.org/3/library/functions.html#int)) ：底层通信库使用的设备 ID。


### broadcast(*src: ndarray |*Tensor, *dst:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *in_group:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)

向所有 worker 广播一个数组。
* **参数：**
   * **src** (*Union**[****np.ndarray,*[NDArray](https://tvm.apache.org/docs/reference/api/python/runtime/ndarray.html#tvm.runtime.ndarray.NDArray)*]*) ：要广播的数组。
   * **dst** (*Optional[*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)*]*) ：输出数组。如果为 None，则将在每个 Worker 上分配一个与src的形状和数据类型匹配的数组。
   * **in_group** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：广播操作默认是全局执行还是按组执行。
* **返回：output_array：** 包含所有 Worker 上广播数据的 DRef。如果提供了 dst ，则此返回值与 dst 相同 。否则，返回新分配的空间。
* **返回类型：** [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)。


### broadcast_from_worker0(*src:*[DRef](https://tvm.apache.org/docs/reference/api/python/runtime/disco.html#tvm.runtime.disco.DRef), *dst:*[DRef](https://tvm.apache.org/docs/reference/api/python/runtime/disco.html#tvm.runtime.disco.DRef), *in_group:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodreff)

将数组从 worker-0 广播到所有其他 worker。
* **参数：**
   * **src** (*Union**[****np.ndarray,*[NDArray](https://tvm.apache.org/docs/reference/api/python/runtime/ndarray.html#tvm.runtime.ndarray.NDArray)*]*) ：要广播的数组。
   * **dst** (*Optional[*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)*]*) ：输出数组。如果为 None，则将在每个 Worker 上分配一个与src的形状和数据类型匹配的数组。
   * **in_group** ([bool](https://docs.python.org/3/library/functions.html#bool)) 广播操作默认是全局执行还是按组执行。


### scatter(*src: ndarray |*[NDArray](https://tvm.apache.org/docs/reference/api/python/runtime/ndarray.html#tvm.runtime.ndarray.NDArray), *dst:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *in_group:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)

将数组分散到所有 worker 中。
* **参数：**
   * **src** (*Union**[****np.ndarray,*[NDArray](https://tvm.apache.org/docs/reference/api/python/runtime/ndarray.html#tvm.runtime.ndarray.NDArray)*]*) ：待分散的数组。该数组的第一个维度src.shape[0]必须等于工作线程的数量。
   * **dst** (*Optional[*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)*]*) ：输出数组。如果为 None，则将在每个 Worker 上分配一个形状兼容且数据类型与 src 相同的数组。
   * **in_group** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：分散操作是全局执行还是默认在组中执行。
* **返回：output_array**：包含所有 Worker 的分散数据的 DRef。如果提供了dst ，则此返回值与dst相同 。否则，返回新分配的空间。
* **返回类型：**[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)。


### scatter_from_worker0(*from_array:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref), *to_array:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref), *in_group:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [None](https://docs.python.org/3/library/constants.html#None)

将数组从 worker-0 分散到所有其他 workers。
* **参数：**
   * **src** (*Union**[****np.ndarray,*[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*]*) ：待分散的数组。该数组的第一个维度src.shape[0]必须等于工作线程的数量。
   * **dst** (*Optional[*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)*]*) ：输出数组。如果为 None，则将在每个 Worker 上分配一个形状兼容且数据类型与src相同的数组。
   * **in_group** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：分散操作是全局执行还是默认在组中执行。


### gather_to_worker0(*from_array:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref), *to_array:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref), *in_group:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [None](https://docs.python.org/3/library/constants.html#None)

将所有其他 worker 的数组收集到 worker-0。
* **参数：**
   * **from_array** ([DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)) ：要从中收集的数组。
   * **to_array** ([DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)) ：要收集的数组。
   * **in_group** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：收集操作是全局执行还是默认以组执行。


### allreduce(*src:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref), *dst:*[DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref), *op:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'sum'*, *in_group:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)

对数组执行 allreduce 操作。
* **参数：**  
   *  **array** ([DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)) ：要缩减的数组。
   * **op** (*str = "sum"*) ：要执行的归约操作。可用选项包括：- “sum”（求和）- “prod”（求积）- “min”（求最小值）- “max”（求最大值）- “avg”（求平均值）。
   * **in_group** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：减少操作是否全局执行或默认在组中执行。


### allgather(src: DRef, dst: DRef, in_group: bool = True)→ DRef

对一个数组执行 allgather 操作。
* **参数：**
   * **src** ([DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)) ：要收集的数组。
   * **dst** ([DRef](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-runtime-disco#class-tvmruntimediscodref)) ：要收集的数组。
   * **in_group** ([bool](https://docs.python.org/3/library/functions.html#bool)) ：减少操作是否全局执行或默认在组中执行。


## *classtvm.runtime.disco.ThreadedSession(num_workers: int, num_groups: int = 1)*

一个基于多线程的 Disco 会话。


## *class* tvm.runtime.disco.SocketSession(*num_nodes:*[int](https://docs.python.org/3/library/functions.html#int), *num_workers_per_node:*[int](https://docs.python.org/3/library/functions.html#int), *num_groups:*[int](https://docs.python.org/3/library/functions.html#int), *host:*[str](https://docs.python.org/3/library/stdtypes.html#str), *port:*[int](https://docs.python.org/3/library/functions.html#int))
 由基于套接字的多节点通信驱动的 Disco 会话。

