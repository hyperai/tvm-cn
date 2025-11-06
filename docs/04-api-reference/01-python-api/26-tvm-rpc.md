---

title: tvm.rpc

---


轻量级 TVM RPC 模块。


RPC 支持连接到远程服务器、上传和启动函数。这对于交叉编译和远程测试非常有用。编译器堆栈在本地服务器上运行，而我们使用 RPC 服务器在没有可用编译器的远程运行时环境中运行。


测试程序在本地服务器上编译程序，上传并运行远程 RPC 服务器，返回结果验证正确性。


TVM RPC 服务器假定用户是可信的，需要在可信网络环境和加密通道中使用。它允许将任意文件写入服务器，并为任何可以访问此 API 的人提供完整的远程代码执行能力。


**类：**

|[Server](/docs/api-reference/python-api/tvm-rpc#class-tvmrpcserverhost0000-port9091-port_end9199-is_proxyfalse-tracker_addrnone-key-load_librarynone-custom_addrnone-silentfalse-no_forkfalse-server_init_callbacknone-reuse_addrtrue-timeoutnone)([host, port, port_end, is_proxy, …])|在单独的进程上启动 RPC 服务器。|
|:----|:----|
|[RPCSession](/docs/api-reference/python-api/tvm-rpc#class-tvmrpcrpcsessionsess)(sess)|RPC 客户端会话模块。|
|[LocalSession](/docs/api-reference/python-api/tvm-rpc#class-tvmrpclocalsession)()|由本地环境支持的 RPCSession 接口。|
|[PopenSession](/docs/api-reference/python-api/tvm-rpc#class-tvmrpcpopensessionbinary)(binary)|由 popen 支持的 RPCSession 接口。|
|[TrackerSession](/docs/api-reference/python-api/tvm-rpc#class-tvmrpctrackersessionaddr)(addr)|跟踪器客户端会话。|

**函数：**

|[connect](/docs/api-reference/python-api/tvm-rpc#tvmrpcconnecturl-port-key-session_timeout0-session_constructor_argsnone-enable_loggingfalse)(url, port[, key, session_timeout, …])|连接到 RPC 服务器。|
|:----|:----|
|[connect_tracker](/docs/api-reference/python-api/tvm-rpc#tvmrpcconnect_trackerurl-port)(url, port)|连接到 RPC 跟踪器。|
|[with_minrpc](/docs/api-reference/python-api/tvm-rpc#tvmrpcwith_minrpccompile_func-serverposix_popen_server-runtimelibtvm)(compile_func[, server, runtime])|使用 minrpc 相关选项附加将编译器函数。|

## *class* tvm.rpc.Server(*host='0.0.0.0'*, *port=9091*, *port_end=9199*, *is_proxy=False*, *tracker_addr=None*, *key=''*, *load_library=None*, *custom_addr=None*, *silent=False*, *no_fork=False*, *server_init_callback=None*, *reuse_addr=True*, *timeout=None*)


在单独的进程上启动 RPC 服务器。


这是一个基于多处理的简单 Python 实现。也可以使用 TVM 运行时实现类似的基于 C 的服务器，该服务器不依赖于 Python。
* **参数：**
   * **host** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：服务器的主机 URL。
   * **port**（[int](https://docs.python.org/3/library/functions.html#int)）：要绑定的端口。
   * **port_end** ( [int](https://docs.python.org/3/library/functions.html#int)*，可选*)：要搜索的结束端口。
   * **is_proxy**（[bool](https://docs.python.org/3/library/functions.html#bool)*，可选*）：指定的地址是否为代理。如果为真，则主机和端口实际上对应于代理服务器的地址。
   * **tracker_addr** ( [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*(*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[int](https://docs.python.org/3/library/functions.html#int)*),可选*)：RPC Tracker 的地址，以 tuple(host, ip) 格式表示。如果不为 None，则服务器会将自身注册到 Tracker。
   * **key**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：用于识别跟踪器中的设备类型的密钥。
   * **load_library**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：执行期间要加载的附加库列表。
   * **custom_addr**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：向 RPC 跟踪器报告的自定义 IP 地址。
   * **silent**（[bool](https://docs.python.org/3/library/functions.html#bool)*，可选*）：是否以静默模式运行此服务器。
   * **no_fork** ( [bool](https://docs.python.org/3/library/functions.html#bool)*,可选*)：是否禁止在多处理中分叉。
   * server_init_callback（*可调用*，可选）**：** 启动服务器时的附加初始化函数。
   * **reuse_addr** ( [bool](https://docs.python.org/3/library/functions.html#bool)*，可选*)：允许内核重用处于 TIME_WAIT 状态的本地套接字。
   * **timeout**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：设置套接字上所有操作的超时时间。

:::note

TVM RPC 服务器假定用户是可信的，需要在可信网络环境和加密通道中使用。它允许将任意文件写入服务器，并为任何可以访问此 API 的人提供完整的远程代码执行能力。


RPC 服务器只能访问 tvm 命名空间中的函数。要将其他自定义函数引入服务器环境，可以使用 server_init_callback。

```python
def server_init_callback()：
    import tvm
    # 必须在这里导入 mypackage。
    import mypackage

    tvm.register_func("function", mypackage.func)

server = rpc.Server(host, server_init_callback=server_init_callback)
```
:::

**方法：**

|[terminate](/docs/api-reference/python-api/tvm-rpc#terminate)()|终止服务器进程|
|:----|:----|

### terminate()

终止服务器进程。

## tvm.rpc.connect(*url*, *port*, *key=''*, *session_timeout=0*, *session_constructor_args=None*, *enable_logging=False*)


连接到 RPC 服务器。
* **参数：**
   * **url** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：主机的 url。
   * **port**（[int](https://docs.python.org/3/library/functions.html#int)）：要连接的端口。
   * **key**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：匹配服务器的附加键。
   * **session_timeout**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：会话的持续时间（以秒为单位），允许服务器在会话持续时间超过此值时终止连接。当会话持续时间为零时，表示请求必须始终保持活动状态。
   * **session_constructor_args** ( *List* )：作为远程会话构造函数传递的附加参数列表。列表的第一个元素始终是一个字符串，用于指定会话构造函数的名称，后续参数是该函数的位置参数。
   * **enable_logging**（*布尔值*）**：** 启用/禁用日志记录的标志。默认情况下，日志记录处于禁用状态。
* **返回：sess：** 连接的会话。
* **返回类型：**[RPCSession](/docs/api-reference/python-api/tvm-rpc#class-tvmrpcrpcsessionsess)。


**示例**

Normal usage .. code-block：： python.

client = rpc.connect(server_url, server_port, server_key).

Session_constructor 可用于自定义远程中的会话以下代码通过在代理机器上构造另一个 RPCClientSession 并通过代理连接到远程内部服务器并将其用作代理端点的服务会话。

```python
client_via_proxy = rpc.connect(
    proxy_server_url, proxy_server_port, proxy_server_key, enable_logging
    session_constructor_args=[
        "rpc.Connect", internal_url, internal_port, internal_key, internal_logging])
```
## tvm.rpc.connect_tracker(*url*, *port*)


连接到 RPC 跟踪器。
* **参数：**
   * **url** ( [str](https://docs.python.org/3/library/stdtypes.html#str))：主机的 url。
   * **port**（[int](https://docs.python.org/3/library/functions.html#int)）：要连接的端口。
* **返回：sess** ：连接的跟踪器会话。
* **返回类型：**[TrackerSession](/docs/api-reference/python-api/tvm-rpc#class-tvmrpctrackersessionaddr)。

## *class* tvm.rpc.RPCSession(*sess*)


RPC 客户端会话模块。


不要直接创建对象，调用 connect。


**方法：**

|[system_lib](/docs/api-reference/python-api/tvm-rpc#system_lib)()|获取系统范围的库模块。|
|:----|:----|
|[get_function](/docs/api-reference/python-api/tvm-rpc#get_functionname)(name)|从会话中获取函数。|
|[device](/docs/api-reference/python-api/tvm-rpc#devicedev_type-dev_id0)(dev_type[, dev_id])|构建一个远程设备。|
|[upload](/docs/api-reference/python-api/tvm-rpc#uploaddata-targetnone)(data[, target])|将文件上传到远程运行时临时文件夹。|
|[download](/docs/api-reference/python-api/tvm-rpc#downloadpath)(path)|从远程临时文件夹下载文件。|
|[remove](/docs/api-reference/python-api/tvm-rpc#removepath)(path)|从远程临时文件夹中删除文件。|
|[listdir](/docs/api-reference/python-api/tvm-rpc#listdirpath)(path)|从远程临时文件夹中 ls 文件。|
|[load_module](/docs/api-reference/python-api/tvm-rpc#load_modulepath)(path)|加载远程模块，需要先上传文件。|
|[download_linked_module](/docs/api-reference/python-api/tvm-rpc#download_linked_modulepath)(path)|链接远程模块并下载。|
|[cpu](/docs/api-reference/python-api/tvm-rpc#cpudev_id0)([dev_id])|构建 CPU 设备。|
|[cuda](/docs/api-reference/python-api/tvm-rpc#cudadev_id0)([dev_id])|构建 CUDA GPU 设备。|
|[cl](/docs/api-reference/python-api/tvm-rpc#cldev_id0)([dev_id])|构建 OpenCL 设备。|
|[vulkan](/docs/api-reference/python-api/tvm-rpc#vulkandev_id0)([dev_id])|构建 Vulkan 设备。|
|[metal](/docs/api-reference/python-api/tvm-rpc#metaldev_id0)([dev_id])|建造金属装置。|
|[rocm](/docs/api-reference/python-api/tvm-rpc#rocmdev_id0)([dev_id])|构建 ROCm 设备。|
|[ext_dev](/docs/api-reference/python-api/tvm-rpc#ext_devdev_id0)([dev_id])|构建扩展设备。|
|[hexagon](/docs/api-reference/python-api/tvm-rpc#hexagondev_id0)([dev_id])|构建六边形装置。|
|[webgpu](/docs/api-reference/python-api/tvm-rpc#webgpudev_id0)([dev_id])|构建 WebGPU 设备。|

### system_lib()

获取系统范围的库模块。
* **返回：module：** 系统范围的库模块。
* **返回类型：** runtime.Module。


:::info 另见

`tvm.runtime.system_lib`

:::

### get_function(*name*)

从会话中获取函数。
* **参数：name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：函数名称
* **返回：f** ：结果函数。
* **返回类型：**[Function](/docs/api-reference/python-api/tvm-relax#classtvmrelaxfunctionparamslistvarbodyrelaxexprret_struct_infostructinfononenoneis_pureboolnonetrueattrsdictattrsnonenonespanspannonenone)。

### device(*dev_type*, *dev_id=0*)


构建一个远程设备。
* **参数：**
   * **dev_type** ([int](https://docs.python.org/3/library/functions.html#int)*or*[str](https://docs.python.org/3/library/stdtypes.html#str))。
   * **dev_id** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)。
* **返回：dev** ：相应编码的远程设备。
* **返回类型：** Device。

### upload(*data*, *target=None*)


将文件上传到远程运行时临时文件夹
* **参数：**
   * **data**（[str](https://docs.python.org/3/library/stdtypes.html#str)*或*[bytearray](https://docs.python.org/3/library/stdtypes.html#bytearray)）：要上传的本地文件名或二进制文件。
   * **target**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：远程路径。

### download(*path*)


从远程临时文件夹下载文件。
* **参数：path** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：远程临时文件夹的相对位置。
* **返回：blob** ：来自文件的结果 blob。
* **返回类型：**[bytearray](https://docs.python.org/3/library/stdtypes.html#bytearray)。

### remove(*path*)


从远程临时文件夹中删除文件。
* **参数：path** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：远程临时文件夹的相对位置。

### listdir(*path*)


从远程临时文件夹中 ls 文件。
* **参数：path** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：远程临时文件夹的相对位置。
* **返回：dirs** ：给定目录中带有分割标记“，”的文件。
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)。

### load_module(*path*)


加载远程模块，需要先上传文件。
* **参数：path** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：远程临时文件夹的相对位置。
* **返回：m** ：包含远程函数的远程模块。
* **返回类型：** Module。

### download_linked_module(*path*)


链接远程模块并下载。
* **参数：path** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：远程临时文件夹的相对位置。
* **返回：blob** ：来自文件的结果 blob。
* **返回类型：**[bytearray](https://docs.python.org/3/library/stdtypes.html#bytearray)。

:::note

当本地客户端上没有可用的链接器时，此函数会很有帮助。

:::


**示例**

```python
mod = build_module_with_cross_compilation()
# 将模块导出为 tar 包，因为本地没有可用的链接器
mod.export_library("lib.tar")
remote.upload("lib.tar")
# 在远程调用链接器，将模块链接为库
# 注意，该库只能在与远程相同的环境下运行

with open("lib.so", "wb") as file：
    file.write(remote.download_linked_module("lib.tar"))
```
### cpu(*dev_id=0*)

构建 CPU 设备。

### cuda(*dev_id=0*) 


构建 CUDA GPU 设备。

### cl(*dev_id=0*) 


构建 OpenCL 设备。

### vulkan(*dev_id=0*)


构建 Vulkan 设备。

### metal(*dev_id=0*)

建造金属装置。

### rocm(*dev_id=0*)

构建 ROCm 设备。

### ext_dev(*dev_id=0*)

构建扩展设备。

### hexagon(*dev_id=0*)

构建六边形装置。

### webgpu(*dev_id=0*)


构建 WebGPU 设备。

## *class* tvm.rpc.LocalSession


由本地环境支持的 RPCSession 接口。


该类可用于实现需要在本地和远程运行的函数。

## *class* tvm.rpc.PopenSession(*binary*)

由 popen 支持的 RPCSession 接口。
* **参数：binary** (*List**[****Union**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,** [bytes](https://docs.python.org/3/library/stdtypes.html#bytes)***]****]*) [：](https://docs.python.org/3/library/stdtypes.html#tuple)要执行的二进制文件。

## *class* tvm.rpc.TrackerSession(*addr*)

跟踪器客户端会话。
* **参数：addr** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)) ：地址元组。


**方法：**

|[close](/docs/api-reference/python-api/tvm-rpc#close)()|关闭跟踪器连接。|
|:----|:----|
|[summary](/docs/api-reference/python-api/tvm-rpc#summary)()|获取跟踪器的摘要字典。|
|[text_summary](/docs/api-reference/python-api/tvm-rpc#text_summary)()|获取跟踪器的文本摘要。|
|[request](/docs/api-reference/python-api/tvm-rpc#requestkey-priority1-session_timeout0-max_retry5-session_constructor_argsnone)(key[, priority, session_timeout, …])|向跟踪器请求新的连接。|
|[request_and_run](/docs/api-reference/python-api/tvm-rpc#request_and_runkey-func-priority1-session_timeout0-max_retry2)(key, func[, priority, …])|从跟踪器请求资源并运行该函数。|

### close()

关闭跟踪器连接。

### summary()


获取跟踪器的摘要字典。

### text_summary()


获取跟踪器的文本摘要。

### request(*key*, *priority=1*, *session_timeout=0*, *max_retry=5*, *session_constructor_args=None*)


向跟踪器请求新的连接。
* **参数：**
   * **key** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：设备的类型键。
   * **优先级**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）请求的优先级。
   * **session_timeout**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：会话的持续时间，允许服务器在会话持续时间超过此值时终止连接。当会话持续时间为零时，表示请求必须始终保持活动状态。
   * **max_retry**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）放弃之前重试的最大次数。
   * **session_constructor_args**（[list](https://docs.python.org/3/library/stdtypes.html#list)*，可选*）作为远程会话构造函数传递的附加参数列表。列表的第一个元素始终是一个字符串，用于指定会话构造函数的名称，后续参数是该函数的位置参数。

### request_and_run(*key*, *func*, *priority=1*, *session_timeout=0*, *max_retry=2*)


从跟踪器请求资源并运行该函数。


此函数在执行过程中会避免罕见的服务器节点掉线。在这种情况下，将请求新的资源并再次运行函数。
* **参数：**
   * **key** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：设备的类型键。
   * **func** ( *session**函数***–> value)：无状态函数。
   * **优先级**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：请求的优先级。
   * **session_timeout**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：会话的持续时间，允许服务器在会话持续时间超过此值时终止连接。当会话持续时间为零时，表示请求必须始终保持活动状态。
   * **max_retry**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：放弃之前重试该函数的最大次数。

## tvm.rpc.with_minrpc(*compile_func*, *server='posix_popen_server'*, *runtime='libtvm'*)

使用 minrpc 相关选项附加将编译器函数。
* **参数：**
   * **compile_func** ( *Union[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,Callable[[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*,Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]],None]]* )：要装饰的编译函数。
   * **服务器**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：服务器类型。
   * **运行时**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：运行时库。
* **返回：fcompile** ：返回编译。
* **返回类型：** function。


