---

title: tvm.error

---


TVM 中的结构化错误类。


每个错误类都接受一个错误消息作为输入。有关建议的消息格式，请参见示例部分。为了提高代码可读性，建议开发者复制示例并使用相同的消息格式来抛出错误。

:::note

另请参阅[错误处理指南](https://tvm.apache.org/docs/contribute/error_handling.html#error-handling-guide)。

:::


**函数：**

|[register_error](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.register_error)([name_or_cls, cls])|注册一个错误类，以便 FFI 错误处理器能够识别它。|
|:----|:----|


**异常：**

|[TVMError](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.TVMError)|TVM 的通用错误基类。|
|:----|:----|
|[InternalError](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.InternalError)|系统内部错误。|
|[RPCError](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.RPCError)|由远程服务器在处理 RPC 调用时抛出的错误。|
|[RPCSessionTimeoutError](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.RPCSessionTimeoutError)|当 RPC 会话过期时由远程服务器抛出的错误。|
|[OpError](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.OpError)|前端所有算子错误的基类。|
|[OpNotImplemented](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.OpNotImplemented)|算子未实现。|
|[OpAttributeRequired](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.OpAttributeRequired)|找不到所需的算子属性。|
|[OpAttributeInvalid](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.OpAttributeInvalid)|前端算子接收的属性值无效。|
|[OpAttributeUnImplemented](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.OpAttributeUnImplemented)|在某些前端中不支持该属性。|
|[DiagnosticError](https://tvm.apache.org/docs/reference/api/python/error.html#tvm.error.DiagnosticError)|在执行某个 Pass 时报告的错误诊断。|



## **tvm.error.register_error(*name_or_cls=None*,*cls=None*)**

注册一个错误类，以便 FFI 错误处理器能够识别它。
* **参数：**
   * **name_or_cls** ([str](https://docs.python.org/3/library/stdtypes.html#str) 或 class) – 错误类的名称或类对象。 
   * **cls** (class) – 要注册的类。
* **返回：fregister** – 如果未指定 f，则返回用于注册的函数。
* **返回类型：** function


**示例：**

```plain
@tvm.error.register_error
class MyError(RuntimeError):
    pass

err_inst = tvm.error.create_ffi_error("MyError: xyz")
assert isinstance(err_inst, MyError)
```


## ***exception*tvm.error.TVMError**

## ***exception*tvm.error.InternalError**
系统内部错误。


**示例**


```plain
// C++ 示例
LOG(FATAL) << "InternalError: internal error detail.";
```

```plain
# Python 示例
raise InternalError("internal error detail")
```


## ***exception*tvm.error.RPCError**

由远程服务器在处理 RPC 调用时抛出的错误。


## ***exception*tvm.error.RPCSessionTimeoutError**

当 RPC 会话过期时由远程服务器抛出的错误。


## ***exception*tvm.error.OpError**

前端所有算子错误的基类。


## ***exception*tvm.error.OpNotImplemented**
算子未实现。


**示例**

```plain
raise OpNotImplemented(
    "Operator {} is not supported in {} frontend".format(
        missing_op, frontend_name))
```


## ***exception*tvm.error.OpAttributeRequired**

所需属性未找到。


**示例**

```plain
raise OpAttributeRequired(
    "Required attribute {} not found in operator {}".format(
        attr_name, op_name))
```


## ***exception*tvm.error.OpAttributeInvalid**

在接收前端算子时，属性值无效。


**示例**

```plain
raise OpAttributeInvalid(
    "Value {} in attribute {} of operator {} is not valid".format(
        value, attr_name, op_name))
```


## ***exception*tvm.error.OpAttributeUnImplemented**

属性在该前端中不支持。


**示例**

```plain
raise OpAttributeUnImplemented(
    "Attribute {} is not supported in operator {}".format(
        attr_name, op_name))
```


## ***exception*tvm.error.DiagnosticError**
在执行 Pass 时报告的错误诊断。


有关详细错误信息，请参见配置的诊断渲染器。


