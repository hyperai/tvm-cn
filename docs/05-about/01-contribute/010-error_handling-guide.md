---

title: 错误处理指南

---

* [在 C++ 中抛出特定错误](https://tvm.apache.org/docs/contribute/error_handling.html#raise-a-specific-error-in-c)
* [如何选择错误类型](https://tvm.apache.org/docs/contribute/error_handling.html#how-to-choose-an-error-type)


 TVM 包含结构化的错误类，用于指示特定类型的错误。请尽可能抛出特定的错误类型，以便用户在必要时可以编写代码处理某一类特定错误。在 Python 中，你可以直接抛出具体的错误对象。而在 C++ 等其他语言中，你只需在错误信息前添加 `<ErrorType>:` 前缀（见下文）。

**注意**

 请参阅 `tvm.error` 获取完整的错误类型列表。


## 在 C++ 中抛出特定错误


 你可以在错误信息前添加 `<ErrorType>:` 前缀，以抛出对应类型的错误。注意，如果消息中没有前缀，则默认抛出 `tvm.error.TVMError`。该机制适用于 `LOG(FATAL)` 和 `ICHECK` 宏。以下代码展示了如何实现这一点：

```plain
// src/api_test.cc
void ErrorTest(int x, int y) {
  ICHECK_EQ(x, y) << "ValueError: expect x and y to be equal."
  if (x == 1) {
    LOG(FATAL) << "InternalError: cannot reach here";
  }
}
```


 上述函数作为 PackedFunc 注册到 Python 前端，名称为 `tvm._api_internal._ErrorTest`。以下是调用该函数时的表现：

```plain
>>> import tvm
>>> tvm.testing.ErrorTest(0, 1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/path/to/tvm/python/tvm/_ffi/_ctypes/function.py", line 190, in __call__
    raise get_last_ffi_error()
ValueError: Traceback (most recent call last):
  [bt] (3) /path/to/tvm/build/libtvm.so(TVMFuncCall+0x48) [0x7fab500b8ca8]
  [bt] (2) /path/to/tvm/build/libtvm.so(+0x1c4126) [0x7fab4f7f5126]
  [bt] (1) /path/to/tvm/build/libtvm.so(+0x1ba2f8) [0x7fab4f7eb2f8]
  [bt] (0) /path/to/tvm/build/libtvm.so(+0x177d12) [0x7fab4f7a8d12]
  File "/path/to/tvm/src/api/api_test.cc", line 80
ValueError: Check failed: x == y (0 vs. 1) : expect x and y to be equal.
>>>
>>> tvm.testing.ErrorTest(1, 1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/path/to/tvm/python/tvm/_ffi/_ctypes/function.py", line 190, in __call__
    raise get_last_ffi_error()
tvm.error.InternalError: Traceback (most recent call last):
  [bt] (3) /path/to/tvm/build/libtvm.so(TVMFuncCall+0x48) [0x7fab500b8ca8]
  [bt] (2) /path/to/tvm/build/libtvm.so(+0x1c4126) [0x7fab4f7f5126]
  [bt] (1) /path/to/tvm/build/libtvm.so(+0x1ba35c) [0x7fab4f7eb35c]
  [bt] (0) /path/to/tvm/build/libtvm.so(+0x177d12) [0x7fab4f7a8d12]
  File "/path/to/tvm/src/api/api_test.cc", line 83
InternalError: cannot reach here
TVM hint: You hit an internal error. Please open a thread on https://discuss.tvm.ai/ to report it.
```


 如上例所示，TVM 的 ffi 系统会将 Python 和 C++ 的堆栈信息合并成一条错误消息，并自动生成对应的错误类。

## 如何选择错误类型


 你可以查看下方列出的错误类型，结合常识，并参考现有代码中的用法进行选择。我们希望维持合理数量的错误类型。如果你认为需要新增错误类型，请按以下步骤操作：
*  提交一份 RFC 提案，包含描述及当前代码中的使用示例。 
*  在 `tvm.error` 中添加新的错误类型，并编写清晰的文档。 
*   更新本文档，将新错误类型加入列表。 
*   修改代码，使用新的错误类型。 


 我们也建议在编写简短的错误信息时尽量减少抽象层级。这样代码更易读，也便于在必要时构造具体的错误提示。

```plain
def preferred():
    
    # 明确指出抛出的是哪类错误及其信息。
    raise OpNotImplemented("Operator relu is not implemented in the MXNet frontend")

def _op_not_implemented(op_name):
    return OpNotImplemented("Operator {} is not implemented.").format(op_name)

def not_preferred():
    
    # 引入了额外的间接层级。
    raise _op_not_implemented("relu")
```


 如果需要引入构建多行错误信息的包装函数，请将该函数放在同一个文件中，以便其他开发者能方便地查看实现。

