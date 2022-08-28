---
title: 处理报错
sidebar_position: 10
---

TVM 包含结构化的错误类以表示特定类型的错误。请尽可能提出特定的错误类型，以便用户可以在必要时写代码来处理特定的错误类别。可以直接在 Python 中抛出特定的错误对象。在 C++ 等其他语言中，只需给错误消息添加 `<ErrorType>:` 前缀（见下文）。

::: note
::: title
Note
:::

Please refer to :py`tvm.error`{.interpreted-text role="mod"} for the
list of errors.
:::

## 在 C++ 中抛出特定错误

可以给错误消息添加 `<ErrorType>:` 前缀来抛出相应类型的错误。注意，当消息中没有错误类型前缀时，不必添加新类型，默认会抛出 :py`tvm.error.TVMError`{.interpreted-text role="class"} 错误。此机制适用于 `LOG(FATAL)` 和 `ICHECK` 宏。具体示例见以下代码：

``` c
// src/api_test.cc
void ErrorTest(int x, int y) {
  ICHECK_EQ(x, y) << "ValueError: expect x and y to be equal."
  if (x == 1) {
    LOG(FATAL) << "InternalError: cannot reach here";
  }
}
```

上述函数作为 PackedFunc 注册到 Python 前端，名称为 `tvm._api_internal._ErrorTest`。如果我们调用注册函数会发生以下情况：

``` 
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

如上面的示例所示，TVM 的 ffi 系统将 Python 和 C++ 的 StackTrace 合并为一条消息，并自动生成相应的错误类。

## 如何选择错误类型

可以浏览下面列出的错误类型，试着用常识并参考已有代码中的选择来决定。我们尽量将错误类型保持在一个合理的数目。如果你觉得需要添加新的错误类型，请这样做：

-   在当前代码库中发送带有描述和使用示例的 RFC 提案。
-   使用清晰的文档将新的错误类型添加到 :py`tvm.error`{.interpreted-text
    role="mod"}。
-   将新的错误类型添加到此文件中的列表里。
-   在代码里使用新的错误类型。

创建简短的错误消息时推荐使用较少的抽象。代码以这种方式更具可读性，并且在必要时还打开了制作特定错误消息的路径。

``` python
def preferred():
    # 清楚知道抛出什么类型的错误以及错误消息是什么。
    raise OpNotImplemented("Operator relu is not implemented in the MXNet frontend")

def _op_not_implemented(op_name):
    return OpNotImplemented("Operator {} is not implemented.").format(op_name)

def not_preferred():
    # In引入另一个间接方法
    raise _op_not_implemented("relu")
```

如果需要引入构造多行错误消息的包装函数，请将包装器放在同一个文件中，以便其他开发者可以轻松找到。
