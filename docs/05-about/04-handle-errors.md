---

title: 处理 TVM 错误

---

运行 TVM 时，你可能会遇到如下错误信息：

```plain
---------------------------------------------------------------
An error occurred during the execution of TVM.
For more information, please see: https://tvm.apache.org/docs/errors.html
---------------------------------------------------------------
```

下面提供了一些解读这些错误信息的提示，以及遇到错误时可以采取的措施。

## 这些错误从何而来？

此错误是 TVM 执行过程中内部不变量被违反时产生的。从技术角度来说，该消息由 `TVM_FFI_ICHECK` 宏生成，该宏位于 [TVM-FFI](https://github.com/tlc-pack/tvm-ffi) 的 `include/tvm/ffi/error.h` 中。`TVM_FFI_ICHECK` 宏在 TVM 代码中的很多地方用于断言某个条件在执行期间为真；当断言失败时，TVM 将退出并显示上述错误信息。

有关 TVM 如何处理和生成错误的更多详细信息，请参阅[错误处理指南](/docs/how-to/development-guides/error-handling-guide)。

## 遇到错误时应该怎么做？

首先，*不要慌张*。慌张也无济于事。

最好的做法是在 [Apache TVM 讨论论坛](https://discuss.tvm.apache.org/) 搜索你遇到的错误，看看是否有其他人遇到过同样的问题，以及解决方案是什么。如果此错误是已在较新版本的 TVM 中修复的 bug，你可能需要更新到新版本。

如果你没有找到关于该问题的已有论坛帖子，欢迎在论坛上开一个新帖子并附上问题详情。**请**在帖子中包含以下关键信息：

* 你使用的 TVM 版本（例如，源码树的 git commit hash）。
* 你运行 TVM 的硬件和操作系统版本。
* 你 TVM 编译目标的硬件设备和操作系统。
* 模型、输入或其他工作负载的详细信息，以便复现问题。

没有这些细节，TVM 开发者很难为你提供帮助。
