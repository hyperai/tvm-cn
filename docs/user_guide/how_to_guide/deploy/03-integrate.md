# 将 TVM 集成到项目中

TVM runtime 具有轻量级和可移植性的特点，有几种方法可将 TVM 集成到项目中。

本文介绍如何将 TVM 作为 JIT 编译器集成到项目中，从而用它在系统上生成函数的方法

## DLPack 支持

TVM 的生成函数遵循 PackedFunc 约定，它是一个可以接受位置参数（包括标准类型，如浮点、整数、字符串）的函数。PackedFunc 采用 [DLPack](https://github.com/dmlc/dlpack) 约定中的 DLTensor 指针。唯一要做的是创建一个对应的 DLTensor 对象。

## 集成用户自定义的 C++ 数组

在 C++ 中唯一要做的就是将你的数组转换为 DLTensor，并将其地址作为 `DLTensor*` 传递给生成的函数。

## 集成用户自定义的 Python 数组

针对 Python 对象 `MyArray`，需要做：

* 将 `_tvm_tcode` 字段添加到返回 `tvm.TypeCode.ARRAY_HANDLE` 的数组中
* 在对象中支持 `_tvm_handle` 属性（以 Python 整数形式返回 DLTensor 的地址）
* 用 `tvm.register_extension` 注册这个类

``` python
# 示例代码
import tvm

class MyArray(object):
    _tvm_tcode = tvm.TypeCode.ARRAY_HANDLE

    @property
    def _tvm_handle(self):
        dltensor_addr = self.get_dltensor_addr()
        return dltensor_addr

# 将注册的步骤放在单独的文件 mypkg.tvm.py 中
# 根据需要选择性地导入依赖
tvm.register_extension(MyArray)
```