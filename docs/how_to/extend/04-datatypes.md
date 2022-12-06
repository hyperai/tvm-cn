---
title: 在 TVM 中使用 Bring Your Own Datatypes
---

# 在 TVM 中使用 Bring Your Own Datatypes

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/extend_tvm/bring_your_own_datatypes.html#sphx-glr-download-how-to-extend-tvm-bring-your-own-datatypes-py) 下载完整的示例代码
:::

**作者**：[Gus Smith](https://github.com/gussmith23), [Andrew Liu](https://github.com/hypercubestart)

本教程将展示如何利用 Bring Your Own Datatypes 框架在 TVM 中使用自定义数据类型。注意，Bring Your Own Datatypes 框架目前仅处理**数据类型的软件模拟版本**。该框架不支持开箱即用地编译自定义加速器数据类型。

## 数据类型库

Bring Your Own Datatypes 允许用户在 TVM 的原生数据类型（例如 `float`）旁边注册自己的数据类型实现。这些数据类型实现通常以库的形式出现。例如：

* [libposit](https://github.com/cjdelisle/libposit)，一个位置库
* [Stillwater Universal](https://github.com/stillwater-sc/universal)，一个包含位置、定点数和其他类型的库
* [SoftFloat](https://github.com/ucb-bar/berkeley-softfloat-3)，伯克利的 IEEE 754 浮点软件实现

Bring Your Own Datatypes 使用户能够将这些数据类型实现插入 TVM！

本节中我们将用到一个已经实现的示例库（位于 `3rdparty/byodt/myfloat.cc`）。这种称之为「myfloat」的数据类型实际上只是一个 IEE-754 浮点数，但它提供了一个有用的示例，表明任何数据类型都可以在 BYODT 框架中使用。

## 设置

由于不使用任何 3rdparty 库，因此无需设置。

若要用自己的数据类型库尝试，首先用 `CDLL` 把库的函数引入进程空间：

``` cmake
ctypes.CDLL('my-datatype-lib.so', ctypes.RTLD_GLOBAL)
```

## 一个简单的 TVM 程序

从在 TVM 中编写一个简单的程序开始，之后进行重写，从而使用自定义数据类型。

``` python
import tvm
from tvm import relay

# 基本程序：Z = X + Y
x = relay.var("x", shape=(3,), dtype="float32")
y = relay.var("y", shape=(3,), dtype="float32")
z = x + y
program = relay.Function([x, y], z)
module = tvm.IRModule.from_expr(program)
```

现使用 numpy 为程序创建随机输入：

``` python
import numpy as np

np.random.seed(23)  # 可重复性

x_input = np.random.rand(3).astype("float32")
y_input = np.random.rand(3).astype("float32")
print("x: {}".format(x_input))
print("y: {}".format(y_input))
```

输出结果：

``` bash
x: [0.51729786 0.9469626  0.7654598 ]
y: [0.28239584 0.22104536 0.6862221 ]
```

最后，准备运行程序：

``` python
z_output = relay.create_executor(mod=module).evaluate()(x_input, y_input)
print("z: {}".format(z_output))
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
z: [0.7996937 1.168008  1.4516819]
```

## 添加自定义数据类型

接下来使用自定义数据类型进行中间计算。

使用与上面相同的输入变量 `x` 和 `y`，但在添加 `x + y` 之前，首先通过调用 `relay.cast(...)` 将 `x` 和 `y` 转换为自定义数据类型。

注意如何指定自定义数据类型：使用特殊的 `custom[...]` 语法来表示。此外，注意数据类型后面的「32」：这是自定义数据类型的位宽，告诉 TVM `myfloat` 的每个实例都是 32 位宽。

``` python
try:
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        x_myfloat = relay.cast(x, dtype="custom[myfloat]32")
        y_myfloat = relay.cast(y, dtype="custom[myfloat]32")
        z_myfloat = x_myfloat + y_myfloat
        z = relay.cast(z_myfloat, dtype="float32")
except tvm.TVMError as e:
    # 打印最后一行错误
    print(str(e).split("\n")[-1])
```

尝试生成此程序会从 TVM 引发错误。 TVM 不知道如何创造性地处理所有自定义数据类型！因此首先要向 TVM 注册自定义类型，给它一个名称和一个类型代码：

``` python
tvm.target.datatype.register("myfloat", 150)
```

注意，类型代码 150 目前由用户手动选择。参阅 [include/tvm/runtime/c_runtime_api.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/data_type.h) 中的 `TVMTypeCode::kCustomBegin`。下面再次生成程序：

``` python
x_myfloat = relay.cast(x, dtype="custom[myfloat]32")
y_myfloat = relay.cast(y, dtype="custom[myfloat]32")
z_myfloat = x_myfloat + y_myfloat
z = relay.cast(z_myfloat, dtype="float32")
program = relay.Function([x, y], z)
module = tvm.IRModule.from_expr(program)
module = relay.transform.InferType()(module)
```

现在有了一个使用 myfloat 的 Relay 程序！

``` python
print(program)
```

输出结果：

``` bash
fn (%x: Tensor[(3), float32], %y: Tensor[(3), float32]) {
  %0 = cast(%x, dtype="custom[myfloat]32");
  %1 = cast(%y, dtype="custom[myfloat]32");
  %2 = add(%0, %1);
  cast(%2, dtype="float32")
}
```

现在可以准确无误地表达程序，尝试运行！

``` python
try:
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        z_output_myfloat = relay.create_executor("graph", mod=module).evaluate()(x_input, y_input)
        print("z: {}".format(y_myfloat))
except tvm.TVMError as e:
    # 打印最后一行错误
    print(str(e).split("\n")[-1])
```

输出结果：

``` bash
Check failed: (lower) is false: Cast lowering function for target llvm destination type 150 source type 2 not found
```

编译该程序会引发错误，下面来剖析这个报错。

该报错发生在代码降级的过程中，即将自定义数据类型代码，降级为 TVM 可以编译和运行的代码。TVM 显示，当从源类型 2（`float`，在 TVM 中）转换到目标类型 150（自定义数据类型）时，它无法找到 `Cast` 操作的*降级函数*。

当对自定义数据类型进行降级时，若 TVM 遇到对自定义数据类型的操作，它会查找用户注册的*降级函数*，这个函数告诉 TVM 如何将操作降级为 TVM 理解的数据类型的操作。由于我们还没有告诉 TVM 如何降级自定义数据类型的 `Cast` 操作，因此会报错。

要修复这个错误，只需要指定一个降级函数：

``` python
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func(
        {
            (32, 32): "FloatToCustom32",  # cast from float32 to myfloat32 # 从 float32 转换为 myfloat32
        }
    ),
    "Cast",
    "llvm",
    "float",
    "myfloat",
)
```

`register_op(...)` 调用接受一个降级函数和一些参数，这些参数准确地指定了应该使用提供的降级函数降级的操作。在这种情况下，传递的参数指定此降级函数用于将 target `“llvm”` 的 `Cast` 从 `float` 降级到 `myfloat`。

传递给此调用的降级函数非常通用：它应该采用指定类型的操作（在本例中为 *Cast*）并返回另一个仅使用 TVM 理解的数据类型的操作。

通常，我们希望用户借助对外部库的调用，来对其自定义数据类型进行操作。在示例中，`myfloat` 库在函数 `FloatToCustom32` 中实现了从 `float` 到 32 位 `myfloat` 的转换。一般情况下，创建一个辅助函数 `create_lower_func(...)`，它的作用是：给定一个字典，它将给定的 `Call`的操作，替换为基于操作和位宽的适当函数名称。它还通过将自定义数据类型存储在适当宽度的不透明 `uint` 中，从而删除自定义数据类型的使用；在我们的例子中，如 `uint32_t`。有关更多信息，参阅 [源代码](https://github.com/apache/tvm/blob/main/python/tvm/target/datatype.py)。

``` python
# 现在重新尝试运行程序：
try:
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        z_output_myfloat = relay.create_executor("graph", mod=module).evaluate()(x_input, y_input)
        print("z: {}".format(z_output_myfloat))
except tvm.TVMError as e:
    # 打印最后一行错误
    print(str(e).split("\n")[-1])
```

输出结果：

``` bash
Check failed: (lower) is false: Add lowering function for target llvm type 150 not found
```

新报错提示无法找到 `Add` 降级函数，这并不是坏事儿，这表明错误与 `Cast`无关！接下来只需要在程序中为其他操作注册降级函数。

注意，对于 `Add`，`create_lower_func` 接受一个键（key）是整数的字典。对于 `Cast` 操作，需要一个 2 元组来指定 `src_bit_length` 和 `dest_bit_length`，对于其他操作，操作数之间的位长度相同，因此只需要一个整数来指定 `bit_length`。

``` python
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Add"}),
    "Add",
    "llvm",
    "myfloat",
)
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({(32, 32): "Custom32ToFloat"}),
    "Cast",
    "llvm",
    "myfloat",
    "float",
)

# 现在，可以正常运行程序了。
with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
    z_output_myfloat = relay.create_executor(mod=module).evaluate()(x_input, y_input)
print("z: {}".format(z_output_myfloat))

print("x:\t\t{}".format(x_input))
print("y:\t\t{}".format(y_input))
print("z (float32):\t{}".format(z_output))
print("z (myfloat32):\t{}".format(z_output_myfloat))

# 或许正如预期的那样，``myfloat32`` 结果和 ``float32`` 是完全一样的！
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
z: [0.7996937 1.168008  1.4516819]
x:              [0.51729786 0.9469626  0.7654598 ]
y:              [0.28239584 0.22104536 0.6862221 ]
z (float32):    [0.7996937 1.168008  1.4516819]
z (myfloat32):  [0.7996937 1.168008  1.4516819]
```

## 使用自定义数据类型运行模型

首先选择要使用 myfloat 运行的模型，本示例中，我们使用的是 [Mobilenet](https://arxiv.org/abs/1704.04861)。选择 Mobilenet 是因为它足够小。在 Bring Your Own Datatypes 框架的这个 alpha 状态下，还没有为运行自定义数据类型的软件仿真实现任何软件优化；由于多次调用数据类型仿真库，导致性能不佳。

首先定义两个辅助函数，获取 mobilenet 模型和猫图像。

``` python
def get_mobilenet():
    dshape = (1, 3, 224, 224)
    from mxnet.gluon.model_zoo.vision import get_model

    block = get_model("mobilenet0.25", pretrained=True)
    shape_dict = {"data": dshape}
    return relay.frontend.from_mxnet(block, shape_dict)

def get_cat_image():
    from tvm.contrib.download import download_testdata
    from PIL import Image

    url = "https://gist.githubusercontent.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/fa7ef0e9c9a5daea686d6473a62aacd1a5885849/cat.png"
    dst = "cat.png"
    real_dst = download_testdata(url, dst, module="data")
    img = Image.open(real_dst).resize((224, 224))
    # CoreML's standard model image format is BGR
    img_bgr = np.array(img)[:, :, ::-1]
    img = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, :]
    return np.asarray(img, dtype="float32")

module, params = get_mobilenet()
```

输出结果：

``` bash
Downloading /workspace/.mxnet/models/mobilenet0.25-9f83e440.zipe0e3327d-26bc-4c47-aed4-734a16b0a3f8 from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/mobilenet0.25-9f83e440.zip...
```

用原生 TVM 很容易执行 MobileNet：

``` python
ex = tvm.relay.create_executor("graph", mod=module, params=params)
input = get_cat_image()
result = ex.evaluate()(input).numpy()
# 打印前 10 个元素
print(result.flatten()[:10])
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
[ -7.5350165   2.0368009 -12.706646   -5.63786   -12.684058    4.0723605
   2.618876    3.4049501  -9.867913  -24.53311  ]
```

若要更改模型在内部使用 myfloat，需要转换网络。为此首先定义一个函数来帮助转换张量：

``` python
def convert_ndarray(dst_dtype, array):
    """Converts an NDArray into the specified datatype"""
    x = relay.var("x", shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        return relay.create_executor("graph").evaluate(cast)(array)
```

为了实际转换整个网络，我们在 Relay 中编写了 [一个 pass](https://github.com/gussmith23/tvm/blob/ea174c01c54a2529e19ca71e125f5884e728da6e/python/tvm/relay/frontend/change_datatype.py#L21)，它简单地将模型中的所有节点转换为使用新的数据类型。

``` python
from tvm.relay.frontend.change_datatype import ChangeDatatype

src_dtype = "float32"
dst_dtype = "custom[myfloat]32"

module = relay.transform.InferType()(module)

# 目前，自定义数据类型仅在预先运行 simple_inference 时才有效
module = tvm.relay.transform.SimplifyInference()(module)

# 在更改数据类型之前运行类型推断
module = tvm.relay.transform.InferType()(module)

# 将数据类型从 float 更改为 myfloat 并重新推断类型
cdtype = ChangeDatatype(src_dtype, dst_dtype)
expr = cdtype.visit(module["main"])
module = tvm.relay.transform.InferType()(module)

# 转换参数：
params = {k: convert_ndarray(dst_dtype, v) for k, v in params.items()}

# 还需要转换输入：
input = convert_ndarray(dst_dtype, input)

# 最后，可以尝试运行转换后的模型：
try:
    # 向量化不是用自定义数据类型实现的。
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        result_myfloat = tvm.relay.create_executor("graph", mod=module).evaluate(expr)(
            input, **params
        )
except tvm.TVMError as e:
    print(str(e).split("\n")[-1])
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
  Check failed: (lower) is false: Intrinsic lowering function for target llvm, intrinsic name tir.sqrt, type 150 not found
```

尝试运行模型时，会收到一个熟悉的报错，提示需要为 myfloat 注册更多函数。

因为这是一个神经网络，所以需要更多的操作。下面注册所有需要的函数：

``` python
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "FloatToCustom32"}),
    "FloatImm",
    "llvm",
    "myfloat",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.lower_ite, "Call", "llvm", "myfloat", intrinsic_name="tir.if_then_else"
)

tvm.target.datatype.register_op(
    tvm.target.datatype.lower_call_pure_extern,
    "Call",
    "llvm",
    "myfloat",
    intrinsic_name="tir.call_pure_extern",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Mul"}),
    "Mul",
    "llvm",
    "myfloat",
)
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Div"}),
    "Div",
    "llvm",
    "myfloat",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Sqrt"}),
    "Call",
    "llvm",
    "myfloat",
    intrinsic_name="tir.sqrt",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Sub"}),
    "Sub",
    "llvm",
    "myfloat",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Exp"}),
    "Call",
    "llvm",
    "myfloat",
    intrinsic_name="tir.exp",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Max"}),
    "Max",
    "llvm",
    "myfloat",
)

tvm.target.datatype.register_min_func(
    tvm.target.datatype.create_min_lower_func({32: "MinCustom32"}, "myfloat"),
    "myfloat",
)
```

注意，我们使用的是：`register_min_func` 和 `create_min_lower_func`。

`register_min_func` 接收一个整数 `num_bits` 作为位长，然后返回一个表示最小有限可表示值的操作，这个值是具有指定位长的自定义数据类型。

与 `register_op` 和 `create_lower_func` 类似，`create_min_lower_func` 处理通过调用一个外部库，实现最小可表示的自定义数据类型值的一般情况。

接下来运行模型：

``` python
# 向量化不是用自定义数据类型实现的。
with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
    result_myfloat = relay.create_executor(mod=module).evaluate(expr)(input, **params)
    result_myfloat = convert_ndarray(src_dtype, result_myfloat).numpy()
    # 打印前 10 个元素
    print(result_myfloat.flatten()[:10])

# 再次注意，使用 32 位 myfloat 的输出与 32 位浮点数完全相同，
# 因为 myfloat 就是一个浮点数！
np.testing.assert_array_equal(result, result_myfloat)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
[ -7.5350165   2.0368009 -12.706646   -5.63786   -12.684058    4.0723605
   2.618876    3.4049501  -9.867913  -24.53311  ]
```

[下载 Python 源代码：bring_your_own_datatypes.py](https://tvm.apache.org/docs/_downloads/ee99205e9f2e4f54c0fb7925008a5354/bring_your_own_datatypes.py)

[下载 Jupyter Notebook：bring_your_own_datatypes.ipynb](https://tvm.apache.org/docs/_downloads/b11795df0596a55e4982bf895d0c8c38/bring_your_own_datatypes.ipynb)
