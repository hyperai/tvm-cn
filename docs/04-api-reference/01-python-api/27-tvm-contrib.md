---

title: tvm.contrib

---


TVM python 包的 Contrib API。


Contrib API 提供了许多实用的非核心函数。其中一些是与第三方库和工具交互的实用工具。



# tvm.contrib.cblas



BLAS 库的外部函数接口。

## tvm.contrib.cblas.matmul(*lhs*, *rhs*, *transa=False*, *transb=False*, ***kwargs*)


创建一个外部操作，使用 CrhsLAS 计算 A 和 rhs 的矩阵乘法，此函数作为如何调用外部库的示例。
* **参数：**
   * **lhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：左矩阵操作数。
   * **rhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：右矩阵操作数。
   * **transa** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置 lhs。
   * **transb** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置 rhs。
* **返回：C**：结果张量。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。

## tvm.contrib.cblas.batch_matmul(*lhs*, *rhs*, *transa=False*, *transb=False*, *iterative=False*, ***kwargs*)


创建一个外部操作，使用 CBLAS 计算 A 和 rhs 的分批矩阵乘法。此函数作为如何调用外部库的示例。
* **参数：**
   * **lhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：左矩阵操作数。
   * **rhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：右矩阵操作数。
   * **transa** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置 lhs。
   * **transb** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置 rhs。
* **返回：C**：结果张量。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。





# tvm.contrib.clong




用于在系统中调用 clang 的实用程序。

## tvm.contrib.clang.find_clang(*required=True*)


在系统中找到 clang。
* **参数：required** ([bool](https://docs.python.org/3/library/functions.html#bool))**：** 是否需要，如果需要编译器，则会引发运行时错误。
* **返回：valid_list**：可能路径的列表。
* **返回类型：**[list](https://docs.python.org/3/library/stdtypes.html#list) of [str](https://docs.python.org/3/library/stdtypes.html#str).

:::Note

此函数将首先搜索与使用 tvm 构建的主要 llvm 版本匹配的 clang。

:::

## tvm.contrib.clang.create_llvm(*inputs*, *output=None*, *options=None*, *cc=None*)


创建 llvm 文本 ir。
* **参数：**
   * **inputs** ([list](https://docs.python.org/3/library/stdtypes.html#list)*of*[str](https://docs.python.org/3/library/stdtypes.html#str))*：* 输入文件名称或代码源的列表。
   * **output** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：输出文件，如果没有，则创建临时文件。
   * **options** ([list](https://docs.python.org/3/library/stdtypes.html#list))：附加选项字符串的列表。
   * **cc** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：clang 编译器，如果未指定，我们将尝试猜测匹配的 clang 版本。
* **返回：code**：生成的 llvm 文本 IR。
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)。





# tvm.contrib.cc




用于调用系统中的 C/C++ 编译器。

## tvm.contrib.cc.get_cc()


返回默认 C/C++ 编译器的路径。
* **返回：out**：默认 C/C++ 编译器的路径，如果未找到则为 None。
* **返回类型：** Optional[[str](https://docs.python.org/3/library/stdtypes.html#str)]。

## tvm.contrib.cc.create_shared(*output*, *objects*, *options=None*, *cc=None*, *cwd=None*, *ccache_env=None*)


创建共享库。
* **参数：**
   * **output** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 目标共享库。
   * **objects** (*List[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)*：* 目标文件列表。
   * **options** (*List[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：附加选项字符串列表。
   * **cc** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)*：* 编译器命令。
   * **cwd** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：当前工作目录。
   * **ccache_env** (*Optional**[****Dict**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,*** [str](https://docs.python.org/3/library/stdtypes.html#str)***]****]*)：ccache 的环境变量。设置为 None 则默认禁用 ccache。

## tvm.contrib.cc.create_staticlib(*output*, *inputs*, *ar=None*)

创建静态库。
* **参数：**
   * **output** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 目标共享库。
   * **inputs** (*List[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：输入文件列表。每个输入文件可以是对象的 tarball 或对象文件。
   * **ar** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：要使用的 ar 命令的路径。

## tvm.contrib.cc.create_executable(*output*, *objects*, *options=None*, *cc=None*, *cwd=None*, *ccache_env=None*)


创建可执行二进制文件。
* **参数：**
   * **output** ([str](https://docs.python.org/3/library/stdtypes.html#str))：目标可执行文件。
   * **objects** (*List[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)*：* 目标文件列表。
   * **options** (*List[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)*：* 附加选项字符串列表。
   * **cc** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：编译器命令。
   * **cwd** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：当前工作目录。
   * **ccache_env** (*Optional*[****Dict**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,** [str](https://docs.python.org/3/library/stdtypes.html#str)***]****]*)：ccache 的环境变量。设置为 None 则默认禁用 ccache。

## tvm.contrib.cc.get_global_symbol_section_map(*path*, *, *nm=None*) → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]


通过 nm -g 从库中获取全局符号。
* **参数：**
   * **path** ([str](https://docs.python.org/3/library/stdtypes.html#str))：库路径。
   * **nm** ([str](https://docs.python.org/3/library/stdtypes.html#str))：nm 命令的路径。
* **返回：symbol_section_map**：从定义的全局符号到其部分的映射。
* **返回类型：** Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]。

## tvm.contrib.cc.get_target_by_dump_machine(*compiler*)


get_target_triple 的函子，可以使用编译器获取目标三元组。
* **参数：compiler** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：编译器。
* **返回：out** *：* 根据编译器的 dumpmachine 选项获取目标三元组的函数。
* **返回类型：** Callable。

## tvm.contrib.cc.cross_compiler(*compile_func*, *options=None*, *output_format=None*, *get_target_triple=None*, *add_files=None*)


通过使用选项专门化 compile_func 来创建交叉编译器函数。


此函数可用于构建可传递给 AutoTVM 测量或 export_library 的编译函数。
* **参数：**
   * **compile_func** (*Union*[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***Callable****[**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,** [str](https://docs.python.org/3/library/stdtypes.html#str)***,*** ***Optional***[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**],None**]*]*)：执行实际编译的函数。
   * **options** (*Optional**[****List**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***]****]*)：附加可选字符串列表。
   * **output_format** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：库输出格式。
   * **get_target_triple** (*Optional*[*Callable]*)：根据编译器的 dumpmachine 选项，可以定位三重函数。
   * **add_files** (*Optional**[****List**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***]****]*)：作为编译的一部分传递的附加对象、源、库文件的路径列表。
* **返回：fcompile**：可以传递给 export_library 的编译函数。
* **返回类型：** Callable[[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), Optional[[str](https://docs.python.org/3/library/stdtypes.html#str)]], None]。


**示例**

```python
from tvm.contrib import cc, ndk
# 使用 ARM GCC 导出
mod = build_runtime_module()
mod.export_library(path_dso,
                   fcompile=cc.cross_compiler("arm-linux-gnueabihf-gcc"))
# 针对 NDK 编译选项进行特化。
specialized_ndk = cc.cross_compiler(
    ndk.create_shared,
    ["--sysroot=/path/to/sysroot", "-shared", "-fPIC", "-lm"])
mod.export_library(path_dso, fcompile=specialized_ndk)
```

# tvm.contrib.cublas

cuBLAS 库的外部函数接口。

## tvm.contrib.cublas.matmul(*lhs*, *rhs*, *transa=False*, *transb=False*, *dtype=None*)


创建一个外部操作，使用 cuBLAS 计算矩阵 A 和 rhs 的乘数。
* **参数：**
   * **lhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：左矩阵操作数。
   * **rhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))*：* 右矩阵操作数。
   * **transa** ([bool](https://docs.python.org/3/library/functions.html#bool))*：* 是否转置 lhs。
   * **transb** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置 rhs。
* **返回：C**：结果张量。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。

## tvm.contrib.cublas.batch_matmul(*lhs*, *rhs*, *transa=False*, *transb=False*, *dtype=None*)

创建一个外部操作，使用 cuBLAS 计算批量矩阵 A 和 rhs 的乘数。
* **参数：**
   * **lhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：左矩阵操作数。
   * **rhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：右矩阵操作数。
   * **transa** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置 lhs。
   * **transb** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置 rhs。
* **返回：C**：结果张量。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。



# tvm.contrib.dlpack



包装函数以桥接框架，并通过 DLPack 支持 TVM。

## tvm.contrib.dlpack.convert_func(*tvm_func*, *tensor_type*, *to_dlpack_func*)


将一个 TVM 函数转换为能够接受来自其他框架的张量的函数，前提是该框架支持 DLPACK。
* **参数：**
   * **tvm_func** ([Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function))：构建对数组进行操作的 tvm 函数。
   * **tensor_type** ([Type](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.Type))：目标框架的张量类型。
   * **to_dlpack_func** ([Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function))：将源张量转换为 DLPACK 的函数。

## tvm.contrib.dlpack.to_pytorch_func(*tvm_func*)

将 tvm 函数转换为接受 PyTorch 张量的函数。
* **参数：tvm_func** ([Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function))：构建对数组进行操作的 tvm 函数。
* **返回：wrapped_func**：对 PyTorch 张量进行操作的包装 tvm 函数。
* **返回类型：**[Function](https://tvm.apache.org/docs/reference/api/python/relax/relax.html#tvm.relax.Function)。


# tvm.contrib.emcc



用于调用系统中的 emscripten 编译器。

## tvm.contrib.emcc.create_tvmjs_wasm(*output*, *objects*, *options=None*, *cc='emcc'*, *libs=None*)


创建应该与 tvmjs 一起运行的 wasm。
* **参数：**
   * **output** ([str](https://docs.python.org/3/library/stdtypes.html#str))：目标共享库。
   * **objects** ([list](https://docs.python.org/3/library/stdtypes.html#list))：目标文件列表。
   * **options** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 附加选项。
   * **cc** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)*：* 编译字符串。
   * **libs** ([list](https://docs.python.org/3/library/stdtypes.html#list))*：* 要添加到 wasm 中的用户定义库文件列表（例如 .bc 文件）。

# tvm.contrib.miopen


MIOpen 库的外部函数接口。

## tvm.contrib.miopen.conv2d_forward(*x*, *w*, *stride_h=1*, *stride_w=1*, *pad_h=0*, *pad_w=0*, *dilation_h=1*, *dilation_w=1*, *conv_mode=0*, *data_type=1*, *group_count=1*)


创建一个使用 MIOpen 计算 2D 卷积的外部操作。
* **参数：**
   * **x** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：输入特征图。
   * **w** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：卷积权重。
   * **stride_h** ([int](https://docs.python.org/3/library/functions.html#int))：身高步幅。
   * **stride_w** ([int](https://docs.python.org/3/library/functions.html#int))：宽度步幅。
   * **pad_h** ([int](https://docs.python.org/3/library/functions.html#int))**：** 高度垫。
   * **pad_w** ([int](https://docs.python.org/3/library/functions.html#int))：权重垫。
   * **dilation_h** ([int](https://docs.python.org/3/library/functions.html#int))：高度扩张。
   * **dilation_w** ([int](https://docs.python.org/3/library/functions.html#int))：宽度扩张。
   * **conv_mode** ([int](https://docs.python.org/3/library/functions.html#int))**：** 0：miopenConvolution 1：miopenTranspose。
   * **data_type** ([int](https://docs.python.org/3/library/functions.html#int))**：** 0：miopenHalf（fp16）1：miopenFloat（fp32）。
   * **group_count** ([int](https://docs.python.org/3/library/functions.html#int))：组数。
* **返回：y**：结果张量。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。

## tvm.contrib.miopen.softmax(*x*, *axis=-1*)

使用 MIOpen 计算 softmax。
* **参数：**
   * **x** ([tvm.te.Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：输入张量。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int))：计算 softmax 的轴。
* **返回：ret**：结果张量。
* **返回类型：**[tvm.te.Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。

## tvm.contrib.miopen.log_softmax(*x*, *axis=-1*)

使用 MIOpen 计算对数 softmax。
* **参数：**
   * **x** ([tvm.te.Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：输入张量。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int))*：* 计算 log softmax 的轴。
* **返回：ret**：结果张量。
* **返回类型：**[tvm.te.Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。




# tvm.contrib.ndk


用于调用 NDK 编译器工具链。

## tvm.contrib.ndk.create_shared(*output*, *objects*, *options=None*)


创建共享库。
* **参数：**
   * **output** ([str](https://docs.python.org/3/library/stdtypes.html#str))：目标共享库。
   * **objects** ([list](https://docs.python.org/3/library/stdtypes.html#list))：目标文件列表。
   * **options** ([list](https://docs.python.org/3/library/stdtypes.html#list)*of*[str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：附加选项 *。*

## tvm.contrib.ndk.create_staticlib(*output*, *inputs*)


创建静态库：
* **参数：**
   * **output** ([str](https://docs.python.org/3/library/stdtypes.html#str))：目标静态库。
   * **inputs** ([list](https://docs.python.org/3/library/stdtypes.html#list))：目标文件或 tar 文件列表。

## tvm.contrib.ndk.get_global_symbol_section_map(*path*, ***, *nm=None*) → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]


通过 NDK 中的 nm -gU 从库中获取全局符号。
* **参数：**
   * **path** ([str](https://docs.python.org/3/library/stdtypes.html#str))：库路径。
   * **nm** ([str](https://docs.python.org/3/library/stdtypes.html#str))：nm 命令的路径。
* **返回：symbol_section_map**：从定义的全局符号到其部分的映射。
* **返回类型：** Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]。



# tvm.contrib.nnpack



NNPACK 库的外部函数接口。

## tvm.contrib.nnpack.is_available()


检查 NNPACK 是否可用，即 nnp_initialize() 返回 nnp_status_success。

## tvm.contrib.nnpack.fully_connected_inference(*lhs*, *rhs*, *nthreads=1*)


创建一个外部操作，使用 nnpack 计算 1D 张量 lhs 和 2D 张量 rhs 的完全连接。
* **参数：**
   * **lhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：FP32 元素的 lhs 1D 数组输入[input_channels]。
   * **rhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：FP32 元素的 lhs 二维矩阵核[输出通道][输入通道]。
* **返回：C**：FP32 元素的 lhs 1D 数组 out[output_channels]。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。

## tvm.contrib.nnpack.convolution_inference(*data*, *kernel*, *bias*, *padding*, *stride*, *nthreads=1*, *algorithm=0*)


创建一个外部操作，使用 nnpack 对 4D 张量数据和 4D 张量核以及 1D 张量偏差进行推理卷积。
* **参数：**
   * **data** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：FP32 元素的数据 4D 张量输入[batch][input_channels][input_height][input_width]。
   * **kernel** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))**：** FP32 元素的内核 4D 张量 kernel[output_channels][input_channels][kernel_height] [kernel_width]。
   * **bias** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：FP32 元素的偏差 1D 数组偏差[output_channels][input_channels][kernel_height] [kernel_width]。
   * **padding** ([list](https://docs.python.org/3/library/stdtypes.html#list))：padding 一个 4 维列表，包含 [pad_top、pad_bottom、pad_left、pad_right]，表示特征图周围的填充。
   * **stride** ([list](https://docs.python.org/3/library/stdtypes.html#list))：步幅 [stride_height，stride_width] 的 2 维列表，表示步幅。
* **返回：output**：FP32 元素的输出 4D 张量输出[batch][output_channels][output_height][output_width]。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。

## tvm.contrib.nnpack.convolution_inference_without_weight_transform(*data*, *transformed_kernel*, *bias*, *padding*, *stride*, *nthreads=1*, *algorithm=0*) 

创建一个外部操作，使用 nnpack 对 4D 张量数据和 4D 预转换张量核和 1D 张量偏差进行推理卷积。
* **参数：**
   * **data** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))**：** FP32 元素的数据 4D 张量输入[batch][input_channels][input_height][input_width]。
   * **transformed_kernel** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：transformed_kernel 4D 张量 kernel[output_channels][input_channels][tile] [tile] 由 FP32 元素组成。
   * **bias** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：FP32 元素的偏差 1D 数组偏差[output_channels][input_channels][kernel_height] [kernel_width]。
   * **padding** ([list](https://docs.python.org/3/library/stdtypes.html#list))*：* padding 一个 4 维列表，包含 [pad_top、pad_bottom、pad_left、pad_right]，表示特征图周围的填充。
   * **stride** ([list](https://docs.python.org/3/library/stdtypes.html#list))：步幅 [stride_height，stride_width] 的 2 维列表，表示步幅。
* **返回：output**：FP32 元素的输出 4D 张量输出[batch][output_channels][output_height][output_width]。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。

## tvm.contrib.nnpack.convolution_inference_weight_transform(*kernel*, *nthreads=1*, *algorithm=0*, *dtype='float32'*)


创建一个外部操作，使用 nnpack 对 3D 张量数据和 4D 张量核以及 1D 张量偏差进行推理卷积。
* **参数：kernel** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))*：* FP32 元素的内核 4D 张量 kernel[output_channels][input_channels][kernel_height] [kernel_width]。
* **返回：output**：FP32 元素的输出 4D 张量输出[output_channels][input_channels][tile][tile]。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。





# tvm.contrib.nvcc


在系统中调用 nvcc 编译器的实用程序。

## tvm.contrib.nvcc.compile_cuda(*code*, *target_format=None*, *arch=None*, *options=None*, *path_target=None*)


使用 NVCC 从 env 编译 cuda 代码。
* **参数：**
   * **code** ([str](https://docs.python.org/3/library/stdtypes.html#str))：cuda 代码。
   * **target_format** ([str](https://docs.python.org/3/library/stdtypes.html#str))：nvcc 编译器的目标格式。
   * **arch** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：*  cuda 架构。
   * **options** ([str](https://docs.python.org/3/library/stdtypes.html#str)*or*[list](https://docs.python.org/3/library/stdtypes.html#list)*of*[str](https://docs.python.org/3/library/stdtypes.html#str))*：(*[https://docs.python.org/3/library/stdtypes.html#list](https://docs.python.org/3/library/stdtypes.html#list)*"（在 Python v3.13 中）")*[附加*](https://docs.python.org/3/library/stdtypes.html#str)选项。
   * **path_target** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：输出文件。
* **返回：cubin**：cubin 的字节数组。
* **返回类型：**[bytearray](https://docs.python.org/3/library/stdtypes.html#bytearray)。

## tvm.contrib.nvcc.find_cuda_path()


查找 cuda 路径的实用函数
* **返回：path**：cuda 根的路径。
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)。

## tvm.contrib.nvcc.get_cuda_version(*cuda_path=None*)


获取 cuda 版本的实用函数
* **参数：cuda_path** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：cuda 根路径。如果未指定，则 默认使用 find_cuda_path() 。
* **返回：version**：cuda 版本。
* **返回类型：**[float](https://docs.python.org/3/library/functions.html#float)。

## tvm.contrib.nvcc.find_nvshmem_paths() → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]


搜索 NVSHMEM 包含和库目录。:returns: 包含包含目录和库目录路径的元组。
* **返回类型：** 一个元组，包含头文件目录路径和库目录路径。

## tvm.contrib.nvcc.parse_compute_version(*compute_version*)


解析计算能力字符串以划分主版本和次版本。
* **参数：**
   * **compute_version** ([str](https://docs.python.org/3/library/stdtypes.html#str))：GPU 的计算能力（例如“6.0”）。
* **返回：** 
   * **major** (*int*)：主版本号。
   * **minor** (*int*)*：* 次要版本号。

## tvm.contrib.nvcc.have_fp16(*compute_version*)

计算功能中是否提供 fp16 支持。
* 参数:**compute_version** ([str](https://docs.python.org/3/library/stdtypes.html#str))：GPU 的计算能力（例如“6.0”）。

## tvm.contrib.nvcc.have_int8(*compute_version*)

计算功能中是否提供 int8 支持。
* 参数:**compute_version** ([str](https://docs.python.org/3/library/stdtypes.html#str))：GPU 的计算能力（例如“6.1”）。

## tvm.contrib.nvcc.have_tensorcore(*compute_version=None*, *target=None*)

计算功能中是否提供 TensorCore 支持。
* 参数：
   * **compute_version** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：GPU 的计算能力（例如“7.0”）。
   * **target** ([tvm.target.Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)*,optional*)*：*编译目标，如果未指定 compute_version，则将用于确定 arch。

## tvm.contrib.nvcc.have_cudagraph()

提供 CUDA Graph 支持。


# tvm.contrib.pickle_memoize


通过 pickle 记忆函数的结果，用于缓存测试用例。

## *class* tvm.contrib.pickle_memoize.Cache(*key*, *save_at_exit*)


用于结果缓存的缓存对象。
* **参数：**
   * **key** ([str](https://docs.python.org/3/library/stdtypes.html#str))：函数的文件键。
   * **save_at_exit** ([bool](https://docs.python.org/3/library/functions.html#bool))：程序退出时是否保存缓存到文件。

### *property* cache


返回缓存，首次使用时初始化。

## tvm.contrib.pickle_memoize.memoize(*key*, *save_at_exit=False*)


记住函数的结果并多次重复使用。
* **参数：**
   * **key** ([str](https://docs.python.org/3/library/stdtypes.html#str))：文件的唯一键。
   * **save_at_exit** ([bool](https://docs.python.org/3/library/functions.html#bool))：程序退出时是否保存缓存到文件。
* **返回：fmemoize**：执行记忆的装饰函数。
* **返回类型：** function。


# tvm.contrib.random

随机库的外部函数接口。

## tvm.contrib.random.randint(*low*, *high*, *size*, *dtype='int32'*)


返回从低（含）到高（不含）的随机整数。返回指定数据类型在「半开」区间 [low, high] 内服从「离散均匀」分布的随机整数。
* **参数：**
   * **low** ([int](https://docs.python.org/3/library/functions.html#int))：从分布中抽取的最低（有符号）整数。
   * **high** ([int](https://docs.python.org/3/library/functions.html#int))：从分布中抽取的最大（有符号）整数的上一个。
* **返回：out** *：* 具有指定大小和 dtype 的张量。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。

## tvm.contrib.random.uniform(*low*, *high*, *size*)


从均匀分布中抽取样本。


样本在半开区间 [低, 高) 内均匀分布（包含低，但不包括高）。换句话说，给定区间内的任何值都有同等的可能性被均匀分布。
* **参数：**
   * **low** ([float](https://docs.python.org/3/library/functions.html#float))：输出间隔的下边界。所有生成的值都将大于或等于 low。
   * **high** ([float](https://docs.python.org/3/library/functions.html#float))：输出间隔的上限。所有生成的值都将小于 high。
   * **size** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*ofints*)：输出形状。如果给定形状为 (m, n, k)，则绘制 m * n * k 个样本。
* **返回：out**：具有指定大小和 dtype 的张量。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。

## tvm.contrib.random.normal(*loc*, *scale*, *size*)


从正态分布中抽取样本。


从正态分布中返回随机样本。
* **参数：**
   * **loc** ([float](https://docs.python.org/3/library/functions.html#float))**：** 分布的 loc。
   * **scale** ([float](https://docs.python.org/3/library/functions.html#float))：分布的标准差。
   * **size** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*ofints*)：输出形状。如果给定形状为 (m, n, k)，则绘制 m * n * k 个样本。
* **返回：out**：具有指定大小和 dtype 的张量。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。



# tvm.contrib.rocblas


rocBLAS 库的外部函数接口。

## tvm.contrib.rocblas.matmul(*lhs*, *rhs*, *transa=False*, *transb=False*)


创建一个外部操作，使用 rocBLAS 计算 A 和 rhs 的矩阵乘法。
* **参数：**
   * **lhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：左矩阵操作数。
   * **rhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：右矩阵操作数。
   * **transa** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置 lhs。
   * **transb** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置 rhs。
* **返回：C**：结果张量。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。

## tvm.contrib.rocblas.batch_matmul(*lhs*, *rhs*, *transa=False*, *transb=False*)


创建一个外部操作，使用 rocBLAS 计算 A 和 rhs 的矩阵乘法。
* **参数：**
   * **lhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：左批处理矩阵操作数。
   * **rhs** ([Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor))：右批处理矩阵操作数。
   * **transa** ([bool](https://docs.python.org/3/library/functions.html#bool))**：** 是否转置 lhs。
   * **transb** ([bool](https://docs.python.org/3/library/functions.html#bool))：是否转置 rhs。
* **返回：C**：结果张量。
* **返回类型：**[Tensor](https://tvm.apache.org/docs/reference/api/python/te.html#tvm.te.Tensor)。


# tvm.contrib.rocm


ROCm 后端实用程序。

## tvm.contrib.rocm.find_lld(*required=True*)


在系统中查找 ld.lld。
* **参数：required** ([bool](https://docs.python.org/3/library/functions.html#bool))**：** 是否需要，如果需要编译器，则会引发运行时错误。
* **返回：valid_list**：可能路径的列表。
* **返回类型：**[list](https://docs.python.org/3/library/stdtypes.html#list) of [str](https://docs.python.org/3/library/stdtypes.html#str)。

:::Note

此函数将首先搜索与使用 tvm 构建的主要 llvm 版本匹配的 ld.lld

:::

## tvm.contrib.rocm.rocm_link(*in_file*, *out_file*, *lld=None*)


使用 lld 将可重定位 ELF 对象链接到共享 ELF 对象。
* **参数：**
   * **in_file** ([str](https://docs.python.org/3/library/stdtypes.html#str))**：** 输入文件名（可重定位 ELF 目标文件）。
   * **out_file** ([str](https://docs.python.org/3/library/stdtypes.html#str))**：** 输出文件名（共享 ELF 目标文件）。
   * **lld** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：lld 链接器，如果未指定，我们将尝试猜测匹配的 clang 版本。

## tvm.contrib.rocm.parse_compute_version(*compute_version*)


解析计算能力字符串以划分主版本和次版本。
* **参数：compute_version** ([str](https://docs.python.org/3/library/stdtypes.html#str))：GPU 的计算能力（例如“6.0”）。
* **返回：** 
   * **major** (*int*)：主版本号。
   * **minor** (*int*)[：](https://docs.python.org/3/library/stdtypes.html#bytearray)次要版本号。

## tvm.contrib.rocm.have_matrixcore(*compute_version=None*)

计算能力中提供 MatrixCore 支持，或不提供。
* 参数：**compute_version** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)*：* GPU 的计算能力（例如“7.0”）。
* 返回:**have_matrixcore：** 如果提供 MatrixCore 支持则为 True，否则为 False
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

## tvm.contrib.rocm.find_rocm_path()


查找 ROCm 路径的实用函数。
* **返回：path：** ROCm 根的路径。
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)。


# tvm.contrib.spirv



与 SPIRV 工具交互的实用程序。

## tvm.contrib.spirv.optimize(*spv_bin*)


通过 CLI 使用 spirv-opt 优化 SPIRV。


请注意，spirv-opt 仍处于实验阶段。
* **参数：spv_bin** ([bytearray](https://docs.python.org/3/library/stdtypes.html#bytearray))：spirv 文件。
* **返回：cobj_bin**：HSA 代码对象。
* **返回类型：**[bytearray](https://docs.python.org/3/library/stdtypes.html#bytearray)。


# tvm.contrib.tar



用于在系统中调用 tarball 的实用程序。

## tvm.contrib.tar.tar(*output*, *files*)


创建包含根目录中所有文件的 tarball。
* **参数：**
   * **output** ([str](https://docs.python.org/3/library/stdtypes.html#str))：目标共享库。
   * **files** ([list](https://docs.python.org/3/library/stdtypes.html#list))：要捆绑的文件列表。

## tvm.contrib.tar.untar(*tar_file*, *directory*)


将所有 tar 文件解压到目录中。
* **参数：**
   * **tar_file** ([str](https://docs.python.org/3/library/stdtypes.html#str))：源 tar 文件。
   * **directory** ([str](https://docs.python.org/3/library/stdtypes.html#str))：目标目录

## tvm.contrib.tar.normalize_file_list_by_unpacking_tars(*temp*, *file_list*)


通过解压列表中的 tar 文件来规范化文件列表。


当文件名为 tar 文件时，它会将其解压到 temp 中的一个唯一目录中，并返回 tar 文件中的文件列表。当文件名为普通文件时，它会被直接添加到列表中。


这对于解压 tar 中的对象然后将其转换为库很有用。
* **参数：**
   * **temp** ([tvm.contrib.utils.TempDirectory](https://tvm.apache.org/docs/reference/api/python/contrib.html#tvm.contrib.utils.TempDirectory))：用于保存解压文件的临时目录。
   * **file_list** (*List[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：路径列表。
* **返回：ret_list**：更新的文件列表。
* **返回类型：** List[[str](https://docs.python.org/3/library/stdtypes.html#str)]。

# tvm.contrib.utils


常用系统实用程序。

## *exception* tvm.contrib.utils.DirectoryCreatedPastAtExit


在 atexit 钩子运行后创建 TempDirectory 时引发。

## *class* tvm.contrib.utils.TempDirectory(*custom_path=None*, *keep_for_debug=None*)


用于在测试期间管理临时目录的辅助对象。


当目录超出范围时自动删除该目录。

## *classmethod* set_keep_for_debug(*set_to=True*)


保留程序退出后的临时目录以进行调试。


### **remove()**

删除 tmp 目录。


### **relpath(*name*)**


临时目录中的相对路径。
* **参数：name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：文件的名称。
* **返回：path**：连接路径。
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)。


### **listdir()**


列出目录中的内容。
* **返回：names**：目录的内容。
* **返回类型：**[list](https://docs.python.org/3/library/stdtypes.html#list)。

## tvm.contrib.utils.tempdir(*custom_path=None*, *keep_for_debug=None*)


创建临时目录，退出时删除其内容。
* **参数：**
   * **custom_path** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：手动指定准确的临时目录路径。
   * **keep_for_debug** ([bool](https://docs.python.org/3/library/functions.html#bool))：保留临时目录以用于调试目的。
* **返回：temp**：临时目录对象。
* **返回类型：**[TempDirectory](https://tvm.apache.org/docs/reference/api/python/contrib.html#tvm.contrib.utils.TempDirectory)。

## *class* tvm.contrib.utils.FileLock(*path*)


文件锁对象。
* **参数：path** ([str](https://docs.python.org/3/library/stdtypes.html#str))：锁的路径。


### **release()**


解除锁定。

## tvm.contrib.utils.filelock(*path*)


创建一个锁定路径的文件锁。
* **参数：path** ([str](https://docs.python.org/3/library/stdtypes.html#str))*：* 锁的路径。
* **返回：lock。**
* **返回类型：** File lock object。

## tvm.contrib.utils.is_source_path(*path*)


检查路径是否是源代码路径。
* **参数：path** ([str](https://docs.python.org/3/library/stdtypes.html#str))：可能的路径。
* **返回：valid**：路径是否是可能的源路径。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

## tvm.contrib.utils.which(*exec_name*)


尝试查找 exec_name 的完整路径。
* **参数：exec_name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：可执行文件名称。
* **返回：path**：如果找到则返回可执行文件的完整路径，否则返回 None。
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)。



# tvm.contrib.xcode


调用 Xcode 编译器工具链的实用程序

## tvm.contrib.xcode.xcrun(*cmd*)


运行 xcrun 并返回输出。
* **参数：cmd** ([list](https://docs.python.org/3/library/stdtypes.html#list)of[str](https://docs.python.org/3/library/stdtypes.html#str))[：](https://docs.python.org/3/library/stdtypes.html#str)[命令](https://docs.python.org/3/library/stdtypes.html#str)序列。
* **返回：out***：*输出字符串。
* **返回类型：**[str](https://docs.python.org/3/library/stdtypes.html#str)。

## tvm.contrib.xcode.create_dylib(*output*, *objects*, *arch*, *sdk='macosx'*, *min_os_version=None*)


创建动态库。
* **参数：**
   * **output** ([str](https://docs.python.org/3/library/stdtypes.html#str))：目标共享库。
   * **objects** ([list](https://docs.python.org/3/library/stdtypes.html#list))：目标文件列表。
   * **options** ([str](https://docs.python.org/3/library/stdtypes.html#str))：附加选项。
   * **arch** ([str](https://docs.python.org/3/library/stdtypes.html#str))：目标主要架构。
   * **sdk** ([str](https://docs.python.org/3/library/stdtypes.html#str))：要使用的 sdk。

## tvm.contrib.xcode.compile_metal(*code*, *path_target=None*, *sdk='macosx'*, *min_os_version=None*)


使用 CLI 工具从环境中编译金属。
* **参数：**
   * **code** ([str](https://docs.python.org/3/library/stdtypes.html#str))：cuda 代码。
   * **path_target** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：输出文件。
   * **sdk** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：目标平台 SDK。
* **返回：metallib**：metallib 的字节数组。
* **返回类型：**[bytearray](https://docs.python.org/3/library/stdtypes.html#bytearray)。

## tvm.contrib.xcode.compile_coreml(*model*, *model_name='main'*, *out_dir='.'*)


 编译 coreml 模型，并返回编译后的模型路径。

