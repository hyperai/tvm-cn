---

title: tvm.relax.frontend


---


用于构建 Relax 程序的前端，以及模型导入器。

### tvm.relax.frontend.detach_params(*mod:*[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[NDArray]]]


将输入 IRModule 函数中的属性「params」分离为单独的参数字典。
* **参数：**
   * **mod** (*tvm.IRModule*)：即将分离其函数「param」属性的 IRModule。
* **返回：**
   * **detached_mod** (*tvm.IRModule*)：分离后的 IRModule。 
   * **params_dict** (*Dict[str, List[tvm.nd.NDArray]]*)：分离的参数。字典的键对应于输入 IRModule 中具有属性“params”的函数名称。

## tvm.relax.frontend.nn

用于构建 IRModules 的类似 PyTorch 的 API。

## *class* tvm.relax.frontend.nn.Effect 

效果是一种特殊的非面向用户的类型，用于表示具有副作用的作，例如打印。它用于表示计算的输出。

### emit_init(*name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str), *builder:*[BlockBuilder](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_builderblockbuildermodirmodulenone-none)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[DataflowVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxdataflowvarname_hintstridstruct_infostructinfononenonespanspannonenone)] 


发出效果的初始化。此方法由编译器调用来初始化效果。

### create(*name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)] 


创建代表副作用的 Relax.Function 的隐式输入。

### set_state(*state_vars:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*]*) → [None](https://docs.python.org/3/library/constants.html#None) 


设置代表效果的变量。

### finalize() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)] 


将效果最终确定为 Relax.Function 的隐式返回值。

### to(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None) 


将效果转换为特定的数据类型。通常，大多数效果都是无操作的。

## *class* tvm.relax.frontend.nn.Module 


一个在 relax.Expr 之上的包装器，其 struct_info 是基础 ObjectStructInfo（而不是其任何子类）。Object 实际上表示非张量前端组件，如 KV 缓存。

### named_parameters(*prefix:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= ''*) → [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), Parameter]] 


该方法提供了模块参数的迭代器，产生参数名称及其对应的值。
* **参数：prefix** ([str](https://docs.python.org/3/library/stdtypes.html#str))：添加在所有参数名称前面的前缀。
* **生成:** (str, Parameter) - 包含名称和参数的元组。

### parameters() → [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator)[Parameter] 


此方法提供了模块参数的迭代器，仅产生参数值。
* **生成:** *Parameter -模块的参数。*

### state_dict(, prefix: [str](https://docs.python.org/3/library/stdtypes.html#str) = '', destination: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Parameter] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Parameter] 


返回包含对模块整个状态的引用的字典。
* **参数：**
   * **prefix** ([str](https://docs.python.org/3/library/stdtypes.html#str))：添加在所有参数名称前面的前缀。
   * **返回：dict**：包含模块整个状态的字典。
* **返回：** dict：包含模块完整状态的字典
* **返回类型:**  Dict[ str, 参数]


### load_state_dict(*state_dict:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, Parameter]*, *strict:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] 

此函数将 state_dict 中的参数和缓冲区复制到当前模块及其后代模块中。如果 strict 设置为 True，则 state_dict 中的键必须与该模块的 state_dict()函数返回的键完全匹配。
* **参数：**
   * **state_dict** (*Dict*[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***Parameter*** *]*)：包含模块整个状态的字典。
   * **返回：(missing_keys, unexpected_keys)**：两个列表的元组：缺失的键和意外的键。
* **返回**：(missing_keys, unexpected_keys)：一个包含两个列表的元组：缺失的键和意外的键。
* **返回类型**：[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)[List[[str](https://docs.python.org/3/library/stdtypes.html#str)], List[[str](https://docs.python.org/3/library/stdtypes.html#str)]]。

### to(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None) 


递归地将模块转换为特定的数据类型。

### export_tvm(*spec: *spec.ModuleSpecType*, *debug: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *allow_extern: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), Parameter]]] | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), Parameter]], [List](https://docs.python.org/3/library/typing.html#typing.List)[ExternModule]] 


将模块导出到 TVM IRModule 及参数。
* **参数:**
   * spec (_spec.ModuleSpecType)：一个字典，将每个输入名称映射到定义输入形状和 dtype 的规范。
   * 8debug (bool)：如果设置为 True，则导出的模块将支持效果。这启用了在图中打印等功能。
* **返回:**
   * irmodule (tvm.ir.IRModule)：模型的 tvm IR 表示形式。
   * params (List[Tuple[str, Parameter]])：与模型权重对应的 Parameters 列表。
   * ext_mods (List[nn.ExternModule])：模型中使用的 ExternModule 列表。


### jit(*spec: *spec.ModuleSpec*, *device: [str](https://docs.python.org/3/library/stdtypes.html#str) | Device = 'cpu'*, *pipeline: [None](https://docs.python.org/3/library/constants.html#None) | [str](https://docs.python.org/3/library/stdtypes.html#str) | [Pass](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass) = 'default_build'*, *out_format: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'torch'*, *debug: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any) 


即时编译 nn.model 为可执行文件。

## *class* tvm.relax.frontend.nn.ModuleList(*modules:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[Module]*) 


将子模块保存在列表中。

### **append(*module: Module*)**

将模块添加到 ModuleList 的末尾。



### to(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None) 


递归地将模块转换为特定的数据类型。


### **forward(*x*)**


模块的前馈传递。

## *class* tvm.relax.frontend.nn.Object(***, expr:*[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) 


基于 Relax.Expr 的包装器，其 struct_info 是基础 ObjectStructInfo（而非其任何子类）。Object 有效地表示非张量前端组件，例如键值缓存。

## *class* tvm.relax.frontend.nn.Parameter(*shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


参数表示神经网络层的权重。它是一种特殊的张量，可以绑定或不绑定到具体值。如果参数绑定到具体值，则称为绑定参数；否则，称为非绑定参数。

### *property* data:*NDArray*|[None](https://docs.python.org/3/library/constants.html#None) 


如果参数绑定到具体值，则返回该参数的具体值，否则返回 None。返回值是一个 tvm.runtime.NDArray。

### to(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None) 


如果参数未绑定到任何具体数据，则更改其 dtype。

## *class* tvm.relax.frontend.nn.Tensor(* *, expr:*[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr)) 


基于 Relax.Expr 的包装器，其 struct_info 为 TensorStructInfo，提供更便捷的形状和数据类型信息访问。张量始终为符号，不绑定任何具体值。形状和数据类型推断在张量创建时立即完成，即，当运算符应用于张量时，形状和数据类型信息已可用。

### *static* from_const(*data*) → Tensor 


从 numpy 常量构造一个张量。

### *static* from_scalar(*data:*[int](https://docs.python.org/3/library/functions.html#int)*|*[float](https://docs.python.org/3/library/functions.html#float), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → Tensor 


从具有指定 dtype 的标量构造张量。

### *static* from_struct_info(*struct_info:*[TensorStructInfo](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtensorstructinfoshaperelaxexprnonelistprimexprnonedtypestrfloat32vdevicevdevicenonestrnonendimint-1spanspannonenone), *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'tensor'*) → Tensor 


从 Relax TensorStructInfo 构建一个 nn.Tensor。

### *static* placeholder(*shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'tensor'*) → Tensor 


创建一个具有给定形状和数据类型的占位符张量。通常情况下，用户不应直接创建占位符张量，唯一的例外是指示外部函数返回值的形状/数据类型。


如果形状是字符串名称，我们将创建一个符号形状 tvm.tir.Var(name, “int64”)。

### *property* shape*:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]


以整数列表形式返回张量的形状。


整数可以是 python int 或 tvm.tir.PrimExpr，具体取决于形状是否完全静态，例如，[1, 2, tvm.tir.Var(“n”)] 是一个有效形状，其中最后一个维度是动态的，而前两个维度始终是静态常量。
* **返回：shape**：张量的形状。
* **返回类型:** List[Union[[int](https://docs.python.org/3/library/functions.html#int), tir.PrimExpr]]。


### *property* ndim:[int](https://docs.python.org/3/library/functions.html#int) 

返回张量的维数。
* **返回：ndim**：张量的维数。
* **返回类型:**[int](https://docs.python.org/3/library/functions.html#int)。


### *property* dtype:[str](https://docs.python.org/3/library/stdtypes.html#str) 


返回张量的数据类型。
* **返回：dtype**：张量的数据类型。
* 返回类型：[str](https://docs.python.org/3/library/stdtypes.html#str)。

## tvm.relax.frontend.nn.add_extern(*mod: ExternModule*) → [None](https://docs.python.org/3/library/constants.html#None) 


向导出器添加外部模块。

## *class* tvm.relax.frontend.nn.ExternModule(*symbols:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*]*) 


外部模块的抽象基类。外部模块旨在帮助将用户提供的手工编写的内核合并到导出的 TVM IRModule 中。

### load() → Module 


将外部模块加载到 TVM 运行时模块中。

## *class* tvm.relax.frontend.nn.ObjectModule(*symbols:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*]*, *filepath:*[Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) 


nn.ExternModule 的子类，允许用户提供一个对象.o 文件以链接到编译的工件中；

### load() → Module 


将外部模块加载到 TVM 运行时模块中。

## *class* tvm.relax.frontend.nn.SourceModule(*symbols:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*]*, *source_code:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path), *source_format:*[str](https://docs.python.org/3/library/stdtypes.html#str), *compile_options:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *compiler:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *output_format:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'obj'*) 


nn.ExternModule 的子类。它编译 C++/CUDA 源代码并将其链接到最终的 IRModule 中。


**形状/数据类型推断。nn.ExternModule**系统需要用户提供额外的信息，即符号(symbols )。它是一个字典，将外部目标文件中的每个符号映射到其形状/数据类型推断函数。假设函数 my_func 接受两个张量，形状为(x, y, 1)的 a 和形状为(y, z, 5)的 b，并生成一个形状为(x, y, z, 9)的张量 c，则形状/数据类型推断函数应如下所示：

```python
def shape_dtype_inference(a, b):
    x, y, * = a.shape
    *, z, * = b.shape
    return nn.Tensor.placeholder((x, y, z, 9), dtype="float32")
```


符号字典应提供为：

```python
symbols={
    "my_func": shape_dtype_inference,
}
```


**调用约定。** 所有外部模块现在都遵循「目标传递式」（DPS）调用约定，这意味着返回的张量已由系统预先分配，并作为外部函数的参数传入。


复用上面的示例，my_func 的实现应该在其签名中包含三个参数，其中张量使用 DLPack 中的 DLTensor 表示，DLPack 是张量内存表示的事实标准。更多详情：

 [https://github.com/dmlc/dlpack/blob/v0.8/include/dlpack/dlpack.h#L163-L206](https://github.com/dmlc/dlpack/blob/v0.8/include/dlpack/dlpack.h#L163-L206)。


为了公开符号，保证 TVM_FFI_DLL_EXPORT_TYPED_FUNC(symbol, function)可用：

```plain
// 这些头文件必须存在
#include <dlpack/dlpack.h>

namespace {
// 匿名名称空间从其他翻译单元中隐藏符号 `*my_func_impl`
int *my_func_impl(DLTensor* a, DLTensor* b, DLTensor* c) {
    // `a` and `b` are inputs, and `c` is the output
}
}
// 暴露符号 `my_func` 代替 `*my_func_impl`
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_func, *my_func_impl);
```


**编译器过程「AttachExternModules」。它的作用是** 在编译流水线的任意阶段将 nn.ExternModule 列表附加 到 IRModule 中，并将编译后的外部模块作为「runtime.Module」附加到 IRModule 的「external_mods」属性中。在 tvm.compile 中进行链接时需要它，但有了此过程，源代码编译可以推迟到 TVM 编译的任意阶段。


**注意事项：** 在 export_tvm 期间，需要调用 nn.add_extern 来注册外部模块，且仅注册一次。每个符号都应仅注册一次，以避免潜在的冲突，否则将引发错误。

### *static* tvm_home() → [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) 


查找 TVM 的主目录。如果已设置 TVM_HOME 环境变量，则使用该变量。否则，请使用 tvm Python 包的安装目录。为了确保完整性，需要将 include 和3rdparty 作为直接子目录。
* **返回：tvm_home**：TVM 主目录，保证包含 include 和 3rdparty 作为直接子目录。
* 返回类型：[pathlib.Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)。

### *static* get_includes(*tvm_pkg:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)] 


根据 tvm_home() 返回默认包含路径。默认情况下，它包含 TVM、DLPack 和 DMLC-Core。如果提供 tvm_pkg ，它还会包含 tvm_home/3rdparty 下的指定软件包。
* **参数：tvm_pkg** (Optional[**List**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**]])： tvm_home/3rdparty 下要包含的软件包列表。每个元素都应该是 tvm_home/3rdparty 的相对路径。
* **返回：includes**：包含路径的列表。
* 返回类型：List[[pathlib.Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]

### *static* get_compile_options(*source_format:*[str](https://docs.python.org/3/library/stdtypes.html#str), *tvm_pkg:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)] 


根据 source_format 返回默认编译选项，包括 wrt tvm_home() 的默认包含路径、配置 DMLC-Core 的默认标志，默认情况下，它使用「-O3」和「-std=c++17」。
* **参数：**
   * source_format (str)：源代码格式。可以是「cpp」或「cu」。
   * tvm_pkg (Optional[List[str]])：要包含在 tvm_home/3rdparty 下的包列表。每个元素应该是 tvm_home/3rdparty 的相对路径。
* 返回:**编译选项**：编译标志列表。
* **返回类型:** List[[str](https://docs.python.org/3/library/stdtypes.html#str)]

### compile(*output_path:*[Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) → [None](https://docs.python.org/3/library/constants.html#None) 


在提供的目录中编译源代码并返回编译后的工件。

### load() → Module 


将外部模块加载到 TVM 运行时模块中。

## *class* tvm.relax.frontend.nn.GELU 


Relax.frontend.nn.Module 用于 GELU 激活层。

## *class* tvm.relax.frontend.nn.Conv1D(*in_channels:*[int](https://docs.python.org/3/library/functions.html#int), *out_channels:*[int](https://docs.python.org/3/library/functions.html#int), *kernel_size:*[int](https://docs.python.org/3/library/functions.html#int), *stride:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *bias:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


Relax.frontend.nn.Module 用于 conv1d 层。

### forward(*x: Tensor*) → Tensor 


conv1d 层的前向方法。
* **参数：x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
* **返回：ret**：conv1d 层的输出张量。
* 返回类型：-  [Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## *class* tvm.relax.frontend.nn.Conv2D(*in_channels:*[int](https://docs.python.org/3/library/functions.html#int), *out_channels:*[int](https://docs.python.org/3/library/functions.html#int), *kernel_size:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[int](https://docs.python.org/3/library/functions.html#int), *stride:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *bias:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCHW'*) 


Relax.frontend.nn.Module 用于 conv2d 层。

### forward(*x: Tensor*) → Tensor 


conv2d 层的前向方法。
* **参数：x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)输入张量。
* **返回：ret**：conv2d 层的输出张量。
* 返回类型：- [Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## *class* tvm.relax.frontend.nn.Conv3D(*in_channels:*[int](https://docs.python.org/3/library/functions.html#int), *out_channels:*[int](https://docs.python.org/3/library/functions.html#int), *kernel_size:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[int](https://docs.python.org/3/library/functions.html#int), *stride:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *padding:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *bias:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'NCDHW'*) 


Relax.frontend.nn.Module 用于 conv3d 层。

### forward(*x: Tensor*) → Tensor 


conv3d 层的前向方法。
* **参数：x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
* **返回：ret**：conv3d 层的输出张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## *class* tvm.relax.frontend.nn.ConvTranspose1D(*in_channels:*[int](https://docs.python.org/3/library/functions.html#int), *out_channels:*[int](https://docs.python.org/3/library/functions.html#int), *kernel_size:*[int](https://docs.python.org/3/library/functions.html#int), *stride:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *output_padding:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *bias:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


ConvTranspose1D 层的 Relax.frontend.nn.Module。

### forward(*x: Tensor*) → Tensor 


conv 转置 1d 层的前向方法。
* **参数：x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
* **返回：ret**：conv 转置 1d 层的输出张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## *class* tvm.relax.frontend.nn.Embedding(*num:*[int](https://docs.python.org/3/library/functions.html#int)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *dim:*[int](https://docs.python.org/3/library/functions.html#int)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


Relax.frontend.nn.Module 用于嵌入层。


### **forward(*x: Tensor*)**


嵌入层的前向方法。
* **参数：x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
* **返回：ret**：嵌入层的输出张量。
* 返回类型：Tensor 张量。


## *class* tvm.relax.frontend.nn.GroupNorm(*num_groups:*[int](https://docs.python.org/3/library/functions.html#int), *num_channels:*[int](https://docs.python.org/3/library/functions.html#int), *eps:*[float](https://docs.python.org/3/library/functions.html#float)*= 1e-05*, *affine:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


Relax.frontend.nn.Module 用于组规范层。


### **forward(*x: Tensor*,*channel_axis:***[int](https://docs.python.org/3/library/functions.html#int)***= 1*,*axes:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[None](https://docs.python.org/3/library/constants.html#None)***= None*)**


群体范数层的前向方法。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
   * channel_axis (int)：输入数据的通道轴。
   * axes (Optional[List[int]])：可选，计算范数的轴列表，如果未指定，则假设前两个轴保持不变。
* **返回：ret**：组范数层的输出张量。
* 返回类型；[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## *class* tvm.relax.frontend.nn.IOEffect 


建模 IO 副作用，例如在屏幕上打印 NDArrays 的内容、插入调试断点等。

### emit_init(*name_hint*, *builder:*[BlockBuilder](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_builderblockbuildermodirmodulenone-none)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[DataflowVar](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxdataflowvarname_hintstridstruct_infostructinfononenonespanspannonenone)] 


发出效果的初始化。此方法由编译器调用来初始化效果。

### create(*name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)] 


创建代表副作用的 Relax.Function 的隐式输入。

### set_state(*state_vars:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)*]*) → [None](https://docs.python.org/3/library/constants.html#None) 


设置代表效果的变量。

### finalize() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)] 


将效果最终确定为 Relax.Function 的隐式返回值。

## *class* tvm.relax.frontend.nn.KVCache(*init_seq_len:*[int](https://docs.python.org/3/library/functions.html#int), *unit_shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


实现 KVCache 的效果。

### emit_init(*name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str), *bb:*[BlockBuilder](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax_block_builder#class-tvmrelaxblock_builderblockbuildermodirmodulenone-none))


发出 KVCache 效果的初始化。
* **参数：**
   * **name_hint** ([str](https://docs.python.org/3/library/stdtypes.html#str))：初始化绑定 Var 的名称提示。
   * bb (relax.BlockBuilder)：用于生成 relax BlockBuilder。

### **create(*name_hint:***[str](https://docs.python.org/3/library/stdtypes.html#str)**)→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[**[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)**]**

为表示 KVCache 效应的 relax.Function 创建隐式输入。
* 参数 **:**
   * name_hint (str)：relax.Var 的名称提示。
* 返回:ret：KVCache 的 relax.Var。
* **返回类型:** List[[relax.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)]。


### **set_state(*state_vars:***[List](https://docs.python.org/3/library/typing.html#typing.List)***[***[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)***]*)→**[None](https://docs.python.org/3/library/constants.html#None)

设置代表效果的变量。

### finalize() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxvarname_hintstridstruct_infostructinfononenonespanspannonenone)] 


将 KVCache 效果最终确定为 Relax.Function 的隐式返回值。
* **返回：ret**[：](https://docs.python.org/3/library/constants.html#None)输出 Relax.Var 作为 KVCache。
* 返回类型：List[rx.Var]。

### to(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None) 


将 KVCache 效果转换为特定的 dtype。
* **参数：dtype** (*Optional[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*)：要转换的目标数据类型。


### **view(*seq_len:***[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)**)→ Tensor**

查看 KVCache 中的最后元素。
* 参数: seq_len (tir.Var)：查看最后元素的个数。
* 返回: ret：最后查看的张量。
* 返回类型 **:** [Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


### append(*new_element: Tensor*) → [None](https://docs.python.org/3/library/constants.html#None) 


在 KVCache 中附加一个新元素。
* **参数：new_element** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：要附加的新张量。




## *class* tvm.relax.frontend.nn.LayerNorm(*normalized_shape:*[int](https://docs.python.org/3/library/functions.html#int), *eps:*[float](https://docs.python.org/3/library/functions.html#float)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1e-05*, *elementwise_affine:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)

用于层归一化的 relax.frontend.nn.Module。

### forward(*x: Tensor*) → Tensor

层归一化层的正向方法。
* **参数：x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
* **返回：ret**：层规范化层的输出张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## *class* tvm.relax.frontend.nn.Linear(*in_features:*[int](https://docs.python.org/3/library/functions.html#int)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *out_features:*[int](https://docs.python.org/3/library/functions.html#int)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *bias:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


Relax.frontend.nn.Module 用于线性层。

### forward(*x: Tensor*) → Tensor 


线性层的前向方法。
* **参数：x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
* **返回：ret**：线性层的输出张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

### to(*dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [None](https://docs.python.org/3/library/constants.html#None) 


重写 to() ，以便在存在 out_dtype 时不转换 bias 。否则，在计算x + self.bias 时可能会遇到数据类型不匹配的情况， 因为 x 的类型是 out_dtype，而 bias 的类型会发生变化，两者可能存在差异。

## *class* tvm.relax.frontend.nn.ReLU 


Relax.frontend.nn.ReLU 激活层的模块。

## *class* tvm.relax.frontend.nn.RMSNorm(*hidden_size:*[int](https://docs.python.org/3/library/functions.html#int), *axes:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *epsilon:*[float](https://docs.python.org/3/library/functions.html#float)*= 1e-05*, *bias:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


Relax.frontend.nn.Module 用于 rms 范数层。


### **forward(*x: Tensor*)**


均方根范数层的前向方法。
* **参数：x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
* **返回：ret**：均方根范数层的输出张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。



## *class* tvm.relax.frontend.nn.SiLU 


Relax.frontend.nn.SiLU 激活层的模块。

## *class* tvm.relax.frontend.nn.SubroutineMixin 


生成一个 mixin。


包含 tvm.relax.frontend.nn.Module 和 tvm.relax.testing.nn.Module 的通用逻辑。

## *class* tvm.relax.frontend.nn.Mutator 


nn.Module transform 的修改器。用户可以重写 visit_*方法，在不同的结构中应用 transform，甚至可以重写 visit 方法来改变遍历的逻辑。

### visit_module(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *node: Module*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any) 


nn.Module 节点变异的基础访问方法。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：父级属性中当前节点的名称。
   * 节点 (nn.Module)**：** 要变异的当前 nn.Module 节点。
* **返回：ret_node**：替换当前节点的新节点。
* 返回类型：任何。

### visit_effect(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *node: Parameter*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any) 


nn.Parameter 节点变异的基础访问方法。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：父级属性中当前节点的名称。
   * node (nn.Parameter)**：** 当前要变异的 nn.Parameter 节点。
* **返回：ret_node**：替换当前节点的新节点。
* 返回类型：任何。

### visit_param(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *node: Effect*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any) 


nn.Effect 节点变异的基访问方法。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：父级属性中当前节点的名称。
   * 节点 (nn.Effect)**：** 当前要变异的 nn.Effect 节点。
* **返回：ret_node**：替换当前节点的新节点。
* 返回类型：任何。

### visit_modulelist(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *node: ModuleList*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any) 


nn.ModuleList 节点变异的基础访问方法。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：父级属性中当前节点的名称。
   * node (nn.ModuleList)**：** 当前要变异的 nn.ModuleList 节点。
* **返回：ret_node：** 替换当前节点的新节点。
* 返回类型：任何。

### visit(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *node:*[Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [Any](https://docs.python.org/3/library/typing.html#typing.Any) 


所有节点访问的基本调度方法。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：父级属性中当前节点的名称。
   * 节点（Any）：当前要访问的节点。
* **返回：ret_node**：替换当前节点的新节点。
* 返回类型：任何。

## *class* tvm.relax.frontend.nn.TypeVar(*name*, constraints*, *bound=None*, *covariant=False*, *contravariant=False*) 


类型变量。


用法：

```python
T = TypeVar('T')  # Can be anything  # 任意
A = TypeVar('A', str, bytes)  # Must be str or bytes # 必须是 str 或 bytes
```


类型变量主要用于静态类型检查器。它们用作泛型类型以及泛型函数定义的参数。有关泛型类型的更多信息，请参阅 Generic 类。泛型函数的工作原理如下：


def repeat(x: T, n: int) -> List[T]: 

'''返回包含 n 个对 x 的引用的列表。''' return [x]*n


def longest(x: A, y: A) -> A: 

'''返回两个字符串中最长的那个。''' return x if len(x) >= len(y) else y


后一个示例的签名本质上是 (str, str) -> str 和 (bytes, bytes) -> bytes 的重载。另请注意，如果参数是 str 某个子类的实例，则返回类型仍然是普通的 str。


在运行时，isinstance(x, T) 和 issubclass(C, T) 将引发 TypeError。


使用 covariant=True 或 contravariant=True 定义的类型变量可用于声明协变或逆变泛型类型。更多详情请参阅 PEP 484。默认情况下，泛型在所有类型变量中都是不变的。


类型变量可以被自省。例如：

T.**name** == ‘T’ T.**constraints** == () T.**covariant** == False T.**contravariant** = False A.**constraints** == (str, bytes)



请注意，只有在全局范围内定义的类型变量才能被腌制。

## tvm.relax.frontend.nn.add(*a: Tensor*, *b: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'add'*) → Tensor 


添加 numpy 风格的广播。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * b (Tensor)**：** 第二个输入张量。
   * name (str)[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

c = add(a, b)

## tvm.relax.frontend.nn.argsort(*data: Tensor*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*, *descending:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'int32'*, *name='argsort'*) 


沿给定轴执行排序，并返回与按排序顺序索引数据的输入数组具有相同形状的索引数组。
* **参数：**
   * **data** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入数据张量。
   * axis (int)：按此轴对输入张量进行排序。
   * 降序（bool）是否按降序排序，默认为 False
* **返回：out**：排序张量的索引。

## tvm.relax.frontend.nn.astype(*x: Tensor*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str), *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'astype'*) → Tensor 

将输入张量转换为给定的数据类型。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))*：*运算符的输入数据。
   * dtype (str)：目标数据类型。
   * name (str)：名称提示。
* **返回：result**：转换结果。
* **返回类型**：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.broadcast_to(*x: Tensor*, *shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'broadcast_to'*) → Tensor 


将张量广播到指定形状。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * shape (Sequence[IntExpr])：目标形状。
   * name (str)：名称提示。
* **返回：result**：广播的张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)

## tvm.relax.frontend.nn.ccl_allgather(*x: Tensor*, *num_workers:*[int](https://docs.python.org/3/library/functions.html#int), *name='ccl_allgather'*) 

CCL Allgather 运算符。
* **参数：**
   * **x** (*relax.Expr*)：输入张量。
   * num_workers (int)：工作线程数。
   * name (str)[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)该操作的名称提示。
* **返回：result**：allgather 的结果张量。
* 返回类型：[Tensor 张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.ccl_allreduce(*x: Tensor*, *op_type:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'sum'*, *in_group:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *name='ccl_allreduce'*) 

CCL Allreduce 运算符。
* **参数：**
   * **x** (*relax.Expr*)：输入张量。
   * op_type (str)：应用于输入数据的归约操作类型。目前支持「sum」、「prod」、「min」、「max」和「avg」。
   * name (str)：该操作的名称提示。
* **返回：result**：allreduce 的结果张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.ccl_broadcast_from_worker0(*x: Tensor*, *name='broadcast_from_worker'*) 


将数据从 worker-0 广播到所有其他 worker。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：要广播的张量。
   * name (str)：该操作的名称提示。
* **返回：result**：相同的张量，已广播给所有其他工作者。
* 返回类型：Tensor  张量。

## tvm.relax.frontend.nn.chunk(*x: Tensor*, *chunks:*[int](https://docs.python.org/3/library/functions.html#int), *dim:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'chunk'*) → Tensor 


将张量沿 dim 分成指定数量的块。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：要分割的输入张量。
   * chunks (int)：将 x 切成多少块。
   * dim (int)：在哪个维度上分割 x。
   * name (str)：该操作的名称提示。
* **返回：result**：具有包含 x 切片的块元素的元组。
* 返回类型；[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)[[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)]。

## tvm.relax.frontend.nn.concat(*x:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[Tensor]*, *dim:*[int](https://docs.python.org/3/library/functions.html#int), *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'concat'*) → Tensor 


沿轴连接张量列表。
* **参数：**
   * **x** (*List[*[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*]*)：要连接的张量列表。
   * dim (int)：连接的维度。
   * name (str)：该算子的名称提示。
* **返回：result**：扩展结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.conv1d(*x: Tensor*, *weight: Tensor*, *bias: Tensor |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *stride:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 0*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'conv1d'*) → Tensor 


一维卷积。


该运算符将权重作为一维卷积核，并将其与数据卷积以产生输出。


默认情况下，data_layout 为 NCW ，kernel_layout 为 OIW，conv1d 接受一个形状为(batch_size, in_channels, width) 的数据张量和一个形状为(channels, in_channels, kernel_w)的权重张量，其中 kernel_w 是 W 核维度的长度，以生成具有以下规则的输出张量：

$$\text{out}[b,c,x] = \sum_{dx,k} \text{data}[b,k,\text{strides} \ast x + dx] \ast \text{weight}[c,k,dx]$$

在计算之前，分别对数据和权重应用填充和膨胀。该算子接受数据布局规范。从语义上讲，该算子会将布局转换为规范布局（数据为 NCW ，权重为 OIW），执行计算，然后转换为 out_layout。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * weight (Tensor)：权重表达式。
   * bias (Optional[Tensor])：可选的偏置张量，形状为 [O]。
   * strides (Optional[Union[int, Tuple]])：卷积的步长。必须具有长度 1。
   * padding (Optional[Union[int, Tuple, str]])：卷积前输入两边的填充。必须具有长度 1 或 2。
   * dilation (Optional[Union[int, Tuple]])：指定用于扩张卷积的扩张率。必须具有长度 1。
   * groups (Optional[int])[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)分组卷积将输入分成多少组。输入和输出通道的数量应该能被组数整除。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.conv1d_transpose(*x: Tensor*, *weight: Tensor*, *bias: Tensor |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *stride:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*, …] |*[None](https://docs.python.org/3/library/constants.html#None)*= 0*, *output_padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= 0*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'conv1d_transpose'*) → Tensor 


1D 转置卷积算子。


该算子可以看作是 conv1d 的梯度算子。


输出形状可以用 data_layout == “NCW”和 kernel_layout == “IOW” 的简单情况来解释。假设数据形状为(N, in_channel, in_w)，权重形状为(in_channel, out_channel, weight_w)，我们需要确保 in_channel % groups == 0。输出形状为(N, out_channel * groups, out_w)，其中
* out_w = ((in_w - 1) * strides[0] + weight_w - 2 * padding[0] + output_padding[0])
* **参数：**
   * **data** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * 权重（Tensor）权重张量。
   * 步长（Union[int, Tuple[int]]）卷积的步长。必须具有长度 1。
   * 填充（Union[int, Tuple[int, ...]]）卷积前在输入两侧的填充。必须具有长度 1 或 2。
   * 输出填充（Union[int, Tuple[int, ...]], 可选）用于区分输出形状。
   * dilation (Union[int, Tuple[int]])：指定用于扩张卷积的扩张率。必须具有长度为 1。
   * groups (int)：分组卷积将输入分成多少组。输入和输出通道的数量应该能被组数整除。
   * data_layout (str)：输入的布局。
   * kernel_layout (str)：权重的布局。
   * out_layout (Optional[str])：输出的布局。如果未指定，则与 data_layout 相同。
   * out_dtype (Optional[Union[str, DataType]])*：*指定混合精度 conv2d 的输出数据类型。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.conv2d(*x: Tensor*, *weight: Tensor*, *bias: Tensor |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *stride:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 0*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 'NCHW'*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'conv2d'*) → Tensor 

对由多个输入平面组成的输入图像应用二维卷积。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：形状为 [B, N, H, W] 的输入张量。
   * 权重 (Tensor)：形状为 [O, N/groups, kH, kW] 的过滤器
   * 偏置 (Optional[Tensor])：形状为 [O] 的可选偏置张量。
   * 步长 (Optional[Union[int, Tuple]])：卷积核的步长。可以是单个数字或 (sH, sW) 元组。
   * padding (可选[[Union[int, Tuple]]])：输入两边的隐式填充。
   * dilation (可选[Union[int, Tuple]])：卷积核元素之间的间距。可以是单个数字或一个元组（dH, dW）。
   * groups (可选[int])：将输入分成若干组。
   * data_layout (Optional[str])：输入和输出数据的布局。
   * name (str)：名称提示。
* **返回：result**：计算结果，形状为 [B, O, oH, oW]。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


## tvm.relax.frontend.nn.conv3d(*x: Tensor*, *weight: Tensor*, *bias: Tensor |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *stride:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 0*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *groups:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 1*, *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 'NCDHW'*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'conv3d'*) → Tensor 


对由多个输入平面组成的输入图像应用 3D 卷积。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：形状为 [B, N, D, H, W] 的输入张量。
   * weight (Tensor)：滤波器，形状为 [O, N/groups, kD, kH, kW]
   * bias (Optional[Tensor])：可选的偏置张量，形状为 [O].
   * stride (Optional[Union[int, Tuple]])：卷积核的步长。可以是单个数字或 (sD, sH, sW) 元组。
   * padding (可选[[Union[int, Tuple]]])：输入两边的隐式填充。
   * dilation (Optional[Union[int, Tuple]])：卷积核元素之间的间距。可以是一个数字或一个元组（dD, dH, dW）。
   * groups (可选[int])[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)将输入分成若干组。
   * data_layout (Optional[str])：可选的输入和输出数据布局。
   * name (str)：名称提示。
* **返回：result**：计算结果，形状为 [B, O, oD, oH, oW]。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.cumsum(*data: Tensor*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *exclusive:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'cumsum'*) → Tensor 


Numpy 风格的 cumsum op。返回沿给定轴的元素的累积包含总和。
* **参数：**
   * **data** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * axis (可选[int])：沿着哪个轴计算累积和。默认值（None）是计算扁平化数组的 cumsum。dtype (Optional[str])**：** 返回数组的类型以及用于累加元素的累加器的类型。如果未指定 dtype，则默认为 data 的类型。exclusive (Optional[bool])[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)如果为 true，将返回不包含第一个元素的排他性求和。
   * name (str)：名称提示。
* **返回：result**：如果 axis 不为 None，则结果的大小与数据相同，形状也与数据相同。如果 axis 为 None，则结果为一维数组。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
a = [[1, 2, 3], [4, 5, 6]]

cumsum(a)  # if axis is not provided, cumsum is done over the flattened input. # 如果未提供轴向，在展平输入上标准化
-> [ 1,  3,  6, 10, 15, 21]

cumsum(a, dtype="float32")
-> [  1.,   3.,   6.,  10.,  15.,  21.]

cumsum(a, axis=0)  # sum over rows for each of the 3 columns # 求和每行的 3 列
-> [[1, 2, 3],
    [5, 7, 9]]

cumsum(a, axis=1)
-> [[ 1,  3,  6],
    [ 4,  9, 15]]

a = [1, 0, 1, 0, 1, 1, 0]  # a is a boolean array # a 是布尔数组
cumsum(a, dtype=int32)  # dtype should be provided to get the expected results # dtype 必须提供，才能得到预期的结果
-> [1, 1, 2, 2, 3, 4, 4]
```
## tvm.relax.frontend.nn.debug_func(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), args: Tensor |*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[int](https://docs.python.org/3/library/functions.html#int)*|*[float](https://docs.python.org/3/library/functions.html#float)*|*[str](https://docs.python.org/3/library/stdtypes.html#str), line_info:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


relax.Call 在运行时调用调试函数。调试函数必须使用以下类型签名注册：

```python
@tvm.register_func(name_of_debug_func)
def debug_func(lineno: str, arg*0, arg*1, ...) -> None:
    ...
```
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：要调用的调试函数的名称。
   * args (Union[Tensor, _tir.PrimExpr, int, float, str])：传递给调试函数的参数。

## tvm.relax.frontend.nn.divide(*a: Tensor*, *b: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'divide'*) → Tensor 


使用 numpy 风格广播进行除法。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * b (Tensor)[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)第二个输入张量。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
c = divide(a, b)
```
## tvm.relax.frontend.nn.empty(*shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'float32'*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'empty'*) → Tensor 


构建一个未初始化的张量，具有输入形状和 dtype。
* **参数：**
   * **shape** (*Sequence*[*IntExpr]*)：创建的张量的形状。
   * dtype (str)：创建的张量的数据类型。
   * name (str)：名称提示。
* **返回：result**：结果张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.equal(*a: Tensor*, *b: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'equal'*) → Tensor 


广播元素比较（lhs == rhs）。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * b (Tensor)**：** 第二个输入张量。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.exp(*x: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'exp'*) → Tensor 


应用指数函数。

$$\text{Exp}(x) = e^x$$
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


:::Note

输入张量需要具有浮点型。

:::

## tvm.relax.frontend.nn.extern(*name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *args:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[Tensor |*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*|*[int](https://docs.python.org/3/library/functions.html#int)*|*[float](https://docs.python.org/3/library/functions.html#float)*|*[str](https://docs.python.org/3/library/stdtypes.html#str)*]*, *out: OutType*) → OutType 


在运行时调用外部函数。该外部函数必须使用 TVM_FFI_REGISTER_GLOBAL (C++) 或 tvm.register_func (Python)在 TVM 运行时注册。
* **参数：**
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str))：要调用的外部函数的名称。
   * args (Sequence[Union[Tensor, _tir.PrimExpr, int, float, str]])：传递给 extern 函数的参数。
   * out (Union[Tensor, List[Tensor]])：仅输出张量。
* **返回：result**：结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.full(*shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *fill_value: Tensor*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'float32'*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'full'*) → Tensor 


用标量值填充数组。
* **参数：**
   * **shape** (*Sequence**[****IntExpr]*)*：创建的张量的形状。
   * args (Sequence[Union[Tensor, _tir.PrimExpr, int, float, str]])[：](https://docs.python.org/3/library/constants.html#None)传递给 extern 函数的参数。
   * out (Union[Tensor, List[Tensor]])：仅输出张量。
* **返回：result**：结果张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.gelu(*x: Tensor*, *approximate:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'gelu'*) → Tensor 


应用高斯误差线性单位函数。

$$\text{GeLU}(x) = 0.5 * x * (1 + \text{erf}(x * 0.5**0.5))$$

在那里 erf 是高斯误差函数。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入数据。
   * approximate (Optional[str])：如果设置为 tanh，则在计算 CDF 时使用近似值。
   * name (str)：名称提示。
* **返回：result**[：](https://docs.python.org/3/library/functions.html#bool)计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


:::Note

输入张量需要具有浮点型

:::

## tvm.relax.frontend.nn.get_default_dtype() → [str](https://docs.python.org/3/library/stdtypes.html#str) 


如果未指定，则获取默认参数 dtype。默认情况下为 float32。
* **返回：dtype**：默认 dtype。
* **返回类型:** [str](https://docs.python.org/3/library/stdtypes.html#str)。


## tvm.relax.frontend.nn.get_timestep_embedding(*x: Tensor*, *embedding_dim:*[int](https://docs.python.org/3/library/functions.html#int), *flip_sin_to_cos:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *downscale_freq_shift:*[float](https://docs.python.org/3/library/functions.html#float)*= 1*, *scale:*[float](https://docs.python.org/3/library/functions.html#float)*= 1*, *max_period:*[int](https://docs.python.org/3/library/functions.html#int)*= 10000*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'get_timestep_embedding'*) → Tensor 


时间步长的计算如去噪扩散概率模型中所述。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：N 个索引的一维张量。
   * embedding_dim (int)：输出的维度。flip_sin_to_cos (bool)：如果为 True，则改变正弦和余弦嵌入的顺序。
   * downscale_freq_shift (float)：调整正弦采样频率。
   * scale (float)：嵌入幅度权重调整。max_period (int)：控制嵌入的最小频率。
   * name (str)：用以标记此算子的名称。
* **返回：result**：[N x dim] 位置嵌入的张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.greater(*a: Tensor*, *b: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'greater'*) → Tensor 


广播元素比较（lhs > rhs）。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * b (Tensor)：第二个输入张量。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.greater_equal(*a: Tensor*, *b: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'greater_equal'*) → Tensor 


广播逐元素比较 (lhs >= rhs)。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))*：*第一个输入张量。
   * b (Tensor)：第二个输入张量。
   * name (str)：名称提示。
* **返回：result**[：](https://arxiv.org/abs/1803.08494)计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.group_norm(*x: Tensor*, *num_groups:*[int](https://docs.python.org/3/library/functions.html#int), *weight: Tensor |*[None](https://docs.python.org/3/library/constants.html#None), *bias: Tensor |*[None](https://docs.python.org/3/library/constants.html#None), *eps:*[float](https://docs.python.org/3/library/functions.html#float)*= 1e-05*, *channel_axis:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *axes:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'group_norm'*) → Tensor 


按照论文 Group Normalization 中的描述，对小批量输入应用[Group Normalization](https://arxiv.org/abs/1803.08494)

$$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{relax.Var}[x] + \epsilon}} * \gamma + \beta$$
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：将应用 rms_norm 的输入。
   * num_groups (int)：将通道分成多少组的数量。
   * weight (Tensor)：伽马缩放因子。
   * bias (Tensor)[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)Beta 偏移因子。
   * epsilon (float)：在平方均值上添加的小浮点数，以避免除以零。channel_axis (int)*：*数据的通道轴。axes (Optional[int])[：](https://docs.python.org/3/library/functions.html#float)计算 groupnorm 的轴。如果为 None，则假定应忽略前两个通道。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.interpolate(*x: Tensor*, *size:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *scale_factor:*[float](https://docs.python.org/3/library/functions.html#float)*|*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[float](https://docs.python.org/3/library/functions.html#float)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *mode:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'nearest'*, *align_corners:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *recompute_scale_factor:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *antialias:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= 'NCHW'*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'interpolate'*) 

使用指定的模式调整张量的大小。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：需要调整大小的输入张量。
   * size (Optional[Union[int, Tuple[int]]])：请求的输出尺寸，仅能指定 size 或 scale_factor 中的一个。
   * scale_factor (Optional[Union[float, Tuple[float]]])：空间尺寸的乘数。mode (str)：采样所使用的算法。
   * align_corners (Optional[bool])：采样前后像素的映射方式。recompute_scale_factor (Optional[bool])：为插值重新计算 scale_factor。抗锯齿 (Optional[bool])*：*对输出应用抗锯齿。
   * data_layout (Optional[str])*：*输入和输出数据的布局。
   * name (str)：该操作的名称提示。
* **返回：result**：具有请求形状的输出张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


## tvm.relax.frontend.nn.layer_norm(*x: Tensor*, *normalized_shape:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *weight: Tensor |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *bias: Tensor |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *eps:*[float](https://docs.python.org/3/library/functions.html#float)*= 1e-05*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'layer_norm'*) → Tensor 


层归一化 (Lei Ba 等，2016)。将层归一化应用于 n 维输入数组。该运算符接受一个 n 维输入数组，并使用给定的轴对输入进行归一化：

$$out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis)+\epsilon}} * gamma + beta$$

与批量标准化不同，平均值和方差是沿着通道维度计算的。


假设输入在轴 1 上的大小为 k，则 gamma 和 beta 的形状均为 (k,)。

:::Note

该运算符可以进行优化以进行推理。

:::
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：将应用 layer_norm 的输入。
   * normalized_shape (Union[int, List[int]])：归一化的轴的形状。如果使用单个整数，它被视为一个单元素列表，此模块将在最后一个维度上归一化。
   * weight (Tensor)：伽马缩放因子。
   * bias (Tensor)：Beta 偏移因子。
   * eps (float)：添加到方差的小浮点数，以避免除以零。
   * name (str)：名称提示。
* **返回：result***：*计算结果。
* 返回了类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.less(*a: Tensor*, *b: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'less'*) → Tensor 


广播元素比较（lhs < rhs）。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * b (Tensor)：第二个输入张量。
   * name (str)**：** 名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.less_equal(*a: Tensor*, *b: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'less_equal'*) → Tensor 


广播逐元素比较 (lhs <= rhs)。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))[：](https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html)第一个输入张量。
   * b (Tensor)[：](https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html)第二个输入张量。
   * name (str)[：](https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html)名称提示。
* **返回：result**：计算结果。
* 返回类型：Tensor  张量。

## tvm.relax.frontend.nn.matmul(*a: Tensor*, *b: Tensor*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'matmul'*) → Tensor 


两个张量的一般矩阵乘法，在分批维度上进行广播。


语义和输出形状推断规则指定为 [https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html](https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html)。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * b (Tensor)：第二个输入张量。
   * out_dtype (Optional[Union[str, DataType]])*：*矩阵乘法结果的数值类型。当未指定时，输出 dtype 将与输入 dtype 相同。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。




**示例**

```python
c = matmul(a, b)
```
## tvm.relax.frontend.nn.max(*x: Tensor*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'max'*) → Tensor 


计算给定轴上张量元素的最大值。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))**：** 输入数据张量。
   * axis (Optional[Union[int, List[int]]])：指定执行最大值操作的轴或轴。默认值 axis=None 将对输入张量的所有元素执行最大值操作。支持负索引。
   * keepdims (bool)：如果设置为 True，被缩减的轴将保留为结果中的尺寸为一的维度。使用此选项，结果将正确地广播与输入张量。
   * name (str)：该操作的名称提示。
* **返回：result**[：](https://docs.python.org/3/library/functions.html#bool)计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.maximum(*x1: Tensor*, *x2: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'maximum'*) 


元素最大值。
* **参数：**
   * **x1** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * x2 (Tensor)[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)第二个输入张量。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。



**示例**

```python
c = maximum(a, b)
```
## tvm.relax.frontend.nn.min(*x: Tensor*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'min'*) → Tensor 


计算给定轴上张量元素的最小值。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入数据张量。
   * axis (Optional[Union[int, List[int]]])**：** 指定执行最小值操作的轴或轴。默认值 axis=None 将对输入张量的所有元素执行最小值操作。支持负索引。
   * keepdims (bool)**：** 如果设置为 True，被缩减的轴将保留为结果中的尺寸为一的维度。使用此选项，结果将正确地广播与输入张量。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.minimum(*x1: Tensor*, *x2: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'minimum'*) 


元素最小值。
* **参数：**
   * **x1** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * x2 (Tensor)：第二个输入张量。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
c = minimum(a, b)
```
## tvm.relax.frontend.nn.multinomial_from_uniform(*prob: Tensor*, *uniform_sample: Tensor*, *sample_indices: Tensor |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'int64'*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'multinomial_from_uniform'*) 


返回一个张量，其中每一行包含从位于张量 prob 相应行的多项概率分布中采样的索引。

**注意**

为了获得更好的 CPU 性能，请使用「vm.builtin.multinomial_from_uniform」。为了获得准确的结果，请确保概率介于 0 到 1 之间，且总和为 1。
* **参数：**
   * **prob** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：一个二维张量，形状为 (batch, vocab_size)，表示概率分布。每一行代表一个批次中词汇表的分布，其中：值的范围为 [0, 1]，表示每个词汇表项目的概率。每行值的总和为 1，构成一个有效的分布。
   * uniform_sample (Tensor)：均匀采样的 2-D 张量，形状为(n, 1)。值范围从 0 到 1，表示均匀采样的概率。
   * sample_indices (Optional[Tensor])：可选的 2-D 张量，形状为[n, 1]，用于指示需要从中采样的特定概率分布。sample_indices[i]的值决定了第 i 个 token 应该从第 sample_indices[i]个概率分布中采样。例如，如果有 3 个不同的概率分布，并且需要从每个分布中采样 2、3 和 4 个 token，那么 sample_indices 将是[0, 0, 1, 1, 1, 2, 2, 2, 2]。
   * dtype (str)：输出张量的数据类型。
* **返回：result**：计算的形状为 (n, 1) 的张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
prob = [[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]]
usample = [[0.4], [0.9]]
sample_indices = [[0], [1]]

multinomial_from_uniform(prob, usample)
-> [[1], [2]]
multinomial_from_uniform(prob, usample, sample_indices)
-> [[1], [2]]
```
## tvm.relax.frontend.nn.multiply(*a: Tensor*, *b: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'mul'*) → Tensor 


使用 numpy 风格广播进行乘法。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * b (Tensor)：第二个输入张量。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
c = multiply(a, b)
```
## tvm.relax.frontend.nn.negative(*x: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'neg'*) → Tensor 


输入张量的数值负数。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * name (str)：名称提示。
   * result：计算结果。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.not_equal(*a: Tensor*, *b: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'not_equal'*) → Tensor 


广播逐元素比较 (lhs != rhs)。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * b (Tensor)：第二个输入张量。
   * name (str)**：** 名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.ones(*shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'float32'*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'ones'*) → Tensor 


构建一个全零的张量，具有输入形状和 dtype。
* **参数：**
   * **shape** (*Sequence*[*IntExpr]*)：创建的张量的形状。
   * dtype (str)：创建的张量的数据类型。
   * name (str)：名称提示。
* **返回：result**：结果张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.pad(*x: Tensor*, *pad:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *mode:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'constant'*, *value:*[float](https://docs.python.org/3/library/functions.html#float)*= 0.0*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'pad'*) → Tensor 


对输入张量应用空间填充。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：要填充的输入张量。
   * pad (List[int])：格式为 [before_0, after_0, before_1, after_1, …] 的列表，表示如何对 x 的每个轴进行填充。
   * mod (str)：使用的填充模式，constant 表示填充的元素将使用 value 参数的值。
   * value (float)：常数模式下用什么来填充。
   * name (str)：该算子的名称提示。
* **返回：result**：填充的输出张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.permute(*x: Tensor*, *axes:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None), *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'permute'*) → Tensor 


排列输入张量的维度。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * axes (Optional[List[int]])：目标轴顺序。
   * name (str)：名称提示。
* **返回：result**：转置的结果。
* 返回类型：[Tensor 张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.permute_dims(*x: Tensor*, *axes:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → Tensor 


排列数组的维度。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * axes (Optional[List[int]])：目标轴顺序，若未指定则逆序。
   * name (str)：名称提示。
* **返回：result**：转置的结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.prelu(*x: Tensor*, *alpha: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'prelu'*) 


参数 ReLU 激活函数。

$$\begin{split}\text{PReLU}(x) = \begin{cases}    x & \text{if } x \geq 0 \\    \alpha \cdot x & \text{if } x < 0\end{cases}\end{split}$$
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入数据。
   * alpha (Tensor)：输入负部分的斜率系数。
   * name (str, optional)：可选的操作名称。默认为“prelu”。
* **返回：result**：计算结果。
* 返回类型；[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.print*(*tensor: Tensor*) 


在运行时调试打印张量。

## tvm.relax.frontend.nn.relu(*x: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'relu'*) → Tensor 


整流线性单元 (ReLU) 激活函数。

$$ext{ReLU}(x) =  ext{max}(x, 0)$$
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))**：** 输入数据。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.relu6(*x: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'relu6'*) → Tensor 


ReLU6 激活函数。

$$\text{ReLU6}(x) = \min(\max(x, 0), 6)$$
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入数据。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型；[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.renormalize_top_p_top_k_prob(*prob*, *sorted_prob*, *top_p*, *top_k*) 


使用 top_p 和 top_k 过滤后重新规范化概率，确保它们的总和为 1。

:::Note

为了获得准确的结果，请确保概率介于 0 和 1 之间且总和为 1。

:::
* **参数：**
   * **prob** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：表示概率分布的形状为（batch，vocab_size）的二维张量。
   * sorted_prob (Tensor)[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)按降序排列的概率。
   * top_p (Tensor)：核心采样中使用的累积概率阈值，形状为 (batch, 1)。
   * top_k (Tensor)：形状为 (batch, 1) 的张量，表示用于 top-k 采样时考虑的顶级概率数量。
* **返回：result**：经过过滤和标准化的张量，以样本形状作为输入概率。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

### tvm.relax.frontend.nn.repeat(*x: Tensor*, *repeats:*[int](https://docs.python.org/3/library/functions.html#int), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *name='repeat'*) → Tensor 


重复数组的元素。
* **参数：**
   * **data** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
   * repeats (int)：重复次数。
   * axis (Optional[int])：重复值的轴。负数从后向前计数。默认情况下，使用展平的输入数组，并返回一个展平的输出数组。
   * name (str)：名称提示。
* **返回：ret**：计算结果。
* 返回类型：
* Tensor  张量。


**示例**

```python
np_x = numpy.array([[1, 2], [3, 4]])
x = Tensor.from_const(np_x)
lv1 = repeat(x, repeats=2) # lv1 == [1, 1, 2, 2, 3, 3, 4, 4]
lv2 = repeat(x, repeats=2, axis=1)   # lv2 == [[1., 1., 2., 2.],
                                     #         [3., 3., 4., 4.]]
```
## tvm.relax.frontend.nn.reshape(*x: Tensor*, *shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *name='reshape'*) → Tensor 


变形输入数组。


`-1`使用输入数组剩余维度推断输出形状的维度，保持新数组的大小与输入数组的大小相同。形状的维度最多可以为 -1。

>x.shape = (2, 3, 4), shape = (6, 1, -1), result.shape = (6, 1, 4)
>x.shape = (2, 3, 4), shape = (3, -1, 8), result.shape = (3, 1, 8)
>x.shape = (2, 3, 4), shape = (-1,), result.shape = (24,)
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * shape (Sequence[IntExpr])：新的形状。应与原始形状兼容。
   * name (str)：名称提示。
* **返回：result**：变形的结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

:::Note

推断`-1`仅在编译时进行。也就是说，如果`-1`在编译时无法推断出维度的长度，则会抛出错误。

:::

## tvm.relax.frontend.nn.rms_norm(*x: Tensor*, *weight: Tensor*, *axes:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *epsilon:*[float](https://docs.python.org/3/library/functions.html#float)*= 1e-05*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'rms_norm'*) → Tensor 


均方根归一化（Biao Zhang 等，2019）。对 n 维输入数组应用均方根归一化。该运算符接受一个 n 维输入数组，并使用给定的轴对输入进行归一化：

$$out = \frac{data}{\sqrt{mean(data, axis)+\epsilon}} * weight$$
* **参数：**
   * **data** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：将应用 rms_norm 的输入。
   * 权重 (Tensor)：缩放因子。
   * 轴 (Union[int, List[int]])：沿着这些轴应用归一化的轴。
   * epsilon (float)：在平方均值上添加的小浮点数，以避免除以零。
   * name (str)**：** 名称提示。
* **返回：result**：计算结果。
* 返回类型；[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.sample_top_p_top_k_from_sorted_prob(*sorted_prob: Tensor*, *sorted_index: Tensor*, *top_p: Tensor*, *top_k: Tensor*, *uniform_sample: Tensor*, *sample_indices: Tensor |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


根据 top_p 和 top_k 标准从排序的概率张量中抽样索引。

**注意**

为了获得准确的结果，请确保概率介于 0 和 1 之间且总和为 1。
* **参数：**
   * **sorted_prob** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：一个二维张量，形状为（batch，vocab_size），包含按降序排列的概率。
   * sorted_index (张量)：形状为 (batch, vocab_size) 的索引张量，对应于 sorted_prob。可能来自对原始概率张量按降序应用 argsort。
   * top_p (Tensor)：核心采样中使用的累积概率阈值，形状为 (batch, 1)。top_k (Tensor)：形状为 (batch, 1) 的张量，表示用于 top-k 采样时考虑的顶级概率数量。
   * uniform_sample (张量)：使用形状为 (n, 1) 的均匀采样值来选择输出索引。
   * sample_indices (Optional[Tensor])：可选的 2-D 张量，形状为[n, 1]，用于指示需要从中采样的特定概率分布。sample_indices[i]的值决定了第 i 个 token 应该从第 sample_indices[i]个概率分布中采样。例如，如果有 3 个不同的概率分布，并且需要从每个分布中采样 2、3 和 4 个 token，那么 sample_indices 将是[0, 0, 1, 1, 1, 2, 2, 2, 2]。
* **返回：result**：选定的形状为 (n, 1) 的索引。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
prob = [[0.1 , 0.4, 0.5],
        [0.3, 0.3, 0.4]]
sorted_prob = [[0.5, 0.4, 0.1],
               [0.4, 0.3, 0.3]]
sorted_index = [[2, 1, 0],
                [2, 0, 1]]
top_p = [[0.6],[0.9]]
top_k = [[3],[2]]
uniform_sample = [[0.5], [0.6]]
sample_indices = [[0], [1]]

    sorted_prob, sorted_index,top_p, top_k, uniform_sample, sample_indices)
-> [2, 0]
```
## tvm.relax.frontend.nn.scaled_dot_product_attention(*query: Tensor*, *key: Tensor*, *value: Tensor*, *attn_mask: Tensor |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *is_causal:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= False*, *scale:*[float](https://docs.python.org/3/library/functions.html#float)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'scaled_dot_product_attention'*) 


计算给定注意力查询、键和值的缩放点积注意力。与 torch 函数实现兼容。
* **参数：**
   * **query** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：表示当前形状为 [batch, seq_len, num_heads, head_size] 的注意力。
   * key (Tensor)：代表交叉注意力映射的张量，形状为 [batch, seq_len_kv, num_heads_kv, head_size]。查找的张量。
   * value (Tensor)：代表嵌入注意力值的张量，形状为 [batch, seq_len_kv, num_heads_kv, head_size_value]。
   * attn_mask (Optional[Tensor])：可选的注意力掩码，目前尚未支持。
   * is_causal (Optional[bool])：如果设置，则使用因果注意力掩码。
   * scale (Optional[float])：可选的额外缩放参数，应用于注意力。
   * name (str)：此函数的名称提示。

## tvm.relax.frontend.nn.sigmoid(*x: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'sigmoid'*) → Tensor 


计算 S 形函数。

$$\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}$$
* **参数：**
   * **data** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

:::note

输入张量需要具有浮点型。

:::

## tvm.relax.frontend.nn.silu(*x: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'silu'*) → Tensor 


Sigmoid 线性单元函数。

$$\text{SiLU}(x) = x * \text{sigmoid}(x)$$
* **参数：**
   * **data** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入数据。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：Tensor  张量。

:::Note

输入张量需要具有浮点型。

:::

## tvm.relax.frontend.nn.softmax(*x: Tensor*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'softmax'*) → Tensor 


计算 softmax。

$$\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
* **参数：**
   * **data** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * axis (int)：计算 softmax 时沿其进行求和的轴。如果未指定，默认为输入张量的最后一个轴。支持负索引。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

:::Note

输入张量需要具有浮点型。

:::

## tvm.relax.frontend.nn.softplus(*x: Tensor*, *beta:*[float](https://docs.python.org/3/library/functions.html#float)*= 1.0*, *threshold:*[float](https://docs.python.org/3/library/functions.html#float)*= 20.0*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'softplus'*) 


Softplus 激活函数。

$$\text{Softplus}(x) = \frac{1}{\beta} \log(1 + e^{\beta x})$$
* **参数：**
   * **data** (*relax.Expr*)：输入数据。
   * beta (float, 可选)[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)控制过渡的平滑度。默认值为 1.0。
   * 阈值（float，可选）超过该值后，函数被视为线性以避免数值不稳定性。默认为 20.0。
* **返回：result**：计算结果。
* 返回类型：relax.Expr。

## tvm.relax.frontend.nn.sort(*x: Tensor*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*, *descending:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *name='sort'*) 


沿给定轴执行排序并按排序顺序返回数组。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))[：](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)输入张量。
   * axis (int)：沿着哪个轴对输入张量进行排序。默认情况下使用输入的最后一个轴。
   * 降序（bool）是否按降序排序，默认为 False。
   * name (str)：名称提示。
* **返回：out**：排序后的张量。
* 返回类型：[Tensor 张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.split(*ary: Tensor*, *indices_or_sections:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'split'*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[Tensor, …] 


将数组拆分为多个子数组。
* **参数：**
   * **ary** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：要分割的输入张量。
   * indices_or_sections (Union[int, Sequence[int]])：用于分割的索引或片段。
   * axis (int = 0)：沿其分割的轴，默认为 0。
   * name (str)**：** 名称提示。
* **返回：result**：拆分结果的子数组列表。
* 返回类型：[Tuple](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)[[Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor), …]



## tvm.relax.frontend.nn.sqrt(*x: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'sqrt'*) → Tensor 


计算输入张量的逐元素平方根。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

:::Note

输入张量需要具有浮点型。


## tvm.relax.frontend.nn.square(*x: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'square'*) → Tensor 


计算输入张量的逐元素平方。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入张量。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.squeeze(*x: Tensor*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'squeeze'*) → Tensor 


挤压阵列中的轴。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * axis (可选[Union[int, List[int]])：要移除的轴集。如果 axis = None，则移除所有维度为 1 的轴。如果任何指定的轴的维度不等于 1，则会产生错误。
   * name (str)：名称提示。
* **返回：result**：压缩的结果。
* 返回类型：Tensor  张量。

## tvm.relax.frontend.nn.subtract(*a: Tensor*, *b: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'subtract'*) → Tensor 


使用 numpy 风格的广播进行减法。
* **参数：**
   * **a** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：第一个输入张量。
   * b (Tensor)：第二个输入张量。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor 张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
c = subtract(a, b)
```
## tvm.relax.frontend.nn.sum(*x: Tensor*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keepdims:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'sum'*) → Tensor 


计算给定轴上张量元素的总和。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))[：](https://pytorch.org/docs/stable/generated/torch.take.html)输入数据张量。
   * axis (Optional[Union[int, List[int]]])：指定进行求和的轴或轴。默认值 axis=None 将对输入张量的所有元素求和。支持负索引。
   * keepdims (bool)[：](https://pytorch.org/docs/stable/generated/torch.take.html)如果设置为 True，被缩减的轴将保留为结果中的尺寸为一的维度。使用此选项，结果将正确地广播与输入张量。
   * name (str)[：](https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-13)该操作的名称提示。
* **返回：result**：计算结果。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。



## tvm.relax.frontend.nn.take(*x: Tensor*, *indices: Tensor*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *name='take'*) → Tensor 


沿某个轴从张量中获取元素。其语义与 numpy.take ( [https://numpy.org/doc/stable/reference/generated/numpy.take.html](https://numpy.org/doc/stable/reference/generated/numpy.take.html) )非常相似，后者可以覆盖 torch.take ( [https://pytorch.org/docs/stable/generated/torch.take.html](https://pytorch.org/docs/stable/generated/torch.take.html) ) 和 onnx.gather ( [https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-13](https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-13) ) 。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：源张量。
   * indices (Tensor)：要提取的值的索引。
   * axis (可选[int])：指定选择值的轴。如果为 none，输入张量必须是一维的。
   * name (str)：名称提示。
* **返回：ret**：获取的结果。
* 返回类型：[Tensor 张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.tanh(*x: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'tanh'*) → Tensor 


应用双曲正切函数。

$$\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：运算符的输入数据。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：Tensor  张量。

:::Note

输入张量需要具有浮点型。

:::

## tvm.relax.frontend.nn.tensor_expr_op(*tensor_expr_func:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable), *name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str), *args:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[Tensor |*[Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*|*[int](https://docs.python.org/3/library/functions.html#int)*]*, *, *attrs: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None*) 


使用 te 构建给定的 tensor_expr_func。
* **参数：**
   * **tensor_expr_func** (*Callable*)：返回 te 张量或张量列表的函数。
   * name_hint (str)：名称提示。
   * args (List[Union[Tensor, _tir.Var]])：传递给函数的参数。
   * attrs (Optional[Dict[str, Any]])：应用于函数的属性字典。
* **返回：result**：结果张量。
* 返回类型：Tensor  张量。

## tvm.relax.frontend.nn.tensor_ir_inplace_op(*func:*[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone), *name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str), *args: Tensor |*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[Tensor |*[ShapeExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxshapeexprvalueslistprimexprtupleprimexprarrayspanspannonenone)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *inplace_indices:*[int](https://docs.python.org/3/library/functions.html#int)*|*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *out: OutType*) → OutType 


使用给定的 PrimFunc 创建 call_tir_inplace 绑定
* **参数：**
   * **func** (tir.PrimFunc*)[：](https://docs.python.org/3/library/typing.html#typing.Sequence)要调用的 PrimFunc。
   * name_hint (str)：名称提示。
   * args (Union[Tensor, Sequence[Union[Tensor, rx.ShapeExpr, _tir.PrimExpr]]])*：*传递给 PrimFunc 的参数。
   * inplace_indices (Union[int, List[int]])：指定哪些参数应用于原地计算。如果 inplace_indices 是一个整数，它将被转换为一个单元素列表。假设 inplace_indices[i] = j，其中 j >= 0。那么第 i 个输出将是 args[j] 的别名。如果 inplace_indices[i] = -1，那么第 i 个输出将是一个新分配的张量。inplace_indices 至少有一个成员不能是 -1。
   * out (Union[Tensor, List[Tensor]])：输出张量。
* **返回：result**：结果张量。
* 返回类型：Tensor  张量。

## tvm.relax.frontend.nn.tensor_ir_op(*func:*[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone), *name_hint:*[str](https://docs.python.org/3/library/stdtypes.html#str), *args: Tensor |*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[Tensor |*[ShapeExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-relax#classtvmrelaxshapeexprvalueslistprimexprtupleprimexprarrayspanspannonenone)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *out: OutType*) → OutType 


使用给定的 PrimFunc 创建 call_tir 绑定。
* **参数：**
   * **func** (tir.PrimFunc*)：要调用的 PrimFunc。
   * name_hint (str)：名称提示。
   * args (Union[Tensor, Sequence[Union[Tensor, rx.ShapeExpr, _tir.PrimExpr]]])：传递给 PrimFunc 的参数。
   * out (Union[Tensor, List[Tensor]])：输出张量。
* **返回：result**：结果张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.topk(*data: Tensor*, *k:*[int](https://docs.python.org/3/library/functions.html#int)*= 1*, *axis:*[int](https://docs.python.org/3/library/functions.html#int)*= -1*, *ret_type:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'both'*, *largest:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'int32'*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'topk'*) 


获取输入张量中沿给定轴的前 k 个元素。

ret_type specifies the return type, can be one of (“both”, “values”, “indices”).
* **参数：**
   * **data** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入数据张量。
   * k (int)**：** 选择顶部元素的数量。如果 k < 1，则返回所有元素。
   * axis (int)：按此轴对输入张量进行排序。
   * ret_type (str)：返回类型 [both, values, indices]。“both”：同时返回顶部 k 数据和索引。“values”：仅返回顶部 k 数据。“indices”：仅返回顶部 k 索引。
   * largest (bool)：是否返回最大或最小的元素。如果 largest 为 False，则返回 k 个最小的元素。
   * dtype (str)：索引输出的数据类型。
   * name (str)：名称提示。
* **返回：out**：计算结果。
* 返回类型：张量或元组[张量, 张量]。

## tvm.relax.frontend.nn.triu(*x: Tensor*, *diagonal:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'triu'*) → Tensor 


返回矩阵或一批矩阵的上三角部分。
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))**：** 将应用 triu 的张量。它至少需要两个维度。
   * k (int)：指示要置零元素的 diagonals 以下索引。如果 k = 0，对角线是主对角线。如果 k < 0，对角线在主对角线下方。如果 k > 0，对角线在主对角线上方。
   * name (str)：名称提示。
* **返回：ret**：结果张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


## tvm.relax.frontend.nn.unsqueeze(*x: Tensor*, *dim:*[int](https://docs.python.org/3/library/functions.html#int), *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'unsqueeze'*) → Tensor 


向张量添加新轴
* **参数：**
   * **x** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：要扩展的输入张量。
   * dim (int)：要扩展的维度。
   * name (str)：该算子的名称提示。
* **返回：result**：扩展结果。
* 返回类型：Tensor  张量。

## tvm.relax.frontend.nn.where(*condition: Tensor*, *x1: Tensor*, *x2: Tensor*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'where'*) → Tensor 


根据条件的值从输入张量中选择元素。


对于给定的位置，如果条件为 True，则返回 x1中的对应值，否则返回 x2中的对应值。
* **参数：**
   * **condition** ([Tensor](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：为 True 时，输出 x1；否则，输出 x2。必须与 x1和 x2广播兼容。必须为布尔类型。
   * x1 (Tensor)：第一个输入张量。必须与 condition 和 x2 兼容广播。
   * x2 (Tensor)：第二个输入张量。必须与 condition 和 x1 兼容广播。
   * name (str)：名称提示。
* **返回：result**：结果张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.relax.frontend.nn.wrap_nested(*expr:*[RelaxExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirrelaxexpr), *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)) → Tensor | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[Tensor] 


包装给定的 relax.Expr，使用当前的 BlockBuilder 发出它，并且如果 expr 代表 Tuple 则自动处理嵌套情况。
* **参数：**
   * **expr** (*relax.Expr*)：要包装的 Expr。
   * name (str)：名称提示。
* **返回：result**：计算结果。
* 返回类型：Union[Tensor, Tuple[Tensor]]。

## tvm.relax.frontend.nn.zeros(*shape:*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*|*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'float32'*, *name:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'zeros'*) → Tensor 

构建一个全零的张量，具有输入形状和 dtype。
* **参数：**
   * **shape** (*Sequence[*IntExpr]*)：创建的张量的形状。
   * dtype (str)：创建的张量的数据类型。
   * name (str)**：** 名称提示。
* **返回：result** ：结果张量。
* 返回类型：[Tensor  张量](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## 


## tvm.relax.frontend.onnx


将 ONNX 图转换为 Relax 图的工具。

## tvm.relax.frontend.onnx.from_onnx(*model: GraphProto*, *shape_dict:*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[List](https://docs.python.org/3/library/typing.html#typing.List)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *dtype_dict:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*,*[str](https://docs.python.org/3/library/stdtypes.html#str)*] |*[None](https://docs.python.org/3/library/constants.html#None)*= 'float32'*, *opset:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *keep_params_in_input:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*, *sanitize_input_names:*[bool](https://docs.python.org/3/library/functions.html#bool)*= True*) → [IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)


将 ONNX 模型转换为等效的 Relax 函数。ONNX 图表示为 Python Protobuf 对象。


当前实现假设输入模型在 ONNX v1.1.0 之后。
* **参数：**
   * **model** (*protobuf object*)**：** ONNX v1.1.0 之后的 ONNX ModelProto。
   * shape_dict (str 到元组的字典，可选)：图的输入形状。
   * dtype_dict (字符串或 str 到字符串的字典，可选)：图的输入类型。
   * opset (整数，可选)**：** 覆盖自动检测的操作集。这在某些测试中很有帮助。
   * keep_params_in_input (bool)：如果为 True，参数将被视为输入变量。如果为 False，参数将被视为常量并直接折叠到图中。
   * sanitize_input_names (bool, optional)：是否对输入名称进行清理以确保它们是有效的 Relax 标识符。
* **返回：mod**：用于编译的 relax 模块。
* 返回类型：tvm.IRModule。



## tvm.relax.frontend.stablehlo


用于构建 Relax 程序的 StableHLO 前端，包含模型导入器。

## tvm.relax.frontend.stablehlo.from_stablehlo(*stablehlo_module*, *input_info:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*],*[str](https://docs.python.org/3/library/stdtypes.html#str)*]] |*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)


将 StableHLO 模块转换为 Relax 程序
* **参数：stablehlo_module** (*Union*[***[str](https://docs.python.org/3/library/stdtypes.html#str)*** ***mlir.ir.Module***])：要转换的 StableHLO 模块。
* **返回：output**：结果 IRModule 的入口函数为“main”。

## 


## tvm.relax.frontend.torch

用于构建 Relax 程序的 PyTorch 前端，带有模型导入器。

## tvm.relax.frontend.torch.from_exported_program(*exported_program: ExportedProgram*, *, *keep_params_as_input: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *unwrap_unit_return_tuple: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *no_bind_return_tuple: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)


将 PyTorch ExportedProgram 转换为 Relax 程序。
* **参数：**
   * **exported_program** (*torch.export.ExportedProgram*)：要转换的 PyTorch ExportedProgram。
   * keep_params_as_input (bool)：是否将模型参数保留为输入变量。
   * unwrap_unit_return_tuple (bool)：当返回值是单元元组时，是否对返回值进行解包。当返回值不是单元元组时，不会进行解包。
   * no_bind_return_tuple (bool)：一个布尔标志，指示是否将返回元组绑定为一个 relax 变量。如果标志为 true 且返回值是一个元组，它将不会将其绑定到一个变量。
* **返回：output**：导入结果 IRModule，其中函数“main”包含翻译后的逻辑。
* 返回类型：tvm.IRModule


**示例**


用户可以使用 torch.export.export() 从 PyTorch 模型中提取一个 torch.export.ExportedProgram。以下代码展示了如何将 PyTorch 模型转换为 Relax 程序。


```python
# 导入导入器
import tvm
from tvm.relax.frontend.torch import from_exported_program
import torch
from torch.export import export

# 定义模块
class MyModule(torch.nn.Module):
    def **init**(self):
        super().**init**()

    def forward(self, input):
        return self.linear(input)

# 实例化模型并创建输入信息字典
torch_model = MyModule()


# 使用 torch.export.export() 转换 PyTorch 模型为 ExportedProgram
example_args = (torch.rand(128, 10, dtype=torch.float32),)
exported_program = export(torch_model, args=example_args)

# 使用导入器导入 ExportedProgram 到 Relax。
mod: tvm.IRModule = from_exported_program(exported_program)
```
## tvm.relax.frontend.torch.from_fx(*model*, *input_info:*[List](https://docs.python.org/3/library/typing.html#typing.List)*[*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[int](https://docs.python.org/3/library/functions.html#int)*],*[str](https://docs.python.org/3/library/stdtypes.html#str)*]]*, *, *keep_params_as_input: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *unwrap_unit_return_tuple: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *no_bind_return_tuple: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *custom_convert_map: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None*) → [IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)


将 PyTorch FX GraphModule 转换为 Relax 程序。
* **参数：**
   * **model** (*fx.GraphModule*)：要转换的 PyTorch FX GraphModule。
   * input_info (List[Tuple[Tuple[int], str]])：输入张量的形状和数据类型的列表。
   * keep_params_as_input (bool)：是否将模型参数保留为输入变量。
   * unwrap_unit_return_tuple (bool)：当返回值是单元元组时，是否对返回值进行解包。当返回值不是单元元组时，不会进行解包。
   * no_bind_return_tuple (bool)：一个布尔标志，指示是否将返回元组绑定为一个 relax 变量。如果标志为 true 且返回值是一个元组，它将不会将其绑定到一个变量。
   * custom_convert_map (str 到 Relax 操作的字典)：与TorchFXImporter.convert_map 相同格式的自定义操作转换映射。
* **返回：output**：导入结果 IRModule，其中函数“main”包含转换后的逻辑。如果 keep_params_as_input 为 true，则“main”函数将包含一个属性“params”，该属性包含输入模型的权重。这些权重可以通过 Relax.frontend.detach_params 分离。
* 返回类型；tvm.IRModule。


**示例**


用户可以使用 FX 追踪器或 dynamo.export() 从 PyTorch 模型中提取 fx.GraphModule。以下代码演示如何将 PyTorch 模型转换为 Relax 程序。


```python
# 导入导入器
import numpy as np
from tvm.relax.frontend.torch_fx import from_fx

# 定义模块
class MyModule(torch.nn.Module):
    def **init**(self):
        super().**init**()

    def forward(self, input):
        return self.linear(input)

# 实例化模型并创建输入信息字典
torch_model = MyModule()
input_info = [((128, 10), "float32")]
    torch.astensor(np.random.randn(*shape).astype(dtype))
    for shape, dtype in input_info
]

# 使用 FX 追踪器追踪 PyTorch 模型
graph_module = fx.symbolic_trace(torch_model)

# 使用 dynamo.export() 导入 PyTorch 模型到 FX
try:
    graph_module = dynamo.export(torch_model, *input_tensors)
except:
    raise RuntimeError("Failed to export the PyTorch model to FX.")

# 使用导入器导入 PyTorch 模型到 Relax
mod: tvm.IRModule = from_fx(graph_module, input_info)

# 输出导入的模型
print(mod.script())
```


对于给定的 PyTorch 模型，要在 FX 中查找模型输入的名称，可以使用。



```python
fx.symbolic_trace(model).graph.print_tabular()
```


打印出 PyTorch 模块的表格表示，然后检查表格开头的占位符行。

## tvm.relax.frontend.torch.relax_dynamo(*pipeline:*[Pass](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*)

用于创建 Relax 后端的辅助函数。
* **参数：pipeline** (*Optional[*[tvm.transform.Pass](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)*]*)：在发送构建之前要应用于 Relax 模块的管道。
* **返回：backend**：Relax Dynamo 后端。
* 返回类型：Callable[[torch.fx.GraphModule, List[torch.Tensor]], Callable]。

## tvm.relax.frontend.torch.dynamo_capture_subgraphs(*model*, params*, ***kwargs*) → [IRModule](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirirmodulefunctionsnone-attrsnone-global_infosnone)


使用 torch.compile 将 PyTorch 模型的子图捕获到 IRModule 中。
* **参数：**
   * **model** (*torch.nn.Module*)：要捕获的 PyTorch 模型。
   * params (List[torch.Tensor])：PyTorch 模型的参数。
   * keep_params_as_input (bool)：是否将模型参数保留为捕获的 Relax 函数的输入变量。
* **返回：output**：翻译的输出，包含已翻译的 IRModule。如果 keep_params_as_input 为 true，则 IRModule 中的函数将具有一个属性「params」，其中包含输入模型的权重。这些权重可以通过 Relax.frontend.detach_params 分离。
* 返回类型：ImporterOutput。


# 

