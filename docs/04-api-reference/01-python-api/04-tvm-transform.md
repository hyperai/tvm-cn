---

title: tvm.transform

---




跨 IR 变体的通用基础设施。

## tvm.transform.ApplyPassToFunction(*transform:*[Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass), *func_name_regex:*[str](https://docs.python.org/3/library/stdtypes.html#str), *error_if_no_function_matches_regex:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*) → [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)


将 pass 应用于 IRModule 中特定函数的实用工具。

TVM 在降阶的所有阶段都使用 IRModule 到 IRModule 的转换。这些转换在手动编写优化模型或对 IRModule 中的特定内核进行优化时非常有用。此实用工具允许将 pass 应用于指定函数，而不改变模块中的其他函数。
* **参数:**
   * **transform**([Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)) – 要应用的 IRModule 到 IRModule pass。
   * **func_name_regex** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – 用于选择要更新函数的正则表达式。pass 将应用于所有名称匹配该正则表达式的函数。
   * **error_if_no_function_matches_regex** ([bool](https://docs.python.org/3/library/functions.html#bool)) – 指定当 IRModule 不包含任何匹配正则表达式的函数时的行为。如果为 true，将引发错误；如果为 false（默认值），则返回未修改的 IRModule。
* **返回: new_transform** – 修改后的 IRModule 到 IRModule pass。
* **返回类型:** [Pass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)。


## *class* tvm.transform.ModulePass


一个作用于 tvm.IRModule 的 pass。用户无需直接与此类交互，而应通过 module_pass 创建模块 pass，因为 module_pass API 的设计足够灵活，可以处理不同方式的模块 pass 创建。此外，模块 pass 的所有成员都可以从基类访问。此规则同样适用于 FunctionPass。

## *class* tvm.transform.Pass


所有 pass 的基类。此处所有方法只是后端实现的简单封装，定义这些方法是为了方便用户与基类交互。

### *property* info


获取 pass 元信息。

## *classtvm.transform.PassContext(opt_level=2, required_pass=None, disabled_pass=None, instruments=None, config=None)*



TVM 优化/分析运行的基础环境。每个 pass 上下文包含用于辅助优化 pass 的若干辅助信息，例如记录优化过程中错误的错误报告器等。


**opt_level：Optional[int]**

此 pass 的优化级别。


**required_pass：Optional[Union[List[str], Set[str], Tuple[str]]]**

特定 pass 所需的 pass 列表。


**disabled_pass：Optional[Union[List[str], Set[str], Tuple[str]]]**

被禁用的 pass 列表。


**instruments：Optional[Sequence[PassInstrument]]**

pass 检测工具的实现列表。


**config：Optional[Dict[str, Object]]**

特定 pass 的额外配置。


## override_instruments(*instruments*)


覆盖此 PassContext 中的检测工具。

如果存在现有检测工具，将调用其 `exit*pass*ctx` 回调函数。然后切换到新检测工具并调用新的 `enter*pass*ctx` 回调函数。


**instruments：Sequence[PassInstrument]**

pass 检测工具的实现列表。


### *static* current()

返回当前 pass 上下文。


### *static* list_configs()

列出所有已注册的 PassContext 配置名称和元数据。
* **返回: configs。**
* **返回类型:** Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), Dict[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]]。


## *class* tvm.transform.PassInfo(*opt_level*, *name*, *required=None*, *traceable=False*)

包含 pass 所需元数据的类。它是运行优化或分析所需信息的容器，当需要更多元数据时可以通过添加新成员来扩展此类。
* **参数:**
   * **opt_level**([int](https://docs.python.org/3/library/functions.html#int)) – 此 pass 的优化级别。
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – pass 名称。
   * **required** (List[[str](https://docs.python.org/3/library/stdtypes.html#str)]) – 特定 pass 所需的 pass 列表。


## tvm.transform.PrintIR(*header=''*, *show_meta_data=False*)

打印标题和 IR 的特殊跟踪 pass。
* **参数:**
   * **header** ([_str_](https://docs.python.org/3/library/stdtypes.html#str)) – 与 dump 内容一起显示的标题。
   * **show_meta_data** ([_bool_](https://docs.python.org/3/library/functions.html#bool)) – 是否打印元数据的布尔标志。
* **返回类型:**  pass。

## *class* tvm.transform.Sequential(*passes=None*, *opt_level=0*, *name='sequential'*, *required=None*, *traceable=False*)

一个作用于一系列 pass 对象的 pass。使用此类可以顺序执行多个 pass。


注意用户也可以提供一系列在运行顺序 pass 时不希望应用的 pass。pass 依赖关系也将在后端解析。
* **参数:**
   * **passes** (_Optional**[**List[_[_Pass_](/docs/api-reference/python-api/tvm-transform#class-tvmtransformpass)]]) – 待优化的 pass 候选序列。
   * **opt_level** (_Optional[_[_int_](https://docs.python.org/3/library/functions.html#int)_]_) – 此顺序 pass 的优化级别。默认顺序 pass 的 opt_level 设置为 0。注意如果 Sequantial 中某些 pass 的 opt_level 高于提供的 opt_level，它们可能仍不会被执行。
   * **name** (_Optional[_[_str_](https://docs.python.org/3/library/stdtypes.html#str)_]_) – 顺序 pass 的名称。
   * **required** (_Optional**[**List[_[_str_](https://docs.python.org/3/library/stdtypes.html#str)]]) – 顺序 pass 所依赖的 pass 列表。


## tvm.transform.module_pass(*pass_func=None*, *opt_level=None*, *name=None*, *required=None*, *traceable=False*)

装饰一个模块 pass。

当提供 pass_func 时，此函数返回回调函数；否则作为装饰器函数使用。

pass_func 也可以是具有 transform_module 方法的类类型。此函数将使用 transform_module 作为 pass 函数创建装饰后的 ModulePass。
* **参数:** 
   * **pass_func** (_Optional**[**Callable**[**(**Module**,_ [_PassContext_](/docs/api-reference/python-api/tvm-transform#classtvmtransformpasscontextopt_level2required_passnonedisabled_passnoneinstrumentsnoneconfignone)_)_ _->Module**]**]_) – 转换函数或类。
   * **opt_level** ([_int_](https://docs.python.org/3/library/functions.html#int)) – 此模块 pass 的优化级别。
   * **name** (_Optional[_[_str_](https://docs.python.org/3/library/stdtypes.html#str)_]_) – 模块 pass 名称。名称可为空，此时将使用优化函数的名称作为 pass 名称。
   * **required** (_Optional**[**List[_[_str_](https://docs.python.org/3/library/stdtypes.html#str)]]) – 模块 pass 所依赖的 pass 列表。
   * **traceable** (_Boolean_) – 模块 pass 是否可跟踪的布尔值。
* **返回:** **create_module_pass** – 如果未提供 pass_func 则返回装饰器，否则返回装饰后的结果。返回的装饰器根据输入有两种行为：当装饰 pass 函数时返回新的 ModulePass；当装饰类类型时返回新的 ModulePass 类。
* **返回类型:**  Union[Callable, [ModulePass](/docs/api-reference/python-api/tvm-transform#class-tvmtransformmodulepass)]。


**示例**

以下代码块装饰模块 pass 类：

```python
@tvm.ir.transform.module_pass
class CustomPipeline:
    def __init__(self, enable_fold):
        self.enable_fold = enable_fold
        self.const_fold = relax.transform.FoldConstant()

    def transform_module(self, mod, ctx):
        if self.enable_fold:
            mod = self.const_fold(mod, ctx)
        return mod

# 创建自定义流水线实例
pipeline = CustomPipeline(enable_fold=False)
assert isinstance(pipeline, transform.ModulePass)
# 运行流水线
output_module = pipeline(input_module)
```
以下代码通过装饰用户定义的转换函数来创建一个模块 pass。
```python
@tvm.ir.transform.module_pass(opt_level=2)
def transform(mod, ctx):
    return relax.transform.FoldConstant(mod)

module_pass = transform
assert isinstance(module_pass, transform.ModulePass)
assert module_pass.info.opt_level == 2

# 给定模块 m，可按以下方式调用优化：
updated_mod = module_pass(m)
# 现在函数 abs 应被添加到模块 m 中
```

# 

