---

title: tvm.instrument

---



IR 变体的通用 pass 工具。

## *class*tvm.instrument.PassInstrument


pass 工具的实现。


要使用 PassInstrument，用户类可以直接从 PassInstrument 继承子类，也可以应用`pass_instrument()`包装器。无论哪种情况，都可以定义 enter_pass_ctx、exit_pass_ctx、should_run、 run_before_pass 和 run_after_pass 方法来调整工具的行为。有关每个方法的更多信息，请参阅此类定义中的无操作实现。

### enter_pass_ctx()


进入检测上下文时调用。
* **返回类型：** 无。

### exit_pass_ctx()


退出检测上下文时调用。
* **返回类型：** 无。

### should_run(*mod*, *info*)

确定是否运行 pass。


当检测上下文处于活动状态时，每次运行都会调用一次。
* **参数：**
   * **mod** ([tvm.ir.module.IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule))  – 正在运行优化过程的模块。
   * **info** ([tvm.transform.PassInfo](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.PassInfo))  –  Pass 信息。
* **返回：should_run**  – True 表示运行 pass ，False 表示跳过 pass 。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

### run_before_pass(*mod*, *info*)


pass 运行前的工具。


当检测上下文处于活动状态时，每次运行 pass 都会调用一次。
* **参数：**
   * **mod** ([tvm.ir.module.IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule))  – 正在运行优化过程的模块。
   * **info** ([tvm.transform.PassInfo](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.PassInfo))  –  Pass 信息。
* **返回类型：** 无。

### run_after_pass(*mod*, info)


pass 后的工具运行。


当检测上下文处于活动状态时，每次运行都会调用一次。
* **参数：**
  * **mod** ([tvm.ir.module.IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule))  – 正在运行优化过程的模块。
   * **info** ([tvm.transform.PassInfo](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.PassInfo))  – [ Pass ](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.PassInfo)信息。
* **返回类型：** 无。

## *class* tvm.instrument.PassPrintingInstrument(*print_before_pass_names*, *print_after_pass_names*)


一个 pass 工具，用于在制定 pass 的每个步骤前后打印 IR。

## *class*tvm.instrument.PassTimingInstrument

用 C++ 实现的用于创建 pass 时间工具的包装器

### *static* render()

检索渲染的时间配置文件结果：**返回:** string – 时间分析结果的渲染字符串：返回类型：string


**示例**

```plain
timing_inst = PassTimingInstrument()
with tvm.transform.PassContext(instruments=[timing_inst]):
    relax_mod = relax.transform.FuseOps()(relax_mod)
    # 在退出前得到分析结果
    profiles = timing_inst.render()
```
## *class*tvm.instrument.PrintAfterAll(*args, ****kwargs*)


仅在 Pass 执行后才打印 Pass 的名称、IR。

## *class*tvm.instrument.PrintBeforeAll(*args, ***kwargs)


仅在执行 pass 之前打印 pass 的名称、IR。

## tvm.instrument.pass_instrument(*pi_cls=None*)


装饰 pass 工具。
* **参数：pi_class** (*class*) – 工具类，见下例。


**示例**

```plain
@tvm.instrument.pass_instrument
class SkipPass:
    def __init__(self, skip_pass_name):
        self.skip_pass_name = skip_pass_name

    # 取消注释以实现自定义。
    # def enter_pass_ctx(self):
    #    pass

    # 取消注释以实现自定义。
    # def exit_pass_ctx(self):
    #    pass

    # 如果Pass名称包含关键词则跳过（返回False），否则不跳过（返回True）。
    def should_run(self, mod, pass_info)
        if self.skip_pass_name in pass_info.name:
            return False
        return True

    # 取消注释以实现自定义。
    # def run_before_pass(self, mod, pass_info):
    #    pass

    # 取消注释以实现自定义。
    # def run_after_pass(self, mod, pass_info):
    #    pass

skip_annotate = SkipPass("AnnotateSpans")
with tvm.transform.PassContext(instruments=[skip_annotate]):
    tvm.compile(mod, "llvm")
```


