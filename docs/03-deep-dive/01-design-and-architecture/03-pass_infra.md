---

title: Pass 基础设施

---

Relax 与 TVM IR 都包含一系列优化传递（optimization
passes），用于改进模型在特定设备上的性能指标，例如推理平均时间、内存占用或功耗。这些优化包括标准优化与机器学习特定优化，如常量折叠（constant folding）、死代码消除、算子布局变换、算子融合、缓冲区处理和循环变换等。每个传递都是基于收集的分析结果进行的 IR-to-IR 转换。

然而，随着 TVM 的快速发展，越来越需要一种系统化且高效的方式来管理这些传递。此外，一个通用的框架能够在 TVM 栈的不同层次（例如 Relax 和 tir）之间管理传递，这为开发者快速原型化和集成新传递铺平了道路。

本文档介绍了这种基础设施的设计，它结合了生产级编译器中用于管理优化传递的方式，以及现代深度学习框架用于构建层次化结构的风格。

例如，许多现有的生产级编译器（如 GCC 与 LLVM） 采用「传递管理器（pass manager）」来高效管理传递执行。最初传递数量较少时管理很简单，但成熟编译器可能包含数百个独立传递。外部用户往往希望添加自定义传递，并能正确调度，而无需手动修改固定顺序。

类似地，现代深度学习框架（如 Pytorch 与 MXNet Gluon）也倾向于通过[Sequential](https://pytorch.org/docs/stable/nn.html?highlight=sequential#torch.nn.Sequential)和[Block](https://mxnet.apache.org/api/python/docs/api/gluon/block.html#gluon-block)实现类似「传递式」层构建机制。
借助这些构造，框架能够轻松将模块或层添加到容器中，从而快速搭建神经网络。

TVM 的传递基础设施设计灵感主要来自 LLVM 的层次化传递管理器
以及流行深度学习框架的模块化容器。 该系统的主要目标包括：

1)  支持更灵活的优化编排，让用户能自由构建自定义优化流水线。

2)  提供便捷的调试机制。

3)  让开发者无需手动解决传递之间的依赖。

4)  简化新传递的实现方式，例如允许用户直接用 Python
    实现一个传递，由系统自动管理其执行。

## 设计概述

系统重点关注可扩展性，使用户能快速添加新传递而不破坏兼容性。
其结构包括后端与前端：后端实现核心逻辑，前端则提供简单的 API
供用户创建与控制优化流程。

### C++ 后端

我们提供 `PassInfo`对象来存储单个传递所需的基本信息：`name`为传递名，`opt_level`指示该传递在哪个优化级别启用，`required`表示执行该传递前所需的其他传递（详见[include/tvm/ir/transform.h](https://github.com/apache/tvm/blob/main/include/tvm/ir/transform.h)）。
在注册传递时，开发者可以指定传递名称、优化级别与依赖。 `opt_level`可帮助系统在给定优化级别下判断某个传递是否需要执行； `required`字段用于自动解析传递依赖。

``` c++
class PassInfoNode : public Object {
  ffi::String name;
  int opt_level;
  ffi::Array<ffi::String> required;
};
```

#### PassContext

`PassContext` 携带优化传递所需的关键信息。例如，它包含错误报告系统，方便优化作者诊断失败原因。 `PassContext`也取代了旧的
`BuildConfig`（用于配置编译选项，如优化级别、必需/禁用传递等）。例如，我们可以配置在 `opt_level=3` 下执行所有传递，并通过`disabled_pass=xx` 禁用某些传递；系统会聚合该级别的所有传递并排除被禁用的项。`PassContext`还提供对所有传递进行"检测（instrumentation）"的能力，见 `pass_instrument_cpp_backend`。

该类支持 Python `with` 语法，便于在给定配置下执行优化。
同时，用户可以通过 `PassContext::Current()`在线程安全的方式获取当前上下文， 因为系统使用线程本地存储`PassContextThreadLocalStore` 来保存上下文对象。

``` c++
class PassContextNode : public Object {
 public:
  int opt_level{2};
  tvm::ffi::Array<tvm::Expr> required_pass;
  tvm::ffi::Array<tvm::Expr> disabled_pass;
  mutable ffi::Optional<DiagnosticContext> diag_ctx;
  ffi::Map<ffi::String, Any> config;
  ffi::Array<instrument::PassInstrument> instruments;
};

class PassContext : public NodeRef {
 public:
  TVM_DLL static PassContext Create();
  TVM_DLL static PassContext Current();
  TVM_DLL void InstrumentEnterPassContext();
  TVM_DLL void InstrumentExitPassContext();
  TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
  TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;
  /* 其他字段省略 */

 private:
  // 进入 pass 上下文作用域
  TVM_DLL void EnterWithScope();
  // 离开 pass 上下文作用域
  TVM_DLL void ExitWithScope();

  // 用于支持 Python `with` 语法
  friend class tvm::With<PassContext>;
};

struct PassContextThreadLocalEntry {
  /*! rief 默认 pass 上下文 */
  PassContext default_context;
  /*! rief 当前 pass 上下文 */
  std::stack<PassContext> context_stack;
  PassContextThreadLocalEntry() {
    default_context = PassContext(make_node<PassContextNode>());
  }
};

/*! rief 线程本地存储，用于保存 pass 上下文 */
typedef dmlc::ThreadLocalStore<PassContextThreadLocalEntry>
     PassContextThreadLocalStore;
```

#### Pass 构造

传递（Pass）基础设施以分层结构设计，可在 Relax/tir
程序的不同粒度上工作。 系统定义了一个纯虚类`PassNode`，作为各种优化传递的基类。此类包含多个必须在子类中实现的虚函数，适用于模块级、函数级或顺序传递级别。


``` c++
class PassNode : Object {
  virtual PassInfo Info() const = 0;
  virtual Module operator()(const IRModule& mod,
                            const PassContext& pass_ctx) const = 0;
};
```

该函数对象定义了传递的执行方式： 每个传递都在特定上下文 `PassContext`下作用于一个 `IRModule`， 并以 `Module` 到 `Module` 的方式实现。因此，所有传递都以模块为单位更新整个 IR。

系统实现了多个 `PassNode` 子类来支持不同类型的优化：
包括函数级传递、模块级传递与顺序传递（sequential pass）。
每个子类本身都可充当一个传递管理器，例如：它们可以收集所需传递并执行，或基于元信息建立依赖图。完整定义见[src/ir/transform.cc](https://github.com/apache/tvm/blob/main/src/ir/transform.cc)。

#### 模块级传递

模块级传递主要用于全局或过程间优化（IPO），类似于 LLVM 中的模块传递。Relax 中一些典型需要全局视图的优化（如 A-normal form 转换、lambda 提升）就属于此类。 在该级别，用户可以在模块中添加或删除函数。

``` c++
class ModulePassNode : PassNode {
  PassInfo pass_info;
  std::function<Module(Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
  // 其他成员/方法省略
};
```

`pass_info` 存储模块传递的相关信息，`pass_func` 定义实际优化逻辑。例如，在模块上执行死代码消除可在 `pass_func` 中实现，它将删除模块中未使用的函数。 此字段被设计为「打包函数（packed function）」， 因此优化逻辑既可用 C++ 实现，也可用 Python 实现。

### 函数级传递

函数级传递用于实现 Relax/tir 模块中函数内的优化。它一次提取模块中的一个函数进行优化，输出优化后的 Relax `Function` 或 tir `PrimFunc`。多数优化都属于此类，如 Relax 的公共子表达式消除、推理简化，或 tir 的向量化与内存扁平化。

函数级传递仅作用于单个函数（Relax 或 tir），因此无法通过此类传递添加或删除函数，因为其不具备全局信息。

``` c++
class FunctionPassNode : PassNode {
  PassInfo pass_info;
  std::function<Function(Function, Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
  bool SkipFunction(const Function& func) const;
  // 其他成员/方法省略
};
```

`pass_info` 与模块级传递相同。 `pass_func`接受函数与模块作为输入，可在函数上执行优化； 函数若被注解为`SkipOptimization`，将被跳过。

#### 顺序传递（Sequential Pass） 

`SequentialPass` 类似于 PyTorch 的 `nn.Sequential`，可包含多个顺序执行的传递。

``` c++
class SequentialPassNode : PassNode {
  PassInfo pass_info;
  // 需要执行的传递列表
  ffi::Array<Pass> passes;
  bool PassEnabled(const PassInfo& info) const;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
};
```

以下展示顺序传递的执行逻辑：系统会按照传递添加的顺序依次执行。

``` c++
Module SequentialNode::operator()(const Module& module,
                                  const PassContext& pass_ctx) const {
  Module mod = module;
  for (const Pass& pass : passes) {
    ICHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!PassEnabled(pass_info))  continue;
    for (const auto& it : pass_info->required) {
      const auto* name = it.as<tvm::ir::StringImm>();
      ICHECK(name);
      mod = GetPass(name->value)(mod, pass_ctx);
    }
    mod = pass(mod, pass_ctx);
  }
  return mod;
}
```

在执行传递前，系统会判断该传递是否启用：首先检查是否被用户禁用，其次查看是否被显式声明为必需。若仍未确定，则根据 `opt_level` 判断是否执行。

执行时，系统会根据传递名从注册表中获取对应实现：

``` c++
Pass GetPass(const std::string& pass_name) {
  using tvm::runtime::Registry;
  std::string fpass_name = "relax.transform." + pass_name;
  const std::optional<tvm::ffi::Function> f = tvm::ffi::Function::GetGlobal(fpass_name);
  ICHECK(f.has_value()) << "Cannot find " << fpass_name
                        << "to create the pass " << pass_name;
  return (*f)();
}
```

系统还提供辅助函数用于创建各类传递，并暴露给 Python 前端：

``` c++
Pass CreateFunctionPass(
    std::function<Function(Function, IRModule, PassContext)> pass_func,
    int opt_level,
    ffi::String name,
    ffi::Array<ffi::String> required);

Pass CreatePrimFuncPass(
    std::function<PrimFunc(PrimFunc, IRModule, PassContext)> pass_func,
    int opt_level,
    ffi::String name,
    ffi::Array<ffi::String> required);

Pass CreateModulePass(
    std::function<IRModule(IRModule, PassContext)> pass_func,
    int opt_level,
    ffi::String name,
    ffi::Array<ffi::String> required);

Pass Sequential(tvm::ffi::Array<Pass> passes, PassInfo pass_info);
```

#### 传递注册

前文介绍了不同粒度的传递和编译上下文。
下面展示如何注册一个传递。以常量折叠（constant folding）为例， 它用于在 Relax 函数中折叠常量（实现位于 [src/relax/transforms/fold_constant.cc](https://github.com/apache/tvm/blob/main/src/relax/transforms/fold_constant.cc)）。

该传递提供了 `Expr` 到 `Expr` 的转换 API：

``` c++
Expr FoldConstant(const Expr& expr);
```

要将其注册到传递基础设施中，首先需要确定传递的粒度。常量折叠作用于函数级，因此通过 `CreateFunctionPass` 创建：`pass_func` 以打包函数形式返回，用于对 [IRModule]{.title-ref} 中的每个函数调用该转换 API。 `{}` 表示该传递没有前置依赖；若有依赖，开发者需明确列出。

同时，注册名为 `"relax.transform.FoldConstant"` 的 API 入口，使该传递可被 C++ （例如以上的 `GetPass` ）与 Python 访问：

``` c++
namespace transform {

Pass FoldConstant() {
  auto pass_func =
      [=](Function f, IRModule m, PassContext pc) { return ConstantFolder::Fold(f, m); };
  return CreateFunctionPass(pass_func, 0, "FoldConstant", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.FoldConstant", FoldConstant);
}

}  // namespace transform
```

为方便其他 C++ 模块调用，在[include/tvm/relax/transform.h](https://github.com/apache/tvm/blob/main/include/tvm/relax/transform.h)中声明：

``` c++
TVM_DLL Pass FoldConstant();
```


#### 传递检测（Pass Instrument） 

传递检测机制用于分析传递本身，例如统计执行时间与内存占用，或观察 IR 如何被改变。

我们在 `PassContext` 生命周期中引入四个检测点：

``` c++
TVM_DLL void InstrumentEnterPassContext();
TVM_DLL void InstrumentExitPassContext();
TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;
```

`InstrumentEnterPassContext` 在进入 `PassContext` 作用域时调用。

`InstrumentExitPassContext` 在离开 `PassContext` 或执行发生异常时调用。当通过 :py`tvm.transform.PassContext`的`override_instruments` 覆盖检测器时也会触发，见`pass_instrument_overriden`。

`InstrumentBeforePass` 在传递执行前调用； 若该传递应执行，则在执行后调用
`InstrumentAfterPass`。其伪代码如下：

``` c++
if (pass_ctx.InstrumentBeforePass(ir_module, pass_info)) {
  new_ir_module = run_pass(ir_module, pass_ctx);
  pass_ctx.InstrumentAfterPass(new_ir_module, pass_info);
  return new_ir_module;
}
```

`PassInstrument`接口允许你在上述四个阶段插入自定义逻辑。 可向单个`PassContext` 注册多个检测器实例，它们将按 `instruments`指定的顺序依次调用。

接口定义如下：

``` c++
namespace instrument {

class PassInstrumentNode : public Object {
 public:
  ffi::String name;
  virtual void EnterPassContext() const = 0;
  virtual void ExitPassContext() const = 0;
  virtual bool ShouldRun(const IRModule& mod, const transform::PassInfo& info) const = 0;
  virtual void RunBeforePass(const IRModule& mod, const transform::PassInfo& info) const = 0;
  virtual void RunAfterPass(const IRModule& mod, const transform::PassInfo& info) const = 0;
  /* 其他字段省略 */
};

class PassInstrument : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PassInstrument, ObjectRef, PassInstrumentNode);
};

}  // namespace instrument
```

Python 前端提供了便捷方式来实现 `PassInstrument`，见`pass_instrument_py_frontend`。

在一个 `PassContext` 中，某个 `PassInstrument` 实例的调用顺序如下：

    with PassContext(instruments=[pi])  # pi 为某个 PassInstrument 实现
        pi.EnterPassContext()

        if pi.ShouldRun(Pass1):
            pi.RunBeforePass()
            Pass1()
            pi.RunAfterPass()

        if pi.ShouldRun(Pass2):
            pi.RunBeforePass()
            Pass2()
            pi.RunAfterPass()

        pi.ExitPassContext()

以下简述 `PassInstrument` 与 `PassContext` 方法之间的关系，详见
[src/ir/transform.cc](https://github.com/apache/tvm/blob/main/src/ir/transform.cc)：

-   `InstrumentEnterPassContext`
    -   `EnterPassContext()` 按传入 `instruments` 的顺序执行。
    -   若执行中抛出异常，`PassContext` 会清空所有已注册的检测器。
    -   然后对已成功执行 `EnterPassContext()` 的检测器依次调用
        `ExitPassContext()`。
    -   例如，注册了 A、B、C 三个检测器，A 成功，B 抛异常，则 C
        不会执行；随后调用 A 的 `ExitPassContext()`。
-   `InstrumentExitPassContext`
    -   各检测器的 `ExitPassContext()` 按 `instruments` 顺序执行。
    -   若发生异常，`instruments` 会被清空。
    -   抛出异常后注册的检测器不会执行 `ExitPassContext`。
-   `InstrumentBeforePass`
    -   若该传递未被显式列为"必需"，则会调用 `ShouldRun`。
    -   若未被 `ShouldRun` 阻塞，则按顺序调用 `RunBeforePass`。
    -   该函数返回布尔值，指示该传递是否应执行。
    -   若发生异常，将立即抛出；Python
        依靠上下文管理器安全退出（确保各检测器的 `ExitPassContext`
        被调用；C++ 见
        [include/tvm/support/with.h](https://github.com/apache/tvm/blob/main/include/tvm/support/with.h)）。
-   `InstrumentAfterPass`
    -   按顺序调用 `RunAfterPass`。
    -   若发生异常，将立即抛出；依靠上下文管理器或 `With`
        类（[include/tvm/support/with.h](https://github.com/apache/tvm/blob/main/include/tvm/support/with.h)）安全退出。

#### 内置检测器

系统内置若干检测器（标注 *TODO* 的尚未实现）：

-   **PassTimingInstrument**（见
    [src/ir/instrument.cc](https://github.com/apache/tvm/blob/main/src/ir/instrument.cc)）
    -   用于分析各传递的执行时间。
-   **PrintIRBefore**（TODO）
    -   在传递执行前打印 IR。也可通过
        :py`tvm.transform.PrintIR`{.interpreted-text role="func"}
        在传递周围插入打印实现；但使用检测器无需修改传递序列。
-   **PrintAfter**（TODO）
    -   在传递执行后打印 IR。

### Python 前端

前端仅需少量 API 即可创建并执行传递（完整实现见[python/tvm/relax/transform/transform.py](https://github.com/apache/tvm/blob/main/python/tvm/relax/transform/transform.py)与[python/tvm/ir/transform.py](https://github.com/apache/tvm/blob/main/python/tvm/ir/transform.py)）。后端将根据提供的信息决定如何创建 Pass 对象。

#### PassContext

Python 前端为 `PassContext` 提供了包装以支持 `with` 语法，并提供`current` 静态方法：

``` python
@tvm_ffi.register_object("transform.PassContext")
class PassContext(tvm.runtime.Object):
    def __enter__(self):
        _transform.EnterPassContext(self)
        return self

    def __exit__(self, ptype, value, trace, config):
        _transform.ExitPassContext(self)

    @staticmethod
    def current():
        """Return the current pass context."""
        return _transform.GetCurrentPassContext()
```

`PassContext`用于配置编译选项（优化级别、必需/禁用传递等），并可传入配置字典，以便不同传递读取需要的数据（如回退设备信息、循环展开步数/深度等）。若要从 `config` 中获取某项配置，其键名需通过`TVM_REGISTER_PASS_CONFIG_OPTION` 注册，例如循环展开传递：

``` c++
TVM_REGISTER_PASS_CONFIG_OPTION("tir.UnrollLoop", UnrollLoopConfig);
```

详见[src/tir/transforms/unroll_loop.cc](https://github.com/apache/tvm/blob/main/src/tir/transforms/unroll_loop.cc)。

#### Python 中的传递检测 

使用装饰器（[python/tvm/ir/instrument.py](https://github.com/apache/tvm/blob/main/python/tvm/ir/instrument.py)）可以快速实现
`PassInstrument`。 推荐使用装饰器方式而非继承：

-   `enter_pass_ctx`：进入 `PassContext` 时执行；
-   `exit_pass_ctx`：退出 `PassContext` 时执行；
-   `should_run`：在传递执行前调用，返回该传递是否应执行；
-   `run_before_pass`：传递执行前调用；
-   `run_after_pass`：传递执行后调用。

可通过 :py`tvm.transform.PassContext` 的
`instruments` 参数注册实例。更多示例见[use pass
instrument](https://github.com/apache/tvm/blob/main/tutorials/dev/use_pass_instrument.py)教程。

#### 覆盖当前 PassContext 中的检测器

`override_instruments` 方法可覆盖当前 `PassContext` 中的 `instruments`。例如，当未显式创建新 `PassContext`
而直接运行传递时，仍可将检测器注册到全局上下文：

``` python
cur_pass_ctx = tvm.transform.PassContext.current()
# 覆盖 PassInstrument 实例
cur_pass_ctx.override_instruments([pass_inst])
mod = pass_seq(mod)
result = pass_inst.get_result()
```

注意：调用 `override_instruments` 时，旧检测器的 `exit_pass_ctx`会被调用，随后新检测器的 `enter_pass_ctx` 会被调用。
