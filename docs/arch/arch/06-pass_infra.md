---
title: Pass Infrastructure
sidebar_position: 160
---

# Pass Infrastructure

Relay 和 TVM IR 都包含了一系列优化 pass，它们可提高模型的性能指标，如平均推理、内存占用或特定设备的功耗。有一套标准优化以及机器学习特定的优化，包括常量折叠、死代码消除、算子布局更改、算子融合、buffer 处理和循环转换等。每个 pass 都被构造为 ir-to-ir 转换，它们用遍历期间和/或遍历之前收集的分析结果进行转换。

TVM 的快速发展催生更加系统化和有效的方式，来管理这些 pass。此外，管理 TVM 堆栈不同层（例如 Relay 和 tir）pass 的通用框架极大便利了开发者，使他们可以快速原型化并将实现的 pass 插入至系统。

该文档描述了一个 infra 的设计——利用生产编译器来管理优化 pass 的方式，以及用于构建层的现代深度学习框架的风格。

例如，许多现有的生产编译器，如 GCC 和 LLVM，都使用 pass 管理器来有效地管理 pass 的执行。最初管理 pass 很简单，因为 pass 的数量很少，但成熟的编译器包含数百个单独的 pass。外部用户通常希望正确调度自定义 pass，而无需修改单个手工制作的 pass 顺序。

类似地，现代深度学习框架，如 Pytorch 和 MXNet Gluon，也倾向于分别通过 [Sequential](https://pytorch.org/docs/stable/nn.html?highlight=sequential#torch.nn.Sequential) 和 [Block](https://mxnet.apache.org/api/python/docs/api/gluon/block.html#gluon-block) 实现 pass 模式的层构建方案。得到了这样的结构后，这些现代框架能够方便地将模块/层添加到容器中，并轻松地构建神经网络。

Relay pass infra 的设计，很大程度上是受到 LLVM 中使用的分层 pass 管理器，以及流行的深度学习框架中使用的块式容器的启发。Pass infra 的主要目标：

1. 实现更好的优化程序编排。用户可以灵活地定制和构建自己的优化 pipeline。
2. 提供一种对用户更友好的方式来调试优化 pass。
3. 减轻开发者手动解决 pass 之间的依赖关系的工作量。
4. 为开发者简化新 pass 的实现。允许用户在 Python 中实现 pass，并让 pass 基础架构操纵其执行。

## 设计

聚焦用户扩展的简易性，使用户可以快速添加新的 pass，且向后兼容。设计包含后端和前端。前者（即后端）实现了 pass infra 的主要逻辑。后者（即前端）为用户提供简单的 API 进行交互，允许用户快速创建自己的优化 pipeline。

### C++ 后端

`PassInfo` 对象包含一个 pass 所需的基本信息，其中 `name` 是 pass 名称，`opt_level` 表示将在哪个优化级别启用 pass，`required` 表示执行某个 pass 所需的 pass（有关详细信息，参阅 [include/tvm/ir/transform.h](https://github.com/apache/tvm/blob/main/include/tvm/ir/transform.h)）。例如，在注册 pass  期间（稍后介绍），pass 开发者可以指定 pass 的名称、执行的优化级别和/或所需的 pass。`opt_level` 可帮助 pass infra 识别在用户提供的优化级别下运行时，是否需要执行某个 pass。pass infra 可以用 `required` 字段来解决 pass 依赖。

``` c++
class PassInfoNode : public Object {
  String name;
  int opt_level;
  Array<String> required;
};
```

#### PassContext

`PassContext` 具备优化 pass 的有用信息。例如，它包含错误报告系统，可以提供优化失败原因的诊断。 `PassContext` 用来替换旧的 `BuildConfig`，`BuildConfig` 用于帮助用户配置编译选项，包括优化级别和必需/禁用的 pass 等。例如，有一个配置，在 `opt_level=3` 执行所有 pass ，同时用 `PassContext` 提供的 `disabled_pass=xx` 来禁用一些 pass 。在 `opt_level=3` 处全局化所有 pass ，并排除禁用 pass 列表中的所有 pass。`PassContext` 还提供了一种检测所有 pass 的方法。参阅 [Pass Instrument](https://tvm.apache.org/docs/arch/pass_infra.html#pass-instrument-cpp-backend) 这一节。

用户可以用这个类编写 Python `with` 语法，从而在一定的配置下进行优化。此外，用户可以通过 `PassContext::Current()` 以线程安全的方式获取某个程序范围内可用的上下文，因为线程本地存储的 `PassContextThreadLocalStore` 用于保存创建的 pass 上下文对象。稍后用示例来展示如何用 C++ 和 Python API 来通过 pass 上下文创建编译 pipeline。

``` c++
class PassContextNode : public Object {
 public:
  int opt_level{2};
  tvm::Array<tvm::Expr> required_pass;
  tvm::Array<tvm::Expr> disabled_pass;
  mutable Optional<DiagnosticContext> diag_ctx;
  Map<String, ObjectRef> config;
  Array<instrument::PassInstrument> instruments;
};

class PassContext : public NodeRef {
 public:
  TVM_DLL static PassContext Create();
  TVM_DLL static PassContext Current();
  TVM_DLL void InstrumentEnterPassContext();
  TVM_DLL void InstrumentExitPassContext();
  TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
  TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;
  /* 省略其他字段。 */

 private:
  // pass 上下文范围的入口。
  TVM_DLL void EnterWithScope();
  // pass 上下文范围的退出。
  TVM_DLL void ExitWithScope();

  // 获取 Python `with` 语法的类。
  friend class tvm::With<PassContext>;
};

struct PassContextThreadLocalEntry {
  /*！ \摘要：默认 pass 上下文。 */
  PassContext default_context;
  /*! \摘要：当前 pass 上下文 */
  std::stack<PassContext> context_stack;
  PassContextThreadLocalEntry() {
    default_context = PassContext(make_node<PassContextNode>());
  }
};

/*！ 保存 pass 上下文的线程本地存储 */
typedef dmlc::ThreadLocalStore<PassContextThreadLocalEntry>
     PassContextThreadLocalStore;
```

#### Pass 构造

pass infra 以分层的方式设计，可以在不同粒度的 Relay/tir 程序下工作。我们引入一个纯虚类 `PassNode`，作为不同优化 pass 的基础。这个类包含几个必须由子类在模块、函数或 pass 序列级别实现的虚拟方法。

``` c++
class PassNode : Object {
  virtual PassInfo Info() const = 0;
  virtual Module operator()(const IRModule& mod
                            const PassContext& pass_ctx) const = 0;
};
```

仿函数展示了 pass 是如何实现的，即它始终在特定上下文对 `IRModule` 起作用。所有 pass  都以 `Module` 到 `Module` 的方式设计。因此，由 pass infra 管理的优化将始终更新整个模块。

已经创建了几个子类来实现不同类型的优化 pass，例如，函数级 pass、模块级 pass 和顺序 pass。每个子类本身都可以充当 pass 管理器。例如，它们可以收集所需的 pass 并执行，或是基于给定的元数据构建依赖关系图。访问 [src/relay/ir/transform.cc](https://github.com/apache/tvm/blob/main/src/relay/ir/transform.cc) 和 [src/ir/transform.cc](https://github.com/apache/tvm/blob/main/src/ir/transform.cc)，查看完整定义。

#### 模块级 Pass

模块级 pass 主要针对全局和过程间优化 (IPO)，类似于 LLVM 中的模块 pass。Relay 中一些需要模块全局图的典型 pass，如 A-normal form 转换和 lambda 提升等，都属于这个集合。在这个级别，用户甚至可以在模块中添加和/或删除功能。注意，是所有 pass。

``` c++
class ModulePassNode : PassNode {
  PassInfo pass_info;
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
  // 其他成员/方法省略
};
```

`pass_info` 维护模块级 pass 所需的信息。`pass_func` 描述真正的优化。例如，可能需要消除模块的死代码。可在 `pass_func` 中实现算法，并让它在模块上运行。然后它将删除死代码，包括模块中未使用的函数。注意，该字段被设计为一个打包函数，可以在 C++ 和 Python 中实现优化。

#### 函数级 Pass

函数级 pass 用于对给定的 Relay/tir 模块进行各种函数内的优化。它每次从模块的函数列表中获取一个函数进行优化，并产生一个重写的 Relay `Function` 或 tir `PrimFunc`。大部分 pass 都可以归为这一类，比如 Relay 中常见的子表达式消除和推理简化，以及 tir 中的向量化和展平存储等。

注意，这个级别的 pass 范围是 Relay 函数或 tir 原始函数。因为它们不知道全局信息，所以无法通过这些 pass 添加或删除功能。

``` c++
class FunctionPassNode : PassNode {
  PassInfo pass_info;
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
  bool SkipFunction(const Function& func) const;
  // 其他成员/方法省略...
};
```

`pass_info` 与模块 pass 中的描述相同。`pass_func` 接收一个函数进行优化，还需要一个模块，可以使用模块来报错。用「SkipOptimization」注解函数，从而在优化期间将忽略它。

#### 顺序 Pass

`SequentialPass` 类似于 Pytorch 中 `nn.Sequential`，包含许多用于执行的 pass。

``` c++
class SequentialPassNode : PassNode {
  PassInfo pass_info;
  // 要执行的 Pass。
  Array<Pass> passes;
  bool PassEnabled(const PassInfo& info) const;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
};
```

目前在 Relay 中只有少数 pass 被放入该组。例如，`FoldScaleAxis` 需要在内部调度 `ForwardFoldScaleAxis` 和 `BackwardFoldScaleAxis`。另外，推荐先实现 `BackwardFoldScaleAxis`。因此，这个 pass 是 `SequentialPass` 的理想候选。

以下代码展示了如何调用顺序 pass 中的各个 pass。本质上，按照它们在 pass 列表中的顺序依次执行每个 pass。

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

调用 pass 首先会检查这个 pass 是否启用——首先检查 pass 是否被用户明确禁用，然后检查它是否被用户指定为必需 pass。如果仍不能确定，则检查其 `opt_level`。只有当它的优化级别不低于在 pass 上下文中配置的优化级别时，才会启用并执行此 pass。

要执行 pass，首先要用 pass 名称在 TVM 打包函数注册表中检索已注册的 pass。每个 pass 都注册了一个 API 端点，将在后面展示。

``` c++
Pass GetPass(const std::string& pass_name) {
  using tvm::runtime::Registry;
  std::string fpass_name = "relay._transform." + pass_name;
  const auto* f = Registry::Get(fpass_name);
  ICHECK(f != nullptr) << "Cannot find " << fpass_name
                      << "to create the pass " << pass_name;
  return (*f)();
}
```

一些辅助函数可以创建上述这些 pass 的每种类型。这些辅助函数也提供给 Python 前端，以便用户更好地使用 Python API 来创建特定的 pass 对象。

``` c++
Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
    int opt_level,
    String name,
    Array<String> required);

Pass CreatePrimFuncPass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level,
    String name,
    Array<String> required);

Pass CreateModulePass(
    const runtime::TypedPackedFunc<IRModule(IRModule, PassContext)>& pass_func,
    int opt_level,
    String name,
    Array<String> required);

Pass Sequential(tvm::Array<Pass> passes, PassInfo pass_info);
```

#### Pass 注册

前面已经介绍了不同级别 pass 的概念和用于编译的上下文。接下来以常量折叠为例，介绍用户注册 pass。这个 pass 可以在 Relay 函数（在 [src/relay/transforms/fold_constant.cc](https://github.com/apache/tvm/blob/main/src/relay/transforms/fold_constant.cc) 中）中折叠常量。

提供一个 API 来执行 `Expr` 到 `Expr` 的转换。

``` c++
Expr FoldConstant(const Expr& expr);
```

为了将这个 pass 注册到 pass infra，首先决定这个 pass 要在哪个级别执行。由于常量折叠发生在单个函数上，应该通过 `CreateFunctionPass` 直观地为它创建一个 `FunctionPass`。`pass_func` 作为一个打包函数返回，该函数在 *IRModule* 中的每个函数上调用 `Expr` 到 `Expr` API。`{}` 表示此 pass 不需要任何先决条件。否则，pass 开发者必须识别并列出它们。

同时，通过名称 `relay._transform.FoldConstant` 注册了一个 pass API 端点。因此，此 pass 成为注册表中的一个条目，可以在需要时由 C++（例如上面的 `GetPass`）和 Python 访问。

``` c++
namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(FoldConstant(f));
  };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldConstant")
.set_body_typed(FoldConstant);

}  // 命名空间变换
```

为了允许其他 C++ 模块应用此 pass，在 [include/tvm/relay/transform.h](https://github.com/apache/tvm/blob/main/include/tvm/relay/transform.h) 中声明了一个自由函数，如下所示：

``` c++
TVM_DLL Pass FoldConstant();
```

#### Pass Instrument

Pass Instrument 是一种分析 pass 本身的机制。例如，可以用基础架构来了解 pass 需要多少时间和内存，或者 pass 如何转换 IR 模块。

在 `PassContext` 的生命周期中引入了四个检测点。

``` c++
TVM_DLL void InstrumentEnterPassContext();
TVM_DLL void InstrumentExitPassContext();
TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;
```

进入 `PassContext` 实例的范围时，立即调用 `InstrumentEnterPassContext`。

离开了 `PassContext` 的范围或者 pass 执行过程中发生了异常，会调用 `InstrumentExitPassContext`。当工具类被 `tvm.transform.PassContext` 中的 `override_instruments` 覆盖时，也会调用此方法。参阅 [在当前 PassContext 中复写工具类](https://tvm.apache.org/docs/arch/pass_infra.html#pass-instrument-overriden)。

`InstrumentBeforePass` 在执行前被调用。如果要在执行后运行 pass，则调用 `InstrumentAfterPass`。行为如下：

``` c++
if (pass_ctx.InstrumentBeforePass(ir_module, pass_info)) {
  new_ir_module = run_pass(ir_module, pass_ctx);
  pass_ctx.InstrumentAfterPass(new_ir_module, pass_info);
  return new_ir_module;
}
```

`PassInstrument` 接口允许在上述四种方法中运行任意代码。多个 `PassInstrument` 实例可以注册到一个 `PassContext` 中。`PassInstrument` 实例按照传递给 `PassContext` 的 `instruments` 参数的顺序依次调用。

`PassInstrument` 提供以下接口：

``` c++
namespace instrument {

class PassInstrumentNode : public Object {
 public:
  String name;
  virtual void EnterPassContext() const = 0;
  virtual void ExitPassContext() const = 0;
  virtual bool ShouldRun(const IRModule& mod, const transform::PassInfo& info) const = 0;
  virtual void RunBeforePass(const IRModule& mod, const transform::PassInfo& info) const = 0;
  virtual void RunAfterPass(const IRModule& mod, const transform::PassInfo& info) const = 0;
  /* 省略其他字段。 */
};

class PassInstrument : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(PassInstrument, ObjectRef, PassInstrumentNode);
};

}  // 命名空间工具
```

提供 Python 前端快速实现 `PassInstrument`。参阅 [Pass 工具](https://tvm.apache.org/docs/arch/pass_infra.html#pass-instrument-py-frontend)。

在 `PassContext` 中，`PassInstrument` 实例的调用顺序如下：

```  python
with PassContext(instruments=[pi]) # pi = a PassInstrument implementation. # pi = PassInstrument 实现。
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
```

接下来简单介绍 `PassInstrument` 接口和 `PassContext` 方法的关系。更多详细信息，参阅 ([src/ir/transform.cc](https://github.com/apache/tvm/blob/main/src/ir/transform.cc))。

* `InstrumentEnterPassContext`
  * `EnterPassContext()` 按照传递给 `PassContext` 的 `instruments` 的顺序执行。
  * 当抛出异常时，`PassContext` 通过清除所有已注册的 `PassInstrument` 实例来禁用 pass 工具。
  * 然后 `PassContext` 执行每个成功完成 `EnterPassContext()` 的 `PassInstrument` 实例的 `ExitPassContext()` 方法
  * 例如，如果将 `PassInstrument`  A、B 和 C 注册到 `PassContext` 并且 A 完成 `EnterPassContext()` 而 B 抛出异常，则永远不会执行 C；而执行 A 的 `ExitPassContext()`。
* `InstrumentExitPassContext`
  * 按照传递给 `PassContext` 的 `instruments` 的顺序执行每个 `PassInstrument` 实例的 `ExitPassContext()`。
  * 发生异常时，`instruments` 被清除。
  * 在抛出异常后注册的 `PassInstrument` 实例不执行 `ExitPassContext`。
* `InstrumentBeforePass`
  * 如果该 pass 未列为必需 pass，则执行 `ShouldRun`。
  * 如果 pass 没有被 `ShouldRun` 阻止，则 `RunBeforePass` 将按照 `instruments` 的顺序执行。
  * 请注意， `InstrumentBeforePass` 返回一个布尔值，指示是否应该运行 pass。
  * 当异常发生时，立即抛出。我们依赖 Python 上下文管理器安全退出 `PassContext`（意味着每个工具的 `ExitPassContext` 都会运行。对于 C++，参考 [include/tvm/support/with.h](https://github.com/apache/tvm/blob/main/include/tvm/support/with.h)。）
* `InstrumentAfterPass`
  * `RunAfterPass` 按照传递给 `PassContext` 的 `instruments` 的顺序执行。
  * 当异常发生时，立即抛出。依靠 Python 上下文管理器或 `With` 类 ([include/tvm/support/with.h](https://github.com/apache/tvm/blob/main/include/tvm/support/with.h)) 安全退出 `PassContext`。

#### 内置工具

以下是几种内置工具。标有 *TODO* 的还没有实现。

* PassTimingInstrument（参考 [src/ir/instrument.cc](https://github.com/apache/tvm/blob/main/src/ir/instrument.cc)）
  * 分析 pass 的执行时间
* PrintIRBefore (TODO)
  * 在 pass 转换之前打印 IR 模块。若在 pass 周围插入 `tvm.transform.PrintIR()` 也可以达到这个目的。但使用 `PassInstrument`，不需要修改 pass 的顺序。
* PrintAfter (TODO)
  * 在 pass 转换后打印 IR 模块。

### Python 前端

前端只需要一些简单的 API。例如，可以为用户提供以下 API 来创建和执行 pass（完整的实现在 [python/tvm/relay/transform/transform.py](https://github.com/apache/tvm/blob/main/python/tvm/relay/transform/transform.py) 和 [python/tvm/ir/transform.py](https://github.com/apache/tvm/blob/main/python/tvm/ir/transform.py) 中）。后端接收信息，并决定用哪个函数来创建 Pass 对象。

#### PassContext

Python 前端为 `PassContext` 提供了一个 wrapper，通过覆盖 `__enter__` 和 `__exit__` 来启用 `with` 语法。提供一种 `current` 静态方法，供用户获取在一定范围内使用的上下文。

``` python
@tvm._ffi.register_object("transform.PassContext")
class PassContext(tvm.runtime.Object):
    def __enter__(self):
        _transform.EnterPassContext(self)
        return self

    def __exit__(self, ptype, value, trace, config):
        _transform.ExitPassContext(self)

    @staticmethod
    def current():
        """返回当前 pass 上下文。"""
        return _transform.GetCurrentPassContext()
```

`PassContext` 用于配置编译选项，包括优化级别和必需及禁用的 pass。它还接收一个配置字典，以便不同的 pass 可以方便地获取传递的数据，例如回退设备信息和循环展开的步长/深度等。为了能够获取所需的配置，必须通过 `TVM_REGISTER_PASS_CONFIG_OPTION` 注册密钥。例如，循环展开 pass 使用以下内容。

``` python
TVM_REGISTER_PASS_CONFIG_OPTION("tir.UnrollLoop", UnrollLoopConfig);
```

更多详情参阅 [src/tir/transforms/unroll_loop.cc](https://github.com/apache/tvm/blob/main/src/tir/transforms/unroll_loop.cc)。

#### Pass 对象

`Pass` 是所有 pass 对象的基类。这里的所有方法都只是在后端实现的简单 wrapper。是为方便用户与 Python 中的基类交互而定义的。在 pass 基类中只定义了一个 `__call__`，使子类成为可调用对象，进而可以轻松调用它们（例如 `pass_xx(arg)`）进行执行。

``` python
@register_relay_node
class Pass(RelayNode):
   def __call__(self, mod):
       return _transform.RunPass(self, mod)
```

提供一些辅助 API，方便从 Python 前端轻松创建 pass，并让 pass infra 控制执行。例如，`module_pass`、`function_pass` 和 `sequential` 提供给用户，方便用户自定义 pass 或 pass pipeline。

对于在 C++ 后端实现的所有 pass，分别在 [python/tvm/ir/transform.py](https://github.com/apache/tvm/blob/main/python/tvm/ir/transform.py) 和 [python/tvm/relay/transform/transform.py](https://github.com/apache/tvm/blob/main/python/tvm/relay/transform/transform.py) 中提供了相应的 Python API。例如，常量折叠有一个 Python API，如下所示：

``` python
def FoldConstant():
    return _transform.FoldConstant()
```

通过装饰器进行装饰，创建一个 pass，如下所示：

``` python
 @relay.transform.module_pass(opt_level=2)
 def transform(mod, ctx):
    tp = relay.TensorType((10,), "float32")
    x = relay.var("x", tp)
    gv = relay.GlobalVar("abs")
    func = relay.Function([x], relay.abs(x))
    new_mod = tvm.IRModule({gv: func})
    new_mod.update(mod)
    return new_mod

module_pass = transform
assert isinstance(module_pass, transform.ModulePass)
assert module_pass.info.opt_level == 2
```

这里的 `transform` 函数为输入模块添加了一个 `abs` 函数，它可以是模块级别的任何自定义优化。创建此 `module_pass` 后，用户可以将其应用于任何 Relay 模块。例如，可以构建一个空模块，应用这个 pass，添加一个 `abs` 函数。

``` python
mod = tvm.IRModule()
mod = module_pass(mod)
```

相应地，也为 `function_pass` 提供了这样的功能。例如，函数级 pass 编写示例如下：

``` python
@relay.transform.function_pass(opt_level=1)
class TestReplaceFunc:
   def __init__(self, new_func):
      self.new_func = new_func
      def transform_function(self, func, mod, ctx):
         # 仅用于演示
         # 将 func 转换为 new_func
         return self.new_func

x = relay.var("x", shape=(10, 20))
f1 = relay.Function([x], x)
f2 = relay.Function([x], relay.log(x))
# fpass 现在是一个特殊的 pass，它取代了每一个
# 函数为 f1
fpass = TestReplaceFunc(f1)
# 现在 input_mod 中的每个函数都被 f1 替换了
res_mod = fpass(input_mod)
```

或者，用户也可以不使用装饰器直接注册一个 pass，然后调用它。有关如何自定义优化 pipeline 和调试 Relay 和 tir pass 的更多示例，参阅 [使用 pass 基础架构](https://github.com/apache/tvm/blob/main/tutorials/dev/use_pass_infra.py) 教程。

#### Pass Instrument

可以通过在实现以下方法的类上使用 `pass_instrument` 装饰器 ([python/tvm/ir/instrument.py](https://github.com/apache/tvm/blob/main/python/tvm/ir/instrument.py)) 来实现 `PassInstrument`。注意，推荐使用 `pass_instrument` 装饰器来实现 `PassInstrument`，而不是覆盖或子类化。

* `enter_pass_ctx`
  * 该方法在进入 `PassContext` 时运行。
* `exit_pass_ctx`
  * 此方法在退出 `PassContext` 时运行。
* `should_run`
  * 此方法在执行 pass 之前运行，返回一个布尔值，指示是否应该运行 pass。
* `run_before_pass`
  * 若要运行某 pass，则在 pass 执行之前运行此方法。
* `run_after_pass`
  * 此方法在执行 pass 后立即运行。
  * 

`PassInstrument` 实例可以通过 `tvm.transform.PassContext` 中的 `instruments` 参数注册。

[使用 pass Instrument](https://github.com/apache/tvm/blob/main/tutorials/dev/use_pass_instrument.py) 教程提供了如何使用 Python API 实现 `PassInstrument` 的示例。

#### 在当前 PassContext 中覆盖工具

提供 `override_instruments` 方法来覆盖当前 `PassContext` 的 `instruments`。例如，若在没有显式创建新 `PassContext` 的情况下运行 pass，仍然可以通过以下方式将 `PassInstrument` 注册到全局 `PassContext` 中：

``` python
cur_pass_ctx = tvm.transform.PassContext.current()
# 覆盖 PassInstrument 实例
cur_pass_ctx.override_instruments([pass_inst])
mod = pass_seq(mod)
result = pass_inst.get_result()
```

注意，调用 `override_instruments` 同时也会调用旧 `PassInstrument` 实例的 `exit_pass_ctx` 方法。然后调用新 `PassInstrument` 的 `enter_pass_ctx` 方法。