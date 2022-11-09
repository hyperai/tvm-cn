---
title: 如何使用 TVM Pass Infra
---

# 如何使用 TVM Pass Infra

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/extend_tvm/use_pass_infra.html#sphx-glr-download-how-to-extend-tvm-use-pass-infra-py) 下载完整的示例代码
:::

**作者**：[Zhi Chen](https://github.com/zhiics)

随着 Relay/tir 中优化 Pass 数的增加，手动执行并维护它们的依赖关系变得难以处理。因此我们引入了一个基础架构来管理优化 Pass，并使其适用于 TVM 堆栈中 IR 的不同层。

Relay/tir 程序的优化可以在各种粒度上应用，即分别使用 `tvm.relay.transform.FunctionPass`/ `tvm.tir.transform.PrimFuncPass` 和 `tvm.transform.ModulePass` 的功能级和模块级，或者用户可以依靠 `tvm.transform.Sequential` 在 Relay/tir 程序上应用一系列 Pass，其中 Pass 之间的依赖关系可以通过 pass infra 来解决。有关这些 Pass 类型的更多详细信息，参阅 [Pass Infrastructure](https://tvm.apache.org/docs/arch/pass_infra.html#pass-infra)。

本教程主要演示开发者如何使用 pass infra 进行某种优化，并为 Relay 程序创建优化 Pass。同样的方法也可以用于 tir。

``` python
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
```

## 创建 Relay 程序示例

首先为教程创建一个简单的 Relay 程序，该程序用于本教程中示例的各种优化。同样，用户可以编写 tir 原始函数并应用 tir pass。

``` python
def example():
    shape = (1, 64, 54, 54)
    c_data = np.empty(shape).astype("float32")
    c = relay.const(c_data)
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    x = relay.var("x", relay.TensorType((1, 64, 56, 56), "float32"))
    conv = relay.nn.conv2d(x, weight)
    y = relay.add(c, c)
    y = relay.multiply(y, relay.const(2, "float32"))
    y = relay.add(conv, y)
    z = relay.add(y, c)
    z1 = relay.add(y, c)
    z2 = relay.add(z, z1)
    return relay.Function([x, weight], z2)
```

## 优化程序

接下来优化程序，Relay 具有许多优化功能，选择其中一部分应用到这个示例程序中。

有多种方法可以优化 Relay 程序。下面将逐一讲解。

### 手动应用优化 Pass

``` python
# 首先创建一个包含一个或多个 Relay 的 relay 模块
# 优化函数。
f = example()
mod = tvm.IRModule.from_expr(f)

# 现在可以在模块上应用常量折叠。
# fold_const 是一个不带任何参数的回调函数。
fold_const = relay.transform.FoldConstant()
# 然后，在给定的模块上调用 pass。注意，常数
# folding pass 在函数级别工作。话虽如此，每个
# 模块中的函数将应用优化。用户无需迭代
# 通过各个函数手动应用此 pass。
mod = fold_const(mod)
# 从更新后的程序中可以看到常量被折叠了。
print(mod)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %3 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %3) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
```

可以以类似的方式应用更多优化。例如，可以消除 *z* 和 *z1* 使用的常用表达式。

``` python
mod = relay.transform.EliminateCommonSubexpr()(mod)
print(mod)
```

输出结果：

``` bash
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
```

融合也是参数化的，例如，opt level 0 将不允许算子融合在一起。用户可以通过 fuse_opt_level 来启用它。

``` python
mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)

# 可以观察到优化后的模块包含的函数只有
# 一个单一的原始操作。
print(mod)
```

输出结果：

``` bash
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = fn (%p03: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p12: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    nn.conv2d(%p03, %p12, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %1 = %0(%x, %weight) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = fn (%p02: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p11: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p02, %p11) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %3 = %2(%1, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %4 = fn (%p01: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p1: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p01, %p1) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %5 = %4(%3, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %6 = fn (%p0: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p0, %p0) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %6(%5) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
```

### 使用 Sequential 应用一系列 Pass

如上所述应用 Pass 实际上很乏味，并且可能需要用户更好地了解它们之间的依赖关系。例如，融合目前在 let 绑定上效果不佳。因此，如果在融合之前应用了 `relay.transform.ToANormalForm()`，将无法融合可融合的算子，因为此过程会为每个表达式生成 let 绑定以规范 Relay 程序。

因此，Relay 提供了 `tvm.transform.Sequential`，使得开发者能够更容易地处理这些问题。他们通过显式指定每个 pass 所需的 pass，然后将它们打包为一个整体来实现。

例如，使用下面的 sequential 来应用相同的 pass。`tvm.transform.Sequential` 类似于 [torch.nn.sequential](https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential) 和 [mxnet.gluon.block](https://mxnet.apache.org/api/python/docs/_modules/mxnet/gluon/block.html)。例如，torch.nn.sequential 包含一系列 PyTorch 模块，这些模块将会用来构建网络。它侧重于网络层。相反，我们的 pass infra 中的 `tvm.transform.Sequential` 用于优化 pass。

``` python
# 通过 :py:class:`tvm.transform.Sequential` 执行一些传递
f = example()
mod = tvm.IRModule.from_expr(f)
# Glob 感兴趣的 passes。
seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(fuse_opt_level=2),
    ]
)
mod1 = seq(mod)
print(mod1)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %4 = fn (%p0: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p1: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %p2: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p3: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    %0 = nn.conv2d(%p0, %p1, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %1 = add(%0, %p2) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %2 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %3 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    add(%2, %3) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32], Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %4(%x, %weight, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
```

从改造后的 Relay 程序中，可以看到仍然有两个相同的加法运算。这是因为 `EliminateCommonSubexpr` 并未实际执行。原因是在 `tvm.transform.Sequential` 下，只有优化级别小于或等于 2 的 pass 才会默认执行。pass infra 为用户提供了一个配置界面来自定义想要执行的优化级别。

``` python
with tvm.transform.PassContext(opt_level=3):
    mod2 = seq(mod)
print(mod2)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %3 = fn (%p0: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p1: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %p2: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p3: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    %0 = nn.conv2d(%p0, %p1, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %1 = add(%0, %p2) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %2 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    add(%2, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32], Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %3(%x, %weight, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
```

现在可以看到两个相同的加法只保留一个。

用户可以使用 *disabled_pass* 配置选择性地禁用某些 pass，这与 Clang 和 GCC 等通用编译器使用的 *-fno-xxx* 选项类似。例如，可以禁用以下 EliminateCommonSubexpr，打印的模块将再次显示两个相同的加法操作。

``` python
with tvm.transform.PassContext(opt_level=3, disabled_pass=["EliminateCommonSubexpr"]):
    mod3 = seq(mod)
print(mod3)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %4 = fn (%p0: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p1: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %p2: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p3: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    %0 = nn.conv2d(%p0, %p1, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %1 = add(%0, %p2) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %2 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %3 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    add(%2, %3) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32], Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %4(%x, %weight, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
```

## 使用 Python 装饰器实现 Pass

下一个示例说明如何使用 Python 装饰器通过 pass infra 来自定义优化 pipeline。此功能极大地简化了pass 的实现。例如，用户可以简单地定义一个装饰类来进行函数级优化，如下例所示。 *transform_function* 包装了一个类，用 c 的倍数替换所有常量。稍后，访问给定模块中的每个函数，并且调用自定义 pass 时，函数中的每个常量都将被替换。

``` python
@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    """Simple test function to replace one argument to another."""

    def __init__(self, multiplier):
        self.multiplier = multiplier

    # 这个函数可以定义一个pass。
    def transform_function(self, func, mod, ctx):
        obj = self

        class ReplaceConstant(tvm.relay.ExprMutator):
            def visit_constant(self, c):
                return relay.multiply(obj.multiplier, c)

        return ReplaceConstant().visit(func)

f = example()
mod = tvm.IRModule.from_expr(f)
custom_pass = CustomPipeline(multiplier=relay.const(3, "float32"))
assert custom_pass.info.name == "CustomPipeline"
mod3 = custom_pass(mod)
print(mod3)
```

输出结果：

``` bash
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = multiply(3f /* ty=float32 */, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, %0) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = multiply(3f /* ty=float32 */, 2f /* ty=float32 */) /* ty=float32 */;
  %3 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %4 = multiply(%1, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %5 = add(%3, %4) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %6 = add(%5, %0) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %7 = add(%5, %0) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%6, %7) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
```

## 调试 Pass

TVM 为用户提供了即插即用式调试 Pass，通过特殊 pass (`PrintIR`) 转储整个模块的 IR，在完成某个 pass 后打印 IR。pass 序列示例的轻微修改版本如下所示，用于启用 IR 转储以进行 `FoldConstant` 优化。

``` python
f = example()
mod = tvm.IRModule.from_expr(f)
seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        tvm.transform.PrintIR(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(),
    ]
)
```

通过在 `FoldConstant` 之后插入 `PrintIR` pass，pass infra 将在 `FoldConstant` 完成时转储模块 IR。用户可以在任何想要调试的 pass 之后插入这个 pass 来查看优化效果。

此外，还有一个更灵活的调试机制，可以实现一个 `PassInstrument` 类来执行任意代码，不仅在每次传递之前和/或之后，而且在进入/退出 `PassContext` 时也可以。有关详细信息，参阅 [Pass Instrument](https://tvm.apache.org/docs/arch/pass_infra.html#pass-instrument-cpp-backend)。

这里使用 `tvm.instrument.pass_instrument` 装饰器来实现一个 PassInsturment 类，在每次执行之前打印 IR：

``` python
@tvm.instrument.pass_instrument
class PrintIR:
    """仅在 pass 执行之前打印 pass 的名称，IR。"""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):
    with tvm.target.Target("llvm"):
        # 执行优化。
        mod = seq(mod)
print(mod)

print("done")
```

输出结果：

``` bash
Running pass: {} The meta data of the pass - pass name: sequential, opt_level: 0, required passes: []

def @main(%x: Tensor[(1, 64, 56, 56), float32], %weight: Tensor[(64, 64, 3, 3), float32]) {
  %0 = add(meta[relay.Constant][0], meta[relay.Constant][0]);
  %1 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]);
  %2 = multiply(%0, 2f);
  %3 = add(%1, %2);
  %4 = add(%3, meta[relay.Constant][0]);
  %5 = add(%3, meta[relay.Constant][0]);
  add(%4, %5)
}

Running pass: {} The meta data of the pass - pass name: FoldConstant, opt_level: 2, required passes: []

def @main(%x: Tensor[(1, 64, 56, 56), float32], %weight: Tensor[(64, 64, 3, 3), float32]) {
  %0 = add(meta[relay.Constant][0], meta[relay.Constant][0]);
  %1 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]);
  %2 = multiply(%0, 2f);
  %3 = add(%1, %2);
  %4 = add(%3, meta[relay.Constant][0]);
  %5 = add(%3, meta[relay.Constant][0]);
  add(%4, %5)
}

/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
Running pass: {} The meta data of the pass - pass name: InferType, opt_level: 0, required passes: []

def @main(%x: Tensor[(1, 64, 56, 56), float32], %weight: Tensor[(64, 64, 3, 3), float32]) {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]);
  %1 = add(%0, meta[relay.Constant][0]);
  %2 = add(%1, meta[relay.Constant][1]);
  %3 = add(%1, meta[relay.Constant][1]);
  add(%2, %3)
}

Running pass: {} The meta data of the pass - pass name: PrintIR, opt_level: 0, required passes: []

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %3 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %3) /* ty=Tensor[(1, 64, 54, 54), float32] */
}

Running pass: {} The meta data of the pass - pass name: InferType, opt_level: 0, required passes: []

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %3 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %3) /* ty=Tensor[(1, 64, 54, 54), float32] */
}

Running pass: {} The meta data of the pass - pass name: EliminateCommonSubexpr, opt_level: 3, required passes: [
InferType, ]

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %3 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %3) /* ty=Tensor[(1, 64, 54, 54), float32] */
}

Running pass: {} The meta data of the pass - pass name: InferType, opt_level: 0, required passes: []

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */
}

Running pass: {} The meta data of the pass - pass name: InferType, opt_level: 0, required passes: []

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */
}

Running pass: {} The meta data of the pass - pass name: FuseOps, opt_level: 0, required passes: [
InferType, ]

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */
}

Running pass: {} The meta data of the pass - pass name: InferType, opt_level: 0, required passes: []

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %3 = fn (%p0: Tensor[(1, 64, 56, 56), float32], %p1: Tensor[(64, 64, 3, 3), float32], %p2: Tensor[(1, 64, 54, 54), float32], %p3: Tensor[(1, 64, 54, 54), float32], Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    %0 = nn.conv2d(%p0, %p1, padding=[0, 0, 0, 0]);
    %1 = add(%0, %p2);
    %2 = add(%1, %p3);
    add(%2, %2)
  };
  %3(%x, %weight, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */)
}

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %3 = fn (%p0: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p1: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %p2: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p3: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    %0 = nn.conv2d(%p0, %p1, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %1 = add(%0, %p2) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %2 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    add(%2, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32], Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %3(%x, %weight, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */
}

done
```

## 总结

本教程介绍了如何使用 pass infra 更方便地在 TVM 中编写和调用 pass。还讨论了调用 pass 的不同方式。使用 `tvm.transform.Sequential` 可以在很大程度上帮助用户简化处理多个优化过程及其依赖关系的工作。此外，还提供了一个示例来说明如何使用 `PrintIR` 和跟踪来调试 pass。

[下载 Python 源代码：use_pass_infra.py](https://tvm.apache.org/docs/_downloads/5499fb7d70e17c6aabf49246a978db52/use_pass_infra.py)

[下载 Jupyter Notebook：use_pass_infra.ipynb](https://tvm.apache.org/docs/_downloads/7ef14586a3b62fe120d97d5fedf72879/use_pass_infra.ipynb)
