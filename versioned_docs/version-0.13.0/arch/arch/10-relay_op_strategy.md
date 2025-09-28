---
title: Relay 算子策略
sidebar_position: 200
---

# Relay 算子策略

为了将 Relay 算子降级为 TOPI 库中定义的实现，需要为每个 Relay 算子注册一个 compute 和 schedule 函数。然而，compute 和 schedule 函数通常是针对每个 target 定制化的，而且，即使对于同一个 target，也可能有多种可用算法和实现。算子策略是为了应对这种复杂的局面而引入的，它让开发者可以为每个算子和 target 定义灵活的降级策略。

## 算子策略设计

算子策略的基本元素是 `OpImplementation`。它包括一对 compute 和 schedule 函数、实现的名称和优先级（优先级的使用在 [从算子策略中选择实现](#select-implementation-from-op-strategy) 中进行了解释）。

`OpStrategy` 包括一个 `OpSpecialization` 列表。每个 `OpSpecialization` 都包含一个与 `SpecializedCondition` 关联的 `OpImplementation` 列表（参见 `include/tvm/te/schedule.h` 中的定义）。 `SpecializedCondition` 可以为 null，表示实现普遍适用；否则，仅在满足特定条件时才考虑实现。`SpecializedCondition` 包含一个子句列表（它们在张量表达式中以联合范式（CNF）的形式定义），并且仅支持在张量 shape 上的情况。

最后，策略函数或 `FTVMStrategy` 确定在给定工作负载的情况下，使用哪一对 compute 和  schedule 函数，并且要注册到每个 Relay 算子。 `FTVMStrategy` 是一个通用函数（参见 `include/tvm/target/generic_func.h`），所有的 target 都可以覆盖它。函数签名是

``` c++
OpStrategy(const Attrs& attrs, const Array<Tensor>& inputs, const Type& out_type, const Target& target)
```

该函数在给定 op 属性、输入张量、输出类型和要编译到的 target 的情况下，可返回一个 `OpStrategy`。

## 编写策略函数

推荐开发者用 Python 编写策略函数，因为大多数 TOPI 计算和 schedule 函数都是用 Python 编写的。Python 的 `pyton/tvm/relay/op/op.py` 中提供 `OpStrategy` 类。它只有一个API，即为策略添加一个实现：

``` python
def add_implementation(self, compute, schedule, name="default", plevel=10)
```

接下来以 `topk` 为例，说明如何编写 `FTVMStrategy` 函数：

``` python
# 添加到 python/tvm/relay/op/strategy/generic.py
@override_native_generic_func("topk_strategy")
def topk_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_topk(topi.topk),
        wrap_topi_schedule(topi.generic.schedule_topk),
        name="topk.generic")
    return strategy

# 添加到 python/tvm/relay/op/strategy 中的每个目标文件，例如 x86.py、cuda.py 等。
@topk_strategy.register(["cuda", "gpu"])
def topk_strategy_cuda(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_my_new_op(topi.cuda.topk),
        wrap_topi_schedule(topi.cuda.schedule_topk),
        name="topk.cuda")
    return strategy
```

本例使用 `topi.cuda.topk` 和 `topi.cuda.schedule_topk` 作为 CUDA 或 GPU target 的 compute 和  schedule 函数，而对其余 target 使用 TOPI 通用计算和  schedule。

注意，用两个包装函数来包装 topi compute 和  schedule，使其符合所需的函数签名（参阅 `include/tvm/relay/op_attr_types.h` 中的 `FTVMCompute` 和 `FTVMSchedule`）。通常需要为每个算子编写一个自定义的计算包装函数，来获取 op 属性中不同的字段。

以上例子展示了一个非常基本的策略函数，它只在策略中添加了一个实现。但是对于许多复杂的算子而言，可能需要添加使用不同算法的多个实现。例如，可以同时使用 direct 算法和 winograd 算法来计算 conv2d 操作。编写如下策略函数来实现：

``` python
strategy.add_implementation(
    wrap_compute_conv2d(topi.cuda.conv2d_nchw),
    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw),
    name="conv2d_nchw.cuda",
    plevel=10)

if winograd_condition:
    strategy.add_implementation(
        wrap_compute_conv2d(topi.cuda.conv2d_nchw_winograd),
        wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_winograd),
        name="conv2d_nchw_winograd.cuda",
        plevel=15)
```

这个例子中，我们向 conv2d 策略添加了两个实现，其中 winograd 算法仅在 `winograd_condition` 为 True 时添加。当 winograd_condition 为 True 时，`“conv2d_nchw_winograd.cuda”` 实现用于编译 conv2d，因为它具有更高的优先级（AutoTVM 模板的实现可以更改此设置。详细信息参阅 [从 op 策略中选择实现](#select-implementation-from-op-strategy)）。否则，使用 `“conv2d_nchw.cuda”`。

可以将上面的示例扩展到第三方库实现。例如，当 cblas 包含在 target 中时，可以在 cblas 库中添加调用内核的实现。

``` python
if "cblas" in target.libs:
    strategy.add_implementation(
        wrap_compute_dense(topi.x86.dense_cblas),
        wrap_topi_schedule(topi.x86.schedule_dense_cblas),
        name="dense_cblas.x86",
        plevel=15)
```

此外，可以添加针对特定 shape 范围的实现。以下代码显示了一个密集策略示例，该示例添加了一个专门用于 `m` > 16 的实现。硬编码 Python 条件（如上例所示）和特定条件之间的主要区别在于，当输入张量具有符号特征的 shapes 时，前者允许 TVM 生成多个内核 shape。编译引擎会生成一个 dispatch 函数，当满足相应条件时会调用专门的内核；否则，调用没有关联特殊条件的内核（本例中为 `dense_common`）。这部分仍在开发中。完成后将提供更多详细信息。

``` python
def dense_strategy(attrs, inputs, out_type, target):
    m = inputs[0].shape[0]
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dense(dense_compute1),
        wrap_topi_schedule(dense_schedule1),
        name="dense_common")

    with tvm.te.SpecializedCondition(m > 16):
        strategy.add_implementation(
            wrap_compute_dense(dense_compute2),
            wrap_topi_schedule(dense_schedule2),
            name="dense_for_large_m",
            plevel=15)

    return strategy
```

## 将策略函数注册到算子

定义了一个算子的策略函数之后，可以将策略函数注册到这个算子

``` python
register_strategy("topk", strategy.topk_strategy)
```

但是，为算子编写策略函数需要花费很多精力。因此，我们提供了两种更简单的方法。

第一种方法是，对于具有单射、广播或归约模式的算子，可以分别调用 `register_injective_schedule`、`register_broadcast_schedule` 和 `register_reduce_schedule`。每个 target 都已经注册了这些模式的 schedule 函数，并且可以应用于这些算子。假设所有 target 上的 compute 函数都是相同的，并且 `FTVMCompute` 要在调用注册 schedule 前注册到 op。

``` python
register_broadcast_schedule("add")
```

第二种方法是，对于不具备前面提到的这些常见模式，但对所有 targets 具有相同 compute 函数的算子，可以使用 `register_schedule` API。`FTVMSchedule` 函数更容易编写，因为只需提供要用的 schedule 函数即可。以下代码片段展示了用于池化的 `FTVMSchedule` 函数。

``` python
# 添加到 python/tvm/relay/op/strategy/generic.py
@generic_func
def schedule_pool(attrs, outs, target):
    with target:
        return topi.generic.schedule_pool(outs, attrs.layout)

# 添加到 python/tvm/relay/op/strategy 中的每个目标文件，例如 x86.py、cuda.py 等。
@schedule_pool.register("cpu")
def schedule_pool_cpu(attrs, outs, target):
    ...
```

为算子创建 `FTVMSchedule` 后，可以使用 `register_schedule` 注册策略：

``` python
register_schedule("nn.max_pool2d", strategy.schedule_pool)
```

## 为新 Target 注册策略

有两种方法可以为新 target 注册策略。更直接的方法是在目录 `python/tvm/relay/op/strategy` 中添加一个新的目标文件。只需为在新 target 上实现的算子自定义策略，其他的算子复用通用策略。

或者，也可以在 TVM Python 库之外为新 target 注册策略。以下代码片段显示了如何执行此操作的示例。`vta/python/vta/top/op.py` 中可以找到更多示例。

``` python
@relay.op.strategy.conv2d_strategy.register("mytarget")
def conv2d_strategy_mytarget(attrs, inputs, out_type, target):
    ...
```

## 从 Op 策略中选择实现

在编译过程中，Relay 编译引擎要确定当有多个算子时，使用哪个算子来实现。选择策略的工作原理如下。

当算子或融合算子的输入张量都具有恒定 shape 时，编译引擎首先会根据 AutoTVM 调整日志找到最佳实现。如果没有 AutoTVM 模板的实现，或所有 AutoTVM 模板都有回退配置，则选择具有最高优先级的实现。在这种情况下，具有相同优先级的实现会导致未定义的行为，并且实现方法的选择具有随机性。

具有符号特征输入 shape 的算子的选择策略仍在开发中。目前，如果所有输入张量都具有符号特征的 shape，则只有具有最高优先级的实现将用于此算子。实现后将更新这部分。

可以在编译 Relay 模型之前添加以下代码行进行调试，来了解每个算子的实现（implementation）。

``` python
logging.getLogger("te_compiler").setLevel(logging.INFO)
logging.getLogger("te_compiler").addHandler(logging.StreamHandler(sys.stdout))
```
