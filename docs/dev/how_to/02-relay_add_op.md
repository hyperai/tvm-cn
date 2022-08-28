---
title: 向 Relay 中添加算子
---

# 向 Relay 中添加算子

本文档将以添加 [cumulative product](https://github.com/apache/tvm/pull/7722) 算子的 PR（基于 [cumulative sum](https://github.com/apache/tvm/pull/7334) 算子 PR）为例，介绍在 Relay 中注册一个新的 TVM 算子所需的步骤。

注册一个新的算子需要如下几个步骤：

1. 添加一个属性节点，声明在编译时已知的固定参数
2. 为算子编写一个类型关系，以整合到 Relay 的类型系统中
3. 使用 C++ 中的 `RELAY_REGISTER_OP` 宏，为编译器注册算子的数量、类型和其他提示
4. 编写算子的计算方式
5. **Register the compute, schedule with the relay operator【待确认】**
6. 定义一个为算子产生调用节点的 C++ 函数，并为该函数注册一个 Python API hook
7. 将上述 Python API hook 放在一个更简洁的接口中
8. 为新的 Relay 算子编写测试

## 1. 定义属性节点

属性是在编译时已知的固定参数。卷积算子的步长和扩张可能属于卷积算子属性节点字段的恰当示例。

属性应在文件夹 [include/tvm/relay/attrs/](https://github.com/apache/tvm/tree/main/include/tvm/relay/attrs) 内的文件中定义。

最终我们要创建一个算子，它的接口可以在最终的 Python 接口中清晰可见：

``` python
def cumprod(data, axis=None, dtype=None, exclusive=None):
    """Numpy style cumprod op. Return the cumulative inclusive product of the elements along a given axis.
    参数
    ----------
    data : relay.Expr 类型
        算子的输入数据。
    axis : int 类型，可选
        Axis along which the cumulative product is computed. The default (None) is to compute the cumprod over the flattened array.
    dtype : string 类型，可选
        Type of the returned array and of the accumulator in which the elements are multiplied.
        如果 dtype 没有被指定, 那么它默认为 data 的 dtype。
    exclusive : bool 类型，可选
        If true will return exclusive product in which the first element is not included. In other terms, if true, the j-th output element would be the product of the first (j-1) elements. Otherwise, it would be the product of the first j elements. The product of zero elements will be 1.
    返回
    -------
    result : relay.Expr 类型
        如果 axis 不为空的话，结果的大小和形状和 data 一样。
        如果 axis 为空的话, 结果是一个一维数组。
    """
```

`cumsum()` 存在类似的接口。

因此，在 `include/tvm/relay/attrs/transform.h` 中定义属性时，可以选择算子的 axis、accumulation dtype 及 exclusivity 作为结构体的合适字段。

``` c++
/*! 用在 cumsum 和 cumprod 算子中的简单属性 */
struct ScanopAttrs : public tvm::AttrsNode<ScanopAttrs> {
  Integer axis;
  DataType dtype;
  Bool exclusive = Bool(false);
  TVM_DECLARE_ATTRS(ScanopAttrs, "relay.attrs.ScanopAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis to operate over").set_default(NullValue<Integer>());
    TVM_ATTR_FIELD(dtype).describe("Output data type").set_default(NullValue<DataType>());
    TVM_ATTR_FIELD(exclusive)
        .describe("The first element is not included")
        .set_default(Bool(false));
  }
};
```

## 2. 编写类型关系

为了提高注册算子的灵活性，在 Relay 中表示类型时更突出，算子使用输入和输出类型之间的关系进行类型化。这些关系被表示为函数，它接收一个输入类型和输出类型的列表（这些类型中的任何一个都可能是不完整的），然后返回一个满足关系的输入和输出类型的列表，包括可以在编译时静态确定的形状信息。基本上，一个算子的关系除了计算输出类型外，还可以执行所有必要的类型化规则（即通过检查输入类型）。

在 `src/relay/op/tensor/transform.cc` 中可以找到 cumulative product 与 cumulative product 算子的类型关系。

``` c++
TVM_REGISTER_NODE_TYPE(ScanopAttrs);
bool ScanopRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
    // types: [data, output]
    ICHECK_EQ(types.size(), 2) << "Expects two types, one for the input and another for the output";
    const auto* data = types[0].as<TensorTypeNode>();
    if (data == nullptr) {
        ICHECK(types[0].as<IncompleteTypeNode>())
        << "Scanop: expect input type to be TensorType but get " << types[0];
        return false;
    }

    const auto* param = attrs.as<ScanopAttrs>();

    auto dtype = param->dtype;
    if (dtype.is_void()) {
        dtype = data->dtype;
    }

    if (param->axis.defined()) {
        reporter->Assign(types[1], TensorType(data->shape, dtype));
    } else {
        auto prod = data->shape[0];
        for (size_t i = 1; i < data->shape.size(); ++i) {
            prod = prod * data->shape[i];
        }
        reporter->Assign(types[1], TensorType({prod}, dtype));
    }

    return true;
}
```

## 3. 将参数数量和属性与算子关联起来

注册新算子的名称，并为其添加调用接口的注解。C++ 中的 `RELAY_REGISTER_OP` 宏允许开发者在 Relay 中指定一个算子的以下信息：

* 参数数量
* 位置参数的名称和描述
* 支持级别（1 表示内部固有的；更高的数字表示集成度低或外部支持的算子）
* 算子的类型关系
* 其他在优化算子时有用的注解

再次将其添加到 `src/relay/op/tensor/transform.cc` 中：

``` c++
RELAY_REGISTER_OP("cumsum")
    .describe(
        R"doc(Return the cumulative sum of the elements along a given axis.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Cumsum", ScanopRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

RELAY_REGISTER_OP("cumprod")
    .describe(
        R"doc(Return the cumulative product of the elements along a given axis.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Cumprod", ScanopRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
```

在这种情况下，`TOpPattern` 是对编译器关于算子执行的计算模式的提示，这对于融合算子可能很有用。`kOpaque` 提示 TVM 无需融合这个算子。

## 4. 定义算子的计算

为算子定义接口后，仍需定义如何执行 cumulative sum 和 cumulative product 的实际计算。

假设算子计算的实现方式，经过了多轮测试且表现良好。推荐查看 [张量表达式教程](../../user_guide/user_tutorial/tensor_expr)、[TVM 算子清单 (topi)](../../user_guide/user_tutorial/TOPI)、[python/tvm/topi/scan.py](https://github.com/apache/tvm/blob/main/python/tvm/topi/scan.py) 中 cumulative sum 及 cumulative product 相关实现案例，以及 [python/tvm/topi/cuda/scan.py](https://github.com/apache/tvm/blob/main/python/tvm/topi/cuda/scan.py) 中的 GPU 版本。在 cumulative sum 及 cumulative product 算子中，可以直接用 [TIR](https://tvm.apache.org/docs/reference/api/python/tir.html#api-python-tir)，张量表达式及 topi 降级后表示为 TIR。

## 5. 将计算 (compute) 和策略 (strategy) 与 Relay 关联起来

实现计算函数后，需要将其与 Relay 算子粘合在一起。在 TVM 中，这意味着不仅要定义 computation，还要定义算子的 schedule。策略决定使用哪种 computation 及 schedule。例如，对于二维卷积，识别出这属于一种深度卷积后，最终将其分配给一个更有效的 computation 和 schedule。

实际上除了在 CPU 和 GPU 的实现之间进行调度外，基本没有类似需求。在  `python/tvm/relay/op/strategy/generic.py` 和 `python/tvm/relay/op/strategy/cuda.py` 中，我们添加了如下策略：

``` python
def wrap_compute_scanop(topi_compute):
    """Wrap scanop style topi compute"""

    def _compute_scanop(attrs, inputs, _):
        return [topi_compute(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]

    return _compute_scanop

@override_native_generic_func("cumsum_strategy")
def cumsum_strategy(attrs, inputs, out_type, target):
    """cumsum 基本策略"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cumsum),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="cumsum.generic",
    )
    return strategy

@override_native_generic_func("cumprod_strategy")
def cumprod_strategy(attrs, inputs, out_type, target):
    """cumprod 基本策略"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cumprod),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="cumprod.generic",
    )
    return strategy

@cumsum_strategy.register(["cuda", "gpu"])
def cumsum_strategy_cuda(attrs, inputs, out_type, target):
    """cumsum cuda 策略"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cuda.cumsum),
        wrap_topi_schedule(topi.cuda.schedule_scan),
        name="cumsum.cuda",
    )
    return strategy

@cumprod_strategy.register(["cuda", "gpu"])
def cumprod_strategy_cuda(attrs, inputs, out_type, target):
    """cumprod cuda 策略"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cuda.cumprod),
        wrap_topi_schedule(topi.cuda.schedule_scan),
        name="cumprod.cuda",
    )
    return strategy
```

每个策略都定义了写入的 compute 以及在 `add_implementation()` 中使用的 schedule。最后，将 strategy 和 compute 与`python/tvm/relay/op/_transform.py` 中定义的 Relay 算子关联起来。

``` python
# cumsum
@_reg.register_compute("cumsum")
def compute_cumsum(attrs, inputs, output_type):
    """cumsum 的计算定义"""
    return [topi.cumsum(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]

_reg.register_strategy("cumsum", strategy.cumsum_strategy)
_reg.register_shape_func("cumsum", False, elemwise_shape_func)

# cumprod
@_reg.register_compute("cumprod")
def compute_cumprod(attrs, inputs, output_type):
    """cumprod 的计算定义"""
    return [topi.cumprod(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]

_reg.register_strategy("cumprod", strategy.cumprod_strategy)
_reg.register_shape_func("cumprod", False, elemwise_shape_func)
```

shape 函数用于确定 output shape，给定一个动态 shaped tensor。在这种情况下，TVM 的 output shape 与 input shape 保持一致。

## 6. 创建 Relay 调用节点并提供 Python Hook

现在已经有了一个可以运行的算子，接下来只需通过一个 Relay 调用节点 (Relay Call Node) 正确地调用即可。这一步需要简单地编写一个函数，接收算子的参数（作为 Relay 表达式），并向算子返回一个的调用节点（即应该被放在调用算子的 Relay AST 中的节点）。

目前不支持调用属性和类型参数（最后两个字段），所以只需使用  `Op::Get` 从算子注册表中获取算子信息，并将参数传递给调用节点（如下所示）。在 `src/relay/op/tensor/transform.cc`：

``` c++
Expr MakeCumsum(Expr data, Integer axis, DataType dtype, Bool exclusive) {
    auto attrs = make_object<ScanopAttrs>();
    attrs->dtype = dtype;
    attrs->axis = axis;
    attrs->exclusive = exclusive;
    static const Op& op = Op::Get("cumsum");
    return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.cumsum").set_body_typed(MakeCumsum);

Expr MakeCumprod(Expr data, Integer axis, DataType dtype, Bool exclusive) {
    auto attrs = make_object<ScanopAttrs>();
    attrs->dtype = dtype;
    attrs->axis = axis;
    attrs->exclusive = exclusive;
    static const Op& op = Op::Get("cumprod");
    return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.cumsum").set_body_typed(MakeCumprod);
```

其中 `TVM_REGISTER_GLOBAL` 通过 `relay.op._make.cumsum(...)` 和 `relay.op._make.cumsum(...)` 分别暴露 (expose) Python 中的 `MakeCumsum` 和 `MakeCumprod` 函数。

## 7. 包含一个更简洁的 Python API hook

通常 Relay 中约定俗成的是，通过 `TVM_REGISTER_GLOBAL` 导出的函数应该包装在单独的 Python 函数中，而不是直接在 Python 中调用。对于算子，我们在  `python/tvm/relay/op/transform.py` 中提供了更简洁的接口：

``` python
def cumsum(data, axis=None, dtype=None, exclusive=None):
    return _make.cumsum(data, axis, dtype, exclusive)

def cumprod(data, axis=None, dtype=None, exclusive=None):
    return _make.cumprod(data, axis, dtype, exclusive)
```

注意，这些 Python wrapper 也可能为算子提供更简洁的接口。例如 `concat` 算子被注册为只接受一个算子（即一个带有要连接的张量的元组），但是 Python wrapper 将张量作为参数，并在产生调用节点之前将它们组合成一个元组。

``` python
def concat(*args):
    """围绕零轴连接输入张量。

    参数
    ----------
    args: Tensor 列表

    返回
    -------
    tensor: 连接的张量。
    """
    tup = Tuple(list(args))
    return _make.concat(tup)
```

## 8. 编写单元测试

更多用于 cumulative sum 和 cumulative product 算子的单元测试示例，请查看 [tests/python/relay/test_op_level3.py](https://github.com/apache/tvm/blob/main/tests/python/relay/test_op_level3.py)。

# 其他主题

## 梯度算子

梯度算子对于在 Relay 中编写可微分程序很重要。虽然 Relay 的 autodiff 算法可以得到优秀的语言结构的微分，但算子是不透明的。因为 Relay 无法查看它的实现，所以必须提供明确的微分规则。

Python 和 C++ 都可用于编写梯度算子，这里重点介绍更为常用的 Python 实例。

## 在 Python 中添加梯度算子

Python 梯度算子集合可以在 `python/tvm/relay/op/_tensor_grad.py` 中找到 。本部分内容将详细介绍两个有代表性的例子：`sigmoid` 和 `multiply`。

``` python
@register_gradient("sigmoid")
def sigmoid_grad(orig, grad):
    """返回 [grad * sigmoid(x) * (1 - sigmoid(x))]."""
    return [grad * orig * (ones_like(orig) - orig)]
```

这里的输入是原始算子 `orig` 以及梯度算子 `grad`，返回是一个列表，其中第 i 个索引的元素，是算子相对于算子第 i 个输入的导数。通常，梯度算子将返回一个列表，其元素的个数和基础算子 (base operator) 的输入一样多。

进一步分析这个定义之前，先回忆一下 sigmoid 函数的导数：∂σ/∂x=σ(x)(1−σ(x))。上面的定义看起来类似于数学定义，但有一个重要的补充：

术语 `orig * (ones_like(orig) - orig)` 直接匹配导数，因为这里的 `orig` 是 sigmoid 函数。除了要了解如何计算该函数的梯度之外，还要掌握该梯度与其他梯度组合的方法，即在整个程序中累积梯度。这就是 `grad` 的作用。在表达式  `grad * orig * (ones_like(orig) - orig)`中，乘以 `grad` 指定了到目前为止如何用梯度组合导数。

接下来请看 `multiply`的示例：

``` python
@register_gradient("multiply")
def multiply_grad(orig, grad):
    """返回 [grad * y, grad * x]"""
    x, y = orig.args
    return [collapse_sum_like(grad * y, x),
            collapse_sum_like(grad * x, y)]
```

在此示例中，返回列表中有两个元素，因为 `multiply` 是二元运算符 (binary operator)。如果 f(x,y) = xy，偏导数是 ∂f / ∂x = y 和 ∂f / ∂y = x。

与 `sigmoid` 相比，`multiply` 需要一个额外的步骤，因为 `multiply` 具有广播语义 (broadcasting semantics)。由于 `grad` 的 shape 可能与输入 shape 不匹配，所以我们使用 `collapse_sum_like` 来获取 `grad * <var>` 项的内容，并使其 shape 与做微分的输入 shape 相匹配。

## 在 C++ 中添加梯度算子

在 C++ 中添加梯度算子的方法，与在 Python 中添加梯度算子类似，但注册的接口略有不同。

首先，确保 `src/relay/transforms/pattern_utils.h` 被包含在内。它提供了用于在 Relay AST 中创建节点的辅助函数。定义梯度算子的方式与 Python 类似：

``` c++
tvm::Array<Expr> MultiplyGrad(const Expr& orig_call, const Expr& output_grad) {
    const Call& call = orig_call.Downcast<Call>();
    return { CollapseSumLike(Multiply(output_grad, call.args[1]), call.args[0]),
             CollapseSumLike(Multiply(output_grad, call.args[0]), call.args[1]) };
}
```

注意，在 C++ 中不能使用与 Python 相同的运算符重载 (operator overloading)，而是需要向下转换，因此实现更加冗长。即便如此，我们仍然可以轻易地验证这个定义反映了 Python 中先前的例子。

要注册梯度算子，这里无需使用 Python 修饰器，只需要在基础算子注册的末尾添加 `set_attr` 调用 "FPrimalGradient" 即可。

``` c++
RELAY_REGISTER_OP("multiply")
    // ...
    // 设置其他属性
    // ...
    .set_attr<FPrimalGradient>("FPrimalGradient", MultiplyGrad);
```