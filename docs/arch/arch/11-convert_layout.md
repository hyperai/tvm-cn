---
title: Convert Layout Pass
sidebar_position: 210
---

# Convert Layout Pass

**作者：**[Animesh Jain](https://github.com/anijain2305)

## 1. 背景

数据布局格式（Data layout format）描述了数据在内存中的布局方式。例如，卷积算子的 TensorFlow 框架默认数据布局是 NHWC，即数据是四维的，并以行优先格式布局，N 为第一维，C 为最后一维。

数据布局在模型性能中起主要作用，对空间和时间局部性有重大影响。例如，TVM 中的 Intel x86 后端往往采用 NCHWc 布局，其中 C 维度以二维形式平铺，从而有效利用数据局部性（data locality）。同样，CUDA 后端的数据布局往往采用 NCHW 格式。

本质上，TVM 必须处理整个编译器工具链中的数据布局——框架解析器、Relay 布局转换和 TOPI schedule。随着我们转向第三方 codegen 集成（它可能有自己的数据布局限制），处理 TVM 工具链中所有级别的布局变得更具挑战性。因此，我们开发了一个新的 Relay pass——**ConvertLayout**——减少布局处理引发的问题。

若要了解 ConvertLayout Pass 的用法，请直接跳到本节第 4 部分 - 用法。

## 2. 动机和概述

下面看一个简单的场景，了解由于不同布局引发的问题——假设要为 ARM 边缘设备编译一个 TensorFlow NHWC 计算图。但是，假设目前在 TOPI for ARM 中仅支持 NCHW schedule。因此，框架布局和 TOPI 支持的布局之间存在不匹配。

处理这种不匹配的方法之一是，在每次卷积之前和之后插入布局转换，这样得到的卷积具有 NCHW 输入数据布局，并且可以使用 TOPI schedule。但是，由于存在过多的布局转换，这可能会导致性能下降。

其他用例中也遇到了类似的问题

* 无法在 NVIDIA GPU 上运行 TFLite 计算图。 TOPI 为 GPU 提供了仅限 NCHW 的 schedules。
* 为了支持不同的布局转换对，AlterOpLayout 用于卷积的逻辑越来越复杂。
* 由于额外的布局转换，导致 TF 计算图的性能并非最优。
* 第三方 codegen 集成中的复杂性，例如 TensorRT，它往往采用一种格式的数据布局。

为了解决这些问题，我们引入了 *ConvertLayout* pass。该 pass 设置了基础架构，它通过最小数量的数据布局转换，来更改整个计算图的数据布局。在理想情况下，只有 2 个数据布局转换，一个在开头，一个在结尾。以下是转换的示例

``` c++
# 原始计算图 - NHWC 格式的 2 个卷积。
fn (%x: Tensor[(1, 56, 56, 64), float32], %weight1: Tensor[(3, 3, 64, 32), float32], %weight2: Tensor[(3, 3, 32, 32), float32]) {
  %0 = nn.conv2d(%x, %weight1, padding=[1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %1 = nn.relu(%0);
  %2 = nn.conv2d(%1, %weight2, padding=[1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  nn.relu(%2)
}

# 在 ConvertLayout 之后 - 对于数据，在开始和结束时都有一个变换。
# 对于权重，有适应 NCHW 布局的变换。这些将被 FoldConstant pass 删除。
fn (%x: Tensor[(1, 56, 56, 64), float32], %weight1: Tensor[(3, 3, 64, 32), float32], %weight2: Tensor[(3, 3, 32, 32), float32]) {
  %0 = layout_transform(%x, src_layout="NHWC", dst_layout="NCHW") /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %1 = layout_transform(%weight1, src_layout="HWIO", dst_layout="OIHW") /* ty=Tensor[(32, 64, 3, 3), float32] */;
  %2 = nn.conv2d(%0, %1, padding=[1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 56, 56), float32] */;
  %3 = nn.relu(%2) /* ty=Tensor[(1, 32, 56, 56), float32] */;
  %4 = layout_transform(%weight2, src_layout="HWIO", dst_layout="OIHW") /* ty=Tensor[(32, 32, 3, 3), float32] */;
  %5 = nn.conv2d(%3, %4, padding=[1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 56, 56), float32] */;
  %6 = nn.relu(%5) /* ty=Tensor[(1, 32, 56, 56), float32] */;
  layout_transform(%6, src_layout="NCHW", dst_layout="NHWC") /* ty=Tensor[(1, 56, 56, 32), float32] */
}
```

## 3. 设计

在深入研究 ConvertLayout pass 之前，让我们根据算子对数据布局的敏感性将它们分为 3 类。此分类稍后将有助于了解 Convertlayout pass 的详细信息。

* **布局无关** - relu、pow 等。这些算子的功能和性能都不受数据布局的影响。
* **轻度布局敏感** - pad、concatenate 和 reduce算子（如 sum 等）。如果在它们之前进行布局转换，这些算子的某些属性会在功能上受到影响。但是，性能的差异不是很明显。对于这些算子来说，只需要适应之前的算子输出的数据布局。
* **重度布局敏感** - convolution、conv2d_transpose 等。这些算子在功能和性能方面都受到数据布局的严重影响。它们还具有数据布局作为 op 属性。通常，修改这些算子的输入数据布局是有益的（如果它不是高性能数据布局），而其余的*布局无关*和*轻度布局敏感*的算子会适应受*重度布局敏感*算子输出控制的布局。

现在来看两个相关的 Relay 算子属性。每个 Relay 算子都有属性，例如 InferType，可以由 TVM 开发者定义。通常，Relay pass 会逐个算子遍历计算图并读取这些算子属性。例如，InferType pass 查看算子的 InferType 属性，确定其输出 shape 和类型，然后将其传给下一个算子的 InferType 属性。

同样，在上下文中，有 2 个这样的属性——*FTVMConvertLayout* 和 *FInferCorrectLayout*。ConvertLayout pass 遍历计算图，并查看这两个属性和自动布局转换插入模块，从而处理数据布局。所以，整个过程可以分为 3 个步骤：

* 运行 FTVMConvertLayout 属性——这允许开发者将原始 Relay expr 转换为具有新布局的新 Relay expr，从而允许用户定义的布局更改。为了方便开发者，可以借助一个 Python 回调函数（仅用于重度布局敏感算子）。
* 运行 FTVMInferCorretLayout 属性——可以将其视为布局推理。它查看原始输入布局和新输入布局，它们要么来自先前的算子，要么来自 FTVMConvertLayout 修改后的 expr（如果已使用）。轻度布局敏感的算子可以借助它，让属性适应新的数据布局。每个算子都会有布局推理。
* 布局转换的自动插入——上一步的布局推理为输入 expr 设置新布局。若这些布局与原始布局不同，则此组件会自动插入布局转换。因此，开发者不需要为此组件做额外操作。

这些步骤按顺序发生在所有算子上，其中 ConvertLayout pass 不断将新布局传给下一个算子属性，最终导致逐个算子地修改整个计算图。下面看几个定义这两个属性的示例。

**FTVMConvertLayout - 布局更改的 Python 回调函数** - 用于*重度布局敏感*的算子。例如，可以返回具有新数据和内核布局的新卷积算子。若需要，其他 2 个组件将推理布局，并插入布局转换。转换为 NCHW 布局的卷积算子的示例如下：

``` python
@reg.register_convert_op_layout("nn.conv2d")
def convert_conv2d(attrs, inputs, tinfos, desired_layouts):
    """为 conv2d 算子注册 Convert Layout pass。

    参数
    ----------
    attrs : tvm.attrs.Attrs
        当前卷积的属性
    inputs : list of tvm.relay.Expr
        Relay expr 的 args 将合法化
    tinfos : list of types
        输入输出类型列表
    desired_layouts : list of layout strings
            定义我们想要的布局列表
            分别用于数据和内核输入的布局。

    返回
    -------
    result : tvm.relay.Expr
        转换后的 expr
    """

    from tvm import relay
    data, weight = inputs
    new_attrs = dict(attrs)

    # 我们期望指定 2 个所需的布局，一个用于数据，一个用于内核。
    assert len(desired_layouts) == 2, "A desired layout is expected for both of nn.conv2d's inputs"

    # 使用指定数据布局的所需布局中的第一个条目。
    # 此算子的预期布局顺序由此函数定义。
    desired_data_layout, desired_kernel_layout = map(str, desired_layouts)

    assert desired_data_layout != "default", "Data layout cannot be default"

    new_attrs['data_layout'] = desired_data_layout

    if desired_data_layout == 'NCHW':
        if desired_kernel_layout != 'default':
            new_attrs['kernel_layout'] = desired_kernel_layout
        else:
            new_attrs['kernel_layout'] = 'OIHW'
        # 布局转换的实际插入在内部进行
        # 通过 ConvertLayout  pass。
        return relay.nn.conv2d(data, weight, **new_attrs)

    raise ValueError('Layout %s is not yet supported' % desired_data_layout)
```

**FInferCorrectLayout - 布局推理** - 目前，此属性仅在 C++ 中公开。此函数采用原始输入布局和新输入布局（由前一个算子传递，或由用于布局更改的 Python 回调函数传递），并推理最终数据布局。每个算子都会调用布局推理。算子类别不同，其使用也会有所不同。

对于布局无关的算子，只要返回这个函数中新的数据布局。对于轻度布局和重度布局敏感的算子，可以更改算子属性（如用于连接的轴，和用于填充的 pad_width），以便适应新的数据布局，防止插入布局转换。下面看几个例子来更好地理解这一点。

第一个例子是布局无关的算子。这些算子没有属性受数据布局影响，所以只适应新的布局。

``` c++
// 设置算子属性，如下所示
// .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

// 采用任意输入布局，并复制到输出。
inline Array<Array<Layout> > ElemwiseArbitraryLayout(const Attrs& attrs,
                                                     const Array<Layout>& new_in_layouts,
                                                     const Array<Layout>& old_in_layouts,
                                                     const Array<Array<IndexExpr>> &old_in_shapes) {
  Layout ret;

  if (new_in_layouts.defined()) {
    ICHECK_GE(new_in_layouts.size(), 1);
    ret = new_in_layouts[0];
  } else {
    for (size_t i = 0; i < old_in_layouts.size(); ++i) {
      if (old_in_layouts[i].defined()) {
        ret = old_in_layouts[i];
        break;
      }
    }
  }

  return Array<Array<Layout> >{Array<Layout>(old_in_layouts.size(), ret), {ret}};
}
```

第二个示例适用于轻度布局敏感的算子——batch normalization。若从 NHWC 转到 NCHW 数据布局时，BatchNorm 的 axis 算子必须更改。 （重度布局敏感的算子处理类似。）

``` c++
Array<Array<Layout>> BatchNormInferCorrectLayout(const Attrs& attrs,
                                                 const Array<Layout>& new_in_layouts,
                                                 const Array<Layout>& old_in_layouts,
                                                 const Array<Array<IndexExpr>>& old_in_shapes) {
  BatchNormAttrs* param = const_cast<BatchNormAttrs*>(attrs.as<BatchNormAttrs>());

  size_t axis =
      param->axis < 0 ? param->axis + old_in_shapes[0].size() : static_cast<size_t>(param->axis);

  Layout ret = Layout::Undef();

  // 例如，考虑 old_layout = NHWC, and new_layout = NCHW, and param->axis = 3
  if (new_in_layouts.defined() && old_in_layouts.defined()) {
    // 获取新的 C 轴。提取旧布局中的 dim。在下一个布局中找到该 dim 的索引。

    // 以下代码给出 bn_dim = C as old_layout = NHWC, axis = 3
    const auto& bn_dim = old_in_layouts[0][axis];

    // new_index 为 1，因为 new_layout = NCHW 且 bn_dim 为 C
    auto new_index = new_in_layouts[0].IndexOf(bn_dim);

    // 修改 layout-dependent 属性 - axis 为 1。
    param->axis = new_index;

    // 最后，适应新的布局。
    ret = new_in_layouts[0];

  } else if (old_in_layouts.defined()) {
    ret = old_in_layouts[0];
  }

  // 如果新旧布局都未定义，则无需更改。
  // 在这种情况下，ConvertLayout pass 会跳过布局转换的自动插入。

  // 以下代码对教程并不重要。但是，布局推理需要定义
  // 所有输入和输出数据布局的布局。对于 batch norm，其他输入
  // 并且输出是输入中长度为 C dim 的向量。所以，我们设置另一个
  // 布局为 C。BN 有 5 个输入，3 个输出。最后 4 个输入和最后 2 个输出
  // 有「C」布局。
  Layout c_layout = Layout("C");

  return Array<Array<Layout>>{{ret, c_layout, c_layout, c_layout, c_layout},
                              {ret, c_layout, c_layout}};
}
```

## 4. 用法

ConvertLayout pass 非常易于使用。pass 不是默认 relay.build pipeline 的一部分。预期用途是在 framework-to-relay 解析器和 relay.build 模块调用之间调用它。

为了指定要转换到的布局，创建一个映射，该映射由重度布局敏感的算子指向该算子期望布局的列表。以下第一个示例指定了数据布局，允许内核布局自动转换为 TVM 支持的布局（针对特定的数据布局和算子）。这是通过使用「default」关键字指定的。

第二个示例显示了如何转换为指定内核布局。注意，以下示例将转换为相同的布局，即 {‘nn.conv2d’: [‘NCHW’, ‘default’]} == {‘nn.conv2d’: [‘NCHW’, ‘OIHW’]}

``` python
# TFlite 框架到 Relay 解析器 - 默认布局是 NHWC
mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict=shape_dict,
                                         dtype_dict=dtype_dict)

# 假设模型的重度布局敏感算子仅包含 nn.conv2d
desired_layouts = {'nn.conv2d': ['NCHW', 'default']}

# 将布局转换为 NCHW
# RemoveUnunsedFunctions 用于清理计算图。
seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                relay.transform.ConvertLayout(desired_layouts)])
with tvm.transform.PassContext(opt_level=3):
    mod = seq(mod)

# 调用 Relay 编译
with relay.build_config(opt_level=3):
     graph, lib, params = relay.build(mod, target, params=params)
```

``` python
desired_layouts = {'nn.conv2d': ['NCHW', 'OIHW']}
pass = relay.transform.ConvertLayout(desired_layouts)
```

布局的顺序由 register_convert_op_layout(“OPNAME”) 的实现定义，可以参考明确说明了预期布局的文档字符串。以上示例中是 [data_layout, kernel_layout]。

当前的实现几乎支持图像分类模型中所有常用算子。但是，如果在计算图中遇到太多数据布局转换，则很可能存在个别算子，需要特殊处理其布局，如第 3 节所述。在这种情况下可以参考的 pull requests 是

* [Batch Norm](https://github.com/apache/tvm/pull/4600) 的布局推理 - Batch normalization 属于轻度敏感算子。 PR 展示了如何处理 batch norm 的布局推理。
* [Convolution](https://github.com/apache/tvm/pull/4335) Python 回调函数 - 对于重度敏感的算子，还需要执行 Python 回调函数。 PR 展示了如何为卷积算子定义 Python 回调函数。
