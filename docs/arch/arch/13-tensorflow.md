---
title: TensorFlow 前端
sidebar_position: 230
---

# TensorFlow 前端

TensorFlow 前端有助于将 TensorFlow 模型导入 TVM。

支持的版本：

* 1.12 及以下

测试模型：

* Inception（V1/V2/V3/V4）
* Resnet（全部）
* Mobilenet（V1/V2 全部）
* Vgg（16/19）
* BERT（基础/3 层）

## 为推理准备模型

### 删除不需要的节点

导出过程将删除许多不需要的推理节点，但还是会留下一些。需要手动删除的节点有：

* Dropout，包括 [Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) 和 [DropoutWrapper](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/nn/rnn_cell/DropoutWrapper?hl=hr)
* [Assert](https://www.tensorflow.org/api_docs/python/tf/debugging/Assert)

### 将 None Dimensions 转换为常量

TVM 对动态张量 shape 的支持最少。为 `None` 的维度应替换为常量。例如，模型可以接受 shape 为 `(None,20)` 的输入。这应该转换为类似 `(1,20)` 的 shape。应相应地修改模型以确保这些 shape 在整个计算图中匹配。

### 导出

TensorFlow 前端需要一个冻结的 protobuf（.pb）或保存的模型作为输入。它目前不支持检查点（.ckpt）。TensorFlow 前端所需的 graphdef 可以从活动 session 中提取，或者使用 [TFParser](https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/tensorflow_parser.py) 辅助类。

导出模型时应进行一些转换，以准备模型进行推理。设置 `add_shapes=True` 也很重要，因为这会将每个节点的输出 shape 嵌入到图中。这是一个将模型导出为给定会话的 protobuf 的函数：

``` python
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

def export_pb(session):
    with tf.gfile.GFile("myexportedmodel.pb", "wb") as f:
        inputs = ["myinput1", "myinput2"] # 替换为你的输入名称

        outputs = ["myoutput1"] # 替换为你的输出名称
        graph_def = session.graph.as_graph_def(add_shapes=True)
        graph_def = tf.graph.util.convert_variables_to_constants(session, graph_def, outputs)
        graph_def = TransformGraph(
            graph_def,
            inputs,
            outputs,
            [
                "remove_nodes(op=Identity, op=CheckNumerics, op=StopGradient)",
                "sort_by_execution_order", # 每次转换后按执行顺序排序，以确保正确的节点排序
                "remove_attribute(attribute_name=_XlaSeparateCompiledGradients)",
                "remove_attribute(attribute_name=_XlaCompile)",
                "remove_attribute(attribute_name=_XlaScope)",
                "sort_by_execution_order",
                "remove_device",
                "sort_by_execution_order",
                "fold_batch_norms",
                "sort_by_execution_order",
                "fold_old_batch_norms",
                "sort_by_execution_order"
            ]
        )
        f.write(graph_def.SerializeToString())
```

另一种方法是 [导出并冻结计算图](https://github.com/tensorflow/models/tree/master/research/slim#exporting-the-inference-graph)。

## 导入模型

### 显式 shape：

为确保可以在整个计算图中了解 shape，将 `shape` 参数传递给 `from_tensorflow`。该字典将输入名称映射到输入 shape。有关示例，参阅这些 [测试用例](https://github.com/apache/tvm/blob/main/tests/python/frontend/tensorflow/test_forward.py#L36)。

### 数据布局

大多数 TensorFlow 模型都是使用 NHWC 布局发布的。NCHW 布局通常提供更好的性能，尤其是在 GPU 上。TensorFlow 前端可以通过将参数 `layout='NCHW'` 传递给 `from_tensorflow` 来自动转换模型的数据布局。

## 最佳实践

* 使用静态张量 shape，而非动态 shape（删除「None」尺寸）。
* 使用静态 RNN，而非动态 RNN，因为尚不支持 `TensorArray`。

## 支持的算子

* Abs
* Add
* AddN
* All
* Any
* ArgMax
* ArgMin
* AvgPool
* BatchMatMul
* BatchMatMulV2
* BatchNormWithGlobalNormalization
* BatchToSpaceND
* BiasAdd
* BroadcastTo
* Cast
* Ceil
* CheckNumerics
* ClipByValue
* Concat
* ConcatV2
* Conv2D
* Cos
* Tan
* CropAndResize
* DecodeJpeg
* DepthwiseConv2dNative
* DepthToSpace
* Dilation2D
* Equal
* Elu
* Enter
* Erf
* Exit
* Exp
* ExpandDims
* Fill
* Floor
* FloorDiv
* FloorMod
* FusedBatchNorm
* FusedBatchNormV2
* Gather
* GatherNd
* GatherV2
* Greater
* GreaterEqual
* Identity
* IsFinite
* IsInf
* IsNan
* LeakyRelu
* LeftShift
* Less
* LessEqual
* Log
* Log1p
* LoopCond
* LogicalAnd
* LogicalOr
* LogicalNot
* LogSoftmax
* LRN
* LSTMBlockCell
* MatMul
* Max
* MaxPool
* Maximum
* Mean
* Merge
* Min
* Minimum
* MirrorPad
* Mod
* Mul
* Neg
* NextIteration
* NotEqual
* OneHot
* Pack
* Pad
* PadV2
* Pow
* Prod
* Range
* Rank
* RealDiv
* Relu
* Relu6
* Reshape
* ResizeBilinear
* ResizeBicubic
* ResizeNearestNeighbor
* ReverseV2
* RightShift
* Round
* Rsqrt
* Select
* Selu
* Shape
* Sigmoid
* Sign
* Sin
* Size
* Slice
* Softmax
* Softplus
* SpaceToBatchND
* SpaceToDepth,
* Split
* SplitV
* Sqrt
* Square
* SquareDifference
* Squeeze
* StridedSlice
* Sub
* Sum
* Switch
* Tanh
* TensorArrayV3
* TensorArrayScatterV3
* TensorArrayGatherV3
* TensorArraySizeV3
* TensorArrayWriteV3
* TensorArrayReadV3
* TensorArraySplitV3
* TensorArrayConcatV3
* Tile
* TopKV2
* Transpose
* TruncateMod
* Unpack
* UnravelIndex
* Where
* ZerosLike
