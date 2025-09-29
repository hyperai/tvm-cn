---

title: TensorIR

---


TensorIR 是 Apache TVM 栈中的核心抽象之一，用于表示和优化原始的张量函数。
* [张量程序抽象](https://tvm.apache.org/docs/deep_dive/tensor_ir/abstraction.html)
   * [张量程序的关键元素](https://tvm.apache.org/docs/deep_dive/tensor_ir/abstraction.html#key-elements-of-tensor-programs)
   * [TensorIR 中的额外结构](https://tvm.apache.org/docs/deep_dive/tensor_ir/abstraction.html#extra-structure-in-tensorir)
* [理解 TensorIR 抽象](https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html)
   * [函数参数与缓冲区](https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html#function-parameters-and-buffers)
   * [循环迭代](https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html#loop-iterations)
   * [计算块](https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html#computational-block)
   * [块轴属性](https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html#block-axis-properties)
   * [为什么计算块中需要额外信息](https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html#why-extra-information-in-block)
   * [绑定块轴的简洁写法](https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html#sugars-for-block-axes-binding)
* [TensorIR 的创建](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html)
   * [使用 TVMScript 创建 TensorIR](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html#create-tensorir-using-tvmscript)
   * [使用张量表达式创建 TensorIR](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html#create-tensorir-using-tensor-expression)
* [变换](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html)
   * [初始化调度](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html#initialization-schedule)
   * [循环切分（Tiling）](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html#loop-tiling)
   * [利用数据局部性](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html#leverage-localities)
   * [重写归约操作](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html#rewrite-reduction)
   * [追踪变换过程](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html#trace-the-transformation)


