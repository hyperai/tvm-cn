---

title: TensorIR

---


TensorIR 是 Apache TVM 栈中的核心抽象之一，用于表示和优化原始的张量函数。
* [张量程序抽象](https://tvm.hyper.ai/docs/deep-dive/tensorir/tensor-program-abstraction)
   * [张量程序的关键元素](https://tvm.hyper.ai/docs/deep-dive/tensorir/tensor-program-abstraction#%E5%BC%A0%E9%87%8F%E7%A8%8B%E5%BA%8F%E7%9A%84%E5%85%B3%E9%94%AE%E5%85%83%E7%B4%A0)
   * [TensorIR 中的额外结构](https://tvm.hyper.ai/docs/deep-dive/tensorir/tensor-program-abstraction#tensorir-%E4%B8%AD%E7%9A%84%E9%A2%9D%E5%A4%96%E7%BB%93%E6%9E%84)
* [理解 TensorIR 抽象](https://tvm.hyper.ai/docs/deep-dive/tensorir/understand-tensorir-abstraction)
   * [函数参数与缓冲区](https://tvm.hyper.ai/docs/deep-dive/tensorir/understand-tensorir-abstraction#%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0%E4%B8%8E%E7%BC%93%E5%86%B2%E5%8C%BA)
   * [循环迭代](https://tvm.hyper.ai/docs/deep-dive/tensorir/understand-tensorir-abstraction#%E5%BE%AA%E7%8E%AF%E8%BF%AD%E4%BB%A3)
   * [计算块](https://tvm.hyper.ai/docs/deep-dive/tensorir/understand-tensorir-abstraction#%E8%AE%A1%E7%AE%97%E5%9D%97)
   * [块轴属性](https://tvm.hyper.ai/docs/deep-dive/tensorir/understand-tensorir-abstraction#%E5%9D%97%E8%BD%B4%E5%B1%9E%E6%80%A7)
   * [为什么计算块中需要额外信息](https://tvm.hyper.ai/docs/deep-dive/tensorir/understand-tensorir-abstraction#%E4%B8%BA%E4%BB%80%E4%B9%88%E5%9C%A8%E5%9D%97%E4%B8%AD%E6%9C%89%E9%A2%9D%E5%A4%96%E7%9A%84%E4%BF%A1%E6%81%AF)
   * [绑定块轴的语法](https://tvm.hyper.ai/docs/deep-dive/tensorir/understand-tensorir-abstraction#%E5%9D%97%E8%BD%B4%E7%BB%91%E5%AE%9A%E7%9A%84%E8%AF%AD%E6%B3%95)
* [TensorIR 的创建](https://tvm.hyper.ai/docs/deep-dive/tensorir/tir_creation)
   * [使用 TVMScript 创建 TensorIR](https://tvm.hyper.ai/docs/deep-dive/tensorir/tir_creation#%E4%BD%BF%E7%94%A8-tvmscript-%E5%88%9B%E5%BB%BA-tensorir)
   * [使用张量表达式创建 TensorIR](https://tvm.hyper.ai/docs/deep-dive/tensorir/tir_creation#%E4%BD%BF%E7%94%A8-tensor-expression-%E5%88%9B%E5%BB%BA-tensorir)
* [转换](https://tvm.hyper.ai/docs/deep-dive/tensorir/tir_transformation)
   * [初始化调度](https://tvm.hyper.ai/docs/deep-dive/tensorir/tir_transformation#%E5%88%9D%E5%A7%8B%E5%8C%96%E8%AE%A1%E5%88%92)
   * [循环切分（Tiling）](https://tvm.hyper.ai/docs/deep-dive/tensorir/tir_transformation#%E5%BE%AA%E7%8E%AF%E5%88%86%E5%9D%97loop-tiling)
   * [利用数据局部性](https://tvm.hyper.ai/docs/deep-dive/tensorir/tir_transformation#%E5%88%A9%E7%94%A8%E5%B1%80%E9%83%A8%E6%80%A7)
   * [重写归约操作](https://tvm.hyper.ai/docs/deep-dive/tensorir/tir_transformation#%E9%87%8D%E5%86%99%E5%BD%92%E7%BA%A6%E6%93%8D%E4%BD%9C)
   * [追踪变换过程](https://tvm.hyper.ai/docs/deep-dive/tensorir/tir_transformation#%E8%BF%BD%E8%B8%AA%E8%BD%AC%E6%8D%A2)


