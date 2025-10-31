---

title: Relax

---



Relax 是 Apache TVM 栈中用于图优化和转换的高级抽象层。此外，Apache TVM 将 Relax 和 TensorIR 结合在一起，作为跨层优化的统一策略。因此，Relax 通常与 TensorIR 紧密协作，用于表示和优化整个 IRModule。
* [机器学习模型的图抽象](https://tvm.hyper.ai/docs/deep-dive/relax/graph-abstraction-for-ml-models)
   * [什么是图抽象？](https://tvm.hyper.ai/docs/deep-dive/relax/graph-abstraction-for-ml-models#%E4%BB%80%E4%B9%88%E6%98%AF%E5%9B%BE%E6%8A%BD%E8%B1%A1)
   * [Relax 的关键特性](https://tvm.hyper.ai/docs/deep-dive/relax/graph-abstraction-for-ml-models#relax-%E7%9A%84%E5%85%B3%E9%94%AE%E7%89%B9%E6%80%A7)
* [理解 Relax 抽象层](https://tvm.hyper.ai/docs/deep-dive/relax/understand-relax-abstraction)
   * [端到端模型执行](https://tvm.hyper.ai/docs/deep-dive/relax/understand-relax-abstraction#%E7%AB%AF%E5%88%B0%E7%AB%AF%E6%A8%A1%E5%9E%8B%E6%89%A7%E8%A1%8C)
   * [Relax 的核心元素](https://tvm.hyper.ai/docs/deep-dive/relax/understand-relax-abstraction#relax-%E7%9A%84%E5%85%B3%E9%94%AE%E5%85%83%E7%B4%A0)
* [创建 Relax ](https://tvm.hyper.ai/docs/deep-dive/relax/relax-creation)
   * [使用 TVMScript 创建 Relax 程序](https://tvm.hyper.ai/docs/deep-dive/relax/relax-creation#%E4%BD%BF%E7%94%A8-tvmscript-%E5%88%9B%E5%BB%BA-relax-%E7%A8%8B%E5%BA%8F)
   * [使用 NNModule API 创建 Relax 程序](https://tvm.hyper.ai/docs/deep-dive/relax/relax-creation#%E4%BD%BF%E7%94%A8-nnmodule-api-%E6%9E%84%E5%BB%BA-relax-%E7%A8%8B%E5%BA%8F)
   * [使用 Block Builder API 创建 Relax 程序](https://tvm.hyper.ai/docs/deep-dive/relax/relax-creation#%E4%BD%BF%E7%94%A8-block-builder-api-%E5%88%9B%E5%BB%BA-relax-%E7%A8%8B%E5%BA%8F)
   * [总结](https://tvm.hyper.ai/docs/deep-dive/relax/relax-creation#%E6%80%BB%E7%BB%93)
* [转换与优化](https://tvm.hyper.ai/docs/deep-dive/relax/transformation)
   * [应用转换](https://tvm.hyper.ai/docs/deep-dive/relax/transformation#%E5%BA%94%E7%94%A8%E8%BD%AC%E6%8D%A2)
   * [自定义 Pass](https://tvm.hyper.ai/docs/deep-dive/relax/transformation#%E8%87%AA%E5%AE%9A%E4%B9%89-pass)
   * [总结](https://tvm.hyper.ai/docs/deep-dive/relax/transformation#%E6%80%BB%E7%BB%93)



