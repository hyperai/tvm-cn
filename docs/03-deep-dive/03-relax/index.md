---

title: Relax

---



Relax 是 Apache TVM 栈中用于图优化和转换的高级抽象层。此外，Apache TVM 将 Relax 和 TensorIR 结合在一起，作为跨层优化的统一策略。因此，Relax 通常与 TensorIR 紧密协作，用于表示和优化整个 IRModule。
* 机器学习模型的图抽象
   * 什么是图抽象？
   * Relax 的关键特性
* 理解 Relax 抽象层
   * 端到端模型执行
   * Relax 的核心元素
* 创建 Relax 
   * 使用 TVMScript 创建 Relax 程序
   * 使用 NNModule API 创建 Relax 程序
   * 使用 Block Builder API 创建 Relax 程序
   * 总结
* 转换与优化
   * 应用转换
   * 自定义 Pass
   * 总结



