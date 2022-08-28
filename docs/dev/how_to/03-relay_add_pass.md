---
title: 向 Relay 中添加 Compiler Pass
---

# 向 Relay 中添加 Compiler Pass

Compiler Pass 是扩展 Relay 功能集及优化 Relay 程序的主要接口。通过编写 compiler pass，用户可以基于最终目标，修改 AST 或收集 AST 相关信息。事实上，Relay 内置的一些重要特性（如自动微分和类型推断）都“标准”的 compiler pass。

整体来看，编写 pass 包括两个关键组成部分：

* 创建一个或多个遍历程序的 C++ 类
* 将遍历实现及其在 pass manager API 中的元数据包装，从而方便与 [Pass Infrastructure](https://tvm.apache.org/docs/arch/pass_infra.html#pass-infra) 轻松交互

首先，我们将概述编写 compiler pass 的关键机制。然后通过 Relay 中常量折叠 pass 的具体示例进行演示。

# AST 遍历器 (Traversers)

用于遍历 Relay 程序的基类是 `ExprFunctor`。它提供的公共接口是一个 `VisitExpr` 方法，该方法接收一个表达式以及零个或多个参数，并返回某种类型的实例。扩展此类时，可以通过覆盖每种表达式类型的 `VisitExpr_` 实现，来定义 AST 遍历模式。

`VisitExpr` 和 `VisitExpr_` 之间的关系与调度有关。每个 `VisitExpr_` 定义都针对特定类型的表达式，但用户无法每次都得知要访问的节点类型。为了解决这个问题，`ExprFunctor` 提供了一个 `VisitExpr` 函数，将给定表达式路由转换为 `VisitExpr_` 实例进而解决问题。尽管 C++ 已经提供了动态调度，但 `ExprFunctor` 定义了自己的虚表供 `VisitExp` 使用。通过定义虚表可以更好地控制调度。例如，定义一个在每次访问之前都打印 "Here" 的 `PrintVisitor` 遍历器，可以覆盖 `VisitExpr`：

``` c++
void PrintVisitor::VisitExpr(const Expr& expr) {
  std::cout << "Here" << std::endl;
  ExprFunctor::VisitExpr(expr);
}
```

`ExprFunctor` 本身是一个非常通用的类，这就是为什么更多时候你会扩展 `ExprVisitor` 或 `ExprMutator`。这些类扩展了 `ExprFunctor`，并提供了 `VisitExpr_` 的默认实现，这些实现捕获了每种表达式类型的常见遍历模式。有了这些默认的实现，开发者只需针对想要不同行为的表达式类型，提供覆盖的实现。后续章节将针对每个子类进行详细描述。

## 表达式访问器 (Expression Visitors)

`ExprVisitor` 不用于修改程序的pass，而是用于实施程序分析和收集信息的 pass。使用这个类，`VisitExpr` 和私有 counterparts 不会返回任何内容。此类提供的 `VisitExpr_` 实现只是访问表达式的所有表达式字段。 `IfNode` 的默认实现如下所示：

``` c++
void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);
}
```

注意，这里调用的是 `VisitExpr` 而非 `VisitExpr_`，因此用户可以使用 `ExprFunctor` 中的虚表进行路由。

如果要编写一个 `CallChecker` 类来检查程序中是否出现函数调用，只需扩展 `ExprVisitor` 并定义以下 `VisitExpr_` 方法：

``` c++
void VisitExpr_(const CallNode* n) final {
  result_ = true;
}
```

其中 `result_` 是一个字段。在该示例中，无需在 `CallNode` 字段上进一步递归，因为 `result_` 已经为 true，原始表达式中包含一个调用。为了使该访问器可用，可以采用以下方法：

``` c++
bool Check(const Expr& expr) final {
  result_ = false;
  VisitExpr(expr);
  return result_;
}
```

以上就是全部操作。在调用 top-level 的递归之前，定义一个执行一些记录的公有接口是很常见的操作。用户也可以通过创建一个生成 `CallChecker` 实例，并在其上调用 `Check` 的独立程序来进一步包装 API，重要的是用尽可能少的资源用实现目标。

## 表达式变异器 (Expression Mutators)

`ExprMutator` 用于以某种方式转换程序的 pass。通过这个类，`VisitExpr` 及其对应的私有部分返回 `Expr`。此类提供的默认 `VisitExpr_` 实现访问表达式的所有表达式字段，并将字段设置为访问它们的结果。`TupleGetItemNode` 的默认实现如下所示：

``` c++
Expr ExprMutator::VisitExpr_(const TupleGetItemNode* g) {
  auto t = this->Mutate(g->tuple);
  if (g->tuple == t) {
    return GetRef<Expr>(g);
  } else {
    return TupleGetItem(t, g->index);
  }
}
```

这里有几点需要注意。首先，`Mutate` 是 `ExprMutator` 中 `VisitExpr` 的别名。其次，如果对 `Mutate` 的调用修改了 `tuple` 字段，则只返回一个新节点。这种更新的方法称为功能更新，这样做可以避免不必要的分配。

`ExprMutator` 有、而 `ExprVisitor` 没有的一个功能，是用于缓存结果的内置 `memo_` 字段。`ExprMutator` 有一个记忆器 (memoizer) 这是合理的，因为用户知道正在缓存哪些类型的结果（即 `Expr`），而 `ExprVisitor` 的访问方法不返回任何内容。通常，当用户要在 `ExprVisitor` 的子类中缓存结果时，需要自行定义缓存。

如果希望编写一个 `IfCollapser` 类，用它的真实分支替换每个 if 语句，用户将为 `IfNode` 覆盖 `VisitExpr_`：

``` c++
Expr ExprMutator::VisitExpr_(const IfNode* op) {
  return this->Mutate(op->true_branch);
}
```

注意：返回的表达式不一定是 `IfNode`，这是正常的，因为返回类型是 `Expr`。接下来创建一个公有接口：

``` c++
Expr CollapseIfs(const Expr& expr) final {
  return this->Mutate(expr);
}
```

虽然使用这个变异器无需做任何记录，但仍然鼓励用户将描述性方法作为接口。

# 示例：常量折叠

为了更好地理解编写 pass 的过程，本部分将以常量折叠 pass（可在 [src/relay/transforms/fold_constant.cc](https://github.com/apache/tvm/blob/main/src/relay/transforms/fold_constant.cc) 中找到）作为示例进行讲解。常量折叠 pass 相对简单，且包含两种类型的遍历。

常量折叠涉及只包含常量的程序评估表达式 (evaluating expression)，然后用评估它们的结果替换这些表达式。此过程的目的是预加载可以进行的所有计算。为了实现这一点，常量折叠 pass 使用了一个访问器（`ConstantChecker`）和一个修改器（`ConstantFolder`）。

## `ConstantChecker` 访问器

此访问器用于检查表达式是否为常量。在 Relay 中，用户将 `ConstantNode` 或者只有常量字段的 `TupleNode` 的表达式定义为常量。

使用 `memo_` 字段从节点映射到它们是否为常量，并缓存这些结果。下面是 `ConstantChecker` 中的 `VisitExpr_` 定义。

``` c++
void VisitExpr_(const ConstantNode* n) final {
  memo_[GetRef<Constant>(n)] = true;
}

void VisitExpr_(const TupleNode* n) final {
  bool result = true;
  for (const auto& field : n->fields) {
    if (!Check(field)) {
      result = false;
      break;
    }
  }
  memo_[GetRef<Tuple>(n)] = result;
}
```

用于协调这些定义的记录是一个 `Check` 方法，它返回给定的表达式是否被认定为常量。

``` c++
bool Check(const Expr& expr) {
  const auto it = memo_.find(expr);
  if (it != memo_.end())
    return it->second;
  VisitExpr(expr);
  return memo_[expr];
}
```

并不是所有遇到的节点都要修改 `memo_`；相反，用户只有在遇到的节点有可能是常数时，才修改 `memo_`。当 `memo_` 不包含 `expr` 时，需要依赖默认的 false 值。

## `ConstantFolder` 变异器

这个变异器执行了大部分的常量折叠过程，并在内部使用 `ConstantChecker`。在 Relay 中，常量折叠涉及三种节点类型：`LetNode`、`TupleItemGetNode` 和 `CallNode`。后续段落中将进行详细讲解。

``` c++
Expr VisitExpr_(const LetNode* op) final {
  Expr value = this->Mutate(op->value);
  if (value.as<ConstantNode>()) {
    memo_[op->var] = value;
    return this->Mutate(op->body);
  } else {
    Var var = Downcast<Var>(this->Mutate(op->var));
    Expr body = this->Mutate(op->body);
    if (var.same_as(op->var) &&
        value.same_as(op->value) &&
        body.same_as(op->body)) {
      return GetRef<Expr>(op);
    } else {
      return Let(var, value, body);
    }
  }
}
```

在 `LetNode` 示例里，首先尝试常量折叠绑定在表达式的值。如果可以，填充 `memo_` 并返回访问主体的结果——本质上是将绑定的值传到主体中的使用点。如果无法常量折叠绑定的值，可以参照默认的实现方法：

``` c++
Expr VisitExpr_(const TupleGetItemNode* op) final {
  Expr res = ExprMutator::VisitExpr_(op);
  op = res.as<TupleGetItemNode>();
  if (const auto* tuple = op->tuple.as<TupleNode>()) {
    return tuple->fields[op->index];
  } else {
    return res;
  }
}
```

在 `TupleItemGetNode` 的例子里，需要检查 `op->tuple` 字段是否为 `TupleNode`。如果是，我们将 get 元组替换为 `op->index` 指向的元组的字段。这样做的原因是因为 `op->tuple` 可能被错误评估为一个元组。

``` c++
Expr VisitExpr_(const CallNode* call) final {
  static auto op_stateful = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");
  Expr res = ExprMutator::VisitExpr_(call);
  call = res.as<CallNode>();
  // 我们不使用零参数的常量折叠函数。
  // 这是一个很有用的启发式方法。
  // 例如折叠那些 shape=(4, 5) 是有害的。
  if (call->args.size() == 0) return res;
  const OpNode* op = call->op.as<OpNode>();
  if (op == nullptr) return res;
  // 跳过有状态的算子。
  if (op_stateful.get(GetRef<Op>(op), false)) return res;
  bool all_const_args = true;
  for (Expr arg : call->args) {
    if (!checker_.Check(arg)) {
      all_const_args = false;
    }
  }
  if (all_const_args) {
    return ConstEvaluate(res);
  } else {
    return res;
  }
}
```

在 `CallNode` 示例中，首先使用 `ExprMutator` 的 `VisitExpr_` 来访问调用，它将调用的所有字段都常量折叠了。之所以使用 `ExprMutator::VisitExpr_` 而不是 `VisitExpr`，是因为我们想要绕过虚表（以避免死循环）并使用 `ExprMutator` 提供的默认实现。只有当所有参数都是常量时，才评估调用（使用 `ConstantChecker`）。评估调用会产生一个**值**，因此这里使用辅助方法 `ValueToExpr` ，将评估的表达式放回 AST 中。

现在，我们为常量文件夹构造了一个更方便的接口 `FoldConstant`。`FoldConstant` 是 `ConstantFolder` 类之外的一个独立函数，它负责接收表达式并在内部创建和使用 `ConstantFolder` 实例（其完整的定义在 [src/relay/transforms/fold_constant.cc](https://github.com/apache/tvm/blob/main/src/relay/transforms/fold_constant.cc) 中）。

## 用 Pass Manager 注册 Pass

*注意：更多详情请参阅 :ref:*`pass-infra` 中的文档。

编写 AST 遍历器后，用以下代码可将 pass 注册为 TVM API 端点：

``` c++
namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(FoldConstant(f));
  };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}

}  // 命名空间转换
```

将上述代码生成的 `Pass` 对象提供给 pass 基础架构，可以使得 AST 遍历应用于给定 Relay 模块中的所有函数，这是常量折叠过程预期的行为（它应该尽可能折叠所有常量）。

函数 `CreateFunctionPass` 允许注册 pass 的优化级别（在本例中为 2），可用于根据 pass 的一般实用性、 pass 名称和 pass 中的任何依赖项将 pass 组合在一起。pass 的依赖项以列表形式给出，罗列了当前 pass 运行所必需的所有 pass 的结果。`FoldConstant` 没有任何依赖，但是很多 Relay pass 确实依赖有类型信息，所以 `InferType` 是一个常见的依赖；其他的可能依赖于程序为 A-范式，通过 `ToANormalForm` pass。

注意，`PassContext` 对象包含 pass 用于错误报告和配置选项的信息； `FoldConstant` 不需要此信息，但其他 pass 可能会引用它们的 `PassContext` 对象。

现在可以通过 pass 基础结构调用 pass 了，推荐为 pass 添加 Python 绑定，如以下代码片段所示：

``` c++
TVM_REGISTER_GLOBAL("relay._transform.FoldConstant")
.set_body_typed(FoldConstant);
```

通过以上方法定义了 `Pass` 对象后，就可以用 pass 基础架构的 `Sequential` 结构来调用了。 `Sequential` 接收一个 pass 列表，并将其按顺序应用于 Relay 模块，从而获得转换后的模块。例如，下面的代码将 `FoldConstant` 和 `ToANormalForm` pass 逐一应用于 `mod` 中的每个函数，并获得一个新模块。

``` c++
seq = transform.Sequential([
    relay.transform.FoldConstant(),
    relay.transform.ToANormalForm()
])
new_mod = seq(mod)
```

更多注册相关的内容，请查看 [TVM Runtime 系统](https://tvm.apache.org/docs/arch/runtime.html#tvm-runtime-system)；pass 管理器接口相关的更多信息，请查看 [Pass 基础架构](https://tvm.apache.org/docs/arch/pass_infra.html#pass-infra)； Relay 的标准 pass 列表及实现方式，请分别查看 [include/tvm/relay/transform.h](https://github.com/apache/tvm/blob/main/include/tvm/relay/transform.h) 及 [src/relay/transforms/](https://github.com/apache/tvm/tree/main/src/relay/transforms)。