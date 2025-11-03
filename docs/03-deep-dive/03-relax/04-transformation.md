---

title: 转换

---


:::note

本教程可通过 Google Colab 交互式运行！也可点击[此处](/docs/deep-dive/relax/transformation#%E6%80%BB%E7%BB%93)在本地运行 Jupyter Notebook。

[在 Google Colab 中打开](https://colab.research.google.com/github/apache/tvm-site/blob/asf-site/docs/_downloads/315fcda965a8d605f81705edf19ea2c6/relax_creation.ipynb)

:::

本节我们将深入介绍 Relax 程序的转换。转换是编译流程中实现优化和对接硬件后端的关键环节之一。


我们首先像[上一节](/docs/deep-dive/relax/relax-creation)那样，创建一个简单的 Relax 程序。

```plain
import tvm
from tvm import IRModule, relax
from tvm.relax.frontend import nn


class NNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


origin_mod, params = NNModule().export_tvm(
    {"forward": {"x": nn.spec.Tensor(("n", 784), "float32")}}
)
origin_mod.show()
```


输出:

```plain
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def forward(x: R.Tensor(("n", 784), dtype="float32"), fc1_weight: R.Tensor((128, 784), dtype="float32"), fc1_bias: R.Tensor((128,), dtype="float32"), fc2_weight: R.Tensor((10, 128), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor(("n", 10), dtype="float32"):
        n = T.int64()
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims: R.Tensor((784, 128), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
            matmul: R.Tensor((n, 128), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((n, 128), dtype="float32") = R.add(matmul, fc1_bias)
            relu: R.Tensor((n, 128), dtype="float32") = R.nn.relu(add)
            permute_dims1: R.Tensor((128, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
            matmul1: R.Tensor((n, 10), dtype="float32") = R.matmul(relu, permute_dims1, out_dtype="void")
            add1: R.Tensor((n, 10), dtype="float32") = R.add(matmul1, fc2_bias)
            gv: R.Tensor((n, 10), dtype="float32") = add1
            R.output(gv)
        return gv
```


# 应用转换

在 Relax 中，Pass 是应用程序转换的主要方式。我们可以通过应用 Pass 来改变程序的形式。首先，我们应用一个内置 `LegalizeOps`，它会将高层操作符（High-level Operators）转换为低层操作符。

```plain
mod = tvm.relax.transform.LegalizeOps()(origin_mod)
mod.show()
```


输出:

```plain
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(var_matmul: T.handle, fc1_bias: T.Buffer((T.int64(128),), "float32"), var_T_add: T.handle):
        T.func_attr({"tir.noalias": True})
        n = T.int64()
        matmul = T.match_buffer(var_matmul, (n, T.int64(128)))
        T_add = T.match_buffer(var_T_add, (n, T.int64(128)))
        # with T.block("root"):
        for ax0, ax1 in T.grid(n, T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(matmul[v_ax0, v_ax1], fc1_bias[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = matmul[v_ax0, v_ax1] + fc1_bias[v_ax1]

    @T.prim_func(private=True)
    def add1(var_matmul1: T.handle, fc2_bias: T.Buffer((T.int64(10),), "float32"), var_T_add: T.handle):
        T.func_attr({"tir.noalias": True})
        n = T.int64()
        matmul1 = T.match_buffer(var_matmul1, (n, T.int64(10)))
        T_add = T.match_buffer(var_T_add, (n, T.int64(10)))
        # with T.block("root"):
        for ax0, ax1 in T.grid(n, T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(matmul1[v_ax0, v_ax1], fc2_bias[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = matmul1[v_ax0, v_ax1] + fc2_bias[v_ax1]

    @T.prim_func(private=True)
    def matmul(var_x: T.handle, permute_dims: T.Buffer((T.int64(784), T.int64(128)), "float32"), var_matmul: T.handle):
        T.func_attr({"tir.noalias": True})
        n = T.int64()
        x = T.match_buffer(var_x, (n, T.int64(784)))
        matmul = T.match_buffer(var_matmul, (n, T.int64(128)))
        # with T.block("root"):
        for i0, i1, k in T.grid(n, T.int64(128), T.int64(784)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], permute_dims[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + x[v_i0, v_k] * permute_dims[v_k, v_i1]

    @T.prim_func(private=True)
    def matmul1(var_relu: T.handle, permute_dims1: T.Buffer((T.int64(128), T.int64(10)), "float32"), var_matmul: T.handle):
        T.func_attr({"tir.noalias": True})
        n = T.int64()
        relu = T.match_buffer(var_relu, (n, T.int64(128)))
        matmul = T.match_buffer(var_matmul, (n, T.int64(10)))
        # with T.block("root"):
        for i0, i1, k in T.grid(n, T.int64(10), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(relu[v_i0, v_k], permute_dims1[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + relu[v_i0, v_k] * permute_dims1[v_k, v_i1]

    @T.prim_func(private=True)
    def relu(var_add: T.handle, var_compute: T.handle):
        T.func_attr({"tir.noalias": True})
        n = T.int64()
        add = T.match_buffer(var_add, (n, T.int64(128)))
        compute = T.match_buffer(var_compute, (n, T.int64(128)))
        # with T.block("root"):
        for i0, i1 in T.grid(n, T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(add[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(add[v_i0, v_i1], T.float32(0.0))

    @T.prim_func(private=True)
    def transpose(fc1_weight: T.Buffer((T.int64(128), T.int64(784)), "float32"), T_transpose: T.Buffer((T.int64(784), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(784), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(fc1_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = fc1_weight[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose1(fc2_weight: T.Buffer((T.int64(10), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(128), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(128), T.int64(10)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(fc2_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = fc2_weight[v_ax1, v_ax0]

    @R.function
    def forward(x: R.Tensor(("n", 784), dtype="float32"), fc1_weight: R.Tensor((128, 784), dtype="float32"), fc1_bias: R.Tensor((128,), dtype="float32"), fc2_weight: R.Tensor((10, 128), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor(("n", 10), dtype="float32"):
        n = T.int64()
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            permute_dims = R.call_tir(cls.transpose, (fc1_weight,), out_sinfo=R.Tensor((784, 128), dtype="float32"))
            matmul = R.call_tir(cls.matmul, (x, permute_dims), out_sinfo=R.Tensor((n, 128), dtype="float32"))
            add = R.call_tir(cls.add, (matmul, fc1_bias), out_sinfo=R.Tensor((n, 128), dtype="float32"))
            relu = R.call_tir(cls.relu, (add,), out_sinfo=R.Tensor((n, 128), dtype="float32"))
            permute_dims1 = R.call_tir(cls.transpose1, (fc2_weight,), out_sinfo=R.Tensor((128, 10), dtype="float32"))
            matmul1 = R.call_tir(cls.matmul1, (relu, permute_dims1), out_sinfo=R.Tensor((n, 10), dtype="float32"))
            add1 = R.call_tir(cls.add1, (matmul1, fc2_bias), out_sinfo=R.Tensor((n, 10), dtype="float32"))
            gv: R.Tensor((n, 10), dtype="float32") = add1
            R.output(gv)
        return gv
```


从输出可以看到，程序中的高层操作符（即 `relax.op`）已被其对应的低层操作符（即 `relax.call_tir`）所替代。


接下来我们尝试应用算子融合（Operator Fusion），这是一种在 ML 编译器中被广泛使用的优化技术。需要注意的是，在 Relax 中，融合优化是由一系列 Pass 协作完成的，我们可以按顺序依次应用它们。

```plain
mod = tvm.ir.transform.Sequential(
    [
        tvm.relax.transform.AnnotateTIROpPattern(),
        tvm.relax.transform.FuseOps(),
        tvm.relax.transform.FuseTIR(),
    ]
)(mod)
mod.show()
```


输出:

```plain
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_matmul1_add1(p_relu: T.handle, permute_dims1: T.Buffer((T.int64(128), T.int64(10)), "float32"), fc2_bias: T.Buffer((T.int64(10),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": True})
        n = T.int64()
        relu = T.match_buffer(p_relu, (n, T.int64(128)))
        T_add_intermediate = T.match_buffer(p_output0, (n, T.int64(10)))
        # with T.block("root"):
        matmul_intermediate = T.alloc_buffer((n, T.int64(10)))
        for i0, i1, k in T.grid(n, T.int64(10), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(relu[v_i0, v_k], permute_dims1[v_k, v_i1])
                T.writes(matmul_intermediate[v_i0, v_i1])
                with T.init():
                    matmul_intermediate[v_i0, v_i1] = T.float32(0.0)
                matmul_intermediate[v_i0, v_i1] = matmul_intermediate[v_i0, v_i1] + relu[v_i0, v_k] * permute_dims1[v_k, v_i1]
        for ax0, ax1 in T.grid(n, T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(matmul_intermediate[v_ax0, v_ax1], fc2_bias[v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = matmul_intermediate[v_ax0, v_ax1] + fc2_bias[v_ax1]

    @T.prim_func(private=True)
    def fused_matmul_add_relu(p_x: T.handle, permute_dims: T.Buffer((T.int64(784), T.int64(128)), "float32"), fc1_bias: T.Buffer((T.int64(128),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": True})
        n = T.int64()
        x = T.match_buffer(p_x, (n, T.int64(784)))
        compute_intermediate = T.match_buffer(p_output0, (n, T.int64(128)))
        # with T.block("root"):
        matmul_intermediate = T.alloc_buffer((n, T.int64(128)))
        T_add_intermediate = T.alloc_buffer((n, T.int64(128)))
        for i0, i1, k in T.grid(n, T.int64(128), T.int64(784)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], permute_dims[v_k, v_i1])
                T.writes(matmul_intermediate[v_i0, v_i1])
                with T.init():
                    matmul_intermediate[v_i0, v_i1] = T.float32(0.0)
                matmul_intermediate[v_i0, v_i1] = matmul_intermediate[v_i0, v_i1] + x[v_i0, v_k] * permute_dims[v_k, v_i1]
        for ax0, ax1 in T.grid(n, T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(matmul_intermediate[v_ax0, v_ax1], fc1_bias[v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = matmul_intermediate[v_ax0, v_ax1] + fc1_bias[v_ax1]
        for i0, i1 in T.grid(n, T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_add_intermediate[v_i0, v_i1])
                T.writes(compute_intermediate[v_i0, v_i1])
                compute_intermediate[v_i0, v_i1] = T.max(T_add_intermediate[v_i0, v_i1], T.float32(0.0))

    @T.prim_func(private=True)
    def transpose(fc1_weight: T.Buffer((T.int64(128), T.int64(784)), "float32"), T_transpose: T.Buffer((T.int64(784), T.int64(128)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(784), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(fc1_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = fc1_weight[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose1(fc2_weight: T.Buffer((T.int64(10), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(128), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(128), T.int64(10)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(fc2_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = fc2_weight[v_ax1, v_ax0]

    @R.function
    def forward(x: R.Tensor(("n", 784), dtype="float32"), fc1_weight: R.Tensor((128, 784), dtype="float32"), fc1_bias: R.Tensor((128,), dtype="float32"), fc2_weight: R.Tensor((10, 128), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor(("n", 10), dtype="float32"):
        n = T.int64()
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            permute_dims = R.call_tir(cls.transpose, (fc1_weight,), out_sinfo=R.Tensor((784, 128), dtype="float32"))
            lv = R.call_tir(cls.fused_matmul_add_relu, (x, permute_dims, fc1_bias), out_sinfo=R.Tensor((n, 128), dtype="float32"))
            permute_dims1 = R.call_tir(cls.transpose1, (fc2_weight,), out_sinfo=R.Tensor((128, 10), dtype="float32"))
            gv = R.call_tir(cls.fused_matmul1_add1, (lv, permute_dims1, fc2_bias), out_sinfo=R.Tensor((n, 10), dtype="float32"))
            R.output(gv)
        return gv
```



最终可以看到，`matmul`、`add` 和 `relu` 等操作符被融合成一个内核（也就是一个 `call_tir`）。

关于所有内置 Pass 的详细信息，请参考 `relax.transform`。


## 自定义 Pass

我们也可以定义自己的 Pass。例如，我们可以将程序中的 `relu` 操作符重写为 `gelu` 操作符。


首先，我们需要编写一个 Relax IR Mutator 来执行这个替换操作。

```plain
from tvm.relax.expr_functor import PyExprMutator, mutator


@mutator
class ReluRewriter(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        # 访问 relax.Call 表达式，并且只处理当操作（op）是 relax.nn.relu 的情况。
        if call.op.name == "relax.nn.relu":
            return relax.op.nn.gelu(call.args[0])

        return super().visit_call_(call)
```


然后我们可以编写一个 Pass，将这个 Mutator 应用于整个模块。

```plain
@tvm.transform.module_pass(opt_level=0, name="ReluToGelu")
class ReluToGelu:  # pylint: disable=too-few-public-methods
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        rewriter = ReluRewriter(mod)
        for g_var, func in mod.functions_items():
            if isinstance(func, relax.Function):
                func = rewriter.visit_expr(func)
                rewriter.builder_.update_func(g_var, func)
        return rewriter.builder_.get()


mod = ReluToGelu()(origin_mod)
mod.show()

```


输出:

```plain
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def forward(x: R.Tensor(("n", 784), dtype="float32"), fc1_weight: R.Tensor((128, 784), dtype="float32"), fc1_bias: R.Tensor((128,), dtype="float32"), fc2_weight: R.Tensor((10, 128), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor(("n", 10), dtype="float32"):
        n = T.int64()
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims: R.Tensor((784, 128), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
            matmul: R.Tensor((n, 128), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((n, 128), dtype="float32") = R.add(matmul, fc1_bias)
            relu: R.Tensor((n, 128), dtype="float32") = R.nn.gelu(add)
            permute_dims1: R.Tensor((128, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
            matmul1: R.Tensor((n, 10), dtype="float32") = R.matmul(relu, permute_dims1, out_dtype="void")
            add1: R.Tensor((n, 10), dtype="float32") = R.add(matmul1, fc2_bias)
            gv: R.Tensor((n, 10), dtype="float32") = add1
            R.output(gv)
        return gv
```


从打印的输出中可以看到，`relax.nn.relu` 操作符被成功替换为 `relax.nn.gelu` 操作符。

关于 Mutator 的更多细节，请参考 `relax.expr_functor.PyExprMutator`。



## 总结

在本节中，我们展示了如何对 Relax 程序进行转换，并介绍了如何定义和应用自定义转换（可右键另存为下载）。
* [下载 Jupyter notebook：relax_transformation.ipynb](https://tvm.apache.org/docs/_downloads/31d077cfa7c55c0edcd16ad5d4faf483/relax_transformation.ipynb)
* [下载 Python 源码：relax_transformation.py](https://tvm.apache.org/docs/_downloads/4e684410fb30ce02332f55fde123c42e/relax_transformation.py)
* [下载压缩包：relax_transformation.zip](https://tvm.apache.org/docs/_downloads/7d201684dfa095a5ea48d98e9a2ef7ad/relax_transformation.zip)


