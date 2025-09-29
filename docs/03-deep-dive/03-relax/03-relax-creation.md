---

title: 创建 Relax

---


:::note

本教程可通过 Google Colab 交互式运行！也可点击[此处](https://tvm.apache.org/docs/deep_dive/relax/tutorials/relax_creation.html#sphx-glr-download-deep-dive-relax-tutorials-relax-creation-py)在本地运行 Jupyter Notebook。

[在 Google Colab 中打开](https://colab.research.google.com/github/apache/tvm-site/blob/asf-site/docs/_downloads/315fcda965a8d605f81705edf19ea2c6/relax_creation.ipynb)

:::


本教程演示了如何创建 Relax 函数和程序。我们将介绍使用 TVMScript 和 Relax NNModule API 定义 Relax 函数的各种方式。



## 使用 TVMScript 创建 Relax 程序

TVMScript 是一个用于表示 Apache TVM 中间表示（IR）的领域特定语言（DSL）。它是 Python 的一种变体，可用于定义一个包含 TensorIR 和 Relax 函数的 IRModule。


在本节中，我们将展示如何使用 TVMScript 定义一个仅使用高级 Relax 运算符的简单 MLP 模型。

```plain
from tvm import relax, topi
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


@I.ir_module
class RelaxModule:
    @R.function
    def forward(
        data: R.Tensor(("n", 784), dtype="float32"),
        w0: R.Tensor((128, 784), dtype="float32"),
        b0: R.Tensor((128,), dtype="float32"),
        w1: R.Tensor((10, 128), dtype="float32"),
        b1: R.Tensor((10,), dtype="float32"),
    ) -> R.Tensor(("n", 10), dtype="float32"):
        with R.dataflow():
            lv0 = R.matmul(data, R.permute_dims(w0)) + b0
            lv1 = R.nn.relu(lv0)
            lv2 = R.matmul(lv1, R.permute_dims(w1)) + b1
            R.output(lv2)
        return lv2


RelaxModule.show()
```
输出:
```plain
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def forward(data: R.Tensor(("n", 784), dtype="float32"), w0: R.Tensor((128, 784), dtype="float32"), b0: R.Tensor((128,), dtype="float32"), w1: R.Tensor((10, 128), dtype="float32"), b1: R.Tensor((10,), dtype="float32")) -> R.Tensor(("n", 10), dtype="float32"):
        n = T.int64()
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(w0, axes=None)
            lv1: R.Tensor((n, 128), dtype="float32") = R.matmul(data, lv, out_dtype="void")
            lv0: R.Tensor((n, 128), dtype="float32") = R.add(lv1, b0)
            lv1_1: R.Tensor((n, 128), dtype="float32") = R.nn.relu(lv0)
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(w1, axes=None)
            lv5: R.Tensor((n, 10), dtype="float32") = R.matmul(lv1_1, lv4, out_dtype="void")
            lv2: R.Tensor((n, 10), dtype="float32") = R.add(lv5, b1)
            R.output(lv2)
        return lv2
```


Relax 不仅是图层级的中间表示（IR），还支持跨层级的表示与转换。具体来说，我们可以在 Relax 函数中直接调用 TensorIR 函数。


```plain
@I.ir_module
class RelaxModuleWithTIR:
    @T.prim_func
    def relu(x: T.handle, y: T.handle):
        n, m = T.int64(), T.int64()
        X = T.match_buffer(x, (n, m), "float32")
        Y = T.match_buffer(y, (n, m), "float32")
        for i, j in T.grid(n, m):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(X[vi, vj], T.float32(0))

    @R.function
    def forward(
        data: R.Tensor(("n", 784), dtype="float32"),
        w0: R.Tensor((128, 784), dtype="float32"),
        b0: R.Tensor((128,), dtype="float32"),
        w1: R.Tensor((10, 128), dtype="float32"),
        b1: R.Tensor((10,), dtype="float32"),
    ) -> R.Tensor(("n", 10), dtype="float32"):
        n = T.int64()
        cls = RelaxModuleWithTIR
        with R.dataflow():
            lv0 = R.matmul(data, R.permute_dims(w0)) + b0
            lv1 = R.call_tir(cls.relu, lv0, R.Tensor((n, 128), dtype="float32"))
            lv2 = R.matmul(lv1, R.permute_dims(w1)) + b1
            R.output(lv2)
        return lv2


RelaxModuleWithTIR.show()
```
输出:
```plain
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def relu(x: T.handle, y: T.handle):
        n, m = T.int64(), T.int64()
        X = T.match_buffer(x, (n, m))
        Y = T.match_buffer(y, (n, m))
        # with T.block("root"):
        for i, j in T.grid(n, m):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(X[vi, vj])
                T.writes(Y[vi, vj])
                Y[vi, vj] = T.max(X[vi, vj], T.float32(0.0))

    @R.function
    def forward(data: R.Tensor(("n", 784), dtype="float32"), w0: R.Tensor((128, 784), dtype="float32"), b0: R.Tensor((128,), dtype="float32"), w1: R.Tensor((10, 128), dtype="float32"), b1: R.Tensor((10,), dtype="float32")) -> R.Tensor(("n", 10), dtype="float32"):
        n = T.int64()
        cls = Module
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(w0, axes=None)
            lv1: R.Tensor((n, 128), dtype="float32") = R.matmul(data, lv, out_dtype="void")
            lv0: R.Tensor((n, 128), dtype="float32") = R.add(lv1, b0)
            lv1_1 = R.call_tir(cls.relu, (lv0,), out_sinfo=R.Tensor((n, 128), dtype="float32"))
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(w1, axes=None)
            lv5: R.Tensor((n, 10), dtype="float32") = R.matmul(lv1_1, lv4, out_dtype="void")
            lv2: R.Tensor((n, 10), dtype="float32") = R.add(lv5, b1)
            R.output(lv2)
        return lv2
```


:::note

你可能会注意到：打印输出的内容与我们编写的 TVMScript 代码不同。这是因为我们在打印 IRModule 时使用的是标准格式，而 TVMScript 支持语法糖简化输入。


例如，我们可以将多个操作合并为一行写成：

```plain
lv0 = R.matmul(data, R.permute_dims(w0)) + b0
```


但规范化后的表达式要求一个绑定中只能包含一个操作，因此打印输出是以下形式：

```plain
lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(w0, axes=None)
lv1: R.Tensor((n, 128), dtype="float32") = R.matmul(data, lv, out_dtype="void")
lv0: R.Tensor((n, 128), dtype="float32") = R.add(lv1, b0)
```
:::


## 使用 NNModule API 构建 Relax 程序

除了 TVMScript，我们还提供了类似 PyTorch 的 API，用于定义神经网络模型。这个接口更直观，也更容易使用。


在本节中，我们将展示如何使用 Relax 的 NNModule API 定义相同的 MLP 模型。

```plain
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
```


在定义完 NNModule 后，我们可以通过 `export_tvm` 将其导出为 TVM 的 IRModule。


```plain
mod, params = NNModule().export_tvm({"forward": {"x": nn.spec.Tensor(("n", 784), "float32")}})
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
            relu: R.Tensor((n, 128), dtype="float32") = R.nn.relu(add)
            permute_dims1: R.Tensor((128, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
            matmul1: R.Tensor((n, 10), dtype="float32") = R.matmul(relu, permute_dims1, out_dtype="void")
            add1: R.Tensor((n, 10), dtype="float32") = R.add(matmul1, fc2_bias)
            gv: R.Tensor((n, 10), dtype="float32") = add1
            R.output(gv)
        return gv
```


我们还可以在 NNModule 中插入自定义的函数调用，例如张量表达式（Tensor Expression，简称 TE）、TensorIR 函数，或其他 TVM 的打包函数（Packed Functions）。


```plain
@T.prim_func
def tir_linear(x: T.handle, w: T.handle, b: T.handle, z: T.handle):
    M, N, K = T.int64(), T.int64(), T.int64()
    X = T.match_buffer(x, (M, K), "float32")
    W = T.match_buffer(w, (N, K), "float32")
    B = T.match_buffer(b, (N,), "float32")
    Z = T.match_buffer(z, (M, N), "float32")
    for i, j, k in T.grid(M, N, K):
        with T.block("linear"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                Z[vi, vj] = 0
            Z[vi, vj] = Z[vi, vj] + X[vi, vk] * W[vj, vk]
    for i, j in T.grid(M, N):
        with T.block("add"):
            vi, vj = T.axis.remap("SS", [i, j])
            Z[vi, vj] = Z[vi, vj] + B[vj]


class NNModuleWithTIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        n = x.shape[0]
        # We can call external functions using nn.extern
        x = nn.extern(
            "env.linear",
            [x, self.fc1.weight, self.fc1.bias],
            out=nn.Tensor.placeholder((n, 128), "float32"),
        )
        # We can also call TensorIR via Tensor Expression API in TOPI
        x = nn.tensor_expr_op(topi.nn.relu, "relu", [x])
        # We can also call other TVM packed functions
        x = nn.tensor_ir_op(
            tir_linear,
            "tir_linear",
            [x, self.fc2.weight, self.fc2.bias],
            out=nn.Tensor.placeholder((n, 10), "float32"),
        )
        return x


mod, params = NNModuleWithTIR().export_tvm(
    {"forward": {"x": nn.spec.Tensor(("n", 784), "float32")}}
)
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
    def relu(var_env_linear: T.handle, var_compute: T.handle):
        T.func_attr({"tir.noalias": True})
        n = T.int64()
        env_linear = T.match_buffer(var_env_linear, (n, T.int64(128)))
        compute = T.match_buffer(var_compute, (n, T.int64(128)))
        # with T.block("root"):
        for i0, i1 in T.grid(n, T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(env_linear[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(env_linear[v_i0, v_i1], T.float32(0.0))

    @T.prim_func
    def tir_linear(x: T.handle, w: T.handle, b: T.handle, z: T.handle):
        M, K = T.int64(), T.int64()
        X = T.match_buffer(x, (M, K))
        N = T.int64()
        W = T.match_buffer(w, (N, K))
        B = T.match_buffer(b, (N,))
        Z = T.match_buffer(z, (M, N))
        # with T.block("root"):
        for i, j, k in T.grid(M, N, K):
            with T.block("linear"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(X[vi, vk], W[vj, vk])
                T.writes(Z[vi, vj])
                with T.init():
                    Z[vi, vj] = T.float32(0.0)
                Z[vi, vj] = Z[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(M, N):
            with T.block("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(Z[vi, vj], B[vj])
                T.writes(Z[vi, vj])
                Z[vi, vj] = Z[vi, vj] + B[vj]

    @R.function
    def forward(x: R.Tensor(("n", 784), dtype="float32"), fc1_weight: R.Tensor((128, 784), dtype="float32"), fc1_bias: R.Tensor((128,), dtype="float32"), fc2_weight: R.Tensor((10, 128), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor(("n", 10), dtype="float32"):
        n = T.int64()
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            env_linear = R.call_dps_packed("env.linear", (x, fc1_weight, fc1_bias), out_sinfo=R.Tensor((n, 128), dtype="float32"))
            lv = R.call_tir(cls.relu, (env_linear,), out_sinfo=R.Tensor((n, 128), dtype="float32"))
            lv1 = R.call_tir(cls.tir_linear, (lv, fc2_weight, fc2_bias), out_sinfo=R.Tensor((n, 10), dtype="float32"))
            gv: R.Tensor((n, 10), dtype="float32") = lv1
            R.output(gv)
        return gv
```


## 使用 Block Builder API 创建 Relax 程序

除了上述 API，我们还提供了 Block Builder API 来创建 Relax 程序。它是一种 IR 构建 API，属于更底层的接口，广泛应用于 TVM 的内部逻辑中，例如编写自定义的 Pass。


```plain
bb = relax.BlockBuilder()
n = T.int64()
x = relax.Var("x", R.Tensor((n, 784), "float32"))
fc1_weight = relax.Var("fc1_weight", R.Tensor((128, 784), "float32"))
fc1_bias = relax.Var("fc1_bias", R.Tensor((128,), "float32"))
fc2_weight = relax.Var("fc2_weight", R.Tensor((10, 128), "float32"))
fc2_bias = relax.Var("fc2_bias", R.Tensor((10,), "float32"))
with bb.function("forward", [x, fc1_weight, fc1_bias, fc2_weight, fc2_bias]):
    with bb.dataflow():
        lv0 = bb.emit(relax.op.matmul(x, relax.op.permute_dims(fc1_weight)) + fc1_bias)
        lv1 = bb.emit(relax.op.nn.relu(lv0))
        gv = bb.emit(relax.op.matmul(lv1, relax.op.permute_dims(fc2_weight)) + fc2_bias)
        bb.emit_output(gv)
    bb.emit_func_output(gv)

mod = bb.get()
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
    def forward(x: R.Tensor(("v", 784), dtype="float32"), fc1_weight: R.Tensor((128, 784), dtype="float32"), fc1_bias: R.Tensor((128,), dtype="float32"), fc2_weight: R.Tensor((10, 128), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor(("v", 10), dtype="float32"):
        v = T.int64()
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
            lv1: R.Tensor((v, 128), dtype="float32") = R.matmul(x, lv, out_dtype="void")
            lv2: R.Tensor((v, 128), dtype="float32") = R.add(lv1, fc1_bias)
            lv3: R.Tensor((v, 128), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
            lv5: R.Tensor((v, 10), dtype="float32") = R.matmul(lv3, lv4, out_dtype="void")
            lv6: R.Tensor((v, 10), dtype="float32") = R.add(lv5, fc2_bias)
            gv: R.Tensor((v, 10), dtype="float32") = lv6
            R.output(gv)
        return lv6
```


Block Builder API 同样支持构建跨层级的 IRModule，既可以包含 Relax 函数、TensorIR 函数，也可以包含其他 TVM 的打包函数。


```plain
bb = relax.BlockBuilder()
with bb.function("forward", [x, fc1_weight, fc1_bias, fc2_weight, fc2_bias]):
    with bb.dataflow():
        lv0 = bb.emit(
            relax.call_dps_packed(
                "env.linear",
                [x, fc1_weight, fc1_bias],
                out_sinfo=relax.TensorStructInfo((n, 128), "float32"),
            )
        )
        lv1 = bb.emit_te(topi.nn.relu, lv0)
        tir_gv = bb.add_func(tir_linear, "tir_linear")
        gv = bb.emit(
            relax.call_tir(
                tir_gv,
                [lv1, fc2_weight, fc2_bias],
                out_sinfo=relax.TensorStructInfo((n, 10), "float32"),
            )
        )
        bb.emit_output(gv)
    bb.emit_func_output(gv)
mod = bb.get()
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
    def relu(var_lv: T.handle, var_compute: T.handle):
        T.func_attr({"tir.noalias": True})
        v = T.int64()
        lv = T.match_buffer(var_lv, (v, T.int64(128)))
        compute = T.match_buffer(var_compute, (v, T.int64(128)))
        # with T.block("root"):
        for i0, i1 in T.grid(v, T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(lv[v_i0, v_i1], T.float32(0.0))

    @T.prim_func
    def tir_linear(x: T.handle, w: T.handle, b: T.handle, z: T.handle):
        M, K = T.int64(), T.int64()
        X = T.match_buffer(x, (M, K))
        N = T.int64()
        W = T.match_buffer(w, (N, K))
        B = T.match_buffer(b, (N,))
        Z = T.match_buffer(z, (M, N))
        # with T.block("root"):
        for i, j, k in T.grid(M, N, K):
            with T.block("linear"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(X[vi, vk], W[vj, vk])
                T.writes(Z[vi, vj])
                with T.init():
                    Z[vi, vj] = T.float32(0.0)
                Z[vi, vj] = Z[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(M, N):
            with T.block("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(Z[vi, vj], B[vj])
                T.writes(Z[vi, vj])
                Z[vi, vj] = Z[vi, vj] + B[vj]

    @R.function
    def forward(x: R.Tensor(("v", 784), dtype="float32"), fc1_weight: R.Tensor((128, 784), dtype="float32"), fc1_bias: R.Tensor((128,), dtype="float32"), fc2_weight: R.Tensor((10, 128), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor(("v", 10), dtype="float32"):
        v = T.int64()
        cls = Module
        with R.dataflow():
            lv = R.call_dps_packed("env.linear", (x, fc1_weight, fc1_bias), out_sinfo=R.Tensor((v, 128), dtype="float32"))
            lv1 = R.call_tir(cls.relu, (lv,), out_sinfo=R.Tensor((v, 128), dtype="float32"))
            lv2 = R.call_tir(cls.tir_linear, (lv1, fc2_weight, fc2_bias), out_sinfo=R.Tensor((v, 10), dtype="float32"))
            gv: R.Tensor((v, 10), dtype="float32") = lv2
            R.output(gv)
        return lv2
```



需要注意的是，Block Builder API 的使用体验不如前述 API 那么友好，但它是最低层的 API，并且与 IR 定义紧密耦合。我们推荐普通用户在仅需定义和转换机器学习模型的场景中使用前面提到的 API；而对于需要进行更复杂转换的用户，Block Builder API 提供了更高的灵活性。


## 总结

本教程演示了如何使用 TVMScript、NNModule API、Block Builder API 以及 PackedFunc API，根据不同的应用场景来创建 Relax 程序。
* [下载 Jupyter notebook: relax_creation.ipynb](https://tvm.apache.org/docs/_downloads/315fcda965a8d605f81705edf19ea2c6/relax_creation.ipynb)
* [下载 Python 源码: relax_creation.py](https://tvm.apache.org/docs/_downloads/a67d9d8c1ea48d5e0c95591f8c9bfd6f/relax_creation.py)
* [下载压缩包: relax_creation.zip](https://tvm.apache.org/docs/_downloads/4753776bbe68e7c9ee4d19117973fc8b/relax_creation.zip)


