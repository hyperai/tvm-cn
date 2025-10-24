---

title: IRModule

---


:::note

本教程可通过 Google Colab 交互式运行！也可点击[此处](https://tvm.hyper.ai/docs/getting-started/irmodule/#%E9%83%A8%E7%BD%B2%E5%88%B0%E5%85%B6%E4%BB%96%E5%90%8E%E7%AB%AF)在本地运行 Jupyter Notebook。

[在 Google Colab 中打开](https://colab.research.google.com/github/apache/tvm-site/blob/asf-site/docs/_downloads/a6d7947451d373bc811080cffa18dc7c/ir_module.ipynb)

:::


本教程介绍了 Apache TVM Unity 的核心抽象——IRModule。IRModule 表示整个机器学习模型，涵盖计算图、张量程序以及对外部库的潜在调用。


**目录**
* [创建 IRModule](https://tvm.hyper.ai/docs/getting-started/irmodule/#%E5%88%9B%E5%BB%BA-irmodule)
* [IRModule 的属性](https://tvm.hyper.ai/docs/getting-started/irmodule/#irmodule-%E7%9A%84%E5%B1%9E%E6%80%A7)
* [对 IRModule 进行转换](https://tvm.hyper.ai/docs/getting-started/irmodule/#%E5%AF%B9-irmodule-%E8%BF%9B%E8%A1%8C%E8%BD%AC%E6%8D%A2)
* [通用部署 IRModule](https://tvm.hyper.ai/docs/getting-started/irmodule/#%E9%80%9A%E7%94%A8%E9%83%A8%E7%BD%B2-irmodule)


```plain
import numpy as np
import tvm
from tvm import relax
```


## 创建 IRModule

IRModule 可以通过多种方式初始化。下面我们演示几种常见的方法。

```plain
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
```

### 从已有模型导入

最常见的初始化方式是从已有模型导入。Apache TVM Unity 支持从多个主流框架（如 PyTorch 和 ONNX）导入模型。本教程中仅演示如何从 PyTorch 导入。

```plain
# 创建一个简单的模型
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


# 提供一个示例输入用于 torch.export
example_args = (torch.randn(1, 784, dtype=torch.float32),)

# 将模型转换为 IRModule
with torch.no_grad():
    exported_program = export(TorchModel().eval(), example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
    )

mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)
# 打印 IRModule
mod_from_torch.show()
```


输出：

```plain
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32"), p_fc1_weight: R.Tensor((256, 784), dtype="float32"), p_fc1_bias: R.Tensor((256,), dtype="float32"), p_fc2_weight: R.Tensor((10, 256), dtype="float32"), p_fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            lv: R.Tensor((784, 256), dtype="float32") = R.permute_dims(p_fc1_weight, axes=None)
            lv1: R.Tensor((1, 256), dtype="float32") = R.matmul(x, lv, out_dtype="float32")
            lv2: R.Tensor((1, 256), dtype="float32") = R.add(lv1, p_fc1_bias)
            lv3: R.Tensor((1, 256), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((256, 10), dtype="float32") = R.permute_dims(p_fc2_weight, axes=None)
            lv5: R.Tensor((1, 10), dtype="float32") = R.matmul(lv3, lv4, out_dtype="float32")
            lv6: R.Tensor((1, 10), dtype="float32") = R.add(lv5, p_fc2_bias)
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return gv
```


### 使用 Relax NN 模块编写

Apache TVM Unity 还提供了一套类似于 PyTorch 的 API，帮助用户直接编写 IRModule。

```plain
from tvm.relax.frontend import nn


class RelaxModel(nn.Module):
    def __init__(self):
        super(RelaxModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


mod_from_relax, params_from_relax = RelaxModel().export_tvm(
    {"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
mod_from_relax.show()
```


输出：

```plain
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def forward(x: R.Tensor((1, 784), dtype="float32"), fc1_weight: R.Tensor((256, 784), dtype="float32"), fc1_bias: R.Tensor((256,), dtype="float32"), fc2_weight: R.Tensor((10, 256), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims: R.Tensor((784, 256), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
            matmul: R.Tensor((1, 256), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((1, 256), dtype="float32") = R.add(matmul, fc1_bias)
            relu: R.Tensor((1, 256), dtype="float32") = R.nn.relu(add)
            permute_dims1: R.Tensor((256, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
            matmul1: R.Tensor((1, 10), dtype="float32") = R.matmul(relu, permute_dims1, out_dtype="void")
            add1: R.Tensor((1, 10), dtype="float32") = R.add(matmul1, fc2_bias)
            gv: R.Tensor((1, 10), dtype="float32") = add1
            R.output(gv)
        return gv
```


### 通过 TVMScript 创建

TVMScript 是一种基于 Python 的 DSL（领域特定语言），用于编写 IRModule。我们可以直接以 TVMScript 语法输出 IRModule，或者解析 TVMScript 来生成 IRModule。


```plain
from tvm.script import ir as I
from tvm.script import relax as R


@I.ir_module
class TVMScriptModule:
    @R.function
    def main(
        x: R.Tensor((1, 784), dtype="float32"),
        fc1_weight: R.Tensor((256, 784), dtype="float32"),
        fc1_bias: R.Tensor((256,), dtype="float32"),
        fc2_weight: R.Tensor((10, 256), dtype="float32"),
        fc2_bias: R.Tensor((10,), dtype="float32"),
    ) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims = R.permute_dims(fc1_weight, axes=None)
            matmul = R.matmul(x, permute_dims, out_dtype="void")
            add = R.add(matmul, fc1_bias)
            relu = R.nn.relu(add)
            permute_dims1 = R.permute_dims(fc2_weight, axes=None)
            matmul1 = R.matmul(relu, permute_dims1, out_dtype="void")
            add1 = R.add(matmul1, fc2_bias)
            gv = add1
            R.output(gv)
        return gv


mod_from_script = TVMScriptModule
mod_from_script.show()


```


输出：

```plain
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32"), fc1_weight: R.Tensor((256, 784), dtype="float32"), fc1_bias: R.Tensor((256,), dtype="float32"), fc2_weight: R.Tensor((10, 256), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims: R.Tensor((784, 256), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
            matmul: R.Tensor((1, 256), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((1, 256), dtype="float32") = R.add(matmul, fc1_bias)
            relu: R.Tensor((1, 256), dtype="float32") = R.nn.relu(add)
            permute_dims1: R.Tensor((256, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
            matmul1: R.Tensor((1, 10), dtype="float32") = R.matmul(relu, permute_dims1, out_dtype="void")
            add1: R.Tensor((1, 10), dtype="float32") = R.add(matmul1, fc2_bias)
            gv: R.Tensor((1, 10), dtype="float32") = add1
            R.output(gv)
        return gv
```


### IRModule 的属性

IRModule 是一个由函数组成的集合，这些函数通过 `GlobalVar` 索引。

```plain
mod = mod_from_torch
print(mod.get_global_vars())
```


输出：

```plain
[I.GlobalVar("main")]
```


我们可以通过 GlobalVar 或其名称来访问 IRModule 中的函数：

```plain
# 通过全局变量名索引
print(mod["main"])
# 通过 GlobalVar 索引，并验证它们是相同的函数
(gv,) = mod.get_global_vars()
assert mod[gv] == mod["main"]
```


输出：

```plain
# from tvm.script import relax as R

@R.function
def main(x: R.Tensor((1, 784), dtype="float32"), p_fc1_weight: R.Tensor((256, 784), dtype="float32"), p_fc1_bias: R.Tensor((256,), dtype="float32"), p_fc2_weight: R.Tensor((10, 256), dtype="float32"), p_fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
    R.func_attr({"num_input": 1})
    with R.dataflow():
        lv: R.Tensor((784, 256), dtype="float32") = R.permute_dims(p_fc1_weight, axes=None)
        lv1: R.Tensor((1, 256), dtype="float32") = R.matmul(x, lv, out_dtype="float32")
        lv2: R.Tensor((1, 256), dtype="float32") = R.add(lv1, p_fc1_bias)
        lv3: R.Tensor((1, 256), dtype="float32") = R.nn.relu(lv2)
        lv4: R.Tensor((256, 10), dtype="float32") = R.permute_dims(p_fc2_weight, axes=None)
        lv5: R.Tensor((1, 10), dtype="float32") = R.matmul(lv3, lv4, out_dtype="float32")
        lv6: R.Tensor((1, 10), dtype="float32") = R.add(lv5, p_fc2_bias)
        gv: R.Tensor((1, 10), dtype="float32") = lv6
        R.output(gv)
    return gv
```


### 对 IRModule 进行转换

转换是 Apache TVM Unity 的一个重要组成部分。一个转换接受一个 IRModule 并输出另一个 IRModule。我们可以对 IRModule 应用一系列转换，得到一个新的 IRModule。这是优化模型的常见方式。


在本入门教程中，我们仅展示如何对 IRModule 应用转换。关于每个转换的详细信息，请参考 [转换 API 文档](https://tvm.apache.org/docs/reference/api/python/relax/transform.html#api-relax-transformation)。


我们首先对 IRModule 应用 LegalizeOps 转换。该转换会将 Relax 模块转换为一个混合阶段，即在同一个模块中同时包含 Relax 和 TensorIR 的函数。同时，Relax 操作会被转化为 `call_tir` 的形式。


```plain
mod = mod_from_torch
mod = relax.transform.LegalizeOps()(mod)
mod.show()
```


输出：

```plain
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv1: T.Buffer((T.int64(1), T.int64(256)), "float32"), p_fc1_bias: T.Buffer((T.int64(256),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(256)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(256)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv1[v_ax0, v_ax1], p_fc1_bias[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv1[v_ax0, v_ax1] + p_fc1_bias[v_ax1]

    @T.prim_func(private=True)
    def add1(lv5: T.Buffer((T.int64(1), T.int64(10)), "float32"), p_fc2_bias: T.Buffer((T.int64(10),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv5[v_ax0, v_ax1], p_fc2_bias[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv5[v_ax0, v_ax1] + p_fc2_bias[v_ax1]

    @T.prim_func(private=True)
    def matmul(x: T.Buffer((T.int64(1), T.int64(784)), "float32"), lv: T.Buffer((T.int64(784), T.int64(256)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(256)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(256), T.int64(784)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], lv[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + x[v_i0, v_k] * lv[v_k, v_i1]

    @T.prim_func(private=True)
    def matmul1(lv3: T.Buffer((T.int64(1), T.int64(256)), "float32"), lv4: T.Buffer((T.int64(256), T.int64(10)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(10), T.int64(256)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv3[v_i0, v_k], lv4[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + lv3[v_i0, v_k] * lv4[v_k, v_i1]

    @T.prim_func(private=True)
    def relu(lv2: T.Buffer((T.int64(1), T.int64(256)), "float32"), compute: T.Buffer((T.int64(1), T.int64(256)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(256)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv2[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(lv2[v_i0, v_i1], T.float32(0.0))

    @T.prim_func(private=True)
    def transpose(p_fc1_weight: T.Buffer((T.int64(256), T.int64(784)), "float32"), T_transpose: T.Buffer((T.int64(784), T.int64(256)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(784), T.int64(256)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(p_fc1_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = p_fc1_weight[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose1(p_fc2_weight: T.Buffer((T.int64(10), T.int64(256)), "float32"), T_transpose: T.Buffer((T.int64(256), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(256), T.int64(10)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(p_fc2_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = p_fc2_weight[v_ax1, v_ax0]

    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32"), p_fc1_weight: R.Tensor((256, 784), dtype="float32"), p_fc1_bias: R.Tensor((256,), dtype="float32"), p_fc2_weight: R.Tensor((10, 256), dtype="float32"), p_fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.transpose, (p_fc1_weight,), out_sinfo=R.Tensor((784, 256), dtype="float32"))
            lv1 = R.call_tir(cls.matmul, (x, lv), out_sinfo=R.Tensor((1, 256), dtype="float32"))
            lv2 = R.call_tir(cls.add, (lv1, p_fc1_bias), out_sinfo=R.Tensor((1, 256), dtype="float32"))
            lv3 = R.call_tir(cls.relu, (lv2,), out_sinfo=R.Tensor((1, 256), dtype="float32"))
            lv4 = R.call_tir(cls.transpose1, (p_fc2_weight,), out_sinfo=R.Tensor((256, 10), dtype="float32"))
            lv5 = R.call_tir(cls.matmul1, (lv3, lv4), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            lv6 = R.call_tir(cls.add1, (lv5, p_fc2_bias), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return gv
```


经过该转换后，模块中会多出很多函数。我们可以再次打印出全局变量进行查看。

```plain
print(mod.get_global_vars())
```


输出：

```plain
[I.GlobalVar("add"), I.GlobalVar("add1"), I.GlobalVar("main"), I.GlobalVar("matmul"), I.GlobalVar("matmul1"), I.GlobalVar("relu"), I.GlobalVar("transpose"), I.GlobalVar("transpose1")]
```


接着，Apache TVM Unity 提供了一套默认的转换流水线，以简化转换过程。我们可以将默认的转换流水线应用到模块上。其中默认的 zero 流水线包含以下基本转换步骤：
* **LegalizeOps：** 将 Relax 操作转换为 call_tir 调用，配合相应的 TensorIR 函数。此转换会让 IRModule 同时包含 Relax 函数与 TensorIR 函数。 
*  **AnnotateTIROpPattern：** 为 TensorIR 函数打上模式标签，为后续的算子融合做好准备。 
*  **FoldConstant：** 进行常量折叠优化，简化涉及常量的运算。 
*  **FuseOps 和 FuseTIR：** 根据上一阶段 AnnotateTIROpPattern 所打的标签对 Relax 和 TensorIR 函数进行融合优化。


:::note

在此流程中我们对 LegalizeOps 转换应用了两次。第二次虽然没有实际效果，但也不会产生任何副作用。


每个转换步骤都可以在转换流程中重复使用，因为我们确保所有转换都可以处理合法的 IRModule 输入。这种设计有助于用户灵活构建自己的转换流水线。

:::


```plain
mod = relax.get_pipeline("zero")(mod)
mod.show()
```


输出：

```plain
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_matmul1_add1(lv3: T.Buffer((T.int64(1), T.int64(256)), "float32"), lv4: T.Buffer((T.int64(256), T.int64(10)), "float32"), p_fc2_bias: T.Buffer((T.int64(10),), "float32"), T_add_intermediate: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(10)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(10), T.int64(256)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv3[v_i0, v_k], lv4[v_k, v_i1])
                T.writes(matmul_intermediate[v_i0, v_i1])
                with T.init():
                    matmul_intermediate[v_i0, v_i1] = T.float32(0.0)
                matmul_intermediate[v_i0, v_i1] = matmul_intermediate[v_i0, v_i1] + lv3[v_i0, v_k] * lv4[v_k, v_i1]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(matmul_intermediate[v_ax0, v_ax1], p_fc2_bias[v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = matmul_intermediate[v_ax0, v_ax1] + p_fc2_bias[v_ax1]

    @T.prim_func(private=True)
    def fused_matmul_add_relu(x: T.Buffer((T.int64(1), T.int64(784)), "float32"), lv: T.Buffer((T.int64(784), T.int64(256)), "float32"), p_fc1_bias: T.Buffer((T.int64(256),), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(256)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(256)))
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(256)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(256), T.int64(784)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], lv[v_k, v_i1])
                T.writes(matmul_intermediate[v_i0, v_i1])
                with T.init():
                    matmul_intermediate[v_i0, v_i1] = T.float32(0.0)
                matmul_intermediate[v_i0, v_i1] = matmul_intermediate[v_i0, v_i1] + x[v_i0, v_k] * lv[v_k, v_i1]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(256)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(matmul_intermediate[v_ax0, v_ax1], p_fc1_bias[v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = matmul_intermediate[v_ax0, v_ax1] + p_fc1_bias[v_ax1]
        for i0, i1 in T.grid(T.int64(1), T.int64(256)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_add_intermediate[v_i0, v_i1])
                T.writes(compute_intermediate[v_i0, v_i1])
                compute_intermediate[v_i0, v_i1] = T.max(T_add_intermediate[v_i0, v_i1], T.float32(0.0))

    @T.prim_func(private=True)
    def transpose(p_fc1_weight: T.Buffer((T.int64(256), T.int64(784)), "float32"), T_transpose: T.Buffer((T.int64(784), T.int64(256)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(784), T.int64(256)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(p_fc1_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = p_fc1_weight[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose1(p_fc2_weight: T.Buffer((T.int64(10), T.int64(256)), "float32"), T_transpose: T.Buffer((T.int64(256), T.int64(10)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(256), T.int64(10)):
            with T.block("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(p_fc2_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = p_fc2_weight[v_ax1, v_ax0]

    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32"), p_fc1_weight: R.Tensor((256, 784), dtype="float32"), p_fc1_bias: R.Tensor((256,), dtype="float32"), p_fc2_weight: R.Tensor((10, 256), dtype="float32"), p_fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.transpose, (p_fc1_weight,), out_sinfo=R.Tensor((784, 256), dtype="float32"))
            lv_1 = R.call_tir(cls.fused_matmul_add_relu, (x, lv, p_fc1_bias), out_sinfo=R.Tensor((1, 256), dtype="float32"))
            lv4 = R.call_tir(cls.transpose1, (p_fc2_weight,), out_sinfo=R.Tensor((256, 10), dtype="float32"))
            gv = R.call_tir(cls.fused_matmul1_add1, (lv_1, lv4, p_fc2_bias), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(gv)
        return gv
```


## 通用部署 IRModule

在完成优化之后，我们可以将模型编译为 TVM 运行时模块。值得注意的是，Apache TVM Unity 支持通用部署功能，也就是说，我们可以将同一个 IRModule 部署到不同的后端，包括 CPU、GPU 以及其他新兴后端。


### 部署到 CPU

我们可以通过将目标设置为 `llvm`，将 IRModule 部署到 CPU 上。

```plain
exec = tvm.compile(mod, target="llvm")
dev = tvm.cpu()
vm = relax.VirtualMachine(exec, dev)

raw_data = np.random.rand(1, 784).astype("float32")
data = tvm.runtime.tensor(raw_data, dev)
cpu_out = vm["main"](data, *params_from_torch["main"]).numpy()
print(cpu_out)
```


输出：

```plain
[[ 6.4867303e-02  1.6763064e-01  9.3035400e-05  1.8091209e-01
   8.0412276e-02 -1.4292052e-01 -3.2873321e-02 -7.4184828e-02
  -6.7507513e-02  1.5245053e-01]]
```


### 部署到 GPU

除了 CPU 后端之外，我们还可以将 IRModule 部署到 GPU 上。与 CPU 不同，GPU 程序需要额外的信息，比如线程绑定和共享内存分配。我们需要进行进一步的转换来生成适用于 GPU 的程序。


我们使用 `DLight` 来生成 GPU 程序。本教程中不深入介绍 `DLight` 的具体细节。

```plain
from tvm import dlight as dl

with tvm.target.Target("cuda"):
    gpu_mod = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.Fallback(),
    )(mod)
```


接下来我们可以像部署到 CPU 那样，将 IRModule 编译并部署到 GPU 上：

```plain
exec = tvm.compile(gpu_mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(exec, dev)
# 需要在 GPU 设备上分配数据和参数。
data = tvm.runtime.tensor(raw_data, dev)
gpu_params = [tvm.runtime.tensor(p, dev) for p in params_from_torch["main"]]
gpu_out = vm["main"](data, *gpu_params).numpy()
print(gpu_out)

# 检查结果的正确性
assert np.allclose(cpu_out, gpu_out, atol=1e-3)
```


输出：

```plain
[[ 6.4867347e-02  1.6763058e-01  9.3113631e-05  1.8091221e-01
   8.0412284e-02 -1.4292067e-01 -3.2873336e-02 -7.4184842e-02
  -6.7507453e-02  1.5245046e-01]]
```


### 部署到其他后端

Apache TVM Unity 也支持部署到其他后端，包括不同类型的 GPU（如 Metal、ROCm、Vulkan 和 OpenCL）、不同类型的 CPU（如 x86 和 ARM），以及其他新兴平台（例如 WebAssembly）。部署流程与 GPU 后端类似（可右键另存为下载）。
* [下载 Jupyter Notebook: ir_module.ipynb](https://tvm.apache.org/docs/_downloads/a6d7947451d373bc811080cffa18dc7c/ir_module.ipynb) 
* [下载 Python 源代码: ir_module.py](https://tvm.apache.org/docs/_downloads/0b64717d4cc6027368b96fad40119738/ir_module.py) 
* [下载压缩包: ir_module.zip](https://tvm.apache.org/docs/_downloads/11c11e53c7dace51a8be968ee169ed0d/ir_module.zip)


