---
title: 利用 UMA 使硬件加速器可直接用于 TVM
---

# 利用 UMA 使硬件加速器可直接用于 TVM

:::note
单击 [此处](https://tvm.apache.org/docs/tutorial/uma.html#sphx-glr-download-tutorial-uma-py) 下载完整的示例代码
:::

**作者**：[Michael J. Klaiber](https://github.com/MichaelJKlaiber)，[Christoph Gerum](https://github.com/cgerum)，[Paul Palomero Bernardo](https://github.com/PaulPalomeroBernardo/)

本节介绍**通用模块化加速器接口** (UMA)。UMA 提供了一个易用的 API 来将新的硬件加速器集成到 TVM 中。

本教程详细介绍了如何利用 UMA 使得你的硬件加速器可直接用于 TVM。虽然这个问题没有万能的解决方案，但 UMA 旨在提供一个稳定的纯 Python API，从而将许多种类的硬件加速器集成到 TVM 中。

本教程将通过三个逐渐复杂的用例来介绍 UMA API。这些用例引入了三个模拟加速器 **Vanilla**、**Strawberry** 和 **Chocolate**，并用 UMA 将它们集成到 TVM 中。

## Vanilla

**Vanilla** 是一个由 MAC 数组组成的简单加速器，没有内部存储器。它只能处理 Conv2D 层，所有其他层都在 CPU 上执行，同时也协调 **Vanilla**。 CPU 和 Vanilla 共享内存。

**Vanilla** 的 C 接口 `vanilla_conv2dnchw(...)` 用于执行 Conv2D 操作（包括 same-padding），它接收指向输入特征图、权重和结果的指针，以及 Conv2D 的维度：*oc*、*iw*、*ih*、*ic*、*kh* 和 *kw*。

``` cpp
int vanilla_conv2dnchw(float* ifmap, float*  weights, float*  result, int oc, int iw, int ih, int ic, int kh, int kw);
```

脚本 *uma_cli* 为新的加速器创建带有 API (UMA-API) 调用的代码骨架。

**Vanilla** 的使用方式如下：（`--tutorial vanilla` 添加了本部分教程所需的所有附加文件）

``` bash
pip install inflection
cd $TVM_HOME/apps/uma
python uma_cli.py --add_hardware vanilla_accelerator --tutorial vanilla
```

uma_cli.py 在 `vanilla_accelerator` 目录中生成这些文件。

``` bash
backend.py
codegen.py
conv2dnchw.cc
passes.py
patterns.py
run.py
strategies.py
```

Vanilla 后端

vanilla 生成的后端位于 *vanilla_accelerator/backend.py* 中：

``` python
class VanillaAcceleratorBackend(UMABackend):
    """VanillaAccelerator 的 UMA 后端。"""

    def __init__(self):
        super().__init__()

        self._register_pattern("conv2d", conv2d_pattern())
        self._register_tir_pass(PassPhase.TIR_PHASE_0, VanillaAcceleratorConv2DPass())
        self._register_codegen(fmt="c", includes=gen_includes)

    @property
    def target_name(self):
        return "vanilla_accelerator"
```

定义迁移模式

为了指定 *Conv2D* 迁移到 **Vanilla**，*vanilla_accelerator/patterns.py* 中将其描述为 Relay 数据流模式 ([DFPattern](https://tvm.apache.org/docs/reference/langref/relay_pattern.html))。

``` python
def conv2d_pattern():
    pattern = is_op("nn.conv2d")(wildcard(), wildcard())
    pattern = pattern.has_attr({"strides": [1, 1]})
    return pattern
```

为了将输入计算图的 **Conv2D** 算子映射到 **Vanilla** 的底层函数调用 `vanilla_conv2dnchw(...)`，在 VanillaAcceleratorBackend 中注册了 TIR pass *VanillaAcceleratorConv2DPass*（稍后讨论）。

Codegen

文件 `vanilla_accelerator/codegen.py` 定义了静态 C 代码，它被添加到生成的结果 C 代码（由 `gen_includes` 中的 TVM 的 C-Codegen 生成）中，其目的是包含 **Vanilla** 的底层库 `vanilla_conv2dnchw()`。

``` python
def gen_includes() -> str:
    topdir = pathlib.Path(__file__).parent.absolute()

    includes = ""
    includes += f'#include "{topdir}/conv2dnchw.cc"'
    return includes
```

如上面的 *VanillaAcceleratorBackend* 所示，用 *self._register_codegen* 可将其注册到 UMA。

``` python
self._register_codegen(fmt="c", includes=gen_includes)
```

构建神经网络并在 Vanilla 上运行

为了演示 UMA 的功能，将为单个 Conv2D 层生成 C 代码，并在 Vanilla 加速器上运行。文件 `vanilla_accelerator/run.py` 提供了一个使用 Vanilla 的 C-API 运行 Conv2D 层的 demo。

``` python
def main():
    mod, inputs, output_list, runner = create_conv2d()

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)
    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))

    export_directory = tvm.contrib.utils.tempdir(keep_for_debug=True).path
    print(f"Generated files are in {export_directory}")
    compile_and_run(
        AOTModel(module=mod, inputs=inputs, outputs=output_list),
        runner,
        interface_api="c",
        use_unpacked_api=True,
        target=target,
        test_dir=str(export_directory),
    )

main()
```

运行 `vanilla_accelerator/run.py`，将以模型库格式 (MLF) 生成输出文件。

输出结果：

``` bash
Generated files are in /tmp/tvm-debug-mode-tempdirs/2022-07-13T13-26-22___x5u76h0p/00000
```

查看生成的文件：

输出结果：

``` bash
cd /tmp/tvm-debug-mode-tempdirs/2022-07-13T13-26-22___x5u76h0p/00000
cd build/
ls -1

codegen
lib.tar
metadata.json
parameters
runtime
src
```

若要评估生成的 C 代码，请查看 `codegen/host/src/default_lib2.c`。

``` bash
cd codegen/host/src/
ls -1

default_lib0.c
default_lib1.c
default_lib2.c
```

在 *default_lib2.c* 中，可以看到生成的代码调用了 Vanilla 的 C-API，然后执行了一个 Conv2D 层：

``` cpp
TVM_DLL int32_t tvmgen_default_vanilla_accelerator_main_0(float* placeholder, float* placeholder1, float* conv2d_nchw, uint8_t* global_workspace_1_var) {
     vanilla_accelerator_conv2dnchw(placeholder, placeholder1, conv2d_nchw, 32, 14, 14, 32, 3, 3);
     return 0;
}
```

## Strawberry

即将上线

## Chocolate

即将上线

## 征求社区意见

若本教程**不**适合你的加速器，请将你的需求添加到 TVM 论坛中的 [UMA 帖子](https://discuss.tvm.apache.org/t/rfc-uma-universal-modular-accelerator-interface/12039) 中。我们很乐意通过扩展本教程来提供更多指导，例如如何利用 UMA 接口使得更多种类的 AI 硬件加速器可直接用于 TVM。

## 参考

[UMA-RFC] [UMA：通用模块化加速器接口](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0060_UMA_Unified_Modular_Accelerator_Interface.md)，TVM RFC，2022 年 6 月。

[DFPattern] [Relay 中的模式匹配](https://tvm.apache.org/docs/reference/langref/relay_pattern.html)

[下载 Python 源代码：uma.py](https://tvm.apache.org/docs/_downloads/f9c6910c7b4a120c51a9bf48f34f3ad7/uma.py)

[下载 Jupyter Notebook：uma.ipynb](https://tvm.apache.org/docs/_downloads/6e0673ce1f08636c34d0b9a73ea114f7/uma.ipynb)