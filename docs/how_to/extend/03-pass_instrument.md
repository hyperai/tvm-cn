---
title: 如何使用 TVM Pass Instrument
---

# 如何使用 TVM Pass Instrument

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/extend_tvm/use_pass_instrument.html#sphx-glr-download-how-to-extend-tvm-use-pass-instrument-py) 下载完整的示例代码
:::

**作者**：[Chi-Wei Wang](https://github.com/chiwwang)

随着实现的 Pass 越来越多，instrument pass 执行、分析每个 Pass 效果和观察各种事件也愈发重要。

可以通过向 `tvm.transform.PassContext` 提供 `tvm.ir.instrument.PassInstrument` 实例列表来检测 Pass。我们提供了一个用于收集计时信息的 pass 工具（`tvm.ir.instrument.PassTimingInstrument`），可以通过 `tvm.instrument.pass_instrument()` 装饰器使用扩展机制。

本教程演示开发者如何用 `PassContext` 检测 Pass。另请参阅 [Pass Infrastructure](https://tvm.apache.org/docs/arch/pass_infra.html#pass-infra)。

``` python
import tvm
import tvm.relay as relay
from tvm.relay.testing import resnet
from tvm.contrib.download import download_testdata
from tvm.relay.build_module import bind_params_by_name
from tvm.ir.instrument import (
    PassTimingInstrument,
    pass_instrument,
)
```

## 创建 Relay 程序示例

在 Relay 中使用预定义的 ResNet-18 网络。

``` python
batch_size = 1
num_of_image_class = 1000
image_shape = (3, 224, 224)
output_shape = (batch_size, num_of_image_class)
relay_mod, relay_params = resnet.get_workload(num_layers=18, batch_size=1, image_shape=image_shape)
print("Printing the IR module...")
print(relay_mod.astext(show_meta_data=False))
```

输出结果：

``` bash
Printing the IR module...
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 3, 224, 224), float32] /* ty=Tensor[(1, 3, 224, 224), float32] */, %bn_data_gamma: Tensor[(3), float32] /* ty=Tensor[(3), float32] */, %bn_data_beta: Tensor[(3), float32] /* ty=Tensor[(3), float32] */, %bn_data_moving_mean: Tensor[(3), float32] /* ty=Tensor[(3), float32] */, %bn_data_moving_var: Tensor[(3), float32] /* ty=Tensor[(3), float32] */, %conv0_weight: Tensor[(64, 3, 7, 7), float32] /* ty=Tensor[(64, 3, 7, 7), float32] */, %bn0_gamma: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %bn0_beta: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %bn0_moving_mean: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %bn0_moving_var: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit1_bn1_gamma: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit1_bn1_beta: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit1_bn1_moving_mean: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit1_bn1_moving_var: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit1_conv1_weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %stage1_unit1_bn2_gamma: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit1_bn2_beta: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit1_bn2_moving_mean: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit1_bn2_moving_var: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit1_conv2_weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %stage1_unit1_sc_weight: Tensor[(64, 64, 1, 1), float32] /* ty=Tensor[(64, 64, 1, 1), float32] */, %stage1_unit2_bn1_gamma: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit2_bn1_beta: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit2_bn1_moving_mean: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit2_bn1_moving_var: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit2_conv1_weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %stage1_unit2_bn2_gamma: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit2_bn2_beta: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit2_bn2_moving_mean: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit2_bn2_moving_var: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage1_unit2_conv2_weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %stage2_unit1_bn1_gamma: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage2_unit1_bn1_beta: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage2_unit1_bn1_moving_mean: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage2_unit1_bn1_moving_var: Tensor[(64), float32] /* ty=Tensor[(64), float32] */, %stage2_unit1_conv1_weight: Tensor[(128, 64, 3, 3), float32] /* ty=Tensor[(128, 64, 3, 3), float32] */, %stage2_unit1_bn2_gamma: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit1_bn2_beta: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit1_bn2_moving_mean: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit1_bn2_moving_var: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit1_conv2_weight: Tensor[(128, 128, 3, 3), float32] /* ty=Tensor[(128, 128, 3, 3), float32] */, %stage2_unit1_sc_weight: Tensor[(128, 64, 1, 1), float32] /* ty=Tensor[(128, 64, 1, 1), float32] */, %stage2_unit2_bn1_gamma: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit2_bn1_beta: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit2_bn1_moving_mean: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit2_bn1_moving_var: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit2_conv1_weight: Tensor[(128, 128, 3, 3), float32] /* ty=Tensor[(128, 128, 3, 3), float32] */, %stage2_unit2_bn2_gamma: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit2_bn2_beta: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit2_bn2_moving_mean: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit2_bn2_moving_var: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage2_unit2_conv2_weight: Tensor[(128, 128, 3, 3), float32] /* ty=Tensor[(128, 128, 3, 3), float32] */, %stage3_unit1_bn1_gamma: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage3_unit1_bn1_beta: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage3_unit1_bn1_moving_mean: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage3_unit1_bn1_moving_var: Tensor[(128), float32] /* ty=Tensor[(128), float32] */, %stage3_unit1_conv1_weight: Tensor[(256, 128, 3, 3), float32] /* ty=Tensor[(256, 128, 3, 3), float32] */, %stage3_unit1_bn2_gamma: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit1_bn2_beta: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit1_bn2_moving_mean: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit1_bn2_moving_var: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit1_conv2_weight: Tensor[(256, 256, 3, 3), float32] /* ty=Tensor[(256, 256, 3, 3), float32] */, %stage3_unit1_sc_weight: Tensor[(256, 128, 1, 1), float32] /* ty=Tensor[(256, 128, 1, 1), float32] */, %stage3_unit2_bn1_gamma: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit2_bn1_beta: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit2_bn1_moving_mean: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit2_bn1_moving_var: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit2_conv1_weight: Tensor[(256, 256, 3, 3), float32] /* ty=Tensor[(256, 256, 3, 3), float32] */, %stage3_unit2_bn2_gamma: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit2_bn2_beta: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit2_bn2_moving_mean: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit2_bn2_moving_var: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage3_unit2_conv2_weight: Tensor[(256, 256, 3, 3), float32] /* ty=Tensor[(256, 256, 3, 3), float32] */, %stage4_unit1_bn1_gamma: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage4_unit1_bn1_beta: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage4_unit1_bn1_moving_mean: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage4_unit1_bn1_moving_var: Tensor[(256), float32] /* ty=Tensor[(256), float32] */, %stage4_unit1_conv1_weight: Tensor[(512, 256, 3, 3), float32] /* ty=Tensor[(512, 256, 3, 3), float32] */, %stage4_unit1_bn2_gamma: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit1_bn2_beta: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit1_bn2_moving_mean: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit1_bn2_moving_var: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit1_conv2_weight: Tensor[(512, 512, 3, 3), float32] /* ty=Tensor[(512, 512, 3, 3), float32] */, %stage4_unit1_sc_weight: Tensor[(512, 256, 1, 1), float32] /* ty=Tensor[(512, 256, 1, 1), float32] */, %stage4_unit2_bn1_gamma: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit2_bn1_beta: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit2_bn1_moving_mean: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit2_bn1_moving_var: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit2_conv1_weight: Tensor[(512, 512, 3, 3), float32] /* ty=Tensor[(512, 512, 3, 3), float32] */, %stage4_unit2_bn2_gamma: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit2_bn2_beta: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit2_bn2_moving_mean: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit2_bn2_moving_var: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %stage4_unit2_conv2_weight: Tensor[(512, 512, 3, 3), float32] /* ty=Tensor[(512, 512, 3, 3), float32] */, %bn1_gamma: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %bn1_beta: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %bn1_moving_mean: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %bn1_moving_var: Tensor[(512), float32] /* ty=Tensor[(512), float32] */, %fc1_weight: Tensor[(1000, 512), float32] /* ty=Tensor[(1000, 512), float32] */, %fc1_bias: Tensor[(1000), float32] /* ty=Tensor[(1000), float32] */) -> Tensor[(1, 1000), float32] {
  %0 = nn.batch_norm(%data, %bn_data_gamma, %bn_data_beta, %bn_data_moving_mean, %bn_data_moving_var, epsilon=2e-05f, scale=False) /* ty=(Tensor[(1, 3, 224, 224), float32], Tensor[(3), float32], Tensor[(3), float32]) */;
  %1 = %0.0 /* ty=Tensor[(1, 3, 224, 224), float32] */;
  %2 = nn.conv2d(%1, %conv0_weight, strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7]) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %3 = nn.batch_norm(%2, %bn0_gamma, %bn0_beta, %bn0_moving_mean, %bn0_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 112, 112), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %4 = %3.0 /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %5 = nn.relu(%4) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %6 = nn.max_pool2d(%5, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %7 = nn.batch_norm(%6, %stage1_unit1_bn1_gamma, %stage1_unit1_bn1_beta, %stage1_unit1_bn1_moving_mean, %stage1_unit1_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %8 = %7.0 /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %9 = nn.relu(%8) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %10 = nn.conv2d(%9, %stage1_unit1_conv1_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %11 = nn.batch_norm(%10, %stage1_unit1_bn2_gamma, %stage1_unit1_bn2_beta, %stage1_unit1_bn2_moving_mean, %stage1_unit1_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %12 = %11.0 /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %13 = nn.relu(%12) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %14 = nn.conv2d(%13, %stage1_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %15 = nn.conv2d(%9, %stage1_unit1_sc_weight, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %16 = add(%14, %15) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %17 = nn.batch_norm(%16, %stage1_unit2_bn1_gamma, %stage1_unit2_bn1_beta, %stage1_unit2_bn1_moving_mean, %stage1_unit2_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %18 = %17.0 /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %19 = nn.relu(%18) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %20 = nn.conv2d(%19, %stage1_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %21 = nn.batch_norm(%20, %stage1_unit2_bn2_gamma, %stage1_unit2_bn2_beta, %stage1_unit2_bn2_moving_mean, %stage1_unit2_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %22 = %21.0 /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %23 = nn.relu(%22) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %24 = nn.conv2d(%23, %stage1_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %25 = add(%24, %16) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %26 = nn.batch_norm(%25, %stage2_unit1_bn1_gamma, %stage2_unit1_bn1_beta, %stage2_unit1_bn1_moving_mean, %stage2_unit1_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %27 = %26.0 /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %28 = nn.relu(%27) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %29 = nn.conv2d(%28, %stage2_unit1_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %30 = nn.batch_norm(%29, %stage2_unit1_bn2_gamma, %stage2_unit1_bn2_beta, %stage2_unit1_bn2_moving_mean, %stage2_unit1_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 128, 28, 28), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %31 = %30.0 /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %32 = nn.relu(%31) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %33 = nn.conv2d(%32, %stage2_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %34 = nn.conv2d(%28, %stage2_unit1_sc_weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %35 = add(%33, %34) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %36 = nn.batch_norm(%35, %stage2_unit2_bn1_gamma, %stage2_unit2_bn1_beta, %stage2_unit2_bn1_moving_mean, %stage2_unit2_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 128, 28, 28), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %37 = %36.0 /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %38 = nn.relu(%37) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %39 = nn.conv2d(%38, %stage2_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %40 = nn.batch_norm(%39, %stage2_unit2_bn2_gamma, %stage2_unit2_bn2_beta, %stage2_unit2_bn2_moving_mean, %stage2_unit2_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 128, 28, 28), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %41 = %40.0 /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %42 = nn.relu(%41) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %43 = nn.conv2d(%42, %stage2_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %44 = add(%43, %35) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %45 = nn.batch_norm(%44, %stage3_unit1_bn1_gamma, %stage3_unit1_bn1_beta, %stage3_unit1_bn1_moving_mean, %stage3_unit1_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 128, 28, 28), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %46 = %45.0 /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %47 = nn.relu(%46) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %48 = nn.conv2d(%47, %stage3_unit1_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %49 = nn.batch_norm(%48, %stage3_unit1_bn2_gamma, %stage3_unit1_bn2_beta, %stage3_unit1_bn2_moving_mean, %stage3_unit1_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 256, 14, 14), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %50 = %49.0 /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %51 = nn.relu(%50) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %52 = nn.conv2d(%51, %stage3_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %53 = nn.conv2d(%47, %stage3_unit1_sc_weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %54 = add(%52, %53) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %55 = nn.batch_norm(%54, %stage3_unit2_bn1_gamma, %stage3_unit2_bn1_beta, %stage3_unit2_bn1_moving_mean, %stage3_unit2_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 256, 14, 14), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %56 = %55.0 /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %57 = nn.relu(%56) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %58 = nn.conv2d(%57, %stage3_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %59 = nn.batch_norm(%58, %stage3_unit2_bn2_gamma, %stage3_unit2_bn2_beta, %stage3_unit2_bn2_moving_mean, %stage3_unit2_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 256, 14, 14), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %60 = %59.0 /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %61 = nn.relu(%60) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %62 = nn.conv2d(%61, %stage3_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %63 = add(%62, %54) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %64 = nn.batch_norm(%63, %stage4_unit1_bn1_gamma, %stage4_unit1_bn1_beta, %stage4_unit1_bn1_moving_mean, %stage4_unit1_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 256, 14, 14), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %65 = %64.0 /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %66 = nn.relu(%65) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %67 = nn.conv2d(%66, %stage4_unit1_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %68 = nn.batch_norm(%67, %stage4_unit1_bn2_gamma, %stage4_unit1_bn2_beta, %stage4_unit1_bn2_moving_mean, %stage4_unit1_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 512, 7, 7), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %69 = %68.0 /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %70 = nn.relu(%69) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %71 = nn.conv2d(%70, %stage4_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %72 = nn.conv2d(%66, %stage4_unit1_sc_weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %73 = add(%71, %72) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %74 = nn.batch_norm(%73, %stage4_unit2_bn1_gamma, %stage4_unit2_bn1_beta, %stage4_unit2_bn1_moving_mean, %stage4_unit2_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 512, 7, 7), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %75 = %74.0 /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %76 = nn.relu(%75) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %77 = nn.conv2d(%76, %stage4_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %78 = nn.batch_norm(%77, %stage4_unit2_bn2_gamma, %stage4_unit2_bn2_beta, %stage4_unit2_bn2_moving_mean, %stage4_unit2_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 512, 7, 7), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %79 = %78.0 /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %80 = nn.relu(%79) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %81 = nn.conv2d(%80, %stage4_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %82 = add(%81, %73) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %83 = nn.batch_norm(%82, %bn1_gamma, %bn1_beta, %bn1_moving_mean, %bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 512, 7, 7), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %84 = %83.0 /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %85 = nn.relu(%84) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %86 = nn.global_avg_pool2d(%85) /* ty=Tensor[(1, 512, 1, 1), float32] */;
  %87 = nn.batch_flatten(%86) /* ty=Tensor[(1, 512), float32] */;
  %88 = nn.dense(%87, %fc1_weight, units=1000) /* ty=Tensor[(1, 1000), float32] */;
  %89 = nn.bias_add(%88, %fc1_bias, axis=-1) /* ty=Tensor[(1, 1000), float32] */;
  nn.softmax(%89) /* ty=Tensor[(1, 1000), float32] */
}
```

## 使用 Instrument 创建 PassContext

要用 instrument 运行所有 Pass，将其通过参数 `instruments` 传递给构造函数 `PassContext`。`PassTimingInstrument` 用于分析每个 Pass 执行时间的内置函数。

``` python
timing_inst = PassTimingInstrument()
with tvm.transform.PassContext(instruments=[timing_inst]):
    relay_mod = relay.transform.InferType()(relay_mod)
    relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
    # 在退出上下文之前，获取配置文件结果。
    profiles = timing_inst.render()
print("Printing results of timing profile...")
print(profiles)
```

输出结果：

``` bash
Printing results of timing profile...
InferType: 6628us [6628us] (46.29%; 46.29%)
FoldScaleAxis: 7691us [6us] (53.71%; 53.71%)
        FoldConstant: 7685us [1578us] (53.67%; 99.92%)
                InferType: 6107us [6107us] (42.65%; 79.47%)
```

## 将当前 PassContext 与 Instrument 一起使用

也可以使用当前的 `PassContext`，并通过 `override_instruments` 方法注册 `PassInstrument` 实例。注意，如果已经存在了任何 instrument，`override_instruments` 将执行 `exit_pass_ctx` 方法。然后它切换到新 instrument，并调用新 instrument 的 `enter_pass_ctx` 方法。有关这些方法，参阅以下部分和 `tvm.instrument.pass_instrument()`。

``` python
cur_pass_ctx = tvm.transform.PassContext.current()
cur_pass_ctx.override_instruments([timing_inst])
relay_mod = relay.transform.InferType()(relay_mod)
relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
profiles = timing_inst.render()
print("Printing results of timing profile...")
print(profiles)
```

输出结果：

``` bash
Printing results of timing profile...
InferType: 6131us [6131us] (44.86%; 44.86%)
FoldScaleAxis: 7536us [4us] (55.14%; 55.14%)
        FoldConstant: 7532us [1549us] (55.11%; 99.94%)
                InferType: 5982us [5982us] (43.77%; 79.43%)
```

注册空列表以清除现有 instrument。

注意，`PassTimingInstrument` 的 `exit_pass_ctx` 被调用了。配置文件被清除，因此不会打印任何内容。

``` python
cur_pass_ctx.override_instruments([])
# 取消 .render() 的注释以查看如下警告：
# 警告：没有 Pass 分析，您是否启用了 Pass 分析？
# profiles = timing_inst.render()
```

## 创建自定义 Instrument 类

可以使用 `tvm.instrument.pass_instrument()` 装饰器创建自定义 instrument 类。

创建一个工具类（计算每次 Pass 引起的每个算子出现次数的变化）。可以在 Pass 之前和之后查看 `op.name` 来找到每个算子的名称，从而计算差异。

``` python
@pass_instrument
class RelayCallNodeDiffer:
    def __init__(self):
        self._op_diff = []
        # Pass 可以嵌套。
        # 使用堆栈来确保得到之前/之后正确的 pairs。
        self._op_cnt_before_stack = []

    def enter_pass_ctx(self):
        self._op_diff = []
        self._op_cnt_before_stack = []

    def exit_pass_ctx(self):
        assert len(self._op_cnt_before_stack) == 0, "The stack is not empty. Something wrong."

    def run_before_pass(self, mod, info):
        self._op_cnt_before_stack.append((info.name, self._count_nodes(mod)))

    def run_after_pass(self, mod, info):
        # 弹出最新记录的 Pass。
        name_before, op_to_cnt_before = self._op_cnt_before_stack.pop()
        assert name_before == info.name, "name_before: {}, info.name: {} doesn't match".format(
            name_before, info.name
        )
        cur_depth = len(self._op_cnt_before_stack)
        op_to_cnt_after = self._count_nodes(mod)
        op_diff = self._diff(op_to_cnt_after, op_to_cnt_before)
        # 只记导致差异的 Pass。
        if op_diff:
            self._op_diff.append((cur_depth, info.name, op_diff))

    def get_pass_to_op_diff(self):
        """
        return [
          (depth, pass_name, {op_name: diff_num, ...}), ...
        ]
        """
        return self._op_diff

    @staticmethod
    def _count_nodes(mod):
        """Count the number of occurrences of each operator in the module"""
        ret = {}

        def visit(node):
            if isinstance(node, relay.expr.Call):
                if hasattr(node.op, "name"):
                    op_name = node.op.name
                else:
                    # 某些 CallNode 可能没有“名称”，例如 relay.Function
                    return
                ret[op_name] = ret.get(op_name, 0) + 1

        relay.analysis.post_order_visit(mod["main"], visit)
        return ret

    @staticmethod
    def _diff(d_after, d_before):
        """Calculate the difference of two dictionary along their keys.
        The result is values in d_after minus values in d_before.
        """
        ret = {}
        key_after, key_before = set(d_after), set(d_before)
        for k in key_before & key_after:
            tmp = d_after[k] - d_before[k]
            if tmp:
                ret[k] = d_after[k] - d_before[k]
        for k in key_after - key_before:
            ret[k] = d_after[k]
        for k in key_before - key_after:
            ret[k] = -d_before[k]
        return ret
```

## 应用 Pass 和多个 Instrument 类

可以在 `PassContext` 中使用多个 instrument 类。但注意，instrument 方法是按 `instruments` 参数的顺序执行的，所以对于像 `PassTimingInstrument` 这样的 instrument 类，不可避免地要将其他 instrument 类的执行时间计入最终的分析结果。

``` python
call_node_inst = RelayCallNodeDiffer()
desired_layouts = {
    "nn.conv2d": ["NHWC", "HWIO"],
}
pass_seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        relay.transform.ConvertLayout(desired_layouts),
        relay.transform.FoldConstant(),
    ]
)
relay_mod["main"] = bind_params_by_name(relay_mod["main"], relay_params)
# timing_inst 放在 call_node_inst 之后。
# 所以 `call_node.inst.run_after_pass()` 的执行时间也算在内。
with tvm.transform.PassContext(opt_level=3, instruments=[call_node_inst, timing_inst]):
    relay_mod = pass_seq(relay_mod)
    profiles = timing_inst.render()
# 取消注释下一行以查看时序配置文件结果。
# print(profiles)
```

输出结果：

``` bash
/workspace/python/tvm/driver/build_module.py:268: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  "target_host parameter is going to be deprecated. "
```

可以看到每个操作类型增加/减少了多少 CallNode。

``` python
from pprint import pprint

print("Printing the change in number of occurrences of each operator caused by each pass...")
pprint(call_node_inst.get_pass_to_op_diff())
```

输出结果：

``` bash
Printing the change in number of occurrences of each operator caused by each pass...
[(1, 'CanonicalizeOps', {'add': 1, 'nn.bias_add': -1}),
 (1, 'ConvertLayout', {'expand_dims': 1, 'layout_transform': 23}),
 (1, 'FoldConstant', {'expand_dims': -1, 'layout_transform': -21}),
 (0, 'sequential', {'add': 1, 'layout_transform': 2, 'nn.bias_add': -1})]
```

## 异常处理

以下演示了 `PassInstrument` 的方法发生异常的详细情况。

定义在进入/退出 `PassContext` 中引发异常的 `PassInstrument` 类：

```plain
class PassExampleBase:
    def __init__(self, name):
        self._name = name

    def enter_pass_ctx(self):
        print(self._name, "enter_pass_ctx")

    def exit_pass_ctx(self):
        print(self._name, "exit_pass_ctx")

    def should_run(self, mod, info):
        print(self._name, "should_run")
        return True

    def run_before_pass(self, mod, pass_info):
        print(self._name, "run_before_pass")

    def run_after_pass(self, mod, pass_info):
        print(self._name, "run_after_pass")

@pass_instrument
class PassFine(PassExampleBase):
    pass

@pass_instrument
class PassBadEnterCtx(PassExampleBase):
    def enter_pass_ctx(self):
        print(self._name, "bad enter_pass_ctx!!!")
        raise ValueError("{} bad enter_pass_ctx".format(self._name))

@pass_instrument
class PassBadExitCtx(PassExampleBase):
    def exit_pass_ctx(self):
        print(self._name, "bad exit_pass_ctx!!!")
        raise ValueError("{} bad exit_pass_ctx".format(self._name))
```

若 `enter_pass_ctx` 发生异常，`PassContext` 将禁用 pass instrumentation。它将运行每个成功完成 `enter_pass_ctx` 的 PassInstrument 的 `exit_pass_ctx`。

下面的例子可以看到  *PassFine_0* 的 `exit_pass_ctx` 在异常后执行。

``` python
demo_ctx = tvm.transform.PassContext(
    instruments=[
        PassFine("PassFine_0"),
        PassBadEnterCtx("PassBadEnterCtx"),
        PassFine("PassFine_1"),
    ]
)
try:
    with demo_ctx:
        relay_mod = relay.transform.InferType()(relay_mod)
except ValueError as ex:
    print("Catching", str(ex).split("\n")[-1])
```

输出结果：

``` bash
PassFine_0 enter_pass_ctx
PassBadEnterCtx bad enter_pass_ctx!!!
PassFine_0 exit_pass_ctx
Catching ValueError: PassBadEnterCtx bad enter_pass_ctx
```

`PassInstrument` 实例中的异常会导致当前的 `PassContext` 所有 instrument 被清除，因此调用 `override_instruments` 时不会打印任何内容。

``` python
demo_ctx.override_instruments([])  # 没有打印 PassFine_0 exit_pass_ctx....等
```

若 `exit_pass_ctx` 发生异常，则禁用 pass  instrument，然后传播异常。这意味着 `PassInstrument` 在抛出异常之后注册的实例不会执行 `exit_pass_ctx`。

``` python
demo_ctx = tvm.transform.PassContext(
    instruments=[
        PassFine("PassFine_0"),
        PassBadExitCtx("PassBadExitCtx"),
        PassFine("PassFine_1"),
    ]
)
try:
    # PassFine_1 执行 enter_pass_ctx，但不执行 exit_pass_ctx。
    with demo_ctx:
        relay_mod = relay.transform.InferType()(relay_mod)
except ValueError as ex:
    print("Catching", str(ex).split("\n")[-1])
```

输出结果：

``` bash
PassFine_0 enter_pass_ctx
PassBadExitCtx enter_pass_ctx
PassFine_1 enter_pass_ctx
PassFine_0 should_run
PassBadExitCtx should_run
PassFine_1 should_run
PassFine_0 run_before_pass
PassBadExitCtx run_before_pass
PassFine_1 run_before_pass
PassFine_0 run_after_pass
PassBadExitCtx run_after_pass
PassFine_1 run_after_pass
PassFine_0 exit_pass_ctx
PassBadExitCtx bad exit_pass_ctx!!!
Catching ValueError: PassBadExitCtx bad exit_pass_ctx
```

以 `run_before_pass`为例：

`should_run`、`run_before_pass` 和 `run_after_pass` 发生的异常没有明确处理，用上下文管理器（`with` 语法）安全退出 `PassContext`。

``` python
@pass_instrument
class PassBadRunBefore(PassExampleBase):
    def run_before_pass(self, mod, pass_info):
        print(self._name, "bad run_before_pass!!!")
        raise ValueError("{} bad run_before_pass".format(self._name))

demo_ctx = tvm.transform.PassContext(
    instruments=[
        PassFine("PassFine_0"),
        PassBadRunBefore("PassBadRunBefore"),
        PassFine("PassFine_1"),
    ]
)
try:
    # 所有的 exit_pass_ctx 都会被调用。
    with demo_ctx:
        relay_mod = relay.transform.InferType()(relay_mod)
except ValueError as ex:
    print("Catching", str(ex).split("\n")[-1])
```

输出结果：

``` bash
PassFine_0 enter_pass_ctx
PassBadRunBefore enter_pass_ctx
PassFine_1 enter_pass_ctx
PassFine_0 should_run
PassBadRunBefore should_run
PassFine_1 should_run
PassFine_0 run_before_pass
PassBadRunBefore bad run_before_pass!!!
PassFine_0 exit_pass_ctx
PassBadRunBefore exit_pass_ctx
PassFine_1 exit_pass_ctx
Catching ValueError: PassBadRunBefore bad run_before_pass
```

注意，pass instrumentation 未禁用。所以若调用 `override_instruments`，`exit_pass_ctx` 先前注册的 `PassInstrument` 将被调用。

``` python
demo_ctx.override_instruments([])
```

输出结果：

``` bash
PassFine_0 exit_pass_ctx
PassBadRunBefore exit_pass_ctx
PassFine_1 exit_pass_ctx
```

若不用 `with` 语法包装 pass 执行，则不会调用  `exit_pass_ctx`。用当前的 `PassContext`：

``` python
cur_pass_ctx = tvm.transform.PassContext.current()
cur_pass_ctx.override_instruments(
    [
        PassFine("PassFine_0"),
        PassBadRunBefore("PassBadRunBefore"),
        PassFine("PassFine_1"),
    ]
)
```

输出结果：

``` bash
PassFine_0 enter_pass_ctx
PassBadRunBefore enter_pass_ctx
PassFine_1 enter_pass_ctx
```

然后调用 Pass。异常后 `exit_pass_ctx` 不执行。

``` python
try:
    # No ``exit_pass_ctx`` got executed.
    relay_mod = relay.transform.InferType()(relay_mod)
except ValueError as ex:
    print("Catching", str(ex).split("\n")[-1])
```

输出结果：

``` bash
PassFine_0 should_run
PassBadRunBefore should_run
PassFine_1 should_run
PassFine_0 run_before_pass
PassBadRunBefore bad run_before_pass!!!
Catching ValueError: PassBadRunBefore bad run_before_pass
```

清除 instrument。

``` python
cur_pass_ctx.override_instruments([])
```

输出结果：

``` bash
PassFine_0 exit_pass_ctx
PassBadRunBefore exit_pass_ctx
PassFine_1 exit_pass_ctx
```

[下载 Python 源代码：use_pass_instrument.py](https://tvm.apache.org/docs/_downloads/d0a1817b910da958b41d88afe4d4952d/use_pass_instrument.py)

[下载 Jupyter Notebook：use_pass_instrument.ipynb](https://tvm.apache.org/docs/_downloads/f6ff0fbc61d45d2cc0f53ebbf11a5fb5/use_pass_instrument.ipynb)
