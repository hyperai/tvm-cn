---
title: 使用 tvmc 接口在 Adreno™ 上部署预训练模型
---
<style>
    pre{
        overflow-y : auto;
        max-height : 900px;
    }
</style>

# 使用 tvmc 接口在 Adreno™ 上部署预训练模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_adreno_tvmc.html#sphx-glr-download-how-to-deploy-models-deploy-model-on-adreno-tvmc-py) 下载完整的示例代码
:::

**作者**：Siva Rama Krishna

本文是一篇关于在 Adreno™ 上部署预训练 Keras resnet50 模型的逐步教程。

此外，您应该已经为 Android 构建了 TVM。请参阅以下说明，了解如何构建它并设置 RPC 环境。

[在 Adreno GPU 上部署](https://tvm.hyper.ai/docs/how_to/deploy/deploy_adreno)

```python
import os
import tvm
import numpy as np
from tvm import relay
from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCPackage
from tvm.contrib import utils
```

# 配置
在编译以生成纹理之前指定 Adreno 目标以利用内核并获得所有纹理的好处。注意：此生成的示例在我们的 x86 服务器上运行以进行演示。如果在 Android 设备上运行它，我们需要指定其指令集。如果要在实际设备上通过 rpc 运行此教程，请将 `local_demo` 设置为 False。

```python
local_demo = True

# 默认情况下，将在 CPU 目标上执行。
# 选择 'llvm'、'opencl' 和 'opencl -device=adreno'
target = "llvm"

# 更改目标配置。
# 运行 `adb shell cat /proc/cpuinfo` 以查找架构。
arch = "arm64"
target_host = "llvm -mtriple=%s-linux-android" % arch

# 自动调整是计算和耗时的任务，因此默认情况下禁用。
# 如果需要，请启用它。请启用它。
is_tuning = False
tune_log = "adreno-resnet50.log"

# 启用 OpenCLML 加速运算符库。
enable_clml = False
cross_compiler = (
    os.getenv("ANDROID_NDK_HOME", "")
    + "/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
)
```

# 制作 Keras Resnet50 模型
```python
from tensorflow.keras.applications.resnet50 import ResNet50

tmp_path = utils.tempdir()
model_file_name = tmp_path.relpath("resnet50.h5")

model = ResNet50(include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000)
model.save(model_file_name)
```

Out:
```info
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5

     8192/102967424 [..............................] - ETA: 0s
  7208960/102967424 [=>............................] - ETA: 0s
  8380416/102967424 [=>............................] - ETA: 1s
 16769024/102967424 [===>..........................] - ETA: 1s
 23412736/102967424 [=====>........................] - ETA: 1s
 25157632/102967424 [======>.......................] - ETA: 1s
 33546240/102967424 [========>.....................] - ETA: 1s
 40189952/102967424 [==========>...................] - ETA: 1s
 41934848/102967424 [===========>..................] - ETA: 1s
 50143232/102967424 [=============>................] - ETA: 1s
 50323456/102967424 [=============>................] - ETA: 1s
 56967168/102967424 [===============>..............] - ETA: 1s
 58712064/102967424 [================>.............] - ETA: 1s
 65355776/102967424 [==================>...........] - ETA: 0s
 67100672/102967424 [==================>...........] - ETA: 0s
 69296128/102967424 [===================>..........] - ETA: 0s
 71540736/102967424 [===================>..........] - ETA: 0s
 73269248/102967424 [====================>.........] - ETA: 0s
 75489280/102967424 [====================>.........] - ETA: 0s
 83877888/102967424 [=======================>......] - ETA: 0s
 90521600/102967424 [=========================>....] - ETA: 0s
 92266496/102967424 [=========================>....] - ETA: 0s
 99598336/102967424 [============================>.] - ETA: 0s
100646912/102967424 [============================>.] - ETA: 0s
102850560/102967424 [============================>.] - ETA: 0s
102967424/102967424 [==============================] - 3s 0us/step
```

# 加载模型
将模型从任何框架转换为 tvm relay 模块。tvmc.load 支持来自任何框架的模型（例如 tensorflow saves_model、onnx、tflite 等），并自动检测文件类型。
```python
tvmc_model = tvmc.load(model_file_name)

print(tvmc_model.mod)


# tvmc_model 包含 tvmc_mode.mod，即 relay 模块和 tvmc_model.params，即模块的参数。
```

Out:

```python
def @main(%input_2: Tensor[(1, 224, 224, 3), float32], %v_param_1: Tensor[(7, 7, 3, 64), float32], %v_param_2: Tensor[(64), float32], %v_param_3: Tensor[(64), float32], %v_param_4: Tensor[(64), float32], %v_param_5: Tensor[(64), float32], %v_param_6: Tensor[(64), float32], %v_param_19: Tensor[(1, 1, 64, 256), float32], %v_param_20: Tensor[(256), float32], %v_param_23: Tensor[(256), float32], %v_param_24: Tensor[(256), float32], %v_param_25: Tensor[(256), float32], %v_param_26: Tensor[(256), float32], %v_param_7: Tensor[(1, 1, 64, 64), float32], %v_param_8: Tensor[(64), float32], %v_param_9: Tensor[(64), float32], %v_param_10: Tensor[(64), float32], %v_param_11: Tensor[(64), float32], %v_param_12: Tensor[(64), float32], %v_param_13: Tensor[(3, 3, 64, 64), float32], %v_param_14: Tensor[(64), float32], %v_param_15: Tensor[(64), float32], %v_param_16: Tensor[(64), float32], %v_param_17: Tensor[(64), float32], %v_param_18: Tensor[(64), float32], %v_param_21: Tensor[(1, 1, 64, 256), float32], %v_param_22: Tensor[(256), float32], %v_param_27: Tensor[(256), float32], %v_param_28: Tensor[(256), float32], %v_param_29: Tensor[(256), float32], %v_param_30: Tensor[(256), float32], %v_param_31: Tensor[(1, 1, 256, 64), float32], %v_param_32: Tensor[(64), float32], %v_param_33: Tensor[(64), float32], %v_param_34: Tensor[(64), float32], %v_param_35: Tensor[(64), float32], %v_param_36: Tensor[(64), float32], %v_param_37: Tensor[(3, 3, 64, 64), float32], %v_param_38: Tensor[(64), float32], %v_param_39: Tensor[(64), float32], %v_param_40: Tensor[(64), float32], %v_param_41: Tensor[(64), float32], %v_param_42: Tensor[(64), float32], %v_param_43: Tensor[(1, 1, 64, 256), float32], %v_param_44: Tensor[(256), float32], %v_param_45: Tensor[(256), float32], %v_param_46: Tensor[(256), float32], %v_param_47: Tensor[(256), float32], %v_param_48: Tensor[(256), float32], %v_param_49: Tensor[(1, 1, 256, 64), float32], %v_param_50: Tensor[(64), float32], %v_param_51: Tensor[(64), float32], %v_param_52: Tensor[(64), float32], %v_param_53: Tensor[(64), float32], %v_param_54: Tensor[(64), float32], %v_param_55: Tensor[(3, 3, 64, 64), float32], %v_param_56: Tensor[(64), float32], %v_param_57: Tensor[(64), float32], %v_param_58: Tensor[(64), float32], %v_param_59: Tensor[(64), float32], %v_param_60: Tensor[(64), float32], %v_param_61: Tensor[(1, 1, 64, 256), float32], %v_param_62: Tensor[(256), float32], %v_param_63: Tensor[(256), float32], %v_param_64: Tensor[(256), float32], %v_param_65: Tensor[(256), float32], %v_param_66: Tensor[(256), float32], %v_param_79: Tensor[(1, 1, 256, 512), float32], %v_param_80: Tensor[(512), float32], %v_param_83: Tensor[(512), float32], %v_param_84: Tensor[(512), float32], %v_param_85: Tensor[(512), float32], %v_param_86: Tensor[(512), float32], %v_param_67: Tensor[(1, 1, 256, 128), float32], %v_param_68: Tensor[(128), float32], %v_param_69: Tensor[(128), float32], %v_param_70: Tensor[(128), float32], %v_param_71: Tensor[(128), float32], %v_param_72: Tensor[(128), float32], %v_param_73: Tensor[(3, 3, 128, 128), float32], %v_param_74: Tensor[(128), float32], %v_param_75: Tensor[(128), float32], %v_param_76: Tensor[(128), float32], %v_param_77: Tensor[(128), float32], %v_param_78: Tensor[(128), float32], %v_param_81: Tensor[(1, 1, 128, 512), float32], %v_param_82: Tensor[(512), float32], %v_param_87: Tensor[(512), float32], %v_param_88: Tensor[(512), float32], %v_param_89: Tensor[(512), float32], %v_param_90: Tensor[(512), float32], %v_param_91: Tensor[(1, 1, 512, 128), float32], %v_param_92: Tensor[(128), float32], %v_param_93: Tensor[(128), float32], %v_param_94: Tensor[(128), float32], %v_param_95: Tensor[(128), float32], %v_param_96: Tensor[(128), float32], %v_param_97: Tensor[(3, 3, 128, 128), float32], %v_param_98: Tensor[(128), float32], %v_param_99: Tensor[(128), float32], %v_param_100: Tensor[(128), float32], %v_param_101: Tensor[(128), float32], %v_param_102: Tensor[(128), float32], %v_param_103: Tensor[(1, 1, 128, 512), float32], %v_param_104: Tensor[(512), float32], %v_param_105: Tensor[(512), float32], %v_param_106: Tensor[(512), float32], %v_param_107: Tensor[(512), float32], %v_param_108: Tensor[(512), float32], %v_param_109: Tensor[(1, 1, 512, 128), float32], %v_param_110: Tensor[(128), float32], %v_param_111: Tensor[(128), float32], %v_param_112: Tensor[(128), float32], %v_param_113: Tensor[(128), float32], %v_param_114: Tensor[(128), float32], %v_param_115: Tensor[(3, 3, 128, 128), float32], %v_param_116: Tensor[(128), float32], %v_param_117: Tensor[(128), float32], %v_param_118: Tensor[(128), float32], %v_param_119: Tensor[(128), float32], %v_param_120: Tensor[(128), float32], %v_param_121: Tensor[(1, 1, 128, 512), float32], %v_param_122: Tensor[(512), float32], %v_param_123: Tensor[(512), float32], %v_param_124: Tensor[(512), float32], %v_param_125: Tensor[(512), float32], %v_param_126: Tensor[(512), float32], %v_param_127: Tensor[(1, 1, 512, 128), float32], %v_param_128: Tensor[(128), float32], %v_param_129: Tensor[(128), float32], %v_param_130: Tensor[(128), float32], %v_param_131: Tensor[(128), float32], %v_param_132: Tensor[(128), float32], %v_param_133: Tensor[(3, 3, 128, 128), float32], %v_param_134: Tensor[(128), float32], %v_param_135: Tensor[(128), float32], %v_param_136: Tensor[(128), float32], %v_param_137: Tensor[(128), float32], %v_param_138: Tensor[(128), float32], %v_param_139: Tensor[(1, 1, 128, 512), float32], %v_param_140: Tensor[(512), float32], %v_param_141: Tensor[(512), float32], %v_param_142: Tensor[(512), float32], %v_param_143: Tensor[(512), float32], %v_param_144: Tensor[(512), float32], %v_param_157: Tensor[(1, 1, 512, 1024), float32], %v_param_158: Tensor[(1024), float32], %v_param_161: Tensor[(1024), float32], %v_param_162: Tensor[(1024), float32], %v_param_163: Tensor[(1024), float32], %v_param_164: Tensor[(1024), float32], %v_param_145: Tensor[(1, 1, 512, 256), float32], %v_param_146: Tensor[(256), float32], %v_param_147: Tensor[(256), float32], %v_param_148: Tensor[(256), float32], %v_param_149: Tensor[(256), float32], %v_param_150: Tensor[(256), float32], %v_param_151: Tensor[(3, 3, 256, 256), float32], %v_param_152: Tensor[(256), float32], %v_param_153: Tensor[(256), float32], %v_param_154: Tensor[(256), float32], %v_param_155: Tensor[(256), float32], %v_param_156: Tensor[(256), float32], %v_param_159: Tensor[(1, 1, 256, 1024), float32], %v_param_160: Tensor[(1024), float32], %v_param_165: Tensor[(1024), float32], %v_param_166: Tensor[(1024), float32], %v_param_167: Tensor[(1024), float32], %v_param_168: Tensor[(1024), float32], %v_param_169: Tensor[(1, 1, 1024, 256), float32], %v_param_170: Tensor[(256), float32], %v_param_171: Tensor[(256), float32], %v_param_172: Tensor[(256), float32], %v_param_173: Tensor[(256), float32], %v_param_174: Tensor[(256), float32], %v_param_175: Tensor[(3, 3, 256, 256), float32], %v_param_176: Tensor[(256), float32], %v_param_177: Tensor[(256), float32], %v_param_178: Tensor[(256), float32], %v_param_179: Tensor[(256), float32], %v_param_180: Tensor[(256), float32], %v_param_181: Tensor[(1, 1, 256, 1024), float32], %v_param_182: Tensor[(1024), float32], %v_param_183: Tensor[(1024), float32], %v_param_184: Tensor[(1024), float32], %v_param_185: Tensor[(1024), float32], %v_param_186: Tensor[(1024), float32], %v_param_187: Tensor[(1, 1, 1024, 256), float32], %v_param_188: Tensor[(256), float32], %v_param_189: Tensor[(256), float32], %v_param_190: Tensor[(256), float32], %v_param_191: Tensor[(256), float32], %v_param_192: Tensor[(256), float32], %v_param_193: Tensor[(3, 3, 256, 256), float32], %v_param_194: Tensor[(256), float32], %v_param_195: Tensor[(256), float32], %v_param_196: Tensor[(256), float32], %v_param_197: Tensor[(256), float32], %v_param_198: Tensor[(256), float32], %v_param_199: Tensor[(1, 1, 256, 1024), float32], %v_param_200: Tensor[(1024), float32], %v_param_201: Tensor[(1024), float32], %v_param_202: Tensor[(1024), float32], %v_param_203: Tensor[(1024), float32], %v_param_204: Tensor[(1024), float32], %v_param_205: Tensor[(1, 1, 1024, 256), float32], %v_param_206: Tensor[(256), float32], %v_param_207: Tensor[(256), float32], %v_param_208: Tensor[(256), float32], %v_param_209: Tensor[(256), float32], %v_param_210: Tensor[(256), float32], %v_param_211: Tensor[(3, 3, 256, 256), float32], %v_param_212: Tensor[(256), float32], %v_param_213: Tensor[(256), float32], %v_param_214: Tensor[(256), float32], %v_param_215: Tensor[(256), float32], %v_param_216: Tensor[(256), float32], %v_param_217: Tensor[(1, 1, 256, 1024), float32], %v_param_218: Tensor[(1024), float32], %v_param_219: Tensor[(1024), float32], %v_param_220: Tensor[(1024), float32], %v_param_221: Tensor[(1024), float32], %v_param_222: Tensor[(1024), float32], %v_param_223: Tensor[(1, 1, 1024, 256), float32], %v_param_224: Tensor[(256), float32], %v_param_225: Tensor[(256), float32], %v_param_226: Tensor[(256), float32], %v_param_227: Tensor[(256), float32], %v_param_228: Tensor[(256), float32], %v_param_229: Tensor[(3, 3, 256, 256), float32], %v_param_230: Tensor[(256), float32], %v_param_231: Tensor[(256), float32], %v_param_232: Tensor[(256), float32], %v_param_233: Tensor[(256), float32], %v_param_234: Tensor[(256), float32], %v_param_235: Tensor[(1, 1, 256, 1024), float32], %v_param_236: Tensor[(1024), float32], %v_param_237: Tensor[(1024), float32], %v_param_238: Tensor[(1024), float32], %v_param_239: Tensor[(1024), float32], %v_param_240: Tensor[(1024), float32], %v_param_241: Tensor[(1, 1, 1024, 256), float32], %v_param_242: Tensor[(256), float32], %v_param_243: Tensor[(256), float32], %v_param_244: Tensor[(256), float32], %v_param_245: Tensor[(256), float32], %v_param_246: Tensor[(256), float32], %v_param_247: Tensor[(3, 3, 256, 256), float32], %v_param_248: Tensor[(256), float32], %v_param_249: Tensor[(256), float32], %v_param_250: Tensor[(256), float32], %v_param_251: Tensor[(256), float32], %v_param_252: Tensor[(256), float32], %v_param_253: Tensor[(1, 1, 256, 1024), float32], %v_param_254: Tensor[(1024), float32], %v_param_255: Tensor[(1024), float32], %v_param_256: Tensor[(1024), float32], %v_param_257: Tensor[(1024), float32], %v_param_258: Tensor[(1024), float32], %v_param_271: Tensor[(1, 1, 1024, 2048), float32], %v_param_272: Tensor[(2048), float32], %v_param_275: Tensor[(2048), float32], %v_param_276: Tensor[(2048), float32], %v_param_277: Tensor[(2048), float32], %v_param_278: Tensor[(2048), float32], %v_param_259: Tensor[(1, 1, 1024, 512), float32], %v_param_260: Tensor[(512), float32], %v_param_261: Tensor[(512), float32], %v_param_262: Tensor[(512), float32], %v_param_263: Tensor[(512), float32], %v_param_264: Tensor[(512), float32], %v_param_265: Tensor[(3, 3, 512, 512), float32], %v_param_266: Tensor[(512), float32], %v_param_267: Tensor[(512), float32], %v_param_268: Tensor[(512), float32], %v_param_269: Tensor[(512), float32], %v_param_270: Tensor[(512), float32], %v_param_273: Tensor[(1, 1, 512, 2048), float32], %v_param_274: Tensor[(2048), float32], %v_param_279: Tensor[(2048), float32], %v_param_280: Tensor[(2048), float32], %v_param_281: Tensor[(2048), float32], %v_param_282: Tensor[(2048), float32], %v_param_283: Tensor[(1, 1, 2048, 512), float32], %v_param_284: Tensor[(512), float32], %v_param_285: Tensor[(512), float32], %v_param_286: Tensor[(512), float32], %v_param_287: Tensor[(512), float32], %v_param_288: Tensor[(512), float32], %v_param_289: Tensor[(3, 3, 512, 512), float32], %v_param_290: Tensor[(512), float32], %v_param_291: Tensor[(512), float32], %v_param_292: Tensor[(512), float32], %v_param_293: Tensor[(512), float32], %v_param_294: Tensor[(512), float32], %v_param_295: Tensor[(1, 1, 512, 2048), float32], %v_param_296: Tensor[(2048), float32], %v_param_297: Tensor[(2048), float32], %v_param_298: Tensor[(2048), float32], %v_param_299: Tensor[(2048), float32], %v_param_300: Tensor[(2048), float32], %v_param_301: Tensor[(1, 1, 2048, 512), float32], %v_param_302: Tensor[(512), float32], %v_param_303: Tensor[(512), float32], %v_param_304: Tensor[(512), float32], %v_param_305: Tensor[(512), float32], %v_param_306: Tensor[(512), float32], %v_param_307: Tensor[(3, 3, 512, 512), float32], %v_param_308: Tensor[(512), float32], %v_param_309: Tensor[(512), float32], %v_param_310: Tensor[(512), float32], %v_param_311: Tensor[(512), float32], %v_param_312: Tensor[(512), float32], %v_param_313: Tensor[(1, 1, 512, 2048), float32], %v_param_314: Tensor[(2048), float32], %v_param_315: Tensor[(2048), float32], %v_param_316: Tensor[(2048), float32], %v_param_317: Tensor[(2048), float32], %v_param_318: Tensor[(2048), float32], %v_param_319: Tensor[(1000, 2048), float32], %v_param_320: Tensor[(1000), float32]) {
  %0 = nn.pad(%input_2, 0, pad_width=[[0, 0], [3, 3], [3, 3], [0, 0]]);
  %1 = nn.conv2d(%0, %v_param_1, strides=[2, 2], padding=[0, 0, 0, 0], channels=64, kernel_size=[7, 7], data_layout="NHWC", kernel_layout="HWIO");
  %2 = nn.bias_add(%1, %v_param_2, axis=-1);
  %3 = nn.batch_norm(%2, %v_param_3, %v_param_4, %v_param_5, %v_param_6, axis=3, epsilon=1.001e-05f);
  %4 = %3.0;
  %5 = nn.relu(%4);
  %6 = nn.pad(%5, 0, pad_width=[[0, 0], [1, 1], [1, 1], [0, 0]]);
  %7 = nn.max_pool2d(%6, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0], layout="NHWC");
  %8 = nn.conv2d(%7, %v_param_19, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %9 = nn.bias_add(%8, %v_param_20, axis=-1);
  %10 = nn.batch_norm(%9, %v_param_23, %v_param_24, %v_param_25, %v_param_26, axis=3, epsilon=1.001e-05f);
  %11 = nn.conv2d(%7, %v_param_7, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %12 = nn.bias_add(%11, %v_param_8, axis=-1);
  %13 = nn.batch_norm(%12, %v_param_9, %v_param_10, %v_param_11, %v_param_12, axis=3, epsilon=1.001e-05f);
  %14 = %13.0;
  %15 = nn.relu(%14);
  %16 = nn.conv2d(%15, %v_param_13, padding=[1i64, 1i64, 1i64, 1i64], channels=64, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %17 = nn.bias_add(%16, %v_param_14, axis=-1);
  %18 = nn.batch_norm(%17, %v_param_15, %v_param_16, %v_param_17, %v_param_18, axis=3, epsilon=1.001e-05f);
  %19 = %18.0;
  %20 = nn.relu(%19);
  %21 = nn.conv2d(%20, %v_param_21, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %22 = nn.bias_add(%21, %v_param_22, axis=-1);
  %23 = nn.batch_norm(%22, %v_param_27, %v_param_28, %v_param_29, %v_param_30, axis=3, epsilon=1.001e-05f);
  %24 = %10.0;
  %25 = %23.0;
  %26 = add(%24, %25);
  %27 = nn.relu(%26);
  %28 = nn.conv2d(%27, %v_param_31, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %29 = nn.bias_add(%28, %v_param_32, axis=-1);
  %30 = nn.batch_norm(%29, %v_param_33, %v_param_34, %v_param_35, %v_param_36, axis=3, epsilon=1.001e-05f);
  %31 = %30.0;
  %32 = nn.relu(%31);
  %33 = nn.conv2d(%32, %v_param_37, padding=[1i64, 1i64, 1i64, 1i64], channels=64, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %34 = nn.bias_add(%33, %v_param_38, axis=-1);
  %35 = nn.batch_norm(%34, %v_param_39, %v_param_40, %v_param_41, %v_param_42, axis=3, epsilon=1.001e-05f);
  %36 = %35.0;
  %37 = nn.relu(%36);
  %38 = nn.conv2d(%37, %v_param_43, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %39 = nn.bias_add(%38, %v_param_44, axis=-1);
  %40 = nn.batch_norm(%39, %v_param_45, %v_param_46, %v_param_47, %v_param_48, axis=3, epsilon=1.001e-05f);
  %41 = %40.0;
  %42 = add(%27, %41);
  %43 = nn.relu(%42);
  %44 = nn.conv2d(%43, %v_param_49, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %45 = nn.bias_add(%44, %v_param_50, axis=-1);
  %46 = nn.batch_norm(%45, %v_param_51, %v_param_52, %v_param_53, %v_param_54, axis=3, epsilon=1.001e-05f);
  %47 = %46.0;
  %48 = nn.relu(%47);
  %49 = nn.conv2d(%48, %v_param_55, padding=[1i64, 1i64, 1i64, 1i64], channels=64, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %50 = nn.bias_add(%49, %v_param_56, axis=-1);
  %51 = nn.batch_norm(%50, %v_param_57, %v_param_58, %v_param_59, %v_param_60, axis=3, epsilon=1.001e-05f);
  %52 = %51.0;
  %53 = nn.relu(%52);
  %54 = nn.conv2d(%53, %v_param_61, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %55 = nn.bias_add(%54, %v_param_62, axis=-1);
  %56 = nn.batch_norm(%55, %v_param_63, %v_param_64, %v_param_65, %v_param_66, axis=3, epsilon=1.001e-05f);
  %57 = %56.0;
  %58 = add(%43, %57);
  %59 = nn.relu(%58);
  %60 = nn.conv2d(%59, %v_param_79, strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %61 = nn.bias_add(%60, %v_param_80, axis=-1);
  %62 = nn.batch_norm(%61, %v_param_83, %v_param_84, %v_param_85, %v_param_86, axis=3, epsilon=1.001e-05f);
  %63 = nn.conv2d(%59, %v_param_67, strides=[2, 2], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %64 = nn.bias_add(%63, %v_param_68, axis=-1);
  %65 = nn.batch_norm(%64, %v_param_69, %v_param_70, %v_param_71, %v_param_72, axis=3, epsilon=1.001e-05f);
  %66 = %65.0;
  %67 = nn.relu(%66);
  %68 = nn.conv2d(%67, %v_param_73, padding=[1i64, 1i64, 1i64, 1i64], channels=128, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %69 = nn.bias_add(%68, %v_param_74, axis=-1);
  %70 = nn.batch_norm(%69, %v_param_75, %v_param_76, %v_param_77, %v_param_78, axis=3, epsilon=1.001e-05f);
  %71 = %70.0;
  %72 = nn.relu(%71);
  %73 = nn.conv2d(%72, %v_param_81, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %74 = nn.bias_add(%73, %v_param_82, axis=-1);
  %75 = nn.batch_norm(%74, %v_param_87, %v_param_88, %v_param_89, %v_param_90, axis=3, epsilon=1.001e-05f);
  %76 = %62.0;
  %77 = %75.0;
  %78 = add(%76, %77);
  %79 = nn.relu(%78);
  %80 = nn.conv2d(%79, %v_param_91, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %81 = nn.bias_add(%80, %v_param_92, axis=-1);
  %82 = nn.batch_norm(%81, %v_param_93, %v_param_94, %v_param_95, %v_param_96, axis=3, epsilon=1.001e-05f);
  %83 = %82.0;
  %84 = nn.relu(%83);
  %85 = nn.conv2d(%84, %v_param_97, padding=[1i64, 1i64, 1i64, 1i64], channels=128, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %86 = nn.bias_add(%85, %v_param_98, axis=-1);
  %87 = nn.batch_norm(%86, %v_param_99, %v_param_100, %v_param_101, %v_param_102, axis=3, epsilon=1.001e-05f);
  %88 = %87.0;
  %89 = nn.relu(%88);
  %90 = nn.conv2d(%89, %v_param_103, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %91 = nn.bias_add(%90, %v_param_104, axis=-1);
  %92 = nn.batch_norm(%91, %v_param_105, %v_param_106, %v_param_107, %v_param_108, axis=3, epsilon=1.001e-05f);
  %93 = %92.0;
  %94 = add(%79, %93);
  %95 = nn.relu(%94);
  %96 = nn.conv2d(%95, %v_param_109, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %97 = nn.bias_add(%96, %v_param_110, axis=-1);
  %98 = nn.batch_norm(%97, %v_param_111, %v_param_112, %v_param_113, %v_param_114, axis=3, epsilon=1.001e-05f);
  %99 = %98.0;
  %100 = nn.relu(%99);
  %101 = nn.conv2d(%100, %v_param_115, padding=[1i64, 1i64, 1i64, 1i64], channels=128, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %102 = nn.bias_add(%101, %v_param_116, axis=-1);
  %103 = nn.batch_norm(%102, %v_param_117, %v_param_118, %v_param_119, %v_param_120, axis=3, epsilon=1.001e-05f);
  %104 = %103.0;
  %105 = nn.relu(%104);
  %106 = nn.conv2d(%105, %v_param_121, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %107 = nn.bias_add(%106, %v_param_122, axis=-1);
  %108 = nn.batch_norm(%107, %v_param_123, %v_param_124, %v_param_125, %v_param_126, axis=3, epsilon=1.001e-05f);
  %109 = %108.0;
  %110 = add(%95, %109);
  %111 = nn.relu(%110);
  %112 = nn.conv2d(%111, %v_param_127, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %113 = nn.bias_add(%112, %v_param_128, axis=-1);
  %114 = nn.batch_norm(%113, %v_param_129, %v_param_130, %v_param_131, %v_param_132, axis=3, epsilon=1.001e-05f);
  %115 = %114.0;
  %116 = nn.relu(%115);
  %117 = nn.conv2d(%116, %v_param_133, padding=[1i64, 1i64, 1i64, 1i64], channels=128, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %118 = nn.bias_add(%117, %v_param_134, axis=-1);
  %119 = nn.batch_norm(%118, %v_param_135, %v_param_136, %v_param_137, %v_param_138, axis=3, epsilon=1.001e-05f);
  %120 = %119.0;
  %121 = nn.relu(%120);
  %122 = nn.conv2d(%121, %v_param_139, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %123 = nn.bias_add(%122, %v_param_140, axis=-1);
  %124 = nn.batch_norm(%123, %v_param_141, %v_param_142, %v_param_143, %v_param_144, axis=3, epsilon=1.001e-05f);
  %125 = %124.0;
  %126 = add(%111, %125);
  %127 = nn.relu(%126);
  %128 = nn.conv2d(%127, %v_param_157, strides=[2, 2], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %129 = nn.bias_add(%128, %v_param_158, axis=-1);
  %130 = nn.batch_norm(%129, %v_param_161, %v_param_162, %v_param_163, %v_param_164, axis=3, epsilon=1.001e-05f);
  %131 = nn.conv2d(%127, %v_param_145, strides=[2, 2], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %132 = nn.bias_add(%131, %v_param_146, axis=-1);
  %133 = nn.batch_norm(%132, %v_param_147, %v_param_148, %v_param_149, %v_param_150, axis=3, epsilon=1.001e-05f);
  %134 = %133.0;
  %135 = nn.relu(%134);
  %136 = nn.conv2d(%135, %v_param_151, padding=[1i64, 1i64, 1i64, 1i64], channels=256, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %137 = nn.bias_add(%136, %v_param_152, axis=-1);
  %138 = nn.batch_norm(%137, %v_param_153, %v_param_154, %v_param_155, %v_param_156, axis=3, epsilon=1.001e-05f);
  %139 = %138.0;
  %140 = nn.relu(%139);
  %141 = nn.conv2d(%140, %v_param_159, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %142 = nn.bias_add(%141, %v_param_160, axis=-1);
  %143 = nn.batch_norm(%142, %v_param_165, %v_param_166, %v_param_167, %v_param_168, axis=3, epsilon=1.001e-05f);
  %144 = %130.0;
  %145 = %143.0;
  %146 = add(%144, %145);
  %147 = nn.relu(%146);
  %148 = nn.conv2d(%147, %v_param_169, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %149 = nn.bias_add(%148, %v_param_170, axis=-1);
  %150 = nn.batch_norm(%149, %v_param_171, %v_param_172, %v_param_173, %v_param_174, axis=3, epsilon=1.001e-05f);
  %151 = %150.0;
  %152 = nn.relu(%151);
  %153 = nn.conv2d(%152, %v_param_175, padding=[1i64, 1i64, 1i64, 1i64], channels=256, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %154 = nn.bias_add(%153, %v_param_176, axis=-1);
  %155 = nn.batch_norm(%154, %v_param_177, %v_param_178, %v_param_179, %v_param_180, axis=3, epsilon=1.001e-05f);
  %156 = %155.0;
  %157 = nn.relu(%156);
  %158 = nn.conv2d(%157, %v_param_181, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %159 = nn.bias_add(%158, %v_param_182, axis=-1);
  %160 = nn.batch_norm(%159, %v_param_183, %v_param_184, %v_param_185, %v_param_186, axis=3, epsilon=1.001e-05f);
  %161 = %160.0;
  %162 = add(%147, %161);
  %163 = nn.relu(%162);
  %164 = nn.conv2d(%163, %v_param_187, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %165 = nn.bias_add(%164, %v_param_188, axis=-1);
  %166 = nn.batch_norm(%165, %v_param_189, %v_param_190, %v_param_191, %v_param_192, axis=3, epsilon=1.001e-05f);
  %167 = %166.0;
  %168 = nn.relu(%167);
  %169 = nn.conv2d(%168, %v_param_193, padding=[1i64, 1i64, 1i64, 1i64], channels=256, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %170 = nn.bias_add(%169, %v_param_194, axis=-1);
  %171 = nn.batch_norm(%170, %v_param_195, %v_param_196, %v_param_197, %v_param_198, axis=3, epsilon=1.001e-05f);
  %172 = %171.0;
  %173 = nn.relu(%172);
  %174 = nn.conv2d(%173, %v_param_199, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %175 = nn.bias_add(%174, %v_param_200, axis=-1);
  %176 = nn.batch_norm(%175, %v_param_201, %v_param_202, %v_param_203, %v_param_204, axis=3, epsilon=1.001e-05f);
  %177 = %176.0;
  %178 = add(%163, %177);
  %179 = nn.relu(%178);
  %180 = nn.conv2d(%179, %v_param_205, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %181 = nn.bias_add(%180, %v_param_206, axis=-1);
  %182 = nn.batch_norm(%181, %v_param_207, %v_param_208, %v_param_209, %v_param_210, axis=3, epsilon=1.001e-05f);
  %183 = %182.0;
  %184 = nn.relu(%183);
  %185 = nn.conv2d(%184, %v_param_211, padding=[1i64, 1i64, 1i64, 1i64], channels=256, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %186 = nn.bias_add(%185, %v_param_212, axis=-1);
  %187 = nn.batch_norm(%186, %v_param_213, %v_param_214, %v_param_215, %v_param_216, axis=3, epsilon=1.001e-05f);
  %188 = %187.0;
  %189 = nn.relu(%188);
  %190 = nn.conv2d(%189, %v_param_217, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %191 = nn.bias_add(%190, %v_param_218, axis=-1);
  %192 = nn.batch_norm(%191, %v_param_219, %v_param_220, %v_param_221, %v_param_222, axis=3, epsilon=1.001e-05f);
  %193 = %192.0;
  %194 = add(%179, %193);
  %195 = nn.relu(%194);
  %196 = nn.conv2d(%195, %v_param_223, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %197 = nn.bias_add(%196, %v_param_224, axis=-1);
  %198 = nn.batch_norm(%197, %v_param_225, %v_param_226, %v_param_227, %v_param_228, axis=3, epsilon=1.001e-05f);
  %199 = %198.0;
  %200 = nn.relu(%199);
  %201 = nn.conv2d(%200, %v_param_229, padding=[1i64, 1i64, 1i64, 1i64], channels=256, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %202 = nn.bias_add(%201, %v_param_230, axis=-1);
  %203 = nn.batch_norm(%202, %v_param_231, %v_param_232, %v_param_233, %v_param_234, axis=3, epsilon=1.001e-05f);
  %204 = %203.0;
  %205 = nn.relu(%204);
  %206 = nn.conv2d(%205, %v_param_235, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %207 = nn.bias_add(%206, %v_param_236, axis=-1);
  %208 = nn.batch_norm(%207, %v_param_237, %v_param_238, %v_param_239, %v_param_240, axis=3, epsilon=1.001e-05f);
  %209 = %208.0;
  %210 = add(%195, %209);
  %211 = nn.relu(%210);
  %212 = nn.conv2d(%211, %v_param_241, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %213 = nn.bias_add(%212, %v_param_242, axis=-1);
  %214 = nn.batch_norm(%213, %v_param_243, %v_param_244, %v_param_245, %v_param_246, axis=3, epsilon=1.001e-05f);
  %215 = %214.0;
  %216 = nn.relu(%215);
  %217 = nn.conv2d(%216, %v_param_247, padding=[1i64, 1i64, 1i64, 1i64], channels=256, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %218 = nn.bias_add(%217, %v_param_248, axis=-1);
  %219 = nn.batch_norm(%218, %v_param_249, %v_param_250, %v_param_251, %v_param_252, axis=3, epsilon=1.001e-05f);
  %220 = %219.0;
  %221 = nn.relu(%220);
  %222 = nn.conv2d(%221, %v_param_253, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %223 = nn.bias_add(%222, %v_param_254, axis=-1);
  %224 = nn.batch_norm(%223, %v_param_255, %v_param_256, %v_param_257, %v_param_258, axis=3, epsilon=1.001e-05f);
  %225 = %224.0;
  %226 = add(%211, %225);
  %227 = nn.relu(%226);
  %228 = nn.conv2d(%227, %v_param_271, strides=[2, 2], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %229 = nn.bias_add(%228, %v_param_272, axis=-1);
  %230 = nn.batch_norm(%229, %v_param_275, %v_param_276, %v_param_277, %v_param_278, axis=3, epsilon=1.001e-05f);
  %231 = nn.conv2d(%227, %v_param_259, strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %232 = nn.bias_add(%231, %v_param_260, axis=-1);
  %233 = nn.batch_norm(%232, %v_param_261, %v_param_262, %v_param_263, %v_param_264, axis=3, epsilon=1.001e-05f);
  %234 = %233.0;
  %235 = nn.relu(%234);
  %236 = nn.conv2d(%235, %v_param_265, padding=[1i64, 1i64, 1i64, 1i64], channels=512, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %237 = nn.bias_add(%236, %v_param_266, axis=-1);
  %238 = nn.batch_norm(%237, %v_param_267, %v_param_268, %v_param_269, %v_param_270, axis=3, epsilon=1.001e-05f);
  %239 = %238.0;
  %240 = nn.relu(%239);
  %241 = nn.conv2d(%240, %v_param_273, padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %242 = nn.bias_add(%241, %v_param_274, axis=-1);
  %243 = nn.batch_norm(%242, %v_param_279, %v_param_280, %v_param_281, %v_param_282, axis=3, epsilon=1.001e-05f);
  %244 = %230.0;
  %245 = %243.0;
  %246 = add(%244, %245);
  %247 = nn.relu(%246);
  %248 = nn.conv2d(%247, %v_param_283, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %249 = nn.bias_add(%248, %v_param_284, axis=-1);
  %250 = nn.batch_norm(%249, %v_param_285, %v_param_286, %v_param_287, %v_param_288, axis=3, epsilon=1.001e-05f);
  %251 = %250.0;
  %252 = nn.relu(%251);
  %253 = nn.conv2d(%252, %v_param_289, padding=[1i64, 1i64, 1i64, 1i64], channels=512, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %254 = nn.bias_add(%253, %v_param_290, axis=-1);
  %255 = nn.batch_norm(%254, %v_param_291, %v_param_292, %v_param_293, %v_param_294, axis=3, epsilon=1.001e-05f);
  %256 = %255.0;
  %257 = nn.relu(%256);
  %258 = nn.conv2d(%257, %v_param_295, padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %259 = nn.bias_add(%258, %v_param_296, axis=-1);
  %260 = nn.batch_norm(%259, %v_param_297, %v_param_298, %v_param_299, %v_param_300, axis=3, epsilon=1.001e-05f);
  %261 = %260.0;
  %262 = add(%247, %261);
  %263 = nn.relu(%262);
  %264 = nn.conv2d(%263, %v_param_301, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %265 = nn.bias_add(%264, %v_param_302, axis=-1);
  %266 = nn.batch_norm(%265, %v_param_303, %v_param_304, %v_param_305, %v_param_306, axis=3, epsilon=1.001e-05f);
  %267 = %266.0;
  %268 = nn.relu(%267);
  %269 = nn.conv2d(%268, %v_param_307, padding=[1i64, 1i64, 1i64, 1i64], channels=512, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
  %270 = nn.bias_add(%269, %v_param_308, axis=-1);
  %271 = nn.batch_norm(%270, %v_param_309, %v_param_310, %v_param_311, %v_param_312, axis=3, epsilon=1.001e-05f);
  %272 = %271.0;
  %273 = nn.relu(%272);
  %274 = nn.conv2d(%273, %v_param_313, padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="HWIO");
  %275 = nn.bias_add(%274, %v_param_314, axis=-1);
  %276 = nn.batch_norm(%275, %v_param_315, %v_param_316, %v_param_317, %v_param_318, axis=3, epsilon=1.001e-05f);
  %277 = %276.0;
  %278 = add(%263, %277);
  %279 = nn.relu(%278);
  %280 = nn.global_avg_pool2d(%279, layout="NHWC");
  %281 = nn.batch_flatten(%280);
  %282 = nn.dense(%281, %v_param_319, units=1000);
  %283 = nn.bias_add(%282, %v_param_320);
  nn.softmax(%283)
}
```

# 自动调优
现在，可以使用下面的 api 为任何目标对模型进行自动调优。调整需要 RPC 设置，请参阅[在 Adreno GPU 上部署](https://tvm.hyper.ai/docs/how_to/deploy/deploy_adreno)

```python
rpc_tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
rpc_tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
rpc_key = "android"
rpc_tracker = rpc_tracker_host + ":" + str(rpc_tracker_port)

# 自动调整是计算密集型和耗时的任务。
# 它在上述配置中被设置为 False，因为此脚本在 x86 上运行以进行演示。
# 请将 :code:`is_tuning` 设置为 True 以启用自动调整。

# 此外，:code:`test_target` 设置为 :code:`llvm`，因为此示例以使其与 x86 演示兼容。
# 请在上述配置中将其更改为 :code:`opencl` 或 :code:`opencl -device=adreno` 以用于 RPC 目标。

if is_tuning:
    tvmc.tune(
        tvmc_model,
        target=target,
        tuning_records=tune_log,
        target_host=target_host,
        hostname=rpc_tracker_host,
        port=rpc_tracker_port,
        rpc_key=rpc_key,
        tuner="xgb",
        repeat=30,
        trials=3,
        early_stopping=0,
    )
```

# 编译
编译以生成 tvm 产品

```python
# 此生成的示例在我们的 x86 服务器上运行以进行演示。
# 要在真实目标上的 RPC 上部署和调优，请在上述配置部分将 :code:`local_demo` 设置为 False。

# OpenCLML 卸载将尝试通过使用 OpenCLML 专有运算符库加速受支持的运算符。
# 默认情况下，在上述配置部分 :code:`enable_clml` 设置为 False。

if not enable_clml:
    if local_demo:
        tvmc_package = tvmc.compile(
            tvmc_model,
            target=target,
        )
    else:
        tvmc_package = tvmc.compile(
            tvmc_model,
            target=target,
            target_host=target_host,
            cross=cross_compiler,
            tuning_records=tune_log,
        )
else:
    # 或者，我们可以保存编译输出并将其保存为 TVMCPackage。
    # 这种方式避免了再次编译时加载编译的模块。
    target = target + ", clml"
    pkg_path = tmp_path.relpath("keras-resnet50.tar")
    tvmc.compile(
        tvmc_model,
        target=target,
        target_host=target_host,
        cross=cross_compiler,
        tuning_records=tune_log,
        package_path=pkg_path,
    )

    # 加载已编译的包
    tvmc_package = TVMCPackage(package_path=pkg_path)

# tvmc_package 包括 tvmc_package.lib_path, tvmc_package.graph, tvmc_package.params
# 已保存的 TVMPackage 实际上是 mod.so、mod.json 和 mod.params的 tar 存档。
```

# 部署和运行
通过让 tvmc 使用随机数据填充输入在 RPC 上部署和运行已编译的模型。

```python
# 在 RPC 设置上运行
if local_demo:
    result = tvmc.run(tvmc_package, device="cpu", fill_mode="random")
else:
    result = tvmc.run(
        tvmc_package,
        device="cl",
        rpc_key=rpc_key,
        hostname=rpc_tracker_host,
        port=rpc_tracker_port,
        fill_mode="random",
    )

# result 是输出的字典。
print("Result:", result)
```

Out:
```python
Result: []
Output Names:
 ['output_0']
```