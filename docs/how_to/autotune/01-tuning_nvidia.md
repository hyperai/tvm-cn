---
title: 在 NVIDIA GPU 上调优高性能卷积
---

# 在 NVIDIA GPU 上调优高性能卷积

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_conv2d_cuda.html#sphx-glr-download-how-to-tune-with-autotvm-tune-conv2d-cuda-py) 下载完整的示例代码
:::

**作者**：[Lianmin Zheng](https://github.com/merrymercy)

本教程介绍如何为 NVIDIA GPU 编写高性能可调模板。通过在此模板上运行自动调优器，可在许多情况下胜过供应商提供的 cuDNN 库。

注意，本教程不会在 Windows 或最新版本的 macOS 上运行。如需运行，请将本教程的主体放在 `if __name__ == "__main__":` 代码块中。

## 安装依赖

要在 TVM 中使用 autotvm 包，需要安装额外的依赖（如果用的是 Python2，请将「3」更改为「2」）：

``` bash
pip3 install --user psutil xgboost tornado cloudpickle
```

为了让 TVM 在调优中运行更快，推荐使用 Cython 作为 TVM 的 FFI。在 TVM 的根目录下，执行如下命令：

``` bash
pip3 install --user cython
sudo make cython3
```

在 Python 代码中导入包：

``` python
import logging
import sys
import numpy as np

import tvm
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing

from tvm import autotvm
```

## 第 1 步：定义搜索空间

TVM 中有很多有用的调度原语，详细教程，例如（1）[如何在 GPU 上优化卷积](/docs/how_to/optimize/gpu_conv)（2）[在 NVIDIA GPU 上优化 DepthwiseConv](https://tvm.apache.org/2017/08/22/Optimize-Deep-Learning-GPU-Operators-with-TVM-A-Depthwise-Convolution-Example)。

但是，它们的实现是针对一些特殊的输入 shape 手动调整的。本节将构建足够大的空间，涵盖这些教程中使用的技术，然后依靠高效的自动调优器，对空间进行搜索并选择合适的配置。

熟悉 CUDA schedule 的开发者，对以下通用模板并不陌生。可以修改此模板以调整其他算子，例如深度卷积和 GEMM。要完全理解这个模板，需要熟悉调度原语和自动调优 API，可以参考上面的教程和 [AutoTVM 教程](/docs/tutorial/ops_AutoTVM)。

需要注意 conv2d 算子的搜索空间可能非常大（某些输入 shape 为 10^9 级别）

``` python
@autotvm.template("tutorial/conv2d_no_batching")
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    s = te.create_schedule([conv.op])

    #### 空间定义开始 ####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    #### 空间定义结束 ####

    # 内联填充
    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, "local")

    # 创建 cache 阶段
    AA = s.cache_read(data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])
    AL = s.cache_read(AA, "local", [OL])
    WL = s.cache_read(WW, "local", [OL])

    # 平铺和绑定空间轴
    n, f, y, x = s[output].op.axis
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
    kernel_scope = n  # 这是在此内核中附加全局配置的范围

    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction 轴
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, rym, ryi = cfg["tile_rx"].apply(s, OL, ry)
    rxo, rxm, rxi = cfg["tile_ry"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

    # 协作获取
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # 调优 unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    return s, [raw_data, kernel, conv]
```

## 第 2 步：在空间中搜索

选择 resnet 上的最后一层作为测试用例。由于空间足够大，所以 `XGBoostTuner` 最适合。这里只做 20 次试验来演示。实际上试验 1000 次可以为这个模板找到更合适的内核。

``` python
# logging 配置（用于将调优日志打印到屏幕）
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

# resnet 中的最后一层
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
task = autotvm.task.create(
    "tutorial/conv2d_no_batching", args=(N, H, W, CO, CI, KH, KW, strides, padding), target="cuda"
)
print(task.config_space)

# 使用本地 gpu，为每个配置测量 10 次以减少方差
# 编译程序超时 10 秒，运行超时 4 秒
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
)

# 开始调优，将 log 记录到 `conv2d.log`
# 在调优过程中，会尝试很多无效的配置，所以你应该
# 查看许多错误报告。只要能看到非零的 GFLOPS 就可以。
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(
    n_trial=20,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("conv2d.log")],
)
```

输出结果：

``` bash
ConfigSpace (len=10454400, space_map=
   0 tile_f: Split(policy=factors, product=512, num_outputs=4) len=220
   1 tile_y: Split(policy=factors, product=7, num_outputs=4) len=4
   2 tile_x: Split(policy=factors, product=7, num_outputs=4) len=4
   3 tile_rc: Split(policy=factors, product=512, num_outputs=3) len=55
   4 tile_ry: Split(policy=factors, product=3, num_outputs=3) len=3
   5 tile_rx: Split(policy=factors, product=3, num_outputs=3) len=3
   6 auto_unroll_max_step: OtherOption([0, 512, 1500]) len=3
   7 unroll_explicit: OtherOption([0, 1]) len=2
)
waiting for device...
device available
Get devices for measurement successfully!
No: 1   GFLOPS: 0.00/0.00       result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 32, 8, 2]), ('tile_y', [-1, 1, 1, 1]), ('tile_x', [-1, 7, 1, 1]), ('tile_rc', [-1, 8, 64]), ('tile_ry', [-1, 3, 1]), ('tile_rx', [-1, 3, 1]), ('auto_unroll_max_step', 0), ('unroll_explicit', 1)],None,6171524
No: 2   GFLOPS: 0.00/0.00       result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 128, 1, 4]), ('tile_y', [-1, 1, 1, 1]), ('tile_x', [-1, 1, 1, 1]), ('tile_rc', [-1, 4, 128]), ('tile_ry', [-1, 1, 1]), ('tile_rx', [-1, 3, 1]), ('auto_unroll_max_step', 512), ('unroll_explicit', 0)],None,2502827
No: 3   GFLOPS: 0.00/0.00       result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 4, 1, 32]), ('tile_y', [-1, 1, 1, 1]), ('tile_x', [-1, 1, 1, 7]), ('tile_rc', [-1, 512, 1]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 512), ('unroll_explicit', 0)],None,3325707
No: 4   GFLOPS: 0.00/0.00       result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 8, 4, 2]), ('tile_y', [-1, 1, 1, 7]), ('tile_x', [-1, 1, 1, 1]), ('tile_rc', [-1, 4, 8]), ('tile_ry', [-1, 3, 1]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 0)],None,4942815
No: 5   GFLOPS: 0.00/0.00       result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 4, 1, 128]), ('tile_y', [-1, 1, 1, 7]), ('tile_x', [-1, 7, 1, 1]), ('tile_rc', [-1, 2, 64]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 0)],None,5197272
No: 6   GFLOPS: 0.00/0.00       result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 32, 2, 4]), ('tile_y', [-1, 1, 1, 1]), ('tile_x', [-1, 1, 7, 1]), ('tile_rc', [-1, 8, 8]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 1]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 0)],None,3979473
No: 7   GFLOPS: 0.00/0.00       result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 8, 4, 8]), ('tile_y', [-1, 1, 7, 1]), ('tile_x', [-1, 1, 1, 1]), ('tile_rc', [-1, 4, 32]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 512), ('unroll_explicit', 0)],None,3439632
No: 8   GFLOPS: 0.00/0.00       result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 142, in build
    res = future.result()
  File "/usr/lib/python3.7/concurrent/futures/_base.py", line 435, in result
    return self.__get_result()
  File "/usr/lib/python3.7/concurrent/futures/_base.py", line 384, in __get_result
    raise self._exception
  File "/usr/lib/python3.7/concurrent/futures/thread.py", line 57, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/workspace/python/tvm/contrib/popen_pool.py", line 404, in <lambda>
    worker = lambda *args: self._worker_run(*args)
  File "/workspace/python/tvm/contrib/popen_pool.py", line 373, in _worker_run
    return proc.recv()
  File "/workspace/python/tvm/contrib/popen_pool.py", line 297, in recv
    raise TimeoutError()
TimeoutError

        [('tile_f', [-1, 2, 1, 64]), ('tile_y', [-1, 1, 1, 7]), ('tile_x', [-1, 1, 7, 1]), ('tile_rc', [-1, 1, 4]), ('tile_ry', [-1, 3, 1]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 0)],None,4909501
No: 9   GFLOPS: 174.65/174.65   result: MeasureResult(costs=(0.0013254985555555556,), error_no=MeasureErrorNo.NO_ERROR, all_cost=2.0453152656555176, timestamp=1658801307.741073)       [('tile_f', [-1, 1, 4, 8]), ('tile_y', [-1, 7, 1, 1]), ('tile_x', [-1, 1, 1, 1]), ('tile_rc', [-1, 2, 2]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 0)],None,5072689
No: 10  GFLOPS: 0.00/174.65     result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 4, 4, 8]), ('tile_y', [-1, 1, 1, 1]), ('tile_x', [-1, 1, 1, 7]), ('tile_rc', [-1, 64, 2]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 0)],None,5092711
No: 11  GFLOPS: 260.18/260.18   result: MeasureResult(costs=(0.0008897650828729283,), error_no=MeasureErrorNo.NO_ERROR, all_cost=1.7439014911651611, timestamp=1658801308.6655772)      [('tile_f', [-1, 8, 2, 1]), ('tile_y', [-1, 7, 1, 1]), ('tile_x', [-1, 1, 7, 1]), ('tile_rc', [-1, 2, 1]), ('tile_ry', [-1, 3, 1]), ('tile_rx', [-1, 3, 1]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 0)],None,4264713
No: 12  GFLOPS: 0.00/260.18     result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 128, 1, 2]), ('tile_y', [-1, 1, 7, 1]), ('tile_x', [-1, 1, 1, 1]), ('tile_rc', [-1, 1, 256]), ('tile_ry', [-1, 1, 1]), ('tile_rx', [-1, 1, 1]), ('auto_unroll_max_step', 0), ('unroll_explicit', 0)],None,183542
No: 13  GFLOPS: 0.00/260.18     result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 4, 8, 8]), ('tile_y', [-1, 1, 7, 1]), ('tile_x', [-1, 1, 1, 1]), ('tile_rc', [-1, 1, 64]), ('tile_ry', [-1, 1, 1]), ('tile_rx', [-1, 3, 1]), ('auto_unroll_max_step', 512), ('unroll_explicit', 0)],None,2482196
No: 14  GFLOPS: 0.00/260.18     result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 64, 1, 4]), ('tile_y', [-1, 1, 7, 1]), ('tile_x', [-1, 1, 1, 7]), ('tile_rc', [-1, 4, 2]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 1)],None,10306226
No: 15  GFLOPS: 5.42/260.18     result: MeasureResult(costs=(0.042678113000000004,), error_no=MeasureErrorNo.NO_ERROR, all_cost=1.8313236236572266, timestamp=1658801313.2114427)       [('tile_f', [-1, 2, 2, 8]), ('tile_y', [-1, 1, 1, 7]), ('tile_x', [-1, 7, 1, 1]), ('tile_rc', [-1, 4, 8]), ('tile_ry', [-1, 1, 1]), ('tile_rx', [-1, 1, 1]), ('auto_unroll_max_step', 0), ('unroll_explicit', 1)],None,5330964
No: 16  GFLOPS: 3.34/260.18     result: MeasureResult(costs=(0.0693738225,), error_no=MeasureErrorNo.NO_ERROR, all_cost=4.546748638153076, timestamp=1658801314.4557946)        [('tile_f', [-1, 8, 4, 4]), ('tile_y', [-1, 1, 1, 7]), ('tile_x', [-1, 1, 1, 7]), ('tile_rc', [-1, 4, 1]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 1]), ('auto_unroll_max_step', 512), ('unroll_explicit', 0)],None,2140058
No: 17  GFLOPS: 0.00/260.18     result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 142, in build
    res = future.result()
  File "/usr/lib/python3.7/concurrent/futures/_base.py", line 435, in result
    return self.__get_result()
  File "/usr/lib/python3.7/concurrent/futures/_base.py", line 384, in __get_result
    raise self._exception
  File "/usr/lib/python3.7/concurrent/futures/thread.py", line 57, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/workspace/python/tvm/contrib/popen_pool.py", line 404, in <lambda>
    worker = lambda *args: self._worker_run(*args)
  File "/workspace/python/tvm/contrib/popen_pool.py", line 373, in _worker_run
    return proc.recv()
  File "/workspace/python/tvm/contrib/popen_pool.py", line 297, in recv
    raise TimeoutError()
TimeoutError

        [('tile_f', [-1, 2, 2, 1]), ('tile_y', [-1, 1, 7, 1]), ('tile_x', [-1, 7, 1, 1]), ('tile_rc', [-1, 4, 16]), ('tile_ry', [-1, 3, 1]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 1)],None,10195251
No: 18  GFLOPS: 27.80/260.18    result: MeasureResult(costs=(0.008326810928571429,), error_no=MeasureErrorNo.NO_ERROR, all_cost=1.2733991146087646, timestamp=1658801325.4546382)       [('tile_f', [-1, 4, 8, 4]), ('tile_y', [-1, 1, 1, 1]), ('tile_x', [-1, 1, 1, 1]), ('tile_rc', [-1, 1, 4]), ('tile_ry', [-1, 3, 1]), ('tile_rx', [-1, 3, 1]), ('auto_unroll_max_step', 0), ('unroll_explicit', 1)],None,6068603
No: 19  GFLOPS: 0.00/260.18     result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 16, 4, 8]), ('tile_y', [-1, 1, 7, 1]), ('tile_x', [-1, 7, 1, 1]), ('tile_rc', [-1, 4, 128]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 0), ('unroll_explicit', 1)],None,6956993
No: 20  GFLOPS: 0.00/260.18     result: Traceback (most recent call last):
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 588, in __call__
    func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 540, in _build_func_common
    func = build(s, args, target_host=task.target_host, runtime=runtime)
  File "/workspace/python/tvm/driver/build_module.py", line 228, in build
    input_mod = lower(inputs, args, name=name, binds=binds)
  File "/workspace/python/tvm/driver/build_module.py", line 134, in lower
    return ffi.lower_schedule(inp, args, name, binds, simple_mode)
  File "tvm/_ffi/_cython/./packed_func.pxi", line 331, in tvm._ffi._cy3.core.PackedFuncBase.__call__
  File "tvm/_ffi/_cython/./packed_func.pxi", line 276, in tvm._ffi._cy3.core.FuncCall
  File "tvm/_ffi/_cython/./base.pxi", line 181, in tvm._ffi._cy3.core.CHECK_CALL
tvm._ffi.base.TVMError: Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel

Traceback (most recent call last):
  24: TVMFuncCall
        at ../src/runtime/c_runtime_api.cc:477
  23: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  22: Call
        at ../include/tvm/runtime/packed_func.h:1213
  21: operator()
        at ../include/tvm/runtime/packed_func.h:1731
  20: unpack_call<tvm::IRModule, 5, tvm::<lambda(tvm::te::Schedule, const tvm::runtime::Array<tvm::runtime::ObjectRef>&, const tvm::runtime::String&, const tvm::runtime::Map<tvm::te::Tensor, tvm::tir::Buffer>&, bool)> >
        at ../include/tvm/runtime/packed_func.h:1671
  19: run<>
        at ../include/tvm/runtime/packed_func.h:1631
  18: run<tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  17: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  16: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  15: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1631
  14: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
        at ../include/tvm/runtime/packed_func.h:1646
  13: operator()
        at ../src/driver/driver_api.cc:365
  12: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
        at ../src/driver/driver_api.cc:352
  11: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
        at ../src/driver/driver_api.cc:252
  10: tvm::transform::Pass::operator()(tvm::IRModule) const
        at ../src/ir/transform.cc:258
  9: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  8: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:453
  7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/ir/transform.cc:274
  6: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
        at ../src/tir/ir/transform.cc:100
  5: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::operator()(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext) const
        at ../include/tvm/runtime/packed_func.h:1750
  4: tvm::tir::PrimFunc tvm::runtime::detail::typed_packed_call_dispatcher<tvm::tir::PrimFunc>::run<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::runtime::PackedFunc const&, tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&)
        at ../include/tvm/runtime/packed_func.h:1694
  3: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext>(tvm::tir::PrimFunc&&, tvm::IRModule&&, tvm::transform::PassContext&&) const
        at ../include/tvm/runtime/packed_func.h:1618
  2: tvm::runtime::PackedFuncObj::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
        at ../include/tvm/runtime/packed_func.h:1217
  1: Call
        at ../include/tvm/runtime/packed_func.h:1213
  0: operator()
        at ../src/runtime/c_runtime_api.cc:534
  File "tvm/_ffi/_cython/./packed_func.pxi", line 56, in tvm._ffi._cy3.core.tvm_callback
  File "/workspace/python/tvm/autotvm/measure/measure_methods.py", line 871, in verify_pass
    raise InstantiationError("Skipped because of invalid gpu kernel")
tvm.autotvm.task.space.InstantiationError: Skipped because of invalid gpu kernel        [('tile_f', [-1, 16, 1, 2]), ('tile_y', [-1, 7, 1, 1]), ('tile_x', [-1, 1, 7, 1]), ('tile_rc', [-1, 32, 4]), ('tile_ry', [-1, 1, 3]), ('tile_rx', [-1, 1, 3]), ('auto_unroll_max_step', 512), ('unroll_explicit', 0)],None,3377719
```

最后从日志文件中检查最佳配置，检查正确性并测试运行时间。

``` python
# 检查最佳配置
dispatch_context = autotvm.apply_history_best("conv2d.log")
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)

# 从日志文件中应用历史最好记录
with autotvm.apply_history_best("conv2d.log"):
    with tvm.target.Target("cuda"):
        s, arg_bufs = conv2d_no_batching(N, H, W, CO, CI, KH, KW, strides, padding)
        func = tvm.build(s, arg_bufs)

# 验证正确性
a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

dev = tvm.cuda()
a_tvm = tvm.nd.array(a_np, device=dev)
w_tvm = tvm.nd.array(w_np, device=dev)
c_tvm = tvm.nd.empty(c_np.shape, device=dev)
func(a_tvm, w_tvm, c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-2)

# 评估运行时间。这里选择一个较大的重复次数 (400) 来减少噪音以及内核启动的开销。还可用 nvprof 来验证结果。
evaluator = func.time_evaluator(func.entry_name, dev, number=400)
print("Time cost of this operator: %f" % evaluator(a_tvm, w_tvm, c_tvm).mean)
```

输出结果：

``` bash
Finish loading 20 records

Best config:
[('tile_f', [-1, 8, 2, 1]), ('tile_y', [-1, 7, 1, 1]), ('tile_x', [-1, 1, 7, 1]), ('tile_rc', [-1, 2, 1]), ('tile_ry', [-1, 3, 1]), ('tile_rx', [-1, 3, 1]), ('auto_unroll_max_step', 1500), ('unroll_explicit', 0)],None,4264713
Finish loading 20 records
Time cost of this operator: 0.001299
```

[下载 Python 源代码：tune_conv2d_cuda.py](https://tvm.apache.org/docs/_downloads/6ad550da5092845382b1197f58a93816/tune_conv2d_cuda.py)

[下载 Jupyter notebook：tune_conv2d_cuda.ipynb](https://tvm.apache.org/docs/_downloads/732ed130cbc15432e737da8cc47e1734/tune_conv2d_cuda.ipynb)
