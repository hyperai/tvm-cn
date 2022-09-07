---
title: PAPI 入门
---

# PAPI 入门

性能应用程序编程接口 (Performance Application Programming Interface，简称 PAPI) 是一个可在各种平台上提供性能计数器的库。在指定的运行期间，性能计数器提供处理器行为的准确底层信息，包含简单的指标，如总循环计数、缓存未命中和执行的指令，以及更高级的信息（如总 FLOPS 和 warp 占用率）。PAPI 的这些指标在 profiling 时可用。

## 安装 PAPI

PAPI 可以用包管理器（Ubuntu 上用 `apt-get install libpapi-dev` 命令）来安装，也可以从 [源代码](https://bitbucket.org/icl/papi/src/master/) 安装。

由于之前从源代码 pull 最新版本的 PAPI 导致了构建问题，因此推荐 checkout 标记版本 `papi-6-0-0-1-t`。

## 使用 PAPI 构建 TVM

若要在 TVM 构建中包含 PAPI，在 `config.cmake` 中设置：

``` cmake
set(USE_PAPI ON)
```

若 PAPI 安装在非标准位置，可指定它的位置：

``` cmake
set(USE_PAPI path/to/papi.pc)
```

## 在 Profiling 时使用 PAPI

若 TVM 是用 PAPI 构建的（见上文），可将 `tvm.runtime.profiling.PAPIMetricCollector` 传给 `tvm.runtime.GraphModule.profile()` 来收集性能指标：

``` python
target = "llvm"
dev = tvm.cpu()
mod, params = mlp.get_workload(1)

exe = relay.vm.compile(mod, target, params=params)
vm = profiler_vm.VirtualMachineProfiler(exe, dev)

data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"), device=dev)
report = vm.profile(
    [data],
    func_name="main",
    collectors=[tvm.runtime.profiling.PAPIMetricCollector()],
)
print(report)
```
``` bash
Name                                    perf::CACHE-MISSES   perf::CYCLES  perf::STALLED-CYCLES-BACKEND  perf::INSTRUCTIONS  perf::STALLED-CYCLES-FRONTEND
fused_nn_dense_nn_bias_add_nn_relu                   2,494      1,570,698                        85,608             675,564                         39,583
fused_nn_dense_nn_bias_add_nn_relu_1                 1,149        655,101                        13,278             202,297                         21,380
fused_nn_dense_nn_bias_add                             288        600,184                         8,321             163,446                         19,513
fused_nn_batch_flatten                                 301        587,049                         4,636             158,636                         18,565
fused_nn_softmax                                       154        575,143                         8,018             160,738                         18,995
----------
Sum                                                  4,386      3,988,175                       119,861           1,360,681                        118,036
Total                                               10,644      8,327,360                       179,310           2,660,569                        270,044
```

还可以指定收集哪些指标：

``` python
report = vm.profile(
    [data],
    func_name="main",
    collectors=[tvm.runtime.profiling.PAPIMetricCollector({dev: ["PAPI_FP_OPS"])],
)
```

``` bash
Name                                  PAPI_FP_OPS
fused_nn_dense_nn_bias_add_nn_relu        200,832
fused_nn_dense_nn_bias_add_nn_relu_1       16,448
fused_nn_dense_nn_bias_add                  1,548
fused_nn_softmax                              160
fused_nn_batch_flatten                          0
----------
Sum                                       218,988
Total                                     218,988
```

运行 `papi_avail` 和 `papi_native_avail` 命令可得到可用指标列表。