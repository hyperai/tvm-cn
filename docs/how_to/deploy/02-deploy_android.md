# 部署到 Android

## 为 Android Target 构建模型

针对 Android target 的 Relay 模型编译遵循和 android_rpc 相同的方法，以下代码会保存 Android target 所需的编译输出：

``` python
lib.export_library("deploy_lib.so", ndk.create_shared)
with open("deploy_graph.json", "w") as fo:
    fo.write(graph.json())
with open("deploy_param.params", "wb") as fo:
    fo.write(runtime.save_param_dict(params))
```

deploy_lib.so、deploy_graph.json、deploy_param.params 将转到 Android target。

## 适用于 Android Target 的 TVM Runtime

参考 [此处](https://github.com/apache/tvm/blob/main/apps/android_deploy/README.md#build-and-installation) 为 Android target 构建 CPU/OpenCL 版本的 TVM runtime。参考这个 [Java](https://github.com/apache/tvm/blob/main/apps/android_deploy/app/src/main/java/org/apache/tvm/android/demo/MainActivity.java) 示例来了解 Android Java TVM API，以及如何加载和执行模型。