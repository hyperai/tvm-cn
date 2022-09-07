# 使用 C++ API 部署 TVM 模块

[apps/howto_deploy](https://github.com/apache/tvm/tree/main/apps/howto_deploy) 中给出了部署 TVM 模块的示例，执行下面的命令运行该示例：

``` bash
cd apps/howto_deploy
./run_example.sh
```

## 获取 TVM Runtime 库

唯一要做的是链接到 target 平台中的 TVM runtime。 TVM 给出了一个最小 runtime，它的开销大约在 300K 到 600K 之间，具体值取决于使用模块的数量。大多数情况下，可用 `libtvm_runtime.so` 文件去构建。

若构建 `libtvm_runtime` 有困难，可查看 [tvm_runtime_pack.cc](https://github.com/apache/tvm/tree/main/apps/howto_deploy/tvm_runtime_pack.cc)（集成了 TVM runtime 的所有示例）。用构建系统来编译这个文件，然后将它包含到项目中。

查看 [apps](https://github.com/apache/tvm/tree/main/apps/) 获取在 iOS、Android 和其他平台上，用 TVM 构建的应用示例。

## 动态库 vs. 系统模块

TVM 有两种使用编译库的方法，查看 [prepare_test_libs.py](https://github.com/apache/tvm/tree/main/apps/howto_deploy/prepare_test_libs.py) 了解如何生成库，查看 [cpp_deploy.cc](https://github.com/apache/tvm/tree/main/apps/howto_deploy/cpp_deploy.cc) 了解如何使用它们。

* 把库存储为共享库，并动态加载到项目中。
* 将编译好的库以系统模块模式绑定到项目中。

动态加载更加灵活，能快速加载新模块。系统模块是一种更 `static` 的方法，可用在动态库加载不可用的地方。