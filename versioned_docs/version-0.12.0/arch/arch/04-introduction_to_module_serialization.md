---
title: 模块序列化简介
sidebar_position: 140
---

# 模块序列化简介

当部署 TVM runtime 模块的时候，无论是 CPU 还是 GPU，TVM 只需要一个动态共享库。关键是统一的模块序列化机制。本节将介绍 TVM 模块序列化的格式标准和实现细节。

## 模块导出示例

首先，为 GPU 构建一个 ResNet-18 工作负载，以此示例：

``` python
from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
import tvm

# Resnet18 工作负载
resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)

# 构建
with relay.build_config(opt_level=3):
    _, resnet18_lib, _ = relay.build_module.build(resnet18_mod, "cuda", params=resnet18_params)

# # 创建一个临时目录
temp = utils.tempdir()

# 路径库
file_name = "deploy.so"
path_lib = temp.relpath(file_name)

# 导出库
resnet18_lib.export_library(path_lib)

# 加载回来
loaded_lib = tvm.runtime.load_module(path_lib)
assert loaded_lib.type_key == "library"
assert loaded_lib.imported_modules[0].type_key == "cuda"
```

## 序列化

入口 API 是 `tvm.module.Module` 的 `export_library`。函数将执行以下步骤：

1. 收集所有 DSO 模块（LLVM 模块和 C 模块）
2. 有了 DSO 模块后，调用 `save` 函数，并将其保存至文件。
3. 接下来，检查模块是否导入，例如 CUDA、OpenCL 或其他任何模块。这里不限制模块类型。导入模块后，创建一个名为 `devc.o` / `dev.cc` 的文件（以便可将导入模块的二进制 blob 数据嵌入到一个动态共享库中），然后调用 `_PackImportsToLLVM` 或 `_PackImportsToC` 函数进行模块序列化。
4. 最后，调用 `fcompile`，它会调用 `_cc.create_shared` 来获取动态共享库。

:::note
1. 对于 C 源代码模块，先编译，然后将它们与 DSO 模块链接。
2. 用 `_PackImportsToLLVM` 还是 `_PackImportsToC` 取决于 TVM 是否启用 LLVM。它们的目的其实是一样的。
:::

## 序列化和格式标准的底层

如前所述，序列化工作将在 `_PackImportsToLLVM` 或 `_PackImportsToC` 中进行。它们都调用 `SerializeModule` 来序列化 runtime 模块。在 `SerializeModule` 函数中，首先构造一个辅助类 `ModuleSerializer`。`module` 要做一些初始化工作，比如标记模块索引。然后可以用它的 `SerializeModule` 来序列化模块。

为了方便大家理解，接下来我们将详细讲解这个类的实现。

以下代码用于构造 `ModuleSerializer`：

``` c++
explicit ModuleSerializer(runtime::Module mod) : mod_(mod) {
  Init();
}
private:
void Init() {
  CreateModuleIndex();
  CreateImportTree();
}
```

在 `CreateModuleIndex()` 中，会用 DFS 检查模块导入关系，并为它们创建索引。注意，根模块固定在位置 0。示例中，模块关系如下：

``` c++
llvm_mod:imported_modules
  - cuda_mod
```

因此，LLVM 模块的索引为 0，CUDA 模块的索引为 1。

模块索引构建后，用 `CreateImportTree()` 来构建导入树（import tree），其作用是在加载导出的库时，恢复模块导入关系。在我们的设计中，用 CSR 格式存储导入树，每一行都是父索引，子索引（child indices）对应它的孩子索引（children index）。代码中用 `import_tree_row_ptr_` 和 `import_tree_child_indices_` 来表示它们。

初始化后，可以用 `SerializeModule` 函数来序列化模块。在其功能逻辑中，假设序列化格式如下：

``` text
binary_blob_size
binary_blob_type_key
binary_blob_logic
binary_blob_type_key
binary_blob_logic
...
_import_tree
_import_tree_logic
```

`binary_blob_size` 是这个序列化步骤中的 blob 数量。示例中，分别为 LLVM 模块、CUDA 模块和 `_import_tree` 创建了三个 blob。

`binary_blob_type_key` 是模块的 blob 类型键。对于 LLVM/C 模块，其 blob 类型键为 `_lib`。对于 CUDA 模块，则是 `cuda`，可以通过 `module->type_key()` 获取。

`binary_blob_logic` 是 blob 的逻辑处理。对于大多数 blob（如 CUDA、OpenCL），可以调用 `SaveToBinary` 函数来将 blob 序列化为二进制。但是，类似 LLVM/C 模块，写成 `_lib` 表示这是一个 DSO 模块。

:::note
是否需要实现 SaveToBinary 虚函数，取决于模块的使用方式。例如，加载动态共享库时，若模块有我们需要的信息，则应该实现 SaveToBinary 虚函数。它类似于 CUDA 模块，加载动态共享库时，要将其二进制数据传递给 GPU 驱动程序，因此实现 `SaveToBinary` 来序列化其二进制数据。但是对于主机模块（如 DSO），加载动态共享库时不需要其他信息，因此无需实现`SaveToBinary`。但是，若之后要记录一些 DSO 模块的元信息，也可以为 DSO 模块实现 `SaveToBinary`。
:::

最后，除非模块只有一个 DSO 模块，并且在根目录下，否则要编写一个主要的 `_import_tree`。如前所述，它用于将导出的库加载回来时，重建模块导入关系。`import_tree_logic` 只是将 `import_tree_row_ptr_` 和 `import_tree_child_indices_` 写入流（stream）中。

完成这一步后，将其打包到符号 `runtime::symbol::tvm_dev_mblob` 中，这个符号可以在动态库中恢复。

至此，完成了序列化部分。可以看到，现在已经可以理想地支持任意模块的导入了。

## 反序列化

入口 API 是 `tvm.runtime.load`。这个函数本质上的作用就是调用 `_LoadFromFile`，更具体一点，就是 `Module::LoadFromFile`。示例中，文件是 `deploy.so`，根据函数逻辑，将调用 `dso_library.cc` 中的 `module.loadfile_so`。关键点如下：

``` c++
// 加载导入的模块
const char* dev_mblob = reinterpret_cast<const char*>(lib->GetSymbol(runtime::symbol::tvm_dev_mblob));
Module root_mod;
if (dev_mblob != nullptr) {
    root_mod = ProcessModuleBlob(dev_mblob, lib);
} else {
    // 只有一个 DSO 模块
    root_mod = Module(n);
}
```

如前所述，blob 会被打包到符号 `runtime::symbol::tvm_dev_mblob` 中。可以在反序列化部分对其进行检查。若有 `runtime::symbol::tvm_dev_mblob`，则调用 `ProcessModuleBlob`，逻辑如下：

``` c++
READ(blob_size)
READ(blob_type_key)
for (size_t i = 0; i < blob_size; i++) {
    if (blob_type_key == "_lib") {
      // 用 lib 构建 dso 模块
    } else if (blob_type_key == "_import_tree") {
      // READ(_import_tree_row_ptr)
      // READ(_import_tree_child_indices)
    } else {
      // 调用 module.loadbinary_blob_type_key，如module.loadbinary_cuda 来恢复。
    }
}
// 用 _import_tree_row_ptr 和 _import_tree_child_indices 来恢复模块导入关系。
// 根据之前说的不变性，第一个模块是根模块。
return root_module;
```

之后，将 `ctx_address` 设置为 `root_module`，使得可以从根目录查找符号（因此所有符号都是可见的）。

至此，完成了反序列化部分。
