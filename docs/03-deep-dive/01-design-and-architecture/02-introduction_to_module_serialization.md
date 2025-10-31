---

title: 模块序列化指南


---

在部署 TVM 运行时模块时，无论目标是 CPU 还是 GPU，TVM **最终只需要一个动态共享库（dynamic shared library）**。实现这一点的关键就在于 **统一的模块序列化机制**。本文将介绍 TVM 模块序列化的格式标准与实现细节。

## 序列化（Serialization）

入口 API 为 `tvm.module.Module` 的 `export_library`。在此函数内部，我们会执行以下步骤：

1. **收集所有 DSO 模块**（例如 LLVM 模块和 C 模块）。
2. 在获得 DSO 模块后，调用 `save` 函数将它们保存到文件。
3. 随后检查是否存在已导入的模块（imported modules），例如 CUDA、OpenCL 等。这里对模块类型不做限制。  
   如果存在导入模块，我们将创建一个名为 `devc.o` / `dev.cc` 的文件（用于将这些导入模块的二进制数据打包进最终的动态库中），然后调用 `_PackImportsToLLVM` 或 `_PackImportsToC` 来执行模块序列化。
4. 最后，调用 `fcompile`，其内部会调用`_cc.create_shared`，生成动态共享库。


:::note

1. 对于 C 源码模块（CSourceModule），我们会将它们编译并与 DSO 模块一同进行链接。
2. 是否使用 `_PackImportsToLLVM` 或 `_PackImportsToC` **取决于 TVM 是否启用了 LLVM**。它们本质上实现的是相同的目标。

:::

## 序列化底层机制与格式标准

序列化主要发生在 `_PackImportsToLLVM` 或`_PackImportsToC` 中。它们都会调用 `SerializeModule` 来序列化 runtime module。在 `SerializeModule` 函数中，我们首先会构造一个辅助类 `ModuleSerializer`。它会以 `module` 为输入进行初始化，例如分配模块索引。随后可以调用其 `SerializeModule` 方法执行序列化。

为了更好地理解，让我们更深入地挖掘这个类的实现。

下面的代码用于构造 `ModuleSerializer`：

```c++
explicit ModuleSerializer(runtime::Module mod) : mod_(mod) {
  Init();
}
private:
void Init() {
  CreateModuleIndex();
  CreateImportTree();
}
```


在 `CreateModuleIndex()` 中，我们使用 DFS 遍历模块的导入关系并为每个模块分配索引。根模块固定为索引 `0`。

例如：

```
llvm_mod:imported_modules
  - cuda_mod
```

因此，LLVM 模块的索引将是 0，CUDA 模块的索引将是 1。

在构建完模块索引之后，我们将尝试构建导入树（`CreateImportTree()`），该导入树会在我们重新加载导出的库时用于恢复模块之间的导入关系。在我们的设计中，我们使用 CSR 格式来存储导入树，每一行对应父节点索引，而子数组中的索引对应其子模块索引。在代码中，我们使用 `import_tree_row_ptr_ `和`import_tree_child_indices_` 来表示它们。

在完成初始化之后，我们就可以使用 `SerializeModule` 函数来序列化模块。

在该函数的逻辑中，我们假设序列化格式如下所示：

```c++
binary_blob_size
binary_blob_type_key
binary_blob_logic
binary_blob_type_key
binary_blob_logic
...
_import_tree
_import_tree_logic
```
`binary_blob_size` 是我们在本次序列化步骤中将会包含的 blob 数量。在我们的示例中会有三个 blob，分别对应 LLVM 模块、CUDA 模块以及 `_import_tree`。

`binary_blob_type_key` 是模块的 blob 类型键。
对于 LLVM / C 模块，其 blob 类型键为 `_lib`。对于 CUDA 模块，其类型键为 `cuda`，可以通过 `module->type_key()` 获取。

`binary_blob_logic` 是处理该 blob 的逻辑。
对于大多数 blob（例如 CUDA、OpenCL），我们会调用 `SaveToBinary` 函数将 blob 序列化为二进制。然而，对于 LLVM / C 模块，我们只会写入 `_lib`，用于表示这是一个 DSO 模块。


:::note

是否需要实现 SaveToBinary 虚函数取决于模块的使用方式。例如，如果模块中包含我们在重新加载动态共享库时需要的信息，那么我们就应该实现该函数。像 CUDA 模块，在重新加载动态共享库时我们需要将其二进制数据传递给 GPU 驱动，因此我们需要实现 `SaveToBinary` 来序列化其二进制数据。但对于主机侧模块（如 DSO 模块），在加载动态共享库时我们并不需要额外信息，因此不需要实现 `SaveToBinary`。不过，如果未来我们希望记录一些关于 DSO 模块的元信息，我们也可以为 DSO 模块实现 `SaveToBinary`。

:::

最后，除非我们的模块中仅有一个 DSO 模块并且它位于根位置，否则我们会写入一个键` _import_tree`。该键用于在重新加载导出的库时恢复模块导入关系，如前文所述。`import_tree_logic` 的内容则是将 `import_tree_row_ptr_ `和 `import_tree_child_indices_` 写入到流中。

在上述步骤完成后，我们会将最终结果打包进一个符号
`runtime::symbol::tvm_ffi_library_bin`，该符号可在动态库中恢复。

现在，我们已经完成序列化部分。正如你所看到的，我们理论上可以支持导入任意模块。


---

## 反序列化

入口 API 是 `tvm.runtime.load`。实际上，该函数会调用 `_LoadFromFile`。  如果进一步展开，可以看到其对应的是 `Module::LoadFromFile`。

在我们的示例中，文件是 `deploy.so`。根据其函数逻辑，我们会在 `dso_library.cc` 中调用 `module.loadfile_so`，关键代码如下：

```c++
// Load the imported modules
const char* library_bin = reinterpret_cast<const char*>(
   lib->GetSymbol(runtime::symbol::tvm_ffi_library_bin));
Module root_mod;
if (library_bin != nullptr) {
   root_mod = ProcessLibraryBin(library_bin, lib);
} else {
   // Only have one single DSO Module
   root_mod = Module(n);
}```

如前所述，我们会将 blob 打包进符号 `runtime::symbol::tvm_ffi_library_bin`· 中。
在反序列化阶段，我们会检查它。如果存在 `runtime::symbol::tvm_ffi_library_bin`，我们将调用 `ProcessLibraryBin`，其逻辑如下：

```c++
READ(blob_size)
READ(blob_type_key)
for (size_t i = 0; i < blob_size; i++) {
    if (blob_type_key == "_lib") {
      // construct dso module using lib
    } else if (blob_type_key == "_import_tree") {
      // READ(_import_tree_row_ptr)
      // READ(_import_tree_child_indices)
    } else {
      // call module.loadbinary_blob_type_key, such as module.loadbinary_cuda
      // to restore.
    }
}
// Using _import_tree_row_ptr and _import_tree_child_indices to
// restore module import relationship. The first module is the
// root module according to our invariance as said before.
return root_module;
```
完成上述步骤后，我们会将 `ctx_address` 设置为 `root_module`，
以便能够从根模块查找符号（使所有符号可见）。

最终，我们就完成了反序列化部分。





