---

title: 代码指南和提示

---

* [C++ 代码风格](/docs/about/contribute/code_guide_and_Tips#c-%E4%BB%A3%E7%A0%81%E9%A3%8E%E6%A0%BC)
* [Python 代码风格](/docs/about/contribute/code_guide_and_Tips#python-%E4%BB%A3%E7%A0%81%E6%A0%B7%E5%BC%8F)
* [编写 Python 测试](/docs/about/contribute/code_guide_and_Tips#%E7%BC%96%E5%86%99-python-%E6%B5%8B%E8%AF%95)
* [网络](/docs/about/contribute/code_guide_and_Tips#%E7%BD%91%E7%BB%9C%E8%B5%84%E6%BA%90%E5%A4%84%E7%90%86)[资源](/docs/about/contribute/code_guide_and_Tips#%E7%BD%91%E7%BB%9C%E8%B5%84%E6%BA%90%E5%A4%84%E7%90%86)
* [整型常量表达式处理](/docs/about/contribute/code_guide_and_Tips#%E6%95%B4%E5%9E%8B%E5%B8%B8%E9%87%8F%E8%A1%A8%E8%BE%BE%E5%BC%8F%E5%A4%84%E7%90%86)


本文档记录 TVM 代码库中供评审者和贡献者参考的实用技巧，多数内容源自贡献过程中的经验总结。


## C++ 代码风格
* 采用 Google C/C++ 代码风格。
* 公开函数使用 doxygen 格式文档注释。
* 类型声明优先使用具体类型而非 `auto`（当类型名称较短时）。
* 参数传递优先使用常量引用（如 `const Expr&`），但当函数需要通过拷贝或移动构造函数消费参数时，传值优于传常量引用。
* 尽可能使用 `const` 成员函数。


我们使用 `clang-format` 强制执行代码风格。由于不同版本的 clang-format 可能存在差异，建议使用与主版本相同的 clang-format 版本。也可以通过 docker 执行以下命令：

```plain

# 对指定文件运行 clang-format。
docker/bash.sh ci_lint clang-format-10 [path-to-file]


# 运行所有检查工具（包括 clang-format）。
python tests/scripts/ci.py lint
```


当特定代码区域需要规避格式化时，可使用以下方式：

```plain
// clang-format off
void Test() {
   // 此区域内禁用 clang-format。
}
// clang-format on
```


由于 clang-format 可能无法正确识别宏，建议宏的使用风格保持与普通函数一致：

```plain
#define MACRO_IMPL { custom impl; }
#define MACRO_FUNC(x)


// 不推荐（clang-format 可能误识别为类型声明）。
virtual void Func1() MACRO_IMPL


// 推荐方式。
virtual void Func2() MACRO_IMPL;

void Func3() {
  
  // 推荐方式。
  MACRO_FUNC(xyz);
}
```


## Python 代码样式
* 函数和类采用 [numpydoc](https://numpydoc.readthedocs.io/en/latest/) 格式文档。
* 使用 `python tests/scripts/ci.py lint` 检查代码风格。
* 限定使用 Python 3.7 的语言特性。
* 对于包含提前返回的函数：当各条件分支逻辑平行且简短时（如参数简单映射），推荐使用 `if`/`elif`/`else` 链式结构。当最终 `else` 块明显长于其他分支时（常见于流程型函数），应取消最终 `else` 的缩进。


我们已禁用 pylint 的 `no-else-return` 检查以支持这种区分，详见相关讨论： <https://github.com/apache/tvm/pull/11327>。


```plain
# 各分支流程相似的情况（虽然可用连续if实现，但读者需检查每个分支才能确认唯一执行路径）。
def sign(x):
    if x > 0:
        return "+"
    elif x < 0:
        return "-"
    else:
        return ""

# 特殊情况提前返回模式（使用else会导致后续主要逻辑不必要缩进）。
def num_unique_subsets(values):
    if len(values)==0:
        return 1

   
    # 后续是较长的主要处理逻辑。
    ...
```

## 编写 Python 测试

我们使用 [pytest](https://docs.pytest.org/en/stable/) 进行所有 Python 测试，测试用例位于 `tests/python` 目录。

多设备测试应使用 `tvm.testing.parametrize_targets()` 装饰器：

```plain
@tvm.testing.parametrize_targets
def test_mytest(target, dev):
  ...
```


这将自动使用 `target="llvm"`、`target="cuda"` 等参数运行测试，并确保 CI 在正确的硬件上执行。若只需测试特定设备，可使用 `@tvm.testing.parametrize_targets("target_1", "target_2")`。单设备测试直接使用对应的装饰器（如 CUDA 测试使用 `@tvm.testing.requires_cuda`）。


## 网络资源处理

在 CI 环境中，网络文件下载是导致测试不稳定的主要因素（如远程服务器宕机或延迟），应尽量避免测试过程中访问网络。但某些情况下并不合理（如文档教程需要下载模型）。


对于这些情况，你可以将文件托管至 S3 以保证 CI 快速访问；项目维护者可通过 upload_ci_resource.yml 工作流的 workflow_dispatch 事件上传文件（需指定文件名、SHA256 校验和及 S3 路径）；文件上传后需提交 PR 更新 request_hook.py 中的 URL_MAP。

## 整型常量表达式处理

处理整型常量表达式前，首先应评估是否真的需要常量值。若符号表达式也能满足逻辑流程，应优先使用符号表达式以使代码能处理运行时才确定的形状。


注意：对于无法确定的信息（如符号变量的正负），可在特定情况下做出假设。同时应为常量情况保留精确支持。


如果我们确实需要获取常量整数表达式，应该使用 `int64_t` 而不是 `int` 来获取常量值，以避免潜在的整数溢出。我们可以始终通过 `make_const` 使用相应的表达式类型重新构建一个整数。以下代码给出了一个示例。

```plain
Expr CalculateExpr(Expr value) {
  int64_t int_value = GetConstInt<int64_t>(value);
  int_value = CalculateExprInInt64(int_value);
  return make_const(value.type(), int_value);
}
```


