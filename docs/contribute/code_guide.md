---
title: 代码指南及 Tips
sidebar_position: 6
---

本文档为 reviewer 和贡献者汇总了 TVM 代码库中的技巧，大部分是在贡献过程中总结的经验教训。

## C++ 代码风格

-   使用 Google C/C++ 风格。
-   面向公众的功能以 doxygen 格式记录。
-   如果代码很短，使用具体类型声明而不是 `auto`。
-   通过 const 引用传递（例如 `const Expr&`）而不是按值传递。除非函数通过拷贝构造函数或移动构造函数使用该值，在这种情况下，按值传递优于通过 const 引用传递。
-   尽可能使用 `const` 成员函数。

我们可以用 `clang-format` 来规整代码风格。因为不同版本的 clang-format 可能会因版本而异，所以建议使用相同版本的 clang-format 作为主要版本。您还可以通过 docker 使用以下命令。

``` bash
# 通过 clang-format 运行指定文件
docker/bash.sh ci_lint clang-format-10 [path-to-file]

# 运行所有 linter，包括 clang-format
python tests/scripts/ci.py lint
```

clang-format 并不完美，必要时可以在某些代码区域上禁用 clang-format。

``` c++
// clang-format off
void Test() {
    // clang-format 将在这个区域禁用。
}
// clang-format on
```

因为 clang-format 可能无法识别宏，所以建议像普通函数样式一样使用宏。

``` c++
#define MACRO_IMPL { custom impl; }
#define MACRO_FUNC(x)

// 不是首选，因为 clang-format 可能会将其识别为类型。

// 首选
virtual void Func2() MACRO_IMPL;

void Func3() {
    // 首选
    MACRO_FUNC(xyz);
}
```

## Python 代码风格

-   函数和类以 [numpydoc](https://numpydoc.readthedocs.io/en/latest/) 格式记录。
-   使用 `python tests/scripts/ci.py lint` 检查代码风格
-   使用 `python 3.7` 中的语言特性

## 编写 Python 测试

用 [pytest](https://docs.pytest.org/en/stable/) 进行所有 Python 测试。 `tests/python` 包含所有测试。

如果您希望测试在各种 target 上运行，请使用 `tvm.testing.parametrize_targets()` 装饰器。例如：

``` python
@tvm.testing.parametrize_targets
def test_mytest(target, dev):
  ...
```

用 `target="llvm"`、`target="cuda"` 和其他几个运行 `test_mytest`。这可以确保测试由 CI 在正确的硬件上运行。如果只想针对几个 target 进行测试，请使用 `@tvm.testing.parametrize_targets("target_1", "target_2")`。如果想在单个 target 上进行测试，请使用来自 `tvm.testing()` 的相关装饰器。例如，CUDA 测试使用 `@tvm.testing.requires_cuda` 装饰器。

## 网络资源
在CI中，从互联网下载文件是导致测试失败的一个主要原因（例如，远程服务器可能宕机或速度较慢），因此在测试期间尽量避免使用网络。在某些情况下，这并不是一个合理的建议（例如，需要下载模型的文档教程时）。

在这些情况下，您可以在CI中重新托管文件到S3，以便快速访问。Committer可以在[upload_ci_resource.yml GitHub Actions flow](https://github.com/apache/tvm/actions/workflows/upload_ci_resource.yml)上使用`workflow_dispatch`来上传一个文件，该文件由S3中的名称、哈希和路径指定。sha256必须与文件匹配，否则将无法上传。上传路径由用户定义，可以是任意路径（不允许尾随或前导斜杠），但要小心不要意外地与现有资源发生冲突。上传完成后，您应该发送一个PR，上传新的URL来更新[request_hook.py](https://github.com/apache/tvm/blob/main/tests/scripts/request_hook/request_hook.py)中的`URL_MAP`。

## 处理整型常量表达式

TVM 中经常需要处理整型常量表达式。在此之前，首先要考虑的是，是否真的有必要获取一个整型常量。如果符号表达式也有效并且逻辑行得通，那么尽可能用符号表达式。所以生成的代码也适用于未知的 shape。

注意，某些情况下，无法知道符号变量的符号这样的信息，这样的情况可做出一些假设。如果变量是常量，则添加精确的支持。

如果必须获取整型常量表达式，应该用 `int64_t` 类型而不是 `int` 来获取常量值，以避免整数溢出。通过 `make_const` 可以重构一个具有相应表达式类型的整数。相关示例如下所示：

``` c++
Expr CalculateExpr(Expr value) {
  int64_t int_value = GetConstInt<int64_t>(value);
  int_value = CalculateExprInInt64(int_value);
  return make_const(value.type(), int_value);
}
```
