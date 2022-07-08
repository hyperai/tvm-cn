---
title: Code Guide and Tips
---

::: {.contents depth="2" local=""}
:::

This is a document used to record tips in TVM codebase for reviewers and
contributors. Most of them are summarized through lessons during the
contributing and process.

## C++ Code Styles

-   Use the Google C/C++ style.
-   The public facing functions are documented in doxygen format.
-   Favor concrete type declaration over `auto` as long as it is short.
-   Favor passing by const reference (e.g. `const Expr&`) over passing
    by value. Except when the function consumes the value by copy
    constructor or move, pass by value is better than pass by const
    reference in such cases.
-   Favor `const` member function when possible.

We use `clang-format` to enforce the code style. Because different
version of clang-format might change by its version, it is recommended
to use the same version of the clang-format as the main one. You can
also use the following command via docker.

``` bash
# Run a specific file through clang-format
docker/bash.sh ci_lint clang-format-10 [path-to-file]

# Run all linters, including clang-format
python tests/scripts/ci.py lint
```

clang-format is also not perfect, when necessary, you can use disble
clang-format on certain code regions.

> // clang-format off void Test() { // clang-format will be disabled in
> this region. } // clang-format on

Because clang-format may not recognize macros, it is recommended to use
macro like normal function styles.

> #define MACRO_IMPL { custom impl; } #define MACRO_FUNC(x)
>
> // not preferred, because clang-format might recognize it as types.
> virtual void Func1() MACRO_IMPL
>
> // preferred virtual void Func2() MACRO_IMPL;
>
> void Func3() {
>
> :   // preferred MACRO_FUNC(xyz);
>
> }

## Python Code Styles

-   The functions and classes are documented in
    [numpydoc](https://numpydoc.readthedocs.io/en/latest/) format.
-   Check your code style using `python tests/scripts/ci.py lint`
-   Stick to language features in `python 3.7`

## Writing Python Tests

We use [pytest](https://docs.pytest.org/en/stable/) for all python
testing. `tests/python` contains all the tests.

If you want your test to run over a variety of targets, use the
:py`tvm.testing.parametrize_targets`{.interpreted-text role="func"}
decorator. For example:

``` python
@tvm.testing.parametrize_targets
def test_mytest(target, dev):
  ...
```

will run `test_mytest` with `target="llvm"`, `target="cuda"`, and few
others. This also ensures that your test is run on the correct hardware
by the CI. If you only want to test against a couple targets use
`@tvm.testing.parametrize_targets("target_1", "target_2")`. If you want
to test on a single target, use the associated decorator from
:py`tvm.testing`{.interpreted-text role="func"}. For example, CUDA tests
use the `@tvm.testing.requires_cuda` decorator.

## Handle Integer Constant Expression

We often need to handle constant integer expressions in TVM. Before we
do so, the first question we want to ask is that is it really necessary
to get a constant integer. If symbolic expression also works and let the
logic flow, we should use symbolic expression as much as possible. So
the generated code works for shapes that are not known ahead of time.

Note that in some cases we cannot know certain information, e.g. sign of
symbolic variable, it is ok to make assumptions in certain cases. While
adding precise support if the variable is constant.

If we do have to get constant integer expression, we should get the
constant value using type `int64_t` instead of `int`, to avoid potential
integer overflow. We can always reconstruct an integer with the
corresponding expression type via `make_const`. The following code gives
an example.

``` c++
Expr CalculateExpr(Expr value) {
  int64_t int_value = GetConstInt<int64_t>(value);
  int_value = CalculateExprInInt64(int_value);
  return make_const(value.type(), int_value);
}
```
