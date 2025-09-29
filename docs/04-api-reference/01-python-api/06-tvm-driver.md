---

title: tvm.driver

---

驱动程序 API 的命名空间

## **tvm.compile(*mod:***[PrimFunc](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.PrimFunc)***|***[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)**,*target:***[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)***|***[None](https://docs.python.org/3/library/constants.html#None)***= None*,***,*relax_pipeline:***[Pass](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.Pass)***|***[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)***|***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[None](https://docs.python.org/3/library/constants.html#None)***= 'default'*,*tir_pipeline:***[Pass](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.Pass)***|***[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)***|***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[None](https://docs.python.org/3/library/constants.html#None)***= 'default'*)→ Executable*

将 IRModule 编译为运行时可执行文件。

此函数作为编译 TIR 和 Relax 模块的统一入口点。它会自动检测模块类型并路由到相应的构建函数。
* **参数：**
   * **mod** ( *Union [*[PrimFunc](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.PrimFunc)*,*[IRModule](https://tvm.apache.org/docs/reference/api/python/ir.html#tvm.ir.IRModule)*]* ) ：待编译的输入模块。可以是包含 TIR 或 Relax 函数的 PrimFunc 或 IRModule。
   * **target** (*Optional[*[Target](https://tvm.apache.org/docs/reference/api/python/target.html#tvm.target.Target)*]*) ：要编译的目标平台。
   * **Relax_pipeline** ( *Optional [ Union [*[tvm.transform.Pass](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.Pass)*, Callable ,*[str](https://docs.python.org/3/library/stdtypes.html#str)*] ]* ) ：Relax 函数使用的编译管道。仅当模块包含 Relax 函数时使用。
   * **tir_pipeline** ( *Optional [ Union [*[tvm.transform.Pass](https://tvm.apache.org/docs/reference/api/python/transform.html#tvm.transform.Pass)*, Callable ,*[str](https://docs.python.org/3/library/stdtypes.html#str)*] ]* ) ：用于 TIR 函数的编译管道。
* **返回：** 可以加载和执行的运行时可执行文件。
* **返回类型：** Executable.



