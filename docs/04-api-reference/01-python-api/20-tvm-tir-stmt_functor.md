---

title: tvm.tir.stmt_functor

---


用于 IR 变换的语句函子实用程序。

## tvm.tir.stmt_functor.ir_transform(*stmt*, *preorder*, *postorder*, *only_enable=None*)


按照 DFS 后顺序递归访问和转换 ir 节点。
* **参数：**
   * **stmt** ([tvm.tir.Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt))：要转换的输入。
   * **preorder** (*function*)：递归变异之前调用的函数。如果 preorder 返回 None，则变换将继续进行递归调用。如果 preorder 返回非 None 的 tvm.tir.Stmt/Expr，则变换器将直接返回该值，而不会进行进一步的递归。
   * **postorder** (*function*)：递归变异后调用的函数。
   * **only_enable** (*Optional**[****List**[***[str](https://docs.python.org/3/library/stdtypes.html#str)***]***]*)：我们仅启用的类型列表。
* **返回：result**：结果。
* **返回类型：**[tvm.tir.Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

## tvm.tir.stmt_functor.post_order_visit(*stmt*, *fvisit*)

递归访问后 DFS 顺序节点，应用 fvisit。


保证每个节点只被访问一次。
* **参数：fvisit** (*function*)：访问者函数。

## tvm.tir.stmt_functor.pre_order_visit(*stmt*, *fvisit*)



对 stmt AST 进行递归预序访问，在每个节点上应用 fvisit。


如果 fvisit 返回 False，它将不会访问该节点的子节点。
* **参数：fvisit** (*functionofthe signature Object -> bool*)：访问者函数。

## tvm.tir.stmt_functor.substitute(*node*, *vmap*)


替换 vmap 指定的 var。
* **参数：**
   * **node** (*ObjectRef*)：输入。
   * **vmap** (*Dict[*[tir.Var](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,*[PrimExpr](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]*)：变量映射。
* **返回：result**：结果。
* **返回类型：**[tvm.tir.Stmt](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)。

## tvm.tir.stmt_functor.renew_defs(*func:*[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))


重新生成 TIR 的定义节点，包括 VarDef 和 BufferDef。此过程相当于一个简单的 DeepCopy，用于复制具有不同 Var 和 Buffer 但行为相同的函数。
* **参数：func** ([PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone))：输入函数。
* **返回：result**：新生成的函数。
* **返回类型：**[PrimFunc](https://tvm.hyper.ai/docs/api-reference/python-api/tvm-tir#class-tvmtirprimfuncparams-body-ret_typenone-buffer_mapnone-attrsnone-spannone)。


