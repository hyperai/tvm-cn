---

title: tvm.topi

---


TVM 操作符清单。


TOPI 是 TVM 的操作符集合库，提供构建计算声明以及优化调度的语法。


某些调度函数可能针对特定工作负载进行了专门优化。




**类：**

|**Analyzer**()|整数算术分析器。|
|:----|:----|
|**Cast**(dtype,value[,span])|类型转换表达式|
|**PrimExpr**|所有原始表达式的基类。|

**函数：**

|**abs**(x)|逐个元素地取 x 输入的绝对值。|
|:----|:----|
|**acos**(x)|取输入 x 的反余弦值。|
|**acosh**(x)|取输入 x 的反余弦值。|
|**add**(lhs,rhs)|自动广播加法。|
|**adv_index**(data,indices)|使用张量进行 Numpy 样式索引。|
|**all**(data[,axis,keepdims])|对给定轴或轴列表上的数组元素进行逻辑与。|
|**any**(data[,axis,keepdims])|对给定轴或轴列表上的数组元素进行逻辑或。|
|**arange**(start[,stop,step,dtype])|创建在给定间隔内具有均匀分布值的张量。|
|**argmax**(data[,axis,keepdims,select_last_index])|返回沿轴的最大值的索引。|
|**argmin**(data[,axis,keepdims,select_last_index])|返回沿轴的最小值的索引。|
|**argsort**(data[,valid_count,axis,…])|沿给定轴执行排序，并返回与按排序顺序索引数据的输入数组具有相同形状的索引数组。|
|**asin**(x)|对输入 x 取反正弦值。|
|**asinh**(x)|对输入 x 取反正弦值。|
|**atan**(x)|对输入 x 取正切值。|
|**atanh**(x)|对输入 x 进行 atanh 处理。|
|**binary_search**(ib,sequence_offset,…)|CPU 和 GPU 后端使用的二进制搜索的通用 IR 生成器。|
|**bitwise_and**(lhs,rhs)|逐元素按位计算数据的「与」。|
|**bitwise_not**(data)|逐元素按位计算数据的「非」。|
|**bitwise_or**(lhs,rhs)|逐元素按位或计算数据。|
|**bitwise_xor**(lhs,rhs)|计算数据的逐元素按位异或。|
|**broadcast_to**(data,shape)|将 src 广播到目标形状。|
|**cast**(x,dtype[,span])|将输入转换为指定的数据类型。|
|**ceil**(x)|取输入 x 的上限。|
|**ceil_log2**(x)|使用 Vulkan 的特殊代码路径计算整数 ceil log2。SPIR-V 不支持 fp64 上的 log2。因此，当目标平台为 Vulkan 时，我们通过 clz 内在函数计算整数 ceil_log2。|
|**clip**(x,a_min,a_max)|裁剪（限制）数组中的值。给定一个区间，区间之外的值将被裁剪到区间边缘。|
|**collapse_sum**(data,target_shape)|将数据总和返回到给定的形状。|
|**concatenate**(a_tuple[,axis])|沿现有轴连接一系列数组。|
|**const_vector**(vector[,name])|将 const numpy 一维向量转换为 TVM 张量。|
|**cos**(x)|取输入 x 的 cos。|
|**cosh**(x)|取输入 x 的 cosh。|
|**cumprod**(data[,axis,dtype,exclusive])|  Numpy 风格的 cumprod op。|
|**cumsum**(data[,axis,dtype,exclusive])|Numpy 风格的 cumsum op。|
|**decl_buffer**(shape[,dtype,name,data,…])|声明一个新的符号缓冲区。|
|**dft**(re_data,im_data,inverse)|计算输入的离散傅里叶变换（沿最后一个轴计算）。这将给出信号随时间变化的频率分量。|
|**div**(a,b[,span])|按照 C/C++ 语义计算 a / b。|
|**divide**(lhs,rhs)|自动广播分工。|
|**dynamic_strided_slice**(a,begin,end,…)|数组的切片。|
|**einsum**(subscripts,*operand)|评估操作数的爱因斯坦求和约定。|
|**elemwise_sum**(xs)|对输入执行元素求和。|
|**equal**(lhs,rhs)|使用自动广播计算 (lhs==rhs)。|
|**erf**(x)|取输入 x 的高斯误差函数。|
|**exp**(x)|取输入 x 的指数。|
|**expand_dims**(a,axis[,num_newaxis])|扩展数组的形状。|
|**expand_like**(a,shape_like,axis)|将输入数组扩展为第二个数组的形状。此操作始终可以由对未压缩轴上的维度进行解压缩和扩展组成。|
|**extern**(shape,inputs,fcompute[,name,…])|通过外部函数计算多个张量。|
|**eye**(n[,m,k,dtype])|生成一个单位矩阵或第 k 个对角线为 1 的矩阵。|
|**fast_erf**(x)|使用 fast_erf 实现获取输入 x 的高斯误差函数。|
|**fast_exp**(x)|使用 fast_exp 实现对输入 x 进行指数运算。|
|**fast_tanh**(x)|使用 fast_tanh 实现对输入 x 进行双曲正切。|
|**fixed_point_multiply**(x,multiplier,shift)|数据与定点常数之间的定点乘法表示为乘数 * 2^(-shift)，其中乘数是一个具有 31 个小数位的 Q 数。|
|**fixed_point_multiply_per_axis**(x,y,lshift,…)|数据与定点常数之间的定点乘法表示为乘数 * 2^(-shift)，其中乘数是一个具有 31 个小数位的 Q 数。|
|**flip**(a[,axis])|在特定轴上翻转/反转数组的元素。|
|**floor**(x)|取输入 x 的底数。|
|**floor_divide**(lhs,rhs)|自动广播的楼层划分。|
|**floor_mod**(lhs,rhs)|自动广播的楼层模量。|
|**floordiv**(a,b[,span])|计算两个表达式的 floordiv。|
|**floormod**(a,b[,span])|计算两个表达式的 floormod。|
|**full**(shape,dtype,fill_value)|用 fill_value 填充张量。|
|**full_like**(x,fill_value)|构造与输入张量形状相同的张量。|
|**gather**(data,axis,indices)|从给定的索引沿给定的轴收集值。|
|**gather_nd**(a,indices[,batch_dims])|从 n 维数组中收集元素..|
|**get_const_tuple**(in_tuple)|验证输入元组是 IntImm 还是 Var，返回 int 或 Var 的元组。|
|**greater**(lhs,rhs)|使用自动广播计算 (lhs>rhs)|
|**greater_equal**(lhs,rhs)|使用自动广播计算 (lhs>=rhs)|
|**hamming_window**(window_size,periodic,alpha,…)|汉明窗函数。|
|**identity**(x)|取输入 x 的恒等式。|
|**index_put**(data,indices,values[,accumulate])|根据索引将值放入数组中。|
|**index_tensor**(data,indices)|高级张量索引（NumPy/PyTorch 风格）。|
|**isfinite**(x)|检查 x 的值是否是有限的、元素有限的。|
|**isinf**(x)|检查 x 的值是否为无限的（按元素）。|
|**isnan**(x)|逐个元素检查 x 的值是否为 NaN。|
|**layout_transform**(array,src_layout,dst_layout)|根据 src_layout 和 dst_layout 转换布局。|
|**left_shift**(lhs,rhs)|左移并自动广播。|
|**less**(lhs,rhs)|使用自动广播计算 (lhs<rhs)。|
|**less_equal**(lhs,rhs)|使用自动广播计算 (lhs<=rhs)。|
|**log**(x)|对输入 x 取对数。|
|**log10**(x)|对输入 x 取以 10 为底的对数。|
|**log2**(x)|对输入 x 取以 2 为底的对数。|
|**log_add_exp**(lhs,rhs)|自动广播的对数和指数运算。|
|**logical_and**(lhs,rhs)|计算元素级别的数据的逻辑与。|
|**logical_not**(data)|逐元素计算数据的逻辑非。|
|**logical_or**(lhs,rhs)|计算元素的逻辑或数据。|
|**logical_xor**(lhs,rhs)|计算数据的元素逻辑异或。|
|**make_idx**(b,e,s,z,i)|返回与完整数组中的数组位置相对应的选择中的数组位置。|
|**matmul**(a,b[,transp_a,transp_b])|创建一个计算矩阵乘法（行主表示法）的运算：如果 trans_a == trans_b，则为 A(i, k) * B(k, j)，否则为通常的转置组合。|
|**matrix_set_diag**(data,diagonal[,k,align])|返回一个张量，其中输入张量的对角线被提供的对角线值替换。|
|**max**(data[,axis,keepdims])|给定轴或轴列表上的数组元素的最大值。|
|**maximum**(lhs,rhs)|使用自动广播，逐个元素地取两个张量的最大值。|
|**meshgrid**(a_tuple,indexing)|从坐标向量创建坐标矩阵。|
|**min**(data[,axis,keepdims])|给定轴或轴列表上的数组元素的最小值。|
|**minimum**(lhs,rhs)|使用自动广播，逐个元素地取两个张量的最大值。|
|**mod**(lhs,rhs)|自动广播模块。|
|**multiply**(lhs,rhs)|自动广播乘法。|
|**ndarray_size**(array[,dtype])|获取输入数组元素的数量。|
|**negative**(x)|对输入 x 取否定。|
|**not_equal**(lhs,rhs)|使用自动广播计算 (lhs!=rhs)。|
|**one_hot**(indices,on_value,off_value,depth,…)|返回一个独热张量，其中索引所代表的位置的值为 on_value，其他位置的值为 off_value。最终维度为 <索引外维度> x 深度 x <索引内维度>。|
|**power**(lhs,rhs)|自动广播幂方。|
|**prod**(data[,axis,keepdims])|给定轴或轴列表上的数组元素的乘积。|
|**reinterpret**(x,dtype)|将输入重新解释为指定的数据类型。|
|**repeat**(a,repeats,axis)|重复数组的元素。|
|**reshape**(a,newshape)|重塑数组。|
|**reverse_sequence**(a,seq_lengths[,seq_axis,…])|将张量反转为可变长度切片。输入首先沿批处理轴进行切片，然后沿序列轴反转元素。|
|**right_shift**(lhs,rhs)|右移并自动广播。|
|**round**(x)|将 x 的元素四舍五入为最接近的整数。|
|**rsqrt**(x)|取输入 x 的平方根的倒数。|
|**scanop**(data,binop,identity_value,op_name)|累积二元运算符（扫描），其轴行为与 np.cumsum 和 np.cumprod 类似。|
|**scatter_elements**(data,indices,updates[,…])|将更新中的元素分散到复制数据的相应索引中。|
|**scatter_nd**(data,indices,updates,mode)|从 n 维数组中分散元素。|
|**searchsorted**(sorted_sequence,values[,…])|查找应插入元素以维持顺序的索引。|
|**sequence_mask**(data,valid_length[,…])|将序列预期长度之外的所有元素设置为常量值。|
|**shape**(array[,dtype])|获取输入数组的形状。|
|**sigmoid**(x)|对输入 x 进行 S 型 tanh 运算。|
|**sign**(x)|根据 x 的符号返回 -1、0、1。|
|**sin**(x)|对输入 x 取正弦值。|
|**sinh**(x)|对输入 x 取 sinh。|
|**slice_scatter**(input_tensor,src,start,end,…)|将 src 的切片沿给定轴（SSA 形式）分散到输入中。|
|**sliding_window**(data,axis,window_shape,strides)|在数据张量上滑动一个窗口。|
|**sort**(data[,axis,is_ascend])|沿给定轴执行排序并按排序顺序返回数组。|
|**sparse_reshape**(sparse_indices,prev_shape,…)|重塑稀疏张量。|
|**sparse_to_dense**(sparse_indices,…[,…])|将稀疏表示转换为密集张量。|
|**split**(ary,indices_or_sections[,axis])|将数组拆分为多个子数组。|
|**sqrt**(x)|对输入 x 取平方根。|
|**squeeze**(a[,axis])|从数组形状中删除一维条目。|
|**stack**(tensors[,axis])|沿新轴连接一系列张量。|
|**stft**(data,n_fft,hop_length,win_length,…)|STFT 计算输入短重叠窗口的傅里叶变换。这给出了信号随时间变化的频率分量。:param data: 一维张量或二维批量张量。:type data: te.Tensor :param n_fft: 傅里叶变换的大小 :type n_fft: int :param hop_length: 相邻滑动窗口帧之间的距离 :type hop_length: int :param win_length: 窗口帧和 STFT 滤波器的大小 :type win_length: int :param window: 一维张量窗口帧 :type window: te.Tensor :param normalized: 是否返回归一化的 STFT 结果 :type normalized: bool :param onesided: 是否返回单侧结果或使用共轭对称性填充 :type onesided: bool。|
|**strided_set**(a,v,begin,end[,strides])|设置数组的切片。|
|**strided_slice**(a,begin,end[,strides,…])|数组的切片。|
|**subtract**(lhs,rhs)|自动广播减法。|
|**sum**(data[,axis,keepdims])|给定轴或轴列表上的数组元素的总和。|
|**take**(a,indices[,axis,batch_dims,mode])|沿轴从数组中获取元素。|
|**tan**(x)|对输入 x 取 tan。|
|**tanh**(x)|对输入 x 取双曲 tanh。|
|**tensordot**(a,b,axes)|矩阵乘法到张量的推广。|
|**tile**(a,reps)|重复整个数组多次。|
|**topk**(data[,k,axis,ret_type,is_ascend,dtype])|获取输入张量中沿给定轴的前 k 个元素。|
|**transpose**(a[,axes])|排列数组的维度。|
|**trilu**(data,k,upper)|给定一个二维矩阵或一批二维矩阵，返回张量的上三角或下三角部分。|
|**trunc**(x)|逐个元素地取 x 输入的截断值。|
|**unravel_index**(indices,shape)|将平面索引或平面索引数组转换为坐标数组的元组。|
|**where**(condition,x,y)|根据条件从 x 或 y 获取元素。|
|**within_index**(b,e,s,i)|返回一个布尔值，指示 i 是否在给定索引内。|

**异常：**

|**InvalidShapeError**|topi 函数的形状无效。|
|:----|:----|

## *class* tvm.topi.Analyzer 


整数算术分析器。


这是一个有状态的分析器类，可用于执行各种符号整数分析。


**方法：**

|**const_int_bound**(expr)|为 expr 找到常量整数边界。|
|:----|:----|
|**const_int_bound_is_bound**(var)|检查一个变量是否绑定到某个范围。|
|**modular_set**(expr)|找到一个模集，该表达式属于该模集。|
|**simplify**(expr[, steps])|通过重写和规范化简化表达式。|
|**rewrite_simplify**(expr)|通过重写规则简化表达式。|
|**canonical_simplify**(expr)|通过规范化简化表达式。|
|**int_set**(expr, dom_map)|计算一个符号 IntSet，该集合覆盖了在 dom_map 中所有值下的 expr。|
|**can_prove**(expr[, strength])|检查我们是否能够证明 expr 为真。|
|**bind**(var, expr)|将变量绑定到表达式。|
|**constraint_scope**(constraint)   **constraint_scope** (约束)|创建一个约束作用域。|
|**update**(var, info[, override])|更新关于 var 的信息。|
|**can_prove_equal**(lhs, rhs)|我们能否证明 lhs == rhs。|


**属性：**

|**enabled_extensions**|返回当前启用的扩展。|
|:----|:----|


### const_int_bound（*expr ：*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)  ） → ConstIntBound


查找 expr 的常数整数界限。
* **参数：expr** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：表达式。
* **返回：bound**：结果边界。
* **返回类型：** ConstIntBound。

### const_int_bound_is_bound(*var:*[Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)) → [bool](https://docs.python.org/3/library/functions.html#bool)



检查变量是否绑定到某个范围。
* **参数：var** ([tvm.tir.Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none))**：** 变量。
* **返回：result**：变量是否绑定到某个范围。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。


### **modular_set(*expr:***[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)**)→ ModularSet**


找到 expr 所属的模集。
* **参数：expr** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：表达式。
* **返回：result**：结果。
* **返回类型：** ModularSet。


### **simplify(*expr:***[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)**,*steps:***[int](https://docs.python.org/3/library/functions.html#int)***= 2*)→**[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


通过重写和规范化来简化表达。
* **参数：**
   * **expr** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：表达式。
   * **步骤**（*简化按以下顺序运行*）：rewrite_simplify（步骤 1）–> canonical_simplify（步骤 2）–> rewrite_simplify（步骤 3）–> canonical_simplify（步骤 4）–> … 参数 steps 控制运行步骤数。默认值为 2，即 rewrite_simplify + canonical_simplify。
* **返回：result**：结果。
* **返回类型：** Expr。


### rewrite_simplify（*expr ：*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) ） → [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


通过重写规则来简化表达。
* **参数：expr** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))**：** 表达式。
* **返回：result**：结果。
* **返回类型：** Expr。

### canonical_simplify(*expr:*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)) → [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)


通过规范化来简化表达。
* **参数：expr** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：表达式。
* **返回：result** *：* 结果。
* **返回类型：** Expr。


### int_set（*expr ：*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)  ， *dom_map ：*[dict](https://docs.python.org/3/library/stdtypes.html#dict)*[*[Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*， IntSet ]* ） → IntSet


计算一个符号 IntSet，覆盖 dom_map 中所有值的 expr。
* **参数：**
   * **expr** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：表达式。
   * **dom_map** ( *Dict[*[tvm.tir.Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,tvm.arith.IntSet]* )：要放宽的变量的域。
* **返回：result** ：结果。
* **返回类型：** IntSet。


### can_prove（*expr ：*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) *，strength ：ProofStrength = ProofStrength.DEFAULT* ） → [bool](https://docs.python.org/3/library/functions.html#bool)


检查我们是否可以证明 expr 为真。
* **参数：**
   * **expr** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：表达式。
   * **强度**（*ProofStrength*）**：** 证明强度。
* **返回：result** ：结果。
* **返回类型：** Expr。


### **bind(*var:***[Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)**,*expr:***[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)***|***[Range](/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)**)→**[None](https://docs.python.org/3/library/constants.html#None)


将变量绑定到表达式。
* **参数：**
   * **var** ( [tvm.tir.Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none) )：变量。
   * expr（Union [tir.PrimExpr，[ir.Range](/docs/api-reference/python-api/tvm-ir#class-tvmirrangebeginprimexpr-endprimexprnone-none-spanspannone-none)]）**：** 要绑定到的表达式或范围。


### **constraint_scope(*constraint:***[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)**)→ ConstraintScope**

创建约束范围。
* **参数：**
   * **constraint** ([PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr))：约束表达式。
* **返回：scope**：约束范围。
* **返回类型：** ConstraintScope。


**示例**

```python
x = te.var("x")
analyzer = tvm.arith.Analyzer()
with analzyer.constraint_scope(x % 3 == 0):
    # 约束生效中
    assert analyzer.modular_set(x).coeff == 3
# 约束不再生效
assert analyzer.modular_set(x).coeff != 3

update(*var: [Var](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.Var"tvm.tir.expr.Var""tvm.tir.expr.Var")*, *info: ConstIntBound*, *override: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") = False*) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")

Update infomation about var
```
### **update(*var:***[Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)**,*info: ConstIntBound*,*override:***[bool](https://docs.python.org/3/library/functions.html#bool)***= False*)→**[None](https://docs.python.org/3/library/constants.html#None)

更新有关 var 的信息。
* **参数：**
   * **var** ( [tvm.tir.Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none) )：变量。
   * **info**（*tvm.Object*）：相关信息。
   * **override** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：是否允许覆盖。

### can_prove_equal(*lhs:*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr), *rhs:*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)) → [bool](https://docs.python.org/3/library/functions.html#bool)



我们是否可以证明 lhs == rhs
* **参数：**
   * **lhs** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：比较的左侧。
   * **rhs** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：比较的右侧。
* **返回：result**：我们是否可以证明 lhs == rhs。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

### *property* enabled_extensions*: Extension*


返回当前启用的扩展。

## *class* tvm.topi.Cast(*dtype*, *value*, *span:*[Span](/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) 


转换表达式。
* **参数：**
   * **dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：数据类型。
   * **值**（[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)）：函数的值。
   * *span*（*可选*[[Span](/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此表达式在源代码中的位置。

## *class* tvm.topi.PrimExpr 


所有原始表达式的基类。


PrimExpr 用于低级代码优化和整数分析。

## tvm.topi.abs(*x*) 


逐个元素地取 x 输入的绝对值。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)

## tvm.topi.acos(*x*) 


取输入 x 的反余弦值。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.acosh(*x*) 


取输入 x 的反余弦值。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** *：* 结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.add(*lhs*, *rhs*) 


自动广播加法
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.adv_index(*data*, *indices*) 


使用张量进行 Numpy 样式索引。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入数据。
   * **indices**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*列表）：张量*索引。
* **返回：result**：输出张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.all(*data*, *axis=None*, *keepdims=False*) 


对给定轴或轴列表上的数组元素进行逻辑与。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入 tvm 布尔张量。
   * **axis**（*None、*[int](https://docs.python.org/3/library/functions.html#int)*或*[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：执行逻辑与运算的轴。默认值 axis=None，表示对输入数组的所有元素执行逻辑与运算。如果 axis 为负数，则从最后一个轴计数到第一个轴。
   * **keepdims** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果设置为 True，则缩小的轴将保留在结果中，作为大小为 1 的维度。使用此选项，结果将根据输入数组正确广播。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.any(*data*, *axis=None*, *keepdims=False*) 


对给定轴或轴列表上的数组元素进行逻辑或
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入 tvm 布尔张量。
   * **axis**（*None、*[int](https://docs.python.org/3/library/functions.html#int)*或*[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：执行逻辑或运算的轴。默认值 axis=None，将对输入数组的所有元素执行逻辑或运算。如果 axis 为负数，则从最后一个轴计数到第一个轴。
   * **keepdims** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果设置为 True，则缩小的轴将保留在结果中，作为大小为 1 的维度。使用此选项，结果将根据输入数组正确广播。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.arange(*start*, *stop=None*, *step=1*, *dtype='float32'*) 


创建在给定间隔内具有均匀分布值的张量。
* **参数：**
   * *start*（*tvm.Expr*，*可选*）：区间的起始值。区间包含此值。默认起始值为 0。
   * **stop** ( *tvm.Expr* )：区间停止。区间不包含此值。
   * *step*（*tvm.Expr*，*可选*）：值之间的间距。默认步长为 1。
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：目标数据类型。
* **返回：result** **：** 结果张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.argmax(*data*, *axis=None*, *keepdims=False*, *select_last_index=False*) 


返回沿轴的最大值的索引。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入 tvm 张量。
   * **axis**（*None、*[int](https://docs.python.org/3/library/functions.html#int)*或*[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：执行 argmax 运算的一个或多个轴。默认值 axis=None 将查找输入数组元素中最大元素的索引。如果 axis 为负数，则从最后一个轴计数到第一个轴。
   * **keepdims** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果设置为 True，则缩小的轴将保留在结果中，作为大小为 1 的维度。使用此选项，结果将根据输入数组正确广播。
   * **select_last_index** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果最大元素出现多次，是否选择最后一个索引，否则选择第一个索引。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.argmin(*data*, *axis=None*, *keepdims=False*, *select_last_index=False*) 


返回沿轴的最小值的索引。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 输入 tvm 张量。
   * **axis**（*None、*[int](https://docs.python.org/3/library/functions.html#int)*或*[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：执行 argmin 操作的轴。默认值 axis=None，将查找输入数组所有元素中最小元素的索引。如果 axis 为负数，则从最后一个轴计数到第一个轴。
   * **keepdims** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果设置为 True，则缩小的轴将保留在结果中，作为大小为 1 的维度。使用此选项，结果将根据输入数组正确广播。
   * **select_last_index** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果最小元素出现多次，是否选择最后一个索引，否则选择第一个索引。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.argsort(*data*, *valid_count=None*, *axis=-1*, *is_ascend=1*, *dtype='float32'*) 


沿给定轴执行排序，并返回与按排序顺序索引数据的输入数组具有相同形状的索引数组。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 输入张量。
   * **valid_count**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*）**：** 有效框数量的一维张量。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：对输入张量进行排序的轴。默认情况下使用扁平数组。
   *   *is_ascend**（*布尔值**，*可选*）：按升序还是降序排序。
   *   *dtype**（*字符串**，*可选*）**：** 输出索引的 DType。
* **返回：out** ：排序索引张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
# 使用 argsort 的示例
dshape = (1, 5, 6)
data = te.placeholder(dshape, name="data")
axis = 0
is_ascend = False
out = argsort(data, axis=axis, is_ascend=is_ascend)
np_data = np.random.uniform(dshape)
s = topi.generic.schedule_argsort(out)
f = tvm.compile(s, [data, out], "llvm")
dev = tvm.cpu()
tvm_data = tvm.runtime.tensor(np_data, dev)
tvm_out = tvm.runtime.tensor(np.zeros(dshape, dtype=data.dtype), dev)
f(tvm_data, tvm_out)
```
## tvm.topi.asin(*x*) 


对输入 x 取反正弦值。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.asinh(*x*) 


对输入 x 取反正弦值。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) **：** 输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.atan(*x*) 


对输入 x 取正切值。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.atanh(*x*) 


对输入 x 进行 atanh 处理。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.binary_search(*ib*, *sequence_offset*, *search_range*, *sorted_sequence*, *value*, *right*, *out_dtype*) 


CPU 和 GPU 后端使用的二进制搜索的通用 IR 生成器。


sorted_sequence 是一个 ND 缓冲区，我们要在其最内层维度中搜索值，search_range 是最内层维度的大小。sequence_offset 是一个一维线性偏移量，指定要搜索哪个最内层序列。


因此，对值的搜索是在 sorted_sequence[sequence_offset:(sequence_offset + search_range)]上进行的。请注意，我们通过一维线性化索引对 ND 缓冲区进行索引。

## tvm.topi.bitwise_and(*lhs*, *rhs*) 


逐元素按位计算数据的「与」。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret** ：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.bitwise_not(*data*) 


逐元素按位计算数据的「非」。
* **参数：数据**([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*orExpr*)。
* **返回：ret** ：如果操作数是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.bitwise_or(*lhs*, *rhs*) 


逐元素按位或计算数据。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret** ：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.bitwise_xor(*lhs*, *rhs*) 


计算数据的逐元素按位异或。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）*：* 左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret** ：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.broadcast_to(*data*, *shape*) 


将 src 广播到目标形状。


我们遵循 numpy 广播规则。另请参阅[https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入数据。
   * **shape**（[列表](https://docs.python.org/3/library/stdtypes.html#list)*或*[元组](https://docs.python.org/3/library/stdtypes.html#tuple)）**：** 要广播的目标形状。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.cast(*x*, *dtype*, *span=None*) 


将输入转换为指定的数据类型。
* **参数：**
   * **x**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：输入参数。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str))[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)数据类型。
   * *span**（*可选**[ [Span](/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：源中演员的位置。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.ceil(*x*) 


取输入 x 的上限。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.ceil_log2(*x*) 


使用 Vulkan 的特殊代码路径计算整数 ceil log2。SPIR-V 不支持 fp64 上的 log2。因此，当目标平台为 Vulkan 时，我们通过 clz 内在函数计算整数 ceil_log2。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.clip(*x*, *a_min*, *a_max*) 


裁剪（限制）数组中的值。给定一个区间，区间之外的值将被裁剪到区间边缘。
* **参数：**
   * **x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)输入参数。
   * **a_min** (*tvm.tir.PrimExpr*) *：* 最小值。
   * **a_max** (*tvm.tir.PrimExpr*)：最大值。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.collapse_sum(*data*, *target_shape*) 


将数据总和返回到给定的形状。


crash_sum 旨在作为自动微分过程中 topi 广播操作符的后向操作符。


我们期望数据是通过某些广播操作广播某个 target_shape 的张量的结果。因此 target_shape 和 data.shape 必须遵循广播规则。


计算过程中，data.shape 和 target_shape 的轴会从右到左进行检查。对于每个轴，如果满足以下任一条件：- 存在于数据中但不存在于 target_shape 中；或 - 在数据中大于 1 且在 target_shape 中等于 1，则数据将在该轴上进行求和。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入张量。
   * **shape** ([Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*)：要折叠的形状。
* **返回：ret**：求和后的结果张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.concatenate(*a_tuple*, *axis=0*) 


沿现有轴连接一系列数组。
* **参数：**
   * **a_tuple**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*的*[元组](https://docs.python.org/3/library/stdtypes.html#tuple)）：要连接的数组。
   * **axis** （[int](https://docs.python.org/3/library/functions.html#int)*， 可选*）：数组将沿其连接的轴。默认值为 0。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.const_vector(*vector*, *name='const_vector'*) 

将一个常量 numpy 一维向量转换为 tvm 张量。
* **参数：**
   * **vector** (*numpy.ndarray*)：常量输入数组。
   * **name** (str, 可选)**：** 输出操作的名称。
* **返回：tensor** ：创建的 tensor。
* **返回类型：**[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.cos(*x*) 

对输入 x 取余弦。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor))[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)输入参数。
* **返回：y**：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.cosh(*x*) 

对输入 x 取双曲余弦。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor))*：* 输入参数。
* **返回：y** *：* 结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.cumprod(*data:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *dtype:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *exclusive:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) 

Numpy 风格的累积乘积操作。返回沿给定轴的元素的累积乘积。
* **参数：**
   * **data** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：操作的输入数据。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)：计算累积乘积的轴。默认（None）是计算展平数组的累积乘积。
   * **dtype** (*string,optional*)：返回数组的类型以及元素相乘的累加器的类型。如果未指定 dtype，则默认为 data 的 dtype。
   * **exclusive** ([bool](https://docs.python.org/3/library/functions.html#bool)*,optional*) ：如果为 True，将返回一个排他性乘积，其中第一个元素不包含在内。换句话说，如果为 True，第 j 个输出元素将是前 (j-1) 个元素的乘积。否则，将是前 j 个元素的乘积。 
* **返回：result** ：结果与 data 具有相同的大小，如果 axis 不是 None，则形状也相同。如果 axis 是 None，结果是一个一维数组。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.cumsum(*data:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *exclusive:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) 

Numpy 风格的 cumsum 操作。返回沿给定轴的元素累积和。
* **参数：**
   * **data** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：运算符的输入数据。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)：沿着哪个轴计算累积和。默认值（None）是计算扁平化数组的 cumsum。
   * **dtype** (*string,optional*) ：返回数组的类型以及用于累加元素的累加器的类型。如果未指定 dtype，则默认为 data 的 dtype。
   * **exclusive** ([bool](https://docs.python.org/3/library/functions.html#bool)*,optional*) ：如果为 True，将返回排他性求和，其中第一个元素不包含在内。换句话说，如果为 True，第 j 个输出元素将是前 (j-1) 个元素的和。否则，它将是前 j 个元素的和。
* **返回：result**：结果与 data 的大小相同，如果 axis 不是 None，则形状也与 data 相同。如果 axis 是 None，结果是一个 1 维数组。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)

## tvm.topi.decl_buffer(*shape*, *dtype=None*, *name='buffer'*, *data=None*, *strides=None*, *elem_offset=None*, *scope=''*, *data_alignment=-1*, *offset_factor=0*, *buffer_type=''*, *axis_separators=None*, *span=None*) 

声明一个新的符号缓冲区。

通常在降低和构建过程中会自动创建缓冲区。只有当用户想要指定自己的缓冲区布局时才需要这样做。

有关缓冲区使用的详细讨论，请参阅下方的注释。
* **参数：**
   * **shape** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*ofExpr*) ：缓冲区的形状。
   * **dtype** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：缓冲区的数据类型。
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：缓冲区的名称。
   * **data** ([tir.Var](/docs/api-reference/python-api/tvm-tir#class-tvmtirvarnamestr-dtypestrtype-spanspannone-none)*,optional*) *：* 缓冲区中的数据指针。
   * **strides** (*arrayofExpr*)：缓冲区的步长。
   * **elem_offset** (*Expr,optional*) ：数组到数据的起始偏移量。以 dtype 元素数量为单位。
   * **scope** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional*)：缓冲区的存储范围，如果不是全局的。如果 scope 等于空字符串，表示它是全局内存。
   * **data_alignment** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*) ：数据指针的字节对齐方式。如果传入-1，对齐方式将设置为 TVM 的内部默认值。
   * **offset_factor** ([int](https://docs.python.org/3/library/functions.html#int)*,optional*)：elem_offset 字段的因子，当设置时，elem_offset 必须是 offset_factor 的倍数。如果传入 0，对齐将被设置为 1。如果传入非零值，当 elem_offset 不为 None 时，我们将为 elem_offset 创建一个 Var。
   * **buffer_type** ([str](https://docs.python.org/3/library/stdtypes.html#str)*,optional,* ***{""****,"auto_broadcast"}*)*：* auto_broadcast buffer 允许在不考虑维度大小是否等于一的情况下实现广播计算。TVM 将 buffer[i][j][k] 映射到 buffer[i][0][k]，如果维度 j 的形状等于 1。
   * **axis_separators** ([list](https://docs.python.org/3/library/stdtypes.html#list)*of*[int](https://docs.python.org/3/library/functions.html#int)*,optional*) ：如果传入，则是一个分隔轴组的列表，每个组将被展平为一个输出轴。对于扁平内存空间，应该是 None 或一个空列表。
   * **span** (*Optional[*[Span](/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*) ：在源代码中创建 decl_buffer 的位置。
* **返回：buffer** ：创建的缓冲区。
* **返回类型：**[tvm.tir.Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)。

:::Note

Buffer 数据结构反映了 dlpack 中的 DLTensor 结构。虽然 DLTensor 数据结构非常通用，但通常创建只处理特定数据结构情况的函数是有帮助的，这样编译后的函数可以从中受益。

如果用户在构建函数时传递了步长（strides）并且 elem_offset 传递为 None，那么该函数将针对紧凑且对齐的 DLTensor 进行特化。如果用户向步长传递一个完全通用的符号数组，那么生成的函数将变为完全通用的。

:::

## tvm.topi.dft(*re_data:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *im_data:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *inverse:*[IntImm](/docs/api-reference/python-api/tvm-tir#class-tvmtirintimmdtypestr-valueint-spanspannone-none)) 


计算输入的离散傅里叶变换（沿最后一个轴计算）。这将给出信号随时间变化的频率分量。
* **参数：**
   *  **re_data**（[te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)ND 张量，输入信号的实部。
   * **im_data** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：ND 张量，输入信号的虚部。如果信号为实数，则该张量的值为零。
   * **inverse**（[bool](https://docs.python.org/3/library/functions.html#bool)）：是否执行逆离散傅里叶变换。
* **返回：**
   *  **re_output** (*te.Tensor*)：输入的傅里叶变换（实部）。
   *  im_output (te.Tensor)*：* 输入的傅里叶变换（虚部）。



## tvm.topi.div(*a*, *b*, *span=None*) 


按照 C/C++ 语义计算 a / b。
* **参数：**
   * **a** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：左侧操作数，已知为非负数。
   * **b** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：右侧操作数，已知为非负。
   * *span**（*可选**[ [Span](/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：res** ：结果表达式。
* **返回类型：**[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

:::Note

当操作数为整数时，返回 truncdiv(a, b, span)。

:::

## tvm.topi.divide(*lhs*, *rhs*) 


自动广播分工。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret** ：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.dynamic_strided_slice(*a*, *begin*, *end*, *strides*, *output_shape*) 


数组的切片。
* **参数：**
   * **a** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：要切片的张量。
   * **begin** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)切片中开始的索引。
   * **end** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) *：* 指示切片结束的索引。
   * **strides** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor))：指定步幅值，在这种情况下它可以为负，输入张量将在该特定轴上反转。
   * **output_shape**（*PrimExpr*[列表）](https://docs.python.org/3/library/stdtypes.html#list)[–](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)指定输出形状。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.einsum(*subscripts*, operand*) 


评估操作数的爱因斯坦求和约定。
* **参数：**
   * **subscripts**（*字符串*）：将求和的下标指定为以逗号分隔的下标标签列表。除非包含显式指示符“–>”以及精确输出形式的下标标签，否则将执行隐式（经典爱因斯坦求和）计算。
   * **a_tuple**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*的*[元组](https://docs.python.org/3/library/stdtypes.html#tuple)）：这些是用于运算的张量。tvm 和 numpy 中 einsum 的唯一区别在于，einsum 需要额外的括号来表示张量。例如，topi.einsum(“ij, jk –> ik”, (A, B))。
* **返回：out**[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)基于爱因斯坦求和约定的计算。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.elemwise_sum(*xs*) 


对输入执行元素求和。
* **参数：xs** ([list](https://docs.python.org/3/library/stdtypes.html#list)*of*[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) *：*–] "tvm.te.张量")输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.equal(*lhs*, *rhs*) 


使用自动广播计算 (lhs==rhs)
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）**：** 左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.erf(*x*) 


取输入 x 的高斯误差函数。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** **：** 结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.exp(*x*) 


取输入 x 的指数。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) **：** 输入参数。
* **返回：y** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.expand_dims(*a*, *axis*, *num_newaxis=1*) 


扩展数组的形状。
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 要扩展的张量。
   * **num_newaxis** ( [int](https://docs.python.org/3/library/functions.html#int)*，可选*)：要在轴上插入的新轴的数量。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.expand_like(*a*, *shape_like*, *axis*) 


将输入数组扩展为第二个数组的形状。此操作始终可以由对未压缩轴上的维度进行解压缩和扩展组成。


**示例**

```python
input = [ 12.  19.  27.]
input.shape = (3,)

new_shape_array = [[[1,2],[2,3],[1,3]],
                [[1,4],[4,3],[5,2]],
                [[7,1],[7,2],[7,3]]]
new_shape_array.shape = (3, 3, 2)

expand_like(input, [1,2], new_shape_array) =
                [[[12,12],[12,12],[12,12]],
                [[19,19],[19,19],[19,19]],
                [[27,27],[27,27],[27,27]]]
```
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 要扩展的张量。
   * **shape_like**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)具有目标形状的张量。
   * **axis**（*int*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：要扩展的[轴](https://docs.python.org/3/library/functions.html#int)。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.extern(*shape*, *inputs*, *fcompute*, *name='extern'*, *dtype=None*, *in_buffers=None*, *out_buffers=None*, *tag=''*, *attrs=None*) 


通过外部函数计算多个张量。
* **参数：**
   * **shape**（[元组](https://docs.python.org/3/library/stdtypes.html#tuple)*或元组*[列表](https://docs.python.org/3/library/stdtypes.html#list)*。*）：输出的形状。
   * **输入**（*Tensor*[列表）](https://docs.python.org/3/library/stdtypes.html#list)[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)输入。
   * *fcompute**（*输入**的**lambda 函数**，*输出–> stmt*）：指定用于执行计算的 IR 语句。请参阅以下注释以了解 fcompute 的函数签名。

:::Note
* **参数:**
   * **ins** (list of [tvm.tir.Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)) ：每个输入的占位符
   * **outs** (list of [tvm.tir.Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)) ：每个输出的占位符
* **返回：stmt** ([tvm.tir.Stmt](/docs/api-reference/python-api/tvm-tir#class-tvmtirstmt)) ：执行数组计算的语句。 

:::
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）**：** 张量的名称提示
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)[列表](https://docs.python.org/3/library/stdtypes.html#list)*，**可选）***[：](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)***输出的数据类型，默认****情况*下 dtype 与输入相同。
   * **in_buffers**（[tvm.tir.Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*或**tvm.tir.Buffer***[列表](https://docs.python.org/3/library/stdtypes.html#list)***，****可选*）***：*** 输入缓冲区[。](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)
   * **out_buffers**（[tvm.tir.Buffer](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)*或**tvm.tir.Buffer***[列表](https://docs.python.org/3/library/stdtypes.html#list)***，*** *可选*）：输出缓冲区[。](/docs/api-reference/python-api/tvm-tir#class-tvmtirbuffer)


**tag: str, optional**


有关计算的附加标记信息。

**attrs: dict, optional**


有关计算的其他辅助属性。
* **返回：tensor**：创建的张量或张量元组包含多个输出。
* **返回类型：**[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or [list](https://docs.python.org/3/library/stdtypes.html#list) of Tensors


**示例**


在下面的代码中，C 是通过调用外部 PackedFunc tvm.contrib.cblas.matmul 生成的。

```python
A = te.placeholder((n, l), name="A")
B = te.placeholder((l, m), name="B")
C = te.extern((n, m), [A, B],
               lambda ins, outs: tvm.tir.call_packed(
                  "tvm.contrib.cblas.matmul",
                    ins[0], ins[1], outs[0], 0, 0), name="C")
```
## tvm.topi.eye(*n:*[int](https://docs.python.org/3/library/functions.html#int), *m:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *k:*[int](https://docs.python.org/3/library/functions.html#int)*= 0*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= 'float32'*) → [Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) 


生成一个单位矩阵或第 k 个对角线为 1 的矩阵。
* **参数：**
   * **n**（[int](https://docs.python.org/3/library/functions.html#int)）：行数。
   * **m**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：列数。如果为 None ，则默认为 n 。
   * **k**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：对角线的索引。0（默认值）表示主对角线。正值表示上对角线，负值表示下对角线。
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：返回数组的数据类型。
* **返回：y** **：** 结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.fast_erf(*x*) 


使用 fast_erf 实现获取输入 x 的高斯误差函数。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.fast_exp(*x*) 


使用 fast_exp 实现对输入 x 进行指数运算。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) *：* 输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.fast_tanh(*x*) 


使用 fast_tanh 实现对输入 x 进行双曲正切
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.fixed_point_multiply(*x*, *multiplier*, *shift*) 


数据与定点常数之间的定点乘法表示为乘数 * 2^(-shift)，其中乘数是一个具有 31 个小数位的 Q 数
* **参数：**
   * **x**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：输入参数。
   * **乘数**（[int](https://docs.python.org/3/library/functions.html#int)）*：* 固定浮点数的乘数，表示为乘数*2^(–shift)。
   * **shift**（[int](https://docs.python.org/3/library/functions.html#int)）：固定浮点数的移位，描述为乘数*2^(–shift)。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.fixed_point_multiply_per_axis(*x:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *y:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *lshift:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *rshift:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *is_lshift_required:*[int](https://docs.python.org/3/library/functions.html#int), *is_rshift_required:*[int](https://docs.python.org/3/library/functions.html#int), *axes*) 


数据与定点常数之间的定点乘法表示为乘数 * 2^(-shift)，其中乘数是一个具有 31 个小数位的 Q 数。
* **参数：**
   * **x**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入参数。
   * **y**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：固定浮点数的乘数，描述为乘数*2^(–shift)。
   * **lshift**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：固定浮点数的左移，描述为乘数*2^(–shift)。
   * **rshift**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：固定浮点数的右移，描述为乘数*2^(–shift)。
   * **is_lshift_required** ( [int](https://docs.python.org/3/library/functions.html#int) )[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)我们是否需要进行左移。
   * **is_rshift_required** ( [int](https://docs.python.org/3/library/functions.html#int) )：我们是否需要进行右移。
* **返回：z** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.flip(*a*, *axis=0*) 


在特定轴上翻转/反转数组的元素。
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)要扩展的张量。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：张量将沿其反转的轴。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.floor(*x*) 


取输入 x 的底数。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.floor_divide(*lhs*, *rhs*) 


使用自动广播的整数除法。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.floor_mod(*lhs*, *rhs*) 


使用自动广播的向下取模。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)右操作数。
* **返回：ret**[：](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.floordiv(*a*, *b*, *span=None*) 


计算两个表达式的 floordiv。
* **参数：**
   * **a** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：左侧操作数。
   * **b** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：右侧操作数。
   * span**（*可选*[*[Span](/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]）：此运算符在源中的位置。
* **返回：res**：结果表达式。
* **返回类型：**[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.topi.floormod(*a*, *b*, *span=None*) 


计算两个表达式的 floormod。
* **参数：**
   * **a** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：左侧操作数。
   * **b** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )**：** 右侧操作数。
   * *span**（*可选**[ [Span](/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：res：** 结果表达式。
* **返回类型：**[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)。

## tvm.topi.full(*shape*, *dtype*, *fill_value*) 


用 fill_value 填充张量。
* **参数：**
   * **形状**（[元组](https://docs.python.org/3/library/stdtypes.html#tuple)）：输入张量形状。
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：数据类型。
   * **fill_value** ( [float](https://docs.python.org/3/library/functions.html#float) )：要填充的值。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.full_like(*x*, *fill_value*) 


构造与输入张量形状相同的张量。


然后用 fill_value 填充张量。
* **参数：**
   * **x**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入参数。
   * **fill_value** ( [float](https://docs.python.org/3/library/functions.html#float) )：要填充的值。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.gather(*data*, *axis*, *indices*) 


从给定的索引沿给定的轴收集值。


例如，对于 3D 张量，输出计算如下：

```plain
out[i][j][k] = data[indices[i][j][k]][j][k]  # if axis == 0
out[i][j][k] = data[i][indices[i][j][k]][k]  # if axis == 1
out[i][j][k] = data[i][j][indices[i][j][k]]  # if axis == 2
```



`indices`必须具有与 相同的形状`data`，但维度`axis` 必须不为空。输出将具有与 相同的形状`indices`。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 运算符的输入数据。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)）：沿其进行索引的轴。
   * **indices**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要提取的值的索引。
* **返回：ret。**
* **返回类型：**[tvm.te.Tenso](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.gather_nd(*a*, *indices*, *batch_dims=0*) 


从 n 维数组中收集元素..
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：源数组。
   * **indices**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 要提取的值的索引。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.get_const_tuple(*in_tuple*) 


验证输入元组是 IntImm 还是 Var，返回 int 或 Var 的元组。
* **参数：in_tuple** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*ofExpr*)**：** 输入。
* **返回：out_tuple**：输出。
* **返回类型：**[tuple](https://docs.python.org/3/library/stdtypes.html#tuple) of [int](https://docs.python.org/3/library/functions.html#int)。

## tvm.topi.greater(*lhs*, *rhs*) 


使用自动广播计算 (lhs>rhs)。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.greater_equal(*lhs*, *rhs*) 


使用自动广播计算 (lhs>=rhs)。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.hamming_window(*window_size*, *periodic*, *alpha*, *beta*, *dtype*) 


汉明窗函数。
* **参数：**
   * **window_size**（*tvm.Expr*）**：** 返回窗口的大小。
   * **period**（*tvm.Expr*）**：** 如果为 True，则返回一个用作周期函数的窗口。如果为 False，则返回一个对称窗口。
   * **alpha** ( *tvm.Expr* )**：** 系数 alpha。
   * **beta**（*tvm.Expr*）：系数 beta。
* **返回：ret**：结果张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.identity(*x*) 


取输入 x 的恒等式。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.index_put(*data*, *indices*, *values*, *accumulate=False*) 


根据索引将值放入数组中。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)要修改的源数组。
   * **indices** ( [Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)*[*[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*]* )：指定位置的 1D 索引张量的元组（每个维度一个）。
   * **值**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：放置在指定索引处的值。
   * **累积**（[bool](https://docs.python.org/3/library/functions.html#bool)*，可选*）：是否累积（添加）值而不是替换。如果为 True，则执行 tensor[indices] += values；如果为 False，则执行 tensor[indices] = values。默认值为 False。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.index_tensor(*data*, *indices*) 


高级张量索引（NumPy/PyTorch 风格）。


给定 k 个索引张量，该运算符从中选择元素，就像在 NumPy/PyTorch 中编写的那样 ：`indices = (I0, I1, …, Ik‑1)``data``data[I0, I1, …, Ik‑1]`
* 所有索引张量必须具有整数数据类型。
* `B`它们的形状以通常的 NumPy 方式一起广播成一个共同的形状。
* 结果形状是（即广播形状后跟未索引的其余轴*）* 。`B + data.shape[k:]``data`
* `k`不能超过`data.ndim`；否则会引发编译时错误。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要索引的张量。
   * *indices*( *Sequence[*[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*]* )：一个包含 k 个索引的 Python`list`张**量**`tuple`，或一个 tvm.te.Tensor 元组表达式。每个张量必须具有整数数据类型。
* **返回：result**：高级索引后得到的张量。其 dtype 等于 `data.dtype`。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
x     = te.placeholder((3, 3),  name="x")        # 形状 (3,3)
row   = te.placeholder((2,),    name="row", dtype="int32")
col   = te.placeholder((2,),    name="col", dtype="int32")

# 等价于 NumPy / PyTorch 中的 x[row, col]
y = topi.index_tensor(x, [row, col])             # 形状 (2,)

# 广播示例：
row = te.placeholder((2, 1), name="row", dtype="int32")
col = te.placeholder((1, 3), name="col", dtype="int32")
z = topi.index_tensor(x, [row, col])             # 形状 (2, 3)
```
## tvm.topi.isfinite(*x*) 


检查 x 的值是否是有限的、元素有限的。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.isinf(*x*) 


检查 x 的值是否为无限的（按元素）。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.isnan(*x*) 


逐个元素检查 x 的值是否为 NaN。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) **：** 输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.layout_transform(*array*, *src_layout*, *dst_layout*, *schedule_rule='None'*) 


根据 src_layout 和 dst_layout 转换布局。
* **参数：**
   * **array**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：源数组。
   * **src_layout** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：源布局。
   * **dst_layout** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：目标布局。
   * **Schedule_rule** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)适用的调度规则（如果有）。

## tvm.topi.left_shift(*lhs*, *rhs*) 


左移并自动广播。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）**：** 左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.less(*lhs*, *rhs*) 


使用自动广播计算 (lhs<rhs)。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）**：** 左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.less_equal(*lhs*, *rhs*) 


使用自动广播计算 (lhs<=rhs)。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret** ：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.log(*x*) 


对输入 x 取对数。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.log10(*x*) 


对输入 x 取以 10 为底的对数。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.log2(*x*) 


对输入 x 取以 2 为底的对数。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.log_add_exp(*lhs*, *rhs*) 


自动广播的对数和指数运算。
* **参数：**
   * **x1**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：第一个输入张量或表达式。
   * **x2**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：第二个输入张量或表达式。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则，返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.logical_and(*lhs*, *rhs*) 


计算元素级别的数据的逻辑与。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.logical_not(*data*) 


逐元素计算数据的逻辑非。
* **参数：data** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*orExpr*)。
* **返回：ret**：如果操作数是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.logical_or(*lhs*, *rhs*) 

计算元素的逻辑或数据。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.logical_xor(*lhs*, *rhs*) 


计算数据的元素逻辑异或。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.make_idx(*b*, *e*, *s*, *z*, *i*) 


返回与完整数组中的数组位置相对应的选择中的数组位置。


仅当 within_index() 对于同一组参数返回 True 时，返回值才有意义。
* **参数：**
   * **b** ( *Expr* )：索引的开头。
   * **e**（*Expr*）**：** 索引的结尾。
   * **s** ( *Expr* )*：* 索引的步幅。
   * **z**（*Expr*）*：* 索引维度的大小。
   * **i** ( *Expr* )：数组位置。
* **返回：position** *：* 与选择中的数组位置相对应的 int 表达式。
* **返回类型：** Expr。

## tvm.topi.matmul(*a*, *b*, *transp_a=False*, *transp_b=False*) 


创建一个计算矩阵乘法（行主表示法）的运算：如果 trans_a == trans_b，则为 A(i, k) * B(k, j)，否则为通常的转置组合。
* **参数:**
   * **a**（*矩阵 A*）。
   * **b**（*矩阵 B*）。**trans_a**（*A 的布局是转置的吗？）。*
   * **trans_b**（*B 的布局是转置的吗？）。*
* **返回类型:** 一个张量（Tensor），其 op 成员是矩阵乘法（matmul）操作。




## tvm.topi.matrix_set_diag(*data*, *diagonal*, *k=0*, *align='RIGHT_LEFT'*) 


返回一个张量，其中输入张量的对角线被提供的对角线值替换。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入张量。
   * **diagonal**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要填充对角线的值。
   * **k**（[int](https://docs.python.org/3/library/functions.html#int)*或*[int](https://docs.python.org/3/library/functions.html#int)*元*[组](https://docs.python.org/3/library/stdtypes.html#tuple)*，可选*）：对角线偏移量。要设置的对角线或对角线范围。（默认为 0）正值表示超对角线，0 表示主对角线，负值表示次对角线。k 可以是单个整数（表示单个对角线）或一对整数，分别指定矩阵带的低端和高端。k[0] 不能大于 k[1]。
   * *align**（*字符串**，*可选*）：某些对角线比 max_diag_len 短，需要填充。align 是一个字符串，指定上对角线和下对角线分别如何对齐。有四种可能的对齐方式：“RIGHT_LEFT”（默认）、“LEFT_RIGHT”、“LEFT_LEFT”和“RIGHT_RIGHT”。“RIGHT_LEFT”将上对角线向右对齐（左填充行），将下对角线向左对齐（右填充行）。这是 LAPACK 使用的打包格式。cuSPARSE 使用“LEFT_RIGHT”，这是相反的对齐方式。
* **返回：result**：具有给定对角线值的新张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
data = [[[7, 7, 7, 7],
         [7, 7, 7, 7],
         [7, 7, 7, 7]],
        [[7, 7, 7, 7],
         [7, 7, 7, 7],
         [7, 7, 7, 7]]]

diagonal = [[1, 2, 3],
            [4, 5, 6]]

topi.matrix_set_diag(input, diagonal) =
    [[[1, 7, 7, 7],
      [7, 2, 7, 7],
      [7, 7, 3, 7]],
     [[4, 7, 7, 7],
      [7, 5, 7, 7],
      [7, 7, 6, 7]]]
```
## tvm.topi.max(*data*, *axis=None*, *keepdims=False*) 


给定轴或轴列表上的数组元素的最大值。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入 tvm 张量。
   * **axis**（*None、*[int](https://docs.python.org/3/library/functions.html#int)*或*[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)执行最大值运算的轴。默认值 axis=None 表示从输入数组的所有元素中查找最大值元素。如果 axis 为负数，则从最后一个轴计数到第一个轴。
   * **keepdims** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果设置为 True，则缩小的轴将保留在结果中，作为大小为 1 的维度。使用此选项，结果将根据输入数组正确广播。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.maximum(*lhs*, *rhs*) 


对两个张量执行元素级最大值操作，支持自动广播（auto-broadcasting）。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.meshgrid(*a_tuple*, *indexing*) 


从坐标向量创建坐标矩阵。
* **参数：**
   * **a_tuple**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*的*[元组](https://docs.python.org/3/library/stdtypes.html#tuple)）：坐标向量或标量。
   * **indexing** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：索引模式，可以是「ij」或「xy」。
* **返回：result**：每个轴的结果网格。
* **返回类型：**[tuple](https://docs.python.org/3/library/stdtypes.html#tuple) of [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.min(*data*, *axis=None*, *keepdims=False*) 


给定轴或轴列表上的数组元素的最小值。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 输入 tvm 张量。 
   * **axis**（*None、*[int](https://docs.python.org/3/library/functions.html#int)*或*[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)*：* 执行最小值运算的轴。默认值 axis=None 将从输入数组的所有元素中查找最小元素。如果 axis 为负数，则从最后一个轴计数到第一个轴。
   * **keepdims** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果设置为 True，则缩小的轴将保留在结果中，作为大小为 1 的维度。使用此选项，结果将根据输入数组正确广播。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.minimum(*lhs*, *rhs*) 


使用自动广播，逐个元素地取两个张量的最大值。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.mod(*lhs*, *rhs*) 


自动广播模块
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.multiply(*lhs*, *rhs*) 


自动广播乘法。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）*：* 右操作数。
* **返回：ret** ：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.ndarray_size(*array*, *dtype='int32'*) 


获取输入数组元素的数量
* **参数：**
   * **数组**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)源张量。
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：目标数据类型。
* **返回：result**：结果张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.negative(*x*) 


对输入 x 取否定。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.not_equal(*lhs*, *rhs*) 


使用自动广播计算 (lhs!=rhs)。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）*：* 右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.one_hot(*indices*, *on_value*, *off_value*, *depth*, *axis*, *dtype*) 


返回一个独热张量，其中索引所代表的位置的值为 on_value，其他位置的值为 off_value。最终维度为 <索引外维度> x 深度 x <索引内维度>。
* **参数：**
   * **indices**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 设置为 on_value 的位置。
   * **on_value**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：填充索引的值。
   * **off_value**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：除了索引之外的所有其他位置填充的值。
   * **深度**（[int](https://docs.python.org/3/library/functions.html#int)）：独热维度的深度。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)）：要填充的轴。
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：输出张量的数据类型。
* **返回：ret**：独热张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
indices = [0, 1, 2]

topi.one_hot(indices, 3) =
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
```
## tvm.topi.power(*lhs*, *rhs*) 


自动广播幂方。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.prod(*data*, *axis=None*, *keepdims=False*) 


给定轴或轴列表上的数组元素的乘积。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入 tvm 张量。
   * **axis**（*None、*[int](https://docs.python.org/3/library/functions.html#int)*或*[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)**：** 执行 prod 操作的轴。默认值 axis=None 表示将获取输入数组所有元素上的 prod 元素。如果 axis 为负数，则从最后一个轴计数到第一个轴。
   * **keepdims** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果设置为 True，则缩小的轴将保留在结果中，作为大小为 1 的维度。使用此选项，结果将根据输入数组正确广播。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.reinterpret(*x*, *dtype*) 


将输入重新解释为指定的数据类型。
* **参数：**
   * **x**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入参数。
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：数据类型。
* **返回：y** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.repeat(*a*, *repeats*, *axis*) 


重复数组的元素。
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 要重复的张量。
   * **repeats**（[int](https://docs.python.org/3/library/functions.html#int)*，必需*）：每个元素的重复次数。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：重复值的轴。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.reshape(*a*, *newshape*) 


重塑数组。
* **参数：**
   * **a** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：需要重塑的张量。
   * **newshape**（*整数元组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：新形状。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.reverse_sequence(*a*, *seq_lengths*, *seq_axis=1*, *batch_axis=0*) 


将张量反转为可变长度切片。输入首先沿批处理轴进行切片，然后沿序列轴反转元素。
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要反转的张量。
   * **seq_lengths** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)长度为 a.dims[batch_axis] 的一维张量，必须是以下类型之一：int32、int64，如果 seq_lengths[i] > a.dims[seq_axis]，则四舍五入为 a.dims[seq_axis]，如果 seq_lengths[i] < 1，则四舍五入为 1。
   * **seq_axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：元素反转的轴。默认值为 1。
   * **batch_axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）*：* 张量切片的轴。默认值为 0。
* **返回：ret**：与输入具有相同形状和类型的计算结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.right_shift(*lhs*, *rhs*) 


右移并自动广播。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.round(*x*) 


将 x 的元素四舍五入为最接近的整数。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.rsqrt(*x*) 


取输入 x 的平方根的倒数。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.scanop(*data:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *binop:*[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*[[tvm.Expr, tvm.Expr], tvm.Expr]*, *identity_value: tvm.Expr*, *op_name:*[str](https://docs.python.org/3/library/stdtypes.html#str), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *exclusive:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) 


累积二元运算符（扫描），其轴行为与 np.cumsum 和 np.cumprod 类似。


请参阅 cumprod 和 cumsum 了解使用示例。


例如，如果 * 是二元运算符，输入张量为 [1, 2, 3, 4]，则输出可能是 [1, 1 * 2, 1 * 2 * 3, 1 * 2 * 3 * 4]
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：运算符的输入数据。
   * *binop* ( *Callable(tvm.Expr,tvm.Expr)–> tvm.Expr* )：一个二元运算符，它必须满足结合律和交换律。例如，如果你的运算符是 * ，那么 a * (b * c) = (a * b) * c ，并且 a * b = b * a
   * **身份值**( *tvm.Expr* )：提供身份属性的二元运算值。例如，如果 * 是运算符，i 是身份值，那么对于运算域中的所有 a，a * i = a。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：计算操作所沿的轴。默认值（无）是计算展平数组上的累积运算。
   * *dtype**（*string*，*可选*）：返回数组的类型，以及用于计算元素的累加器的类型。如果未指定 dtype，则默认为 data 的 dtype。
   * **exclusive** （[bool](https://docs.python.org/3/library/functions.html#bool)*，可选*）：如果为 True，则返回独占累积运算，其中不包含第一个元素。换句话说，如果为 True，则第 j 个输出元素将是前 (j–1) 个元素的累积运算。否则，它将是前 j 个元素的累积运算。零个元素的累积运算被假定为身份值。
* **返回：result**：如果 axis 不为 None，则结果的大小与数据相同，形状也与数据相同。如果 axis 为 None，则结果为一维数组。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.scatter_elements(*data*, *indices*, *updates*, *axis=0*, *reduction='update'*) 


将更新中的元素分散到复制数据的相应索引中。


数据、索引、更新和输出具有相同的形状。如果 reduction == “update”，则索引不能有重复项（如果 idx1 != idx2，则 indices[idx1] != indices[idx2]）。

```plain
output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0
output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1
```


其中更新函数 f 由约简确定。该函数支持五种类型：“update”、“add”、“mul”、“min”和“max”（见下文）
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：源数组。
   * **indices**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要提取的值的索引。
   * **更新**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：应用于索引的更新。
   * *axis**（*可选**，[int](https://docs.python.org/3/library/functions.html#int)）：散点图的轴。默认值为零。
   * r*eduction**（*可选**，*字符串*）：算法的更新模式，可以是“update”，“add”，“mul”，“min”或“max”。如果是更新，更新值将替换输入数据。如果是添加，更新值将添加到输入数据中。如果是 mul，输入数据将乘以更新值。如果是平均值，输入数据将是更新值和输入数据之间的平均值。如果是最小值，则可以在更新值和输入数据之间选择最小值。如果是最大值，则可以在更新值和输入数据之间选择最大值。默认情况下为「更新」。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.scatter_nd(*data*, *indices*, *updates*, *mode*) 


从 n 维数组中分散元素。


给定形状为 (Y_0, …, Y_{K-1}, X_M, …, X_{N-1}) 的更新、形状为 (M, Y_0, …, Y_{K-1}) 的索引以及从形状为 (X_0, X_1, …, X_{N-1}) 的数据复制的输出，scatter_nd 计算。

```plain
output[indices[0, y_0, ..., y_{K-1}],
       ...,
       indices[M-1, y_0, ..., y_{K-1}],
       x_M,
       ...,
       x_{N-1}
      ] = f(output[...], updates[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}])
```


其中更新函数 f 由模式决定。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 源数组。
   * **indices**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要提取的值的索引。
   * **更新**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 应用于索引的更新。
   * **mode**（*字符串*）*：* 算法的更新模式，可以是“更新”或“添加”。如果是更新，则更新值将替换输入数据。如果是添加，则更新值将添加到输入数据中。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.searchsorted(*sorted_sequence*, *values*, *right=False*, *out_dtype='int64'*) 


查找应插入元素以维持顺序的索引。


如果 sorted_sequence 是 N 维的，则 在 sorted_sequence 的相应维度中搜索值的最内层维度。
* **参数：**
   * **sorted_sequence**（[te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：ND 或 1–D 张量，包含最内层维度上的单调递增序列。
   * **values** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：包含搜索值的 ND 张量。当 sorted_sequence 为一维时， values 的形状可以是任意的。否则，sorted_sequence 和 values 的秩必须相同，且外 N–1 个轴的大小必须相同。
   * **right**（[bool](https://docs.python.org/3/library/functions.html#bool)*，可选*）**：** 控制当值恰好位于已排序值之一时返回哪个索引。如果为 False，则返回找到的第一个合适位置的索引。如果为 True，则返回最后一个合适的索引。如果没有合适的索引，则返回 0 或 N（其中 N 是最内层维度的大小）。
   * dtype（*字符串*，可选）*：* 输出索引的数据类型。
* **返回：indices**：与值具有相同形状的张量，表示如果值元素插入 sorted_sequence 中则它们的索引。
* **返回类型：**[te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.sequence_mask(*data*, *valid_length*, *mask_value=0*, *axis=0*) 


将序列预期长度之外的所有元素设置为常量值。


此函数采用形式为 [MAX_LENGTH, batch_size, …] 或 [batch_size, MAX_LENGTH, …] 的 n 维输入数组，并返回相同形状的数组。


axis 表示长度维度的轴，只能为 0 或 1。如果 axis 为 0，则数据形状必须为 [MAX_LENGTH, batch_size, …]。否则（axis=1），数据形状必须为 [batch_size, MAX_LENGTH, …]。


valid_length 给出每个序列的长度。valid_length 应该是一个包含正整数的一维 int 数组，维度为 [batch_size,] 。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：根据轴的值，具有形状[MAX_LENGTH，batch_size，…]或[batch_size，MAX_LENGTH，…]的 ND 。
   * **valid_length**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：一维，形状为[batch_size,]。
   * **mask_value**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：掩蔽值，默认为 0。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：长度维度的轴，必须为 0 或 1，默认为 0。
* **返回：output**：ND，形状为 [MAX_LENGTH, batch_size, …] 或 [batch_size, MAX_LENGTH, …]，具体取决于轴的值。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.shape(*array*, *dtype='int32'*) 


获取输入数组的形状
* **参数：**
   * **数组**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 源张量。
   * **dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：目标数据类型。
* **返回：result**：结果张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.sigmoid(*x*) 


对输入 x 进行 S 型 tanh 运算。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) *：* 输入参数。
* **返回：y** *：* 结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.sign(*x*) 


根据 x 的符号返回 -1、0、1。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.sin(*x*) 


对输入 x 取正弦值。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.sinh(*x*) 


对输入 x 取 sinh。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.slice_scatter(*input_tensor*, *src*, *start*, *end*, *step*, *axis*) 


将 src 的切片沿给定轴（SSA 形式）分散到输入中。
* **参数：**
   * **input_tensor**（[te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要散射的输入张量。
   * **src**（[te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要散射的源张量。
   * **start**（[int](https://docs.python.org/3/library/functions.html#int)）*：* 切片的起始索引。
   * **end**（[int](https://docs.python.org/3/library/functions.html#int)）：切片的结束索引。
   * **step**（[int](https://docs.python.org/3/library/functions.html#int)）：切片的步长。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)）：散布的轴。
* **返回：** 包含切片分散的输出张量的列表。
* **返回类型：**[list](https://docs.python.org/3/library/stdtypes.html#list)[[te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)]。

## tvm.topi.sliding_window(*data*, *axis*, *window_shape*, *strides*) 


在数据张量上滑动一个窗口。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：运算符的输入数据。
   * **axis** ( [int](https://docs.python.org/3/library/functions.html#int) )*：* 窗口开始滑动的轴。窗口将在此轴及其所有后续轴上滑动。axis 值决定了窗口的形状（从而决定了步长）：窗口形状和步长的长度都必须为 data.ndim–axis。
   * *window_shape* ( *List[*[int](https://docs.python.org/3/library/functions.html#int)*]* )**：** 在输入上形成的窗口形状。窗口形状的长度必须为 data.ndim–axis。
   * *strides* ( *List[*[int](https://docs.python.org/3/library/functions.html#int)*]* )*：* 如何沿每个维度移动窗口。步幅必须为 data.ndim–axis 的长度。
* **返回：result**：结果张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.sort(*data*, *axis=-1*, *is_ascend=1*) 


沿给定轴执行排序并按排序顺序返回数组。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入张量。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）*：* 对输入张量进行排序的轴。默认情况下使用扁平数组。
   * *is_ascend*（*布尔值*，*可选*）*：* 按升序还是降序排序。
   * *dtype*（*字符串*，*可选*）：输出索引的 DType。
* **返回：out**：排序索引张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.sparse_reshape(*sparse_indices*, *prev_shape*, *new_shape*, *new_sparse_indices_shape*, *new_shape_shape*) 


重塑稀疏张量。
* **参数：**
   * **sparse_indices** ( *te.Expr* )：包含稀疏值位置的整数二维张量 [N, n_dim]，其中 N 是稀疏值的数量，n_dim 是 dense_shape 的维数。
   * **prev_shape** ( *te.Expr* )：包含密集张量的先前形状的一维张量。
   * **new_shape** ( *te.Expr* )：包含稠密张量新形状的一维张量。
* **返回：result**：输出张量。
* **返回类型：** te.Expr。


**示例**

```python
sparse_indices = [[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 2, 3]]
prev_shape = [2, 3, 4]
new_shape = [9, -1]
new_sparse_indices, new_shape = topi.sparse_reshape(
    sparse_indices, prev_shape, new_shape)
new_sparse_indices = [[0, 0],
                      [0, 1],
                      [1, 2],
                      [4, 2],
                      [8, 1]]
new_shape = [9, 4]
```
## tvm.topi.sparse_to_dense(*sparse_indices*, *output_shape*, *sparse_values*, *default_value=0*) 


将稀疏表示转换为密集张量。


示例:: - sparse_to_dense([[0, 0], [1, 1]], [2, 2], [3, 3], 0) = [[3, 0], [0, 3]]。
* **参数：**
   * **sparse_indices**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：包含稀疏值位置的 0–D、1–D 或 2–D 整数张量。
   * **output_shape**（*整数列表*）：密集输出张量的形状 *。*
   * **sparse_values**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：包含稀疏索引的稀疏值的 0–D 或 1–D 张量。
   * **default_value**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：一个 0 维张量，包含剩余位置的默认值。默认为 0。
* **返回：result** ：形状为 output_shape 的稠密张量。类型与 sparse_values 相同。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.split(*ary*, *indices_or_sections*, *axis=0*) 


将数组拆分为多个子数组。
* **参数：**
   * **ary** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor))。
   * **indices_or_sections** ([int](https://docs.python.org/3/library/functions.html#int)*or1-D array*)。
   * **axis** ([int](https://docs.python.org/3/library/functions.html#int))。
* **返回：ret。**
* **返回类型：**[tuple](https://docs.python.org/3/library/stdtypes.html#tuple) of [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.sqrt(*x*) 


对输入 x 取平方根。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.squeeze(*a*, *axis=None*) 


从数组形状中删除一维条目。
* **参数：**
   * **a** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor))。
   * **axis** (*Noneor*[int](https://docs.python.org/3/library/functions.html#int)*or*[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*ofints,optional*)[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)  axis（*None*、***[int](https://docs.python.org/3/library/functions.html#int)***或***int 元**组*[，](https://docs.python.org/3/library/stdtypes.html#tuple)***可选*** *）*：选择形状中单维条目的子集。如果所选轴的形状条目数大于 1，则会引发错误。
* **返回：squeezed。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.stack(*tensors*, *axis=0*) 


沿新轴连接一系列张量。
* **参数：**
   * **张量**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*的*[元组](https://docs.python.org/3/library/stdtypes.html#tuple)*或*[列表](https://docs.python.org/3/library/stdtypes.html#list)）*：* 需要堆叠的张量。所有张量必须具有相同的形状。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）*：* 输入张量将沿着结果张量的轴进行堆叠。负值表示环绕。默认值为 0。
* **返回：ret**：与输入张量相比具有额外维度的堆叠张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.stft(*data*, *n_fft*, *hop_length*, *win_length*, *window*, *normalized*, *onesided*, *output_shape*) 


STFT 计算输入短重叠窗口的傅里叶变换。这给出了信号随时间变化的频率分量。:param data: 一维张量或二维批量张量。:type data: te.Tensor :param n_fft: 傅里叶变换的大小 :type n_fft: int :param hop_length: 相邻滑动窗口帧之间的距离 :type hop_length: int :param win_length: 窗口帧和 STFT 滤波器的大小 :type win_length: int :param window: 一维张量窗口帧 :type window: te.Tensor :param normalized: 是否返回归一化的 STFT 结果 :type normalized: bool :param onesided: 是否返回单侧结果或使用共轭对称性填充 :type onesided: bool。
* **返回：output**：包含 STFT 结果的张量。
* **返回类型：**[te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
data = [1, 2, 3, 4, 5, 6]
window = [4, 3, 2]
[n_fft, hop_length, win_length, normalized, onesided] = [3, 3, 3, False, True]
topi.stft(data, n_fft, hop_length, win_length, window, normalized, onesided)
-> [[[15.0000,  0.0000], [34.0000,  0.0000]], [[ 4.5000,  0.8660], [ 1.0000, -1.7321]]]
```
## tvm.topi.strided_set(*a*, *v*, *begin*, *end*, *strides=None*) 


设置数组的切片。
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 要切片的张量。
   * **v**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要设置的值。
   * **begin**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：切片中开始的索引。
   * **end**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 指示切片结束的索引。
   * **strides**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*）：指定步幅值，在这种情况下可以为负数，输入张量将在该特定轴上反转。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.strided_slice(*a*, *begin*, *end*, *strides=None*, *axes=None*, *slice_mode='end'*, *assume_inbound=True*) 


数组的切片。
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要切片的张量。
   * **begin**（*int*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：切片中开始的索引[。](https://docs.python.org/3/library/functions.html#int)
   * **end**（*int*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：指示切片结束的索引[。](https://docs.python.org/3/library/functions.html#int)
   * **strides**（*整数*[列表](https://docs.python.org/3/library/stdtypes.html#list)*，可选）：指定*[步幅](https://docs.python.org/3/library/functions.html#int)值，在这种情况下可以为负数，输入张量将在该特定轴上反转。
   * **轴**( *int*[列表](https://docs.python.org/3/library/stdtypes.html#list)*，可选)：应用切片*[的](https://docs.python.org/3/library/functions.html#int)轴。指定后，起始、结束步幅和轴需要为相同长度的整数列表。
   * **slice_mode**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）**：** 切片模式 [end, size]。end*：*切片的结束索引 [默认]。size：输入的步幅将被忽略，此模式下的输入 end 表示从 begin 指定位置开始的切片大小。如果 end[i] 为 –1，则该维度上的所有剩余元素都将包含在切片中。
   * **假设_inbound** ( [bool](https://docs.python.org/3/library/functions.html#bool)*，可选*)：一个标志，指示是否假定所有索引都是入站的。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.subtract(*lhs*, *rhs*) 


自动广播减法。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：左操作数。
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）：右操作数。
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr。

## tvm.topi.sum(*data*, *axis=None*, *keepdims=False*) 


给定轴或轴列表上的数组元素的总和。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 输入 tvm 张量。
   * **axis**（*None、*[int](https://docs.python.org/3/library/functions.html#int)*或*[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：执行求和的轴。默认值 axis=None，将对输入数组的所有元素求和。如果 axis 为负数，则从最后一个轴开始计数到第一个轴。
   * **keepdims** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果设置为 True，则缩小的轴将保留在结果中，作为大小为 1 的维度。使用此选项，结果将根据输入数组正确广播。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.take(*a*, *indices*, *axis=None*, *batch_dims=0*, *mode='fast'*) 


沿轴从数组中获取元素。
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：源数组。
   * **indices**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要提取的值的索引。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）**：** 用于选择值的轴。默认情况下，使用扁平化的输入数组。
   * **batch_dims**（[int](https://docs.python.org/3/library/functions.html#int)）：批次维度的数量。默认情况下为 0。
   * **mode**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：指定超出范围的索引将如何表现。*：* fast（默认）：额外的索引会导致段错误（用户必须确保索引在范围内）：nan：为超出范围的索引生成 NaN：wrap：环绕索引：clip：剪辑到范围内。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.tan(*x*) 


对输入 x 取 tan。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.tanh(*x*) 


对输入 x 取双曲 tanh。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.tensordot(*a*, *b*, *axes*) 


矩阵乘法到张量的推广。
* **参数：**
   * **a** (*The tensor A*)。
   * **b** (*The tensor B*)。
   * **axes** (*The numberofdimensions to reduce over*)。
* 返回类型：计算结果的张量。


## tvm.topi.tile(*a*, *reps*) 


重复整个数组多次。
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)要平铺的张量。
   * *reps（[整数](https://docs.python.org/3/library/stdtypes.html#tuple)元组，* 必需）：重复张量的次数。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.topk(*data*, *k=1*, *axis=-1*, *ret_type='both'*, *is_ascend=False*, *dtype='int64'*) 


获取输入张量中沿给定轴的前 k 个元素。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入张量。
   * **k**（[int](https://docs.python.org/3/library/functions.html#int)*或*[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*）：待选元素的数量。如果 k < 1，则返回所有元素。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：用于对输入张量进行排序的轴长。
   * **ret_type**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：返回类型 [both，values，indices]。“both”：返回前 k 个数据和索引。“values”：仅返回前 k 个数据。“indices”：仅返回前 k 个索引。
   * *is_ascend*（*布尔值*，*可选*）：按升序还是降序排序。
   * *dtype*（*字符串**，*可选*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)索引输出的数据类型。
* **返回：out** **：** 计算结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or List[[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)]。

## tvm.topi.transpose(*a*, *axes=None*) 


排列数组的维度。
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要扩展的张量。
   * *轴*（*整数*元[组](https://docs.python.org/3/library/stdtypes.html#tuple)，可选）：默认情况下，反转尺寸。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.trilu(*data*, *k*, *upper*) 


给定一个二维矩阵或一批二维矩阵，返回张量的上三角或下三角部分。
* **参数：**
   * **data** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)  **data** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：trilu 将应用到的张量。必须是二维矩阵或由二维矩阵批次组成的张量。
   * **k** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)  **k**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要排除或包含的主对角线上方或下方的对角线数量。
   * **upper** ([bool](https://docs.python.org/3/library/functions.html#bool)) [：](https://docs.python.org/3/library/functions.html#bool)  **upper** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：如果为 True，则仅保留输入的上三角值；如果为 False，则保留下三角值。
* **返回：ret** ：将适当的对角线设置为零的新张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。


**示例**

```python
x = [[0, 1, 2],
     [3, 4, 5],
     [6, 7, 8]]

topi.trilu(x, True, 0) =
    [[0, 1, 2],
     [0, 4, 5],
     [0, 0, 8]]
```
## tvm.topi.trunc(*x*) 


逐个元素地取 x 输入的截断值。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.unravel_index(*indices*, *shape*) 


将平面索引或平面索引数组转换为坐标数组的元组。


示例:: - unravel_index([22, 41, 37], [7, 6]) = [[3, 6, 6], [4, 5, 1]]。
* **参数：**
   * **indices**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：包含索引的整数数组。
   * **shape**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 数组的形状。
* **返回：result** ：坐标数组的元组。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.where(*condition*, *x*, *y*) 


根据条件从 x 或 y 获取元素。
* **参数：**
   * **条件**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：条件数组。
   * **x**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 要选择的第一个数组。
   * **y**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 要选择的第二个数组。
* **返回：result** ：根据条件从 x 或 y 中选择的张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.within_index(*b*, *e*, *s*, *i*) 


返回一个布尔值，指示 i 是否在给定索引内。
* **参数：**
   * **b** ( *Expr* )：索引的开头。
   * **e**（*Expr*）：索引的结尾。
   * **s** ( *Expr* )：索引的步幅。
   * **i** ( *Expr* )：数组位置。
* **返回：selected**：bool 表达式，如果数组位置由索引选择则为 True，否则为 False。
* **返回类型：** Expr。

## *exception* tvm.topi.InvalidShapeError 


 topi 函数的形状无效。例如，调用非 3x3 内核的 winograd 模板）



# tvm 神经网络算子



神经网络运算符

 

**类：**

|[Workload](/docs/api-reference/python-api/tvm-topi#class-tvmtopinnworkloadin_dtype-out_dtype-height-width-in_filter-out_filter-kernel_h-kernel_w-padt-padl-padb-padr-dilation_h-dilation_w-stride_h-stride_w)(in_dtype,out_dtype,height,width,…)||
|:----|:----|

**函数：**

|[adaptive_pool](/docs/api-reference/python-api/tvm-topi#tvmtopinnadaptive_pooldata-output_size-pool_type-layoutnchw)(data,output_size,pool_type)|对数据的高度和宽度维度进行池化。|
|:----|:----|
|[adaptive_pool1d](/docs/api-reference/python-api/tvm-topi#tvmtopinnadaptive_pool1ddata-output_size-pool_type-layoutncw)(data,output_size,pool_type)|对三维数据进行池化。详情请参阅上面的二维版本。|
|[adaptive_pool3d](/docs/api-reference/python-api/tvm-topi#tvmtopinnadaptive_pool3ddata-output_size-pool_type-layoutncdhw)(data,output_size,pool_type)|对三维数据进行池化。详情请参阅上面的二维版本。|
|[add](/docs/api-reference/python-api/tvm-topi#tvmtopinnaddlhs-rhs)(lhs,rhs)|自动广播附加功能。|
|[batch_matmul](/docs/api-reference/python-api/tvm-topi#tvmtopinnbatch_matmultensor_a-tensor_b-oshapenone-out_dtypenone-transpose_afalse-transpose_btrue-auto_scheduler_rewritten_layout-meta_schedule_original_shapenone)(tensor_a,tensor_b[,oshape,…])|计算 tensor_a 和 tensor_b 的批量矩阵乘法。|
|[batch_norm](/docs/api-reference/python-api/tvm-topi#tvmtopinnbatch_normdatatensor-gammatensor-betatensor-moving_meantensor-moving_vartensor-axisintnone-none-epsilonfloatnone-none-centerboolnone-none-scaleboolnone-none-trainingboolnone-none-momentumfloatnone-none--listtensor)(data,gamma,beta,moving_mean,…)|批量标准化层（Ioffe 和 Szegedy，2014）。|
|[batch_to_space_nd](/docs/api-reference/python-api/tvm-topi#tvmtopinnbatch_to_space_nddata-block_shape-crop_begin_list-crop_end_list)(data,block_shape,…)|对数据执行空间到批量的转换。|
|[binarize_pack](/docs/api-reference/python-api/tvm-topi#tvmtopinnbinarize_packdata-axisnone-namepackedinput)(data[,axis,name])|沿某个轴进行二值化和位打包。|
|[binary_dense](/docs/api-reference/python-api/tvm-topi#tvmtopinnbinary_densedata-weight)(data,weight)|使用异或和位计数进行二进制矩阵乘法。|
|[bitpack](/docs/api-reference/python-api/tvm-topi#tvmtopinnbitpackdata-bits-pack_axis-bit_axis-pack_type-namequantizeinput)(data,bits,pack_axis,bit_axis,…)|将数据打包成位串行计算所需的格式。|
|[bitserial_conv2d_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinnbitserial_conv2d_nchwdata-kernel-stride-padding-activation_bits-weight_bits-pack_dtypeuint32-out_dtypeint16-unipolartrue)(data,kernel,stride,…)|Bitserial Conv2D 运算符。|
|[bitserial_conv2d_nhwc](/docs/api-reference/python-api/tvm-topi#tvmtopinnbitserial_conv2d_nhwcdata-kernel-stride-padding-activation_bits-weight_bits-pack_dtypeuint32-out_dtypeint16-unipolartrue)(data,kernel,stride,…)|Bitserial Conv2D 运算符。|
|[bitserial_dense](/docs/api-reference/python-api/tvm-topi#tvmtopinnbitserial_densedata-weight-data_bits-weight_bits-pack_dtypeuint32-out_dtypeint16-unipolartrue)(data,weight,data_bits,…)|topi 中位串行密集的默认实现。|
|[circular_pad](/docs/api-reference/python-api/tvm-topi#tvmtopinncircular_paddata-pad_before-pad_afternone-namecircularpadinput)(data,pad_before[,pad_after,name])|对输入张量应用圆形填充（环绕）。|
|[concatenate](/docs/api-reference/python-api/tvm-topi#tvmtopinnconcatenatea_tuple-axis0)(a_tuple[,axis])|沿现有轴连接一系列数组。|
|[conv](/docs/api-reference/python-api/tvm-topi#tvmtopinnconvinptensor-filttensor-strideintsequenceint-paddingintsequenceint-dilationintsequenceint-groupsint-data_layoutstr-kernel_layoutstr--out_dtypestrnone-none-auto_scheduler_rewritten_layoutstrnone-none-meta_schedule_original_shapenone-auto_scheduler_should_rewrite_layoutbool-false)(inp,filt,stride,padding,dilation,…)|NCHW 或 NHWC 布局中的卷积操作符。|
|[conv1d](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv1ddata-kernel-strides1-paddingvalid-dilation1-groups1-data_layoutncw-kernel_layout-out_dtypenone)(data,kernel[,strides,padding,…])|1D 卷积前向操作符。|
|[conv1d_ncw](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv1d_ncwdata-kernel-strides1-paddingvalid-dilation1-out_dtypenone)(data,kernel[,strides,padding,…])|NCW 布局中的一维卷积。|
|[conv1d_nwc](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv1d_nwcdata-kernel-strides1-paddingvalid-dilation1-out_dtypenone)(data,kernel[,strides,padding,…])|NWC 布局中的一维卷积。|
|[conv1d_transpose_ncw](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv1d_transpose_ncwdata-kernel-stride-padding-out_dtype-output_padding)(data,kernel,stride,…)|转置的一维卷积 ncw 前向操作符。|
|[conv2d](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2dinput-filter-strides-padding-dilation-data_layoutnchw-kernel_layout-out_dtypenone)(input,filter,strides,padding,dilation)|Conv2D 运算符。|
|[conv2d_NCHWc](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_nchwcdata-kernel-stride-padding-dilation-layout-out_layout-out_dtypefloat32)(data,kernel,stride,padding,…)|nChw[x]c 布局的 Conv2D 运算符。|
|[conv2d_NCHWc_int8](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_nchwc_int8data-kernel-stride-padding-dilation-layout-out_layout-out_dtypeint32-n_elems4)(data,kernel,stride,…)|nChw[x]c 布局的 Conv2D 运算符。|
|[conv2d_hwcn](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_hwcninput-filter-stride-padding-dilation-out_dtypenone)(Input,Filter,stride,padding,…)|HWCN 布局中的卷积操作符。|
|[conv2d_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_nchwinput-filter-stride-padding-dilation-out_dtypenone)(Input,Filter,stride,padding,…)|NCHW 布局中的卷积操作符。|
|[conv2d_nhwc](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_nhwcinput-filter-stride-padding-dilation-out_dtypefloat32-auto_scheduler_rewritten_layout-meta_schedule_original_shapenone)(Input,Filter,stride,padding,…)|NHWC 布局中的卷积操作符。|
|[conv2d_transpose_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_transpose_nchwinput-filter-strides-padding-out_dtype-output_padding)(Input,Filter,…)|转置的二维卷积 nchw 前向操作符。|
|[conv2d_transpose_nchw_preprocess](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_transpose_nchw_preprocessdata-kernel-strides-padding-out_dtype-output_padding)(data,…)|预处理数据和内核，使 conv2d_transpose 的计算模式与 conv2d 相同。|
|[conv2d_winograd_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_winograd_nchwdata-weight-strides-padding-dilation-out_dtype-pre_computedfalse-auto_scheduler_rewritten_layout-meta_schedule_original_shapenone)(data,weight,strides,…)|NCHW 布局中的 Conv2D Winograd。这是一个干净的版本，可供 CPU 和 GPU 的自动调度程序使用。|
|[conv2d_winograd_nchw_without_weight_transform](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_winograd_nchw_without_weight_transformdata-weight-strides-padding-dilation-out_dtype-auto_scheduler_rewritten_layout-meta_schedule_original_shapenone)(…)|在 NCHW 布局中，Conv2D Winograd 无需布局变换。这是一个可供 CPU 和 GPU 元调度使用的干净版本。|
|[conv2d_winograd_nhwc](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_winograd_nhwcdata-weight-strides-padding-dilation-out_dtype-pre_computedfalse-auto_scheduler_rewritten_layout-meta_schedule_original_shapenone)(data,weight,strides,…)|NHWC 布局中的 Conv2D Winograd。这是一个干净的版本，可供 CPU 和 GPU 的自动调度程序使用。|
|[conv2d_winograd_nhwc_without_weight_transform](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_winograd_nhwc_without_weight_transformdata-weight-strides-padding-dilation-out_dtype-auto_scheduler_rewritten_layout-meta_schedule_original_shapenone)(…)|Conv2D Winograd 在 NHWC 布局中无需布局变换。这是一个干净的版本，可供 CPU 和 GPU 自动调度程序使用。|
|[conv2d_winograd_weight_transform](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv2d_winograd_weight_transformkernel-tile_size)(kernel,…)|winograd 的权重转换。|
|[conv3d_ncdhw](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv3d_ncdhwinput-filter-stride-padding-dilation-groups-out_dtypenone)(Input,Filter,stride,padding,…)|NCDHW 布局中的 Conv3D 运算符。|
|[conv3d_ndhwc](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv3d_ndhwcinput-filter-stride-padding-dilation-groups-out_dtypefloat32-auto_scheduler_rewritten_layout-meta_schedule_origin_shapenone)(Input,Filter,stride,padding,…)|NDHWC 布局中的卷积操作符。|
|[conv3d_transpose_ncdhw](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv3d_transpose_ncdhwinput-filter-strides-padding-out_dtype-output_padding)(Input,Filter,…)|转置的3D 卷积 ncdhw 前向操作符。|
|[conv3d_transpose_ncdhw_preprocess](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv3d_transpose_ncdhw_preprocessdata-kernel-strides-padding-out_dtype-output_padding)(data,…)|预处理数据和内核，使 conv3d_transpose 的计算模式与 conv3d 相同。|
|[conv3d_winograd_weight_transform](/docs/api-reference/python-api/tvm-topi#tvmtopinnconv3d_winograd_weight_transformkernel-tile_size)(kernel,…)|3D winograd 的权重变换。|
|[correlation_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinncorrelation_nchwdata1-data2-kernel_size-max_displacement-stride1-stride2-padding-is_multiply)(data1,data2,kernel_size,…)|NCHW 布局中的相关运算符。|
|[declaration_conv2d_transpose_impl](/docs/api-reference/python-api/tvm-topi#tvmtopinndeclaration_conv2d_transpose_impldata-kernel-strides-padding-out_dtype-output_padding)(data,…)|conv2d 转置的实现。|
|[declaration_conv3d_transpose_impl](/docs/api-reference/python-api/tvm-topi#tvmtopinndeclaration_conv3d_transpose_impldata-kernel-strides-padding-out_dtype-output_padding)(data,…)|conv3d 转置的实现。|
|[deformable_conv2d_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinndeformable_conv2d_nchwdata-offset-kernel-strides-padding-dilation-deformable_groups-groups-out_dtype)(data,offset,kernel,…)|NCHW 布局中的可变形 conv2D 运算符。|
|[deformable_conv2d_nhwc](/docs/api-reference/python-api/tvm-topi#tvmtopinndeformable_conv2d_nhwcdata-offset-kernel-strides-padding-dilation-deformable_groups-groups-out_dtype)(data,offset,kernel,…)|NHWC 布局中的可变形 conv2D 运算符。|
|[dense](/docs/api-reference/python-api/tvm-topi#tvmtopinndensedata-weight-biasnone-out_dtypenone-auto_scheduler_rewritten_layout-meta_schedule_original_shapenone)(data,weight[,bias,out_dtype,…])|topi 中致密的默认实现。这是 matmul_nt 运算符的别名，用于非转置格式的数据张量和转置格式的权重张量。|
|[dense_pack](/docs/api-reference/python-api/tvm-topi#tvmtopinndense_packdata-weight-biasnone-out_dtypenone)(data,weight[,bias,out_dtype])|topi 中 dense_pack 的默认实现。|
|[depth_to_space](/docs/api-reference/python-api/tvm-topi#tvmtopinndepth_to_spacedata-block_size-layoutnchw-modedcr)(data,block_size[,layout,mode])|对数据进行深度到空间的变换。|
|[depthwise_conv2d_NCHWc](/docs/api-reference/python-api/tvm-topi#tvmtopinndepthwise_conv2d_nchwcinput-filter-stride-padding-dilation-layout-out_layout-out_dtypenone)(Input,Filter,…[,…])|深度卷积 NCHW[x]c 前向操作符。|
|[depthwise_conv2d_backward_input_nhwc](/docs/api-reference/python-api/tvm-topi#tvmtopinndepthwise_conv2d_backward_input_nhwcfilter-out_grad-oshape-ishape-stride-padding)(Filter,…)|深度卷积 nhwc 后向 wrt 输入运算符。|
|[depthwise_conv2d_backward_weight_nhwc](/docs/api-reference/python-api/tvm-topi#tvmtopinndepthwise_conv2d_backward_weight_nhwcinput-out_grad-oshape-fshape-stride-padding)(Input,…)|深度卷积 nhwc 后向 wrt 权重运算符。|
|[depthwise_conv2d_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinndepthwise_conv2d_nchwinput-filter-stride-padding-dilation-out_dtypenone)(Input,Filter,stride,…)|深度卷积 nchw 前向操作符。|
|[depthwise_conv2d_nhwc](/docs/api-reference/python-api/tvm-topi#tvmtopinndepthwise_conv2d_nhwcinput-filter-stride-padding-dilation-kernel_layouthwoi-out_dtypenone)(Input,Filter,stride,…)|深度卷积 nhwc 前向操作符。|
|[dilate](/docs/api-reference/python-api/tvm-topi#tvmtopinndilatedata-strides-dilation_value00-namedilatedinput)(data,strides[,dilation_value,name])|使用给定的扩张值（默认为 0）扩张数据。|
|[equal_const_int](/docs/api-reference/python-api/tvm-topi#tvmtopinnequal_const_intexpr-value)(expr,value)|如果 expr 等于 value，则返回。|
|[fast_softmax](/docs/api-reference/python-api/tvm-topi#tvmtopinnfast_softmaxx-axis-1)(x[,axis])|对数据执行 softmax 激活。使用近似值计算指数可以提高速度。|
|[fifo_buffer](/docs/api-reference/python-api/tvm-topi#tvmtopinnfifo_bufferdata-buffer-axis)(data,buffer,axis)|FIFO 缓冲区可在具有滑动窗口输入的 CNN 中实现计算重用。|
|[flatten](/docs/api-reference/python-api/tvm-topi#tvmtopinnflattendata)(data)|通过折叠较高维度将输入数组展平为二维数组。|
|[get_const_int](/docs/api-reference/python-api/tvm-topi#tvmtopinnget_const_intexpr)(expr)|验证 expr 是否为整数并获取常数值。|
|[get_const_tuple](/docs/api-reference/python-api/tvm-topi#tvmtopinnget_const_tuplein_tuple)(in_tuple)|验证输入元组是 IntImm 还是 Var，返回 int 或 Var 的元组。|
|[get_pad_tuple](/docs/api-reference/python-api/tvm-topi#tvmtopinnget_pad_tuplepadding-kernel)(padding,kernel)|获取 pad 选项的通用代码。|
|[get_pad_tuple1d](/docs/api-reference/python-api/tvm-topi#tvmtopinnget_pad_tuple1dpadding-kernel)(padding,kernel)|获取 pad 选项的通用代码。|
|[get_pad_tuple3d](/docs/api-reference/python-api/tvm-topi#tvmtopinnget_pad_tuple3dpadding-kernel)(padding,kernel)|获取 pad 选项的通用代码。|
|[get_pad_tuple_generic](/docs/api-reference/python-api/tvm-topi#tvmtopinnget_pad_tuple_genericpadding-kernel)(padding,kernel)|获取 pad 选项的通用代码。|
|[get_padded_shape](/docs/api-reference/python-api/tvm-topi#tvmtopinnget_padded_shapedata-pad_before-pad_afternone)(data,pad_before[,pad_after])|应用填充后计算张量的输出形状。|
|[global_pool](/docs/api-reference/python-api/tvm-topi#tvmtopinnglobal_pooldata-pool_type-layoutnchw)(data,pool_type[,layout])|对数据的高度和宽度维度进行全局池化。|
|[group_conv1d_ncw](/docs/api-reference/python-api/tvm-topi#tvmtopinngroup_conv1d_ncwdata-kernel-strides1-paddingvalid-dilation1-groups1-out_dtypenone)(data,kernel[,strides,…])|用于 NCW 布局的一维卷积前向操作符。|
|[group_conv1d_nwc](/docs/api-reference/python-api/tvm-topi#tvmtopinngroup_conv1d_nwcdata-kernel-strides1-paddingvalid-dilation1-groups1-out_dtypenone)(data,kernel[,strides,…])|用于 NWC 布局的一维卷积前向操作符。|
|[group_conv1d_transpose_ncw](/docs/api-reference/python-api/tvm-topi#tvmtopinngroup_conv1d_transpose_ncwdata-kernel-stride-padding-out_dtype-output_padding-groups)(data,kernel,…)|转置的一维组卷积 ncw 前向操作符。|
|[group_conv2d_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinngroup_conv2d_nchwinput-filter-stride-padding-dilation-groups-out_dtypenone)(Input,Filter,stride,…)|NCHW 布局中的组卷积操作符。|
|[group_conv2d_nhwc](/docs/api-reference/python-api/tvm-topi#tvmtopinngroup_conv2d_nhwcinput-filter-stride-padding-dilation-groups-out_dtypenone)(Input,Filter,stride,…)|NHWC 布局中的组卷积操作符。|
|[group_conv2d_transpose_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinngroup_conv2d_transpose_nchwdata-kernel-stride-padding-out_dtype-output_padding-groups)(data,kernel,…)|NCHW 布局中的组卷积操作符。|
|[group_conv3d_transpose_ncdhw](/docs/api-reference/python-api/tvm-topi#tvmtopinngroup_conv3d_transpose_ncdhwdata-kernel-strides-padding-out_dtype-output_padding-groups)(data,kernel,…)|转置组3D 卷积 ncdhw 前向操作符。|
|[if_then_else](/docs/api-reference/python-api/tvm-topi#tvmtopinnif_then_elsecond-t-f-spannone)(cond,t,f[,span])|条件选择表达式。|
|[leaky_relu](/docs/api-reference/python-api/tvm-topi#tvmtopinnleaky_relux-alpha)(x,alpha)|取输入 x 的 leaky relu。|
|[log_softmax](/docs/api-reference/python-api/tvm-topi#tvmtopinnlog_softmaxx-axis-1)(x[,axis])|对数据执行对数 softmax 激活|
|[lrn](/docs/api-reference/python-api/tvm-topi#tvmtopinnlrndata-size-axis1-alpha00001-beta075-bias2)(data,size[,axis,alpha,beta,bias])|对输入数据执行跨通道局部响应标准化。|
|[lstm](/docs/api-reference/python-api/tvm-topi#tvmtopinnlstmxs-wi-wh-binone-bhnone-h_initnone-c_initnone-projnone-p_inone-p_fnone-p_onone-f_act-g_act-h_act-reversefalse-weight_layout-str--ifgo)(Xs,Wi,Wh[,Bi,Bh,h_init,c_init,…])|使用 TE 扫描实现的通用 LSTM。|
|[matmul](/docs/api-reference/python-api/tvm-topi#tvmtopinnmatmultensor_a-tensor_b-biasnone-out_dtypenone-transpose_afalse-transpose_bfalse-auto_scheduler_rewritten_layout-meta_schedule_original_shapenone)(tensor_a,tensor_b[,bias,…])|topi 中 matmul 的默认实现。|
|[mirror_pad](/docs/api-reference/python-api/tvm-topi#tvmtopinnmirror_paddata-pad_before-pad_afternone-modesymmetric-namemirrorpadinput)(data,pad_before[,pad_after,…])|具有对称或反射功能的镜像平板输入。|
|[namedtuple](/docs/api-reference/python-api/tvm-topi#tvmtopinnnamedtupletypename-field_names--renamefalse-defaultsnone-modulenone)(typename,field_names,*[,…])|返回具有命名字段的元组的新子类。|
|[nll_loss](/docs/api-reference/python-api/tvm-topi#tvmtopinnnll_losspredictions-targets-weights-reduction-ignore_index)(predictions,targets,weights,…)|输入数据的负对数似然损失。|
|[pad](/docs/api-reference/python-api/tvm-topi#tvmtopinnpaddata-pad_before-pad_afternone-pad_value00-namepadinput-attrsnone)(data,pad_before[,pad_after,…])|使用 pad 值的 Pad 输入。|
|[pool1d](/docs/api-reference/python-api/tvm-topi#tvmtopinnpool1ddata-kernel-stride-dilation-padding-pool_type-ceil_modefalse-layoutncw-count_include_padtrue)(data,kernel,stride,dilation,…)|对数据的宽度维度进行池化。|
|[pool2d](/docs/api-reference/python-api/tvm-topi#tvmtopinnpool2ddata-kernel-stride-dilation-padding-pool_type-ceil_modefalse-layoutnchw-count_include_padtrue)(data,kernel,stride,dilation,…)|对数据的高度和宽度维度进行池化。|
|[pool3d](/docs/api-reference/python-api/tvm-topi#tvmtopinnpool3ddata-kernel-stride-dilation-padding-pool_type-ceil_modefalse-layoutncdhw-count_include_padtrue)(data,kernel,stride,dilation,…)|对数据的深度、高度和宽度维度进行池化。|
|[pool_grad](/docs/api-reference/python-api/tvm-topi#tvmtopinnpool_gradgrads-data-kernel-stride-padding-pool_type-ceil_modefalse-count_include_padtrue-layoutnchw)(grads,data,kernel,stride,…)|池化在数据高度和宽度维度上的梯度。|
|[prelu](/docs/api-reference/python-api/tvm-topi#tvmtopinnprelux-slope-axis1)(x,slope[,axis])|PReLU。它接受两个参数：一个输入x和一个权重数组W ，并计算输出为 PReLU(x)y=x>0?x:W∗x， 在哪里∗是批次中每个样本的元素乘法。|
|[reduce](/docs/api-reference/python-api/tvm-topi#tvmtopinnreducefunction-sequence-initial--value)(function,sequence[,initial])|将一个包含两个参数的函数从左到右累加地应用于序列的项，从而将序列简化为单个值。例如，reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) 计算结果为 ((((1+2)+3)+4)+5)。如果指定了 initial，则在计算过程中将其放置在序列的项之前，并在序列为空时用作默认值。|
|[reflect_pad](/docs/api-reference/python-api/tvm-topi#tvmtopinnreflect_paddata-pad_before-pad_afternone-namereflectpadinput)(data,pad_before[,pad_after,name])|将反射填充应用于输入张量。|
|[relu](/docs/api-reference/python-api/tvm-topi#tvmtopinnrelux)(x)|取输入 x 的 relu。|
|[replicate_pad](/docs/api-reference/python-api/tvm-topi#tvmtopinnreplicate_paddata-pad_before-pad_afternone-namereplicatepadinput)(data,pad_before[,pad_after,…])|对输入张量应用重复填充（边缘填充）。|
|[scale_shift_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinnscale_shift_nchwinput-scale-shift)(Input,Scale,Shift)|推理中的批量标准化运算符。|
|[scale_shift_nchwc](/docs/api-reference/python-api/tvm-topi#tvmtopinnscale_shift_nchwcinput-scale-shift)(Input,Scale,Shift)|推理中的批量标准化运算符。|
|[scale_shift_nhwc](/docs/api-reference/python-api/tvm-topi#tvmtopinnscale_shift_nhwcinput-scale-shift)(Input,Scale,Shift)|推理中的批量标准化运算符。|
|[simplify](/docs/api-reference/python-api/tvm-topi#tvmtopinnsimplifyexpr)(expr)|如果是 Expr 则化简表达式，如果是 int 则直接返回。|
|[simulated_dequantize](/docs/api-reference/python-api/tvm-topi#tvmtopinnsimulated_dequantizedata-in_dtype-input_scalenone-input_zero_pointnone-axis-1)(data,in_dtype[,…])|模拟 QNN 反量化运算符，可模拟 QNN 输出，而无需更改数据类型。与真正的 QNN 反量化相比，此运算符的优势在于，它允许动态选择数据类型，并且可以对每个通道、标量尺度和零点进行操作，而 QNN 反量化则需要在编译时修复这两者。|
|[simulated_quantize](/docs/api-reference/python-api/tvm-topi#tvmtopinnsimulated_quantizedata-out_dtype-output_scalenone-output_zero_pointnone-axis-1)(data,out_dtype[,…])|模拟 QNN 量化运算符，可模拟 QNN 输出，无需更改数据类型。与真正的 QNN 量化相比，此运算符的优势在于，它允许动态选择数据类型，并且可以对每个通道、标量尺度和零点进行操作，而 QNN 量化则要求在编译时固定这两个参数。|
|[softmax](/docs/api-reference/python-api/tvm-topi#tvmtopinnsoftmaxx-axis-1)(x[,axis])|对数据执行 softmax 激活。|
|[softmax_common](/docs/api-reference/python-api/tvm-topi#tvmtopinnsoftmax_commonx-axis-use_fast_exp)(x,axis,use_fast_exp)|softmax 和 fast_softmax 的共同部分。|
|[softplus](/docs/api-reference/python-api/tvm-topi#tvmtopinnsoftplusx-beta10-threshold200)(x[,beta,threshold])|计算具有数值稳定性的输入 x 的 Softplus 激活。|
|[space_to_batch_nd](/docs/api-reference/python-api/tvm-topi#tvmtopinnspace_to_batch_nddata-block_shape-pad_before-pad_after-pad_value00)(data,block_shape,…[,…])|对数据执行批量到空间的转换。|
|[space_to_depth](/docs/api-reference/python-api/tvm-topi#tvmtopinnspace_to_depthdata-block_size-layoutnchw)(data,block_size[,layout])|对数据执行空间到深度的转换。|
|[strided_slice](/docs/api-reference/python-api/tvm-topi#tvmtopinnstrided_slicea-begin-end-stridesnone-axesnone-slice_modeend-assume_inboundtrue)(a,begin,end[,strides,…])|数组的切片。|
|[unpack_NCHWc_to_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopinnunpack_nchwc_to_nchwpacked_out-out_dtype)(packed_out,out_dtype)|将 conv2d_NCHWc 输出从布局 NCHWc 解包为 NCHW。|
|[upsampling](/docs/api-reference/python-api/tvm-topi#tvmtopinnupsamplingdata-scale_h-scale_w-layoutnchw-methodnearest_neighbor-align_cornersfalse-output_shapenone)(data,scale_h,scale_w[,layout,…])|对数据执行上采样。|
|[upsampling3d](/docs/api-reference/python-api/tvm-topi#tvmtopinnupsampling3ddata-scale_d-scale_h-scale_w-layoutncdhw-methodnearest_neighbor-coordinate_transformation_modehalf_pixel-output_shapenone)(data,scale_d,scale_h,scale_w)|对数据执行上采样。|
|[winograd_transform_matrices](/docs/api-reference/python-api/tvm-topi#tvmtopinnwinograd_transform_matricestile_size-kernel_size-out_dtype)(tile_size,…)|将 tile_size 的 A、B 和 G 变换矩阵计算为 tvm.Expr。|
|[instance_norm](/docs/api-reference/python-api/tvm-topi#tvmtopinninstance_normdata-gamma-beta-channel_axis-axis-epsilon1e-05)(data,gamma,beta,…[,epsilon])|实例规范化运算符。|
|[layer_norm](/docs/api-reference/python-api/tvm-topi#tvmtopinnlayer_normdata-gamma-beta-axis-epsilon1e-05)(data,gamma,beta,axis[,epsilon])|层归一化运算符。它接受 fp16 和 fp32 作为输入数据类型。它会将输入转换为 fp32 来执行计算。输出将具有与输入相同的数据类型。|
|[group_norm](/docs/api-reference/python-api/tvm-topi#tvmtopinngroup_normdata-gamma-beta-num_groups-channel_axis-axes-epsilon1e-05)(data,gamma,beta,num_groups,…)|组规范化运算符。它接受 fp16 和 fp32 作为输入数据类型。它会将输入转换为 fp32 来执行计算。输出将具有与输入相同的数据类型。|
|[rms_norm](/docs/api-reference/python-api/tvm-topi#tvmtopinnrms_normdata-weight-axis-epsilon1e-05)(data,weight,axis[,epsilon])|均方根归一化运算符。输出将具有与输入相同的数据类型。|

## *class* tvm.topi.nn.Workload(*in_dtype*, *out_dtype*, *height*, *width*, *in_filter*, *out_filter*, *kernel_h*, *kernel_w*, *padt*, *padl*, *padb*, *padr*, *dilation_h*, *dilation_w*, *stride_h*, *stride_w*)

**属性：**

|[dilation_h](/docs/api-reference/python-api/tvm-topi#dilation_h)|字段编号 12 的别名。|
|:----|:----|
|[dilation_w](/docs/api-reference/python-api/tvm-topi#dilation_w)|字段编号 13 的别名。|
|[height](/docs/api-reference/python-api/tvm-topi#height)|字段号 2 的别名。|
|[in_dtype](/docs/api-reference/python-api/tvm-topi#in_dtype)|字段编号 0 的别名。|
|[in_filter](/docs/api-reference/python-api/tvm-topi#in_filter)|字段号 4 的别名。|
|[kernel_h](/docs/api-reference/python-api/tvm-topi#kernel_h)|字段号 6 的别名。|
|[kernel_w](/docs/api-reference/python-api/tvm-topi#kernel_w)|字段号 7 的别名。|
|[out_dtype](/docs/api-reference/python-api/tvm-topi#out_dtype)|字段号 1 的别名。|
|[out_filter](/docs/api-reference/python-api/tvm-topi#out_filter)|字段编号 5 的别名。|
|[padb](/docs/api-reference/python-api/tvm-topi#padb)|字段编号 10 的别名。|
|[padl](/docs/api-reference/python-api/tvm-topi#padl)|字段编号 9 的别名。|
|[padr](/docs/api-reference/python-api/tvm-topi#padr)|字段编号 11 的别名。|
|[padt](/docs/api-reference/python-api/tvm-topi#padt)|字段号 8 的别名。|
|[stride_h](/docs/api-reference/python-api/tvm-topi#stride_h)|字段编号 14 的别名。|
|[stride_w](/docs/api-reference/python-api/tvm-topi#stride_w)|字段编号 15 的别名。|
|[width](/docs/api-reference/python-api/tvm-topi#width)|字段号 3 的别名。|

### dilation_h


字段编号 12 的别名。

### dilation_w


字段编号 13 的别名。

### height

字段号 2 的别名。

### in_dtype


字段编号 0 的别名。

### in_filter


字段号 4 的别名。

### kernel_h


字段号 6 的别名。

### kernel_w


字段号 7 的别名。

### out_dtype


字段号 1 的别名。

### out_filter


字段编号 5 的别名。

### padb


字段编号 10 的别名。

### padl


字段编号 9 的别名。

### padr


字段编号 11 的别名。

### padt


字段号 8 的别名。

### stride_h


字段编号 14 的别名。

### stride_w

字段编号 15 的别名。

### width


字段号 3 的别名。

## tvm.topi.nn.adaptive_pool(*data*, *output_size*, *pool_type*, *layout='NCHW'*)


对数据的高度和宽度维度进行池化。


池化核和步长大小会根据所需的输出大小自动选择。它根据布局字符串确定高度和宽度，其中「W」和「H」分别表示宽度和高度。宽度和高度不能拆分。例如，NCHW、NCHW16c 等适用于池化，而 NCHW16w、NCHW16h 则不适用。有关布局字符串约定的更多信息，请参阅参数布局。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：具有布局形状的 nD。
   * **output_size**（[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)*：* 输出高度和宽度。
   * **pool_type** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：池类型，“max”或“avg”。
   * **layout**（*字符串*）：输入数据的布局。布局应该由大写字母、小写字母和数字组成，其中大写字母表示维度，对应的小写字母（因子大小）表示分割维度。例如，NCHW16c 可以描述一个 5 维张量，其值为 [batch_size, channel, height, width, channel_block]，其中 channel_block=16 表示对 channel 维度的分割。
* **返回：output** ：nD 在同一布局中。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.adaptive_pool1d(*data*, *output_size*, *pool_type*, *layout='NCW'*)


对三维数据进行池化。详情请参阅上面的二维版本。

## tvm.topi.nn.adaptive_pool3d(*data*, *output_size*, *pool_type*, *layout='NCDHW'*)


对三维数据进行池化。详情请参阅上面的二维版本。

## tvm.topi.nn.add(*lhs*, *rhs*)


自动广播附加功能。
* **参数：**
   * **lhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）*：* 左操作数
   * **rhs**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*或Expr*）*：* 右操作数
* **返回：ret**：如果两个操作数都是 Expr，则返回 Expr。否则返回 Tensor。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) or Expr

### tvm.topi.nn.batch_matmul(*tensor_a*, *tensor_b*, *oshape=None*, *out_dtype=None*, *transpose_a=False*, *transpose_b=True*, *auto_scheduler_rewritten_layout=''*, *meta_schedule_original_shape=None*)

计算 tensor_a 和 tensor_b 的批量矩阵乘法。


tensor_a 和 tensor_b 均可转置。由于历史原因，我们默认使用 NT 格式（transpose_a=False，transpose_b=True）。
* **参数：**
   * **tensor_a** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：三维，形状为 [batch, M, K] 或 [batch, K, M]。
   * **tensor_b** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：三维，形状为 [batch, K, N] 或 [batch, N, K]。
   * *oshape*（*列表**[**可选**]）：计算的明确预期输出形状。在输入形状动态变化的情况下很有用。
   * *out_dtype**（*可选**[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]*）：指定混合精度批量 matmul 的输出数据类型。
   * *transpose_a* (*可选[*[bool](https://docs.python.org/3/library/functions.html#bool)*]= False* )*：* 第一个张量是否为转置格式。
   * *transpose_b* (*可选[*[bool](https://docs.python.org/3/library/functions.html#bool)*]= True* )[：](https://docs.python.org/3/library/functions.html#float)第二个张量是否为转置格式。
   * *auto_scheduler_rewritten_layout**（*可选**[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]= ""*）：自动调度程序布局重写传递后的布局。
   * **meta_schedule_original_shape** ( *Optional[List[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]= None* )**：** 张量的原始形状
* **返回：output**：三维，形状为 [batch, M, N]
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)

## tvm.topi.nn.batch_norm(*data:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *gamma:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *beta:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *moving_mean:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *moving_var:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *axis:*[int](https://docs.python.org/3/library/functions.html#int)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *epsilon:*[float](https://docs.python.org/3/library/functions.html#float)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *center:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *scale:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *training:*[bool](https://docs.python.org/3/library/functions.html#bool)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *momentum:*[float](https://docs.python.org/3/library/functions.html#float)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)]


批量标准化层（Ioffe 和 Szegedy，2014）。


对每个批次的输入进行标准化，即应用保持平均激活接近 0 且激活标准差接近 1 的变换。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：要批量标准化的输入。
   * **gamma**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 应用于标准化张量的比例因子。
   * **beta**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：应用于标准化张量的偏移量。
   * **moving_mean**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入的运行平均值。
   * **moving_var**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入的运行方差。
   * *axis*（**[int](https://docs.python.org/3/library/functions.html#int)**，**可选**，默认值 = 1）：指定应沿哪个形状轴进行规范化。
   * *epsilon*（**[浮点数](https://docs.python.org/3/library/functions.html#float)**，**可选**，默认值 = 1e–5）：添加到方差的小浮点数，以避免除以零。
   * *center*（**[bool](https://docs.python.org/3/library/functions.html#bool)**，**可选**，默认 = True）：如果为 True，则将 beta 的偏移量添加到标准化张量，如果为 False，则忽略 beta。
   * *scale*（**[bool](https://docs.python.org/3/library/functions.html#bool)**，**可选**，默认=True）：如果为 True，则按 gamma 缩放标准化张量。如果为 False，则忽略 gamma。
   * *training*（**[bool](https://docs.python.org/3/library/functions.html#bool)**，**可选**，默认=False）：指示是否处于训练模式。如果为 True，则更新 moving_mean 和 moving_var。
   * *动量*（**[浮点数](https://docs.python.org/3/library/functions.html#float)**，**可选**，默认值 = 0.1）：用于 moving_mean 和 moving_var 更新的值。
* **返回：** 
   * **输出**（*tvm.te.Tensor 列表*）*：* 与输入具有相同形状的标准化数据
   * **moving_mean** (*tvm.te.Tensor*)**：moving_mean**（*tvm.te.Tensor*）：输入的运行平均值。
   * **moving_var** (*tvm.te.Tensor*)：**moving_var**（*tvm.te.Tensor*）：输入的运行方差。

## tvm.topi.nn.batch_to_space_nd(*data*, *block_shape*, *crop_begin_list*, *crop_end_list*)

对数据执行空间到批量的转换。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 形状为 [batch, spatial_shape, remaining_shapes] 的 ND Tensor，其中 spatial_shape 有 M 维。
   * **block_shape**（*整数*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：大小为 [M] 的列表，其中 M 是空间维度的数量，指定每个空间维度的块大小。
   * **crop_begin_list**（*整数*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：形状为 [M] 的列表，其中 M 是空间维度的数量，指定每个空间维度的开始裁剪大小。
   * **crop_end_list**（*整数*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：形状为 [M] 的列表，其中 M 是空间维度的数量，指定每个空间维度的最终裁剪大小。
* **返回：output。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.binarize_pack(*data*, *axis=None*, *name='PackedInput'*)


沿某个轴进行二值化和位打包。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：nD 输入，可以是任何布局。
   * **axis**（*None或*[int](https://docs.python.org/3/library/functions.html#int)）：进行二值化和位打包的轴，默认为最后一个轴。
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：运算符生成的名称前缀。
* **返回：output**：nD，与输入相同的布局，dtype 为 uint32。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.binary_dense(*data*, *weight*)


使用异或和位计数进行二进制矩阵乘法。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：二维，形状为 [batch, in_dim]，dtype 为 uint32。
   * **权重**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 二维，形状为 [out_dim, in_dim]，dtype 为 uint32。
* **返回：output**：二维，形状为 [batch, out_dim]，dtype 为 float32。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.bitpack(*data*, *bits*, *pack_axis*, *bit_axis*, *pack_type*, *name='QuantizeInput'*)


将数据打包成位串行计算所需的格式。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入 tvm 张量。
   * **位**（[int](https://docs.python.org/3/library/functions.html#int)）：用于打包的位数。
   * **pack_axis** ( [int](https://docs.python.org/3/library/functions.html#int) )：数据打包轴的索引。
   * **bit_axis** ( [int](https://docs.python.org/3/library/functions.html#int) )：在结果打包数据中放置位轴的轴索引。
   * **pack_type** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：打包的数据类型，必须是以下之一：['uint8', 'uint16', 'uint32', 'uint64']。
   * *name**（*可选**[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]= "QuantizeInput"*）：操作的名称。

## tvm.topi.nn.bitserial_conv2d_nchw(*data*, *kernel*, *stride*, *padding*, *activation_bits*, *weight_bits*, *pack_dtype='uint32'*, *out_dtype='int16'*, *unipolar=True*)


Bitserial Conv2D 运算符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_channel, in_height, in_width]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[num_filter，in_channel，filter_height，filter_width]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或**两个 int***的列表/元组）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个**或**四个 int*的列表/元组*）：填充大小，[pad_height, pad_width], [pad_top, pad_left, pad_down, pad_right]。
   * **激活位**( [int](https://docs.python.org/3/library/functions.html#int) )：用于激活/输入元素的位数。
   * **weight_bits** ( [int](https://docs.python.org/3/library/functions.html#int) )：用于权重元素的位数。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：返回卷积类型。
   * **pack_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )*：* 位打包类型。
   * **单极**（[bool](https://docs.python.org/3/library/functions.html#bool)）：如果二值化样式为单极 1/0 格式，而不是双极 –1/+1 格式。
* **返回：output**：4–D，形状为[batch, out_channel, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.bitserial_conv2d_nhwc(*data*, *kernel*, *stride*, *padding*, *activation_bits*, *weight_bits*, *pack_dtype='uint32'*, *out_dtype='int16'*, *unipolar=True*)


Bitserial Conv2D 运算符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_height, in_width, in_channel]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[filter_height，filter_width，in_channel，num_filter]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)或 **两个 int** *的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或两个****或四个 int****的列表/元组*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)填充大小，[pad_height, pad_width], [pad_top, pad_left, pad_down, pad_right]。
   * **激活位**( [int](https://docs.python.org/3/library/functions.html#int) )：用于激活/输入元素的位数。
   * **weight_bits** ( [int](https://docs.python.org/3/library/functions.html#int) )：用于权重元素的位数。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)返回卷积类型。
   * **pack_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：位打包类型。
   * **单极**（[bool](https://docs.python.org/3/library/functions.html#bool)）：如果二值化样式为单极 1/0 格式，而不是双极 –1/+1 格式。
* **返回：output**：4–D，形状为[batch, out_height, out_width, out_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.bitserial_dense(*data*, *weight*, *data_bits*, *weight_bits*, *pack_dtype='uint32'*, *out_dtype='int16'*, *unipolar=True*)


topi 中位串行密集的默认实现。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 二维，形状为[batch, in_dim]。
   * **权重**( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)形状为 [out_dim, in_dim] 的二维或​​形状为 [out_dim, weight_bits, in_dim] 的三维。
* **返回：output**[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)二维，形状为 [batch, out_dim]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.circular_pad(*data*, *pad_before*, *pad_after=None*, *name='CircularPadInput'*)


对输入张量应用圆形填充（环绕）。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入张量。
   * *pad_before* ( *List[*[int](https://docs.python.org/3/library/functions.html#int)*]* )*：* 每个维度前填充的量。
   * **pad_after**（*List[*[int](https://docs.python.org/3/library/functions.html#int)*]，可选*）**：** 每个维度后的填充量。如果为 None ，则默认为 pad_before 。
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)）*：* 结果张量的名称。
* **返回：out** *：* 圆形填充张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.concatenate(*a_tuple*, *axis=0*)


沿现有轴连接一系列数组。
* **参数：**
   *   **a_tuple**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*的*[元组](https://docs.python.org/3/library/stdtypes.html#tuple)）*：* 要连接的数组。
   *   **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：数组连接的轴。默认值为 0。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv(*inp:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *filt:*[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), *stride:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *padding:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *dilation:*[int](https://docs.python.org/3/library/functions.html#int)*|*[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)*[*[int](https://docs.python.org/3/library/functions.html#int)*]*, *groups:*[int](https://docs.python.org/3/library/functions.html#int), *data_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str), *kernel_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*= ''*, *out_dtype:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *auto_scheduler_rewritten_layout:*[str](https://docs.python.org/3/library/stdtypes.html#str)*|*[None](https://docs.python.org/3/library/constants.html#None)*= None*, *meta_schedule_original_shape=None*, *auto_scheduler_should_rewrite_layout:*[bool](https://docs.python.org/3/library/functions.html#bool)*= False*)


NCHW 或 NHWC 布局中的卷积操作符。


支持 1D、2D、3D …… 和分组。
* **参数：**
   * **inp**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) ）：data_layout 中形状为 [batch, in_channel, in_height, in_width, …] 的 ND。
   * **filt**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) ）：kernel_layout 中形状为 [num_filter, in_channel//groups, filter_height, filter_width, …] 的 ND。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或dim ints列表/元组）*：（其中 dim=2 表示 NCHW，dim=1 表示 NCH，等等）步幅大小，或 [stride_height, stride_width, …]。
   * ***padding***（*[int](https://docs.python.org/3/library/functions.html#int)或**dim**或2*dim ints的列表/元组）：（其中 dim=2 表示 NCHW，dim=1 表示 NCH，等等）填充大小，或 dim ints 的 [pad_height, pad_width, …]，或 2*dim ints 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）：组数。
   * **data_layout** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输入的布局。N 表示批次维度，C 表示通道数，其他字符表示 HW（对于 1D 和 3D，则表示 H 或 HWD）。
   * *kernel_layout*（可选[ [str](https://docs.python.org/3/library/stdtypes.html#str)]）**：** 滤波器的布局。I 表示输入通道数，O 表示输出通道数，其他字符表示滤波器的 HW 维度（一维和三维时为 H 或 HWD）。如果 kernel_layout 为空，则使用 data_layout 推断默认的 kernel_layout。默认 kernel_layout 为 OIHW（NCHW 数据布局）或 HWIO（NHWC 数据布局）。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)在逐元素乘法和求和之前，元素被转换为此类型。
   * **auto_scheduler_rewritten_layout** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：来自自动调度程序布局重写的布局。
   * **meta_schedule_original_shape** (*可选*[***列表[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]* )[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)输入张量的原始形状。
   * **auto_scheduler_should_rewrite_layout** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：是否允许自动调度程序重写滤波器张量的布局。默认为 false。如果与分组卷积一起使用，可能会导致错误。
* **返回：Output**：data_layout 中形状为 [batch, out_channel, out_height, out_width, …] 的 ND。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv1d(*data*, *kernel*, *strides=1*, *padding='VALID'*, *dilation=1*, *groups=1*, *data_layout='NCW'*, *kernel_layout=''*, *out_dtype=None*) 


1D 卷积前向操作符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：data_layout == 'NCW' 的 3–D 输入形状 [batch, in_width, in_channel] 和 data_layout == 'NWC' 的 [batch, in_width, in_channel]。
   * **kernel**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：3D 内核，其中 kernel_layout == 'OIW' 的形状为 [num_filter, in_channel, filter_size]，而 kernel_layout == 'WIO' 的形状为 [filter_size, in_channel, num_filter]。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或*[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)）*：* 沿宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **dilation**（[整数](https://docs.python.org/3/library/functions.html#int)*或*[元组](https://docs.python.org/3/library/stdtypes.html#tuple)）：如果卷积需要扩张，则扩张率。
   * **data_layout** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )*：* 输入数据的布局方式，必须是 ['NCW', 'NWC'] 之一。
   * *kernel_layout* ( *Optiona[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]* )：内核布局。如果未指定，则使用默认布局。如果 data_layout == “NCW”，则为“OIW”；如果 data_layout == “NWC”，则为“WIO”。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据类型。如果为 None ，则输出与输入类型相同。

## tvm.topi.nn.conv1d_ncw(*data*, *kernel*, *strides=1*, *padding='VALID'*, *dilation=1*, *out_dtype=None*)


NCW 布局中的一维卷积。有关参数的详细信息，请参阅 [conv()](/docs/api-reference/python-api/tvm-topi#tvmtopinnconvinptensor-filttensor-strideintsequenceint-paddingintsequenceint-dilationintsequenceint-groupsint-data_layoutstr-kernel_layoutstr--out_dtypestrnone-none-auto_scheduler_rewritten_layoutstrnone-none-meta_schedule_original_shapenone-auto_scheduler_should_rewrite_layoutbool-false)。

## tvm.topi.nn.conv1d_nwc(*data*, *kernel*, *strides=1*, *padding='VALID'*, *dilation=1*, *out_dtype=None*)


NWC 布局中的一维卷积。有关参数的详细信息，请参阅 [conv()](/docs/api-reference/python-api/tvm-topi#tvmtopinnconvinptensor-filttensor-strideintsequenceint-paddingintsequenceint-dilationintsequenceint-groupsint-data_layoutstr-kernel_layoutstr--out_dtypestrnone-none-auto_scheduler_rewritten_layoutstrnone-none-meta_schedule_original_shapenone-auto_scheduler_should_rewrite_layoutbool-false)。

## tvm.topi.nn.conv1d_transpose_ncw(*data*, *kernel*, *stride*, *padding*, *out_dtype*, *output_padding*)


转置的一维卷积 ncw 前向操作符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：三维，形状为[batch, in_channel, in_width]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：3–D，形状为[in_channel，num_filter，filter_width]。
   * **步幅**（*ints*）*：* 沿宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）*：* 填充大小，或 ['VALID', 'SAME']。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据类型。用于混合精度。
   * **output_padding**（*ints*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)用于在存在多个可能形状的情况下恢复实际的输出形状。必须小于步幅。
* **返回：output**：3–D，形状为[batch, out_channel, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d(*input*, *filter*, *strides*, *padding*, *dilation*, *data_layout='NCHW'*, *kernel_layout=''*, *out_dtype=None*)


Conv2D 运算符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 4–D，形状为 [batch, in_channel, in_height, in_width]，位于 data_layout 中。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为 [num_filter, in_channel, filter_height, filter_width]，位于 kernel_layout 中。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int*的列表/元组）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)或 2 个或 4 个 int的列表/元组）：填充大小，或 2 个 int 的 [pad_height, pad_width]，或 4 个 int 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * **data_layout** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：数据布局。
   * *kernel_layout*（*可选*[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]*）*：* 内核布局。如果未指定，则使用从 data_layout 推断出的默认布局。如果 data_layout == “NCHW”，则为“OIHW”；如果 data_layout == “NHWC”，则为“HWIO”。
* **返回：output**：4–D，形状为[batch, out_channel, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_NCHWc(*data*, *kernel*, *stride*, *padding*, *dilation*, *layout*, *out_layout*, *out_dtype='float32'*)


nChw[x]c 布局的 Conv2D 运算符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：5–D，形状为[batch，in_channel_chunk，in_height，in_width，in_channel_block]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：6D，形状为[num_filter_chunk，in_channel_chunk，filter_height，filter_width，in_channel_block，num_filter_block]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int****的列表/元组*）*：* 步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)或2 个或4 个 int的列表/元组）：填充大小，或 2 个 int 的 [pad_height, pad_width]，或 4 个 int 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int的列表/元组*）*：* 扩张大小，或 [dilation_height, dilation_width]。
   * **布局**（[str](https://docs.python.org/3/library/stdtypes.html#str)）*：* 输入数据布局。
   * **out_layout** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据布局。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据类型。
* **返回：output**：5–D，形状为[batch, out_channel_chunk, out_height, out_width, out_channel_block]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_NCHWc_int8(*data*, *kernel*, *stride*, *padding*, *dilation*, *layout*, *out_layout*, *out_dtype='int32'*, *n_elems=4*)


nChw[x]c 布局的 Conv2D 运算符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：5–D，形状为[batch，in_channel_chunk，in_height，in_width，in_channel_block]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：7–D，形状为[num_filter_chunk，in_channel_chunk，filter_height，filter_width，in_channel_block/4，num_filter_block，4]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int 的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)或 2 个或 4 个 int 的列表/元组）：填充大小，或 2 个 int 的 [pad_height, pad_width]，或 4 个 int 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int 的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * **布局**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：输入数据布局。
   * **out_layout** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据布局。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据类型。
   * **n_elems** ( [int](https://docs.python.org/3/library/functions.html#int) )：累计的 int8 元素数量。
* **返回：output** *：* 5–D，形状为[batch, out_channel_chunk, out_height, out_width, out_channel_block]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_hwcn(*Input*, *Filter*, *stride*, *padding*, *dilation*, *out_dtype=None*)


HWCN 布局中的卷积操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[in_height, in_width, in_channel, batch]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 4–D，形状为[filter_height，filter_width，in_channel，num_filter]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int 的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或**2 个****或**4 个 int****的列表/元组*）：填充大小，或 2 个 int 的 [pad_height, pad_width]，或 4 个 int 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int 的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
* **返回：output**：4–D，形状为[out_height, out_width, out_channel, batch]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_nchw(*Input*, *Filter*, *stride*, *padding*, *dilation*, *out_dtype=None*)


NCHW 布局中的卷积操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_channel, in_height, in_width]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](https://docs.python.org/3/library/functions.html#int)4–D，形状为[num_filter，in_channel，filter_height，filter_width]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int 的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或2 个或 4 个 int 的列表/元组*）：填充大小，或 2 个 int 的 [pad_height, pad_width]，或 4 个 int 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int 的列表/元组*）*：* 扩张大小，或 [dilation_height, dilation_width]。
* **返回：Output**：4–D，形状为 [batch, out_channel, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_nhwc(*Input*, *Filter*, *stride*, *padding*, *dilation*, *out_dtype='float32'*, *auto_scheduler_rewritten_layout=''*, *meta_schedule_original_shape=None*)

NHWC 布局中的卷积操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_height, in_width, in_channel]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）***：*** 4–D，形状为[filter_height，filter_width，in_channel，num_filter]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或**两个 int**的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或**2 个**或**4 个 int**的列表/元组*）：填充大小，或 2 个 int 的 [pad_height, pad_width]，或 4 个 int 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或**两个 int**的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * *out_dtype* ( *str = "float32",* )：输出张量的类型。
   * **auto_scheduler_rewritten_layout** ( *str = ""* )：自动调度程序布局重写传递后的布局。
   * **meta_schedule_original_shape** ( *Optional[List[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]= None* )：输入张量的原始形状。
* **返回：output**：4–D，形状为[batch, out_height, out_width, out_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_transpose_nchw(*Input*, *Filter*, *strides*, *padding*, *out_dtype*, *output_padding*)


转置的二维卷积 nchw 前向操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_channel, in_height, in_width]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 4–D，形状为[in_channel，num_filter，filter_height，filter_width]。
   * **strides**（*两个整数的元*[组](https://docs.python.org/3/library/stdtypes.html#tuple)）：沿高度和宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据类型。用于混合精度。
   * **output_padding**（*整数元*[组](https://docs.python.org/3/library/stdtypes.html#tuple)）：用于获取渐变的正确输出形状。
* **返回：Output** *：* 4–D，形状为 [batch, out_channel, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_transpose_nchw_preprocess(*data*, *kernel*, *strides*, *padding*, *out_dtype*, *output_padding*)


预处理数据和内核，使 conv2d_transpose 的计算模式与 conv2d 相同。

## tvm.topi.nn.conv2d_winograd_nchw(*data*, *weight*, *strides*, *padding*, *dilation*, *out_dtype*, *pre_computed=False*, *auto_scheduler_rewritten_layout=''*, *meta_schedule_original_shape=None*)

NCHW 布局中的 Conv2D Winograd。这是一个干净的版本，可供 CPU 和 GPU 的自动调度程序使用。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 4–D，形状为[batch, in_channel, in_height, in_width]。
   * **权重**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[filter_height，filter_width，in_channel，num_filter]。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int的列表/元组*）*：* 步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int 的列表/元组）：填充大小，或 [pad_height, pad_width]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)或两个 int 的列表/元组）：扩张大小，或 [dilation_height, dilation_width]。
   * **out_dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：指定输出数据类型。
   * **pre_computed** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：内核是否预先计算。
   * **auto_scheduler_rewritten_layout** ( *str = ""* )**：**自动调度程序布局重写传递后的布局。
   * **meta_schedule_original_shape** ( *Optional[List[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]= None* )：输入张量的原始形状。
* **返回：output**：4–D，形状为[batch, out_height, out_width, out_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_winograd_nchw_without_weight_transform(*data*, *weight*, *strides*, *padding*, *dilation*, *out_dtype*, *auto_scheduler_rewritten_layout=''*, *meta_schedule_original_shape=None*)


在 NCHW 布局中，Conv2D Winograd 无需布局变换。这是一个可供 CPU 和 GPU 元调度使用的干净版本。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_height, in_width, in_channel]。
   * **权重**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[filter_height，filter_width，in_channel，num_filter]。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int***的列表/元组*）*：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或两个 int*的列表/元组*）：填充大小，或 [pad_height, pad_width]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或**两个 int**的列表/元组*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)扩张大小，或 [dilation_height, dilation_width]。
   * **out_dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）*：* 指定输出数据类型。
   * **auto_scheduler_rewritten_layout** ( *str = ""* )*：*自动调度程序布局重写传递后的布局。
   * **meta_schedule_original_shape** ( *Optional[List[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]= None* )：输入张量的原始形状。
* **返回：output**[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)4–D，形状为[batch, out_height, out_width, out_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_winograd_nhwc(*data*, *weight*, *strides*, *padding*, *dilation*, *out_dtype*, *pre_computed=False*, *auto_scheduler_rewritten_layout=''*, *meta_schedule_original_shape=None*)


NHWC 布局中的 Conv2D Winograd。这是一个干净的版本，可供 CPU 和 GPU 的自动调度程序使用。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_height, in_width, in_channel]。
   * **权重**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 4–D，形状为[filter_height，filter_width，in_channel，num_filter]。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）*：* 步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：填充大小，或 [pad_height, pad_width]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * **out_dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：指定输出数据类型。
   * **pre_computed** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：内核是否预先计算。
   * **auto_scheduler_rewritten_layout** ( *str = ""* )*：* 自动调度程序布局重写传递后的布局。
   * **meta_schedule_original_shape** ( *Optional[List[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]= None* )：输入张量的原始形状。
* **返回：output**：4–D，形状为[batch, out_height, out_width, out_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_winograd_nhwc_without_weight_transform(*data*, *weight*, *strides*, *padding*, *dilation*, *out_dtype*, *auto_scheduler_rewritten_layout=''*, *meta_schedule_original_shape=None*)


Conv2D Winograd 在 NHWC 布局中无需布局变换。这是一个干净的版本，可供 CPU 和 GPU 自动调度程序使用。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_height, in_width, in_channel]。
   * **权重**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：**4–D，形状为[filter_height，filter_width，in_channel，num_filter]。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）**：** 填充大小，或 [pad_height, pad_width]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)扩张大小，或 [dilation_height, dilation_width]。
   * **out_dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）*：* 指定输出数据类型。
   * **auto_scheduler_rewritten_layout** ( *str = ""* )：自动调度程序布局重写传递后的布局。
   * **meta_schedule_original_shape** ( *Optional[List[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]= None* )：输入张量的原始形状。
* **返回：output**：4–D，形状为[batch, out_height, out_width, out_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv2d_winograd_weight_transform(*kernel*, *tile_size*)


winograd 的权重转换。
* **参数：**
   * **核**（[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)布局为「NCHW」的原始核张量。
   * **tile_size** ( [int](https://docs.python.org/3/library/functions.html#int) )：winograd 变换的 Tile 大小。例如，F(2x2, 3x3) 为 2，F(4x4, 3x3) 为 4。
* **返回：output**：4–D，形状为 [alpha, alpha, CO, CI]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv3d_ncdhw(*Input*, *Filter*, *stride*, *padding*, *dilation*, *groups*, *out_dtype=None*)


NCDHW 布局中的 Conv3D 运算符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）***：*** 5–D，形状为[batch, in_channel, in_depth, in_height, in_width]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：5–D，形状为[num_filter，in_channel，filter_depth，filter_height，filter_width]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或*三个 int*的列表/元组*）：步幅大小，或 [strid_depth, stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或*三个 int*的列表/元组*）*：* 扩张大小，或 [dilation_depth, dilation_height, dilation_width]。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)组数。
* **返回：Output**：5–D，形状为[batch, out_channel, out_depth, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv3d_ndhwc(*Input*, *Filter*, *stride*, *padding*, *dilation*, *groups*, *out_dtype='float32'*, *auto_scheduler_rewritten_layout=''*, *meta_schedule_origin_shape=None*)


NDHWC 布局中的卷积操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 5–D，形状为[batch, in_depth, in_height, in_width, in_channel]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：5–D，形状为[filter_depth，filter_height，filter_width，in_channel，num_filter]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或**三个 int**的列表/元组*）：步幅大小，或 [stride_depth, stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或**三个 int**的列表/元组*）：扩张大小，或 [dilation_depth, dilation_height, dilation_width]。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）：组数。
   * out_dtype ( *str = "float32",* )：输出张量的类型。
   * **auto_scheduler_rewritten_layout** ( *str = ""* )：自动调度程序布局重写传递后的布局。
   * **meta_schedule_origin_shape** ( *Optional[List[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]= None* )：输入张量的原始形状。
* **返回：Output**：5–D，形状为 [batch, out_depth, out_height, out_width, out_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv3d_transpose_ncdhw(*Input*, *Filter*, *strides*, *padding*, *out_dtype*, *output_padding*)


转置的3D 卷积 ncdhw 前向操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：5–D，形状为[batch, in_channel, in_depth, in_height, in_width]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：5–D，形状为[in_channel，num_filter，filter_depth，filter_height，filter_width]。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或**三个 int***的列表/元组）：沿深度、高度和宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )*：* 输出数据类型。用于混合精度。
   * **output_padding**（*整数元*[组](https://docs.python.org/3/library/stdtypes.html#tuple)）：用于获取渐变的正确输出形状。
* **返回：Output**  ：5–D，形状为[batch, out_channel, out_depth, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.conv3d_transpose_ncdhw_preprocess(*data*, *kernel*, *strides*, *padding*, *out_dtype*, *output_padding*)


预处理数据和内核，使 conv3d_transpose 的计算模式与 conv3d 相同。

## tvm.topi.nn.conv3d_winograd_weight_transform(*kernel*, *tile_size*)


3D winograd 的权重变换。
* **参数：**
   * **核**（[Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 布局为「NCDHW」的原始核张量。
   * **tile_size** ( [int](https://docs.python.org/3/library/functions.html#int) )：winograd 变换的 Tile 大小。例如，F(2x2, 3x3) 为 2，F(4x4, 3x3) 为 4。
* **返回：output** ：5–D，形状为 [alpha, alpha, alpha, CO, CI]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.correlation_nchw(*data1*, *data2*, *kernel_size*, *max_displacement*, *stride1*, *stride2*, *padding*, *is_multiply*)

NCHW 布局中的相关运算符。
* **参数：**
   * **data1** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：4–D，形状为 [batch, channel, height, width]。
   * **data2** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )*：* 4–D，形状为 [batch, channel, height, width]。
   * **kernel_size** ( [int](https://docs.python.org/3/library/functions.html#int) )：相关的核大小，必须是奇数。
   * **max_displacement**（[int](https://docs.python.org/3/library/functions.html#int)）*：* 相关性的最大位移。
   * **stride1**（[int](https://docs.python.org/3/library/functions.html#int)）：数据1的步幅。
   * **stride2**（[int](https://docs.python.org/3/library/functions.html#int)）：以 data1 为中心的邻域内 data2 的步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*2 个***或**4 个 int 的列表/元组*）：填充大小，或 2 个 int 的 [pad_height, pad_width]，或 4 个 int 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **is_multiply** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：运算类型是乘法或减法。
* **返回：Output**  ：4–D，形状为 [batch, out_channel, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.declaration_conv2d_transpose_impl(*data*, *kernel*, *strides*, *padding*, *out_dtype*, *output_padding*)


conv2d 转置的实现。

## tvm.topi.nn.declaration_conv3d_transpose_impl(*data*, *kernel*, *strides*, *padding*, *out_dtype*, *output_padding*)

conv3d 转置的实现。

## tvm.topi.nn.deformable_conv2d_nchw(*data*, *offset*, *kernel*, *strides*, *padding*, *dilation*, *deformable_groups*, *groups*, *out_dtype*)


NCHW 布局中的可变形 conv2D 运算符。


可变形卷积运算描述于[https://arxiv.org/abs/1703.06211](https://arxiv.org/abs/1703.06211)。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 4–D，形状为[batch, in_channel, in_height, in_width]。
   * **offset**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）***：*** 4–D，形状为 [batch, formable_groups filter_height  filter_width * 2, out_height, out_width]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 4–D，形状为[num_filter，in_channel，filter_height，filter_width]。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或**两个 int***的列表/元组*）：填充大小，或 [pad_height, pad_width]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * **deformable_groups**（[int](https://docs.python.org/3/library/functions.html#int)）：可变形组的数量。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）：组数。
* **返回：output**  *：* 4–D，形状为[batch, out_channel, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.deformable_conv2d_nhwc(*data*, *offset*, *kernel*, *strides*, *padding*, *dilation*, *deformable_groups*, *groups*, *out_dtype*)


NHWC 布局中的可变形 conv2D 运算符。


可变形卷积运算描述于[https://arxiv.org/abs/1703.06211](https://arxiv.org/abs/1703.06211)。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：*4–D，形状为[batch, in_height, in_width, in_channel]。
   * **偏移量**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)） –4–D，形状为 [batch, out_height, out_width, 可变形组 * 过滤器高度 * 过滤器宽度 * 2]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](https://docs.python.org/3/library/functions.html#int)4–D，形状为[filter_height，filter_width，in_channel，num_filter]。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：填充大小，或 [pad_height, pad_width]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * **deformable_groups**（[int](https://docs.python.org/3/library/functions.html#int)）：可变形组的数量。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）：组数。
* **返回：output**  ：4–D，形状为[batch, out_height, out_width, out_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.dense(*data*, *weight*, *bias=None*, *out_dtype=None*, *auto_scheduler_rewritten_layout=''*, *meta_schedule_original_shape=None*)

topi 中致密的默认实现。这是 matmul_nt 运算符的别名，用于非转置格式的数据张量和转置格式的权重张量。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：**二维，形状为[batch, in_dim]。
   * **权重**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：*形状为[out_dim，in_dim]的二维。
   * *偏差*（*可选*[ [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*]*）：形状为 [out_dim] 的一维。
   * *out_dtype*（*可选*[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]*）：输出类型。用于混合精度。
   * **auto_scheduler_rewritten_layout** ( *str = ""* )：自动调度程序布局重写传递后的布局。
   * **meta_schedule_original_shape** ( *Optional[List[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]= None* )：输入张量的原始形状。
* **返回：output** *：* 二维，形状为 [batch, out_dim]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.dense_pack(*data*, *weight*, *bias=None*, *out_dtype=None*)


topi 中 dense_pack 的默认实现。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：二维，形状为[batch, in_dim]。
   * **权重**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：形状为[out_dim，in_dim]的二维。
   * *偏差*（*可选*[ [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*]*）：形状为 [out_dim] 的一维。
   * *out_dtype*（*可选*[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]*）：输出类型。用于混合精度。
* **返回：output**  ：二维，形状为 [batch, out_dim]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.depth_to_space(*data*, *block_size*, *layout='NCHW'*, *mode='DCR'*)


对数据进行深度到空间的变换。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：NCHW 或 NHWC 布局中的 4–D 张量。
   * **block_size** ( [int](https://docs.python.org/3/library/functions.html#int) )[：](https://docs.python.org/3/library/stdtypes.html#str)从通道维度组成的块的大小。
   * **布局**（*字符串*）：NCHW 或 NHWC，表示数据布局。
   * **模式**（*字符串*）：DCR 或 CDR，指示应如何访问通道。在 DCR 中，通道以 TensorFlow 风格交织，而在 CDR 中，通道以 Pytorch 风格顺序访问。
* **返回：output** ：形状输出[N，C / block_size**2，H * block_size，W * block_size]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.depthwise_conv2d_NCHWc(*Input*, *Filter*, *stride*, *padding*, *dilation*, *layout*, *out_layout*, *out_dtype=None*)


深度卷积 NCHW[x]c 前向操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：5–D，形状为[batch，in_channel_chunk，in_height，in_width，in_channel_block]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：6维，形状为[out_channel_chunk，1，filter_height，filter_width，1，out_channel_block]。在 NCHWc 深度卷积中，我们将内核的 in_channel 和 channel_multiplier 组合在一起，然后进行平铺。
   * **stride**（*两个整数的元*[组](https://docs.python.org/3/library/stdtypes.html#tuple)）：沿高度和宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * **布局**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：输入数据布局。
   * **out_layout** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据布局。
   * **out_dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：输出数据类型。
* **返回：Output**  **：** 5–D，形状为[batch, out_channel_chunk, out_height, out_width, out_channel_block]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.depthwise_conv2d_backward_input_nhwc(*Filter*, *Out_grad*, *oshape*, *ishape*, *stride*, *padding*)


深度卷积 nhwc 后向 wrt 输入运算符。
* **参数：**
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[filter_height，filter_width，in_channel，channel_multiplier]。
   * **Out_grad** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：4–D，形状为 [batch, out_height, out_width, out_channel]。
   * **stride**（*两个整数的元*[组](https://docs.python.org/3/library/stdtypes.html#tuple)）*：* 沿高度和宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
* **返回：Output**  ：4–D，形状为 [batch, in_height, in_width, in_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.depthwise_conv2d_backward_weight_nhwc(*Input*, *Out_grad*, *oshape*, *fshape*, *stride*, *padding*)


深度卷积 nhwc 后向 wrt 权重运算符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_height, in_width, in_channel]。
   * **Out_grad** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )***：*** 4–D，形状为 [batch, out_height, out_width, out_channel]。
   * **stride**（*两个整数的元*[组](https://docs.python.org/3/library/stdtypes.html#tuple)）：沿高度和宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
* **返回：Output**  [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)4–D，形状为 [filter_height, filter_width, in_channel, channel_multiplier]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.depthwise_conv2d_nchw(*Input*, *Filter*, *stride*, *padding*, *dilation*, *out_dtype=None*)


深度卷积 nchw 前向操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_channel, in_height, in_width]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[in_channel, channel_multiplier, filter_height, filter_width]。
   * **stride** ([int](https://docs.python.org/3/library/functions.html#int)*ora list/tupleoftwo ints*) *：*  **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：空间步幅，或 (stride_height, stride_width)。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）**：** 填充大小，或 ['VALID', 'SAME']。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * **out_dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）*：* 输出数据类型。
* **返回：Output**  *：* 4–D，形状为 [batch, out_channel, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.depthwise_conv2d_nhwc(*Input*, *Filter*, *stride*, *padding*, *dilation*, *kernel_layout='HWOI'*, *out_dtype=None*)

深度卷积 nhwc 前向操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_height, in_width, in_channel]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 4–D，形状为[filter_height，filter_width，in_channel，channel_multiplier]。
   * **stride**（*两个整数的元*[组](https://docs.python.org/3/library/stdtypes.html#tuple)）：沿高度和宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）*：* 扩张大小，或 [dilation_height, dilation_width]。
   * **out_dtype**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：输出数据类型。
* **返回：Output**  *：* 4–D，形状为 [batch, out_height, out_width, out_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.dilate(*data*, *strides*, *dilation_value=0.0*, *name='DilatedInput'*)


使用给定的扩张值（默认为 0）扩张数据。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：nD，可以是任何布局。
   * **strides**（*n 个整数**的**列表/元组*）：每个维度上的扩张步幅，1 表示无扩张。
   * dilation_value（int/float，*可选*）：用于扩大输入的值。
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：生成的名称前缀运算符。
* **返回：Output** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)nD，与数据相同的布局。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.equal_const_int(*expr*, *value*)


如果 expr 等于 value，则返回。
* **参数：expr** (*tvm.Expr*)  [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)输入表达式。
* **返回：equal**  **：** 是否相等。
* **返回类型：**[bool](https://docs.python.org/3/library/functions.html#bool)。

## tvm.topi.nn.fast_softmax(*x*, *axis=-1*)


对数据执行 softmax 激活。使用近似值计算指数可以提高速度。
* **参数：**
   * **x** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：可以是任意维度。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)）：通道轴。
* **返回：output**  ：输出形状与输入相同。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.fifo_buffer(*data*, *buffer*, *axis*)


FIFO 缓冲区可在具有滑动窗口输入的 CNN 中实现计算重用。


计算等价于

```plain
concat(buffer, data, axis=axis)
.slice_axis(axis=axis,
            begin=data.shape[axis],
            end=data.shape[axis]+buffer.shape[axis])
```



适用于：
* 在滑动窗口输入上操作的卷积操作中，对计算的显式重用进行编码。
* 实现 FIFO 队列来缓存中间结果，例如在 Fast WaveNet 中。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入数据。
   * **buffer**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：FIFO 缓冲区的先前值。
   * **axis** ( [int](https://docs.python.org/3/library/functions.html#int) )：指定应该使用哪个轴进行缓冲。
* **返回：result**  *：* 缓冲区的更新值。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.flatten(*data*)


通过折叠较高维度将输入数组展平为二维数组。
* **参数：data** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入数组。
* **返回：output** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)具有折叠高维的二维数组。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.get_const_int(*expr*)

验证 expr 是否为整数并获取常数值。
* **参数：expr** (*tvm.Expror*[int](https://docs.python.org/3/library/functions.html#int)) ：输入表达式。
* **返回：out_value** ：输出。
* **返回类型：**[int](https://docs.python.org/3/library/functions.html#int)。

## tvm.topi.nn.get_const_tuple(*in_tuple*)


验证输入元组是 IntImm 还是 Var，返回 int 或 Var 的元组。
* **参数：in_tuple** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*ofExpr*) **：** 输入。
* **返回：out_tuple** ：输出。
* **返回类型：**[tuple](https://docs.python.org/3/library/stdtypes.html#tuple) of [int](https://docs.python.org/3/library/functions.html#int)。

## tvm.topi.nn.get_pad_tuple(*padding*, *kernel*)


获取 pad 选项的通用代码。
* **参数：**
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **kernel**（[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：卷积核大小。
* **返回：**
   *  **pad_top**（*int*）：顶部填充大小。
   * **pad_left**（*int*）：左侧填充大小。
   * **pad_down**（*int*）：向下填充大小。
   * **pad_right**（*int*）：右侧填充大小。

## tvm.topi.nn.get_pad_tuple1d(*padding*, *kernel*)


获取 pad 选项的通用代码。
* **参数：**
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）*：* 填充大小，或 ['VALID', 'SAME']。
   * **kernel**（[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：卷积核大小。
* **返回：**
   * **pad_left**（*int*）：左侧填充大小。
   * **pad_right**（*int*）：右侧填充大小。

## tvm.topi.nn.get_pad_tuple3d(*padding*, *kernel*)


获取 pad 选项的通用代码。
* **参数：**
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **kernel**（[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：卷积核大小。
* **返回：**
   *    **pad_front**（*int*）：正面的填充尺寸。
   * **pad_top**（*int*）：顶部填充大小。
   * **pad_left**（*int*）：左侧填充大小。
   * **pad_back**（*int*）：背面的填充尺寸。
   * **pad_down**（*int*）：向下填充大小。
   * **pad_right**（*int*）：右侧填充大小。

## tvm.topi.nn.get_pad_tuple_generic(*padding*, *kernel*)


获取 pad 选项的通用代码。
* **参数：**
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **kernel**（[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：卷积核大小。
* **返回：** 
   * **pad_top**（*int*）：顶部填充大小。
   * **pad_down**（*int*）：向下填充大小。
   * **pad_left**（*int*）：左侧填充大小。
   * **pad_right**（*int*）：右侧填充大小。

## tvm.topi.nn.get_padded_shape(*data*, *pad_before*, *pad_after=None*)


应用填充后计算张量的输出形状。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：应用填充的输入张量。
   * **pad_before** *：* n 个整数的列表/元组，在每个维度上填充宽度以在轴开始之前进行填充。
   * **pad_after**：n 个整数的列表/元组，可选填充宽度，每个维度在轴端后进行填充。
* **抛出：**[ValueError](https://docs.python.org/3/library/exceptions.html#ValueError)：如果 pad_before 或 pad_after 长度与数据维度不匹配。
* **返回：** 表示张量填充形状的元组。
* **返回类型：**[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)。

## tvm.topi.nn.global_pool(*data*, *pool_type*, *layout='NCHW'*)


对数据的高度和宽度维度进行全局池化。


它根据布局字符串决定高度和宽度尺寸，其中「W」和「H」分别表示宽度和高度。宽度和高度尺寸不能拆分。例如，NCHW、NCHW16c 等适用于池，而 NCHW16w、NCHW16h 则不适用。有关布局字符串约定的更多信息，请参阅参数布局。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：具有布局形状的 nD。
   * **pool_type** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：池类型，“max”或“avg”。
   * **layout** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )*：* 输入数据的布局。布局应该由大写字母、小写字母和数字组成，其中大写字母表示维度，对应的小写字母（因子大小）表示分割维度。例如，NCHW16c 可以描述一个 5 维张量，其值为 [batch_size, channel, height, width, channel_block]，其中 channel_block=16 表示对 channel 维度的分割。
* **返回：output**  ：nD 采用相同的布局，高度和宽度尺寸为 1。例如，对于 NCHW，输出形状将为 [batch, channel, 1, 1]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.group_conv1d_ncw(*data*, *kernel*, *strides=1*, *padding='VALID'*, *dilation=1*, *groups=1*, *out_dtype=None*)


用于 NCW 布局的一维卷积前向操作符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 三维，形状为[batch, in_channel, in_width]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：形状为[num_filter，in_channel，filter_size]的 3–D。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或*[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)）：沿宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*、*[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小可以是相等填充的整数、（左、右）的元组或 ['VALID', 'SAME'] 中的字符串。
   * **dilation**（[整数](https://docs.python.org/3/library/functions.html#int)*或*[元组](https://docs.python.org/3/library/stdtypes.html#tuple)）：如果卷积需要扩张，则扩张率。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）：组数。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据类型。如果为 None ，则输出与输入类型相同。

## tvm.topi.nn.group_conv1d_nwc(*data*, *kernel*, *strides=1*, *padding='VALID'*, *dilation=1*, *groups=1*, *out_dtype=None*)


用于 NWC 布局的一维卷积前向操作符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：三维，形状为[batch, in_width, in_channel]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：3–D，形状为[filter_size，in_channel，num_filter]。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或*[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)）*：* 沿宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*、*[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小可以是相等填充的整数、（左、右）的元组或 ['VALID', 'SAME'] 中的字符串。
   * **dilation**（[整数](https://docs.python.org/3/library/functions.html#int)*或*[元组](https://docs.python.org/3/library/stdtypes.html#tuple)）：如果卷积需要扩张，则扩张率。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）：组数。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据类型。如果为 None ，则输出与输入类型相同。

## tvm.topi.nn.group_conv1d_transpose_ncw(*data*, *kernel*, *stride*, *padding*, *out_dtype*, *output_padding*, *groups*)


转置的一维组卷积 ncw 前向操作符。
   * **参数：数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)三维，形状为[batch, in_channel, in_width]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 3–D，形状为[in_channel，num_filter，filter_width]。
   * **步幅**（*ints*）：沿宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据类型。用于混合精度。
   * **输出填充**（*整数*）：用于恢复实际输出形状，以防有更多不止一种可能的形状。必须小于步幅。

                    组：int

                        组数
* **返回：output** ：3–D，形状为[batch, out_channel, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.group_conv2d_nchw(*Input*, *Filter*, *stride*, *padding*, *dilation*, *groups*, *out_dtype=None*)


NCHW 布局中的组卷积操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_channel, in_height, in_width]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[num_filter，in_channel // groups，filter_height，filter_width]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*2 个*或4 个 int*的列表/元组）：填充大小，或 2 个 int 的 [pad_height, pad_width]，或 4 个 int 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组）**：** 扩张大小，或 [dilation_height, dilation_width]。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）：组数。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出类型。用于混合精度。
* **返回：Output**  ：4–D，形状为 [batch, out_channel, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.group_conv2d_nhwc(*Input*, *Filter*, *stride*, *padding*, *dilation*, *groups*, *out_dtype=None*)


NHWC 布局中的组卷积操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_height, in_width, in_channel, …]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[filter_height，filter_width，in_channel // groups，num_filter]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或* 2 个*或* 4 个 int*的列表/元组*）：填充大小，或 2 个 int 的 [pad_height, pad_width]，或 4 个 int 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **dilation**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）：组数。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出类型。用于混合精度。
* **返回：Output**  *：* 4–D，形状为 [batch, out_height, out_width, out_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.group_conv2d_transpose_nchw(*data*, *kernel*, *stride*, *padding*, *out_dtype*, *output_padding*, *groups*)


NCHW 布局中的组卷积操作符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 4–D，形状为[batch, in_channel, in_height, in_width]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 4–D，形状为[in_channel，out_channel // groups，filter_height，filter_width]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*2 个***或**4 个 int*的列表/元组）：填充大小，或 2 个 int 的 [pad_height, pad_width]，或 4 个 int 的 [pad_top, pad_left, pad_bottom, pad_right]。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据类型。用于混合精度。
   * **output_padding**（*整数元*[组](https://docs.python.org/3/library/stdtypes.html#tuple)）：用于获取渐变的正确输出形状。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）：组数。
   * **out_dtype**[：](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)输出类型。用于混合精度。
* **返回：Output**  ：4–D，形状为 [batch, out_channel, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.group_conv3d_transpose_ncdhw(*data*, *kernel*, *strides*, *padding*, *out_dtype*, *output_padding*, *groups*)


转置组3D 卷积 ncdhw 前向操作符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：5–D，形状为[batch，in_channel，in_depth，in_height，in_width]。
   * **内核**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 5–D，形状为[in_channel，num_filter，filter_depth，filter_height，filter_width]。
   * **strides**（[int](https://docs.python.org/3/library/functions.html#int)*或*三个 int*的列表/元组*）：沿深度、高度和宽度的空间步幅。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）**：** 填充大小，或 ['VALID', 'SAME']。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出数据类型。用于混合精度。
   * **output_padding**（*整数元*[组](https://docs.python.org/3/library/stdtypes.html#tuple)）：用于获取渐变的正确输出形状。
   * **groups**（[int](https://docs.python.org/3/library/functions.html#int)）：组数。
* **返回：Output**  ：5–D，形状为[batch, out_channel, out_depth, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.if_then_else(*cond*, *t*, *f*, *span=None*)


条件选择表达式。
* **参数：**
   * **cond** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)条件。
   * **t** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：如果 cond 为真，则结果表达式。
   * **f** ( [PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr) )：如果 cond 为假，则结果表达式。
   * *span*（可选[ [Span](/docs/api-reference/python-api/tvm-ir#class-tvmirspansource_name-line-end_line-column-end_column)*]*）：此运算符在源中的位置。
* **返回：result** ：条件表达式的结果。
* **返回类型：**[Node](/docs/api-reference/python-api/tvm-ir#class-tvmirnode)。

:::Note

与 Select 不同，if_then_else 不会执行不满足条件的分支。您可以使用它来防止越界访问。与 Select 不同，如果向量中某些通道的条件不同，则 if_then_else 无法进行向量化。

:::

## tvm.topi.nn.leaky_relu(*x*, *alpha*)

取输入 x 的 leaky relu。
* **参数：**
   *   **x**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）–输入参数。
   *   **alpha**（[float](https://docs.python.org/3/library/functions.html#float)）*：* x < 0 时小梯度的斜率。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.log_softmax(*x*, *axis=-1*)


对数据执行对数 softmax 激活。
* **参数：**
   * **x**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：ND 输入数据。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)）[：](https://docs.python.org/3/library/functions.html#int)通道轴。
* **返回：output**  ：具有相同形状的 ND 输出。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.lrn(*data*, *size*, *axis=1*, *alpha=0.0001*, *beta=0.75*, *bias=2*)


对输入数据执行跨通道局部响应标准化。


sum_sqr_up^i{x, y} = (bias+((alpha/size)* {sum_{j=max(0, i-size/2)}^{min(N-1,i+size/2)} (data^j{x,y})^2}))^beta output^i{x, y} = data^i{x, y}/sum_sqr_up^i{x, y} N 是输入通道数。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 4–D，形状为[批次，通道，高度，宽度]。
   * **size**（[int](https://docs.python.org/3/library/functions.html#int)）：标准化窗口大小。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)）**：** 输入数据布局通道轴默认值为 1（对于 NCHW 格式）。
   * **偏差**（[浮点型](https://docs.python.org/3/library/functions.html#float)）：避免除以 0 的偏移量。
   * **alpha**（[浮点数](https://docs.python.org/3/library/functions.html#float)）：待除。
   * **beta**（[浮点数](https://docs.python.org/3/library/functions.html#float)）：指数。
* **返回：output**  ：具有相同形状的 4–D 输出。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.lstm(*Xs*, *Wi*, *Wh*, *Bi=None*, *Bh=None*, *h_init=None*, *c_init=None*, *proj=None*, *p_i=None*, *p_f=None*, *p_o=None*, *f_act=*, *g_act=*, *h_act=*, *reverse=False*, *weight_layout: str = 'IFGO'*)

使用 TE 扫描实现的通用 LSTM。
* **参数：**
   * **Xs**（[te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：形状为（seq_len、batch_size、in_dim）的输入序列。
   * **Wi** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：输入权重矩阵，形状为(4 * hidden_​​dim, in_dim)。权重根据 weight_layout 进行打包。
   * **Wh** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：隐藏权重矩阵，形状为(4 * hidden_​​dim, hidden_​​dim 或 proj_dim)。打包为 Wh。
   * **Bi** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*)：输入偏差，形状为(4 * hidden_​​dim,)，默认为 None 。打包为 Wh。
   * **Bh**（[te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*）：形状为 Bi 的隐藏偏差，默认为 None 。打包为 Wh。
   * **h_init** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*)：初始隐藏状态，形状为(batch_size, hidden_​​dim 或 proj_dim)，若为 None 则为零。
   * **c_init** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*)*：* 初始单元状态，形状与 h_init 相同，若为 None 则为零。
   * **proj** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*)[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)形状为(proj_dim, hidden_​​dim)的投影矩阵，默认为 None。
   * **p_i** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*)：形状为(batch_size, hidden_​​dim)的 Peephole LSTM 矩阵，默认为 None。
   * **p_f** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*)*：* 形状为(batch_size, hidden_​​dim)的 Peephole LSTM 矩阵，默认为 None。
   * **p_o** ( [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*)：形状为(batch_size, hidden_​​dim)的 Peephole LSTM 矩阵，默认为 None。
   * *f_act**（*F*，*可选*）：门激活函数。
   * **g_act** (*F,optional*) *：* *g_act**（*F*，*可选*）：门激活函数。
   * **h_act** (*F,optional*) ： *h_act**（*F*，*可选*）*：* 门激活函数。
   * **reverse**（[bool](https://docs.python.org/3/library/functions.html#bool)*，可选*）：是否反向处理 X，默认为 False。
   * **weight_layout**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：门控的打包权重布局，默认为“IFGO”。注意：I = 输入，F = 遗忘，G = 单元，O = 输出。
* **返回：result**  *：* 隐藏状态的元组（形状为(seq_len, batch_size, hidden_​​dim 或 proj_dim)）和单元状态（形状为(seq_len, batch_size, hidden_​​dim)）。
* **返回类型：**[te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor), [te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.matmul(*tensor_a*, *tensor_b*, *bias=None*, *out_dtype=None*, *transpose_a=False*, *transpose_b=False*, *auto_scheduler_rewritten_layout=''*, *meta_schedule_original_shape=None*)


topi 中 matmul 的默认实现。
* **参数：**
   * **tensor_a** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：二维，形状为 [batch, in_dim]。
   * **tensor_b** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：二维，形状为 [out_dim, in_dim]。
   * *偏差**（*可选**[ [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*]*）*：* 形状为 [out_dim] 的一维
   * *out_dtype**（*可选**[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]*）*：* 输出类型。用于混合精度。
   * *transpose_a**（*可选**[ [bool](https://docs.python.org/3/library/functions.html#bool)*]= False*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)tensor_a 是否为转置格式。
   * *transpose_b**（*可选**[ [bool](https://docs.python.org/3/library/functions.html#bool)*]= False*）：tensor_b 是否为转置格式。
   * *auto_scheduler_rewritten_layout**（*可选**[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]= ""*）：自动调度程序布局重写传递后的布局。
   * **meta_schedule_original_shape** ( *Optional[List[*[PrimExpr](/docs/api-reference/python-api/tvm-ir#class-tvmirprimexpr)*]]= None* )：输入张量的原始形状。
* **返回：output**  ：二维，形状为 [batch, out_dim]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.mirror_pad(*data*, *pad_before*, *pad_after=None*, *mode='SYMMETRIC'*, *name='MirrorPadInput'*)


具有对称或反射功能的镜像平板输入。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：nD 输入，可以是任何布局。
   * **pad_before**（*n 个整数的列表/元组*）：在每个维度上填充宽度以在轴开始之前进行填充。
   * *pad_after**（*n 个整数**的**列表/元组*，可选）：填充每个维度的宽度以填充轴端之后。
   * **mode**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：镜像填充的类型。必须为 SYMMETRIC 或 REFLECT。
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）*：* 生成的名称前缀运算符。
* **返回：Output** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)nD，与输入相同的布局。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.namedtuple(*typename*, *field_names*, ***, *rename=False*, *defaults=None*, *module=None*)

返回具有命名字段的元组的新子类。

```python
>>> Point = namedtuple('Point', ['x', 'y'])
>>> Point.__doc__                   # 新类的文档字符串
'Point(x, y)'
>>> p = Point(11, y=22)             # 使用位置参数或关键字参数实例化
>>> p[0] + p[1]                     # 可以像普通元组一样通过索引访问
>>> x, y = p                        # 可以像普通元组一样解包
>>> x, y
(11, 22)
>>> p.x + p.y                       # 字段也可以通过名字访问e
33
>>> d = p._asdict()                 # 转换为字典
>>> d['x']
11
>>> Point(**d)                      # 从字典创建
Point(x=11, y=22)
>>> p._replace(x=100)               # _replace() 类似于 str.replace()，但作用于命名字段
Point(x=100, y=22)
```
## tvm.topi.nn.nll_loss(*predictions*, *targets*, *weights*, *reduction*, *ignore_index*)


输入数据的负对数似然损失。


输出{n，i_1，i_2，…，i_k} = -p * w。


其中 t = 目标{n, i_1, i_2, …, i_k}。


p = 预测{n，t，i_1，i_2，i_k} w = 权重{n，i_1，i_2，…，i_k} 如果 t != ignore_index 否则为 0。


结果 = 减少（输出）。
* **参数：**
   * **预测**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：（k + 2）–D，形状为（N，C，d_1，d_2，…，d_k），其中 C 是目标类别的数量。
   * **目标**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 形状为（n，d_1，d_2，…，d_k）的（k + 1）–D 输入的目标值。
   * **权重**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：形状为（C，）的 1–D 每个目标值的权重。
   * **reduction**（*字符串*）**：** 应用于输出的缩减方法。可以是“mean”、“sum”或“none”。
   * **ignore_index** ( [int](https://docs.python.org/3/library/functions.html#int) )：要忽略的目标值。
* **返回：output** ：如果约简类型为“平均值”或“总和”，则为标量，否则与目标形状相同。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.pad(*data*, *pad_before*, *pad_after=None*, *pad_value=0.0*, *name='PadInput'*, *attrs=None*)


使用 pad 值的 Pad 输入。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：nD 输入，可以是任何布局。
   * **pad_before**（*n 个整数的*列表/元组）：在每个维度上填充宽度以在轴开始之前进行填充。
   * *pad_after*（n 个整数**的列表/元组**，*可选*）：填充每个维度的宽度以填充轴端之后。
   * **pad_value**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：要填充的值。
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：生成的名称前缀运算符。
* **返回：Output** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)nD，与输入相同的布局。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.pool1d(*data*, *kernel*, *stride*, *dilation*, *padding*, *pool_type*, *ceil_mode=False*, *layout='NCW'*, *count_include_pad=True*)


对数据的宽度维度进行池化。


宽度轴根据布局字符串确定。其中“w”表示宽度。宽度维度不可拆分。例如，NCW、NCW16c 等适用于池，而 NCW16w 则不适用。有关布局字符串约定的更多信息，请参阅参数布局。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：具有布局形状的 nD。
   * **kernel**（*一个 int**或***[int](https://docs.python.org/3/library/functions.html#int)***的****列表/元组*）：内核大小，[kernel_width]。
   * **stride**（*一个 int**或***[int](https://docs.python.org/3/library/functions.html#int)***的****列表/元组*）：步幅大小，[stride_width]。
   * **dilation**（*两个整数**的***列表/元组*）：扩张大小，[dilation_height，dilation_width]。
   * **padding**（*两个整数**的***列表/元组*）：填充大小，[pad_left, pad_right]。
   * **pool_type** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：池类型，“max”或“avg”。
   * **ceil_mode** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：计算输出大小时是否使用 ceil。
   * **layout**（*字符串*）：输入数据的布局。布局应该由大写字母、小写字母和数字组成，其中大写字母表示维度，对应的小写字母（因子大小）表示分割维度。例如，NCW16c 可以描述一个 [batch_size, channel, width, channel_block] 的四维张量，其中 channel_block=16 表示对 channel 维度的分割。
   * **count_include_pad** ( [bool](https://docs.python.org/3/library/functions.html#bool) )[：](https://docs.python.org/3/library/functions.html#bool)当 pool_type 为 'avg' 时，是否在计算中包含填充。
* **返回：output** ：nD 在同一布局中。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.pool2d(*data*, *kernel*, *stride*, *dilation*, *padding*, *pool_type*, *ceil_mode=False*, *layout='NCHW'*, *count_include_pad=True*)


对数据的高度和宽度维度进行池化。


它根据布局字符串决定高度和宽度尺寸，其中“W”和“H”分别表示宽度和高度。宽度和高度尺寸不能拆分。例如，NCHW、NCHW16c 等适用于池，而 NCHW16w、NCHW16h 则不适用。有关布局字符串约定的更多信息，请参阅参数布局。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：具有布局形状的 nD。
   * **kernel**（*两个整数**的***列表/元组）：内核大小，[kernel_height，kernel_width]。
   * **stride**（*两个整数**的***列表/元组）：步幅大小，[stride_height，stride_width]。
   * **dilation**（*两个整数**的***列表/元组）：扩张大小，[dilation_height，dilation_width]。
   * **padding**（*四个整数**的***列表/元组）：填充大小，[pad_top, pad_left, pad_bottom, pad_right]。
   * **pool_type** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：池类型，“max”或“avg”。
   * **ceil_mode** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：计算输出大小时是否使用 ceil。
   * **layout**（*字符串*）：输入数据的布局。布局应该由大写字母、小写字母和数字组成，其中大写字母表示维度，对应的小写字母（因子大小）表示分割维度。例如，NCHW16c 可以描述一个 5 维张量，其值为 [batch_size, channel, height, width, channel_block]，其中 channel_block=16 表示对 channel 维度的分割。
   * **count_include_pad** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：当 pool_type 为 'avg' 时，是否在计算中包含填充。
* **返回：output** ：nD 在同一布局中。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.pool3d(*data*, *kernel*, *stride*, *dilation*, *padding*, *pool_type*, *ceil_mode=False*, *layout='NCDHW'*, *count_include_pad=True*)


对数据的深度、高度和宽度维度进行池化。


它根据布局字符串决定深度、高度和宽度尺寸，其中“D”、“W”和“H”分别表示深度、宽度和高度。深度、宽度和高度尺寸不能拆分。例如，NCDHW、NCDHW16c 等适用于池，而 NCDHW16d、NCDHW16w 和 NCDHW16h 则不适用。有关布局字符串约定的更多信息，请参阅参数布局。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：具有布局形状的 nD。
   * **kernel**（*三个整数**的***列表/元组）：内核大小，[kernel_depth、kernel_height、kernel_width]。
   * **stride**（*三个整数**的***列表/元组）：步幅大小，[stride_depth，stride_height，stride_width]。
   * **dilation**（*两个整数**的***列表/元组）：扩张大小，[dilation_height，dilation_width]。
   * **padding**（*六个整数**的***列表/元组）：填充尺寸，[pad_front、pad_top、pad_left、pad_back、pad_bottom、pad_right]。
   * **pool_type** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )**：** 池类型，“max”或“avg”。
   * **ceil_mode** ( [bool](https://docs.python.org/3/library/functions.html#bool) )**：** 计算输出大小时是否使用 ceil。
   * **layout**（*字符串*）：输入数据的布局。布局应该由大写字母、小写字母和数字组成，其中大写字母表示维度，对应的小写字母（因子大小）表示分割维度。例如，NCDHW16c 可以描述一个 6 维张量，其值为 [batch_size, channel,depth, height, width, channel_block]，其中 channel_block=16 表示对 channel 维度的分割。
   * **count_include_pad** ( [bool](https://docs.python.org/3/library/functions.html#bool) )**：** 当 pool_type 为 'avg' 时，是否在计算中包含填充。
* **返回：output** ：nD 在同一布局中。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.pool_grad(*grads*, *data*, *kernel*, *stride*, *padding*, *pool_type*, *ceil_mode=False*, *count_include_pad=True*, *layout='NCHW'*)


池化在数据高度和宽度维度上的梯度。


它根据布局字符串决定高度和宽度尺寸，其中“W”和“H”分别表示宽度和高度。宽度和高度尺寸不能拆分。例如，NCHW、NCHW16c 等适用于池，而 NCHW16w、NCHW16h 则不适用。有关布局字符串约定的更多信息，请参阅参数布局。
* **参数：**
   * **grads**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：具有布局形状的 nD。
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：具有布局形状的 nD。
   * **kernel**（*两个整数**的***列表/元组）：内核大小，[kernel_height，kernel_width]。
   * **stride**（*两个整数**的***列表/元组）：步幅大小，[stride_height，stride_width]。
   * **padding**（*四个整数**的*** *列表/元组*）：填充大小，[pad_top, pad_left, pad_bottom, pad_right]。
   * **pool_type** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：池类型，“max”或“avg”。
   * **ceil_mode** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：计算输出大小时是否使用 ceil。
   * **count_include_pad** ( [bool](https://docs.python.org/3/library/functions.html#bool) )**：** 当 pool_type 为 'avg' 时，是否在计算中包含填充。
   * **layout**（*字符串*）**：** 输入数据的布局。布局应该由大写字母、小写字母和数字组成，其中大写字母表示维度，对应的小写字母（因子大小）表示分割维度。例如，NCHW16c 可以描述一个 5 维张量，其值为 [batch_size, channel, height, width, channel_block]，其中 channel_block=16 表示对 channel 维度的分割。
* **返回：output** ：nD 在同一布局中。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.prelu(*x*, *slope*, *axis=1*)


PReLU。它接受两个参数：一个输入`x`和一个权重数组`W` ，并计算输出为 PReLU(x)y=x>0?x:W∗x， 在哪里∗是批次中每个样本的元素乘法。
* **参数：**
   * **x**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入参数。
   * **斜率**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 用于 prelu 的通道化斜率张量。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)）：需要应用通道数据的轴。
* **返回：**
   *  **y** ( *tvm.te.Tensor* )：结果。
   * *Links。*
   * **[http** (*//arxiv.org/pdf/1502.01852v1.pdf]*)。

## tvm.topi.nn.reduce(*function*, *sequence*[, *initial*]) → value

将一个包含两个参数的函数从左到右累加地应用于序列的项，从而将序列简化为单个值。例如，reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) 计算结果为 ((((1+2)+3)+4)+5)。如果指定了 initial，则在计算过程中将其放置在序列的项之前，并在序列为空时用作默认值。

## tvm.topi.nn.reflect_pad(*data*, *pad_before*, *pad_after=None*, *name='ReflectPadInput'*)


将反射填充应用于输入张量。
* **参数：**
   *   **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入张量。
   *   *pad_before* ( *List[*[int](https://docs.python.org/3/library/functions.html#int)*]* )：每个维度前填充的量。
   * **pad_after**（*List[*[int](https://docs.python.org/3/library/functions.html#int)*]，可选*）：每个维度后的填充量。如果为 None ，则默认为 pad_before 。
   * **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) ：  **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：结果张量的名称。
* **返回：out** ：反射填充张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.relu(*x*)

取输入 x 的 relu。
* **参数：x** ([tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)) ：输入参数。
* **返回：y** *：* 结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.replicate_pad(*data*, *pad_before*, *pad_after=None*, *name='ReplicatePadInput'*)


对输入张量应用重复填充（边缘填充）。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入张量。
   * *pad_before* ( *List[*[int](https://docs.python.org/3/library/functions.html#int)*]* )：每个维度前填充的量。
   * **pad_after**（*List[*[int](https://docs.python.org/3/library/functions.html#int)*]，可选*）：每个维度后的填充量。如果为 None ，则默认为 pad_before 。
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)）*：* 结果张量的名称。
* **返回：out** ：复制填充的张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.scale_shift_nchw(*Input*, *Scale*, *Shift*)


推理中的批量标准化运算符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D 输入张量，NCHW 布局 [批次、通道、高度、宽度]。
   * **比例尺**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)比例尺张量，大小为通道数的一维。
   * **Shift** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：移位张量，大小为通道数的一维
* **。返回：Output** ：输出张量，布局为 NCHW。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.scale_shift_nchwc(*Input*, *Scale*, *Shift*) 


推理中的批量标准化运算符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：5D 输入张量，NCHWc 布局[batch，channel_chunk，height，width，channel_block]。
   * **Scale**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：缩放张量，二维，大小为 [channel_chunk, channel_block]。
   * **Shift** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：移位张量，二维，大小为 [channel_chunk, channel_block]。
* **返回：Output** ：输出张量，布局为 NHWC。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.scale_shift_nhwc(*Input*, *Scale*, *Shift*)


推理中的批量标准化运算符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D 输入张量，NHWC 布局[批次，高度，宽度，通道]。
   * **比例尺**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：比例尺张量，大小为通道数的一维。
   * **Shift** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：移位张量，大小为通道数的一维。
* **返回：Output** ：输出张量，布局为 NHWC。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.simplify(*expr*)


如果是 Expr 则化简表达式，如果是 int 则直接返回。
* **参数：expr** (*Expror*[int](https://docs.python.org/3/library/functions.html#int)) ：输入。
* **返回：out** ：简化的输出。
* **返回类型：** Expr or [int](https://docs.python.org/3/library/functions.html#int)。

## tvm.topi.nn.simulated_dequantize(*data*, *in_dtype*, *input_scale=None*, *input_zero_point=None*, *axis=-1*)


模拟 QNN 反量化运算符，可模拟 QNN 输出，而无需更改数据类型。与真正的 QNN 反量化相比，此运算符的优势在于，它允许动态选择数据类型，并且可以对每个通道、标量尺度和零点进行操作，而 QNN 反量化则需要在编译时修复这两者。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：运算符的 ND 输入张量。
   * **in_dtype** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：标量变量，指示使用哪种数据类型来模拟反量化。使用 SQNN_DTYPE_TO_CODE 将 dtype 字符串转换为相应的变量值。
   * **input_scale**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)标量张量，表示从整数数据类型反量化时使用的比例。当它包含多个值时，N 必须与数据的通道数匹配。
   * **input_zero_point**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*）**：** 一维张量，表示从整数数据类型反量化时使用的零点。当它包含多个值时，N 必须与数据的通道数匹配。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：用于量化的通道轴。默认值为 –1，对应于最后一个轴。

## tvm.topi.nn.simulated_quantize(*data*, *out_dtype*, *output_scale=None*, *output_zero_point=None*, *axis=-1*) 


模拟 QNN 量化运算符，可模拟 QNN 输出，无需更改数据类型。与真正的 QNN 量化相比，此运算符的优势在于，它允许动态选择数据类型，并且可以对每个通道、标量尺度和零点进行操作，而 QNN 量化则要求在编译时固定这两个参数。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：运算符的 ND 输入张量。
   * **out_dtype** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：一个标量变量，用于指示要使用哪种数据类型来模拟量化。使用 SQNN_DTYPE_TO_CODE 将 dtype 字符串转换为相应的变量值。
   * **output_scale**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*）*：* 标量张量，表示量化为整数数据类型时使用的比例。当它包含多个值时，N 必须与数据的通道数匹配。
   * **output_zero_point**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)*，可选*）：一个一维张量，表示量化为整数数据类型时使用的零点。当它包含多个值时，N 必须与数据的通道数匹配。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)*，可选*）：用于量化的通道轴。默认值为 –1，对应于最后一个轴。

## tvm.topi.nn.softmax(*x*, *axis=-1*)


对数据执行 softmax 激活。
* **参数：**
   * **x** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：可以是任意维度。
   * **axis**（[int](https://docs.python.org/3/library/functions.html#int)）：通道轴。
* **返回：output** ：输出形状与输入相同。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.softmax_common(*x*, *axis*, *use_fast_exp*)


softmax 和 fast_softmax 的共同部分。

## tvm.topi.nn.softplus(*x*, *beta=1.0*, *threshold=20.0*)


计算具有数值稳定性的输入 x 的 Softplus 激活。
* **参数：**
   * **x**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入张量。
   * **beta**（[浮点数](https://docs.python.org/3/library/functions.html#float)*，可选*）：Softplus 公式中的比例因子 β（默认值为 1.0）。
   * **阈值**（[浮点数](https://docs.python.org/3/library/functions.html#float)*，可选*）[：](https://docs.python.org/3/library/functions.html#float)数值稳定性的阈值（默认值为 20.0）。
* **返回：y** ：结果。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.space_to_batch_nd(*data*, *block_shape*, *pad_before*, *pad_after*, *pad_value=0.0*)


对数据执行批量到空间的转换
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：形状为 [batch, spatial_shape, remaining_shapes] 的 ND Tensor，其中 spatial_shape 有 M 维。
   * **block_shape**（*整数*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：大小为 [M] 的列表，其中 M 是空间维度的数量，指定每个空间维度的块大小。
   * **pad_before**（*整数*[列表](https://docs.python.org/3/library/stdtypes.html#list)*）：形状为 [M] 的列表，其中 M 是空间维度的*数量，指定每个空间维度之前的零填充大小。
   * **pad_after**（*整数*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：形状为 [M] 的列表，*其中 M 是空间维度的数量，指定每个空间维度后的零填充大小。
   * **pad_value**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：用于填充的值。
* **返回：output。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.space_to_depth(*data*, *block_size*, *layout='NCHW'*)


对数据执行空间到深度的转换。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：NCHW 或 NHWC 布局中的 4–D 张量。
   * **block_size** ( [int](https://docs.python.org/3/library/functions.html#int) )：分解为通道维度的块的大小。
   * **布局**（*字符串*）：NCHW 或 NHWC，表示数据布局。
* **返回：output** ：形状输出[N，C * block_size**2，H / block_size，W / block_size]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.strided_slice(*a*, *begin*, *end*, *strides=None*, *axes=None*, *slice_mode='end'*, *assume_inbound=True*)

数组的切片。
* **参数：**
   * **a**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）**：** 要切片的张量。
   * **begin**（*int*[列表](https://docs.python.org/3/library/stdtypes.html#list)）*：* 切片中开始的索引[。](https://docs.python.org/3/library/functions.html#int)
   * **end**（*int*[列表](https://docs.python.org/3/library/stdtypes.html#list)）*：* 指示切片结束的索引[。](https://docs.python.org/3/library/functions.html#int)
   * **strides**（*整数*[列表](https://docs.python.org/3/library/stdtypes.html#list)*，可选）：指定*[步幅](https://docs.python.org/3/library/functions.html#int)值，在这种情况下可以为负数，输入张量将在该特定轴上反转。
   * **轴**( *int*[列表](https://docs.python.org/3/library/stdtypes.html#list)*，可选)：应用切片*[的](https://docs.python.org/3/library/functions.html#int)轴。指定后，起始、结束步幅和轴需要为相同长度的整数列表。
   * **slice_mode**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）*：* 切片模式 [end, size]。end*：*切片的结束索引 [默认]。size：输入的步幅将被忽略，此模式下的输入 end 表示从 begin 指定位置开始的切片大小。如果 end[i] 为 –1，则该维度上的所有剩余元素都将包含在切片中。
   * **假设_inbound** ( [bool](https://docs.python.org/3/library/functions.html#bool)*，可选*)*：* 一个标志，指示是否假定所有索引都是入站的。
* **返回：ret。**
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.unpack_NCHWc_to_nchw(*packed_out*, *out_dtype*) 


将 conv2d_NCHWc 输出从布局 NCHWc 解包为 NCHW。
* **参数：**
   * **packed_out** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：conv2d_NCHWc 的输出张量。
   * **out_dtype** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：输出 dtype。
* **返回：unpacked_out** ：NCHW 布局中解包的输出张量。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.upsampling(*data*, *scale_h*, *scale_w*, *layout='NCHW'*, *method='nearest_neighbor'*, *align_corners=False*, *output_shape=None*)


对数据执行上采样。


支持最近邻和双线性上采样。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入是形状为 [batch, channel, in_height, in_width] 或 [batch, in_height, in_width, channel] 的 4–D 张量。
   * **scale_h** ( [float](https://docs.python.org/3/library/functions.html#float) )：高度的缩放因子。
   * **scale_w** ( [float](https://docs.python.org/3/library/functions.html#float) )*：* 宽度的缩放因子。
   * *布局**（*字符串**，*可选*）：“NCHW”或“NHWC”。
   * **方法**（*{“bilinear”，**“nearest_neighbor”*** ***，****“bicubic”}*）：用于上采样的方法。
   * *output_shape*（*tvm.tir.container.Array，*可选*）：返回的形状。如果为 None，则推断为 None （如果形状是动态确定的，则将 out_dtype.shape 传递为 output_shape）。
* **返回：output** *：* 4–D，形状为 [batch, channel, in_height*scale_h, in_width*scale_w] 或 [batch, in_height*scale, in_width*scale, channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.upsampling3d(*data*, *scale_d*, *scale_h*, *scale_w*, *layout='NCDHW'*, *method='nearest_neighbor'*, *coordinate_transformation_mode='half_pixel'*, *output_shape=None*)


对数据执行上采样。


支持最近邻和双线性上采样。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入是一个 5–D 张量，形状为 [batch, channel, in_depth, in_height, in_width] 或 [batch, in_depth, in_height, in_width, channel]。
   * **scale_d**（[float](https://docs.python.org/3/library/functions.html#float)）：深度的缩放因子。
   * **scale_h** ( [float](https://docs.python.org/3/library/functions.html#float) )：高度的缩放因子。
   * **scale_w** ( [float](https://docs.python.org/3/library/functions.html#float) )：宽度的缩放因子。
   * *布局**（*字符串*，可选*）：“NCDHW”或“NDHWC”。
   * *方法**（*{“trilinear”，*“nearest_neighbor”}*）：用于上采样的方法。
   * *coordinate_transformation_mode**（*字符串**，*可选*）：描述如何将调整大小后的张量中的坐标转换为原始张量中的坐标。有关详细信息，请参阅 ONNX Resize 运算符规范。可用选项包括“half_pixel”、“align_corners”和“asymmetric”。
   * *output_shape**（*tvm.tir.container.Array，****可选*）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)返回的形状。如果为 None，则推断为 None （如果形状是动态确定的，则将 out_dtype.shape 传递为 output_shape）。
* **返回：output** *：*5–D，形状为 [batch, channel, in_depth*scale, in_height*scale, in_width*scale] 或 [batch, in_depth*scale, in_height*scale, in_width*scale, channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.winograd_transform_matrices(*tile_size*, *kernel_size*, *out_dtype*)

将 tile_size 的 A、B 和 G 变换矩阵计算为 tvm.Expr。

## tvm.topi.nn.instance_norm(*data*, *gamma*, *beta*, *channel_axis*, *axis*, *epsilon=1e-05*)

实例规范化运算符。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：形状为（d_0，d_1，…，d_{N–1}）的 ND。
   * **gamma** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)形状为 (r_0, r_1, …, r_{K–1}) 的 KD，其中 K == len(axis) 且 d_{axis_k} == r_k。
   * **beta**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：可选，KD 形状为 (r_0, r_1, …, r_{K–1})，其中 K == len(axis) 且 d_{axis_k} == r_k。
   * **axis**（*int*[列表）](https://docs.python.org/3/library/stdtypes.html#list)[：](https://docs.python.org/3/library/stdtypes.html#list)[应用标准化](https://docs.python.org/3/library/stdtypes.html#list)[的](https://docs.python.org/3/library/functions.html#int)轴（计算平均值和方差的轴）。
   * **epsilon**（[float](https://docs.python.org/3/library/functions.html#float)）：避免被零除的 epsilon 值。。
* **返回：result** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)形状为 (d_0, d_1, …, d_{N–1}) 的 ND。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.layer_norm(*data*, *gamma*, *beta*, *axis*, *epsilon=1e-05*)


层归一化运算符。它接受 fp16 和 fp32 作为输入数据类型。它会将输入转换为 fp32 来执行计算。输出将具有与输入相同的数据类型。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：形状为（d_0，d_1，…，d_{N–1}）的 ND。
   * **gamma** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：形状为 (r_0, r_1, …, r_{K–1}) 的 KD，其中 K == len(axis) 且 d_{axis_k} == r_k。
   * **beta**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)可选，KD 形状为 (r_0, r_1, …, r_{K–1})，其中 K == len(axis) 且 d_{axis_k} == r_k。
   * **axis**（*int*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：应用规范化的[轴](https://docs.python.org/3/library/functions.html#int)。
   * **epsilon**（[float](https://docs.python.org/3/library/functions.html#float)）：避免被零除的 epsilon 值。
* **返回：result** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)形状为 (d_0, d_1, …, d_{N–1}) 的 ND。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.group_norm(*data*, *gamma*, *beta*, *num_groups*, *channel_axis*, *axes*, *epsilon=1e-05*)


组规范化运算符。它接受 fp16 和 fp32 作为输入数据类型。它会将输入转换为 fp32 来执行计算。输出将具有与输入相同的数据类型。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)形状为（d_0，d_1，…，d_{N–1}）的 ND。
   * **gamma** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )：一维，形状为 (r_0)，其中 r_0 == d_{channel_axis}。
   * **beta**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：可选，一维，形状为（r_0），其中 r_0 == d_{channel_axis}。
   * **num_groups**（[int](https://docs.python.org/3/library/functions.html#int)）*：* 组数。
   * **channel_axis**（[int](https://docs.python.org/3/library/functions.html#int)）：通道轴。
   * **轴**（*整数*[列表）](https://docs.python.org/3/library/stdtypes.html#list)[：](https://docs.python.org/3/library/stdtypes.html#list)[应用标准化](https://docs.python.org/3/library/stdtypes.html#list)[的](https://docs.python.org/3/library/functions.html#int)轴，不包括通道轴。
   *   **epsilon**（[float](https://docs.python.org/3/library/functions.html#float)）：避免被零除的 epsilon 值。
* **返回：result** [：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)形状为 (d_0, d_1, …, d_{N–1}) 的 ND。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.nn.rms_norm(*data*, *weight*, *axis*, *epsilon=1e-05*)


均方根归一化运算符。输出将具有与输入相同的数据类型。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：形状为（d_0，d_1，…，d_{N–1}）的 ND。
   * **权重**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 形状为 (r_0, r_1, …, r_{K–1}) 的 KD，其中 K == len(axis) 且 d_{axis_k} == r_k。
   * **axis**（*int*[列表](https://docs.python.org/3/library/stdtypes.html#list)）：应用规范化的[轴](https://docs.python.org/3/library/functions.html#int)。
   * **epsilon**（[float](https://docs.python.org/3/library/functions.html#float)）*：* 避免被零除的 epsilon 值。
* **返回：result** ：形状为 (d_0, d_1, …, d_{N–1}) 的 ND。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。




# tvm.topi.image




IMAGE 网格操作符。


**函数：**

|[affine_grid](/docs/api-reference/python-api/tvm-topi#tvmtopiimageaffine_griddata-target_shape)(data,target_shape)|生成二维采样网格的 affine_grid 操作符。|
|:----|:----|
|[can_convert_multiply_to_intdiv](/docs/api-reference/python-api/tvm-topi#tvmtopiimagecan_convert_multiply_to_intdivorigin_size-scaled_size)(origin_size,…)|检查是否可以将乘法转换为除法。|
|[crop_and_resize](/docs/api-reference/python-api/tvm-topi#tvmtopiimagecrop_and_resizedata-boxes-box_indices-crop_size-layoutnchw-methodbilinear-extrapolation_valuenone-out_dtypenone)(data,boxes,box_indices,…)|对数据执行裁剪和调整大小操作。|
|[dilation2d_nchw](/docs/api-reference/python-api/tvm-topi#tvmtopiimagedilation2d_nchwinput-filter-stride-padding-dilations-out_dtypenone)(input,filter,stride,…)|NCHW 布局中的形态膨胀操作符。|
|[dilation2d_nhwc](/docs/api-reference/python-api/tvm-topi#tvmtopiimagedilation2d_nhwcinput-filter-stride-padding-dilations-out_dtypenone)(input,filter,stride,…)|形态学二维扩张 NHWC 布局。|
|[get_1d_indices](/docs/api-reference/python-api/tvm-topi#tvmtopiimageget_1d_indicesindices-layoutncw)(indices[,layout])|获取一维索引。|
|[get_1d_pixel](/docs/api-reference/python-api/tvm-topi#tvmtopiimageget_1d_pixeldata-layout-image_width-n-c-x-cc-ib-ic)(data,layout,image_width,n,…)|获取 1d 像素。|
|[get_2d_indices](/docs/api-reference/python-api/tvm-topi#tvmtopiimageget_2d_indicesindices-layoutnchw)(indices[,layout])|获取二维索引。|
|[get_2d_pixel](/docs/api-reference/python-api/tvm-topi#tvmtopiimageget_2d_pixeldata-layout-image_height-image_width-n-c-y-x-cc-ib-ic)(data,layout,image_height,…)|获取二维像素。|
|[get_3d_indices](/docs/api-reference/python-api/tvm-topi#tvmtopiimageget_3d_indicesindices-layoutncdhw)(indices[,layout])|获取 3d 索引。|
|[get_3d_pixel](/docs/api-reference/python-api/tvm-topi#tvmtopiimageget_3d_pixeldata-layout-image_depth-image_height-image_width-n-c-z-y-x-cc)(data,layout,image_depth,…)|获取 3d 像素。|
|[get_closest_index](/docs/api-reference/python-api/tvm-topi#tvmtopiimageget_closest_indexin_x-rounding_method-boxes-use_int_divfalse)(in_x,rounding_method,boxes)|根据某种舍入方法获取最接近某个值的索引。|
|[get_inx](/docs/api-reference/python-api/tvm-topi#tvmtopiimageget_inxx-image_width-target_width-coordinate_transformation_mode-start_x0-end_x-1-use_int_divfalse)(x,image_width,target_width,…[,…])|使用各种坐标变换方法从输出 x 推断输入 x。|
|[get_pad_tuple](/docs/api-reference/python-api/tvm-topi#tvmtopiimageget_pad_tuplepadding-kernel)(padding,kernel)|获取 pad 选项的通用代码。|
|[grid_sample](/docs/api-reference/python-api/tvm-topi#tvmtopiimagegrid_sampledata-grid-methodbilinear-layoutnchw-padding_modezeros-align_cornerstrue)(data,grid[,method,layout,…])|将网格采样应用于输入特征图。|
|[nchw_pack_layout](/docs/api-reference/python-api/tvm-topi#tvmtopiimagenchw_pack_layoutlayout_info)(layout_info)|检查布局类型是否为 NCHWinic。|
|[nchw_xc_layout](/docs/api-reference/python-api/tvm-topi#tvmtopiimagenchw_xc_layoutlayout_info)(layout_info)|检查布局类型是否为 NCHWxc。|
|[pad](/docs/api-reference/python-api/tvm-topi#tvmtopiimagepaddata-pad_before-pad_afternone-pad_value00-namepadinput-attrsnone)(data,pad_before[,pad_after,…])|使用 pad 值的 Pad 输入。|
|[resize1d](/docs/api-reference/python-api/tvm-topi#tvmtopiimageresize1ddata-roi-size-layoutncw-methodlinear-coordinate_transformation_modehalf_pixel-rounding_method-bicubic_alpha-075-bicubic_exclude0-extrapolation_value00-out_dtypenone-output_shapenone)(data,roi,size[,layout,method,…])|对数据执行调整大小操作。|
|[resize2d](/docs/api-reference/python-api/tvm-topi#tvmtopiimageresize2ddata-roi-size-layoutnchw-methodlinear-coordinate_transformation_modehalf_pixel-rounding_method-bicubic_alpha-075-bicubic_exclude0-extrapolation_value00-out_dtypenone-output_shapenone)(data,roi,size[,layout,method,…])|对数据执行调整大小操作。|
|[resize3d](/docs/api-reference/python-api/tvm-topi#tvmtopiimageresize3ddata-roi-size-layoutncdhw-methodlinear-coordinate_transformation_modehalf_pixel-rounding_method-bicubic_alpha-075-bicubic_exclude0-extrapolation_value00-out_dtypenone-output_shapenone)(data,roi,size[,layout,method,…])|对数据执行调整大小操作。|
|[simplify](/docs/api-reference/python-api/tvm-topi#tvmtopiimagesimplifyexpr)(expr)|如果是 Expr 则化简表达式，如果是 int 则直接返回。|

## tvm.topi.image.affine_grid(*data*, *target_shape*)


生成二维采样网格的 affine_grid 操作符。


[此操作在 https://arxiv.org/pdf/1506.02025.pdf](https://arxiv.org/pdf/1506.02025.pdf) 中进行了描述。它在目标形状内生成一个均匀的采样网格，并将其归一化到 [-1, 1]。然后将提​​供的仿射变换应用于采样网格。
* **参数：**
   * **数据**（*tvm.Tensor*）：三维，形状为 [batch, 2, 3]。仿射矩阵。
   * **target_shape**（*两个 int**的***列表/元组）[：](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)指定输出形状（H，W）。
* **返回：Output** **：**4–D，形状为 [batch, 2, target_height, target_width]。
* **返回类型：** tvm.Tensor。

## tvm.topi.image.can_convert_multiply_to_intdiv(*origin_size*, *scaled_size*)


检查是否可以将乘法转换为除法。

## tvm.topi.image.crop_and_resize(*data*, *boxes*, *box_indices*, *crop_size*, *layout='NCHW'*, *method='bilinear'*, *extrapolation_value=None*, *out_dtype=None*)


对数据执行裁剪和调整大小操作。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入是形状为 [batch, channel, in_height, in_width] 或 [batch, in_height, in_width, channel] 的 4–D 张量。
   * **boxes** ( [tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor) )*：* 形状为 [num_boxes, 4] 的二维张量。张量的每一行指定一个框的坐标。
   * **box_indices**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：形状为 [num_boxes] 的一维张量，box_indices[i] 指定第 i 个框引用的数据。
   * **crop_size**（[元组](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)）：每个框的目标大小。
   * *布局**（*字符串*，可选）：“NCHW”，“NHWC”。
   * *方法**（*{“bilinear”，*“nearest_neighbor”}*）：用于调整大小的方法。
   * **extrapolation_value**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）*：*用于外推的值（如适用）。
   * *out_dtype*（*字符串*，可选）：返回类型。如果为 None ，则返回与输入类型相同的类型。
* **返回：output** ：4–D，形状为 [num_boxes, channel, crop_height, crop_width] 或 [num_boxes, crop_height, crop_width, channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.image.dilation2d_nchw(*input*, *filter*, *stride*, *padding*, *dilations*, *out_dtype=None*)


NCHW 布局中的形态膨胀操作符。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_channel, in_height, in_width]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：三维，形状为[in_channel, filter_height, filter_width]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int***的列表/元组*）**：步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小。
   * **扩张**（[int](https://docs.python.org/3/library/functions.html#int)*或**两个 int***的列表/元组）：扩张大小，或 [dilation_height, dilation_width]。
   * *out_dtype*（*可选*[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]*）*：* 指定输出数据类型。
* **返回：Output** **：** 4–D，形状为 [batch, in_channel, out_height, out_width]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.image.dilation2d_nhwc(*input*, *filter*, *stride*, *padding*, *dilations*, *out_dtype=None*)


形态学二维扩张 NHWC 布局。
* **参数：**
   * **输入**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：4–D，形状为[batch, in_height, in_width, in_channel]。
   * **过滤器**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）*：* 三维，形状为[filter_height, filter_width, in_channel]。
   * **stride**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）*：* 步幅大小，或 [stride_height, stride_width]。
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)）**：** 填充大小。
   * **扩张**（[int](https://docs.python.org/3/library/functions.html#int)*或*两个 int*的列表/元组*）：扩张大小，或 [dilation_height, dilation_width]。
   * *out_dtype*（*可选*[ [str](https://docs.python.org/3/library/stdtypes.html#str)*]*）：指定输出数据类型。
* **返回：Output** **：** 4–D，形状为 [batch, out_height, out_width, in_channel]。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.image.get_1d_indices(*indices*, *layout='NCW'*)


获取一维索引。

## tvm.topi.image.get_1d_pixel(*data*, *layout*, *image_width*, *n*, *c*, *x*, *cc*, *ib*, *ic*)


获取 1d 像素。

## tvm.topi.image.get_2d_indices(*indices*, *layout='NCHW'*)


获取二维索引。

## tvm.topi.image.get_2d_pixel(*data*, *layout*, *image_height*, *image_width*, *n*, *c*, *y*, *x*, *cc*, *ib*, *ic*)


获取二维像素。

## tvm.topi.image.get_3d_indices(*indices*, *layout='NCDHW'*)


获取 3d 索引。

## tvm.topi.image.get_3d_pixel(*data*, *layout*, *image_depth*, *image_height*, *image_width*, *n*, *c*, *z*, *y*, *x*, *cc*)


获取 3d 像素。

## tvm.topi.image.get_closest_index(*in_x*, *rounding_method*, *boxes*, *use_int_div=False*)


根据某种舍入方法获取最接近某个值的索引。

## tvm.topi.image.get_inx(*x*, *image_width*, *target_width*, *coordinate_transformation_mode*, *start_x=0*, *end_x=-1*, *use_int_div=False*)


使用各种坐标变换方法从输出 x 推断输入 x。

## tvm.topi.image.get_pad_tuple(*padding*, *kernel*)


获取 pad 选项的通用代码。
* **参数：**
   * **padding**（[int](https://docs.python.org/3/library/functions.html#int)*或*[str](https://docs.python.org/3/library/stdtypes.html#str)）：填充大小，或 ['VALID', 'SAME']。
   * **kernel**（[int 元](https://docs.python.org/3/library/functions.html#int)*组*[）](https://docs.python.org/3/library/stdtypes.html#tuple)：卷积核大小。
* **返回：**
   *  **pad_top**（*int*）：顶部填充大小。
   * **pad_left**（*int*）：左侧填充大小。
   * **pad_down**（*int*）：向下填充大小。
   * **pad_right**（*int*）：右侧填充大小。

## tvm.topi.image.grid_sample(*data*, *grid*, *method='bilinear'*, *layout='NCHW'*, *padding_mode='zeros'*, *align_corners=True*)


将网格采样应用于输入特征图。


鉴于 data 和 grid，那么对于 4-D，输出计算如下。

xsrc=grid[batch,0,ydst,xdst]ysrc=grid[batch,1,ydst,xdst]output[batch,channel,ydst,xdst]=G(data[batch,channel,ysrc,xsrc])。


xdst，ydst 枚举所有空间位置 output， 和 G()表示插值函数。


如果 padding_mode 为“zeros”，则外边界点将用零填充；如果 padding_mode 为“border”，则外边界点将用边界像素值填充；如果 padding_mode 为“reflection”，则外边界点将用内部像素值填充。


如果 align_corners 为“True”，则网格左上角 (-1, -1) 和右下角 (1, 1) 将映射到数据的 (0, 0) 和 (h - 1, w - 1)；如果 align_corners 为“False”，则网格左上角 (-0.5, -0.5) 和 (h - 0.5, w - 0.5)。


输出的形状将是 4-D（data.shape[0]、data.shape[1]、grid.shape[2]、grid.shape[3]）或 5-D（data.shape[0]、data.shape[1]、grid.shape[2]、grid.shape[3]、grid.shape[4]）。


操作员假设 grid 已标准化为[-1,1]。


grid_sample 经常与 affine_grid 配合使用，后者为 grid_sample 生成采样网格。
* **参数：**
   * **数据**（*tvm.Tensor*）：形状为 [batch, in_channel, in_height, in_width] 的 4–D 数据，或形状为 [batch, in_channel, in_depth, in_height, in_width] 的 5–D 数据。
   * **网格**（*tvm.Tensor*）*：* 形状为 [batch, 2, out_height, out_width] 的 4 维网格，或形状为 [batch, 3, out_depth, out_height, out_width] 的 5 维网格。
   * **方法**（[str](https://docs.python.org/3/library/stdtypes.html#str)）：插值方法，支持4维「最近」、「双线性」、「双三次」和 5 维「最近」、「双线性」（「三线性」）。
   * **布局**（[str](https://docs.python.org/3/library/stdtypes.html#str)）*：* 输入数据和输出的布局。
   * **padding_mode** ( [str](https://docs.python.org/3/library/stdtypes.html#str) )：外部网格值的填充模式，支持「零」、「边框」、「反射」。
   * **align_corners** ( [bool](https://docs.python.org/3/library/functions.html#bool) )：从几何学角度来看，我们将输入像素视为正方形而非点。如果设置为“True”，则极值（“–1”和“1”）将被视为指向输入角点像素的中心点。如果设置为“False”，则它们将被视为指向输入角点像素的角点，从而使采样更加不受分辨率影响。
* **返回：Output** ：形状为 [batch, in_channel, out_height, out_width] 的 4–D 或形状为 [batch, in_channel, out_depth, out_height, out_width] 的 5–D。
* **返回类型：** tvm.Tensor。

## tvm.topi.image.nchw_pack_layout(*layout_info*)


检查布局类型是否为 NCHWinic。

## tvm.topi.image.nchw_xc_layout(*layout_info*)


检查布局类型是否为 NCHWxc。

## tvm.topi.image.pad(*data*, *pad_before*, *pad_after=None*, *pad_value=0.0*, *name='PadInput'*, *attrs=None*)


使用 pad 值的 Pad 输入。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：nD 输入，可以是任何布局。
   * **pad_before**（*n 个整数**的***列表/元组）：在每个维度上填充宽度以在轴开始之前进行填充。
   * *pad_after**（*n 个整数**的**列表/元组*，*可选*）：填充每个维度的宽度以填充轴端之后。
   * **pad_value**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：要填充的值。
   * **name**（[str](https://docs.python.org/3/library/stdtypes.html#str)*，可选*）：生成的名称前缀运算符。
* **返回：Output** ：nD，与输入相同的布局。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.image.resize1d(*data*, *roi*, *size*, *layout='NCW'*, *method='linear'*, *coordinate_transformation_mode='half_pixel'*, *rounding_method=''*, *bicubic_alpha=-0.75*, *bicubic_exclude=0*, *extrapolation_value=0.0*, *out_dtype=None*, *output_shape=None*)


对数据执行调整大小操作。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入是形状为[batch, channel in_width]或[batch in_width, channel]的三维张量。
   * **roi**（*浮点数**或** **表达式的*[元组](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)）**：** 用于裁剪输入图像的感兴趣区域。预期大小为 2，格式为 [start_w, end_w]。仅在 coordinate_transformation_mode 为 tf_crop_and_resize 时使用。
   * **size**（[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)）*：* 输出分辨率缩放至。
   * *布局*（*字符串*，*可选*）：“NCW”、“NWC”或“NCWc”。
   * *coordinate_transformation_mode*（*字符串**，*可选*）*：* 描述如何将调整大小后的张量中的坐标转换为原始张量中的坐标。有关详细信息，请参阅 ONNX Resize 运算符规范。可用选项包括“half_pixel”、“align_corners”和“asymmetric”。
   * *方法**（*字符串**，可选*）：插值方法（“最近”、“线性”、“双三次”）。
   * **coordinate_transformation_mode**：描述如何将调整大小后的张量中的坐标转换为原始张量中的坐标。[half_pixel、align_corners、asymmetric、pytorch_half_pixel、tf_half_pixel_for_nn 和 tf_crop_and_resize]。
   * **rounding_method**：对坐标位置进行舍入的方法。
   * **bicubic_alpha**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：双三次样条系数。
   * **bicubic_exclude** ( [bool](https://docs.python.org/3/library/functions.html#bool)*,可选:* )：排除图像 fdor 双三次插值之外的值。
   * **extrapolation_value**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：用于外推的值（如适用）。
   * *out_dtype*（*字符串*，*可选*）*：* 返回类型。如果为 None ，则返回与输入类型相同的类型。
   * *output_shape*（tvm.tir.container.Array，*可选*）：返回的形状。如果为 None，则推断为 None （如果形状是动态确定的，则将 out_dtype.shape 传递为 output_shape）。
* **返回：output** ：形状为 [batch, chananel, in_width*scale] 或 [batch, in_width*scale, channel] 的 4–D 或形状为 [batch, channel–major, in_width*scale, channel–minor] 的 5–D。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)

## tvm.topi.image.resize2d(*data*, *roi*, *size*, *layout='NCHW'*, *method='linear'*, *coordinate_transformation_mode='half_pixel'*, *rounding_method=''*, *bicubic_alpha=-0.75*, *bicubic_exclude=0*, *extrapolation_value=0.0*, *out_dtype=None*, *output_shape=None*)


对数据执行调整大小操作。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入是形状为 [batch, channel, in_height, in_width] 或 [batch, in_height, in_width, channel] 的 4–D 张量。
   * **roi**（*浮点数**或****表达式的*[元组](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)）：用于裁剪输入图像的感兴趣区域。预期大小为 4，格式为 [start_h, start_w, end_h, end_w]。仅在 coordinate_transformation_mode 为 tf_crop_and_resize 时使用。
   * **size**（[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)）：输出分辨率缩放至。
   * *布局**（*字符串*，*可选*）：“NCHW”、“NHWC”或“NCHWc”。
   * *方法**（*字符串*，*可选*）：插值方法（「最近」、「线性」、「双三次」）。
   * *coordinate_transformation_mode*（*字符串*，*可选*）：描述如何将调整大小后的张量中的坐标转换为原始张量中的坐标。[half_pixel、align_corners、asymmetric、pytorch_half_pixel、tf_half_pixel_for_nn 和 tf_crop_and_resize]。
   * **rounding_method：** 对坐标位置进行舍入的方法。
   * **bicubic_alpha**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）*：* 双三次样条系数。
   * **bicubic_exclude** ( [bool](https://docs.python.org/3/library/functions.html#bool)*,可选:* )：排除图像 fdor 双三次插值之外的值。
   * **extrapolation_value**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：用于外推的值（如适用）。
   * *out_dtype*（*字符串*，可选）*：* 返回类型。如果为 None ，则返回与输入类型相同的类型。
   * *output_shape*（tvm.tir.container.Array，***可选*）*：* 返回的形状。如果为 None，则推断为 None （如果形状是动态确定的，则将 out_dtype.shape 传递为 output_shape）。
* **返回：output** ：形状为 [batch, channel, in_height*scale, in_width*scale] 或 [batch, in_height*scale, in_width*scale, channel] 的 4–D 或形状为 [batch, channel–major, in_height*scale, in_width*scale, channel–minor] 的 5–D。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.image.resize3d(*data*, *roi*, *size*, *layout='NCDHW'*, *method='linear'*, *coordinate_transformation_mode='half_pixel'*, *rounding_method=''*, *bicubic_alpha=-0.75*, *bicubic_exclude=0*, *extrapolation_value=0.0*, *out_dtype=None*, *output_shape=None*)


对数据执行调整大小操作。
* **参数：**
   * **数据**（[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)）：输入是一个 5–D 张量，形状为 [batch, channel, in_depth, in_height, in_width] 或 [batch, in_depth, in_height, in_width, channel]。
   * **roi**（*浮点数**或*** *表达式的*[元组](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)）**：** 用于裁剪输入图像的感兴趣区域。预期大小为 6，格式为 [start_d, start_h, start_w, end_d, end_h, end_w]。仅在 coordinate_transformation_mode 为 tf_crop_and_resize 时使用。
   * **size**（[Tuple](/docs/api-reference/python-api/tvm-relax#classtvmrelaxtuplefieldslistrelaxexprtuplerelaxexprspanspannonenone)）：输出分辨率缩放至。
   * *布局**（*字符串*，*可选*）：“NCDHW”、“NDHWC”或“NCDHWc”。
   * *方法**（*字符串*，*可选*）：插值方法（“最近”、“线性”、“双三次”）。
   * *coordinate_transformation_mode*（*字符串*，*可选*）：描述如何将调整大小后的张量中的坐标转换为原始张量中的坐标。[half_pixel、align_corners、asymmetric、pytorch_half_pixel、tf_half_pixel_for_nn 和 tf_crop_and_resize]。
   * **rounding_method**：对坐标位置进行舍入的方法。
   * **bicubic_alpha**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：双三次样条系数。
   * **bicubic_exclude** ( [bool](https://docs.python.org/3/library/functions.html#bool)*,可选:* )：排除图像 fdor 双三次插值之外的值。
   * **extrapolation_value**（[float](https://docs.python.org/3/library/functions.html#float)*，可选*）：用于外推的值（如适用）。
   * *out_dtype*（*字符串*，可选）：返回类型。如果为 None ，则返回与输入类型相同的类型。
   * *output_shape*（tvm.tir.container.Array，*可选*）：返回的形状。如果为 None，则推断为 None （如果形状是动态确定的，则将 out_dtype.shape 传递为 output_shape）。
* **返回：output** ：形状为 [batch, channel, in_depth*scale, in_height*scale, in_width*scale] 或 [batch, in_depth*scale, in_height*scale, in_width*scale, channel] 的 4–D 或形状为 [batch, channel–major, in_depth*scale, in_height*scale, in_width*scale, channel–minor] 的 5–D。
* **返回类型：**[tvm.te.Tensor](/docs/api-reference/python-api/tvm-te#class-tvmtetensor)。

## tvm.topi.image.simplify(*expr*)

如果是 Expr 则化简表达式，如果是 int 则直接返回。
* **参数：expr** (*Expror*[int](https://docs.python.org/3/library/functions.html#int)) ：输入。
* **返回：out** ：简化的输出。
* **返回类型：** Expr or [int](https://docs.python.org/3/library/functions.html#int)。


