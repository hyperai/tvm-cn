---
title: 在 CPU 上部署 Hugging Face 剪枝模型
---

# 在 CPU 上部署 Hugging Face 剪枝模型

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/deploy_models/deploy_sparse.html#sphx-glr-download-how-to-deploy-models-deploy-sparse-py) 下载完整的示例代码
:::

**作者**：[Josh Fromm](https://github.com/jwfromm)

本教程演示如何采用剪枝后的模型（本例中模型是 [来自 Hugging Face 的 PruneBert](https://huggingface.co/huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad)），并使用 TVM 来利用模型稀疏支持来加速。

尽管本教程的主要目的是在已经修剪过的模型上实现加速，但评估修剪后模型的速度也十分必要。为此，我们提供了一个函数采用未修剪的模型，并将其权重替换为指定稀疏的随机和修剪权重。确定模型是否值得修剪时，这可能是一个有用的特性。

进入代码前讨论一下稀疏和剪枝，并深入研究两种不同类型的稀疏：**结构化**和**非结构化**。

剪枝是一种主要通过将权重值替换为 0 来减小模型参数大小的技术，尽管选择哪些权重设置为 0 的方法众多，但最直接的方法是选择具有最小值的权重。

通常，权重会被修剪为所需的稀疏百分比。例如，一个 95% 的稀疏模型将只有 5% 的权重非零。修剪成非常高的稀疏通常需要微调，或完全重新训练，因为它是有损近似。尽管通过简单的压缩从修剪后的模型中很容易获得参数大小的好处，但利用模型稀疏来产生 runtime 加速更加复杂。

修剪结构化稀疏权重的目的，是把修剪过的权重聚集在一起。换言之，用它们的值和位置进行修剪。将修剪后的权重捆绑在一起的好处是它允许诸如矩阵乘法之类的算法跳过整个块。

事实证明，在当今可用的大多数硬件上，某种程度的*块稀疏*对于实现显著加速非常重要。这是因为在大多数 CPU 或 GPU 加载内存时，一次跳过读取单个值并不会节省任何工作，而是使用向量化指令之类的东西读入并执行整个块。

非结构化稀疏权重是仅根据原始权重值进行修剪的权重，它们看起来随机分散在整个张量中，而非像块稀疏权重中看到的那样成块。在低稀疏下，非结构化剪枝技术很难加速。然而，在高稀疏下，会出现许多全 0 值的块，这使得加速成为可能。

本教程包含结构化和非结构化稀疏。Hugging Face 的 PruneBert 模型是非结构化的，但 95% 是稀疏的，即使不是最优的，也可以对其应用 TVM 的块稀疏优化。

可以用结构化稀疏为未修剪模型生成随机稀疏权重。将 PruneBert 的真实速度与使用假权重的块稀疏速度比较，可以发现结构化稀疏的优势。

## 加载所需模块

除了 TVM，还需要 scipy（最新的 transformers）和 TensorFlow（版本在 2.2 以上）。

``` python
import os
import tvm
import time
import itertools
import numpy as np
import tensorflow as tf
from tvm import relay, runtime
from tvm.contrib import graph_executor
from tvm.relay import data_dep_optimization as ddo
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
import scipy.sparse as sp

# 要求 TensorFlow 将其 GPU 内存限制为实际需要的内存
# 而不是任其消耗其他内存。
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# 这样对 sphinx-gallery 更友好一点。
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("tensorflow will use experimental.set_memory_growth(True)")
    except RuntimeError as e:
        print("experimental.set_memory_growth option is not available: {}".format(e))
```

## 配置设置

从参数开始，定义要运行的模型类型和稀疏。

``` python
# 要下载和运行的 transformer 模型的名称
name = "huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad"
# 输入的 batches 数目
batch_size = 1
# 每个输入序列的长度。
seq_len = 128
# TVM 平台标识符。注意，可以通过设置 -mcpu 来实现最佳 CPU 性能
# 适合特定机器，还支持 CUDA 和 ROCm。
target = "llvm"
# 在哪个设备上运行，tvm.cpu() 或 tvm.cuda() 。
# 如果为 True，则将运行网络的稀疏变体，并进行 benchmark 测试。
measure_sparse = True
# 结构化稀疏块大小转换权重张量
＃ 进入。更改此参数可能会提高某些平台的速度。
bs_r = 1
# 对于除 PruneBert（95% 稀疏）以外的模型，此参数
# 确定生成的权重稀疏，值越大，稀疏越高，结果越快。
sparsity = 0.85
```

## 下载和转换 Transformers 模型

下面从 transformers 模块获取一个模型，下载并转换为 TensorFlow graphdef，将该 graphdef 转换为可以优化和部署的 Relay 计算图。

``` python
def load_keras_model(module, name, seq_len, batch_size, report_runtime=True):
    model = module.from_pretrained(name)
    dummy_input = tf.keras.Input(shape=[seq_len], batch_size=batch_size, dtype="int32")
    dummy_out = model(dummy_input)  # 通过 Keras 模型传播 shape。
    if report_runtime:
        np_input = np.random.uniform(size=[batch_size, seq_len], low=0, high=seq_len).astype(
            "int32"
        )
        start = time.time()
        repeats = 50
        for i in range(repeats):
            np_out = model(np_input)
        end = time.time()
        print("Keras Runtime: %f ms." % (1000 * ((end - start) / repeats)))
    return model

def convert_to_graphdef(model, batch_size, seq_len):
    model_func = tf.function(lambda x: model(x))
    input_dict = model._saved_model_inputs_spec
    input_spec = input_dict[list(input_dict.keys())[0]]
    model_func = model_func.get_concrete_function(
        tf.TensorSpec([batch_size, seq_len], input_spec.dtype)
    )
    frozen_func = convert_variables_to_constants_v2(model_func)
    return frozen_func.graph.as_graph_def()

def download_model(name, batch_size, seq_len):
    import transformers

    module = getattr(transformers, "TFBertForSequenceClassification")
    model = load_keras_model(module, name=name, batch_size=batch_size, seq_len=seq_len)
    return convert_to_graphdef(model, batch_size, seq_len)
```

## 转换为 Relay 计算图

目前有很多工具可以获得正确格式的 transformers 模型，从而进行 Relay 转换。下面的函数将导入的计算图保存为 Relay 的 json 格式，这样就不必在每次运行此脚本时从 TensorFlow 重新导入了。

``` python
def import_graphdef(
    name,
    batch_size,
    seq_len,
    save_relay=True,
    relay_file="model.json",
    relay_params="model.params",
):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    shape_dict = {"input_1": (batch_size, seq_len)}
    relay_file = ("%s_%d_%d_%s" % (name, batch_size, seq_len, relay_file)).replace("/", "_")
    relay_params = ("%s_%d_%d_%s" % (name, batch_size, seq_len, relay_params)).replace("/", "_")
    if os.path.exists(os.path.join(abs_path, relay_file)) and os.path.exists(
        os.path.join(abs_path, relay_params)
    ):
        with open(os.path.join(abs_path, relay_file), "r") as fi:
            mod = tvm.ir.load_json(fi.read())
        with open(os.path.join(abs_path, relay_params), "rb") as fi:
            params = relay.load_param_dict(fi.read())
    else:
        graph_def = download_model(name, batch_size, seq_len)

        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict)

        if save_relay:
            with open(os.path.join(abs_path, relay_file), "w") as fo:
                fo.write(tvm.ir.save_json(mod))
            with open(os.path.join(abs_path, relay_params), "wb") as fo:
                fo.write(runtime.save_param_dict(params))

    return mod, dict(params.items()), shape_dict
```

## 运行密集计算图

运行导入模型的默认版本。注意，即使权重是稀疏的，也不会看到任何加速，因为在这些密集（但大部分为零）张量上使用的是常规密集矩阵乘法，而非稀疏感知内核。

``` python
def run_relay_graph(mod, params, shape_dict, target, dev):
    with relay.build_config(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    input_shape = shape_dict["input_1"]
    dummy_data = np.random.uniform(size=input_shape, low=0, high=input_shape[1]).astype("int32")

    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input(0, dummy_data)
    m.run()
    tvm_output = m.get_output(0)

    print(m.benchmark(dev, repeat=5, number=5))
    return tvm_output

def run_dense(mod, params, shape_dict, target, dev):
    print("Dense Model Benchmark:")
    return run_relay_graph(mod, params, shape_dict, target, dev)
```

## 运行稀疏计算图

接下来把计算图转换为稀疏表示，并在需要时生成假的稀疏权重。然后用与 dense 相同的 benchmark 测试脚本来测试速度！对计算图应用一些 Relay pass，从而利用稀疏。

首先用 simple_fc_transpose 将密集层的权重转置到参数中，便于矩阵乘法转换为稀疏版本。接下来应用 bsr_dense.convert 来识别所有可以稀疏的权重矩阵，并自动替换它们。

下面的 bsr_dense.convert 函数通过检查 sparse_threshold 百分比稀疏，识别模型中的哪些权重可以变得稀疏，并将这些权重转换为 *Block Compressed Row Format (BSR)*。

BSR 本质上是一种对张量的 nonzero chunks 进行索引的表示，使算法可以轻松加载那些 nonzero chunks，并忽略张量的其余部分。一旦稀疏权重采用 BSR 格式，就会应用 relay.transform.DenseToSparse，实际上是用 relay.sparse_dense 函数来替换 relay.dense 操作，从而运行更快。

``` python
def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype="float32"):
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)
    ]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r : r + BS_R, c : c + BS_C] = np.random.uniform(-0.1, 0.1, (BS_R, BS_C))
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.data.size >= nnz
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (M // BS_R + 1,)
    return s.todense()

def random_sparse_bert_params(func, params, density, BS_R, BS_C):
    def deepcopy(param_dic):
        ret = {}
        for k, v in param_dic.items():
            ret[k] = tvm.nd.array(v.numpy())
        return ret

    new_params = deepcopy(params)
    dense_weight_names = relay.analysis.sparse_dense._search_dense_op_weight(func)
    for item in dense_weight_names:
        name = str(item)
        shape = new_params[name].shape
        if shape[0] % BS_R == 0 and shape[1] % BS_C == 0:
            new_w = random_bsr_matrix(shape[0], shape[1], BS_R, BS_C, density)
            new_params[name] = tvm.nd.array(new_w)
    return new_params

def run_sparse(mod, params, shape_dict, target, dev, bs_r, sparsity, gen_weights):
    mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    if gen_weights:
        params = random_sparse_bert_params(mod, params, BS_R=bs_r, BS_C=1, density=1 - sparsity)
    mod, params = ddo.bsr_dense.convert(mod, params, (bs_r, 1), sparsity_threshold=0.8)
    print("Block Sparse Model with {blocksize}x1 blocks:".format(blocksize=bs_r))
    return run_relay_graph(mod, params, shape_dict, target, dev)
```

## 运行所有代码

调用所有需要的函数，根据设置的参数对模型进行 benchmark 测试。注意，运行这个代码，首先需要取消最后一行的注释。

``` python
def benchmark():
    mod, params, shape_dict = import_graphdef(name, batch_size, seq_len)
    run_dense(mod, params, shape_dict, target, dev)
    if measure_sparse:
        gen_weights = "prune" not in name
        run_sparse(mod, params, shape_dict, target, dev, bs_r, sparsity, gen_weights)

# benchmark()
```

## 样本输出

可参考下面在 AMD CPU 上运行的脚本输出，显示用稀疏模型可提高约 2.5 倍的速度。

``` bash
# Dense Model Benchmark:
# Cannot find config for target=llvm, workload=('dense_nopack.x86', ('TENSOR', (1, 768), 'float32'), ('TENSOR', (2, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('dense_nopack.x86', ('TENSOR', (1, 768), 'float32'), ('TENSOR', (768, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('dense_nopack.x86', ('TENSOR', (128, 3072), 'float32'), ('TENSOR', (768, 3072), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('dense_nopack.x86', ('TENSOR', (128, 768), 'float32'), ('TENSOR', (3072, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('dense_nopack.x86', ('TENSOR', (128, 768), 'float32'), ('TENSOR', (768, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('batch_matmul.x86', ('TENSOR', (12, 128, 128), 'float32'), ('TENSOR', (12, 64, 128), 'float32')). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=llvm, workload=('batch_matmul.x86', ('TENSOR', (12, 128, 64), 'float32'), ('TENSOR', (12, 128, 64), 'float32')). A fallback configuration is used, which may bring great performance regression.
# Runtime:             165.26 ms           (12.83 ms)
# Block Sparse Model with 1x1 blocks:
# Runtime:             67.75 ms            (8.83 ms)

# Here is the output of this script on a GPU (GTX 1070) with the target "cuda -libs=cublas".
#
# Dense Model Benchmark:
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('dense_cublas.cuda', ('TENSOR', (1, 768), 'float32'), ('TENSOR', (2, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('dense_cublas.cuda', ('TENSOR', (1, 768), 'float32'), ('TENSOR', (768, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('dense_cublas.cuda', ('TENSOR', (128, 3072), 'float32'), ('TENSOR', (768, 3072), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('dense_cublas.cuda', ('TENSOR', (128, 768), 'float32'), ('TENSOR', (3072, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('dense_cublas.cuda', ('TENSOR', (128, 768), 'float32'), ('TENSOR', (768, 768), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('batch_matmul_cublas.cuda', ('TENSOR', (12, 128, 128), 'float32'), ('TENSOR', (12, 64, 128), 'float32'), (12, 128, 64)). A fallback configuration is used, which may bring great performance regression.
# Cannot find config for target=cuda -keys=cuda,gpu -libs=cublas -max_num_threads=1024 -thread_warp_size=32, workload=('batch_matmul_cublas.cuda', ('TENSOR', (12, 128, 64), 'float32'), ('TENSOR', (12, 128, 64), 'float32'), (12, 128, 128)). A fallback configuration is used, which may bring great performance regression.
# Runtime:             10.64 ms            (0.29 ms)
# Block Sparse Model with 1x1 blocks:
# Runtime:             6.46 ms             (0.05 ms)
```

[下载 Python 源代码：deploy_sparse.py](https://tvm.apache.org/docs/_downloads/9c3764c88ab3eb57dc223b4eda1e8a2f/deploy_sparse.py)

[下载 Jupyter Notebook：deploy_sparse.ipynb](https://tvm.apache.org/docs/_downloads/0b60295044fd20226a0d5adc52b50b2f/deploy_sparse.ipynb)