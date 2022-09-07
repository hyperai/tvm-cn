---
title: 构建图卷积网络
---

# 构建图卷积网络

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_relay/build_gcn.html#sphx-glr-download-how-to-work-with-relay-build-gcn-py) 下载完整的示例代码
:::

**作者**：[Yulun Yao](https://yulunyao.io/)，[Chien-Yu Lin](https://homes.cs.washington.edu/\~cyulin/)

本文介绍如何用 Relay 构建图卷积网络 (GCN)。本教程演示在 Cora 数据集上运行 GCN。Cora 数据集是图神经网络 (GNN) 的 benchmark，同时是支持 GNN 训练和推理的框架。我们直接从 DGL 库加载数据集来与 DGL 进行同类比较。

有关 DGL 安装，参阅 [DGL 文档](https://docs.dgl.ai/install/index.html)。

有关 PyTorch 安装，参阅 [PyTorch 指南](https://pytorch.org/get-started/locally/)。

## 使用 PyTorch 后端在 DGL 中定义 GCN

这部分重用了 [DGL 示例](https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn) 的代码。

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, g, n_infeat, n_hidden, n_classes, n_layers, activation):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(n_infeat, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_classes))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            # handle api changes for differnt DGL version
            # 处理不同 DGL 版本的不同函数
            if dgl.__version__ > "0.3":
                h = layer(self.g, h)
            else:
                h = layer(h, self.g)
        return h
```

输出结果：

``` bash
Using backend: pytorch
```

## 定义加载数据集和评估准确性的函数

可以将这部分替换为你自己的数据集，本示例中，我们选择从 DGL 加载数据：

``` python
from dgl.data import load_data
from collections import namedtuple

def load_dataset(dataset="cora"):
    args = namedtuple("args", ["dataset"])
    data = load_data(args(dataset))

    # 删除自循环，避免重复将节点的特征传递给自身
    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))
    g.add_edges_from(zip(g.nodes, g.nodes))

    return g, data

def evaluate(data, logits):
    test_mask = data.test_mask  # 未包含在训练阶段的测试集

    pred = logits.argmax(axis=1)
    acc = ((pred == data.labels) * test_mask).sum() / test_mask.sum()

    return acc
```

## 加载数据并设置模型参数

``` python
"""
Parameters
----------
dataset: str
    Name of dataset. You can choose from ['cora', 'citeseer', 'pubmed'].

num_layer: int
    number of hidden layers

num_hidden: int
    number of the hidden units in the hidden layer

infeat_dim: int
    dimension of the input features

num_classes: int
    dimension of model output (Number of classes)
"""

dataset = "cora"
g, data = load_dataset(dataset)

num_layers = 1
num_hidden = 16
infeat_dim = data.features.shape[1]
num_classes = data.num_labels
```

输出结果：

``` bash
Downloading /workspace/.dgl/cora_v2.zip from https://data.dgl.ai/dataset/cora_v2.zip...
Extracting file to /workspace/.dgl/cora_v2
Finished data loading and preprocessing.
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done saving data into cached files.
/usr/local/lib/python3.7/dist-packages/dgl/data/utils.py:286: UserWarning: Property dataset.graph will be deprecated, please use dataset[0] instead.
  warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))
/usr/local/lib/python3.7/dist-packages/dgl/data/utils.py:286: UserWarning: Property dataset.feat will be deprecated, please use g.ndata['feat'] instead.
  warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))
/usr/local/lib/python3.7/dist-packages/dgl/data/utils.py:286: UserWarning: Property dataset.num_labels will be deprecated, please use dataset.num_classes instead.
  warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))
```

## 设置 DGL-PyTorch 模型以取得最好的结果

用 https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/train.py 训练权重。

``` python
from tvm.contrib.download import download_testdata
from dgl import DGLGraph

features = torch.FloatTensor(data.features)
dgl_g = DGLGraph(g)

torch_model = GCN(dgl_g, infeat_dim, num_hidden, num_classes, num_layers, F.relu)

# 下载预训练的权重
model_url = "https://homes.cs.washington.edu/~cyulin/media/gnn_model/gcn_%s.torch" % (dataset)
model_path = download_testdata(model_url, "gcn_%s.pickle" % (dataset), module="gcn_model")

# 将 weights 加载到模型中
torch_model.load_state_dict(torch.load(model_path))
```

输出结果：

``` bash
/usr/local/lib/python3.7/dist-packages/dgl/data/utils.py:286: UserWarning: Property dataset.feat will be deprecated, please use g.ndata['feat'] instead.
  warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))
/usr/local/lib/python3.7/dist-packages/dgl/base.py:45: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  return warnings.warn(message, category=category, stacklevel=1)

<All keys matched successfully>
```

## 运行 DGL 模型并测试准确性

``` python
torch_model.eval()
with torch.no_grad():
    logits_torch = torch_model(features)
print("Print the first five outputs from DGL-PyTorch execution\n", logits_torch[:5])

acc = evaluate(data, logits_torch.numpy())
print("Test accuracy of DGL results: {:.2%}".format(acc))
```

输出结果：

``` bash
Print the first five outputs from DGL-PyTorch execution
 tensor([[-0.2198, -0.7980,  0.0784,  0.9232, -0.9319, -0.7733,  0.9410],
        [-0.4646, -0.6606, -0.1732,  1.1829, -0.3705, -0.5535,  0.0858],
        [-0.0031, -0.4156,  0.0175,  0.4765, -0.5887, -0.3609,  0.2278],
        [-0.8559, -0.8860,  1.4782,  0.9262, -1.3100, -1.0960, -0.0908],
        [-0.0702, -1.1651,  1.1453, -0.3586, -0.4938, -0.2288,  0.1827]])
/usr/local/lib/python3.7/dist-packages/dgl/data/utils.py:286: UserWarning: Property dataset.test_mask will be deprecated, please use g.ndata['test_mask'] instead.
  warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))
/usr/local/lib/python3.7/dist-packages/dgl/data/utils.py:286: UserWarning: Property dataset.label will be deprecated, please use g.ndata['label'] instead.
  warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))
Test accuracy of DGL results: 10.00%
```

## 在 Relay 中定义图卷积层

在 TVM 上运行 GCN 之前，首先实现 Graph Convolution Layer。参考 https://github.com/dmlc/dgl/blob/master/python/dgl/nn/mxnet/conv/graphconv.py 了解在 DGL 中使用 MXNet 后端实现的 GraphConv 层的更多信息。

该层由以下操作定义。注意：我们用两个转置来保持 sparse_dense 算子右侧的邻接矩阵，此方法是临时的，接下来几周内会更新稀疏矩阵转置，使得支持左稀疏算子。

$$
GraphConv(A,H,W)=A∗H∗W= ((H∗W)^{t}∗A^{t})^{t} = (( W^{t} ∗ H^{t})∗ A^{t} )^{t}
$$

``` python
from tvm import relay
from tvm.contrib import graph_executor
import tvm
from tvm import te

def GraphConv(layer_name, input_dim, output_dim, adj, input, norm=None, bias=True, activation=None):
    """
    参数
    ----------
    layer_name: str
    图层名称

    input_dim: int
    每个节点特征的输入维度

    output_dim: int,
    每个节点特征的输出维度

    adj: namedtuple,
    稀疏格式的图形表示（邻接矩阵）(`data`，`indices`，`indptr`)，其中`data`的 shape 为[num_nonzeros]，indices`的 shape 为[num_nonzeros]，`indptr`的 shape 为[num_nodes + 1]

    input: relay.Expr,
    shape 为 [num_nodes, input_dim] 的当前层的输入特征

    norm: relay.Expr,
    范数传给该层，对卷积前后的特征进行归一化。

    bias: bool
    将 bias 设置为 True，在处理 GCN 层时添加偏差

    activation: <function relay.op.nn>,
    激活函数适用于输出，例如 relay.nn.{relu，sigmoid，log_softmax，softmax，leaky_relu}

    返回
    ----------
    输出：tvm.relay.Expr
    该层的输出张量 [num_nodes, output_dim]
    """
    if norm is not None:
        input = relay.multiply(input, norm)

    weight = relay.var(layer_name + ".weight", shape=(input_dim, output_dim))
    weight_t = relay.transpose(weight)
    dense = relay.nn.dense(weight_t, input)
    output = relay.nn.sparse_dense(dense, adj)
    output_t = relay.transpose(output)
    if norm is not None:
        output_t = relay.multiply(output_t, norm)
    if bias is True:
        _bias = relay.var(layer_name + ".bias", shape=(output_dim, 1))
        output_t = relay.nn.bias_add(output_t, _bias, axis=-1)
    if activation is not None:
        output_t = activation(output_t)
    return output_t
```

## 准备 GraphConv 层所需的参数

``` python
import numpy as np
import networkx as nx

def prepare_params(g, data):
    params = {}
    params["infeats"] = data.features.numpy().astype(
        "float32"
    )  # 目前仅支持 float32 格式

    # 生成邻接矩阵
    adjacency = nx.to_scipy_sparse_matrix(g)
    params["g_data"] = adjacency.data.astype("float32")
    params["indices"] = adjacency.indices.astype("int32")
    params["indptr"] = adjacency.indptr.astype("int32")

    # 标准化 w.r.t.节点的度
    degs = [g.in_degree[i] for i in range(g.number_of_nodes())]
    params["norm"] = np.power(degs, -0.5).astype("float32")
    params["norm"] = params["norm"].reshape((params["norm"].shape[0], 1))

    return params

params = prepare_params(g, data)

# 检查特征的 shape 和邻接矩阵的有效性
assert len(params["infeats"].shape) == 2
assert (
    params["g_data"] is not None and params["indices"] is not None and params["indptr"] is not None
)
assert params["infeats"].shape[0] == params["indptr"].shape[0] - 1
```

输出结果：

``` bash
/usr/local/lib/python3.7/dist-packages/dgl/data/utils.py:286: UserWarning: Property dataset.feat will be deprecated, please use g.ndata['feat'] instead.
  warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))
```

## 逐层叠加

``` python
# 在 Relay 中定义输入特征、范数、邻接矩阵
infeats = relay.var("infeats", shape=data.features.shape)
norm = relay.Constant(tvm.nd.array(params["norm"]))
g_data = relay.Constant(tvm.nd.array(params["g_data"]))
indices = relay.Constant(tvm.nd.array(params["indices"]))
indptr = relay.Constant(tvm.nd.array(params["indptr"]))

Adjacency = namedtuple("Adjacency", ["data", "indices", "indptr"])
adj = Adjacency(g_data, indices, indptr)

# 构建 2 层 GCN
layers = []
layers.append(
    GraphConv(
        layer_name="layers.0",
        input_dim=infeat_dim,
        output_dim=num_hidden,
        adj=adj,
        input=infeats,
        norm=norm,
        activation=relay.nn.relu,
    )
)
layers.append(
    GraphConv(
        layer_name="layers.1",
        input_dim=num_hidden,
        output_dim=num_classes,
        adj=adj,
        input=layers[-1],
        norm=norm,
        activation=None,
    )
)

# 分析自由变量并生成 Relay 函数
output = layers[-1]
```

输出结果：

``` bash
/usr/local/lib/python3.7/dist-packages/dgl/data/utils.py:286: UserWarning: Property dataset.feat will be deprecated, please use g.ndata['feat'] instead.
  warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))
```

## 使用 TVM 编译和运行

将权重从 PyTorch 模型导出到 Python 字典：

``` python
model_params = {}
for param_tensor in torch_model.state_dict():
    model_params[param_tensor] = torch_model.state_dict()[param_tensor].numpy()

for i in range(num_layers + 1):
    params["layers.%d.weight" % (i)] = model_params["layers.%d.weight" % (i)]
    params["layers.%d.bias" % (i)] = model_params["layers.%d.bias" % (i)]

# 设置 TVM 构建 target
target = "llvm"  # 目前只支持 `llvm` 作为目标

func = relay.Function(relay.analysis.free_vars(output), output)
func = relay.build_module.bind_params_by_name(func, params)
mod = tvm.IRModule()
mod["main"] = func
# 使用 Relay 构建 
with tvm.transform.PassContext(opt_level=0):  # 目前只支持 opt_level=0
    lib = relay.build(mod, target, params=params)

# 生成图执行器
dev = tvm.device(target, 0)
m = graph_executor.GraphModule(lib["default"](dev))
```

## 运行 TVM 模型，测试准确性并使用 DGL 进行验证

``` python
m.run()
logits_tvm = m.get_output(0).numpy()
print("Print the first five outputs from TVM execution\n", logits_tvm[:5])

labels = data.labels
test_mask = data.test_mask

acc = evaluate(data, logits_tvm)
print("Test accuracy of TVM results: {:.2%}".format(acc))

import tvm.testing

# 使用 DGL 模型验证结果
tvm.testing.assert_allclose(logits_torch, logits_tvm, atol=1e-3)
```

输出结果：

```plain
Print the first five outputs from TVM execution
 [[-0.21976954 -0.7979525   0.07836491  0.9232204  -0.93188703 -0.7732947
   0.9410008 ]
 [-0.4645713  -0.66060466 -0.17316166  1.1828876  -0.37051404 -0.5534965
   0.08579484]
 [-0.00308266 -0.41562504  0.0175378   0.47649348 -0.5886737  -0.3609016
   0.22782072]
 [-0.8559376  -0.8860172   1.4782399   0.9262254  -1.3099641  -1.0960144
  -0.09084877]
 [-0.07015878 -1.1651071   1.1452857  -0.35857323 -0.49377596 -0.22878847
   0.18269953]]
/usr/local/lib/python3.7/dist-packages/dgl/data/utils.py:286: UserWarning: Property dataset.label will be deprecated, please use g.ndata['label'] instead.
  warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))
/usr/local/lib/python3.7/dist-packages/dgl/data/utils.py:286: UserWarning: Property dataset.test_mask will be deprecated, please use g.ndata['test_mask'] instead.
  warnings.warn('Property {} will be deprecated, please use {} instead.'.format(old, new))
Test accuracy of TVM results: 10.00%
```

[下载 Python 源代码：build_gcn.py](https://tvm.apache.org/docs/_downloads/dabb6b43ea9ef9d7bd1a3912001deace/build_gcn.py)

[下载 Jupyter Notebook：build_gcn.ipynb](https://tvm.apache.org/docs/_downloads/825671e45a9bdc4733400384984cd9dd/build_gcn.ipynb)