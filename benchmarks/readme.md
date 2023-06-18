This folder encompasses all the code for conducting benchmarks on NFTGraph. NFTGraph exhibits a wealth of features at node, edge, and subgraph levels. Therefore, three experiments have been conducted on NFTGraph, targeting different levels of the graph.

## Algorithms:
### node classification (anomaly detection)
We select several commonly used anomaly detection models provided on Python Graph Outlier Detection (PyGOD), a benchmark tool for outlier detection. It is a free tool and available at https://github.com/pygod-team/pygod/.

### link prediction

We select eight state-of-the-art graph neural networks on OGB leaderboards. Thanks to `dgl` package, we benchmark them on NFTGraph easily. The reference codes are listed as follows:

SEAL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/seal_ogbl

SEAL+NGNN: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ngnn_seal

SUREL: https://github.com/Graph-COM/SUREL

SIEG: https://github.com/anonymous20221001/SIEG_OGB

AGDN: https://github.com/skepsun/Adaptive-Graph-Diffusion-Networks

GraphSAGE: https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage

GAT: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-products/gat

GCN: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn


### graph self-supervised learning
We select seven commonly used graph self-supervised learning algorithms and evaluate them on NFTGraph. Thanks for the open-sourced codes, we list them in the following:

ParetoGNN: https://github.com/jumxglhf/ParetoGNN

GraphMAE: https://github.com/THUDM/GraphMAE

DGI: https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgi

MVGRL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/mvgrl

CCA-SSG: https://github.com/hengruizhang98/

BGRL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/bgrl

GRACE: https://github.com/dmlc/dgl/tree/master/examples/pytorch/grace


## Results:
### node classification (anomaly detection)

|Models |AUC | AP |
|-|-|-|
|DONE |	0.5241±0.0140	|0.0097 ±0.0003|
|AdONE |0.5292 ±0.0060 |	0.0096 ±0.0002|
|AnomalyDAE |	0.4302 ±0.0310|	0.0070 ±0.0004|
|GCNAE |	0.5813 ±0.0021	|0.0126 ±0.0000|
|MLPAE |	0.5914 ±0.0008  |0.0145 ±0.0000|
|CONAD |	0.6101 ±0.0141| 0.0168 ±0.0034|
|DOMINANT |	0.6077 ±0.0049|	0.0169 ±0.0013|
|GAAN |	0.5771 ±0.0005	|0.0117 ±0.0000|

### link prediction

|Models|MRR|AUC|
|-|-|-|
SEAL |0.9882 ±0.0002|0.9770 ±0.0001|
SEAL+NGNN |0.9885 ±0.0001|0.9773 ±0.0003|
SUREL |0.9762 ±0.0008|0.9529 ±0.0016|
SIEG |0.9754 ±0.0000|0.9540 ±0.0003|
AGDN |0.9587 ±0.0069|0.9181 ±0.0137|
GraphSAGE |0.9560 ±0.0045|0.9126 ±0.0088|
GAT |0.9445 ±0.0051|0.8897 ±0.0098|
GCN |0.7373 ±0.0385|0.4834 ±0.0778|

### graph self-supervised learning

Models|ACC|NMI|METIS|HMEANS|RANK|
|-|-|-|-|-|-|
ParetoGNN |0.8603 ±0.0001|0.0004 ±0.0000}|0.2615 ±0.0064|0.3741|2.67|
GraphMAE |0.8508 ±0.0001|0.0167 ±0.0000|0.1228 ±0.0056|0.3301|4|
DGI |0.8554 ±0.0001|0.0059 ±0.0000|0.1842 ±0.0299|0.3485|3.33|
MVGRL |OOM|OOM|OOM|OOM|OOM|
CCA-SSG |0.8558 ±0.0001|0.0052 ±0.0000|0.1508 ±0.0277|0.3373|3.67|
BGRL |0.9486 ±0.0002|0.3460 ±0.0000|0.2561 ±0.0069|0.5169|1.33|
GRACE |OOM|OOM|OOM|OOM|OOM|
