Given that NFTGraph is a complete graph, it is an ideal candidate for benchmarking link prediction models. 
In comparison to incomplete graphs, NFTGraph provides a more accurate result. 
To this end, we design a link prediction task and select five advanced GNN models, 
namely SEAL, SEAL+NGNN, SUREL, SIEG, and AGDN, as well as three basic GNN models: GraphSAGE, GAT, and GCN. 
We use two metrics, namely Mean Reciprocal Rank (MRR) and AUC, to evaluate their performance. 
MRR is the mean of reciprocal ranks measuring the reciprocal ranks over a set of listed results, 
and AUC is the area under the ROC curve. 


### Settings and Hyperparameters:
| Metrics | Values |
|-|-|
| type of NFTGraph | NFTGraph-User |
| runs | 10 |
epoch | 15 |
batch_size | {AGDN, GAT, GraphSAGE, GCN: 640 ; SEAL, SEAL+NGNN, SIEG: 64 }
hidden_dimensions | 32
learning_rate | 1e-03
eval_batch_size | 64 * 1000
eval_steps | 3
data split | 80/10/10


### Results:

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
