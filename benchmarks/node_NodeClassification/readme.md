We design a node classification task to classify whether the node in NFTGraph is User node or Contract node. 
We select five models, namely GCN, GAT, GraphSAGE, Node2Vec, and MLP. 
We use AUC (Area Under the ROC Curve) as the evaluation metric to assess the performance of node classification, with higher values indicating better performance. 

### Settings and Hyperparameters:
| Metrics | Values |
|-|-|
| type of NFTGraph | NFTGraph-Tiny |
| runs | 5 |
epoch | 100 |
hidden_dim | 64


### Results:

|Models |AUC |
|-|-|
|GCN |	0.7757±0.0121	|
|GAT | 0.9609 ±0.0045 |
|GraphSAGE |	0.9093 ±0.0013|
|Node2Vec |	0.7411 ±0.0202|
|MLP |	0.5570 ±0.0083 |

