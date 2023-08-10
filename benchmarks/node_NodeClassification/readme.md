
In contrast to those synthetic or injected anomaly graph datasets, NFTGraph is an unmasked dataset with many ground-truth anomalies in reality. 
Therefore, we aim to evaluate graph anomaly detection models on it by designing a node classification task to classify 
whether the node in NFTGraph is suspicious or not. 
We select eight models, namely AdONE, CONAD, 
DOMINANT, DONE, 
GAAN, AnomalyDAE, 
GCNAE, and MLPAE. 
We use AUC (Area Under the ROC Curve) and AP (Average Precision) as the evaluation metrics
to assess the performance of anomaly detection, with higher values indicating better performance. 

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
|GAT |0.9609 ±0.0045 |
|GraphSAGE |	0.9093 ±0.0013|
|Node2Vec |	0.7456 ±0.0021|
|MLP |	0.5570 ±0.0083 |

