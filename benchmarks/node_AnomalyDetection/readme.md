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
batch_size | 512


### Results:

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
