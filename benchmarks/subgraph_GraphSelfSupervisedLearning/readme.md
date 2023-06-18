Graph self-supervised learning (GSSL) methods aim to learn effective representations in the pre-training stage 
on unlabeled data and leverage the learned graph embeddings for downstream tasks. 
NFTGraph's rich features make it well-suited for graph self-supervised learning 
since various pretext tasks need to be conducted on the graph. 
We evaluate seven models, namely ParetoGNN, GraphMAE, DGI, MVGRL, CCA-SSG, BGRL, and GRACE, 
on three downstream tasks: node classification, graph partitioning, and node clustering.
The performance of these tasks is measured using ACC, METIS, and NMI, respectively. 
HMEANS takes the average of these three metrics for each model, 
while RANK represents the ranking of these models among all downstream tasks.


### Settings and Hyperparameters:
| Metrics | Values |
|-|-|
| type of NFTGraph | NFTGraph-Tiny |
| runs | 5 |
epoch | {ParetoGNN,CCA-SSG,:100; BGRL: 1000 ; DGI: [100000, 300] ; GraphMAE: 200} |
batch_size |  10240 
learning_rate | {CCA-SSG, DGI: [1e-3,1e-2] ; BGRL: 1e-5 }
hidden dimensions | {ParetoGNN, BGRL: [256,128] ; CCA-SSG: [256,256] ; DGI: 128 } 


### Results:
Models|ACC|NMI|METIS|HMEANS|RANK|
|-|-|-|-|-|-|
ParetoGNN |0.8603 ±0.0001|0.0004 ±0.0000}|0.2615 ±0.0064|0.3741|2.67|
GraphMAE |0.8508 ±0.0001|0.0167 ±0.0000|0.1228 ±0.0056|0.3301|4|
DGI |0.8554 ±0.0001|0.0059 ±0.0000|0.1842 ±0.0299|0.3485|3.33|
MVGRL |OOM|OOM|OOM|OOM|OOM|
CCA-SSG |0.8558 ±0.0001|0.0052 ±0.0000|0.1508 ±0.0277|0.3373|3.67|
BGRL |0.9486 ±0.0002|0.3460 ±0.0000|0.2561 ±0.0069|0.5169|1.33|
GRACE |OOM|OOM|OOM|OOM|OOM|
