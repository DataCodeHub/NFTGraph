# NFTGraph
a new rich featured blockchain dataset for graph learning

## 1. Raw Data
This folder encompasses all the raw data for both NFTGraph-All and NFTGraph-Tiny. 

Additionally, obtaining the raw data for NFTGraph-User is straightforward, 
as it involves keeping all nodes labeled as *User*( `nodelabel` = 0 ) and their edges.

The `crawel` folder contains all the code for downloading all transcations from the Etherscan website.

### NFTGraph-All
There are two files: `nodes.csv` and `edges.csv` representing the node and edge lists of NFTGraph-All.
The files have been uploaded to Google Drive due to their large size.

#### nodes.csv:
download link: https://drive.google.com/file/d/1ksYE5alphvuxDn4CQVTLVh_HlBj2_Uia/view?usp=sharing

There are seven node features: `Address`, `OutAmount`, `OutValue`, `OutTxFee`, `InAmount`, `InValue`, and `InTxFee`. Aside from `Address`, which serves as the unique identifier for each node, the other features represent cumulants from transactions. Specifically, `OutAmount` and `InAmount` represent the cumulative tokens transferred from and to the current node, respectively. Similarly, `OutValue` and `InValue` indicate the cumulative transaction value when the current node is a source and target node separately and `OutTxFee` and `InTxFee` are the cumulative transaction fees paid by the current node as a sender and receiver, respectively.

In addition to these seven node features, it is worth noting that `node_id` corresponds to a unique numerical identifier assigned to each node. Furthermore, the `nodelabel` attribute serves as a binary indicator, where a value of 0 denotes a *User* node, and a value of 1 signifies a *Contract* node. For further elaboration and specific contextual meanings, please refer to the associated research paper.

#### metadata:
| indicator | value |
|-|-|
| total nodes | 1,172,856 |
| nodelabel distribution | { 0 : 1,161,847 ; 1 : 11,009 }
| dimensions of node features | 7 |
| node features | { Address, OutAmount, OutValue, OutTxFee, InAmount, InValue, InTxFee } |


#### preview:
| node_id | address | OutAmount | OutValue | OutTransFee | InAmount | InValue | InTransFee | nodelabel |
|-|-|-|-|-|-|-|-|-|
| 0 | 0x0000000000000000000000000000000000000000 | 7217724.0 | 712324959.5913919 | 21609773.15613218 | 2410795.0 | 20296148.734509576 | 4879685.167745654 | 0 |
|1|0x0000000000000000000000000000000000000001|0.0|0.0|0.0|140.0|0.0|264.18428919757656|0|
|2|0x0000000000000000000000000000000000000002|0.0|0.0|0.0|21.0|0.0|16.970000000000002|0|

#### edges.csv:
download link: https://drive.google.com/file/d/1ONpaBTrW-UupUV31SVYFlMROiL3LwbJ1/view?usp=sharing

Each edge has six features: `TxHash`, `Token`, `Amount`, `Value`, `TxFee`, and `Timestamp`. The unique identifier for each edge is `TxHash`, and the `Token` with a certain `Amount` indicates what and how many tokens are transferred, traded, minted, or burned in the transaction. The `Value` feature represents the number of dollars attached to the transaction, while `TxFee` is the fee charged for recording the transaction on the blockchain. Most importantly, each transaction has a `Timestamp` attribute that records the time of the transaction.

Moreover, `txn_id` represents a unique numerical identifier assigned to each edge, whereas `tokenid` is a unique numberical identifier of the transferred `Token`. The `source` and `target` are the head and tail of an edge, respectively, and `from` and `to` are corresponding node addresses of `source` and `target`. There are six values of `edgelabel`， where `10` represents the *Transfer* edge of *User-to-User*, `11` refers to the *Trade* edge of *User-to-User*, `12` is the *Mint* edge of *Null Address-to-User*, `13` means the *Burn* edge of *User-to-Null Address*, `00` represents the *Transfer* edge of *User-to_Contract* and `01` refers to the *Trade* edge of *User-to_Contract*.


#### metadata:
| indicator | value |
|-|-|
| total edges | 8,668,213 |
| edgelabel distribution | { 10 : 2,432,119 ; 11 : 1,920,546 ; 12 : 1,962,750 ; 13 : 578,549; 00 : 949,141; 01 : 823,466}
| dimensions of edge features | 6 |
| edge features | { TxHash, Token, Amount, Value, TxFee, Timestamp }


#### preview:

txn_id|source|target|tokenid|Timestamp|Amount|Value|TxFee|from|to|Token|Txhash|edgelabel
|-|-|-|-|-|-|-|-|-|-|-|-|-|
0|674618|501095|1170882|20220730055230|1|78.52|2.23|0x9463ea1dadf279e174e1075b49b8b7a13d1e7293|0x6e388502b891ca05eb52525338172f261c31b7d3|0xd07dc4262bcdbf85190c01c996b4c06a461d2430|0xb55b5b44aa556916ab6c8b38c40649c06c6363be5f0034cac678fd44e5f9b420|11
1|0|984132|1170882|20220730055230|14|0.0|0.98|0x0000000000000000000000000000000000000000|0xd8b75eb7bd778ac0b3f5ffad69bcc2e25bccac95|0xd07dc4262bcdbf85190c01c996b4c06a461d2430|0xa50ddcc6c3738761284a9e01427117781dd4810acc9140a3f6f6df6c6e00aeea|12
2|416892|364963|1170882|20220730055138|1|0.0|0.33|0x5b84e08b8883f400120da8a0099ba142641d1abb|0x4fffd4614ef28eb2618a27c5d88a5fd92c6d6580|0xd07dc4262bcdbf85190c01c996b4c06a461d2430|0xa218536b94379dcbc7ec14a298a09cfc366c30d5b8501021bd08698fc754bdf1|10

### NFTGraph-Tiny
We extract the top 20,000 active *Users* and all *Contracts* but without isolated nodes to create NFTGraph-Tiny, 
resulting in a significant reduction in size of the original graph. 
This is done with the consideration that some GNNs may not be able to handle large graphs 
in resource-limited settings, such as a few anomaly detection methods.


This folder contains the raw data of nodes and edges for NFTGraph-Tiny.

The file tinynft_nodes.csv comprises the nodes associated with NFTGraph-Tiny, while tinynft_edges.csv contains the corresponding edges.

The metadata of both nodes and edges aligns with that of NFTGraph-All. For more comprehensive information, kindly refer to the `NFTGraph-All` folder.

### Crawel
This folder contains the code for crawlering transaction details from the Etherscan website.

The crawler is designed to search for and download the transaction list based on token addresses. All tokens are listed in the `tokenlist.csv` file.


## 2. Anomaly Data
This folder comprises all the labels required for anomaly detection.

Specifically, the labels are provided in the format of `addresses` and are assigned to each node. If a node's address is present in the corresponding file, it is considered as an anomalous node.

## 3. OGB Graph
This folder encompasses the codes for handling and obtaining NFTGraph data in mainstream graph library formats, namely `ogb`, `pyg`, and `dgl`.

`pyg_dataset` includes codes for obtaining PyG (PyTorch Geometric) graph.

`dgl_dataset` contains codes for obtaining `dgl` graph.

`ogb_dataset` includes codes for obtaining `ogb` graph.

`official_ogb` contains the official code for the `ogb` library, which is referred to by the code in other directories.

`example_direct_use` are ogb graphs that can be directly used following the instructions on the offical website:
https://ogb.stanford.edu/docs/home/.

## 4. Networkx Graph
This folder contains the code that utilizes the `networkx` package to calculate various metrics on NFTGraph.

Additional statistics of three variants of NFTGraph.

| Metrics | NFTGraph-Tiny | NFTGraph-User | NFTGraph-All|
|-|-|-|-|
|#Nodes	|22,110	|1,161,854	|1,172,865 |
|#Edges|	347,243	| 6,893,964	|8,668,213|
|Density	|0.001421|	4.22E-06|	5.23E-06|
|Reciprocity|0|	0.1009|	0.08153|
|Assortativity|	-0.09156|	-0.1127|	-0.1177|      
|#Triangles|	1,551,432|	4,324,452|	6,883,560|
|Transitivity|	0.007592|	1.82E-05|	2.87E-05|
|Clustering Coefficient|0.1487|0.1747	|0.2128|
|Assortativity|	-0.09156|	-0.1127|	-0.1177| 
|Transitivity|	0.007592|	1.82E-05|	2.87E-05|
|#SCC| 	5,263|	792,965|	798,839|
|#WCC|	12|	5,806|	835|
|(#Nodes / #Edges) of Largest SCC |	(16,846 / 211,786)	|(365,368 / 1,689,063)	|(365,368 / 1,689,063)|
|(#Nodes / #Edges) of Largest WCC |(22,088 / 347,232)|	(1,148,823 / 2,994,341)|	(1,166,696 / 3,714,042)|
|AvgTxFee ($)|	57.3043	|8.5217	|9.6512|
|AvgAmount	|12.5258	|2.8931	|5.63E+41|
|AvgValue ($)|	1126.5053|	284.7976|	297.6778|


## 5. Benchmarks

This folder encompasses all the code for conducting benchmarks on NFTGraph. NFTGraph exhibits a wealth of features at node, edge, and subgraph levels. Therefore, three experiments have been conducted on NFTGraph, targeting different levels of the graph.

### Algorithms:
#### node classification (anomaly detection)
We select several commonly used anomaly detection models provided on Python Graph Outlier Detection (PyGOD), a benchmark tool for outlier detection. It is a free tool and available at https://github.com/pygod-team/pygod/.

#### link prediction

We select eight state-of-the-art graph neural networks on OGB leaderboards. Thanks to `dgl` package, we benchmark them on NFTGraph easily. The reference codes are listed as follows:

SEAL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/seal_ogbl

SEAL+NGNN: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ngnn_seal

SUREL: https://github.com/Graph-COM/SUREL

SIEG: https://github.com/anonymous20221001/SIEG_OGB

AGDN: https://github.com/skepsun/Adaptive-Graph-Diffusion-Networks

GraphSAGE: https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage

GAT: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-products/gat

GCN: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn


#### graph self-supervised learning
We select seven commonly used graph self-supervised learning algorithms and evaluate them on NFTGraph. Thanks for the open-sourced codes, we list them in the following:

ParetoGNN: https://github.com/jumxglhf/ParetoGNN

GraphMAE: https://github.com/THUDM/GraphMAE

DGI: https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgi

MVGRL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/mvgrl

CCA-SSG: https://github.com/hengruizhang98/

BGRL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/bgrl

GRACE: https://github.com/dmlc/dgl/tree/master/examples/pytorch/grace


### Results:
#### node classification (anomaly detection)

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

#### link prediction

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

#### graph self-supervised learning

Models|ACC|NMI|METIS|HMEANS|RANK|
|-|-|-|-|-|-|
ParetoGNN |0.8603 ±0.0001|0.0004 ±0.0000}|0.2615 ±0.0064|0.3741|2.67|
GraphMAE |0.8508 ±0.0001|0.0167 ±0.0000|0.1228 ±0.0056|0.3301|4|
DGI |0.8554 ±0.0001|0.0059 ±0.0000|0.1842 ±0.0299|0.3485|3.33|
MVGRL |OOM|OOM|OOM|OOM|OOM|
CCA-SSG |0.8558 ±0.0001|0.0052 ±0.0000|0.1508 ±0.0277|0.3373|3.67|
BGRL |0.9486 ±0.0002|0.3460 ±0.0000|0.2561 ±0.0069|0.5169|1.33|
GRACE |OOM|OOM|OOM|OOM|OOM|
