There are two files: `nodes.csv` and `edges.csv` representing the node and edge lists of NFTGraph-All.
The files have been uploaded to Google Drive due to their large size.

## nodes.csv:
download link: https://drive.google.com/file/d/1ksYE5alphvuxDn4CQVTLVh_HlBj2_Uia/view?usp=sharing

There are seven node features: `Address`, `OutAmount`, `OutValue`, `OutTxFee`, `InAmount`, `InValue`, and `InTxFee`. Aside from `Address`, which serves as the unique identifier for each node, the other features represent cumulants from transactions. Specifically, `OutAmount` and `InAmount` represent the cumulative tokens transferred from and to the current node, respectively. Similarly, `OutValue` and `InValue` indicate the cumulative transaction value when the current node is a source and target node separately and `OutTxFee` and `InTxFee` are the cumulative transaction fees paid by the current node as a sender and receiver, respectively.

In addition to these seven node features, it is worth noting that `node_id` corresponds to a unique numerical identifier assigned to each node. Furthermore, the `nodelabel` attribute serves as a binary indicator, where a value of 0 denotes a *User* node, and a value of 1 signifies a *Contract* node. For further elaboration and specific contextual meanings, please refer to the associated research paper.

### metadata:
| indicator | value |
|-|-|
| total nodes | 1,172,856 |
| nodelabel distribution | { 0 : 1,161,847 ; 1 : 11,009 }
| dimensions of node features | 7 |
| node features | { Address, OutAmount, OutValue, OutTxFee, InAmount, InValue, InTxFee } |


### preview:
| node_id | address | OutAmount | OutValue | OutTransFee | InAmount | InValue | InTransFee | nodelabel |
|-|-|-|-|-|-|-|-|-|
| 0 | 0x0000000000000000000000000000000000000000 | 7217724.0 | 712324959.5913919 | 21609773.15613218 | 2410795.0 | 20296148.734509576 | 4879685.167745654 | 0 |
|1|0x0000000000000000000000000000000000000001|0.0|0.0|0.0|140.0|0.0|264.18428919757656|0|
|2|0x0000000000000000000000000000000000000002|0.0|0.0|0.0|21.0|0.0|16.970000000000002|0|

## edges.csv:
download link: https://drive.google.com/file/d/1ONpaBTrW-UupUV31SVYFlMROiL3LwbJ1/view?usp=sharing

Each edge has six features: `TxHash`, `Token`, `Amount`, `Value`, `TxFee`, and `Timestamp`. The unique identifier for each edge is `TxHash`, and the `Token` with a certain `Amount` indicates what and how many tokens are transferred, traded, minted, or burned in the transaction. The `Value` feature represents the number of dollars attached to the transaction, while `TxFee` is the fee charged for recording the transaction on the blockchain. Most importantly, each transaction has a `Timestamp` attribute that records the time of the transaction.

Moreover, `txn_id` represents a unique numerical identifier assigned to each edge, whereas `tokenid` is a unique numberical identifier of the transferred `Token`. The `source` and `target` are the head and tail of an edge, respectively, and `from` and `to` are corresponding node addresses of `source` and `target`. There are six values of `edgelabel`， where `10` represents the *Transfer* edge of *User-to-User*, `11` refers to the *Trade* edge of *User-to-User*, `12` is the *Mint* edge of *Null Address-to-User*, `13` means the *Burn* edge of *User-to-Null Address*, `00` represents the *Transfer* edge of *User-to_Contract* and `01` refers to the *Trade* edge of *User-to_Contract*.


### metadata:
| indicator | value |
|-|-|
| total edges | 8,668,213 |
| edgelabel distribution | { 10 : 2,432,119 ; 11 : 1,920,546 ; 12 : 1,962,750 ; 13 : 578,549; 00 : 949,141; 01 : 823,466}
| dimensions of edge features | 6 |
| edge features | { TxHash, Token, Amount, Value, TxFee, Timestamp }


### preview

txn_id|source|target|tokenid|Timestamp|Amount|Value|TxFee|from|to|Token|Txhash|edgelabel
|-|-|-|-|-|-|-|-|-|-|-|-|-|
0|674618|501095|1170882|20220730055230|1|78.52|2.23|0x9463ea1dadf279e174e1075b49b8b7a13d1e7293|0x6e388502b891ca05eb52525338172f261c31b7d3|0xd07dc4262bcdbf85190c01c996b4c06a461d2430|0xb55b5b44aa556916ab6c8b38c40649c06c6363be5f0034cac678fd44e5f9b420|11
1|0|984132|1170882|20220730055230|14|0.0|0.98|0x0000000000000000000000000000000000000000|0xd8b75eb7bd778ac0b3f5ffad69bcc2e25bccac95|0xd07dc4262bcdbf85190c01c996b4c06a461d2430|0xa50ddcc6c3738761284a9e01427117781dd4810acc9140a3f6f6df6c6e00aeea|12
2|416892|364963|1170882|20220730055138|1|0.0|0.33|0x5b84e08b8883f400120da8a0099ba142641d1abb|0x4fffd4614ef28eb2618a27c5d88a5fd92c6d6580|0xd07dc4262bcdbf85190c01c996b4c06a461d2430|0xa218536b94379dcbc7ec14a298a09cfc366c30d5b8501021bd08698fc754bdf1|10
