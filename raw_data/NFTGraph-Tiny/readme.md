## NFTGraph-Tiny
We extract the top 20,000 active *Users* and all *Contracts* but without isolated nodes to create NFTGraph-Tiny, 
resulting in a significant reduction in size of the original graph. 
This is done with the consideration that some GNNs may not be able to handle large graphs 
in resource-limited settings, such as a few anomaly detection methods.


This folder contains the raw data of nodes and edges for NFTGraph-Tiny.

The file tinynft_nodes.csv comprises the nodes associated with NFTGraph-Tiny, while tinynft_edges.csv contains the corresponding edges.

The metadata of both nodes and edges aligns with that of NFTGraph-All. For more comprehensive information, kindly refer to the `NFTGraph-All` folder.
