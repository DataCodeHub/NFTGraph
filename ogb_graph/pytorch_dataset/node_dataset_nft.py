import os
import os.path as osp
import torch
import random
import numpy as np
import pandas as pd

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_gz, extract_tar, extract_zip)
from torch_geometric.data.makedirs import makedirs
from torch_geometric.utils import to_undirected, remove_isolated_nodes, remove_self_loops

class NodeAllNFTGraph(InMemoryDataset):

    available_datasets = {'allnft':{'url':'https://'}}

    def __init__(self, root, name, transform=None, pre_transform=None,
                 use_node_attr: bool = True, use_edge_attr: bool = True):

        self.name = name#.lower()

        # check if dataset name is valid
        assert self.name in self.available_datasets.keys()

        self.url = self.available_datasets[self.name]['url']
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr

        super(NodeAllNFTGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        # get subfolders of each graph
        folder = osp.join(self.raw_dir, self.name)
        raw_file_names = []
        raw_file_names.add(osp.join(self.raw_dir, self.name, f'_nodes.csv'))
        raw_file_names.add(osp.join(self.raw_dir, self.name, f'_edges.csv'))

        print(raw_file_names)
        return [raw_file_names]

    @property
    def processed_file_names(self):
        return 'dataset.pt'

    def _download(self):
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        path = download_url(self.url, self.raw_dir, log=True)
        name = self.available_datasets[self.name]['folder']

        if name.endswith('.tar.gz'):
            extract_tar(path, self.raw_dir)
        elif name.endswith('.tar.xz'):
            extract_tar(path, self.raw_dir)
        elif name.endswith('.gz'):
            extract_gz(path, self.raw_dir)
        elif name.endswith('.zip'):
            extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):

        # reproducible results
        np.random.seed(123)
        torch.manual_seed(123)
        np.random.seed(123)

        # holds all graphs
        data_list = []

        # read csv files for nodes and edges


        df_nodes = pd.read_csv(osp.join(self.raw_dir, f'{self.name}_nodes_processed.csv'), sep=',')
        df_edges = pd.read_csv(osp.join(self.raw_dir, f'{self.name}_edges_processed.csv'), sep=',')

        # PyTorch Geometrics Data Class Object
        data = Data()

        # store keys of node and edge features
        data.node_attr_keys = ['OutAmount','OutValue','OutTransFee','InAmount','InValue','InTransFee']
        data.edge_attr_keys = ['tokenid','timestamp','transferedAmount','value','transactionFee','edgelabel']
        data.nodelabel = ['nodelabel']
        # Node feature matrix with shape [num_nodes, num_node_features]
        data.x = torch.from_numpy(np.array(df_nodes[data.node_attr_keys].to_numpy()))  
        data.y = torch.from_numpy(np.array(df_nodes[data.nodelabel].to_numpy()))  
        # Graph connectivity COO format with shape [2, num_edges]

        edge_index_source = np.array(df_edges[['source']])
        edge_index_sink = np.array(df_edges[['target']])
        edges = np.column_stack((edge_index_source, edge_index_sink))

        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_features = np.array(df_edges[data.edge_attr_keys].to_numpy())

        data.edge_attr = torch.from_numpy(np.array(edge_features))
        data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # convert the graph to an undirected graph
        # data.edge_index, data.edge_attr = to_undirected(edge_index=data.edge_index, edge_attr=data.edge_attr,
        #                                                 num_nodes=data.num_nodes, reduce="add")

        # remove self loops
        data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)

        # filter out isolated nodes
        data.edge_index, data.edge_attr, node_mask = remove_isolated_nodes(edge_index=data.edge_index,
                                                                            edge_attr=data.edge_attr,
                                                                            num_nodes=data.num_nodes)
        data.x = data.x[node_mask]
        data.y = data.y[node_mask]
        # append to other graphs
        data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)