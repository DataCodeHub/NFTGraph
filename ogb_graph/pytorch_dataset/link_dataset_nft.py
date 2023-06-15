import os
import os.path as osp
import torch
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_gz, extract_tar, extract_zip)
from torch_geometric.data.makedirs import makedirs
from torch_geometric.utils import to_undirected, remove_isolated_nodes, remove_self_loops
from torch_sparse import coalesce

from vessap_utils import *


class LinkNFTGraph(InMemoryDataset):

    available_datasets = {'nft':{'url':'https://'}}
    def __init__(self,
                root,
                name,
                splitting_strategy='spatial',
                number_of_workers = 8,
                val_ratio = 0.1,
                test_ratio = 0.1,
                use_edge_attr: bool = True,
                seed = 123,
                transform=None,
                pre_transform=None,
                ):
 
        self.name = name

        print("Available Datasets are:", self.available_datasets.keys())

        # check if dataset name is valid
        assert self.name in self.available_datasets.keys()

        self.url = self.available_datasets[self.name]['url']
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.use_edge_attr = use_edge_attr
        self.splitting_strategy = splitting_strategy
        self.number_of_workers = int(number_of_workers)
    
        super(LinkNFTGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
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
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        data_list = []

        df_nodes = pd.read_csv(osp.join(self.raw_dir, f'{self.name}_nodes.csv'),sep=',')
        df_edges = pd.read_csv(osp.join(self.raw_dir, f'{self.name}_edges.csv'),sep=',')
        data = Data()
        data.node_attr_keys = ['OutAmount','OutValue','OutTransFee','InAmount','InValue','InTransFee']
        # Node feature matrix with shape [num_nodes, num_node_features]
        data.x = torch.from_numpy(np.array(df_nodes[data.node_attr_keys].to_numpy()))      

        edges = np.column_stack((np.array(df_edges[['source']]),np.array(df_edges[['target']])))
        data.edge_attr_keys = ['tokenid','timestamp','transferedAmount','value','transactionFee']
           
        edge_features = np.array(df_edges[data.edge_attr_keys].to_numpy())

        data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.from_numpy(np.array(edge_features))

        # remove self loops
        data.edge_index , data.edge_attr = remove_self_loops(data.edge_index,data.edge_attr)

        # filter out isolated nodes
        data.edge_index, data.edge_attr , node_mask = remove_isolated_nodes(edge_index=data.edge_index,edge_attr = data.edge_attr,num_nodes=data.num_nodes)
        data.x = data.x[node_mask]


        if self.splitting_strategy == 'random': # only randomly sampled edges
            data = custom_train_test_split_edges(data, val_ratio=self.val_ratio, test_ratio = self.test_ratio)
        else:
            raise ValueError('Splitting strategy unknown!')

        if self.use_edge_attr == False:
            del data.train_pos_edge_attr
            del data.test_pos_edge_attr
            del data.val_pos_edge_attr
            del data.edge_attr
            del data.edge_attr_keys
            
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
