import pandas as pd
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import random
from torch_geometric.utils import to_undirected, remove_isolated_nodes, remove_self_loops, train_test_split_edges, negative_sampling
import multiprocessing
from itertools import product
import math
# our modified example
def custom_train_test_split_edges(data, val_ratio: float = 0.05, test_ratio: float = 0.1):
    r"""Splits the edges of a :class:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges.
    As such, it will replace the :obj:`edge_index` attribute with
    :obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
    :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
    :obj:`test_pos_edge_index` attributes.
    If :obj:`data` has edge features named :obj:`edge_attr`, then
    :obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` and
    :obj:`test_pos_edge_attr` will be added as well.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation edges.
            (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test edges.
            (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    original_edge_index = data.edge_index
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]

    # this section is custom
    # -----------------------
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    helper = data.train_pos_edge_index

    # if edge_attr is not None:
    #     out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
    #     data.edge_index, data.edge_attr = out
    # else:
    #     data.edge_index = to_undirected(data.train_pos_edge_index)

    data.train_pos_edge_index = helper

    if edge_attr is not None:
        data.train_pos_edge_attr = edge_attr[n_v + n_t:]
    # -----------------------

    data.edge_index = original_edge_index

    
    # generate negative edge list by randomly sampling the nodes!
    neg_edge_list = np.array(np.random.randint(low=0, high=num_nodes,
                                               size=(2*data.edge_index.shape[1],)). # left and right edge - 2x, to be safe:3.4
                             reshape((data.edge_index.shape[1],2)))

    a = np.min(neg_edge_list, axis=1)
    b = np.max(neg_edge_list, axis=1)

    neg_edge_list = np.vstack((a,b)).transpose()

    # filter for unique edges in the negative edge list

    # obtain the indexes of the first occuring objects
    # _, indices = np.unique(edges[:,[0,1]],return_index=True,axis=0)
    _, indices = np.unique(neg_edge_list[:,[0,1]],return_index=True,axis=0)

    neg_edge_list = neg_edge_list[indices]

    all_edges = np.concatenate((np.array(data.edge_index.t()),neg_edge_list), axis=0) # concat positive edges of graph and negative edges

    # obtain the indexes of unique objects
    _, indices = np.unique(all_edges[:, [0, 1]], return_index=True, axis=0)

    # sort indices

    indices = np.sort(indices)
    indices = indices[indices > data.edge_index.shape[1]] # remove the indices of the positive edges!
    neg_edge_list = torch.tensor(all_edges[indices])

    # sample edges according to percentage

    ind = torch.randperm(neg_edge_list.shape[0])

    data.val_neg_edge_index = neg_edge_list[ind[:n_v]].t()
    data.test_neg_edge_index = neg_edge_list[ind[n_v:n_v+n_t]].t()
    data.train_neg_edge_index = neg_edge_list[ind[n_v+n_t:n_v+n_t+data.train_pos_edge_index.shape[1]]].t()

    """
    #Original Sampling: allocates to much memory

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    ind = torch.randperm(neg_row.size(0))
    perm = ind[:n_v + n_t]
    perm_train = ind[n_v+n_t:n_v+n_t+data.train_pos_edge_index.shape[1]]
    neg_row_train, neg_col_train = neg_row[perm_train], neg_col[perm_train]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row_train , neg_col_train
    data.train_neg_edge_index = torch.stack([row, col], dim=0)
    """

    return data
