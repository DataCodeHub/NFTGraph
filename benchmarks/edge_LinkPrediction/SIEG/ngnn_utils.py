import random
import sys

import numpy as np
import torch
from dgl.sampling import global_uniform_negative_sampling
from scipy.sparse.csgraph import shortest_path

import random
import sys

import numpy as np
import torch
from dgl.sampling import global_uniform_negative_sampling
from scipy.sparse.csgraph import shortest_path
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from dgl.data.utils import load_graphs, save_graphs, Subset
import dgl
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_dgl import read_graph_dgl, read_heterograph_dgl
from ogb.utils.torch_util import replace_numpy_with_torchtensor


from sklearn.metrics import roc_auc_score
import pandas as pd
import os

def k_hop_subgraph(src, dst, num_hops, g, sample_ratio=1.0, directed=False):
    # Extract the k-hop enclosing subgraph around link (src, dst) from g
    nodes = [src, dst]
    visited = set([src, dst])
    fringe = set([src, dst])
    max_nodes_per_hop = 20
    min_nodes_per_hop = 10
    for _ in range(num_hops):
        if not directed:
            _, fringe = g.out_edges(list(fringe))
            fringe = fringe.tolist()
        else:
            _, out_neighbors = g.out_edges(list(fringe))
            in_neighbors, _ = g.in_edges(list(fringe))
            fringe = in_neighbors.tolist() + out_neighbors.tolist()
        fringe = set(fringe) - visited
        visited = visited.union(fringe)

        if len(fringe) > min_nodes_per_hop:
            if len(fringe) > max_nodes_per_hop:
                fringe = random.sample(fringe, max_nodes_per_hop)            
            elif sample_ratio < 1.0:
                tmpfringe = random.sample(fringe, int(sample_ratio * len(fringe)))
                if len(tmpfringe) <= min_nodes_per_hop:
                    fringe = random.sample(fringe, min_nodes_per_hop) 
                else:
                    fringe = tmpfringe

        if len(fringe) == 0:
            break

        nodes = nodes + list(fringe)

    subg = g.subgraph(nodes, store_ids=True)

    return subg


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(
        adj_wo_dst, directed=False, unweighted=True, indices=src
    )
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(
        adj_wo_src, directed=False, unweighted=True, indices=dst - 1
    )
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = (
        torch.div(dist, 2, rounding_mode="floor"),
        dist % 2,
    )

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.0
    z[dst] = 1.0
    # shortest path may include inf values
    z[torch.isnan(z)] = 0.0
    # import pdb; pdb.set_trace()
    return z.to(torch.long)


def get_pos_neg_edges(split, split_edge, g, percent=100):
    pos_edge = split_edge[split]["edge"]
    # if split == "train":
    #     neg_edge = torch.stack(
    #         global_uniform_negative_sampling(
    #             g, num_samples=pos_edge.size(0), exclude_self_loops=True
    #         ),
    #         dim=1,
    #     )
    # else:
    neg_edge = split_edge[split]["edge_neg"]

    # sampling according to the percent param
    np.random.seed(123)
    # pos sampling
    num_pos = pos_edge.size(0)
    perm = np.random.permutation(num_pos)
    perm = perm[: int(percent / 100 * num_pos)]
    pos_edge = pos_edge[perm]
    # neg sampling
    if neg_edge.dim() > 2:  # [Np, Nn, 2]
        neg_edge = neg_edge[perm].view(-1, 2)
    else:
        np.random.seed(123)
        num_neg = neg_edge.size(0)
        perm = np.random.permutation(num_neg)
        perm = perm[: int(percent / 100 * num_neg)]
        neg_edge = neg_edge[perm]
    # import pdb; pdb.set_trace()
    return pos_edge, neg_edge  # ([2, Np], [2, Nn]) -> ([Np, 2], [Nn, 2])


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.runs = runs
        self.results = {
            "valid": [[] for _ in range(runs)],
            "test": [[] for _ in range(runs)],
        }

    def add_result(self, run, result, split="valid"):
        assert run >= 0 and run < len(self.results["valid"])
        assert split in ["valid", "test"]
        self.results[split][run].append(result)

    def print_statistics(self, run=None, f=sys.stdout):
        # if run is not None:
        #     result = torch.tensor(self.results["valid"][run])
        #     print(f"Run {run + 1:02d}:", file=f)
        #     print(f"Highest Valid: {result.max():.4f}", file=f)
        #     print(f"Highest Eval Point(Eval_epoch): {result.argmax().item()+1}", file=f)
        #     if not self.info.no_test:
        #         print(
        #             f'   Final Test Point[1](True Epoch): {self.results["test"][run][0][0]}',
        #             f'   Final Valid: {self.results["test"][run][0][1]}',
        #             f'   Final Test: {self.results["test"][run][0][2]}',
        #             sep="\n",
        #             file=f,
        #         )
        if (run+1) % 3 == 0 or ((run+1) == self.runs):
            # print(self.results["test"])
            best_result = torch.tensor(
                [test_res[0] for test_res in self.results["test"][:run+1]]
            )

            print(f"Runs 1-{run+1}:", file=f)
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.4f} ± {r.std():.4f}", file=f)
            if not self.info.no_test:
                r = best_result[:, 2]
                print(f"   Final Test: {r.mean():.4f} ± {r.std():.4f}", file=f)



class Evaluator:
    def __init__(self, name):
        self.name = name

        meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
        if not self.name in meta_info:
            print(self.name)
            error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
            error_mssg += 'Available datasets are as follows:\n'
            error_mssg += '\n'.join(meta_info.keys())
            raise ValueError(error_mssg)

        self.eval_metric = meta_info[self.name]['eval metric']

        if 'hits@' in self.eval_metric:
            ### Hits@K

            self.K = int(self.eval_metric.split('@')[1])


    def _parse_and_check_input(self, input_dict):
        if 'hits@' in self.eval_metric or 'rocauc' == self.eval_metric:
            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')

            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            '''
                y_pred_pos: numpy ndarray or torch tensor of shape (num_edges, )
                y_pred_neg: numpy ndarray or torch tensor of shape (num_edges, )
            '''

            # convert y_pred_pos, y_pred_neg into either torch tensor or both numpy array
            # type_info stores information whether torch or numpy is used

            type_info = None

            # check the raw tyep of y_pred_pos
            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

            # check the raw type of y_pred_neg
            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

            # if either y_pred_pos or y_pred_neg is torch tensor, use torch tensor
            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                # converting to torch.Tensor to numpy on cpu
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                # put both y_pred_pos and y_pred_neg on the same device
                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'

            else:
                # both y_pred_pos and y_pred_neg are numpy ndarray

                type_info = 'numpy'


            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

            if not y_pred_neg.ndim == 1:
                raise RuntimeError('y_pred_neg must to 1-dim arrray, {}-dim array given'.format(y_pred_neg.ndim))

            return y_pred_pos, y_pred_neg, type_info

        elif 'mrr' == self.eval_metric:

            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')

            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            '''
                y_pred_pos: numpy ndarray or torch tensor of shape (num_edges, )
                y_pred_neg: numpy ndarray or torch tensor of shape (num_edges, num_nodes_negative)
            '''

            # convert y_pred_pos, y_pred_neg into either torch tensor or both numpy array
            # type_info stores information whether torch or numpy is used

            type_info = None

            # check the raw tyep of y_pred_pos
            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

            # check the raw type of y_pred_neg
            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

            # if either y_pred_pos or y_pred_neg is torch tensor, use torch tensor
            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                # converting to torch.Tensor to numpy on cpu
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                # put both y_pred_pos and y_pred_neg on the same device
                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'


            else:
                # both y_pred_pos and y_pred_neg are numpy ndarray

                type_info = 'numpy'


            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

            if not y_pred_neg.ndim == 2:
                raise RuntimeError('y_pred_neg must to 2-dim arrray, {}-dim array given'.format(y_pred_neg.ndim))

            return y_pred_pos, y_pred_neg, type_info

        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))


    def eval(self, input_dict):

        if 'hits@' in self.eval_metric:
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_hits(y_pred_pos, y_pred_neg, type_info)
        elif self.eval_metric == 'mrr':
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_mrr(y_pred_pos, y_pred_neg, type_info)
        elif self.eval_metric == 'rocauc':
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_pred_pos, y_pred_neg, type_info)
        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
        if 'hits@' in self.eval_metric:
            desc += '{\'y_pred_pos\': y_pred_pos, \'y_pred_neg\': y_pred_neg}\n'
            desc += '- y_pred_pos: numpy ndarray or torch tensor of shape (num_edges, ). Torch tensor on GPU is recommended for efficiency.\n'
            desc += '- y_pred_neg: numpy ndarray or torch tensor of shape (num_edges, ). Torch tensor on GPU is recommended for efficiency.\n'
            desc += 'y_pred_pos is the predicted scores for positive edges.\n'
            desc += 'y_pred_neg is the predicted scores for negative edges.\n'
            desc += 'Note: As the evaluation metric is ranking-based, the predicted scores need to be different for different edges.'
        elif self.eval_metric == 'mrr':
            desc += '{\'y_pred_pos\': y_pred_pos, \'y_pred_neg\': y_pred_neg}\n'
            desc += '- y_pred_pos: numpy ndarray or torch tensor of shape (num_edges, ). Torch tensor on GPU is recommended for efficiency.\n'
            desc += '- y_pred_neg: numpy ndarray or torch tensor of shape (num_edges, num_nodes_neg). Torch tensor on GPU is recommended for efficiency.\n'
            desc += 'y_pred_pos is the predicted scores for positive edges.\n'
            desc += 'y_pred_neg is the predicted scores for negative edges. It needs to be a 2d matrix.\n'
            desc += 'y_pred_pos[i] is ranked among y_pred_neg[i].\n'
            desc += 'Note: As the evaluation metric is ranking-based, the predicted scores need to be different for different edges.'
        elif self.eval_metric == 'rocauc':
            desc += '{\'y_pred_pos\': y_pred_pos, \'y_pred_neg\': y_pred_neg}\n'
            desc += '- y_pred_pos: numpy ndarray or torch tensor of shape (num_edges, ).\n'
            desc += '- y_pred_neg: numpy ndarray or torch tensor of shape (num_edges, ).\n'
            desc += 'y_pred_pos is the predicted scores for positive edges.\n'
            desc += 'y_pred_neg is the predicted scores for negative edges.\n'
        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
        if 'hits@' in self.eval_metric:
            desc += '{' + 'hits@{}\': hits@{}'.format(self.K, self.K) + '}\n'
            desc += '- hits@{} (float): Hits@{} score\n'.format(self.K, self.K)
        elif self.eval_metric == 'mrr':
            desc += '{' + '\'hits@1_list\': hits@1_list, \'hits@3_list\': hits@3_list, \n\'hits@10_list\': hits@10_list, \'mrr_list\': mrr_list}\n'
            desc += '- mrr_list (list of float): list of scores for calculating MRR \n'
            desc += '- hits@1_list (list of float): list of scores for calculating Hits@1 \n'
            desc += '- hits@3_list (list of float): list of scores to calculating Hits@3\n'
            desc += '- hits@10_list (list of float): list of scores to calculating Hits@10\n'
            desc += 'Note: i-th element corresponds to the prediction score for the i-th edge.\n'
            desc += 'Note: To obtain the final score, you need to concatenate the lists of scores and take average over the concatenated list.'
        elif self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc\n'
            desc += '- rocauc (float): ROC-AUC score\n'
        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))

        return desc


    def _eval_hits(self, y_pred_pos, y_pred_neg, type_info):
        '''
            compute Hits@K
            For each positive target node, the negative target nodes are the same.

            y_pred_neg is an array.
            rank y_pred_pos[i] against y_pred_neg for each i
        '''

        if len(y_pred_neg) < self.K:
            return {'hits@{}'.format(self.K): 1.}

        if type_info == 'torch':
            kth_score_in_negative_edges = torch.topk(y_pred_neg, self.K)[0][-1]
            hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

        # type_info is numpy
        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-self.K]
            hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

        return {'hits@{}'.format(self.K): hitsK}

    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        '''
            compute mrr
            y_pred_neg is an array with shape (batch size, num_entities_neg).
            y_pred_pos is an array with shape (batch size, )
        '''


        if type_info == 'torch':
            # calculate ranks
            y_pred_pos = y_pred_pos.view(-1, 1)
            # optimistic rank: "how many negatives have at least the positive score?"
            # ~> the positive is ranked first among those with equal score
            optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
            # pessimistic rank: "how many negatives have a larger score than the positive?"
            # ~> the positive is ranked last among those with equal score
            pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            hits1_list = (ranking_list <= 1).to(torch.float)
            hits3_list = (ranking_list <= 3).to(torch.float)
            hits10_list = (ranking_list <= 10).to(torch.float)
            mrr_list = 1./ranking_list.to(torch.float)

            return {'hits@1_list': hits1_list,
                     'hits@3_list': hits3_list,
                     'hits@10_list': hits10_list,
                     'mrr_list': mrr_list}

        else:
            y_pred_pos = y_pred_pos.reshape(-1, 1)
            optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
            pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            hits1_list = (ranking_list <= 1).astype(np.float32)
            hits3_list = (ranking_list <= 3).astype(np.float32)
            hits10_list = (ranking_list <= 10).astype(np.float32)
            mrr_list = 1./ranking_list.astype(np.float32)

            return {'hits@1_list': hits1_list,
                     'hits@3_list': hits3_list,
                     'hits@10_list': hits10_list,
                     'mrr_list': mrr_list}

    def _eval_rocauc(self, y_pred_pos, y_pred_neg, type_info):
        '''
            compute rocauc
        '''
        if type_info == 'torch':
            y_pred_pos_numpy = y_pred_pos.cpu().numpy()
            y_pred_neg_numpy = y_pred_neg.cpu().numpy()
        else:
            y_pred_pos_numpy = y_pred_pos
            y_pred_neg_numpy = y_pred_neg
        
        y_true = np.concatenate([np.ones(len(y_pred_pos_numpy)), np.zeros(len(y_pred_neg_numpy))]).astype(np.int32)
        y_pred = np.concatenate([y_pred_pos_numpy, y_pred_neg_numpy])

        rocauc = roc_auc_score(y_true, y_pred)

        return {'rocauc': rocauc}

class DglLinkPropPredDataset(object):
    '''Adapted from https://docs.dgl.ai/en/latest/_modules/dgl/data/chem/csv_dataset.html#CSVDataset'''
    def __init__(self, name, root = 'dataset', meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''         

        self.name = name ## original name, e.g., ogbl-ppa

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 
            
            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_dgl')):
                self.dir_name = self.dir_name + '_dgl'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user. 
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., ppassoc

        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(DglLinkPropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if osp.exists(pre_processed_file_path):
            self.graph, _ = load_graphs(pre_processed_file_path)

        else:
            ### check if the downloaded file exists
            if self.binary:
                # npz format
                has_necessary_file_simple = osp.exists(osp.join(self.root, 'raw', 'data.npz')) and (not self.is_hetero)
                has_necessary_file_hetero = osp.exists(osp.join(self.root, 'raw', 'edge_index_dict.npz')) and self.is_hetero
            else:
                # csv file
                has_necessary_file_simple = osp.exists(osp.join(self.root, 'raw', 'edge.csv.gz')) and (not self.is_hetero)
                has_necessary_file_hetero = osp.exists(osp.join(self.root, 'raw', 'triplet-type-list.csv.gz')) and self.is_hetero

            has_necessary_file = has_necessary_file_simple or has_necessary_file_hetero
            if not has_necessary_file:
                url = self.meta_info['url']
                if decide_download(url):
                    path = download_url(url, self.original_root)
                    extract_zip(path, self.original_root)
                    os.unlink(path)
                    # delete folder if there exists
                    try:
                        shutil.rmtree(self.root)
                    except:
                        pass
                    shutil.move(osp.join(self.original_root, self.download_name), self.root)
                else:
                    print('Stop download.')
                    exit(-1)

            raw_dir = osp.join(self.root, 'raw')

            add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

            ### pre-process and save
            if self.meta_info['additional node files'] == 'None':
                additional_node_files = []
            else:
                additional_node_files = self.meta_info['additional node files'].split(',')

            if self.meta_info['additional edge files'] == 'None':
                additional_edge_files = []
            else:
                additional_edge_files = self.meta_info['additional edge files'].split(',')


            if self.is_hetero:
                graph = read_heterograph_dgl(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]
            else:
                graph = read_graph_dgl(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]

            print('Saving...')
            save_graphs(pre_processed_file_path, graph, {})

            self.graph, _ = load_graphs(pre_processed_file_path)

    def get_edge_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']
            
        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train = replace_numpy_with_torchtensor(torch.load(osp.join(path, 'train.pt')))
        valid = replace_numpy_with_torchtensor(torch.load(osp.join(path, 'valid.pt')))
        test = replace_numpy_with_torchtensor(torch.load(osp.join(path, 'test.pt')))

        return {'train': train, 'valid': valid, 'test': test}

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph[0]

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))