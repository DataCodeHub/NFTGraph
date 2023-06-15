# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import os, sys
import os.path as osp
import shutil
import copy as cp
from tqdm import tqdm
from functools import partial
import psutil
import pdb

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import scipy.sparse as ssp
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader as PygDataLoader
from torch_geometric.utils import to_networkx, to_undirected

# from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from torch_geometric.datasets import Planetoid
from dataset import SEALDynamicDataset, SEALIterableDataset, SEALDynamicDataset
from preprocess import preprocess, preprocess_full
from graphormer.collator import collator
from utils import *
from models import *
from timer_guard import TimerGuard
import datetime
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', 
                    stream=sys.stdout,
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

# ngnn code
import argparse
import datetime
import os
import sys
import time

import dgl
import torch
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
# from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset
from tqdm import tqdm

import ngnn_models
import ngnn_utils
from ngnn_utils import *

# ngnn dataset
class SEALOGBLDataset(Dataset):
    def __init__(
        self,
        # data_pyg,
        preprocess_fn,
        root,
        graph,
        split_edge,
        percent=100,
        split="train",
        ratio_per_hop=1.0,
        directed=False,
        dynamic=True,
    ) -> None:
        super().__init__()
        # self.data_pyg = data_pyg
        self.preprocess_fn = preprocess_fn
        self.root = root
        self.graph = graph
        self.split = split
        self.split_edge = split_edge
        self.percent = percent
        self.ratio_per_hop = ratio_per_hop
        self.directed = directed
        self.dynamic = dynamic
        # import pdb; pdb.set_trace()
        if "weights" in self.graph.edata:
            self.edge_weights = self.graph.edata["weights"]
        else:
            self.edge_weights = None
        if "feat" in self.graph.ndata:
            self.node_features = self.graph.ndata["feat"]
        else:
            self.node_features = None

        pos_edge, neg_edge = ngnn_utils.get_pos_neg_edges(
            self.split, self.split_edge, self.graph, self.percent
        )
        self.links = torch.cat([pos_edge, neg_edge], 0)  # [Np + Nn, 2] [1215518, 2]
        self.labels = np.array([1] * len(pos_edge) + [0] * len(neg_edge))  # [1215518]

        if not self.dynamic:
            self.g_list, tensor_dict = self.load_cached()
            self.labels = tensor_dict["y"]
        self.degree = None
        # # compute degree from dataset_pyg
        # if 'Graphormer' in args.model:
        #     if 'edge_weight' in data_pyg:
        #         edge_weight = data_pyg.edge_weight.view(-1)
        #     else:
        #         edge_weight = torch.ones(data_pyg.edge_index.size(1), dtype=int)
        #     import scipy.sparse as ssp
        #     A = ssp.csr_matrix(
        #         (edge_weight, (data_pyg.edge_index[0], data_pyg.edge_index[1])), 
        #         shape=(data_pyg.num_nodes, data_pyg.num_nodes))
        #     if directed:
        #         A_undirected = ssp.csr_matrix((np.concatenate([edge_weight, edge_weight]), (np.concatenate([data_pyg.edge_index[0], data_pyg.edge_index[1]]), np.concatenate([data_pyg.edge_index[1], data_pyg.edge_index[0]]))), shape=(data_pyg.num_nodes, data_pyg.num_nodes))
        #         degree_undirected = A_undirected.sum(axis=0).flatten().tolist()[0]
        #         degree_in = A.sum(axis=0).flatten().tolist()[0]
        #         degree_out = A.sum(axis=1).flatten().tolist()[0]
        #         self.degree = torch.Tensor([degree_undirected, degree_in, degree_out]).long()
        #     else:
        #         degree_undirected = A.sum(axis=0).flatten().tolist()[0]
        #         self.degree = torch.Tensor([degree_undirected]).long()



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.dynamic:
            g, y = self.g_list[idx], self.labels[idx]
            x = None if "x" not in g.ndata else g.ndata["x"]
            w = None if "w" not in g.edata else g.eata["w"]
            return g, g.ndata["z"], x, w, y

        src, dst = self.links[idx][0].item(), self.links[idx][1].item()
        y = self.labels[idx]  # 1
        subg = ngnn_utils.k_hop_subgraph(
            src, dst, 1, self.graph, self.ratio_per_hop, self.directed
        )

        # Remove the link between src and dst.
        direct_links = [[], []]
        for s, t in [(0, 1), (1, 0)]:
            if subg.has_edges_between(s, t):
                direct_links[0].append(s)
                direct_links[1].append(t)
        if len(direct_links[0]):
            subg.remove_edges(subg.edge_ids(*direct_links))

        NIDs, EIDs = subg.ndata[dgl.NID], subg.edata[dgl.EID]  # [32] [72]

        z = ngnn_utils.drnl_node_labeling(subg.adj(scipy_fmt="csr"), 0, 1)  # [32]
        edge_weights = (
            self.edge_weights[EIDs] if self.edge_weights is not None else None
        )
        x = self.node_features[NIDs] if self.node_features is not None else None  # [32, 128]

        subg_aug = subg.add_self_loop()
        if edge_weights is not None:  # False
            edge_weights = torch.cat(
                [
                    edge_weights,
                    torch.ones(subg_aug.num_edges() - subg.num_edges()),
                ]
            )

        # compute structure from pyg data
        if 'Graphormer' in args.model:
            subg.x = x
            subg.z = z
            subg.node_id = NIDs
            subg.edge_index = torch.cat([subg.edges()[0].unsqueeze(0), subg.edges()[1].unsqueeze(0)], 0)
            if self.preprocess_fn is not None:
                self.preprocess_fn(subg, directed=self.directed, degree=self.degree)

        return subg_aug, z, x, edge_weights, y, subg

    @property
    def cached_name(self):
        return f"SEAL_{self.split}_{self.percent}%.pt"

    def process(self):
        g_list, labels = [], []
        self.dynamic = True
        for i in tqdm(range(len(self))):
            g, z, x, weights, y = self[i]
            g.ndata["z"] = z
            if x is not None:
                g.ndata["x"] = x
            if weights is not None:
                g.edata["w"] = weights
            g_list.append(g)
            labels.append(y)
        self.dynamic = False
        return g_list, {"y": torch.tensor(labels)}

    def load_cached(self):
        path = os.path.join(self.root, self.cached_name)
        if os.path.exists(path):
            return load_graphs(path)

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        g_list, labels = self.process()
        save_graphs(path, g_list, labels)
        return g_list, labels


def train(num_datas):
    model.train()

    y_pred, y_true = torch.zeros([num_datas]), torch.zeros([num_datas])
    start = 0
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for data in pbar:
        g, z, x, edge_weights, y = [
            item.to(device) if item is not None else None for item in data
        ]
            # g.to(device)没法把这些pairwise结构属性to(device)，只能手动一下
        if 'Graphormer' in args.model:
            g.attn_bias = g.attn_bias.to(device)
            g.edge_index = g.edge_index.to(device)

            try:
                g.x = g.x.to(device)
            except:
                pass
            
            g.z = g.z.to(device)
            if args.use_len_spd:
                g.len_shortest_path = g.len_shortest_path.to(device)
            if args.use_num_spd:
                g.num_shortest_path = g.num_shortest_path.to(device)
            if args.use_cnb_jac:
                g.undir_jac = g.undir_jac.to(device)
            if args.use_cnb_aa:
                g.undir_aa = g.undir_aa.to(device)
            if args.use_cnb_ra:
                g.undir_ra = g.undir_ra.to(device)
            if args.use_degree:
                g.undir_degree = g.undir_degree.to(device)
                if directed:
                    g.in_degree = g.in_degree.to(device)
                    g.out_degree = g.out_degree.to(device)

        num_datas_in_batch = y.numel()  # sieg
        optimizer.zero_grad()
        logits = model(g, z, x, edge_weight=edge_weights)
        loss = BCEWithLogitsLoss()(logits.view(-1), y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * g.batch_size
        # sieg
        end = min(start+num_datas_in_batch, num_datas)
        y_pred[start:end] = logits.view(-1).cpu().detach()
        y_true[start:end] = y.view(-1).cpu().to(torch.float)
        start = end
    
    #result['Confuse'] = confusion_matrix(y_true, y_pred)
    #result['ACC'] = accuracy_score(y_true, y_pred)
    #result['Precision'] = precision_score(y_true, y_pred)
    #result['Recall'] = recall_score(y_true, y_pred)
    #result['F1'] = f1_score(y_true, y_pred)
    result = {}
    result['AUC'] = roc_auc_score(y_true, y_pred)
    return total_loss / len(train_dataset), result

@torch.no_grad()
def test(dataloader, num_datas):
    model.eval()
    y_pred, y_true, pos_y_pred, neg_y_pred = test_model(model, dataloader, num_datas)
    
    if dataset.eval_metric.startswith("hits@"):
        results = evaluate_hits(pos_y_pred, neg_y_pred, hits_K)
    elif dataset.eval_metric == "mrr":
        results = evaluate_mrr(pos_y_pred, neg_y_pred)
    elif dataset.eval_metric == "rocauc":
        results = evaluate_rocauc(pos_y_pred, neg_y_pred)
    
    results['rocauc'] = roc_auc_score(y_true, y_pred)
    return results


def test_model(model, loader, num_datas):
    model.eval()
    y_pred, y_true = torch.zeros([num_datas]), torch.zeros([num_datas])
    start = 0
    for data in tqdm(loader, ncols=70):

        g, z, x, edge_weights, y = [
            item.to(device) if item is not None else None for item in data
        ]
            # import pdb; pdb.set_trace()
            # g.to(device)没法把这些pairwise结构属性to(device)，只能手动一下
        if 'Graphormer' in args.model:
            g.attn_bias = g.attn_bias.to(device)
            g.edge_index = g.edge_index.to(device)
            # g.x = g.x.to(device)
            g.z = g.z.to(device)
            if args.use_len_spd:
                g.len_shortest_path = g.len_shortest_path.to(device)
            if args.use_num_spd:
                g.num_shortest_path = g.num_shortest_path.to(device)
            if args.use_cnb_jac:
                g.undir_jac = g.undir_jac.to(device)
            if args.use_cnb_aa:
                g.undir_aa = g.undir_aa.to(device)
            if args.use_cnb_ra:
                g.undir_ra = g.undir_ra.to(device)
            if args.use_degree:
                g.undir_degree = g.undir_degree.to(device)
                if directed:
                    g.in_degree = g.in_degree.to(device)
                    g.out_degree = g.out_degree.to(device)

        num_datas_in_batch = y.numel()  # sieg
        logits = model(g, z, x, edge_weight=edge_weights)
        # sieg
        end = min(start+num_datas_in_batch, num_datas)
        y_pred[start:end] = logits.view(-1).cpu()
        y_true[start:end] = y.view(-1).cpu().to(torch.float)
        start = end
    pos_test_pred = y_pred[y_true==1]
    neg_test_pred = y_pred[y_true==0]
    return y_pred, y_true, pos_test_pred, neg_test_pred


def eval_model(**kwargs):
    eval_metric = kwargs["eval_metric"]
    if eval_metric == 'hits':
        pos_val_pred = kwargs["pos_val_pred"]
        neg_val_pred = kwargs["neg_val_pred"]
        pos_test_pred = kwargs["pos_test_pred"]
        neg_test_pred = kwargs["neg_test_pred"]
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif eval_metric == 'mrr':
        pos_val_pred = kwargs["pos_val_pred"]
        neg_val_pred = kwargs["neg_val_pred"]
        pos_test_pred = kwargs["pos_test_pred"]
        neg_test_pred = kwargs["neg_test_pred"]
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif eval_metric == 'auc':
        val_pred = kwargs["val_pred"]
        val_true = kwargs["val_true"]
        test_pred = kwargs["test_pred"]
        test_true = kwargs["test_true"]
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results

@torch.no_grad()
def test0(eval_metric):
    model.eval()

    val_pred, val_true, pos_val_pred, neg_val_pred = test_model(model, val_loader, len(val_dataset))

    test_pred, test_true, pos_test_pred, neg_test_pred = val_pred, val_true, pos_val_pred, neg_val_pred #test_model(model, test_loader, len(test_dataset))

    result = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                      val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric=eval_metric)
    if eval_metric != 'auc':
        result_auc = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                          val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric='auc')
        for key in result_auc.keys():
            result[key] = result_auc[key]
    return result

@torch.no_grad()
def final_test(eval_metric):
    model.eval()

    val_pred, val_true, pos_val_pred, neg_val_pred = test_model(model, final_val_loader, len(final_val_dataset))

    test_pred, test_true, pos_test_pred, neg_test_pred = test_model(model, final_test_loader, len(final_test_dataset))

    result = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                      val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric=eval_metric)
    if eval_metric != 'auc':
        result_auc = eval_model(pos_val_pred=pos_val_pred, neg_val_pred=neg_val_pred, pos_test_pred=pos_test_pred, neg_test_pred=neg_test_pred,
                          val_pred=val_pred, val_true=val_true, test_pred=test_pred, test_true=test_true, eval_metric='auc')
        for key in result_auc.keys():
            result[key] = result_auc[key]
    return result


def evaluate_hits(y_pred_pos, y_pred_neg, hits_K):
    results = {}
    hits_K = map(
        lambda x: (int(x.split("@")[1]) if isinstance(x, str) else x), hits_K
    )
    for K in hits_K:
        evaluator.K = K
        hits = evaluator.eval(
            {
                "y_pred_pos": y_pred_pos,
                "y_pred_neg": y_pred_neg,
            }
        )[f"hits@{K}"]

        results[f"hits@{K}"] = hits

    return results


def evaluate_mrr(y_pred_pos, y_pred_neg):
    y_pred_neg = y_pred_neg.view(y_pred_pos.shape[0], -1)
    results = {}
    mrr = (
        evaluator.eval(
            {
                "y_pred_pos": y_pred_pos,
                "y_pred_neg": y_pred_neg,
            }
        )["mrr_list"]
        .mean()
        .item()
    )

    results["mrr"] = mrr

    return results

def evaluate_rocauc(y_pred_pos, y_pred_neg):
    results = {}
    rocauc = evaluator.eval(
        {
            "y_pred_pos": y_pred_pos,
            "y_pred_neg": y_pred_neg,
        }
    )["rocauc"]

    results["rocauc"] = rocauc

    return results



def print_log(*x, sep="\n", end="\n", mode="a"):
    print(*x, sep=sep, end=end)
    with open(log_file, mode=mode) as f:
        print(*x, sep=sep, end=end, file=f)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Data settings
parser = argparse.ArgumentParser(description='OGBL (SEAL)')
parser.add_argument('--cmd_time', type=str, default='ignore_time')
parser.add_argument('--root', type=str, default='dataset',
                    help="root of dataset")
parser.add_argument('--dataset', type=str, default='ogbl-nft_random_no_edge_attr')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")
# GNN settings
parser.add_argument('--model', type=str, default='DGCNN')
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)

# Subgraph extraction settings
parser.add_argument('--sample_type', type=int, default=0)
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='drnl', 
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', action='store_true',
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_feature_GT', action='store_true',
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN")
parser.add_argument('--use_rpe', action='store_true', help="whether to use RPE as GNN input")
parser.add_argument('--replacement', action='store_true', help="whether to enable replacement sampleing in random walk")
parser.add_argument('--trackback', action='store_true', help="whether to enabale trackback path searching in random walk")
parser.add_argument('--num_walk', type=int, default=200, help='total number of random walks')
parser.add_argument('--num_step', type=int, default=4, help='total steps of random walk')
parser.add_argument('--rpe_hidden_dim', type=int, default=16, help='dimension of RPE embedding')
parser.add_argument('--gravity_type', type=int, default=0)
parser.add_argument('--readout_type', type=int, default=0)
# Training settings
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--train_percent', type=float, default=2)
parser.add_argument('--val_percent', type=float, default=1)
parser.add_argument('--test_percent', type=float, default=1)
parser.add_argument('--final_val_percent', type=float, default=100)
parser.add_argument('--final_test_percent', type=float, default=100)
parser.add_argument('--dynamic_train', action='store_true', 
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--dynamic_val', action='store_true')
parser.add_argument('--dynamic_test', action='store_true')
parser.add_argument('--slice_type', type=int, default=0,
                    help="type of saving sampled subgraph in disk")
parser.add_argument('--num_workers', type=int, default=16, 
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--train_node_embedding', action='store_true',
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--dont_z_emb_agg', action='store_true')
parser.add_argument('--pretrained_node_embedding', type=str, default=None, 
                    help="load pretrained node embeddings as additional node features")
# Testing settings
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix', type=str, default='', 
                    help="an appendix to the data directory")
parser.add_argument('--save_appendix', type=str, default='', 
                    help="an appendix to the save directory")
parser.add_argument('--keep_old', action='store_true', 
                    help="do not overwrite old files in the save directory")
parser.add_argument('--continue_from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--only_test', action='store_true', 
                    help="only test without training")
parser.add_argument('--test_multiple_models', type=str, nargs='+', default=[], 
                    help="test multiple models together")
parser.add_argument('--use_heuristic', type=str, default=None,
                    help="test a link prediction heuristic (CN or AA)")
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--use_len_spd', action='store_true', default=False)
parser.add_argument('--use_num_spd', action='store_true', default=False)
parser.add_argument('--use_cnb_jac', action='store_true', default=False)
parser.add_argument('--use_cnb_aa', action='store_true', default=False)
parser.add_argument('--use_cnb_ra', action='store_true', default=False)
parser.add_argument('--use_degree', action='store_true', default=False)
parser.add_argument('--grpe_cross', action='store_true', default=False)
parser.add_argument('--use_ignn', action='store_true', default=False)
parser.add_argument('--mul_bias', action='store_true', default=False,
                    help="add bias to attention if true else multiple")
parser.add_argument('--max_z', type=int, default=1000)  # set a large max_z so that every z has embeddings to look up

# ngnn_args
parser.add_argument('--ngnn_code', action='store_true', default=False)
parser.add_argument('--use_full_graphormer', action='store_true', default=False)

parser.add_argument(
    "--ngnn_type",
    type=str,
    default="none",
    choices=["none", "input", "hidden", "output", "all"],
    help="You can set this value from 'none', 'input', 'hidden' or 'all' " \
            "to apply NGNN to different GNN layers.",
)
parser.add_argument(
    "--num_ngnn_layers", type=int, default=2, choices=[1, 2]
)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument(
    "--test_topk",
    type=int,
    default=1,
    help="select best k models for full validation/test each run.",
)
parser.add_argument(
    "--eval_hits_K",
    type=int,
    nargs="*",
    default=[10],
    help="hits@K for each eval step; " \
            "only available for datasets with hits@xx as the eval metric",
)
parser.add_argument('--eval_metric', type=str, default='mrr')
parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
parser.add_argument('--nthread', type=int, default=16, help='number of thread')

args = parser.parse_args()

args.device = 3
args.hidden_channels= 32
args.runs = 10
args.epochs = 15
args.lr = 1e-03
args.batch_size = 64
args.num_workers = 24 
args.train_percent = 100
args.val_percent = 10
args.final_val_percent = 100
args.test_percent = 100
args.directed = True
args.eval_steps = 3

args.use_len_spd = True
args.use_num_spd = True
args.use_cnb_jac = True

args.no_test = False 
args.model = 'NGNNDGCNNGraphormer'
args.ngnn_type = 'none'

# args.num_ngnn_layers = 2
args.use_feature = False
args.use_edge_weight = False
args.dynamic_train = True
args.dynamic_val = True
args.dynamic_test = True
args.ratio_per_hop = 0.1
args.nthread = 16
args.use_heuristic = False
args.eval_metric == 'mrr'

if args.nthread > 0:
    torch.set_num_threads(args.nthread)


if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = None if args.max_nodes_per_hop < 0 else args.max_nodes_per_hop
if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.data_appendix == '':
    args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
    if args.use_valedges_as_input:
        args.data_appendix += '_uvai'


args.res_dir = os.path.join('./results/{}{}'.format(args.dataset, args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

log_file = os.path.join(args.res_dir, 'log.txt')
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
with open(log_file, 'a') as f:
    f.write('\n' + cmd_input)

directed = False

if args.dataset.startswith('ogbl'):
    evaluator = Evaluator(name=args.dataset)


loggers = {
    "mrr": Logger(args.runs, args),
    "rocauc": Logger(args.runs,args)
}


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:0')
device = 'cpu' if args.device == -1 or not torch.cuda.is_available() else f'cuda:{args.device}'
device = torch.device(device)

print_log(f"{cmd_input}")
print_log(f"{args}")

preprocess_func = preprocess_full if args.use_full_graphormer else preprocess
preprocess_fn = partial(preprocess_func,
                        grpe_cross=args.grpe_cross,
                        use_len_spd=args.use_len_spd,
                        use_num_spd=args.use_num_spd,
                        use_cnb_jac=args.use_cnb_jac,
                        use_cnb_aa=args.use_cnb_aa,
                        use_cnb_ra=args.use_cnb_ra,
                        use_degree=args.use_degree,
                        gravity_type=args.gravity_type,
                )  if args.model.find('Graphormer') != -1 else None


cat_dict = torch.load('/NFTGraph/ogb_graph/example_direct_use/ogbl-nft_random_no_edge_attr/meta_dict.pt')
cat_dict['dir_path'] = '/NFTGraph/ogb_graph/example_direct_use/ogbl-nft_random_no_edge_attr'
cat_dict['eval metric'] = 'mrr'
dataset = DglLinkPropPredDataset(name = 'ogbl-nft_random_no_edge_attr',root = cat_dict['dir_path'] ,meta_dict=cat_dict)



graph = dataset[0]
split_edge = dataset.get_edge_split()


if not args.use_edge_weight and "weight" in graph.edata:
    del graph.edata["weight"]
if not args.use_feature and "feat" in graph.ndata:
    del graph.ndata["feat"]

data_appendix = "_rph{}".format("".join(str(args.ratio_per_hop).split(".")))  # ngnn
path = f"{dataset.root}_seal{data_appendix}"
if not (args.dynamic_train or args.dynamic_val or args.dynamic_test):
    args.num_workers = 0  # ngnn里为啥加这句？ngnn不动态有问题，加不加这句都只有一个worker，且用不了GPU

# dataset_pyg = PygLinkPropPredDataset(name=args.dataset, root=args.root)
# data_pyg = dataset_pyg[0]

train_dataset, val_dataset, test_dataset, final_val_dataset = [
    SEALOGBLDataset(
        # data_pyg,
        preprocess_fn,
        path,
        graph,
        split_edge,
        percent=percent,
        split=split,
        ratio_per_hop=args.ratio_per_hop,
        directed=directed,
        dynamic=dynamic,
    )
    for percent, split, dynamic in zip(
        [
            args.train_percent,
            args.val_percent,
            args.test_percent,
            args.final_val_percent,
        ],
        ["train", "valid", "test", "valid"],
        [
            args.dynamic_train,
            args.dynamic_val,
            args.dynamic_test,
            args.dynamic_val,
        ],)]


def ogbl_collate_fn(batch):
    gs, zs, xs, ws, ys, g_noaugs = zip(*batch)
    batched_g = dgl.batch(gs)
    z = torch.cat(zs, dim=0)
    if xs[0] is not None:
        x = torch.cat(xs, dim=0)
    else:
        x = None
    # if ws[0] is not None:
    #     edge_weights = torch.cat(ws, dim=0)
    # else:
    edge_weights = None
    y = torch.tensor(ys)

    # 把pairwise结构特征组装成batch
    if 'Graphormer' in args.model:
        batched_g.attn_bias = torch.cat([g_noaug.pair_attn_bias for g_noaug in g_noaugs], dim=0)
        batched_g.edge_index = torch.cat([g_noaug.pair_edge_idx for g_noaug in g_noaugs], dim=0)
        # batched_g.x = torch.cat([g_noaug.pair_x for g_noaug in g_noaugs], dim=0)
        batched_g.z = torch.cat([g_noaug.pair_z for g_noaug in g_noaugs], dim=0)
        if args.use_len_spd:
            batched_g.len_shortest_path = torch.cat([g_noaug.pair_len_shortest_path for g_noaug in g_noaugs], dim=0)
        if args.use_num_spd:
            batched_g.num_shortest_path = torch.cat([g_noaug.pair_num_shortest_path for g_noaug in g_noaugs], dim=0)
        if args.use_cnb_jac:
            batched_g.undir_jac = torch.cat([g_noaug.pair_undir_jac for g_noaug in g_noaugs], dim=0)
        if args.use_cnb_aa:
            batched_g.undir_aa = torch.cat([g_noaug.pair_undir_aa for g_noaug in g_noaugs], dim=0)
        if args.use_cnb_ra:
            batched_g.undir_ra = torch.cat([g_noaug.pair_undir_ra for g_noaug in g_noaugs], dim=0)
        if args.use_degree:
            batched_g.undir_degree = torch.cat([g_noaug.pair_undir_degree for g_noaug in g_noaugs], dim=0)
            if directed:
                batched_g.in_degree = torch.cat([g_noaug.pair_in_degree for g_noaug in g_noaugs], dim=0)
                batched_g.out_degree = torch.cat([g_noaug.pair_out_degree for g_noaug in g_noaugs], dim=0)
    return batched_g, z, x, edge_weights, y


train_loader = GraphDataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,  # True-----------------------
    collate_fn=ogbl_collate_fn,
    num_workers=args.num_workers,
)
# pdb.set_trace()
val_loader = GraphDataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=ogbl_collate_fn,
    num_workers=args.num_workers,
)
test_loader = GraphDataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=ogbl_collate_fn,
    num_workers=args.num_workers,
)
final_val_loader = GraphDataLoader(
    final_val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=ogbl_collate_fn,
    num_workers=args.num_workers,
)



emb = None
model_k = 10
print_log(f'\n args: {args} \n')

for run in range(args.runs):
    set_random_seed(args.seed + run)
    stime = datetime.datetime.now()

    print_log(f"\n++++++\n\nstart run [{run+1}], {stime}")

    if args.model == 'NGNNDGCNNGraphormer':
        model = NGNNDGCNNGraphormer(args, args.hidden_channels, args.num_layers, args.max_z,
                k=model_k, feature_dim=graph.ndata["feat"].size(1) if (args.use_feature and "feat" in graph.ndata) else 0,
                use_feature=args.use_feature, use_feature_GT=args.use_feature_GT,
                node_embedding=emb, readout_type=args.readout_type).to(device)
    # print_log(model)
    parameters = list(model.parameters())

    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    
    lr = args.lr
    optimizer = torch.optim.Adam(params=parameters, lr=lr)  # , weight_decay=0.002
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)
    total_params = sum(p.numel() for param in parameters for p in param)
    localtime = time.asctime(time.localtime(time.time()))
    print(f'{localtime} Total number of parameters is {total_params}')
    if args.model.find('DGCNN') != -1:
        print(f'SortPooling k is set to {model_k}')
    with open(log_file, 'a') as f:
        print(f'Total number of parameters is {total_params}', file=f)
        if args.model.find('DGCNN') != -1:
            print(f'SortPooling k is set to {model_k}', file=f)

    start_epoch = 1
    if args.continue_from is not None:
        model.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_model_checkpoint{}.pth'.format(run+1, args.continue_from)))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_optimizer_checkpoint{}.pth'.format(run+1, args.continue_from)))
        )
        start_epoch = args.continue_from + 1
        args.epochs -= args.continue_from
    
    if args.only_test:
        results = test(args.eval_metric)
        for key, result in results.items():
            loggers[key].add_result(run, result)
        for key, result in results.items():
            valid_res, test_res = result
            localtime = time.asctime(time.localtime(time.time()))
            print(f'[{localtime}] {key}')
            print(f'[{localtime}] Run: {run + 1}',
                  f'[{localtime}] Valid: {100 * valid_res}%',
                  f'[{localtime}] Test: {100 * test_res}%')
        # pdb.set_trace()
        exit()

    start_epoch = 1
    # Training starts.
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epo_stime = datetime.datetime.now()
        loss = train(len(train_dataset))
        epo_train_etime = datetime.datetime.now()
        print_log(
            f"[epoch: {epoch}]",
            f"   <Train> starts: {epo_stime}, "
            f"ends: {epo_train_etime}, "
            f"spent time:{epo_train_etime - epo_stime}",
        )
        #Validation starts.
        if epoch % args.eval_steps == 0:
            epo_eval_stime = datetime.datetime.now()
            results = test(val_loader,len(val_dataset))
            epo_eval_etime = datetime.datetime.now()
            print_log(
                f"   <Validation> starts: {epo_eval_stime}, "
                f"ends: {epo_eval_etime}, "
                f"spent time:{epo_eval_etime - epo_eval_stime}"
            )
            print_log('\n Validation: ')
            for key, valid_res in results.items():
                loggers[key].add_result(run, valid_res)
                print_log(f"{key}")
                print_log(f"Run: {run+1},"
                        f"Epoch: {epoch}, "
                        f"Loss: {loss}, "
                        f"Valid ({args.val_percent}%) [{key}]: {valid_res}")
            model_name = os.path.join(
                args.res_dir, f"run{run+1}_model_checkpoint{epoch}.pth"
            )
            optimizer_name = os.path.join(
                args.res_dir, f"run{run+1}_optimizer_checkpoint{epoch}.pth"
            )
            torch.save(model.state_dict(), model_name)
            torch.save(optimizer.state_dict(), optimizer_name)

    print_log()
    tested = dict()
    # Select models according to the eval_metric of the dataset.
    res = torch.tensor(loggers[dataset.eval_metric].results["valid"][run])
    idx_to_test = (
        torch.topk(res, args.test_topk, largest=True).indices + 1
        ).tolist()  # indices of top k valid results

    print_log(
        f"Eval Metric: {dataset.eval_metric}",
        f"Run: {run + 1}, "
        f"Top {args.test_topk} Eval Points/Eval Epoch: {idx_to_test}",
    )
    for _idx, eval_epoch in enumerate(idx_to_test):
        print_log(
            f"Test Point[{_idx+1}]: ",
            f"Eval Epoch {eval_epoch}, ",
            f"Test Metric: {dataset.eval_metric}",
        )
        if eval_epoch not in tested:
            model_name = os.path.join(
                args.res_dir, f"run{run+1}_model_checkpoint{eval_epoch * args.eval_steps}.pth"
            )
            optimizer_name = os.path.join(
                args.res_dir,
                f"run{run+1}_optimizer_checkpoint{eval_epoch * args.eval_steps}.pth",
            )
            model.load_state_dict(torch.load(model_name))
            optimizer.load_state_dict(torch.load(optimizer_name))
            tested[eval_epoch] = (
                test(final_val_loader,len(final_val_dataset)),
                test(test_loader,len(test_dataset)),
            )

        val_res, test_res = tested[eval_epoch]
        for iter_metric in loggers.keys():
            loggers[iter_metric].add_result(
                run, (eval_epoch*args.eval_steps, val_res[iter_metric], test_res[iter_metric]), "test"
            )
            print_log(
                f"   Run: {run + 1}, "
                f"True Epoch: {eval_epoch*args.eval_steps}, "
                f"Valid ({args.val_percent}%) [{iter_metric}]: "
                f"{loggers[iter_metric].results['valid'][run][eval_epoch-1]}, "
                f"Valid (final) [{iter_metric}]: {val_res[iter_metric]}, "
                f"Test [{iter_metric}]: {test_res[iter_metric]}"
            )
    if (run+1) % 3 == 0 or ((run+1) == args.runs):
        for key in loggers.keys():
            print(f"\n{key}")
            loggers[key].print_statistics(run)
            with open(log_file, "a") as f:
                print(f"\n run:{run+1},{key}", file=f)
                # print(f"\n {key}", file=f)
                loggers[key].print_statistics(run,f=f)
    etime = datetime.datetime.now()
    print_log(
        f"end run [{run+1}], {etime}",
        f"spent time:{etime-stime}",
    )
    for key in loggers.keys():
        print_log(f"\n run: {run+1} loggersKey: {key}")
        print_log(f'loggers[{key}].result[\'valid\']:  ',loggers[key].results['valid'])
        print_log(f'loggers[{key}].result[\'test\']:  ',loggers[key].results['test'])

print_log(f"\n Total number of parameters is {total_params}")
print_log(f"Results are saved in {args.res_dir}")
