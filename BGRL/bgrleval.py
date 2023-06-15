import dgl
import src.evaluator
import torch
from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset, WikiCSDataset, CoauthorCSDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorPhysicsDataset
import os, sys
import statistics
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0,
                    help='which GPU to run, -1 for cpu')
parser.add_argument('--batch_size', type=int, default=10240,
                    help='batch size for link prediciton.')
parser.add_argument('--neg_rate', type=int, default=1,
                    help='negative rate for link prediction.')
parser.add_argument('--data', type=str, 
                    help='Dataset to evaluate.')
parser.add_argument('--embedding_path_node', type=str, 
                    help='path for saved node embedding.')
parser.add_argument('--embedding_path_link', type=str, 
                    help='path for save node embedding (intended for link prediction downstream task).')

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


args = parser.parse_args()
args.data = 'tinynft_nodetype'
args.device = 0
args.embedding_path_node = './bgrl/X_bgrl.pt'
device = f'cuda:{args.device}' if args.device != -1 else 'cpu'
batch_size = args.batch_size
neg_rate = args.neg_rate
dataset = args.data



if dataset == 'tinynft_nodetype':
    from ogb.nodeproppred import DglNodePropPredDataset
    cat_dict = torch.load('/NFTGraph/ogb_graph/example_direct_use/submission_ogbn_tinynft_nodetype/meta_dict.pt')
    cat_dict['dir_path'] = '/NFTGraph/ogb_graph/example_direct_use/submission_ogbn_tinynft_nodetype'
    dataset = DglNodePropPredDataset(name = 'ogbn-tinynft_nodetype',root = cat_dict['dir_path'] ,meta_dict=cat_dict)
    g = dataset[0][0]
    g.ndata['label'] = dataset[0][1]

    num_nodes = g.num_nodes()
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    if not torch.is_tensor(train_idx):
        train_idx = torch.as_tensor(train_idx)
        val_idx = torch.as_tensor(val_idx)
        test_idx = torch.as_tensor(test_idx)

    train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    g.ndata["label"] = dataset[0][1].view(-1)
    g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"] = train_mask, val_mask, test_mask
    dataset = 'tinynft_nodetype'


    
metis_label = torch.load(f'/ParetoGNN/pretrain_labels/metis_label_tinynft_nodetype.pt', map_location='cpu')
embedding_path_node = args.embedding_path_node
embedding_path_link = args.embedding_path_link
tvt_edges_file = f'/ParetoGNN/links/tinynft_nodetype_tvtEdges.pkl'


with HiddenPrints():
    X_node = torch.load(embedding_path_node, map_location=device)
if dataset in ['products', 'arxiv','tinynft_nodetype']:
    print("Starting Node Classification...")
    with HiddenPrints():
        ssnc_acc, ssnc_acc_std = src.evaluator.fit_logistic_regression_neural_net_preset_splits(X_node, g.ndata['label'], \
            g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'], repeat=3, device=device)
    
    print("Start Partition Prediction...")
    with HiddenPrints():
        metis, metis_std = src.evaluator.fit_logistic_regression_neural_net(X_node, metis_label, device=device)
else:
    ssnc_acc, ssnc_acc_std = src.evaluator.fit_logistic_regression(X_node.cpu().numpy(), g.ndata['label'], repeat=3)
    metis, metis_std = src.evaluator.fit_logistic_regression(X_node.cpu().numpy(), metis_label, repeat=3)
print("Start Node Clustering...")
with HiddenPrints():
    nmi, nmi_std = src.evaluator.fit_node_clustering(X_node.cpu().numpy(), g.ndata['label'])

print('MEAN: ACC: {:.4f}, NMI:{:.4f}, METIS: {:0.4f}, HMEAN:{:0.4f}'.format(ssnc_acc, nmi, metis, np.mean([ssnc_acc, nmi, metis])))
print('STD : ACC: {:.4f}, NMI:{:.4f}, METIS: {:0.4f}'.format(ssnc_acc_std, nmi_std, metis_std))