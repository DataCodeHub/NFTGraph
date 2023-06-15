import argparse
import math
import time
import dgl
# from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import numpy as np
# import numpy_indexed as npi
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
# from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.utils.data import DataLoader
from dgl.sampling import random_walk
# from torch_cluster import random_walk
import os.path as osp
from gen_model import gen_model
from loss import calculate_loss
from utils import *
import random
import os
import sys
from sklearn.metrics import roc_auc_score
import datetime
from tqdm import tqdm 

def compute_pred(h, predictor, edges, batch_size):
    preds = []
    for perm in DataLoader(range(edges.size(0)), batch_size):
        edge = edges[perm].t()

        preds += [predictor(h[edge[0]], h[edge[1]]).sigmoid().squeeze().cpu().view(-1)]
    pred = torch.cat(preds, dim=0)
    return pred


def train(model, predictor, feat, edge_feat, graph, split_edge, optimizer, batch_size, args):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(feat.device)
    edge_weight_margin = None
    neg_train_edge = split_edge['train']['edge_neg'].to(feat.device)

    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True),ncols=70):
        optimizer.zero_grad()

        h = model(graph, feat, edge_feat)
        
        edge = pos_train_edge[perm]
        neg_edge = neg_train_edge[perm]

        pos_out = predictor(h[edge[:, 0]], h[edge[:, 1]])
        # pos_loss = -torch.log(pos_out + 1e-15).mean()

        neg_out = predictor(h[neg_edge[:,0]], h[neg_edge[:,1]])
        # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        # loss = pos_loss + neg_loss
        weight_margin = edge_weight_margin[perm].to(feat.device) if edge_weight_margin is not None else None

        loss = calculate_loss(pos_out, neg_out, args.n_neg, margin=weight_margin, loss_func_name=args.loss_func)
        # cross_out = predictor(h[edge[:,0].view(-1, 1)], h[neg_edge[:,1].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,0].view(-1, 1)], h[neg_edge[:,0].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,1].view(-1, 1)], h[neg_edge[:,1].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,1].view(-1, 1)], h[neg_edge[:,0].view(-1, args.n_neg)])
        # cross_loss = -torch.log(1 - cross_out.sigmoid() + 1e-15).sum()
        # loss = loss + 0.1 * cross_loss
        loss.backward()

        if args.clip_grad_norm > -1:
            # if 'feat' not in graph.ndata:
            #     torch.nn.utils.clip_grad_norm_(feat, args.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad_norm)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, mode, predictor, feat, edge_feat, graph,  split_edge, batch_size):
    model.eval()
    predictor.eval()

    if mode == 'valid':
        pos_valid_edge = split_edge['eval_train']['edge'].to(feat.device)
        neg_valid_edge = split_edge['eval_train']['edge_neg'].to(feat.device)
        tofeed_pos_edge = pos_valid_edge
        tofeed_neg_edge = neg_valid_edge
    elif mode == 'final_val':
        pos_finalval_edge = split_edge['valid']['edge'].to(feat.device)
        neg_finalval_edge = split_edge['valid']['edge_neg'].to(feat.device)
        tofeed_pos_edge = pos_finalval_edge
        tofeed_neg_edge = neg_finalval_edge
    elif mode == 'test':
        pos_test_edge = split_edge['test']['edge'].to(feat.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(feat.device)
        tofeed_pos_edge = pos_test_edge
        tofeed_neg_edge = neg_test_edge

    h = model(graph, feat, edge_feat)

    pos_y_pred = compute_pred(h, predictor, tofeed_pos_edge, batch_size)
    neg_y_pred = compute_pred(h, predictor, tofeed_neg_edge, batch_size)

    if dataset.eval_metric == "mrr":
        results = evaluate_mrr(pos_y_pred, neg_y_pred)
    
    y_true = torch.cat([torch.ones(pos_y_pred.size(0)),torch.zeros(neg_y_pred.size(0))],dim=0)
    y_pred = torch.cat([pos_y_pred,neg_y_pred],dim=0)
    results['rocauc'] = roc_auc_score(y_true, y_pred)

    return results


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=4)
    # parser.add_argument('--log-steps', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='ogbl-nft_random_no_edge_attr')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage', 'agdn', 'memagdn'])
    parser.add_argument('--clip-grad-norm', type=float, default=1)
    parser.add_argument('--use-valedges-as-input', action='store_true',
                        help='This option can only be used for ogbl-collab')
    parser.add_argument('--no-node-feat', action='store_true')
    parser.add_argument('--use-emb', action='store_true')
    parser.add_argument('--use-edge-feat', action='store_true')
    parser.add_argument('--train-on-subgraph', action='store_true')
    parser.add_argument('--year', type=int, default=0)

    
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--transition-matrix', type=str, default='gat')
    parser.add_argument('--hop-norm', action='store_true')
    parser.add_argument('--weight-style', type=str, default='HA', choices=['HC', 'HA', 'HA+HC', 'HA1', 'sum', 'max_pool', 'mean_pool', 'lstm'])
    parser.add_argument('--no-pos-emb', action='store_true')
    parser.add_argument('--no-share-weights', action='store_true')
    parser.add_argument('--pre-act', action='store_true')
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--n-hidden', type=int, default=32)
    parser.add_argument('--n-heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--input-drop', type=float, default=0.)
    parser.add_argument('--edge-drop', type=float, default=0.)
    parser.add_argument('--attn-drop', type=float, default=0.)
    parser.add_argument('--diffusion-drop', type=float, default=0.)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--output-bn', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--no-dst-attn', action='store_true')
    
    parser.add_argument('--advanced-optimizer', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--eval_steps', type=int, default=3)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--negative-sampler', type=str, default='global', choices=['global', 'strict_global', 'persource'])
    parser.add_argument('--n-neg', type=int, default=1)
    parser.add_argument('--eval_metric', type=str, default='mrr')
    parser.add_argument('--loss-func', type=str, default='CE')
    parser.add_argument('--predictor', type=str, default='MLP')

    
    parser.add_argument('--random_walk_augment', action='store_true')
    parser.add_argument('--walk_start_type', type=str, default='edge')
    parser.add_argument('--walk_length', type=int, default=5)
    parser.add_argument('--adjust-lr', action='store_true')

    parser.add_argument('--use-heuristic', action='store_true')
    parser.add_argument('--n-extra-edges', type=int, default=200000)
    parser.add_argument('--heuristic-method', type=str, default='CN')
    parser.add_argument('--extra-training-edges', action='store_true')
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument('--nthread', type=int, default=16, help='number of thread')

    parser.add_argument("--train_percent", type=float, default=1)
    parser.add_argument("--val_percent", type=float, default=1)
    parser.add_argument("--final_val_percent", type=float, default=100)
    parser.add_argument("--test_percent", type=float, default=100)
    parser.add_argument("--eval_batch_size", type=float, default=64)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=24,
        help="number of workers for dynamic dataloaders; "
        "using a larger value for dynamic dataloading is recommended",
    )

    parser.add_argument(
        "--test_topk",
        type=int,
        default=1,
        help="select best k models for full validation/test each run.",
    )

    args = parser.parse_args()

    args.model = 'gcn'
    args.device= 5
    args.n_hidden= 32
    args.runs = 10
    args.epochs = 15
    args.lr = 1e-03
    args.batch_size = 64 * 10
    args.eval_batch_size = 64 *1000
    args.num_workers = 24 
    args.train_percent = 100
    args.val_percent = 0.1
    # args.directed = True
    args.eval_steps = 3
    
    args.use_feature = False
    args.use_edge_weight = False
    args.dynamic_train = True
    args.dynamic_val = True
    args.dynamic_test = True
    args.ratio_per_hop = 0.1
    args.nthread = 16


    if args.nthread > 0:
        torch.set_num_threads(args.nthread)


    args.res_dir = os.path.join(
        f'./results{"_NoTest" if args.no_test else ""}',
        f'{args.dataset.split("-")[1]}-{args.model}+{time.strftime("%m%d%H%M%S")}',
    )
    print(f"Results will be saved in {args.res_dir}")
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    log_file = os.path.join(args.res_dir, "log.txt")
    # Save command line input.
    cmd_input = "python " + " ".join(sys.argv) + "\n"
    with open(os.path.join(args.res_dir, "cmd_input.txt"), "a") as f:
        f.write(cmd_input)
    print(f"Command line input is saved.")
    print_log(f"{cmd_input}")
    print_log(f"{args}")


    cat_dict = torch.load('/NFTGraph/ogb_graph/example_direct_use/ogbl-nft_random_no_edge_attr/meta_dict.pt')
    cat_dict['dir_path'] = '/NFTGraph/ogb_graph/example_direct_use/ogbl-nft_random_no_edge_attr'

    cat_dict['eval metric'] = 'mrr'
    dataset = DglLinkPropPredDataset(name = 'ogbl-nft_random_no_edge_attr',root = cat_dict['dir_path'] ,meta_dict=cat_dict)

    graph = dataset[0]
    split_edge = dataset.get_edge_split()


    evaluator = Evaluator(name=args.dataset)

    loggers = {
        "mrr": Logger(args.runs, args),
        "rocauc": Logger(args.runs,args)
    }

    device = (
        f"cuda:{args.device}"
        if args.device != -1 and torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device)

    idx = torch.randperm(int(split_edge['valid']['edge'].size(0)*args.val_percent))
    idx = idx[:int(split_edge['valid']['edge'].size(0)*args.val_percent)]
    split_edge['eval_train'] = {'edge': split_edge['valid']['edge'][idx],'edge_neg': split_edge['valid']['edge_neg'][idx]}

    graph = dgl.graph((split_edge['train']['edge'].T[0],split_edge['train']['edge'].T[1]),num_nodes=graph.num_nodes())
    graph = graph.to(device)


    # Use learnable embedding if node attributes are not available
    n_heads = args.n_heads if args.model in ['gat', 'agdn'] else 1
    emb = torch.nn.Embedding(graph.number_of_nodes(), args.n_hidden).to(device)
    feat = emb.weight
    in_feats = feat.shape[1]

    edge_feat = None
    full_edge_feat = None
    in_edge_feats = 0

    print_log(f"training starts: {datetime.datetime.now()}")

    for run in range(args.runs):
        set_random_seed(args.seed + run)
        stime = datetime.datetime.now()
        print_log(f"\n++++++\n\nstart run [{run+1}], {stime}")
        model, predictor = gen_model(args, in_feats, in_edge_feats, device)
        parameters = list(model.parameters()) + list(predictor.parameters())
        if emb is not None:
            parameters = parameters + list(emb.parameters())
            torch.nn.init.xavier_uniform_(emb.weight)
            num_param = count_parameters(model) + count_parameters(predictor) + count_parameters(emb)
        else:
            num_param = count_parameters(model) + count_parameters(predictor)

        optimizer = torch.optim.Adam(parameters,lr=args.lr)
        print_log(f"Total number of parameters is {num_param}")

        start_epoch = 1
        # Training starts.
        for epoch in range(start_epoch, start_epoch + args.epochs):
            epo_stime = datetime.datetime.now()
            loss = train(model, predictor, feat, edge_feat, graph, split_edge, optimizer,
                         args.batch_size, args)
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
                results = test(model, 'valid', predictor, feat, edge_feat, graph,  split_edge, args.eval_batch_size)
                epo_eval_etime = datetime.datetime.now()
                print_log(
                    f"   <Validation> starts: {epo_eval_stime}, "
                    f"ends: {epo_eval_etime}, "
                    f"spent time:{epo_eval_etime - epo_eval_stime}"
                )
                print_log('\n Validation: ')
                for key, valid_res in results.items():
                    loggers[key].add_result(run, valid_res)
                    to_print = (
                        f"Run: {run+1}, "
                        f"Epoch: {epoch}, "
                        f"Loss: {loss}, "
                        f"Valid ({args.val_percent*100}%) [{key}]: {valid_res} \n"
                    )
                    print_log(key, to_print)

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
            f"Run: {run + 1:02d}, "
            f"Top {args.test_topk} Eval Points/Eval Epoch: {idx_to_test}",
        )
        for _idx, eval_epoch in enumerate(idx_to_test):
            print_log(
                f"Test Point[{_idx+1}]: "
                f"Eval Epoch {eval_epoch:02d}, "
                f"Test Metric: {dataset.eval_metric}"
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
                    test(model, 'final_val', predictor, feat, edge_feat, graph,  split_edge,  args.eval_batch_size),
                    test(model, 'test', predictor, feat, edge_feat, graph,  split_edge,  args.eval_batch_size),
                )

            val_res, test_res = tested[eval_epoch]
            for iter_metric in loggers.keys():
                loggers[iter_metric].add_result(
                    run, (eval_epoch*args.eval_steps, val_res[iter_metric], test_res[iter_metric]), "test"
                )
                print_log(
                    f"   Run: {run + 1}, "
                    f"True Epoch: {eval_epoch*args.eval_steps:02d}, "
                    f"Valid ({args.val_percent*100}%) [{iter_metric}]: "
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

    print_log(f"\n Total number of parameters is {num_param}")
    print_log(f"Results are saved in {args.res_dir}")

