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

from models import *
from utils import *


class SEALOGBLDataset(Dataset):
    def __init__(
        self,
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
        self.root = root
        self.graph = graph
        self.split = split
        self.split_edge = split_edge
        self.percent = percent
        self.ratio_per_hop = ratio_per_hop
        self.directed = directed
        self.dynamic = dynamic

        if "weights" in self.graph.edata:
            self.edge_weights = self.graph.edata["weights"]
        else:
            self.edge_weights = None
        if "feat" in self.graph.ndata:
            self.node_features = self.graph.ndata["feat"]
        else:
            self.node_features = None

        pos_edge, neg_edge = get_pos_neg_edges(
            self.split, self.split_edge, self.graph, self.percent
        )
        self.links = torch.cat([pos_edge, neg_edge], 0)  # [Np + Nn, 2]
        self.labels = np.array([1] * len(pos_edge) + [0] * len(neg_edge))

        if not self.dynamic:
            self.g_list, tensor_dict = self.load_cached()
            self.labels = tensor_dict["y"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.dynamic:
            g, y = self.g_list[idx], self.labels[idx]
            x = None if "x" not in g.ndata else g.ndata["x"]
            w = None if "w" not in g.edata else g.eata["w"]
            return g, g.ndata["z"], x, w, y

        src, dst = self.links[idx][0].item(), self.links[idx][1].item()
        y = self.labels[idx]
        subg = k_hop_subgraph(
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

        NIDs, EIDs = subg.ndata[dgl.NID], subg.edata[dgl.EID]

        z = drnl_node_labeling(subg.adj(scipy_fmt="csr"), 0, 1)
        edge_weights = (
            self.edge_weights[EIDs] if self.edge_weights is not None else None
        )
        x = self.node_features[NIDs] if self.node_features is not None else None

        subg_aug = subg.add_self_loop()
        if edge_weights is not None:
            edge_weights = torch.cat(
                [
                    edge_weights,
                    torch.ones(subg_aug.num_edges() - subg.num_edges()),
                ]
            )
        return subg_aug, z, x, edge_weights, y

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

def ogbl_collate_fn(batch):
    gs, zs, xs, ws, ys = zip(*batch)
    batched_g = dgl.batch(gs)
    z = torch.cat(zs, dim=0)
    if xs[0] is not None:
        x = torch.cat(xs, dim=0)
    else:
        x = None
    if ws[0] is not None:
        edge_weights = torch.cat(ws, dim=0)
    else:
        edge_weights = None
    y = torch.tensor(ys)

    return batched_g, z, x, edge_weights, y

def train():
    model.train()
    loss_fnt = BCEWithLogitsLoss()
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for batch in pbar:
        g, z, x, edge_weights, y = [
            item.to(device) if item is not None else None for item in batch
        ]
        optimizer.zero_grad()
        logits = model(g, z, x, edge_weight=edge_weights)
        loss = loss_fnt(logits.view(-1), y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * g.batch_size

    return total_loss / len(train_dataset)


@torch.no_grad()
def test(dataloader, hits_K=["hits@100"]):
    model.eval()

    if isinstance(hits_K, (int, str)):
        hits_K = [hits_K]
    y_pred, y_true = [], []
    for batch in tqdm(dataloader, ncols=70):
        g, z, x, edge_weights, y = [
            item.to(device) if item is not None else None for item in batch
        ]
        logits = model(g, z, x, edge_weight=edge_weights)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(y.view(-1).cpu().to(torch.float))
    y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
    pos_y_pred = y_pred[y_true == 1]
    neg_y_pred = y_pred[y_true == 0]

    if dataset.eval_metric.startswith("hits@"):
        results = evaluate_hits(pos_y_pred, neg_y_pred, hits_K)
    elif dataset.eval_metric == "mrr":
        results = evaluate_mrr(pos_y_pred, neg_y_pred)
    elif dataset.eval_metric == "rocauc":
        results = evaluate_rocauc(pos_y_pred, neg_y_pred)
    
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


if __name__ == "__main__":
    # Data settings
    parser = argparse.ArgumentParser(description="OGBL (SEAL)")
    parser.add_argument("--dataset", type=str, default="ogbl-nft_random_no_edge_attr")
    # GNN settings
    parser.add_argument(
        "--max_z",
        type=int,
        default=1000,
        help="max number of labels as embeddings to look up",
    )
    parser.add_argument("--sortpool_k", type=float, default=0.6)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--ngnn_type",
        type=str,
        default="none",
        choices=["none", "input", "hidden", "output", "all"],
        help="You can set this value from 'none', 'input', 'hidden' or 'all' "
        "to apply NGNN to different GNN layers.",
    )
    parser.add_argument(
        "--num_ngnn_layers", type=int, default=1, choices=[1, 2]
    )
    # Subgraph extraction settings
    parser.add_argument("--ratio_per_hop", type=float, default=0.1)
    parser.add_argument(
        "--use_feature",
        action="store_true",
        help="whether to use raw node features as GNN input",
    )
    parser.add_argument(
        "--use_edge_weight",
        action="store_true",
        help="whether to consider edge weight in GNN",
    )
    # Training settings
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training.",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--train_percent", type=float, default=1)
    parser.add_argument("--val_percent", type=float, default=1)
    parser.add_argument("--final_val_percent", type=float, default=100)
    parser.add_argument("--test_percent", type=float, default=100)
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument(
        "--dynamic_train",
        action="store_true",
        help="dynamically extract enclosing subgraphs on the fly",
    )
    parser.add_argument("--dynamic_val", action="store_true")
    parser.add_argument("--dynamic_test", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=24,
        help="number of workers for dynamic dataloaders; "
        "using a larger value for dynamic dataloading is recommended",
    )
    # Testing settings
    parser.add_argument(
        "--use_valedges_as_input",
        action="store_true",
        help="available for ogbl-collab",
    )
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument(
        "--eval_hits_K",
        type=int,
        nargs="*",
        default=[10],
        help="hits@K for each eval step; "
        "only available for datasets with hits@xx as the eval metric",
    )
    parser.add_argument(
        "--test_topk",
        type=int,
        default=1,
        help="select best k models for full validation/test each run.",
    )

    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--nthread', type=int, default=16, help='number of thread')
    
    args = parser.parse_args([])


    args.device= 2
    args.hidden_channels= 32
    args.runs = 10
    args.epochs = 15
    args.lr = 1e-03
    args.batch_size = 64
    args.num_workers = 24 
    args.train_percent = 100
    args.val_percent = 10
    args.directed = True
    args.eval_steps = 3
    
    args.ngnn_type='none'
    # args.num_ngnn_layers = 2
    args.use_feature = False
    args.use_edge_weight = False
    args.dynamic_train = True
    args.dynamic_val = True
    args.dynamic_test = True
    args.ratio_per_hop = 0.1
    args.nthread = 16

    
    if args.nthread > 0:
        torch.set_num_threads(args.nthread)


    data_appendix = "_rph{}".format("".join(str(args.ratio_per_hop).split(".")))
    if args.use_valedges_as_input:
        data_appendix += "_uvai"

    args.res_dir = os.path.join(
        f'./results{"_NoTest" if args.no_test else ""}',
        f'{args.dataset.split("-")[1]}-{args.ngnn_type}+{time.strftime("%m%d%H%M%S")}',
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


    if not args.use_edge_weight and "weight" in graph.edata:
        del graph.edata["weight"]
    if not args.use_feature and "feat" in graph.ndata:
        del graph.ndata["feat"]



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
    path = f"{dataset.root}_seal{data_appendix}"

    if not (args.dynamic_train or args.dynamic_val or args.dynamic_test):
        args.num_workers = 0

    train_dataset, val_dataset, final_val_dataset, test_dataset = [
        SEALOGBLDataset(
            path,
            graph,
            split_edge,
            percent=percent,
            split=split,
            ratio_per_hop=args.ratio_per_hop,
            directed=args.directed,
            dynamic=dynamic,
        )
        for percent, split, dynamic in zip(
            [
                args.train_percent,
                args.val_percent,
                args.final_val_percent,
                args.test_percent,
            ],
            ["train", "valid", "valid", "test"],
            [
                args.dynamic_train,
                args.dynamic_val,
                args.dynamic_test,
                args.dynamic_test,
            ],
        )
    ]


    # final_val_dataset = val_dataset
    # test_dataset = val_dataset

    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = GraphDataLoader(
        val_dataset,
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
    test_loader = GraphDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ogbl_collate_fn,
        num_workers=args.num_workers,
    )

    model_k = 10

    print_log(f"training starts: {datetime.datetime.now()}")

    for run in range(args.runs):
        set_random_seed(args.seed + run)
        stime = datetime.datetime.now()
        print_log(f"\n++++++\n\nstart run [{run+1}], {stime}")

        model = DGCNN(
            args.hidden_channels,
            args.num_layers,
            args.max_z,
            model_k,
            feature_dim=graph.ndata["feat"].size(1)
            if (args.use_feature and "feat" in graph.ndata)
            else 0,
            dropout=args.dropout,
            ngnn_type=args.ngnn_type,
            num_ngnn_layers=args.num_ngnn_layers,
        ).to(device)
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
        total_params = sum(p.numel() for param in parameters for p in param)
        print_log(
            f"Total number of parameters is {total_params}",
            f"SortPooling k is set to {model.k}",
        )

        start_epoch = 1
        # Training starts.
        for epoch in range(start_epoch, start_epoch + args.epochs):
            epo_stime = datetime.datetime.now()
            loss = train()
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
                results = test(val_loader)
                epo_eval_etime = datetime.datetime.now()
                print_log(
                    f"   <Validation> starts: {epo_eval_stime}, "
                    f"ends: {epo_eval_etime}, "
                    f"spent time:{epo_eval_etime - epo_eval_stime}"
                )
                print_log('\n Validation: ')
                for key, valid_res in results.items():
                    loggers[key].add_result(run, valid_res)
                    print_log(f"\n{key}")
                    print_log(  f"Run: {run+1}, "
                                f"Epoch: {epoch}, "
                                f"Loss: {loss}, "
                                f"Valid ({args.val_percent}%) [{key}]: {valid_res} \n")

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
                f"Test Point[{_idx+1}]: "
                f"Eval Epoch {eval_epoch}, "
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
                    test(final_val_loader),
                    test(test_loader),
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
            print_log(f"run: {run+1} loggersKey: {key}")
            print_log(f'loggers[{key}].result[\'valid\']:  ',loggers[key].results['valid'])
            print_log(f'loggers[{key}].result[\'test\']:  ',loggers[key].results['test'])

    print_log(f"Total number of parameters is {total_params}")
    print_log(f"Results are saved in {args.res_dir}")