import argparse, time

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgi import Classifier, DGI
from dgl import DGLGraph
from dgl.data import load_data, register_data_args
from tqdm import tqdm

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset

    from ogb.nodeproppred import DglNodePropPredDataset
    cat_dict = th.load('/data/sx/ogbn-allnft/data/tinynft/submission_ogbn_tinynft_nodetype/meta_dict.pt')
    cat_dict['dir_path'] = '/data/sx/ogbn-allnft/data/tinynft/submission_ogbn_tinynft_nodetype/tinynft_nodetype'
    dataset = DglNodePropPredDataset(name = 'ogbn-tinynft_nodetype',root = cat_dict['dir_path'] ,meta_dict=cat_dict)
    g = dataset[0][0]
    g.ndata['label'] = dataset[0][1]
    import pickle
    g, labels = dataset[0]
    num_nodes = g.num_nodes()
    tvt_addr = '/ParetoGNN/links/tinynft_nodetype_tvtEdges.pkl'
    mask_edge = True
    if mask_edge:
        _, _, val_edges, _, test_edges, _ = pickle.load(open(tvt_addr, 'rb'))
        lst = []
        lst.append(g.edge_ids(val_edges[:,0], val_edges[:,1]))
        # lst.append(g.edge_ids(val_edges[:,1], val_edges[:,0]))
        lst.append(g.edge_ids(test_edges[:,0], test_edges[:,1]))
        # lst.append(g.edge_ids(test_edges[:,1], test_edges[:,0]))
        lst = torch.cat(lst)
        g.remove_edges(lst)

    # data = load_data(args)
    # g = data[0]
    features = torch.FloatTensor(g.ndata["feat"])
    labels = torch.LongTensor(g.ndata["label"])
    train_mask,val_mask,test_mask = None,None,None
    # if hasattr(torch, "BoolTensor"):
    #     train_mask = torch.BoolTensor(g.ndata["train_mask"])
    #     val_mask = torch.BoolTensor(g.ndata["val_mask"])
    #     test_mask = torch.BoolTensor(g.ndata["test_mask"])
    # else:
    #     train_mask = torch.ByteTensor(g.ndata["train_mask"])
    #     val_mask = torch.ByteTensor(g.ndata["val_mask"])
    #     test_mask = torch.ByteTensor(g.ndata["test_mask"])
    in_feats = features.shape[1]
    n_classes = 2 #g.num_classes
    n_edges = g.number_of_edges()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        # train_mask = train_mask.cuda()
        # val_mask = val_mask.cuda()
        # test_mask = test_mask.cuda()

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    if args.gpu >= 0:
        g = g.to(args.gpu)
    # create DGI model
    dgi = DGI(
        g,
        in_feats,
        args.n_hidden,
        args.n_layers,
        nn.PReLU(args.n_hidden),
        args.dropout,
    )

    if cuda:
        dgi.cuda()

    dgi_optimizer = torch.optim.Adam(
        dgi.parameters(), lr=args.dgi_lr, weight_decay=args.weight_decay
    )

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    for epoch in tqdm(range(args.n_dgi_epochs),ncols=70):
        dgi.train()
        if epoch >= 3:
            t0 = time.time()

        dgi_optimizer.zero_grad()
        loss = dgi(features)
        loss.backward()
        dgi_optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), "best_dgi.pkl")
        else:
            cnt_wait += 1

        # if cnt_wait == args.patience:
        #     print("Early stopping!")
        #     break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print(
            "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
            "ETputs(KTEPS) {:.2f}".format(
                epoch, np.mean(dur), loss.item(), n_edges / np.mean(dur) / 1000
            )
        )

    # # create classifier model
    # classifier = Classifier(args.n_hidden, n_classes)
    # if cuda:
    #     classifier.cuda()

    # classifier_optimizer = torch.optim.Adam(
    #     classifier.parameters(),
    #     lr=args.classifier_lr,
    #     weight_decay=args.weight_decay,
    # )

    # train classifier
    print("Loading {}th epoch".format(best_t))
    dgi.load_state_dict(torch.load("best_dgi.pkl"))
    embeds = dgi.encoder(features, corrupt=False)
    torch.save(embeds,'./X_node_dgi_incomplete.pt')
    # embeds = embeds.detach()
    # dur = []
    # for epoch in range(args.n_classifier_epochs):
    #     classifier.train()
    #     if epoch >= 3:
    #         t0 = time.time()

    #     classifier_optimizer.zero_grad()
    #     preds = classifier(embeds)
    #     loss = F.nll_loss(preds[train_mask], labels[train_mask])
    #     loss.backward()
    #     classifier_optimizer.step()

    #     if epoch >= 3:
    #         dur.append(time.time() - t0)

    #     acc = evaluate(classifier, embeds, labels, val_mask)
    #     print(
    #         "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
    #         "ETputs(KTEPS) {:.2f}".format(
    #             epoch,
    #             np.mean(dur),
    #             loss.item(),
    #             acc,
    #             n_edges / np.mean(dur) / 1000,
    #         )
    #     )

    # print()
    # acc = evaluate(classifier, embeds, labels, test_mask)
    # print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGI")
    register_data_args(parser)
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout probability"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument(
        "--dgi-lr", type=float, default=1e-3, help="dgi learning rate"
    )
    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=1e-2,
        help="classifier learning rate",
    )
    parser.add_argument(
        "--n-dgi-epochs",
        type=int,
        default=100000,
        help="number of training epochs",
    )
    parser.add_argument(
        "--n-classifier-epochs",
        type=int,
        default=300,
        help="number of training epochs",
    )
    parser.add_argument(
        "--n-hidden", type=int, default=128, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=1, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="early stop patience condition"
    )
    parser.add_argument(
        "--self-loop",
        action="store_true",
        help="graph self-loop (default=False)",
    )
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    args.gpu = 2
    args.self_loop = True
    print(args)

    main(args)