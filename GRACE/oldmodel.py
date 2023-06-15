import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GraphConv


# Multi-layer Graph Convolutional Networks
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, num_layers=2):
        super(GCN, self).__init__()

        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, out_dim * 2))
        for _ in range(self.num_layers - 2):
            self.convs.append(GraphConv(out_dim * 2, out_dim * 2))

        self.convs.append(GraphConv(out_dim * 2, out_dim))
        self.act_fn = act_fn

    def forward(self, graph, feat):
        for i in range(self.num_layers):
            feat = self.act_fn(self.convs[i](graph, feat))

        return feat


# Multi-layer(2-layer) Perceptron
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        return self.fc2(z)


class Grace(nn.Module):
    r"""
        GRACE model
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.
    out_dim: int
        Output feature size.
    num_layers: int
        Number of the GNN encoder layers.
    act_fn: nn.Module
        Activation function.
    temp: float
        Temperature constant.
    """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp):
        super(Grace, self).__init__()
        self.encoder = GCN(in_dim, hid_dim, act_fn, num_layers)
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        # s1 = th.mm(z1[:500000], z2.t())
        # s2 = th.mm(z1[500000:], z2.t())
        # s =th.cat([s1,s2],dim=0)
        s = th.mm(z1,z2.t())
        # s = F.normalize(s)
        return s

    def get_loss(self, z1, z2):
        # calculate SimCLR loss
        f = lambda x: th.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1))  # intra-view pairs
        between_sim = f(self.sim(z1, z2))  # inter-view pairs

        # between_sim.diag(): positive pairs
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -th.log(between_sim.diag() / x1)

        return loss

    def get_embedding(self, graph, feat):
        # get embeddings from the model for evaluation
        h = self.encoder(graph, feat)

        return h.detach()

    def forward(self, graph1, graph2, feat1, feat2, device):
        # encoding
        h1 = self.encoder(graph1, feat1)
        h2 = self.encoder(graph2, feat2)

        # projection

        z1 = th.cat([self.proj(h1[:1000000]),self.proj(h1[1000000:])],dim=0)
        z2 = th.cat([self.proj(h2[:1000000]),self.proj(h2[1000000:])],dim=0)
        # z2 = self.proj(h2)
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=64)
        # z1 = th.tensor(pca.fit_transform(z1.cpu().detach().numpy())).to(device)
        # z2 = th.tensor(pca.fit_transform(z2.cpu().detach().numpy())).to(device)
        z1 = z1[:100000]
        z2 = z2[:100000]
        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)

        ret = (l1 + l2) * 0.5

        return ret.nanmean()
