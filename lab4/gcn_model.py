"""Cora 节点分类使用的三种模型：MLP、GCN、GAT。"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class MLP(torch.nn.Module):
    """不使用图结构的基线，只对节点特征做逐节点分类。"""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, **kwargs):
        super().__init__()
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.lins = torch.nn.ModuleList(
            torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def embed(self, x, edge_index=None):
        x = F.relu(self.lins[0](x))
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, **kwargs):
        super().__init__()
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = torch.nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, return_hidden=False):
        hidden = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                hidden.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return (x, hidden) if return_hidden else x

    def embed(self, x, edge_index):
        _, hidden = self.forward(x, edge_index, return_hidden=True)
        return hidden[0] if hidden else x


class GAT(torch.nn.Module):
    """多头图注意力网络，隐藏层每头 hidden_channels 维并拼接。"""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, heads=8, dropout=0.6, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(
                GATConv(in_channels, out_channels, heads=1, concat=False,
                        dropout=dropout)
            )
        else:
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, concat=True,
                        dropout=dropout)
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels, heads=heads,
                            concat=True, dropout=dropout)
                )
            self.convs.append(
                GATConv(hidden_channels * heads, out_channels, heads=1,
                        concat=False, dropout=dropout)
            )

    def forward(self, x, edge_index, return_attention=False):
        attention = None
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            if i == 0 and return_attention:
                x, (edge_index_out, alpha) = conv(
                    x, edge_index, return_attention_weights=True
                )
                attention = (edge_index_out, alpha)
            else:
                x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return (x, attention) if return_attention else x

    def embed(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.convs[0](x, edge_index))
        return x


def build_model(name, in_channels, out_channels, cfg):
    common = dict(
        hidden_channels=cfg.get("hidden", 16),
        num_layers=cfg.get("layers", 2),
        dropout=cfg.get("dropout", 0.5),
    )
    if name == "mlp":
        return MLP(in_channels, out_channels=out_channels, **common)
    if name == "gcn":
        return GCN(in_channels, out_channels=out_channels, **common)
    if name == "gat":
        return GAT(in_channels, out_channels=out_channels,
                   heads=cfg.get("heads", 8), **common)
    raise ValueError(f"未知模型: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
