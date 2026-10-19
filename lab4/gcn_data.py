"""Cora 数据加载、统计与实验划分。"""

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, remove_self_loops

# Planetoid 版本 Cora 的标签下标与类别名按各类样本数对应。
# 各类节点数：351 / 217 / 418 / 818 / 426 / 298 / 180。
CLASS_NAMES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Theory",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
]


def row_normalize(x):
    """按行和归一化节点特征，Cora 的标准预处理。"""
    return x / x.sum(dim=1, keepdim=True).clamp(min=1)


def load_cora(root="./data/Cora", normalize=True):
    """加载 Cora，去掉数据中已有的自环，并默认做特征行归一化。

    GCNConv / GATConv 默认会自行添加自环，若数据里再带一份会重复计算。
    """
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    data.edge_index, _ = remove_self_loops(data.edge_index)
    if normalize:
        data.x = row_normalize(data.x)
    return dataset, data


def dataset_stats(data):
    """汇总数据集规模、度分布、同配性与特征稀疏度。"""
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    y = data.y.cpu().numpy()
    deg = degree(edge_index[0], num_nodes, dtype=torch.long).cpu().numpy()

    same = (data.y[edge_index[0]] == data.y[edge_index[1]]).cpu().numpy()
    global_homophily = float(same.mean())
    per_class_homophily = []
    for c in range(len(CLASS_NAMES)):
        src_is_c = data.y[edge_index[0]].cpu().numpy() == c
        per_class_homophily.append(float(same[src_is_c].mean()))

    # 节点级同配性：每个节点邻居中同类的比例，再对节点取平均
    node_same = np.zeros(num_nodes, dtype=np.float64)
    node_deg = np.zeros(num_nodes, dtype=np.float64)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    np.add.at(node_same, dst, (y[src] == y[dst]).astype(np.float64))
    np.add.at(node_deg, dst, 1.0)
    node_homophily = float((node_same[node_deg > 0] / node_deg[node_deg > 0]).mean())

    adj = coo_matrix(
        (np.ones(edge_index.size(1)), (src, dst)), shape=(num_nodes, num_nodes)
    )
    n_components, labels = connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels)

    x = data.x.cpu().numpy()
    nonzero_per_node = (x > 0).sum(axis=1)

    stats = {
        "num_nodes": int(num_nodes),
        "num_directed_edges": int(edge_index.size(1)),
        "num_undirected_edges": int(edge_index.size(1) // 2),
        "num_features": int(data.num_features),
        "num_classes": int(len(CLASS_NAMES)),
        "class_names": CLASS_NAMES,
        "class_counts": np.bincount(y, minlength=len(CLASS_NAMES)).tolist(),
        "train_size": int(data.train_mask.sum()),
        "val_size": int(data.val_mask.sum()),
        "test_size": int(data.test_mask.sum()),
        "degree_min": int(deg.min()),
        "degree_median": float(np.median(deg)),
        "degree_mean": float(deg.mean()),
        "degree_p90": float(np.percentile(deg, 90)),
        "degree_p99": float(np.percentile(deg, 99)),
        "degree_max": int(deg.max()),
        "isolated_nodes": int((deg == 0).sum()),
        "num_components": int(n_components),
        "largest_component": int(comp_sizes.max()),
        "global_homophily": global_homophily,
        "node_homophily": node_homophily,
        "per_class_homophily": per_class_homophily,
        "feature_nonzero_ratio": float((x > 0).mean()),
        "nonzero_features_per_node_mean": float(nonzero_per_node.mean()),
        "nonzero_features_per_node_median": float(np.median(nonzero_per_node)),
        "nonzero_features_per_node_max": int(nonzero_per_node.max()),
        "edge_index_dtype": str(edge_index.dtype),
    }
    return stats


def make_masks(num_nodes, train_idx, val_idx, test_idx, device=None):
    masks = []
    for idx in (train_idx, val_idx, test_idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.as_tensor(np.asarray(idx), dtype=torch.long)] = True
        masks.append(mask if device is None else mask.to(device))
    return masks


def label_rate_masks(data, per_class, seed, val_size=500):
    """从非测试节点中每类抽 per_class 个作训练，剩余节点抽 val_size 个作验证。"""
    rng = np.random.default_rng(seed)
    y = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    pool = np.where(~test_mask)[0]

    train_idx = []
    for c in range(len(CLASS_NAMES)):
        candidates = pool[y[pool] == c]
        take = min(per_class, candidates.size)
        train_idx.extend(rng.choice(candidates, size=take, replace=False).tolist())
    train_idx = np.array(sorted(train_idx), dtype=np.int64)

    train_set = set(train_idx.tolist())
    remaining = np.array([i for i in pool if i not in train_set], dtype=np.int64)
    val_idx = rng.choice(remaining, size=min(val_size, remaining.size), replace=False)
    test_idx = np.where(test_mask)[0]

    return make_masks(data.num_nodes, train_idx, val_idx, test_idx)


def drop_undirected_edges(edge_index, ratio, seed):
    """按比例随机删除无向边，返回重新对称化后的边索引。"""
    if ratio <= 0:
        return edge_index
    undirected = edge_index[:, edge_index[0] < edge_index[1]]
    rng = np.random.default_rng(seed)
    keep = rng.random(undirected.size(1)) >= ratio
    undirected = undirected[:, torch.as_tensor(keep, device=edge_index.device)]
    return torch.cat([undirected, undirected.flip(0)], dim=1)
