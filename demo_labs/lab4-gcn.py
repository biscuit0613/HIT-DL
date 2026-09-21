# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:35:34 2026

"""

"""
最简单的 GCN 节点分类实验 —— 基于 PyTorch Geometric
数据集：Cora 引文网络
模型：两层 GCN
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 -i https://mirrors.aliyun.com/pypi/simple/
pip uninstall -y transformers
pip install "transformers==4.46.3" -i https://mirrors.aliyun.com/pypi/simple/
pip uninstall numpy
pip install numpy==1.26.4


"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# ==================== 1. 数据加载 ====================
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]  # Cora 只有一张图

print(f"数据集: {dataset}")
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"特征维度: {dataset.num_features}")
print(f"类别数: {dataset.num_classes}")
print(f"训练节点: {data.train_mask.sum().item()}")
print(f"验证节点: {data.val_mask.sum().item()}")
print(f"测试节点: {data.test_mask.sum().item()}")


# ==================== 2. 模型定义 ====================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 第一层 GCN + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # 第二层 GCN（输出 logits）
        x = self.conv2(x, edge_index)
        return x


# ==================== 3. 训练配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes
).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# ==================== 4. 训练与评估 ====================
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum()
        accs.append(correct / mask.sum())

    return accs


# ==================== 5. 主训练循环 ====================
print("\n开始训练...")
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")

print(f"\n最终测试准确率: {test_acc:.4f}")