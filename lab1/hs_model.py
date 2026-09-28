"""MLP 模型、单折训练循环、超参数网格与并行搜索。"""

import time
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from hs_core import format_combo, pick_device, silence_sparse_warnings, sub_banner


class MLP(nn.Module):
    """单隐层多层感知机。第一层以稀疏矩阵乘法实现，输入无需稠密化。"""

    def __init__(self, input_dim, hidden_size, num_classes=2, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)  # 降维：(2048, 50000) -> (2048, hidden)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 已知冗余：is_sparse 只对 COO 为真，CSR 要查 is_sparse_csr，所以传入
        # CSR 时走的是下面的 else 支。结果与稀疏乘法一致且同样不做稠密化——
        # nn.Linear 对 CSR 输入会分发到稀疏实现，这一支保留是为了兼容 COO。
        if x.is_sparse:
            hidden = torch.sparse.mm(x, self.fc1.weight.t()) + self.fc1.bias
        else:
            hidden = self.fc1(x)
        return self.fc2(self.dropout(self.act(hidden)))


def sparse_batch(csr, idx, device):
    """把一个批次的行转成 CSR 稀疏张量并搬到设备上。

    一批是 2048 x 50000，但平均每句只有 14.2 个非零特征，稀疏度 99.97%，
    稠密存要 409 MB。CSR 用行指针 indptr 取代 COO 的逐元素行号，第 i 行的
    数据落在 col[indptr[i]:indptr[i+1]]，定位是 O(1) 查表；稀疏矩阵乘法要
    逐行乘权重，这一点让 CSR 比 COO 快约 1.3 倍。scipy 与 torch 的 CSR 布局
    一致，下面三个数组是原样搬运，没有格式转换。

    .to(device) 放在最后：先在 CPU 侧构造好再整体搬，若先搬再做
    to_sparse_csr() 需要设备端重排，实测多花 3 倍时间。

    另外取数据只占单次迭代的 7.3%，九成时间在模型步——真正的提速来自把
    批量提到 2048 以减少迭代次数，格式只贡献其中一小部分（见 4.6 节）。
    """
    sub = csr[idx]
    return torch.sparse_csr_tensor(
        torch.from_numpy(sub.indptr.astype(np.int64)),
        torch.from_numpy(sub.indices.astype(np.int64)),
        torch.from_numpy(sub.data.astype(np.float32, copy=False)),
        size=sub.shape,
    ).to(device)


def train_fold(combo, fold, cfg, device):
    """在单个折上训练一个组合，早停后在外层验证折上评估。"""
    torch.manual_seed(cfg["seed"] + fold["fold"])

    input_dim = fold["X_inner"].shape[1]
    model = MLP(input_dim, combo["hidden_size"], dropout=combo["dropout"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=combo["lr"], weight_decay=combo["weight_decay"]
    )
    # 调度器和早停盯同一份内层验证损失。学习率减半后验证损失还能再降一段，
    # 所以早停的 patience 必须比调度器的宽，否则刚降完学习率就被停掉。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    y_inner = torch.from_numpy(fold["y_inner"]).to(device)
    y_early = torch.from_numpy(fold["y_early"]).to(device)

    n = fold["X_inner"].shape[0]
    history = {"train_loss": [], "val_loss": []}
    best_loss, best_epoch, best_state = float("inf"), 0, None
    t0 = time.time()

    early_idx = np.arange(len(fold["y_early"]))
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = np.random.default_rng(cfg["seed"] + epoch).permutation(n)
        total = 0.0
        for start in range(0, n, cfg["batch_size"]):
            idx = perm[start:start + cfg["batch_size"]]
            optimizer.zero_grad()
            logits = model(sparse_batch(fold["X_inner"], idx, device))
            loss = criterion(logits, y_inner[torch.from_numpy(idx).to(device)])
            loss.backward()
            optimizer.step()
            # 按批内样本数加权，最后一个不满批不会把均值拉偏，
            # 损失曲线也因此与批量大小无关。
            total += loss.item() * len(idx)

        # eval() 关掉 dropout，否则早停看到的是随机丢弃后的损失，与最终评估
        # 口径不一致，选出的轮次会失真。早停集一次整批前向，每轮开销很小。
        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(sparse_batch(fold["X_early"], early_idx, device)), y_early
            ).item()
        train_loss = total / n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        # 训练损失只会单调下降，判断不了泛化，所以要另拿一份没参与训练的数据
        # 当探针。1e-5 是容差：浮点噪声级别的"改善"不算刷新纪录，否则每轮都在
        # 刷新，早停永远不触发。
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg["patience"]:
            break

    # 跳出时已经比最优轮多走了 patience 轮，权重是变差的，必须回滚。
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(sparse_batch(fold["X_val"], np.arange(len(fold["y_val"])), device))
    preds = logits.argmax(dim=1).cpu().numpy()
    y_true = fold["y_val"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "best_epoch": best_epoch,
        "train_seconds": time.time() - t0,
        "history": history,
        "repeat": fold["repeat"],
        "fold": fold["fold"],
        "preds": preds,
        "y_true": y_true,
    }


def run_combo(combo, folds, cfg):
    """在一个超参数组合上跑完整的所有折。

    每个工作进程固定使用的 CPU 线程数，使串行与并行两支的对比只反映
    进程级并行，而不会因为串行那支独占全部核心而失真。
    """
    silence_sparse_warnings()
    torch.set_num_threads(cfg["torch_threads"])
    device = pick_device(cfg["device"])
    results = [train_fold(combo, fold, cfg, device) for fold in folds]
    accs = np.array([r["accuracy"] for r in results])
    f1s = np.array([r["f1_macro"] for r in results])
    return {
        "combo": combo,
        "folds": results,
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "mean_f1": float(f1s.mean()),
        "std_f1": float(f1s.std()),
        "total_seconds": float(sum(r["train_seconds"] for r in results)),
    }


def parallel_search(combos, folds, cfg, n_jobs, label):
    """对一组超参数组合执行搜索，n_jobs=1 时为串行。"""
    sub_banner(f"{label}：{len(combos)} 个组合 x {len(folds)} 折（n_jobs={n_jobs}）")
    t0 = time.time()
    results = []
    try:
        stream = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(run_combo)(c, folds, cfg) for c in combos
        )
    except TypeError:
        stream = iter(
            Parallel(n_jobs=n_jobs)(delayed(run_combo)(c, folds, cfg) for c in combos)
        )

    for i, res in enumerate(stream, 1):
        results.append(res)
        print(f"  [{i:2d}/{len(combos)}] {format_combo(res['combo'])}  "
              f"acc={res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}  "
              f"f1={res['mean_f1']:.4f}  "
              f"训练耗时={res['total_seconds']:.1f}s")
    wall = time.time() - t0
    print(f"  {label} 完成，墙钟耗时 {wall:.2f}s，累计训练耗时 "
          f"{sum(r['total_seconds'] for r in results):.1f}s")
    return results, wall


# --------------------------------------------------------------------------
# 搜索空间
# --------------------------------------------------------------------------
def build_grid(quick=False):
    if quick:
        return {"hidden_size": [64], "dropout": [0.0, 0.3], "lr": [1e-3],
                "weight_decay": [0.0]}
    return {"hidden_size": [64, 128, 256], "dropout": [0.0, 0.3, 0.5],
            "lr": [1e-3, 5e-3], "weight_decay": [0.0, 1e-4]}


def expand_grid(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]
