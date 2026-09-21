"""稀疏输入喂法的效率基准与批量大小敏感性实验（报告 4.6 节）。

比较 COO / CSR / 稠密三种喂法单轮耗时，拆分「取批数据」与「前向反向」的
开销，并考察批量大小对准确率的影响。
用法：.venv/bin/python -u lab1/efficiency_bench.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs_core import banner, pick_device, silence_sparse_warnings
from hs_data import build_folds, load_sst2
from hs_model import sparse_batch, train_fold

ROOT = Path(__file__).resolve().parent.parent

# 基准用的折规模与主程序一致：5 折下每折约 48491 句训练、特征维 50000
DEFAULT_N = 48491
DEFAULT_DIM = 50000
DEFAULT_NNZ = 14


class BenchNet(nn.Module):
    """与主程序同构的单隐层 MLP，但输入格式由调用方决定。"""

    def __init__(self, input_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        if x.is_sparse:
            h = torch.sparse.mm(x, self.fc1.weight.t()) + self.fc1.bias
        else:
            h = self.fc1(x)
        return self.fc2(torch.relu(h))


def synth_fold(n, dim, nnz, seed=0):
    """造一份与真实折同规模、同稀疏度的 CSR 特征矩阵。

    不读真实数据是为了让基准只反映「喂法」的差异，不被 TF-IDF 拟合时间干扰；
    稀疏度与维度都按实测值设定，因此单次矩阵乘的规模与真实情形一致。
    """
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), nnz)
    cols = rng.integers(0, dim, size=rows.size)
    vals = rng.random(rows.size).astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, dim), dtype=np.float32)


def make_feeders(csr, device):
    """构造三种「取一批数据」的实现，返回同一个批次张量。"""

    def coo(idx):
        sub = csr[idx]
        c = sub.tocoo()
        return torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack([c.row, c.col]).astype(np.int64)),
            torch.from_numpy(c.data.astype(np.float32, copy=False)),
            size=c.shape,
        ).to(device)

    def csr_fmt(idx):
        return sparse_batch(csr, idx, device)

    def dense(idx):
        sub = csr[idx]
        return torch.from_numpy(sub.toarray()).to(device)

    return {"COO": coo, "CSR": csr_fmt, "稠密": dense}


def train_one_epoch(net, optimizer, criterion, fetch, y_all, n, bs, perm, device):
    """跑一轮：按给定顺序遍历全部数据。"""
    for start in range(0, (n // bs) * bs, bs):
        idx = perm[start:start + bs]
        xb = fetch(idx)
        yb = y_all[torch.from_numpy(idx).to(device)]
        loss = criterion(net(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def time_epochs(net, fetch, n, bs, device, epochs=3, seed=0):
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    y_all = torch.randint(0, 2, (n,), device=device)
    d = torch.device(device)

    # 预热一次，排除首次显存分配与算子编译的影响
    train_one_epoch(net, optimizer, criterion, fetch, y_all, n, bs,
                    np.random.default_rng(seed).permutation(n), d)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for e in range(epochs):
        perm = np.random.default_rng(seed + e + 1).permutation(n)
        train_one_epoch(net, optimizer, criterion, fetch, y_all, n, bs, perm, d)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / epochs * 1000, n // bs


def bench_formats(n, dim, nnz, device, batch_sizes, hidden):
    print("\n" + "=" * 78)
    print(f"  一、稀疏输入格式与批量大小对单轮训练耗时的影响（hidden={hidden}）")
    print("=" * 78)
    print(f"  折规模 {n} 句 x {dim} 维，非零特征/句={nnz}，"
          f"设备={device}，第一层参数 {dim * hidden / 1e6:.1f}M")

    csr = synth_fold(n, dim, nnz)
    rows = []
    for bs in batch_sizes:
        feeders = make_feeders(csr, device)
        for name, fetch in feeders.items():
            net = BenchNet(dim, hidden).to(device)
            ms, iters = time_epochs(net, fetch, n, bs, device)
            rows.append((name, bs, ms, iters))
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    base = rows[0][2]
    print(f"\n  {'格式':<6}{'批量':>7}{'每轮迭代数':>11}{'每轮耗时':>12}{'单次迭代':>11}{'相对':>8}")
    print("  " + "-" * 56)
    for name, bs, ms, iters in rows:
        print(f"  {name:<6}{bs:>7}{iters:>11}{ms:>10.1f}ms"
              f"{ms / iters:>9.2f}ms{base / ms:>7.2f}x")
    return rows


def bench_components(n, dim, nnz, device, bs, hidden):
    print("\n" + "=" * 78)
    print(f"  二、单次迭代的耗时构成（hidden={hidden}，批量={bs}）")
    print("=" * 78)
    csr = synth_fold(n, dim, nnz)
    net = BenchNet(dim, hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    y_all = torch.randint(0, 2, (n,), device=device)
    perm = np.random.default_rng(0).permutation(n)
    iters = n // bs

    def slice_only():
        for s in range(0, iters * bs, bs):
            csr[perm[s:s + bs]].tocoo()

    def fetch_only():
        for s in range(0, iters * bs, bs):
            sparse_batch(csr, perm[s:s + bs], device)

    batches = [sparse_batch(csr, perm[s:s + bs], device)
               for s in range(0, iters * bs, bs)]

    def step_only():
        for b in batches:
            loss = criterion(net(b), y_all[:bs])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def full():
        for s in range(0, iters * bs, bs):
            b = sparse_batch(csr, perm[s:s + bs], device)
            loss = criterion(net(b), y_all[:bs])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def timeit(fn, reps=3):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps / iters * 1000

    print(f"  批量={bs}，每轮 {iters} 次迭代")
    parts = [
        ("scipy 行切片（纯 CPU）", timeit(slice_only)),
        ("取一批数据（切片+转 CSR+搬运）", timeit(fetch_only)),
        ("模型步（前向+反向+参数更新）", timeit(step_only)),
        ("两者合计（真实训练）", timeit(full)),
    ]
    for label, ms in parts:
        print(f"    {label:<30}{ms:>8.3f} ms/迭代")
    return parts


def probe_batch_accuracy(args, batch_sizes, lrs):
    """在单个折上比较不同批量大小的验证准确率与耗时。"""
    print("\n" + "=" * 78)
    print("  三、批量大小敏感性：准确率与耗时")
    print("=" * 78)
    texts, labels = load_sst2(args.data_dir / "train.tsv")
    cfg = {
        "seed": 42, "epochs": args.epochs, "patience": 5,
        "n_splits": args.n_splits, "n_repeats": 1,
        "torch_threads": args.torch_threads, "device": args.device,
        "batch_size": batch_sizes[0],
        "tfidf": {"ngram_range": (1, 2), "min_df": args.min_df,
                  "max_features": args.max_features, "sublinear_tf": True},
    }
    folds = build_folds(texts, labels, cfg)
    fold = folds[0]
    device = pick_device(args.device)

    print(f"\n  在折 0 上固定 hidden=256 dropout=0.0 weight_decay=0.0，"
          f"最大 {args.epochs} 轮，早停 patience=5")
    print(f"\n  {'批量':>7}{'学习率':>9}{'最优轮':>8}{'验证准确率':>12}"
          f"{'每折耗时':>11}{'相对耗时':>10}")
    print("  " + "-" * 57)
    ref = None
    for bs in batch_sizes:
        for lr in lrs:
            c = dict(cfg, batch_size=bs)
            combo = {"hidden_size": 256, "dropout": 0.0,
                     "lr": lr, "weight_decay": 0.0}
            r = train_fold(combo, fold, c, device)
            if ref is None:
                ref = r["train_seconds"]
            print(f"  {bs:>7}{lr:>9.0e}{r['best_epoch']:>8}{r['accuracy']:>12.4f}"
                  f"{r['train_seconds']:>10.1f}s{ref / r['train_seconds']:>9.2f}x")


def parse_args():
    p = argparse.ArgumentParser(description="稀疏输入喂法的效率基准")
    p.add_argument("--data-dir", type=Path, default=ROOT / "demo_labs" / "SST-2")
    p.add_argument("--device", default="auto")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--nnz", type=int, default=DEFAULT_NNZ)
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--min-df", type=int, default=3)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64,
                   help="第一层隐藏单元数，决定第一层参数量")
    p.add_argument("--only", choices=["formats", "components", "probe"],
                   help="只跑其中一节")
    p.add_argument("--skip-probe", action="store_true",
                   help="跳过会真正读数据的批量大小敏感性实验")
    return p.parse_args()


def main():
    args = parse_args()
    silence_sparse_warnings()
    torch.set_num_threads(args.torch_threads)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) \
        else args.device
    if device == "auto":
        device = "cpu"

    banner("稀疏输入喂法效率基准")
    print(f"  计算设备      : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"  CPU 线程数    : {args.torch_threads}")

    batch_sizes = [256, 512, 1024, 2048, 4096]
    if args.only in (None, "formats"):
        bench_formats(args.n, args.dim, args.nnz, device, batch_sizes, args.hidden)
    if args.only in (None, "components"):
        bench_components(args.n, args.dim, args.nnz, device, 2048, args.hidden)
    if args.only in (None, "probe") and not args.skip_probe:
        probe_batch_accuracy(args, [256, 512, 1024, 2048], [1e-3, 5e-3])

    if torch.cuda.is_available():
        print(f"\n  显存峰值      : "
              f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print("\n  完成。")


if __name__ == "__main__":
    main()
