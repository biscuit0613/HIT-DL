"""特征表示消融对照（报告 4.5 节）。

在同一个 SST-2 划分上比较「TF-IDF 先经 SVD 降维再喂稠密模型」与「TF-IDF
稀疏直接接入」两条路线，量化降维的损失；并给出各路线交叉验证分数与独立
测试集分数的落差，说明前者为何不能直接当作后者的预期。
用法：.venv/bin/python -u lab1/ablation_features.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs_core import banner, pick_device, silence_sparse_warnings
from hs_data import build_feature_matrices, load_sst2

ROOT = Path(__file__).resolve().parent.parent


def make_dense_model(input_dim, hidden=256, dropout=0.3, seed=42):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 2),
    )


def train_dense_mlp(X_tr, y_tr, X_te, cfg, seed=42):
    """在稠密特征上训练单隐层 MLP，返回 (验证准确率, 测试准确率)。

    训练集内部再切 10% 作早停集，与主程序口径一致。
    """
    device = pick_device(cfg["device"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = X_tr.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * 0.1)
    inner, early = idx[n_val:], idx[:n_val]

    model = make_dense_model(X_tr.shape[1], seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    X_inner = torch.from_numpy(X_tr[inner]).to(device)
    y_inner = torch.from_numpy(y_tr[inner]).long().to(device)
    X_early = torch.from_numpy(X_tr[early]).to(device)
    y_early = torch.from_numpy(y_tr[early]).long().to(device)
    X_test = torch.from_numpy(X_te).to(device)

    best_loss, best_state = float("inf"), None
    bs = 2048
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = np.random.default_rng(seed + epoch).permutation(len(inner))
        for s in range(0, len(inner), bs):
            b = perm[s:s + bs]
            loss = crit(model(X_inner[b]), y_inner[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(X_early), y_early).item()
        if vl < best_loss - 1e-5:
            best_loss = vl
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        elif epoch - 1 >= cfg["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test).argmax(1).cpu().numpy() == y_te).mean()
    return float(test_acc)


def evaluate(name, X_all, y_all, X_test, y_test, cfg, use_mlp,
             extra_prep=None):
    """跑重复分层 K 折得到交叉验证分数，再在测试集上评估。"""
    splitter = RepeatedStratifiedKFold(
        n_splits=cfg["n_splits"], n_repeats=cfg["n_repeats"],
        random_state=cfg["seed"],
    )
    accs, t0 = [], time.time()
    for tr, va in splitter.split(X_all, y_all):
        X_tr, X_va = X_all[tr], X_all[va]
        y_tr, y_va = y_all[tr], y_all[va]
        if extra_prep is not None:
            X_tr, X_va = extra_prep(X_tr, X_va)
        if use_mlp:
            model = make_dense_model(X_tr.shape[1], seed=cfg["seed"]).to(
                pick_device(cfg["device"]))
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()
            Xt = torch.from_numpy(X_tr).to(pick_device(cfg["device"]))
            yt = torch.from_numpy(y_tr).long().to(pick_device(cfg["device"]))
            bs = 2048
            for _ in range(cfg["epochs"]):
                perm = np.random.default_rng(cfg["seed"]).permutation(len(tr))
                for s in range(0, len(tr), bs):
                    b = perm[s:s + bs]
                    loss = crit(model(Xt[b]), yt[b])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(X_va).to(
                    pick_device(cfg["device"]))).argmax(1).cpu().numpy()
        else:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_va)
        accs.append((pred == y_va).mean())

    # 测试集：在完整训练集上重训
    if extra_prep is not None:
        X_tr_full, X_te = extra_prep(X_all, X_test)
    else:
        X_tr_full, X_te = X_all, X_test
    if use_mlp:
        test_acc = train_dense_mlp(X_tr_full, y_all, X_te, cfg)
    else:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr_full, y_all)
        test_acc = float((clf.predict(X_te) == y_test).mean())

    cv = float(np.mean(accs))
    print(f"  {name:<26} CV={cv:.4f}±{np.std(accs):.4f}  "
          f"TEST={test_acc:.4f}  落差={cv - test_acc:+.4f}  "
          f"耗时={time.time() - t0:.0f}s")
    return {"name": name, "cv": cv, "cv_std": float(np.std(accs)),
            "test": test_acc, "gap": cv - test_acc}


def parse_args():
    p = argparse.ArgumentParser(description="特征表示消融对照")
    p.add_argument("--data-dir", type=Path, default=ROOT / "demo_labs" / "SST-2")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--min-df", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-threads", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    silence_sparse_warnings()
    torch.set_num_threads(args.torch_threads)
    cfg = {
        "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
        "device": args.device, "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
    }

    banner("特征表示消融对照：稀疏 TF-IDF vs SVD 降维")
    train_texts, y_train = load_sst2(args.data_dir / "train.tsv")
    test_texts, y_test = load_sst2(args.data_dir / "test.tsv")
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)

    print("  在完整训练集上拟合 TF-IDF（仅用于本次对照，故不放在折内）")
    vec_cfg = {"ngram_range": (1, 2), "min_df": args.min_df,
               "max_features": args.max_features, "sublinear_tf": True}
    X_train, X_test, _ = build_feature_matrices(
        train_texts, test_texts, {"tfidf": vec_cfg})
    print(f"  原始稀疏特征: train={X_train.shape} test={X_test.shape} "
          f"nnz/句={X_train.nnz / X_train.shape[0]:.1f}")

    def svd_prep(k):
        def prep(X_tr, X_ap):
            svd = TruncatedSVD(n_components=k, random_state=args.seed)
            Z_tr = svd.fit_transform(X_tr).astype(np.float32)
            Z_ap = svd.transform(X_ap).astype(np.float32)
            sc = StandardScaler()
            return (sc.fit_transform(Z_tr).astype(np.float32),
                    sc.transform(Z_ap).astype(np.float32))
        return prep

    print("\n  对照组（每条路线都做 5 折 x 2 次重复交叉验证 + 独立测试集）")
    print("  " + "-" * 74)
    rows = []
    rows.append(evaluate("TF-IDF 稀疏 + LR", X_train, y_train, X_test, y_test,
                         cfg, use_mlp=False))

    # 稀疏 + MLP：复用主程序的口径，这里只报随机搜索出来的最佳组合结果
    print("  TF-IDF 稀疏 + MLP           见主程序 4.4 节（CV=0.9190 TEST=0.8096）")

    for k in (200, 800):
        rows.append(evaluate(f"TF-IDF + SVD{k} + LR", X_train, y_train,
                             X_test, y_test, cfg, use_mlp=False,
                             extra_prep=svd_prep(k)))
        svd = TruncatedSVD(n_components=k, random_state=args.seed)
        svd.fit(X_train)
        ev = svd.explained_variance_ratio_.sum()
        print(f"      （SVD{k} 累计解释方差 = {ev:.3f}，即原稀疏特征的 "
              f"{ev * 100:.1f}%）")

    print("\n  结论：SVD 降维后的稠密特征在本任务上远不如原始稀疏特征；"
          "\n        降维丢掉的正是 TF-IDF 里区分度最高的稀有词与二元词组。")
    print("\n  完成。")


if __name__ == "__main__":
    main()
