"""SST-2 情感二分类：MLP 超参数网格搜索 + 重复 K 折交叉验证。

搜索空间 hidden_size [64,128,256] x dropout [0,0.3,0.5] x lr [1e-3,5e-3]
        x weight_decay [0,1e-4]，共 36 组
特征工程 TF-IDF (1-2 gram) 稀疏特征，词表与 IDF 在每折训练集内部拟合
训练策略 Adam + ReduceLROnPlateau + 早停（折内再切一份验证集专供早停）
评估指标 Accuracy / Precision / Recall / F1 (macro) / 混淆矩阵

本文件只负责编排流程与输出，实现分在 hs_core / hs_data / hs_model / hs_viz
四个模块里；两个辅助脚本 efficiency_bench.py、ablation_features.py 也复用它们。
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from hs_core import (
    ROOT,
    CLASS_NAMES,
    banner,
    format_combo,
    pick_device,
    save_cv_csv,
    sub_banner,
    summarize,
    to_per_fold_rows,
)
from hs_data import build_feature_matrices, build_folds, describe_dataset, load_sst2
from hs_model import MLP, build_grid, expand_grid, parallel_search, sparse_batch
from hs_viz import (
    plot_boxplot,
    plot_confusion,
    plot_heatmap,
    plot_loss_curves,
    plot_marginal_effects,
    plot_ranked_bar,
    setup_plot_style,
)


# --------------------------------------------------------------------------
# 最终模型：全量重训与测试集评估
# --------------------------------------------------------------------------
def refit_and_test(texts, labels, test_texts, test_labels, best, epochs, cfg):
    """用最佳组合在完整训练集上重训，并在测试集上评估。

    轮数固定为交叉验证中各折最优轮的平均值，不再早停：训练集已全部用上，
    没有多余数据可留作早停集，硬切一份出来只会白白少训样本。
    """
    sub_banner(f"最终重训：{format_combo(best)}，固定 epoch={epochs}，训练集全量 {len(texts)} 句")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    X_train, X_test, vec = build_feature_matrices(texts, test_texts, cfg)
    print(f"  特征维度={X_train.shape[1]} 词表={len(vec.vocabulary_)} "
          f"非零特征/句={X_train.nnz / X_train.shape[0]:.1f}")

    device = pick_device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    model = MLP(X_train.shape[1], best["hidden_size"], dropout=best["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"],
                                weight_decay=best["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    y = torch.from_numpy(labels).to(device)
    n = X_train.shape[0]
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.default_rng(cfg["seed"] + epoch).permutation(n)
        total = 0.0
        for start in range(0, n, cfg["batch_size"]):
            idx = perm[start:start + cfg["batch_size"]]
            optimizer.zero_grad()
            logits = model(sparse_batch(X_train, idx, device))
            loss = criterion(logits, y[torch.from_numpy(idx).to(device)])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history.append(total / n)
        print(f"    epoch {epoch:3d}/{epochs}  训练损失={history[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(sparse_batch(X_test, np.arange(X_test.shape[0]), device))
    preds = logits.argmax(dim=1).cpu().numpy()
    print(f"  重训耗时 {time.time() - t0:.1f}s")
    return model, preds, history


def print_test_report(test_labels, preds):
    """打印测试集指标。宏平均对两个类别同等加权，避免被多数类带走。"""
    acc = accuracy_score(test_labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="macro", zero_division=0
    )
    print(f"\n  测试集准确率 : {acc:.4f}")
    print(f"  宏平均 Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
    print()
    print(classification_report(test_labels, preds, target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    return {"accuracy": float(acc), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f1)}


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "demo_labs" / "SST-2")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--min-df", type=int, default=3)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=2,
                   help="每个训练进程固定的 CPU 线程数，串行与并行两支一致")
    p.add_argument("--bench-combos", type=int, default=8,
                   help="串行/并行耗时对比所用的组合数，设为 0 则跳过对比")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="小网格冒烟测试")
    return p.parse_args()


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = {
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "tfidf": {"ngram_range": (1, 2), "min_df": args.min_df,
                  "max_features": args.max_features, "sublinear_tf": True},
    }
    figures_dir = args.out_dir / "figures"
    results_dir = args.out_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    program_start = time.time()
    banner("SST-2 情感二分类：MLP 超参数网格搜索 + 重复 K 折交叉验证")
    print(f"  Python        : {sys.version.split()[0]} ({platform.system()})")
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  计算设备      : {pick_device(args.device)}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available()
             and pick_device(args.device).type == "cuda" else ""))
    print(f"  随机种子      : {args.seed}")
    print(f"  折数设置      : {args.n_splits} 折 x {args.n_repeats} 次重复 = "
          f"{args.n_splits * args.n_repeats} 个训练/验证划分")

    banner("数据集", char="-")
    texts, labels = load_sst2(args.data_dir / "train.tsv")
    test_texts, test_labels = load_sst2(args.data_dir / "test.tsv")
    describe_dataset("训练集", labels, np.array([len(t.split()) for t in texts]))
    describe_dataset("测试集", test_labels, np.array([len(t.split()) for t in test_texts]))

    combos = expand_grid(build_grid(args.quick))
    banner("超参数搜索空间", char="-")
    for key, values in build_grid(args.quick).items():
        print(f"  {key:<14}: {values}")
    total_fits = len(combos) * args.n_splits * args.n_repeats
    print(f"  组合数        : {len(combos)}")
    print(f"  预计训练次数  : {len(combos)} x {args.n_splits * args.n_repeats} "
          f"= {total_fits}")

    banner("构建交叉验证划分与折内特征", char="-")
    print("  说明：TF-IDF 的词表与 IDF 只在本折训练集上拟合，再变换验证折；")
    print("        早停使用折内再切出的 10% 验证集，与外层评估折严格分离。")
    t0 = time.time()
    folds = build_folds(texts, labels, cfg)
    print(f"  划分与特征构建总耗时 {time.time() - t0:.2f}s，"
          f"外层验证样本合计 {sum(len(f['y_val']) for f in folds)}")

    timing = None
    if args.bench_combos > 1:
        bench = combos[:args.bench_combos]
        bench_folds = [f for f in folds if f["repeat"] == 0]
        banner("串行与并行搜索耗时对比", char="-")
        print(f"  在 {len(bench)} 个组合 x {len(bench_folds)} 折 上分别执行串行与并行搜索")
        print(f"  两支均固定 torch 线程数 = {args.torch_threads}，"
              f"因此加速比只反映进程级并行")
        _, serial_wall = parallel_search(bench, bench_folds, cfg, 1, "串行搜索")
        _, par_wall = parallel_search(bench, bench_folds, cfg, args.jobs, "并行搜索")
        timing = {"combos": len(bench), "folds": len(bench_folds),
                  "serial_seconds": serial_wall, "parallel_seconds": par_wall,
                  "jobs": args.jobs, "speedup": serial_wall / par_wall}
        print(f"  >> 加速比 {timing['speedup']:.2f}x "
              f"(串行 {serial_wall:.2f}s -> 并行 {par_wall:.2f}s, n_jobs={args.jobs})")

    banner("网格搜索（完整搜索空间，并行执行）", char="-")
    records, search_wall = parallel_search(combos, folds, cfg, args.jobs, "完整网格搜索")

    summaries = [summarize(r) for r in records]
    summaries.sort(key=lambda r: r["mean_accuracy"], reverse=True)
    by_key = {tuple(sorted(r["combo"].items())): r for r in records}
    best_summary = summaries[0]
    best_combo = {k: best_summary[k] for k in ("hidden_size", "dropout", "lr", "weight_decay")}
    best_record = by_key[tuple(sorted(best_combo.items()))]

    banner("交叉验证结果排名（按平均准确率降序）", char="-")
    print(f"  {'#':>3}  {'hidden':>7} {'dropout':>8} {'lr':>8} {'weight_decay':>13} "
          f"{'准确率':>10} {'标准差':>9} {'F1(macro)':>10} {'最优epoch':>9}")
    print(f"  {'-' * 84}")
    for i, row in enumerate(summaries, 1):
        print(f"  {i:>3}  {row['hidden_size']:>7} {row['dropout']:>8.1f} "
              f"{row['lr']:>8.0e} {row['weight_decay']:>13.0e} "
              f"{row['mean_accuracy']:>10.4f} {row['std_accuracy']:>9.4f} "
              f"{row['mean_f1']:>10.4f} {row['mean_best_epoch']:>9.1f}")

    banner("最佳超参数组合", char="-")
    print(f"  {format_combo(best_combo)}")
    print(f"  交叉验证平均准确率 : {best_summary['mean_accuracy']:.4f} "
          f"± {best_summary['std_accuracy']:.4f}")
    print(f"  交叉验证宏平均 F1  : {best_summary['mean_f1']:.4f} "
          f"± {best_summary['std_f1']:.4f}")
    fold_accs = "  ".join(f"{f['accuracy']:.4f}" for f in best_record["folds"])
    print(f"  各折准确率         : [{fold_accs}]")

    worst = summaries[-1]
    print(f"\n  对照：最差组合 {format_combo({k: worst[k] for k in best_combo})}")
    print(f"        交叉验证平均准确率 {worst['mean_accuracy']:.4f}")
    print(f"        最佳组合相对最差组合提升 "
          f"{(best_summary['mean_accuracy'] - worst['mean_accuracy']) * 100:.2f} 个百分点")

    epochs = max(5, int(round(best_summary["mean_best_epoch"])))
    model, test_preds, _ = refit_and_test(texts, labels, test_texts, test_labels,
                                          best_combo, epochs, cfg)

    banner("测试集评估（最佳组合，完整训练集重训）", char="-")
    test_metrics = print_test_report(test_labels, test_preds)

    banner("可视化输出", char="-")
    setup_plot_style()
    plot_ranked_bar(summaries, figures_dir / "01_组合排名.png")
    plot_boxplot(summaries, figures_dir / "02_top12_箱线图.png")
    plot_heatmap(summaries, figures_dir / "03_hidden_dropout_热力图.png")
    plot_marginal_effects(summaries, figures_dir / "04_学习率与权重衰减_边际影响.png")
    plot_loss_curves(best_record, figures_dir / "05_最佳组合损失曲线.png")
    cm = plot_confusion(test_labels, test_preds, figures_dir / "06_测试集混淆矩阵.png",
                        f"测试集混淆矩阵（{format_combo(best_combo)}）")
    for path in sorted(figures_dir.glob("*.png")):
        print(f"  已保存 {path.relative_to(args.out_dir)}")

    banner("结果落盘", char="-")
    summary_csv, fold_csv = save_cv_csv(summaries, to_per_fold_rows(records),
                                        results_dir / "cv_summary.csv")
    torch.save(model.state_dict(), results_dir / "best_model.pt")
    payload = {
        "best_combo": {k: (float(v) if isinstance(v, float) else v)
                       for k, v in best_combo.items()},
        "cv_mean_accuracy": best_summary["mean_accuracy"],
        "cv_std_accuracy": best_summary["std_accuracy"],
        "cv_mean_f1_macro": best_summary["mean_f1"],
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "final_epochs": epochs,
        "timing": timing,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "grid": build_grid(args.quick),
        "total_fits": total_fits,
    }
    with open(results_dir / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    for path in (summary_csv, fold_csv, results_dir / "best_model.pt",
                 results_dir / "best_params.json"):
        print(f"  已保存 {path.relative_to(args.out_dir)}")

    banner("完成")
    print(f"  完整网格搜索墙钟耗时 : {search_wall:.2f}s")
    print(f"  程序总耗时           : {time.time() - program_start:.2f}s")


if __name__ == "__main__":
    main()
