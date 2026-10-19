"""生成实验报告所需的全部图表，并完成注意力与过平滑分析。"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from gcn_data import (CLASS_NAMES, dataset_stats, load_cora)
from gcn_model import build_model
from gcn_train import run_single

RESULTS = "results"
FIGURES = "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_LABELS = {"mlp": "MLP", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}
MODEL_ORDER = ["mlp", "gcn", "sage", "gat"]
MODEL_COLORS = {
    "mlp": "#7f7f7f",
    "gcn": "#1f77b4",
    "sage": "#2ca02c",
    "gat": "#d62728",
}
CLASS_COLORS = plt.cm.tab10(np.arange(len(CLASS_NAMES)))

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP",
                        "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_fig(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name)
    fig.savefig(path)
    plt.close(fig)
    print("保存", path)


def history_mean_std(runs, key):
    arr = np.array([r["history"][key] for r in runs], dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1)


def plot_dataset_stats(data, stats):
    deg = np.bincount(data.edge_index[0].cpu().numpy(),
                      minlength=data.num_nodes)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = stats["class_counts"]
    axes[0].bar(range(len(CLASS_NAMES)), counts, color=CLASS_COLORS)
    axes[0].set_xticks(range(len(CLASS_NAMES)))
    axes[0].set_xticklabels(CLASS_NAMES, rotation=22, ha="right", fontsize=8)
    axes[0].set_ylabel("节点数")
    axes[0].set_title("类别分布")
    for i, count in enumerate(counts):
        axes[0].text(i, count + 8, str(count), ha="center", fontsize=8)

    bins = np.logspace(0, np.log10(deg.max() + 1), 24)
    axes[1].hist(deg[deg > 0], bins=bins, color="#4c72b0", alpha=0.85)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("节点度（对数坐标）")
    axes[1].set_ylabel("节点数（对数坐标）")
    axes[1].set_title("度分布")
    axes[1].axvline(stats["degree_median"], color="tab:orange", ls="--",
                    label=f"中位数 {stats['degree_median']:.0f}")
    axes[1].axvline(stats["degree_mean"], color="tab:red", ls=":",
                    label=f"均值 {stats['degree_mean']:.2f}")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "dataset_overview.png")

    x = data.x.cpu().numpy()
    nz = (x > 0).sum(axis=1)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.hist(nz, bins=40, color="#55a868", alpha=0.88)
    ax.set_yscale("log")
    ax.set_xlabel("每个节点的非零特征数")
    ax.set_ylabel("节点数（对数坐标）")
    ax.set_title("节点特征稀疏度")
    ax.axvline(nz.mean(), color="tab:red", ls="--",
               label=f"均值 {nz.mean():.1f}")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "feature_sparsity.png")

    fig, ax = plt.subplots(figsize=(7.2, 4))
    values = stats["per_class_homophily"]
    ax.bar(range(len(CLASS_NAMES)), values, color=CLASS_COLORS)
    ax.axhline(stats["global_homophily"], color="black", ls="--",
               label=f"全体边同配率 {stats['global_homophily']:.3f}")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=22, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("同类邻居边占比")
    ax.set_title("各类节点的边同配性")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "homophily.png")


def plot_main_curves(main):
    runs = main["runs"]
    epochs = np.arange(1, len(runs["gcn"][0]["history"]["train_acc"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODEL_ORDER):
        for key, label, color in (
            ("train_acc", "训练集", "#9e9e9e"),
            ("val_acc", "验证集", "#1f77b4"),
            ("test_acc", "测试集", "#d62728"),
        ):
            mean, std = history_mean_std(runs[model], key)
            ax.plot(epochs, mean, label=label, color=color, lw=1.6)
            ax.fill_between(epochs, mean - std, mean + std, color=color,
                            alpha=0.15, lw=0)
        best_epochs = [r["best_epoch"] for r in runs[model]]
        ax.axvline(np.mean(best_epochs), color="k", ls=":", lw=1,
                   label=f"最优轮均值 {np.mean(best_epochs):.0f}")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("准确率")
    fig.suptitle("四种模型的训练/验证/测试准确率（10 个随机种子，阴影为 ±1 标准差）",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, "curves_accuracy.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in MODEL_ORDER:
        mean, std = history_mean_std(runs[model], "train_loss")
        axes[0].plot(epochs, mean, label=MODEL_LABELS[model],
                     color=MODEL_COLORS[model], lw=1.5)
        axes[0].fill_between(epochs, mean - std, mean + std,
                             color=MODEL_COLORS[model], alpha=0.12, lw=0)
    axes[0].set_title("训练损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("交叉熵")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for model in MODEL_ORDER:
        mean, std = history_mean_std(runs[model], "val_loss")
        axes[1].plot(epochs, mean, label=MODEL_LABELS[model],
                     color=MODEL_COLORS[model], lw=1.5)
        axes[1].fill_between(epochs, mean - std, mean + std,
                             color=MODEL_COLORS[model], alpha=0.12, lw=0)
    axes[1].set_title("验证损失")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("交叉熵")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "curves_loss.png")


def plot_comparison(main):
    summary = main["summary"]
    means = [summary[m]["test_acc"]["mean"] for m in MODEL_ORDER]
    stds = [summary[m]["test_acc"]["std"] for m in MODEL_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = axes[0].bar(range(len(MODEL_ORDER)), means, yerr=stds, capsize=5,
                       color=[MODEL_COLORS[m] for m in MODEL_ORDER], alpha=0.9)
    axes[0].set_xticks(range(len(MODEL_ORDER)))
    axes[0].set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    axes[0].set_ylim(0.4, 0.9)
    axes[0].set_ylabel("测试准确率")
    axes[0].set_title("测试准确率（10 种子均值 ± 标准差）")
    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width() / 2, mean + std + 0.008,
                     f"{mean:.4f}", ha="center", fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    data = [[r["best_epoch"] for r in main["runs"][m]] for m in MODEL_ORDER]
    bp = axes[1].boxplot(data, tick_labels=[MODEL_LABELS[m] for m in MODEL_ORDER],
                         patch_artist=True, widths=0.55)
    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, (model, values) in enumerate(zip(MODEL_ORDER, data), start=1):
        axes[1].scatter(i + rng.uniform(-0.08, 0.08, len(values)), values,
                        s=14, color="k", alpha=0.7, zorder=3)
    axes[1].set_ylabel("验证集最优轮次")
    axes[1].set_title("最优 checkpoint 出现的轮次")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "comparison.png")


def plot_per_class_f1(main):
    runs = main["runs"]
    x = np.arange(len(CLASS_NAMES))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for offset, model in enumerate(MODEL_ORDER):
        arr = np.array([r["per_class_f1"] for r in runs[model]])
        mean, std = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        ax.bar(x + (offset - 1.5) * width, mean, width, yerr=std, capsize=2,
               label=MODEL_LABELS[model], color=MODEL_COLORS[model], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("测试集 F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("各类别测试 F1（10 种子均值 ± 标准差）")
    ax.legend(ncol=4, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "per_class_f1.png")


def plot_confusions(main):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, model in zip(axes.ravel(), MODEL_ORDER):
        confs = []
        for run in main["runs"][model]:
            c = np.array(run["confusion"], dtype=np.float64)
            c = c / np.maximum(c.sum(axis=1, keepdims=True), 1)
            confs.append(c)
        mean = np.mean(confs, axis=0)
        im = ax.imshow(mean, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                color = "white" if mean[i, j] > 0.55 else "black"
                ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels([f"C{i}" for i in range(len(CLASS_NAMES))])
        ax.set_yticklabels([f"C{i}" for i in range(len(CLASS_NAMES))])
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(MODEL_LABELS[model])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8,
                 label="按真实类别归一化的比例")
    fig.suptitle("测试集混淆矩阵（10 个种子平均；C0–C6 依次为 "
                 + " / ".join(CLASS_NAMES) + "）", fontsize=11)
    save_fig(fig, "confusion_all.png")


def plot_sweeps(sweeps):
    data = sweeps["sweeps"]
    specs = [
        ("hidden", "隐藏层维度", "sweep_hidden.png"),
        ("layers", "GCN 层数", "sweep_layers.png"),
        ("dropout", "Dropout 比例", "sweep_dropout.png"),
        ("lr", "学习率", "sweep_lr.png"),
        ("weight_decay", "权重衰减", "sweep_weight_decay.png"),
    ]
    for key, xlabel, filename in specs:
        entries = data[key]["entries"]
        values = [e["value"] for e in entries]
        test = np.array([e["test_mean"] for e in entries])
        test_std = np.array([e["test_std"] for e in entries])
        val = np.array([e["val_mean"] for e in entries])
        val_std = np.array([e["val_std"] for e in entries])
        x = np.arange(len(values))

        fig, ax = plt.subplots(figsize=(6.8, 4))
        ax.errorbar(x, test, yerr=test_std, marker="o", capsize=3,
                    color="#1f77b4", label="测试准确率")
        ax.errorbar(x, val, yerr=val_std, marker="s", capsize=3,
                    color="#ff7f0e", alpha=0.8, label="验证准确率")
        baseline = data[key]["baseline"]
        if baseline in values:
            ax.axvline(values.index(baseline), color="k", ls=":", lw=1,
                       label=f"基准值 {baseline}")
        if key == "weight_decay":
            tick_labels = {0.0: "0", 1e-5: "1e-5", 1e-4: "1e-4",
                           5e-4: "5e-4", 5e-3: "5e-3", 1e-2: "1e-2"}
            labels = [tick_labels.get(float(v), f"{v:g}") for v in values]
        else:
            labels = [f"{v:g}" for v in values]
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("准确率")
        ax.set_title(f"GCN 的{xlabel}扫描（5 个种子）")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        save_fig(fig, filename)

    grid_data = data["grid_lr_wd"]
    lrs, wds = grid_data["lrs"], grid_data["weight_decays"]
    matrix = np.array([[cell["test_mean"] for cell in row]
                       for row in grid_data["grid"]])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    for i in range(len(lrs)):
        for j in range(len(wds)):
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                    color="white" if matrix[i, j] < matrix.max() - 0.025 else "black",
                    fontsize=8)
    wd_labels = {0.0: "0", 1e-4: "1e-4", 5e-4: "5e-4", 5e-3: "5e-3"}
    ax.set_xticks(range(len(wds)))
    ax.set_xticklabels([wd_labels.get(float(v), f"{v:g}") for v in wds])
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([f"{v:g}" for v in lrs])
    ax.set_xlabel("权重衰减")
    ax.set_ylabel("学习率")
    ax.set_title("GCN 测试准确率：学习率 × 权重衰减（3 个种子）")
    fig.colorbar(im, ax=ax, label="测试准确率")
    fig.tight_layout()
    save_fig(fig, "sweep_grid_lr_wd.png")

    entries = data["gat_heads"]["entries"]
    values = [e["value"] for e in entries]
    test = np.array([e["test_mean"] for e in entries])
    test_std = np.array([e["test_std"] for e in entries])
    params = [e["param_count"] for e in entries]
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.errorbar(values, test, yerr=test_std, marker="o", capsize=4,
                color="#d62728", label="测试准确率")
    ax.axvline(data["gat_heads"]["baseline"], color="k", ls=":", lw=1,
               label="基准头数 8")
    ax.set_xlabel("注意力头数")
    ax.set_ylabel("测试准确率")
    ax.set_title("GAT 注意力头数扫描（5 个种子）")
    ax2 = ax.twinx()
    ax2.plot(values, params, marker="s", color="#7f7f7f", ls="--",
             label="参数量")
    ax2.set_ylabel("可训练参数量")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "gat_heads.png")


def plot_label_rate(label_results):
    rates = label_results["rates"]
    summary = label_results["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, ylabel in ((axes[0], "test_acc", "测试准确率"),
                               (axes[1], "test_f1_macro", "测试宏 F1")):
        for model in MODEL_ORDER:
            mean, std = [], []
            for rate in rates:
                runs = [r for r in label_results["records"]
                        if r["model"] == model and r["rate"] == rate]
                values = np.array([r[metric] for r in runs])
                mean.append(values.mean())
                std.append(values.std(ddof=1))
            mean, std = np.array(mean), np.array(std)
            ax.plot(rates, mean, marker="o", color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model])
            ax.fill_between(rates, mean - std, mean + std,
                            color=MODEL_COLORS[model], alpha=0.13, lw=0)
        ax.axvline(20, color="k", ls=":", lw=1)
        ax.text(21, ax.get_ylim()[0] + 0.02, "标准划分 ≈20/类", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(rates)
        ax.set_xticklabels([str(r) for r in rates])
        ax.set_xlabel("每类训练节点数（对数坐标）")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_title("标签数量对准确率的影响")
    axes[1].set_title("标签数量对宏 F1 的影响")
    fig.tight_layout()
    save_fig(fig, "label_rate.png")


def plot_robustness(robust):
    ratios = robust["ratios"]
    summary = robust["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model in MODEL_ORDER:
        mean = np.array([summary[model][str(r)]["mean"] for r in ratios])
        std = np.array([summary[model][str(r)]["std"] for r in ratios])
        axes[0].errorbar(ratios, mean, yerr=std, marker="o", capsize=3,
                         color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        if model != "mlp":
            base = np.array([summary["mlp"][str(r)]["mean"] for r in ratios])
            base_std = np.array([summary["mlp"][str(r)]["std"] for r in ratios])
            gain = mean - base
            gain_std = np.sqrt(std ** 2 + base_std ** 2)
            axes[1].errorbar(ratios, gain, yerr=gain_std, marker="o", capsize=3,
                             color=MODEL_COLORS[model],
                             label=f"{MODEL_LABELS[model]} − MLP")
    axes[0].set_xlabel("随机删除的无向边比例")
    axes[0].set_ylabel("测试准确率")
    axes[0].set_title("图结构退化时的节点分类性能")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("随机删除的无向边比例")
    axes[1].set_ylabel("相对 MLP 的准确率增益")
    axes[1].set_title("图结构带来的增益")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "robustness.png")


def plot_degree_buckets(data, main):
    logits_data = np.load(f"{RESULTS}/main/logits.npz")
    y = logits_data["y"]
    test_mask = logits_data["test_mask"]
    degree = np.bincount(data.edge_index[0].cpu().numpy(),
                         minlength=data.num_nodes)

    buckets = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]
    bucket_labels = ["1–2", "3–5", "6–10", "11–20", ">20"]
    test_idx = np.where(test_mask)[0]
    acc = {m: [] for m in MODEL_ORDER}
    counts = []
    for lo, hi in buckets:
        idx = test_idx[(degree[test_idx] >= lo) & (degree[test_idx] <= hi)]
        counts.append(len(idx))
        for model in MODEL_ORDER:
            arrays = [logits_data[f"{model}_seed{s}"] for s in main["seeds"]]
            mean_logits = np.mean(arrays, axis=0)
            pred = mean_logits[idx].argmax(axis=1)
            acc[model].append(float((pred == y[idx]).mean()))

    x = np.arange(len(buckets))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for offset, model in enumerate(MODEL_ORDER):
        axes[0].bar(x + (offset - 1.5) * width, acc[model], width,
                    label=MODEL_LABELS[model], color=MODEL_COLORS[model],
                    alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(bucket_labels, counts)],
                            fontsize=8)
    axes[0].set_xlabel("节点度区间")
    axes[0].set_ylabel("测试准确率")
    axes[0].set_title("不同度区间上的准确率（10 种子平均 logits）")
    axes[0].legend(ncol=4, loc="lower right")
    axes[0].grid(axis="y", alpha=0.25)

    for model in ("gcn", "sage", "gat"):
        delta = np.array(acc[model]) - np.array(acc["mlp"])
        axes[1].bar(x + (MODEL_ORDER.index(model) - 1.5) * width, delta, width,
                    label=f"{MODEL_LABELS[model]} − MLP",
                    color=MODEL_COLORS[model], alpha=0.9)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bucket_labels)
    axes[1].set_xlabel("节点度区间")
    axes[1].set_ylabel("相对 MLP 的准确率增益")
    axes[1].set_title("图结构收益按度分布")
    axes[1].legend(ncol=3, loc="upper right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "degree_buckets.png")


def plot_embeddings(data, best_models):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    for ax, model_name in zip(axes.ravel(), MODEL_ORDER):
        entry = best_models[model_name]
        model = build_model(model_name, data.num_features,
                            int(data.y.max()) + 1, entry["config"]).to(DEVICE)
        model.load_state_dict(entry["state_dict"])
        model.eval()
        with torch.no_grad():
            emb = model.embed(data.x, data.edge_index).cpu().numpy()
        tsne = TSNE(n_components=2, perplexity=30, init="pca",
                    random_state=42, max_iter=1000)
        z = tsne.fit_transform(emb)
        for c in range(len(CLASS_NAMES)):
            mask = data.y.cpu().numpy() == c
            ax.scatter(z[mask, 0], z[mask, 1], s=4, alpha=0.7,
                       color=CLASS_COLORS[c], label=CLASS_NAMES[c] if model_name == "mlp" else None)
        ax.set_title(f"{MODEL_LABELS[model_name]}（seed {entry['seed']}）")
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=8)
    fig.suptitle("第一层节点表示的 t-SNE 可视化", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save_fig(fig, "embeddings.png")


def dirichlet_energy(h, edge_index):
    h = h / (h.norm(dim=1, keepdim=True) + 1e-9)
    src, dst = edge_index
    diff = h[src] - h[dst]
    return float((diff.pow(2).sum(dim=1).mean() / 2).item())


def oversmoothing_analysis(data, sweeps):
    print("过平滑分析：重新训练 1–5 层 GCN ...")
    seeds = [0, 1, 2]
    records = []
    layer_energy = []
    for layers in range(1, 6):
        energies, accs = [], []
        for seed in seeds:
            cfg = dict(model="gcn", hidden=16, layers=layers, dropout=0.5,
                       lr=0.01, weight_decay=5e-4, epochs=200)
            run, artifacts = run_single(cfg, data, seed, DEVICE)
            model = build_model("gcn", data.num_features,
                                int(data.y.max()) + 1, cfg).to(DEVICE)
            model.load_state_dict(artifacts["state_dict"])
            model.eval()
            with torch.no_grad():
                logits, hidden = model(data.x, data.edge_index,
                                       return_hidden=True)
            h_final = hidden[-1] if hidden else logits
            energies.append(dirichlet_energy(h_final, data.edge_index))
            accs.append(run["test_acc"])
            if layers == 5:
                layer_energy.append([
                    dirichlet_energy(h, data.edge_index) for h in hidden
                ])
        records.append({
            "layers": layers,
            "energy_mean": float(np.mean(energies)),
            "energy_std": float(np.std(energies, ddof=1)),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)),
        })
        print(f"  {layers} 层: acc {np.mean(accs):.4f}, "
              f"energy {np.mean(energies):.4f}")

    layer_energy = np.array(layer_energy)
    layer_stats = {
        "layers": list(range(1, layer_energy.shape[1] + 1)),
        "mean": layer_energy.mean(axis=0).tolist(),
        "std": layer_energy.std(axis=0, ddof=1).tolist(),
    }
    save_json(f"{RESULTS}/oversmoothing.json",
              {"depths": records, "layer_energy_5layer": layer_stats})

    sweep_entries = sweeps["sweeps"]["layers"]["entries"]
    depth = [e["value"] for e in sweep_entries]
    acc = [e["test_mean"] for e in sweep_entries]
    acc_std = [e["test_std"] for e in sweep_entries]
    # 1 层 GCN 没有隐藏表示，能量曲线只画 2 层及以上
    energy_records = [r for r in records if r["layers"] >= 2]
    depth_energy = [r["layers"] for r in energy_records]
    energy = [r["energy_mean"] for r in energy_records]
    energy_std = [r["energy_std"] for r in energy_records]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.errorbar(depth, acc, yerr=acc_std, marker="o", capsize=3,
                color="#1f77b4", label="测试准确率")
    ax.set_xlabel("GCN 层数")
    ax.set_ylabel("测试准确率", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_title("层数与性能、末层隐藏表示平滑度")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.errorbar(depth_energy, energy, yerr=energy_std, marker="s", capsize=3,
                 color="#d62728", label="末层隐藏表示能量（≥2 层）")
    ax2.set_ylabel("归一化 Dirichlet 能量", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    axes[1].errorbar(layer_stats["layers"], layer_stats["mean"],
                     yerr=layer_stats["std"], marker="o", capsize=3,
                     color="#9467bd")
    axes[1].set_xlabel("5 层 GCN 中的隐藏层序号")
    axes[1].set_ylabel("归一化 Dirichlet 能量")
    axes[1].set_title("5 层 GCN 内逐层能量衰减")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "oversmoothing.png")


def attention_analysis(data, best_models):
    print("GAT 注意力分析 ...")
    entry = best_models["gat"]
    model = build_model("gat", data.num_features, int(data.y.max()) + 1,
                        entry["config"]).to(DEVICE)
    model.load_state_dict(entry["state_dict"])
    model.eval()
    with torch.no_grad():
        logits, (edge_index, alpha) = model(data.x, data.edge_index,
                                            return_attention=True)
    alpha = alpha.cpu().numpy()
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    weight = alpha.mean(axis=1)
    ei = edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    y = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    test_nodes = np.where(test_mask)[0]

    degree_all = np.bincount(data.edge_index[0].cpu().numpy(),
                             minlength=data.num_nodes)

    nonself = src != dst
    test_edge = nonself & test_mask[dst]
    same = (y[src] == y[dst]) & test_edge
    diff = (~(y[src] == y[dst])) & test_edge
    w_same, w_diff = weight[same], weight[diff]
    per_head_same = alpha[same].mean(axis=0)
    per_head_diff = alpha[diff].mean(axis=0)
    per_head_ratio = per_head_same / (per_head_diff + 1e-12)

    share, neighbor_same, entropy, norm_entropy, node_deg = [], [], [], [], []
    for node in test_nodes:
        mask = test_edge & (dst == node)
        w = weight[mask]
        if w.sum() <= 0:
            continue
        s = y[src[mask]] == y[node]
        p = w / w.sum()
        ent = float(-(p * np.log(p + 1e-12)).sum())
        deg = int(degree_all[node])
        share.append(float(w[s].sum() / w.sum()))
        neighbor_same.append(float(s.mean()))
        entropy.append(ent)
        norm_entropy.append(ent / np.log(deg + 1))
        node_deg.append(deg)
    share = np.array(share)
    neighbor_same = np.array(neighbor_same)
    entropy = np.array(entropy)
    norm_entropy = np.array(norm_entropy)
    node_deg = np.array(node_deg)

    per_class_share, per_class_neighbor = [], []
    for c in range(len(CLASS_NAMES)):
        mask = y[test_nodes] == c
        per_class_share.append(float(share[mask].mean()))
        per_class_neighbor.append(float(neighbor_same[mask].mean()))

    stats = {
        "n_test_edges": int(test_edge.sum()),
        "same_mean": float(w_same.mean()),
        "same_median": float(np.median(w_same)),
        "diff_mean": float(w_diff.mean()),
        "diff_median": float(np.median(w_diff)),
        "same_diff_ratio": float(w_same.mean() / (w_diff.mean() + 1e-12)),
        "per_head_same": per_head_same.tolist(),
        "per_head_diff": per_head_diff.tolist(),
        "per_head_ratio": per_head_ratio.tolist(),
        "per_class_attention_share": per_class_share,
        "per_class_neighbor_same_fraction": per_class_neighbor,
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(np.median(entropy)),
        "norm_entropy_mean": float(norm_entropy.mean()),
        "degree_entropy_corr": float(np.corrcoef(node_deg, entropy)[0, 1]),
        "degree_norm_entropy_corr": float(np.corrcoef(node_deg, norm_entropy)[0, 1]),
        "target_seed": int(entry["seed"]),
    }
    save_json(f"{RESULTS}/attention_stats.json", stats)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    bins = np.linspace(0, max(w_same.max(), w_diff.max()), 40)
    axes[0, 0].hist(w_same, bins=bins, density=True, alpha=0.6,
                    color="#1f77b4", label=f"同类邻居（均值 {w_same.mean():.3f}）")
    axes[0, 0].hist(w_diff, bins=bins, density=True, alpha=0.6,
                    color="#d62728", label=f"异类邻居（均值 {w_diff.mean():.3f}）")
    axes[0, 0].set_xlabel("第一层平均注意力权重")
    axes[0, 0].set_ylabel("概率密度")
    axes[0, 0].set_title("注意力权重的类内/类间分布")
    axes[0, 0].legend()

    heads = np.arange(1, len(per_head_same) + 1)
    axes[0, 1].bar(heads - 0.2, per_head_same, 0.4, color="#1f77b4",
                   label="同类邻居")
    axes[0, 1].bar(heads + 0.2, per_head_diff, 0.4, color="#d62728",
                   label="异类邻居")
    axes[0, 1].set_xlabel("注意力头")
    axes[0, 1].set_ylabel("平均注意力权重")
    axes[0, 1].set_title("各注意力头的类内/类间权重（比值轴固定 0.9–1.1）")
    twin = axes[0, 1].twinx()
    twin.plot(heads, per_head_ratio, marker="o", color="k", lw=1.2,
              label="同类/异类比值")
    twin.axhline(1.0, color="gray", ls="--", lw=1)
    twin.set_ylim(0.9, 1.1)
    twin.set_ylabel("同类/异类比值")
    lines1, labels1 = axes[0, 1].get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    axes[0, 1].legend(lines1 + lines2, labels1 + labels2, fontsize=8,
                      loc="upper right")

    x = np.arange(len(CLASS_NAMES))
    axes[1, 0].bar(x - 0.2, per_class_share, 0.4, color="#1f77b4",
                   label="注意力投向同类邻居的比例")
    axes[1, 0].bar(x + 0.2, per_class_neighbor, 0.4, color="#7f7f7f",
                   label="同类邻居在邻居中的比例")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"C{i}" for i in range(len(CLASS_NAMES))])
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xlabel("类别")
    axes[1, 0].set_ylabel("比例")
    axes[1, 0].set_title("各类测试节点的注意力同配性")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].scatter(node_deg, norm_entropy, s=9, alpha=0.35,
                       color="#9467bd")
    if len(node_deg) > 20:
        order = np.argsort(node_deg)
        chunks = np.array_split(order, 8)
        xs = [node_deg[c].mean() for c in chunks if len(c) > 0]
        ys = [norm_entropy[c].mean() for c in chunks if len(c) > 0]
        axes[1, 1].plot(xs, ys, marker="o", color="k", lw=1.2,
                        label="按度分箱均值")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("节点度（对数坐标）")
    axes[1, 1].set_ylabel("归一化注意力熵")
    axes[1, 1].set_title("归一化熵与度的关系"
                         f"（r={stats['degree_norm_entropy_corr']:.2f}）")
    axes[1, 1].legend()
    fig.tight_layout()
    save_fig(fig, "attention_analysis.png")

    # 单个目标节点的径向注意力示意，优先选同类/异类邻居都有的节点
    best_node, best_score = None, -np.inf
    for node in test_nodes:
        deg = int(degree_all[node])
        if not 4 <= deg <= 15:
            continue
        mask = test_edge & (dst == node)
        w = weight[mask]
        if w.sum() <= 0:
            continue
        s = y[src[mask]] == y[node]
        if s.sum() == 0 or (~s).sum() == 0:
            continue
        score = float(w[s].mean() - w[~s].mean())
        if score > best_score:
            best_node, best_score = node, score
    if best_node is None:
        for node in test_nodes:
            deg = int(degree_all[node])
            if not 3 <= deg <= 15:
                continue
            mask = test_edge & (dst == node)
            w = weight[mask]
            if w.sum() > 0:
                s = y[src[mask]] == y[node]
                score = float(w[s].sum() / w.sum())
                if score > best_score:
                    best_node, best_score = node, score
    if best_node is None:
        return

    mask = test_edge & (dst == best_node)
    neighbors = src[mask]
    weights = weight[mask]
    order = np.argsort(-weights)
    neighbors, weights = neighbors[order], weights[order]
    true_class = y[best_node]
    pred_class = logits[best_node].argmax().item()
    n_same = int((y[neighbors] == true_class).sum())
    n_diff = len(neighbors) - n_same
    angles = np.linspace(0, 2 * np.pi, len(neighbors), endpoint=False)
    pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    max_w = weights.max()
    for (px, py), neighbor, w in zip(pos, neighbors, weights):
        ax.plot([0, px], [0, py], color=CLASS_COLORS[y[neighbor]],
                lw=0.6 + 5.5 * w / max_w, alpha=0.75)
    ax.scatter(pos[:, 0], pos[:, 1], s=95,
               c=[CLASS_COLORS[y[n]] for n in neighbors],
               edgecolors="white", linewidths=0.7, zorder=3)
    ax.scatter([0], [0], s=330, c=[CLASS_COLORS[true_class]],
               edgecolors="black", linewidths=1.4, zorder=4)
    ax.text(0, -1.3, f"目标 #{best_node}：真实 {CLASS_NAMES[true_class]}，"
            f"预测 {CLASS_NAMES[pred_class]}\n"
            f"邻居中同类 {n_same} 个、异类 {n_diff} 个；边宽 ∝ 注意力",
            ha="center", va="top", fontsize=8.5)
    present = sorted(set(y[neighbors].tolist()) | {true_class})
    handles = [plt.Line2D([], [], marker="o", ls="", color=CLASS_COLORS[c],
                          label=CLASS_NAMES[c]) for c in present]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.75, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"GAT 第一层对目标节点的注意力（{len(neighbors)} 个邻居）",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, "attention_ego.png")


def main():
    dataset, data_cpu = load_cora(normalize=True)
    _, data_raw = load_cora(normalize=False)
    data = data_cpu.to(DEVICE)
    stats = dataset_stats(data_raw)
    save_json(f"{RESULTS}/dataset_stats.json", stats)
    main_results = load_json(f"{RESULTS}/main/results.json")
    sweeps = load_json(f"{RESULTS}/sweeps/results.json")
    label_results = load_json(f"{RESULTS}/label_rate/results.json")
    robust = load_json(f"{RESULTS}/robustness/results.json")
    best_models = torch.load(f"{RESULTS}/main/best_models.pt",
                             map_location="cpu", weights_only=False)

    plot_dataset_stats(data_raw, stats)
    plot_main_curves(main_results)
    plot_comparison(main_results)
    plot_per_class_f1(main_results)
    plot_confusions(main_results)
    plot_sweeps(sweeps)
    plot_label_rate(label_results)
    plot_robustness(robust)
    plot_degree_buckets(data_cpu, main_results)
    plot_embeddings(data_cpu, best_models)
    oversmoothing_analysis(data, sweeps)
    attention_analysis(data, best_models)
    print("全部图表生成完毕。")


if __name__ == "__main__":
    main()
