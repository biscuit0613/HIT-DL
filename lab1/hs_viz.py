"""六张结果图：组合排名、Top-12 箱线图、hidden×dropout 热力图、
学习率与权重衰减的边际影响、最佳组合损失曲线、测试集混淆矩阵。"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.metrics import confusion_matrix

from hs_core import CLASS_NAMES


def setup_plot_style():
    for font in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
                 "WenQuanYi Micro Hei", "DejaVu Sans"):
        try:
            import matplotlib.font_manager as fm

            if any(font == f.name for f in fm.fontManager.ttflist):
                plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
                break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def plot_ranked_bar(summaries, path):
    rows = sorted(summaries, key=lambda r: r["mean_accuracy"])
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(rows))
    ax.bar(xs, [r["mean_accuracy"] for r in rows],
           yerr=[r["std_accuracy"] for r in rows], capsize=2.5,
           color="#3b6ea5", edgecolor="#1f3f5f", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r['hidden_size']}/{r['dropout']:.1f}/{r['lr']:.0e}/{r['weight_decay']:.0e}"
         for r in rows], rotation=90, fontsize=6)
    ax.set_xlabel("组合 (hidden/dropout/lr/weight_decay)，按平均准确率升序")
    ax.set_ylabel("交叉验证平均准确率")
    ax.set_title(f"全部 {len(rows)} 个超参数组合的交叉验证表现（误差棒为各折标准差）")
    lo, hi = min(r["mean_accuracy"] for r in rows), max(r["mean_accuracy"] for r in rows)
    ax.set_ylim(max(0, lo - 0.02), min(1.0, hi + 0.012))
    ax.axhline(max(r["mean_accuracy"] for r in rows), color="#c0392b",
               linestyle="--", linewidth=0.9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_boxplot(summaries, path, top_n=12):
    rows = sorted(summaries, key=lambda r: r["mean_accuracy"], reverse=True)[:top_n]
    fig, ax = plt.subplots(figsize=(11, 5))
    data = [[f["accuracy"] for f in r["folds"]] for r in rows]
    labels = [f"h={r['hidden_size']} d={r['dropout']:.1f}\nlr={r['lr']:.0e} wd={r['weight_decay']:.0e}"
              for r in rows]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for box in bp["boxes"]:
        box.set(facecolor="#a8c4e0", edgecolor="#1f3f5f", linewidth=0.7)
    for med in bp["medians"]:
        med.set(color="#c0392b", linewidth=1.4)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, 1):
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)),
                   vals, s=10, color="#1f3f5f", alpha=0.75, zorder=3)
    ax.set_ylabel("各折验证准确率")
    ax.set_title(f"交叉验证准确率最高的 {top_n} 个组合（黑点为单折结果）")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(summaries, path):
    hiddens = sorted({r["hidden_size"] for r in summaries})
    dropouts = sorted({r["dropout"] for r in summaries})
    grid = np.zeros((len(hiddens), len(dropouts)))
    count = np.zeros_like(grid)
    for r in summaries:
        i = hiddens.index(r["hidden_size"])
        j = dropouts.index(r["dropout"])
        grid[i, j] += r["mean_accuracy"]
        count[i, j] += 1
    grid = grid / np.maximum(count, 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(dropouts)), [f"{d:.1f}" for d in dropouts])
    ax.set_yticks(range(len(hiddens)), [str(h) for h in hiddens])
    ax.set_xlabel("dropout")
    ax.set_ylabel("hidden_size")
    ax.set_title("hidden_size x dropout 的平均验证准确率\n(对其余超参数取平均)")
    for i in range(len(hiddens)):
        for j in range(len(dropouts)):
            val = grid[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=9,
                    color="white" if val > grid.mean() else "#20303f")
    fig.colorbar(im, ax=ax, label="平均准确率")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_marginal_effects(summaries, path):
    """对每个超参数取值在其余超参数上取平均，画边际影响。

    边际曲线会抹平交互作用：某个取值看着更好，可能只在另一个超参数的
    某一段上成立，要配合二维交叉表读。
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in zip(axes, ("lr", "weight_decay"),
                              ("learning_rate", "weight_decay")):
        values = sorted({r[key] for r in summaries})
        means = [np.mean([r["mean_accuracy"] for r in summaries if r[key] == v])
                 for v in values]
        stds = [np.std([r["mean_accuracy"] for r in summaries if r[key] == v])
                for v in values]
        ax.bar([str(v) for v in values], means, yerr=stds, capsize=4,
               color="#7fa8cd", edgecolor="#1f3f5f", linewidth=0.5)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.002, f"{m:.4f}", ha="center", fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("平均验证准确率")
        ax.set_title(f"{label} 的边际影响")
        ax.set_ylim(min(means) - 0.02, max(means) + 0.016)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss_curves(best_record, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    train_curves = [np.array(f["history"]["train_loss"]) for f in best_record["folds"]]
    val_curves = [np.array(f["history"]["val_loss"]) for f in best_record["folds"]]
    for tc, vc in zip(train_curves, val_curves):
        ax.plot(range(1, len(tc) + 1), tc, color="#3b6ea5", alpha=0.22, linewidth=0.9)
        ax.plot(range(1, len(vc) + 1), vc, color="#c0392b", alpha=0.22, linewidth=0.9)

    def mean_curve(curves):
        longest = max(len(c) for c in curves)
        padded = np.full((len(curves), longest), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        return np.nanmean(padded, axis=0)

    mt, mv = mean_curve(train_curves), mean_curve(val_curves)
    ax.plot(range(1, len(mt) + 1), mt, color="#1f3f5f", linewidth=2.0, label="训练损失(折均值)")
    ax.plot(range(1, len(mv) + 1), mv, color="#8e1b10", linewidth=2.0, label="早停验证损失(折均值)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("交叉熵损失")
    ax.set_title("最佳超参数组合的损失曲线（细线为各折自身曲线）")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "#20303f")
    fig.colorbar(im, ax=ax, label="样本数")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return cm
