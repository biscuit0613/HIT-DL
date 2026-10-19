"""训练结果可视化：长度分布、学习曲线、策略对比、混淆矩阵、置信度、误分类样例。"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from bert_model import MODE_LABELS

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

LAB_DIR = Path(__file__).resolve().parent
RESULTS = LAB_DIR / 'results'
FIGURES = LAB_DIR / 'figures'
FIGURES.mkdir(exist_ok=True)

MODES = ["finetune", "feature-extract", "scratch"]


def read_log(mode):
    return json.loads((RESULTS / mode / 'training_log.json').read_text(encoding='utf-8'))


def read_predictions(mode):
    return json.loads(
        (RESULTS / mode / 'test_predictions.json').read_text(encoding='utf-8')
    )


def plot_length_distribution():
    stats = json.loads((RESULTS / 'length_stats.json').read_text(encoding='utf-8'))
    split = json.loads((RESULTS / 'split_info.json').read_text(encoding='utf-8'))

    edges = stats['hist_edges']
    counts = stats['hist_counts']
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    width = (edges[1] - edges[0]) * 0.9

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centers, counts, width=width, color='steelblue', edgecolor='white')
    ax.axvline(split['max_len'], color='crimson', linestyle='--',
               label=f"max_len = {split['max_len']}")
    ax.set_xlabel('Token length (WordPiece)')
    ax.set_ylabel('Number of samples')
    ax.set_title(
        f"Training set length distribution (mean {stats['mean']:.1f}, "
        f"p90 {stats['p90']:.0f}, truncated {stats['truncated_ratio'] * 100:.2f}%)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / 'length_distribution.png', dpi=150)
    plt.close(fig)
    print('已生成 figures/length_distribution.png')


def plot_learning_curves(mode):
    log = read_log(mode)
    rows = log['epochs']
    epochs = [r['epoch'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(epochs, [r['train_loss'] for r in rows], marker='o', label='train loss')
    axes[0].plot(epochs, [r['val_loss'] for r in rows], marker='s', label='val loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(epochs, [r['val_acc'] for r in rows], marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation accuracy')

    axes[2].plot(epochs, [r['val_f1'] for r in rows], marker='o', color='darkorange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1')
    axes[2].set_title('Validation F1')

    for ax in axes:
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{MODE_LABELS[mode]} ({mode})")
    fig.tight_layout()
    fig.savefig(FIGURES / f'curves_{mode}.png', dpi=150)
    plt.close(fig)
    print(f'已生成 figures/curves_{mode}.png')


def plot_comparison():
    modes = MODES
    accs, f1s, val_f1s = [], [], []
    for mode in modes:
        log = read_log(mode)
        accs.append(log['test']['acc'])
        f1s.append(log['test']['f1'])
        val_f1s.append(log['best_val_f1'])

    x = np.arange(len(modes))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = [
        ax.bar(x - width, val_f1s, width, label='Best val F1', color='#9ecae1'),
        ax.bar(x, accs, width, label='Test accuracy', color='#3182bd'),
        ax.bar(x + width, f1s, width, label='Test F1', color='#e6550d'),
    ]
    for group in bars:
        for rect in group:
            ax.annotate(f'{rect.get_height():.3f}',
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODE_LABELS[m]}\n({m})" for m in modes])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Strategy comparison on SST-2 test set')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / 'comparison.png', dpi=150)
    plt.close(fig)
    print('已生成 figures/comparison.png')


def plot_confusion(mode):
    data = read_predictions(mode)
    cm = confusion_matrix(data['labels'], data['preds'])

    fig, ax = plt.subplots(figsize=(4.4, 4))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)
    ax.set_xticks([0, 1], labels=['Negative', 'Positive'])
    ax.set_yticks([0, 1], labels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f"Confusion matrix – {mode}\n(acc {data['acc']:.4f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES / f'confusion_{mode}.png', dpi=150)
    plt.close(fig)
    print(f'已生成 figures/confusion_{mode}.png')


def plot_confidence(mode):
    data = read_predictions(mode)
    probs = np.array(data['probs'])
    conf = probs.max(axis=1)
    correct = np.array(data['labels']) == np.array(data['preds'])

    bins = np.linspace(0.5, 1.0, 26)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf[correct], bins=bins, alpha=0.6, label=f'Correct (n={correct.sum()})',
            color='#3182bd')
    ax.hist(conf[~correct], bins=bins, alpha=0.6, label=f'Wrong (n={(~correct).sum()})',
            color='#e6550d')
    ax.set_xlabel('Max softmax probability')
    ax.set_ylabel('Number of samples')
    ax.set_yscale('log')
    ax.set_title(f'Prediction confidence – {mode}')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f'confidence_{mode}.png', dpi=150)
    plt.close(fig)
    print(f'已生成 figures/confidence_{mode}.png')


def write_error_examples(mode, top_k=10):
    data = read_predictions(mode)
    probs = np.array(data['probs'])
    conf = probs.max(axis=1)
    preds = np.array(data['preds'])
    labels = np.array(data['labels'])
    wrong = np.where(preds != labels)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:top_k]

    lines = [f"策略 {mode} 测试集上最自信的 {top_k} 条误分类样例", ""]
    for idx in wrong:
        lines.append(
            f"confidence {conf[idx]:.4f} | true {labels[idx]} -> pred {preds[idx]} | {data['texts'][idx]}"
        )
    (FIGURES / f'error_examples_{mode}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'已生成 figures/error_examples_{mode}.txt（{len(wrong)} 条）')


def main():
    plot_length_distribution()
    plot_comparison()
    for mode in MODES:
        plot_learning_curves(mode)
        plot_confusion(mode)
        plot_confidence(mode)
        write_error_examples(mode)


if __name__ == '__main__':
    main()
