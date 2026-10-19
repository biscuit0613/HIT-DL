import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

import torch
from mt_data import Vocab, build_dataloaders
from mt_model import TransformerMT
from mt_decode import decode_with_attention

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

RESULTS = Path(__file__).parent / 'results'
FIGURES = Path(__file__).parent / 'figures'
FIGURES.mkdir(exist_ok=True)


def plot_learning_curves():
    log = json.loads((RESULTS / 'training_log.json').read_text())
    epochs = [r['epoch'] for r in log['epochs']]
    tr_loss = [r['train_loss'] for r in log['epochs']]
    val_loss = [r['val_loss'] for r in log['epochs']]
    val_bleu = [r['val_bleu'] for r in log['epochs']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, tr_loss, label='train loss', marker='o', markersize=3)
    ax1.plot(epochs, val_loss, label='val loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title('Training & Validation Loss')

    ax2.plot(epochs, val_bleu, label='val BLEU', marker='o', markersize=3, color='tab:green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BLEU Score')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_title('Validation BLEU')

    plt.tight_layout()
    plt.savefig(FIGURES / 'learning_curves.png', dpi=150)
    plt.close()
    print('saved learning_curves.png')


def plot_bleu_comparison():
    log = json.loads((RESULTS / 'training_log.json').read_text())
    test = log['test']
    greedy = test['bleu_greedy']
    beam = test['bleu_beam']

    fig, ax = plt.subplots(figsize=(5, 4))
    methods = ['Greedy', 'Beam Search']
    scores = [greedy, beam]
    bars = ax.bar(methods, scores, color=['#38BDF8', '#F97316'], alpha=0.85)
    ax.set_ylabel('BLEU Score')
    ax.set_title('Test Set BLEU: Decoding Strategy Comparison')
    ax.set_ylim(0, max(scores) * 1.15)
    for bar, score in zip(bars, scores):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{score:.2f}',
                ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / 'bleu_comparison.png', dpi=150)
    plt.close()
    print('saved bleu_comparison.png')


def plot_attention_heatmap(n_examples=3, min_src_tokens=5, max_src_tokens=14):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_vocab = Vocab.load(RESULTS / 'src_vocab.json')
    tgt_vocab = Vocab.load(RESULTS / 'tgt_vocab.json')

    log = json.loads((RESULTS / 'training_log.json').read_text())
    cfg = log['config']
    model = TransformerMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_encoder_layers=cfg['enc_layers'],
        num_decoder_layers=cfg['dec_layers'],
        dim_feedforward=cfg['d_ff'],
        dropout=0.0,
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
    ).to(device)
    model.load_state_dict(torch.load(RESULTS / 'best_model.pt', map_location=device))
    model.eval()

    _, _, test_dl, _, _ = build_dataloaders(batch_size=64, min_freq=cfg['min_freq'], max_len=cfg['max_len'],
                                            train_size=cfg['train_size'])

    # Opus-100 里混杂大量长 UN 公文句子，直接取前几条会导致热图 token 数过多、
    # 格子被压缩到肉眼难辨；这里改为挑选源句子长度适中的样例，保证坐标轴标签清晰可读
    candidates = []
    for src, tgt, src_texts, tgt_texts in test_dl:
        for i in range(src.size(0)):
            n_tok = len(src_texts[i].lower().split())
            if min_src_tokens <= n_tok <= max_src_tokens:
                candidates.append((src[i:i+1], src_texts[i], tgt_texts[i]))
            if len(candidates) >= n_examples:
                break
        if len(candidates) >= n_examples:
            break

    for idx_fig, (src_single, src_text, ref) in enumerate(candidates):
        src_single = src_single.to(device)

        tgt_ids, attn = decode_with_attention(model, src_single, src_vocab, tgt_vocab, device=device)
        pred = tgt_vocab.decode(tgt_ids)

        src_tokens = src_text.lower().split()
        tgt_tokens = list(pred)  # character-level Chinese tokens
        attn_np = attn.numpy()[:len(tgt_tokens), :len(src_tokens)]

        # 图幅随 token 数增长但设上限，字号固定为较大值，避免长句样例把坐标轴文字挤得看不清
        fig_w = min(max(len(src_tokens) * 0.55, 6), 9)
        fig_h = min(max(len(tgt_tokens) * 0.5, 5), 8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(attn_np, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=11)
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=11, fontfamily='Noto Sans CJK SC')
        ax.set_xlabel('Source (EN)', fontsize=12)
        ax.set_ylabel('Target (ZH)', fontsize=12)
        ax.set_title(f'Cross-Attention Heatmap #{idx_fig+1}', fontsize=13)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(FIGURES / f'attention_{idx_fig+1}.png', dpi=150)
        plt.close()
        print(f'saved attention_{idx_fig+1}.png')
        print(f'  SRC: {src_text}')
        print(f'  REF: {ref}')
        print(f'  HYP: {pred}\n')


def create_sample_table():
    log = json.loads((RESULTS / 'training_log.json').read_text())
    samples = log['test']['samples'][:5]
    with open(FIGURES / 'sample_translations.txt', 'w', encoding='utf-8') as f:
        for i, (ref, hyp) in enumerate(samples, 1):
            f.write(f'{i}. REF: {ref}\n')
            f.write(f'   HYP: {hyp}\n\n')
    print('saved sample_translations.txt')


if __name__ == '__main__':
    plot_learning_curves()
    plot_bleu_comparison()
    plot_attention_heatmap(n_examples=3)
    create_sample_table()
    print('all visualizations complete.')
