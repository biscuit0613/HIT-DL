import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sacrebleu.metrics import BLEU

from mt_data import build_dataloaders
from mt_model import TransformerMT
from mt_decode import greedy_decode, beam_search_decode

RESULTS = Path(__file__).parent / 'results'
RESULTS.mkdir(exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_bleu(refs, hyps):
    # tokenize='zh' 按字符切分中文，与目标端字符级词表保持一致；effective_order=True 使短句也能计算
    bleu = BLEU(effective_order=True, tokenize='zh')
    return bleu.corpus_score(hyps, [refs]).score


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, n_tokens = 0.0, 0
    with torch.set_grad_enabled(train):
        for src, tgt, _, _ in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            # tgt_in: decoder的输入，tgt_out: decoder的输出
            # teacher forcing: 直接使用真实目标序列的前缀，也就是 tgt_in。
            # 这样模型每一步都能看到“正确答案的历史”，然后学习预测下一个 token。
            # tgt_in 是 decoder 输入 ， 在model的参数中，tgt_in 经过嵌入和位置编码后，进入 decoder 进行自回归预测。
            # tgt_out 是对 decoder 输出的监督标签，训练时整批并行计算，避免逐步解码慢速。
            # 关于切片：a[start:stop:step] 
            # ':' 表示取所有样本，' :-1' 表示去掉最后一个 token，' 1:' 表示去掉第一个 token
            logits = model(src, tgt_in)  # 输出形状[B, T, V]
            #   每个位置 logits[b, t, :] 是一个长度为 V 的向量，表示模型对第 t 个位置的未归一化对数概率（raw scores），需经过 softmax 才能得到概率分布。 
            # PyTorch 的 CrossEntropyLoss 只接受 2D 输入：              
            # - 输入 1（predictions）: [N, C]，N 是样本数，C 是类别数    
            #  把 [B, T, V] 的 logits 看成 N=B*T 个样本、每个样本有 C=V 个类别的分类任务。                                
            # - 输入 2（targets）: [N]，每个位置是类别索引（0 ~ C-1）
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))  # 展平为 [B*T, V] 和 [B*T] 计算交叉熵， 注意这里的监督标签是 tgt_out，它是目标序列的真实 token 索引，经过切片去掉了 <sos>，只保留了实际的输出 token。
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪：L2 范数超过 1.0 时等比缩放，防止梯度爆炸
                optimizer.step()
            mask = tgt_out != model.tgt_pad_idx  # 只统计非 <pad> 位置的损失
            total_loss += loss.item() * mask.sum().item()
            n_tokens += mask.sum().item()
    return total_loss / max(n_tokens, 1)  # 返回每个有效 token 的平均损失（perplexity 的对数）


@torch.no_grad()
def evaluate_bleu(model, loader, src_vocab, tgt_vocab, device, beam=False, n_samples=500):
    model.eval()
    refs, hyps = [], []
    count = 0
    for src, tgt, src_texts, tgt_texts in loader:
        for i in range(src.size(0)):
            if count >= n_samples:  # 只抽样部分数据评估，加速验证阶段
                break
            s = src[i:i+1].to(device)
            if beam:
                ids = beam_search_decode(model, s, src_vocab, tgt_vocab, device=device)
            else:
                ids = greedy_decode(model, s, src_vocab, tgt_vocab, device=device)
            ids = [t for t in ids if t != tgt_vocab.eos_idx and t != tgt_vocab.pad_idx]  # 去掉终止符/填充符再还原成文本
            hyps.append(tgt_vocab.decode(ids))
            refs.append(tgt_texts[i])  # 参考译文用原始文本，交给 sacrebleu 内部按 tokenize='zh' 切分
            count += 1
        if count >= n_samples:
            break
    return compute_bleu(refs, hyps), list(zip(refs[:8], hyps[:8]))


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # 加载数据集，构建 DataLoader
    train_dl, valid_dl, test_dl, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=args.batch_size, min_freq=args.min_freq, max_len=args.max_len,
        train_size=args.train_size,
    )
    print(f'src vocab: {len(src_vocab)}  tgt vocab: {len(tgt_vocab)}')
    print(f'train batches: {len(train_dl)}  valid batches: {len(valid_dl)}')

    src_vocab.save(RESULTS / 'src_vocab.json')
    tgt_vocab.save(RESULTS / 'tgt_vocab.json')

    model = TransformerMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.d_ff,
        dropout=args.dropout,
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
    ).to(device)
    print(f'parameters: {count_params(model):,}')

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx, label_smoothing=0.1)
    # ignore_index: <pad> 位置不计入损失；label_smoothing=0.1: 目标分布平滑化，防止过拟合
    # [0, 0, ..., 1, 0, ..., 0]  onehot:只有 578 位置是 1，其他全是 0
    # [0.001, 0.001, ..., 0.9, 0.001, ..., 0.001]  label smoothing: 平滑化后，其他位置也有微小概率，防止过拟合 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # warmup + inverse sqrt decay scheduler：前 4000 步线性增长，之后按 step^-0.5 衰减
    warmup = args.warmup_steps
    def lr_lambda(step):
        step = max(step, 1)
        return min(step ** -0.5, step * warmup ** -1.5) * (args.d_model ** -0.5) / (warmup ** -0.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log = {'config': vars(args), 'epochs': []}
    best_bleu, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        scheduler.step()
        val_loss = run_epoch(model, valid_dl, criterion, optimizer, device, train=False)
        val_bleu, samples = evaluate_bleu(model, valid_dl, src_vocab, tgt_vocab, device,
                                          beam=False, n_samples=300)
        elapsed = time.time() - t0

        print(f'epoch {epoch:2d} | tr_loss {tr_loss:.4f} | val_loss {val_loss:.4f} '
              f'| val_bleu {val_bleu:.2f} | {elapsed:.0f}s')

        row = {
            'epoch': epoch,
            'train_loss': round(tr_loss, 5),
            'val_loss': round(val_loss, 5),
            'val_bleu': round(val_bleu, 3),
            'elapsed_s': round(elapsed, 1),
            'lr': optimizer.param_groups[0]['lr'],
            'samples': samples,
        }
        log['epochs'].append(row)
        (RESULTS / 'training_log.json').write_text(json.dumps(log, ensure_ascii=False, indent=2))

        if val_bleu > best_bleu:
            best_bleu, best_epoch = val_bleu, epoch
            torch.save(model.state_dict(), RESULTS / 'best_model.pt')  # 只保存参数字典（不含优化器状态）
            print(f'  -> saved best (val_bleu={best_bleu:.2f})')

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), RESULTS / f'ckpt_epoch{epoch}.pt')  # 定期保存检查点

    print(f'\nbest val BLEU {best_bleu:.2f} at epoch {best_epoch}')

    # final test evaluation with beam search
    model.load_state_dict(torch.load(RESULTS / 'best_model.pt', map_location=device))  # 加载验证集最优模型测试
    test_bleu_greedy, _ = evaluate_bleu(model, test_dl, src_vocab, tgt_vocab, device,
                                        beam=False, n_samples=1000)
    test_bleu_beam, test_samples = evaluate_bleu(model, test_dl, src_vocab, tgt_vocab, device,
                                                  beam=True, n_samples=1000)  # beam search 通常比 greedy 高 1~3 BLEU
    print(f'test BLEU greedy={test_bleu_greedy:.2f}  beam={test_bleu_beam:.2f}')

    log['test'] = {
        'bleu_greedy': round(test_bleu_greedy, 3),
        'bleu_beam': round(test_bleu_beam, 3),
        'samples': test_samples,
    }
    (RESULTS / 'training_log.json').write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print('results saved to', RESULTS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--enc-layers', type=int, default=3)
    parser.add_argument('--dec-layers', type=int, default=3)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=4000)
    parser.add_argument('--min-freq', type=int, default=2)
    parser.add_argument('--max-len', type=int, default=64)
    parser.add_argument('--train-size', type=int, default=30000)
    parser.add_argument('--save-every', type=int, default=10)
    args = parser.parse_args()
    train(args)
