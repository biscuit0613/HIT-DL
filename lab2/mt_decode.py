import torch


@torch.no_grad()
def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=64, device='cpu'):
    """单句贪心解码。src: [1, S] 已编码张量。返回 token id 列表（含 eos，不含 sos）。"""
    model.eval()
    src = src.to(device)
    memory, mem_mask = model.encode(src)  # encoder 只跑一次，结果复用于每一步 decode
    ys = torch.tensor([[tgt_vocab.sos_idx]], dtype=torch.long, device=device)  # 从 <sos> 开始自回归生成
    for _ in range(max_len):
        logits = model.decode_step(ys, memory, mem_mask)
        next_id = logits[:, -1].argmax(-1).item()  # 只取最后一个位置、概率最大的 token（贪心，非采样）
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == tgt_vocab.eos_idx:  # 生成 <eos> 即提前终止
            break
    return ys.squeeze(0).tolist()[1:]


@torch.no_grad()
def beam_search_decode(model, src, src_vocab, tgt_vocab, max_len=64, beam_width=5,
                        length_penalty=0.7, device='cpu'):
    """单句 beam search 解码，返回得分最高序列的 token id 列表（不含 sos）。"""
    model.eval()
    src = src.to(device)
    memory, mem_mask = model.encode(src)  # encoder 输出复用，避免重复计算

    # 每个 beam 存为 (已生成序列, 累积对数概率, 是否已完成)
    beams = [(torch.tensor([[tgt_vocab.sos_idx]], dtype=torch.long, device=device), 0.0, False)]
    finished = []

    for _ in range(max_len):
        candidates = []
        for ys, score, done in beams:
            if done:
                candidates.append((ys, score, True))
                continue
            mem = memory.expand(1, -1, -1)
            mmask = mem_mask.expand(1, -1)
            logits = model.decode_step(ys, mem, mmask)
            log_probs = torch.log_softmax(logits[:, -1], dim=-1).squeeze(0)  # [V]，对最后一步取 log 概率
            topk_lp, topk_id = log_probs.topk(beam_width)  # 每个 beam 展开 beam_width 个候选
            for lp, tid in zip(topk_lp.tolist(), topk_id.tolist()):
                new_ys = torch.cat([ys, torch.tensor([[tid]], device=device)], dim=1)
                new_done = tid == tgt_vocab.eos_idx
                candidates.append((new_ys, score + lp, new_done))  # 对数概率可直接累加

        def norm_score(item):
            ys, score, done = item
            length = ys.size(1) - 1
            # 长度惩罚：除以 length^α，防止模型偏好短句（α=length_penalty）
            return score / (length ** length_penalty)

        candidates.sort(key=norm_score, reverse=True)
        beams = candidates[:beam_width]  # 保留得分最高的 beam_width 条路径

        active = [b for b in beams if not b[2]]
        if not active:  # 所有 beam 都已生成 <eos>，提前终止
            break

    best = max(beams, key=lambda it: it[1] / ((it[0].size(1) - 1) ** length_penalty))
    return best[0].squeeze(0).tolist()[1:]


@torch.no_grad()
def decode_with_attention(model, src, src_vocab, tgt_vocab, max_len=64, device='cpu'):
    """贪心解码并抓取最后一层 decoder 的交叉注意力权重，用于可视化。"""
    model.eval()
    src = src.to(device)
    memory, mem_mask = model.encode(src)

    # PyTorch TransformerDecoderLayer 内部调用 MHA 时硬编码 need_weights=False，
    # 导致 attn_output_weights 始终为 None。通过猴子补丁替换最后一层 cross-attention
    # 的 forward，强制 need_weights=True 并截获权重，解码完成后恢复原始方法。
    captured = {}
    last_mha = model.transformer.decoder.layers[-1].multihead_attn  # 只取最后一层，语义最完整
    orig_forward = last_mha.forward

    def patched_forward(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = True  # 多头权重在 head 维度取均值，得到 [B, T, S]
        out = orig_forward(*args, **kwargs)
        captured['w'] = out[1]
        return out

    last_mha.forward = patched_forward

    ys = torch.tensor([[tgt_vocab.sos_idx]], dtype=torch.long, device=device)
    attn_rows = []
    try:
        for _ in range(max_len):
            logits = model.decode_step(ys, memory, mem_mask)
            if 'w' in captured and captured['w'] is not None:
                attn_rows.append(captured['w'][0, -1].cpu())  # 取当前最新位置的注意力行，形状 [S_src]
            next_id = logits[:, -1].argmax(-1).item()
            ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == tgt_vocab.eos_idx:
                break
    finally:
        last_mha.forward = orig_forward  # 无论是否异常，都还原，防止影响后续推理

    tokens = ys.squeeze(0).tolist()[1:]
    if not attn_rows:
        raise RuntimeError('No attention weights captured — check MHA patching.')
    attn_matrix = torch.stack(attn_rows, dim=0)  # 堆叠为 [T_tgt, S_src]，供热力图可视化
    return tokens, attn_matrix
