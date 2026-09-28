import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码：PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(pos/10000^(2i/d))
    Transformer 的 self-attention 对输入序列置换不变，位置编码注入词序信息。"""
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        # 对数空间计算 1/10000^(2i/d_model)，避免数值下溢
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(pos * div)  # 奇数维度用 cos
        # register_buffer: 非训练参数，但会随 model.to(device) 自动移动
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, S, d_model]，位置编码广播到 batch 维度，取前 S 个位置
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerMT(nn.Module):
    """标准 Transformer encoder-decoder 机器翻译模型。
    Encoder 将源句子编码为上下文表示，Decoder 通过 cross-attention 对齐源端信息并自回归生成目标句子。"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512,
                 dropout=0.1, max_len=500, src_pad_idx=0, tgt_pad_idx=0):#0  词表里第一个是 <pad>
        super().__init__()
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # 词嵌入层：token id -> d_model 维向量，padding_idx 位置的梯度恒为 0
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        # V 是目标词表大小，d_model 是嵌入维度，padding_idx=0 表示 <pad> 的嵌入向量不参与训练
        # embedding 每一行对应一个 token 的向量表示，padding_idx=0 表示 <pad> 的嵌入向量不参与训练，保持为零向量。
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)
        # PyTorch 原生 Transformer：包含 num_encoder_layers 个 encoder 层 + num_decoder_layers 个 decoder 层
        # 每个 encoder 层：self-attention + FFN；每个 decoder 层：masked self-attention + cross-attention + FFN
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,  # FFN 中间层维度（通常是 d_model 的 2~4 倍）
            dropout=dropout, batch_first=True,  # batch_first=True 使输入形状为 [B, S, D]
        )
        # 输出投影：d_model -> 目标词表大小，得到每个位置的词分布 logits
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        # Xavier 均匀初始化：使各层输入输出方差一致，缓解深层网络的梯度消失/爆炸
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _src_mask(self, src):
        # key_padding_mask: True 表示该位置是 <pad>，attention 时权重置为 -inf 后 softmax=0
        return src == self.src_pad_idx

    def _tgt_mask(self, tgt):
        return tgt == self.tgt_pad_idx

    def _causal_mask(self, sz, device):
        # 上三角矩阵（diagonal=1 不含对角线）：True 的位置表示"未来"，softmax 前置为 -inf
        # 保证 decoder 位置 t 只能看到 t 之前的 token（自回归约束）
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, src, tgt):
        # 词嵌入乘以 sqrt(d_model)：抵消嵌入初始化方差，使其与位置编码量级匹配（原论文 3.4 节）
        scale = math.sqrt(self.d_model)
        src_emb = self.pos_enc(self.src_embed(src) * scale)  # [B, S_src, d_model]
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * scale)  # [B, S_tgt, d_model]
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=self._causal_mask(tgt.size(1), tgt.device),   # decoder masked self-attention
            src_key_padding_mask=self._src_mask(src),               # 忽略源端 <pad>
            tgt_key_padding_mask=self._tgt_mask(tgt),               # 忽略目标端 <pad>
            memory_key_padding_mask=self._src_mask(src),            # cross-attention 也忽略源端 <pad>
        )
        return self.fc_out(out)  # [# [B, T, 256] → [B, T, V_tgt]，训练时整批并行计算（teacher forcing）

    def encode(self, src):
        """推理时单独调用 encoder，memory 只算一次，供 decode_step 反复使用。"""
        src_m = self._src_mask(src)
        emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(emb, src_key_padding_mask=src_m), src_m

    def decode_step(self, tgt, memory, mem_mask):
        """推理时逐步扩展 tgt，每次重算整个已生成序列的 decoder 输出，取最后一位作为预测。"""
        causal = self._causal_mask(tgt.size(1), tgt.device)
        emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            emb, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=self._tgt_mask(tgt),
            memory_key_padding_mask=mem_mask,
        )
        return self.fc_out(out)  # [B, T, V]
