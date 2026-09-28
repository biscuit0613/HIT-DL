import json
import random
import collections
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

PAD, UNK, SOS, EOS = '<pad>', '<unk>', '<sos>', '<eos>'


def _tok_word(text):
    """英文词级分词：转小写后按空格切分。"""
    return text.lower().split()


def _tok_char(text):
    """中文字符级分词：将字符串拆分为单字列表。"""
    return list(text)


class Vocab:
    def __init__(self, min_freq=2, char_level=False):
        self.min_freq = min_freq
        self.char_level = char_level
        self.token2idx = {}
        self.idx2token = []

    def build(self, sentences):
        counter = collections.Counter() 
        # counter是一个字典子类，用于计数可哈希对象。它是一个无序的集合，其中元素（词）存储为字典键，计数（频次）存储为字典值。
        for s in sentences:
        # 对于每个句子s，如果是字符级别的，就用字符级别的分词器_tok_char(s)来更新计数器counter，否则就用单词级别的分词器_tok_word(s)来更新计数器counter。
            if self.char_level:
                counter.update(_tok_char(s))# 中文 字符分词
            else:
                counter.update(_tok_word(s))# 英文 空格分词
        specials = [PAD, UNK, SOS, EOS]  # 固定占据索引 0~3，与模型默认 pad_idx=0 等约定保持一致
        tokens = specials + [t for t, c in counter.most_common() if c >= self.min_freq]
        # 插入特殊符号：[<pad>, <unk>, <sos>, <eos>] + 高频词/字（低频词过滤掉，压缩词表规模）
        # 低频词（只出现 1 次）几乎不可能在验证/测试集里出现，训练时分配一个独立索引只会浪费 embedding
        # 矩阵的行数，还会让模型花时间学一个几乎用不上的向量。过滤后这些词统一映射到 <unk>。 
        self.idx2token = tokens  # list，支持 O(1) 按索引查 token
        self.token2idx = {t: i for i, t in enumerate(tokens)} # dict，支持 O(1) 按 token 查索引

        # token2idx:  {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3, 'the': 4, 'a': 5, ...}
        # idx2token:  ['<pad>', '<unk>', '<sos>', '<eos>', 'the', 'a', ...] 
        self.pad_idx = self.token2idx[PAD]
        self.unk_idx = self.token2idx[UNK]
        self.sos_idx = self.token2idx[SOS]
        self.eos_idx = self.token2idx[EOS]

    def encode(self, text): # 编码 查查表，把句子变成id列表
        if self.char_level:
            return [self.token2idx.get(t, self.unk_idx) for t in _tok_char(text)]  # 未登录词一律映射为 <unk>
        else:
            return [self.token2idx.get(t, self.unk_idx) for t in _tok_word(text)]

    def decode(self, indices):
        skip = {self.pad_idx, self.sos_idx, self.eos_idx}  # 还原文本时丢弃特殊符号（<unk> 需保留，它是真实的模型输出）
        if self.char_level:
            return ''.join(self.idx2token[i] for i in indices if i not in skip)
        else:
            return ' '.join(self.idx2token[i] for i in indices if i not in skip)

    def __len__(self):
        return len(self.idx2token)

    def save(self, path):
        Path(path).write_text(json.dumps({
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'char_level': self.char_level
        }, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        v = cls(char_level=d.get('char_level', False))
        v.token2idx = d['token2idx']
        v.idx2token = d['idx2token']
        v.pad_idx = v.token2idx[PAD]
        v.unk_idx = v.token2idx[UNK]
        v.sos_idx = v.token2idx[SOS]
        v.eos_idx = v.token2idx[EOS]
        return v


class OpusEnZhDataset(Dataset):
    """en -> zh 翻译数据集，src=英语，tgt=中文。"""

    def __init__(self, raw_data, src_vocab=None, tgt_vocab=None, max_len=64):
        self.pairs = [(r['translation']['en'], r['translation']['zh']) for r in raw_data]
        self.max_len = max_len
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_ids = self.src_vocab.encode(src_text)[: self.max_len]  # 截断到 max_len，避免个别超长句子拖慢训练
        tgt_ids = self.tgt_vocab.encode(tgt_text)[: self.max_len]
        src = torch.tensor(src_ids, dtype=torch.long)
        tgt = torch.tensor(
            [self.tgt_vocab.sos_idx] + tgt_ids + [self.tgt_vocab.eos_idx], dtype=torch.long
        )  # 只有目标端加 <sos>/<eos>：<sos> 是 decoder 自回归的起始输入，<eos> 是训练时的终止监督信号
        return src, tgt, src_text, tgt_text


def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    '''
    batch 内样本长度不同，需 pad 到同一长度才能并行计算。
    batch: list of (src, tgt, src_text, tgt_text)
    产出形状:
        src_pad: (batch_size, src_len)
        tgt_pad: (batch_size, tgt_len)
    '''
    srcs, tgts, src_texts, tgt_texts = zip(*batch)
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=src_pad_idx)  # 右侧补 <pad> 对齐为定长矩阵
    # padding_value=src_vocab.pad_idx 后续会在模型中用来 mask 掉 pad 的位置，避免对模型训练产生干扰。
    tgt_pad = pad_sequence(tgts, batch_first=True, padding_value=tgt_pad_idx)
    return src_pad, tgt_pad, list(src_texts), list(tgt_texts)


def build_dataloaders(batch_size=128, min_freq=2, max_len=64, train_size=30000, num_workers=0):
    """
    加载 Opus-100 en-zh 数据集，构建 DataLoader。
    源 (en): word-level.  目标 (zh): character-level.
    """
    ds = load_dataset('Helsinki-NLP/opus-100', 'en-zh')
    train_raw = ds['train']
    valid_raw = ds['validation']
    test_raw = ds['test']

    # 如果 train_size 小于训练集大小，则随机采样 train_size 个样本
    if len(train_raw) > train_size:
        indices = random.sample(range(len(train_raw)), train_size)
        train_raw = train_raw.select(indices)

    src_vocab = Vocab(min_freq, char_level=False)
    tgt_vocab = Vocab(min_freq, char_level=True)
    src_vocab.build([r['translation']['en'] for r in train_raw])  # 词表只用训练集统计，避免验证/测试集信息泄漏
    tgt_vocab.build([r['translation']['zh'] for r in train_raw])

    train_ds = OpusEnZhDataset(train_raw, src_vocab, tgt_vocab, max_len)
    valid_ds = OpusEnZhDataset(valid_raw, src_vocab, tgt_vocab, max_len)  # 验证/测试集复用训练集词表，未登录词落入 <unk>
    test_ds = OpusEnZhDataset(test_raw, src_vocab, tgt_vocab, max_len)

    collate = partial(collate_fn, src_pad_idx=src_vocab.pad_idx, tgt_pad_idx=tgt_vocab.pad_idx)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)   # 仅训练集打乱顺序
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    return train_dl, valid_dl, test_dl, src_vocab, tgt_vocab
