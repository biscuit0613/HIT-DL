"""SST-2 数据加载与批构造：TSV 解析、分层划分、动态 padding。"""

import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def load_tsv(tsv_path):
    """读取 sentence\\tlabel 两列的 TSV（含表头），返回 [(text, label)]。"""
    samples = [] # list[tuple[str, int]]
    skipped = 0
    with open(tsv_path, encoding="utf-8") as f:
        f.readline()  # 跳过表头 sentence\tlabel
        for line in f:
            line = line.rstrip("\n") # 去掉末尾换行符，避免空行被当作有效样本
            if not line.strip() or "\t" not in line: # 过滤空行或不含制表符的行，避免坏数据中断整个加载
                skipped += 1
                continue
            # 从右侧只切一次：即使正文里出现制表符，末尾的 label 列也不会被切错
            text_part, label_part = line.rsplit("\t", 1) 
            text = text_part.strip()
            try:
                label = int(label_part.strip())
            except ValueError:
                skipped += 1
                continue
            # 空文本或标签越界（不是 0/1）的行按脏数据跳过，不让个别坏行中断整个加载
            if not text or label not in (0, 1):
                skipped += 1
                continue
            samples.append((text, label))
    if skipped:
        print(f"[data] {Path(tsv_path).name}: 跳过 {skipped} 行无效数据")
    return samples # list[tuple[str, int]]


class SST2Dataset(Dataset):
    """初始化时一次性批量编码全部文本，__getitem__ 只做取值，避免训练时逐条分词。"""
    def __init__(self, samples, tokenizer, max_len=64):
        self.texts: list[str] = [t for t, _ in samples]
        self.labels: list[int] = [l for _, l in samples]
        # 一次性批量编码全部文本（tokenizer 内部并行），比训练时逐条分词快一个量级；
        # truncation=True 表示超过 max_len 的部分截断，编码结果长度不一，padding 留到 collate 里按 batch 做
        self.ids: list[list[int]] = tokenizer(
            self.texts, truncation=True, max_length=max_len
        )["input_ids"]# 这里只取 input_ids，attention_mask/token_type_ids 在 collate 里动态生成
        # Tokenizer 负责：分词、加 [CLS]/[SEP]、转 ID、padding（这里没设置）/truncation、生成 attention mask 和 token type ids。


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.ids[index],
            "label": self.labels[index],
            "text": self.texts[index],
        }


def make_collate_fn(pad_id):
    """动态 padding：只补齐到当前 batch 内最长序列，而不是固定 max_len。"""

    def collate_fn(batch):
        # 同一个 batch 内各条长度不同，右侧补 pad_id 到本 batch 最长即可；
        # SST-2 句子普遍很短，比整表固定补到 max_len=64 省下大量无效计算
        # padding过程：
        # DataLoader 会收集 batch_size 个单样本，组成一个列表：
        # batch = [
            # {"input_ids": [101, 2023, 3185, 2003, 2307, 102], "label": 1, "text": "..."},
            # {"input_ids": [101, 1045, 5223, 2023, 2143, 102], "label": 0, "text": "..."},
            # ...  # 共 B 个
        # ]
        input_ids = torch.nn.utils.rnn.pad_sequence( # 第二步：使用 pad_sequence 把它们右侧补齐到本 batch 最大长度
            [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch],  # 第一步：把每个 list[int] 转成 torch.Tensor：
            batch_first=True,
            padding_value=pad_id,
        )
        # input_ids.shape == [B, L_batch]
        return {
            "input_ids": input_ids,
            # attention_mask 区分真实 token（1）与补齐的 pad（0），模型据此在注意力里屏蔽 pad 位置
            "attention_mask": (input_ids != pad_id).long(),
            # 单句输入无句子对关系，token_type_ids 恒为 0
            "token_type_ids": torch.zeros_like(input_ids),
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "texts": [b["text"] for b in batch],
        }

    return collate_fn


def split_train_val(samples, val_ratio, seed):
    """按标签分层切出验证集。"""
    # stratify 保证验证集正负类比例与训练集一致；固定 seed 使三种策略共用同一划分，对比才公平
    labels = [l for _, l in samples]
    train, valid = train_test_split(
        samples, test_size=val_ratio, random_state=seed, stratify=labels
    )
    return train, valid


def _length_stats(ids, max_len):
    # 统计训练集长度分布：确认 max_len 的截断率可接受，同时为报告的长度分布图提供数据
    lens = np.array([len(x) for x in ids])
    hist_counts, hist_edges = np.histogram(lens, bins=np.arange(0, max_len + 5, 4))
    return {
        "mean": float(lens.mean()),
        "p50": float(np.percentile(lens, 50)),
        "p90": float(np.percentile(lens, 90)),
        "p99": float(np.percentile(lens, 99)),
        "max": int(lens.max()),
        "truncated_ratio": float((lens >= max_len).mean()),
        "hist_edges": [int(e) for e in hist_edges],
        "hist_counts": [int(c) for c in hist_counts],
    }


def _pos_ratio(samples):
    return sum(l for _, l in samples) / len(samples)


def build_dataloaders(args, tokenizer):
    data_dir = Path(args.data_dir)
    train_samples: list[tuple[str, int]] = load_tsv(data_dir / "train.tsv")
    test_samples = load_tsv(data_dir / "test.tsv")

    # 冒烟测试用：先采样再划分，保证小样本时各集合的比例关系不变
    if args.train_size and args.train_size < len(train_samples):
        train_samples = random.Random(args.seed).sample(train_samples, args.train_size)

    # 从训练集里分层切出验证集：demo 直接拿 test.tsv 当验证集反复调参，最终指标会偏乐观；
    # 本实现 test.tsv 只在训练全部结束后评估一次
    train_samples, valid_samples = split_train_val(train_samples, args.val_ratio, args.seed)

    train_set = SST2Dataset(train_samples, tokenizer, args.max_len)
    valid_set = SST2Dataset(valid_samples, tokenizer, args.max_len)
    test_set = SST2Dataset(test_samples, tokenizer, args.max_len)

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": args.num_workers,
        # 锁页内存加速 CPU→GPU 的拷贝，仅在 CUDA 环境下有意义
        "pin_memory": torch.cuda.is_available(),
    }
    # 训练集打乱顺序，避免相邻样本相关性影响梯度；验证/测试保持原始顺序便于对照预测结果
    return {
        "train_loader": DataLoader(train_set, shuffle=True, **loader_kwargs),
        # Dataloader shuffle=False 保持原始顺序，便于对照预测结果
        # 调用的是dataset类的__getitem__方法，返回的是一个字典，里面包含input_ids: list[int],  labels: list[int], texts: list[str]
        # 传入自定义的 collate_fn，返回的是一个字典，里面包含 input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, labels: torch.Tensor, texts: list[str]
        "valid_loader": DataLoader(valid_set, shuffle=False, **loader_kwargs),
        "test_loader": DataLoader(test_set, shuffle=False, **loader_kwargs),
        "n_train": len(train_set),
        "n_valid": len(valid_set),
        "n_test": len(test_set),
        "pos_ratio": {
            "train": _pos_ratio(train_samples),
            "valid": _pos_ratio(valid_samples),
            "test": _pos_ratio(test_samples),
        },
        "length_stats": _length_stats(train_set.ids, args.max_len),
    }
