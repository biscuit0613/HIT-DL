# -*- coding: utf-8 -*-
"""
基于 PyTorch 的 BERT 微调实验 —— SST-2 情感分类
数据格式: sentence\tlabel (TSV, 含表头)
预训练模型: ./bert_localpath (bert-base-uncased, pytorch_model.bin)
使用的torch和transformers版本如下
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 -i https://mirrors.aliyun.com/pypi/simple/
pip uninstall -y transformers
pip install "transformers==4.46.3" -i https://mirrors.aliyun.com/pypi/simple/

"""

import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
)
from sklearn.metrics import classification_report, confusion_matrix

# ==================== 1. 环境配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==================== 2. 数据准备 ====================

class SST2Dataset(Dataset):
    """
    加载 SST-2 格式的 TSV 文件。
    第一行是表头: sentence\tlabel
    之后每行: 文本 + \t + 标签(0/1)
    """
    def __init__(self, tsv_path, tokenizer, max_len=64):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(tsv_path, "r", encoding="utf-8") as f:
            f.readline()  # 跳过表头
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if "\t" not in line:
                    continue
                text_part, label_part = line.rsplit("\t", 1)
                text = text_part.strip()
                try:
                    label = int(label_part.strip())
                except ValueError:
                    continue
                if text == "":
                    continue
                self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text, label = self.samples[index]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


tokenizer = AutoTokenizer.from_pretrained("./bert_localpath")
MAX_LEN = 64
BATCH_SIZE = 32

train_dataset = SST2Dataset("./SST-2/train.tsv", tokenizer, MAX_LEN)
test_dataset = SST2Dataset("./SST-2/test.tsv", tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")
print(f"训练集批次数: {len(train_loader)}")
print(f"测试集批次数: {len(test_loader)}")


# ==================== 3. 模型构建 ====================

def build_pretrained_model(num_labels=2):
    """加载本地预训练 BERT + 分类头"""
    model = AutoModelForSequenceClassification.from_pretrained(
        "./bert_localpath", num_labels=num_labels
    )
    return model


def build_scratch_model(num_labels=2):
    """从零开始（随机初始化）的 BERT + 分类头"""
    config = BertConfig.from_pretrained("./bert_localpath", num_labels=num_labels)
    model = BertForSequenceClassification(config)
    return model


# ==================== 4. 训练与评估函数 ====================

class Trainer:
    def __init__(self, model, train_loader, test_loader, device,
                 lr=2e-5, epochs=3, log_every=20):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.log_every = log_every

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_losses = []
        self.train_steps = []
        self.eval_accs = []

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, total_num = 0.0, 0
        step = 0
        steps_per_epoch = len(self.train_loader)
        epoch_start = time.time()

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            bs = input_ids.size(0)
            cur_loss = loss.item()
            total_loss += cur_loss * bs
            total_num += bs
            step += 1

            if step % self.log_every == 0 or step == steps_per_epoch:
                global_step = (epoch - 1) * steps_per_epoch + step
                self.train_losses.append(cur_loss)
                self.train_steps.append(global_step)
                pct = 100.0 * step / steps_per_epoch
                print(f"  [Epoch {epoch}] step {step:4d}/{steps_per_epoch} "
                      f"({pct:5.1f}%) | batch loss = {cur_loss:.4f}")

        avg_loss = total_loss / max(total_num, 1)
        epoch_time = time.time() - epoch_start
        print(f"[Epoch {epoch}] 平均训练 loss = {avg_loss:.4f} | 用时 {epoch_time:.1f}s")
        return avg_loss

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        for batch in self.test_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = (all_preds == all_labels).mean()
        return acc, all_preds, all_labels

    def run(self):
        for epoch in range(1, self.epochs + 1):
            print(f"\n===== 开始 Epoch {epoch} =====")
            self.train_one_epoch(epoch)
            acc, _, _ = self.evaluate()
            self.eval_accs.append(acc)
            print(f"[Epoch {epoch}] 测试集准确率 = {acc:.4f}")


# ==================== 5. 微调预训练模型 ====================

print("\n========== 微调预训练 BERT ==========")
pretrained_model = build_pretrained_model(num_labels=2)
trainer_pre = Trainer(pretrained_model, train_loader, test_loader, device,
                      lr=2e-5, epochs=3)
trainer_pre.run()

acc_pre, preds_pre, labels_pre = trainer_pre.evaluate()
print(f"\n预训练 BERT 最终测试准确率: {acc_pre:.4f}")


# ==================== 6. 详细评估指标 ====================

print("\n========== 详细评估指标 ==========")
print(classification_report(labels_pre, preds_pre,
                            target_names=["Negative", "Positive"],
                            zero_division=0))
print("混淆矩阵:")
print(confusion_matrix(labels_pre, preds_pre))


# ==================== 7. 对比实验：从零开始 ====================

print("\n========== 对比实验：从零开始训练 ==========")
scratch_model = build_scratch_model(num_labels=2)
trainer_scratch = Trainer(scratch_model, train_loader, test_loader, device,
                          lr=2e-5, epochs=3)
trainer_scratch.run()

acc_scratch, _, _ = trainer_scratch.evaluate()
print(f"\n从零开始模型最终测试准确率: {acc_scratch:.4f}")


# ==================== 8. 结果汇总 ====================

print("\n========== 结果汇总 ==========")
print(f"预训练模型准确率:   {acc_pre:.4f}")
print(f"从零开始模型准确率: {acc_scratch:.4f}")
print(f"预训练带来的提升:   {(acc_pre - acc_scratch) * 100:.2f}%")

print("\n训练过程 loss 记录（预训练模型，每 {} 步打印一次）：".format(trainer_pre.log_every))
for s, l in zip(trainer_pre.train_steps, trainer_pre.train_losses):
    print(f"  step {s:5d} | loss {l:.4f}")

print("\n每个 Epoch 结束时的测试准确率（预训练模型）：")
for i, a in enumerate(trainer_pre.eval_accs, start=1):
    print(f"  Epoch {i}: acc = {a:.4f}")

print("\n每个 Epoch 结束时的测试准确率（从零开始）：")
for i, a in enumerate(trainer_scratch.eval_accs, start=1):
    print(f"  Epoch {i}: acc = {a:.4f}")