# -*- coding: utf-8 -*-
"""
基于 MindSpore 2.10.0 的简单 RNN 文本分类（SST-2）
conda activate mindspore
pip uninstall -y numpy
pip install "numpy==1.26.4"
建议：将numpy降低至以上版本
"""

import os
import csv
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import Normal, Zero, initializer

# ---- 设备与上下文 ----
try:
    ms.set_device("CPU")             # 可改为 "GPU" / "Ascend"
except AttributeError:
    ms.set_context(device_target="CPU")

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_seed(42)


# ----------------------------------------------------------------------
# 1. 数据读取与预处理
# ----------------------------------------------------------------------
def load_tsv(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) < 2:
                continue
            sent, label = row[0].strip(), row[1].strip()
            if sent == "":
                continue
            try:
                data.append((sent, int(label)))
            except ValueError:
                continue
    return data


def tokenize(sentence):
    return sentence.lower().split()


def build_vocab(sentences, min_freq=1):
    freq = {}
    for s in sentences:
        for w in tokenize(s):
            freq[w] = freq.get(w, 0) + 1
    words = [w for w, c in sorted(freq.items(), key=lambda x: -x[1]) if c >= min_freq]
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode(sentence, vocab, max_len):
    ids = [vocab.get(w, vocab["<unk>"]) for w in tokenize(sentence)]
    if len(ids) >= max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    return ids


# ----------------------------------------------------------------------
# 2. 数据集封装
# ----------------------------------------------------------------------
class SST2Dataset:
    def __init__(self, data, vocab, max_len):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent, label = self.data[idx]
        ids = encode(sent, self.vocab, self.max_len)
        return np.array(ids, dtype=np.int32), np.array(label, dtype=np.float32)


def make_batches(dataset, batch_size, shuffle=True):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        ids_list, label_list = [], []
        for j in batch_idx:
            ids, label = dataset[j]
            ids_list.append(ids)
            label_list.append(label)
        yield (
            np.stack(ids_list, axis=0),
            np.array(label_list, dtype=np.float32)[:, None]
        )


# ----------------------------------------------------------------------
# 3. 简单 RNN 模型（正向传播过程显式展开）
# ----------------------------------------------------------------------
class SimpleRNNClassifier(nn.Cell):
    """
    网络结构：
        Embedding -> RNN(逐时间步循环) -> 取最后隐状态 -> Linear -> Sigmoid
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # MindSpore 2.10 的 Initializer 需通过 initializer(init, shape, dtype) 使用
        self.W_xh = Parameter(
            initializer(Normal(0.1), (embed_dim, hidden_dim), ms.float32),
            name="W_xh",
        )
        self.W_hh = Parameter(
            initializer(Normal(0.1), (hidden_dim, hidden_dim), ms.float32),
            name="W_hh",
        )
        self.b_h = Parameter(
            initializer(Zero(), (hidden_dim,), ms.float32),
            name="b_h",
        )

        # 输出层
        self.fc = nn.Dense(hidden_dim, 1, weight_init=Normal(0.1), bias_init=Zero())
        self.sigmoid = nn.Sigmoid()

        # 算子
        self.tanh = ops.Tanh()
        self.matmul = ops.MatMul()
        self.transpose = ops.Transpose()

    def construct(self, x):
        # x: [B, L] int32
        emb = self.embedding(x)                       # [B, L, E]
        emb_t = self.transpose(emb, (1, 0, 2))        # [L, B, E]

        B = x.shape[0]
        L = emb_t.shape[0]

        h = ops.zeros((B, self.hidden_dim), dtype=ms.float32)  # [B, H]

        for t in range(L):
            x_t = emb_t[t]                                       # [B, E]
            h = self.tanh(
                self.matmul(x_t, self.W_xh)
                + self.matmul(h, self.W_hh)
                + self.b_h
            )                                                    # [B, H]

        logits = self.fc(h)                                      # [B, 1]
        prob = self.sigmoid(logits)                              # [B, 1]
        return prob


# ----------------------------------------------------------------------
# 4. 训练与评价
# ----------------------------------------------------------------------
def compute_accuracy(pred_prob, labels):
    pred = (pred_prob > 0.5).astype(np.float32)
    return float((pred == labels).mean())


def evaluate(model, dataset, batch_size):
    model.set_train(False)
    losses, accs = [], []
    loss_fn = nn.BCELoss(reduction="mean")
    for x, y in make_batches(dataset, batch_size, shuffle=False):
        x_t = Tensor(x, ms.int32)
        y_t = Tensor(y, ms.float32)
        prob = model(x_t)
        loss = loss_fn(prob, y_t)
        losses.append(loss.asnumpy())
        accs.append(compute_accuracy(prob.asnumpy(), y))
    return float(np.mean(losses)), float(np.mean(accs))


def train(model, train_ds, test_ds, epochs=5, batch_size=32, lr=1e-3):
    loss_fn = nn.BCELoss(reduction="mean")
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

    def forward_fn(x, y):
        prob = model(x)
        loss = loss_fn(prob, y)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(x, y):
        loss, grads = grad_fn(x, y)
        optimizer(grads)
        return loss

    for epoch in range(1, epochs + 1):
        model.set_train(True)
        step_losses, step_accs = [], []
        for x, y in make_batches(train_ds, batch_size, shuffle=True):
            x_t = Tensor(x, ms.int32)
            y_t = Tensor(y, ms.float32)
            loss = train_step(x_t, y_t)
            prob = model(x_t)
            step_losses.append(loss.asnumpy())
            step_accs.append(compute_accuracy(prob.asnumpy(), y))

        tr_loss, tr_acc = float(np.mean(step_losses)), float(np.mean(step_accs))
        te_loss, te_acc = evaluate(model, test_ds, batch_size)
        print(f"Epoch {epoch:2d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}")


# ----------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------
def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SST-2")
    train_path = os.path.join(data_dir, "train.tsv")
    test_path = os.path.join(data_dir, "test.tsv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练集：{train_path}")

    print("加载数据 ...")
    train_data = load_tsv(train_path)

    if os.path.exists(test_path):
        test_data = load_tsv(test_path)
    else:
        print("未找到 test.tsv，从 train.tsv 切分 10% 作为测试集")
        np.random.shuffle(train_data)
        split = int(len(train_data) * 0.9)
        test_data = train_data[split:]
        train_data = train_data[:split]

    print(f"训练样本数：{len(train_data)}，测试样本数：{len(test_data)}")

    vocab = build_vocab([s for s, _ in train_data], min_freq=1)
    print(f"词表大小：{len(vocab)}")

    max_len = 30
    embed_dim = 64
    hidden_dim = 64
    batch_size = 32
    epochs = 5
    lr = 1e-3

    train_ds = SST2Dataset(train_data, vocab, max_len)
    test_ds = SST2Dataset(test_data, vocab, max_len)

    model = SimpleRNNClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        pad_idx=vocab["<pad>"],
    )
    model.set_train(True)

    total_params = sum(p.size for p in model.trainable_params())
    print(f"可训练参数量：{total_params}")

    train(model, train_ds, test_ds, epochs=epochs, batch_size=batch_size, lr=lr)


if __name__ == "__main__":
    main()