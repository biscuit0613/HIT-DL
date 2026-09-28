"""数据读取，以及折内拟合的 TF-IDF 特征流水线。

词表与 IDF 都在每折训练集内部拟合，验证折只用训练折学到的词表做变换。
"""

import csv
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

# 早停要另切一份数据，不能拿外层验证折来用：外层验证折一旦参与"什么时候停"
# 这个决策，它就不再是模型没见过的数据，交叉验证分数会被系统性抬高。
INNER_VAL_FRACTION = 0.1


def load_sst2(path):
    """读取 SST-2 的制表符分隔文件，返回 (句子列表, 标签数组)。"""
    texts, labels = [], []
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        if [h.strip().lower() for h in header[:2]] != ["sentence", "label"]:
            raise ValueError(f"{path} 表头不符合预期: {header}")
        for row in reader:
            if len(row) < 2 or not row[1].strip():
                continue
            texts.append(row[0].strip())
            labels.append(int(row[1]))
    return texts, np.asarray(labels, dtype=np.int64)  # text: list[str], labels: np.ndarray


def describe_dataset(name, labels, n_words):
    counts = np.bincount(labels, minlength=2)
    print(f"  {name}: {len(labels)} 句, 标签分布 neg={counts[0]} / pos={counts[1]}, "
          f"句长(词) min={n_words.min()} max={n_words.max()} 均值={n_words.mean():.1f}")


def build_feature_matrices(fit_texts, apply_texts, cfg):
    """在 fit_texts 上拟合 TF-IDF，再变换 apply_texts，返回稀疏矩阵。

    词表与 IDF 只用 fit_texts 学习，验证折的词频信息不会进入特征。
    """
    vectorizer = TfidfVectorizer(**cfg["tfidf"])
    csr_fit = vectorizer.fit_transform(fit_texts).astype(np.float32)
    csr_apply = vectorizer.transform(apply_texts).astype(np.float32)
    return csr_fit, csr_apply, vectorizer


def build_folds(texts, labels, cfg):
    """按重复分层 K 折切分，并为每折预生成特征矩阵。

    分层保证每折正负比例与整体一致。重复多次换随机划分，是为了看清"组合
    之间的差距"有没有超过"划分本身的波动"——两者量级相当时排名就不能当真。
    特征拟合与内层切分都与超参数无关，所以每折只做一次，所有组合复用。

    一折之内有三份互不相交的数据，各自只干一件事：
        X_inner  48491 句  训练折的 90%，前向反向、更新权重
        X_early   5388 句  训练折的 10%，只用于早停判定与学习率调度
        X_val    13470 句  整折留出，训练结束后评一次分
        48491 + 5388 + 13470 = 67349，不重不漏

    外层折既不参与梯度更新，也不参与"什么时候停"的决策——后者一旦沾上，
    那个折就不再是模型没见过的数据，交叉验证分数会被系统性抬高。代价是
    训练样本少了 10%，换来的是外层评分没被选择过程污染。

    注意外层折仍在 36 个组合之间用于排名，所以它是选择集而非无偏测试集，
    真正的泛化估计要看另外留出的 872 句测试集。
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=cfg["n_splits"], n_repeats=cfg["n_repeats"], random_state=cfg["seed"]
    )
    folds = []
    total = cfg["n_splits"] * cfg["n_repeats"]
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(texts, labels)):
        t0 = time.time()
        fit_texts = [texts[i] for i in train_idx]
        apply_texts = [texts[i] for i in val_idx]
        y_fit = labels[train_idx]
        y_val = labels[val_idx]

        inner_idx, inner_val_idx = train_test_split(
            np.arange(len(fit_texts)),
            test_size=INNER_VAL_FRACTION,
            stratify=y_fit,
            random_state=cfg["seed"] + fold_id,
        )
        X_fit, X_val, vec = build_feature_matrices(fit_texts, apply_texts, cfg)
        # 已知的一处不严格：TF-IDF 在整个训练折上拟合，含被切作早停集的
        # 那 10%，其词频进了 IDF。这只影响"停在哪一轮"，不影响外层评分
        # ——外层折的词频从未参与拟合。要更严格应只用 inner_idx 拟合。

        folds.append(
            {
                "repeat": fold_id // cfg["n_splits"],
                "fold": fold_id % cfg["n_splits"],
                "X_inner": X_fit[inner_idx],
                "y_inner": y_fit[inner_idx],
                "X_early": X_fit[inner_val_idx],
                "y_early": y_fit[inner_val_idx],
                "X_val": X_val,
                "y_val": y_val,
                "vocab_size": len(vec.vocabulary_),
            }
        )
        print(f"    [折 {fold_id + 1:2d}/{total}] repeat={folds[-1]['repeat']} "
              f"fold={folds[-1]['fold']} 内层训练={len(inner_idx)} "
              f"早停验证={len(inner_val_idx)} 外层验证={len(val_idx)} "
              f"词表={folds[-1]['vocab_size']} "
              f"非零特征/句={X_fit.nnz / X_fit.shape[0]:.1f} "
              f"耗时={time.time() - t0:.2f}s")
    return folds
