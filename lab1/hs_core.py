"""公共基础设施：输出格式、设备选择、组合格式化、结果整理，不含实验逻辑。"""

import csv
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["negative", "positive"]

# 每批次都新建稀疏张量，跳过程序化不变量检查可省去重复校验的开销
torch.sparse.check_sparse_tensor_invariants.disable()


def silence_sparse_warnings():
    """屏蔽 PyTorch 建稀疏张量的两条提示（CSR 仍是 beta、不变量检查已关闭）。

    joblib 的工作进程导入后会重置警告过滤器，所以每个组合开跑前要再设一次。
    """
    warnings.filterwarnings("ignore", message=".*Sparse.*")


silence_sparse_warnings()


# --------------------------------------------------------------------------
# 输出格式
# --------------------------------------------------------------------------
def banner(title, char="=", width=78):
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def sub_banner(title, width=78):
    print()
    print("-" * width)
    print(f"  >> {title}")
    print("-" * width)


def format_combo(combo):
    return (f"hidden={combo['hidden_size']:<4d} dropout={combo['dropout']:<4.1f} "
            f"lr={combo['lr']:<7.4g} wd={combo['weight_decay']:<7.4g}")


def pick_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# --------------------------------------------------------------------------
# 结果整理
# --------------------------------------------------------------------------
def summarize(record):
    accs = np.array([f["accuracy"] for f in record["folds"]])
    per_fold = {m: np.array([f[m] for f in record["folds"]])
                for m in ("precision_macro", "recall_macro", "f1_macro")}
    return {
        **record["combo"],
        "folds": record["folds"],
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "mean_precision": float(per_fold["precision_macro"].mean()),
        "mean_recall": float(per_fold["recall_macro"].mean()),
        "mean_f1": float(per_fold["f1_macro"].mean()),
        "std_f1": float(per_fold["f1_macro"].std()),
        "mean_best_epoch": float(np.mean([f["best_epoch"] for f in record["folds"]])),
    }


def save_cv_csv(summaries, per_fold, path):
    fields = ["hidden_size", "dropout", "lr", "weight_decay", "mean_accuracy",
              "std_accuracy", "mean_precision", "mean_recall", "mean_f1", "std_f1",
              "mean_best_epoch"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row[k] for k in fields})

    fold_path = path.with_name("cv_per_fold.csv")
    with open(fold_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["hidden_size", "dropout", "lr", "weight_decay",
                         "repeat", "fold", "accuracy", "precision_macro",
                         "recall_macro", "f1_macro", "best_epoch", "train_seconds"])
        for row in per_fold:
            writer.writerow(row)
    return path, fold_path


def to_per_fold_rows(records):
    rows = []
    for rec in records:
        for f in rec["folds"]:
            rows.append([rec["combo"]["hidden_size"], rec["combo"]["dropout"],
                         rec["combo"]["lr"], rec["combo"]["weight_decay"],
                         f["repeat"], f["fold"], f["accuracy"], f["precision_macro"],
                         f["recall_macro"], f["f1_macro"], f["best_epoch"],
                         f["train_seconds"]])
    return rows
