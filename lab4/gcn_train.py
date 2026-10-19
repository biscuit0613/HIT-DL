"""训练与评估入口：单次运行可供命令行调用，批量实验由 gcn_experiments.py 组织。"""

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from gcn_data import CLASS_NAMES, load_cora
from gcn_model import build_model, count_parameters

DEFAULT_CONFIG = {
    "epochs": 200,
    "hidden": 16,
    "layers": 2,
    "dropout": 0.5,
    "heads": 8,
    "lr": 0.01,
    "weight_decay": 5e-4,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_loss(logits, y, mask):
    return F.cross_entropy(logits[mask], y[mask]).item()


def masked_acc(logits, y, mask):
    pred = logits[mask].argmax(dim=1)
    return (pred == y[mask]).float().mean().item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    result = {"logits": logits}
    for split, mask in (
        ("train", data.train_mask),
        ("val", data.val_mask),
        ("test", data.test_mask),
    ):
        result[f"{split}_loss"] = masked_loss(logits, data.y, mask)
        result[f"{split}_acc"] = masked_acc(logits, data.y, mask)
    return result


def run_single(cfg, data, seed, device="cuda", verbose=False):
    """训练一个配置并返回验证集最优 checkpoint 的完整指标。"""
    config = dict(DEFAULT_CONFIG)
    config.update(cfg)
    set_seed(seed)

    model = build_model(config["model"], data.num_features,
                        int(data.y.max()) + 1, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    history = {key: [] for key in
               ("train_loss", "val_loss", "train_acc", "val_acc", "test_acc")}
    best_val_acc, best_val_loss = -1.0, float("inf")
    best_epoch = 0
    best_state = None

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        metrics = evaluate(model, data)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(metrics["val_loss"])
        history["train_acc"].append(metrics["train_acc"])
        history["val_acc"].append(metrics["val_acc"])
        history["test_acc"].append(metrics["test_acc"])

        improved = (
            metrics["val_acc"] > best_val_acc + 1e-12
            or (abs(metrics["val_acc"] - best_val_acc) <= 1e-12
                and metrics["val_loss"] < best_val_loss)
        )
        if improved:
            best_val_acc = metrics["val_acc"]
            best_val_loss = metrics["val_loss"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  epoch {epoch:03d} | loss {loss.item():.4f} | "
                  f"val {metrics['val_acc']:.4f} | test {metrics['test_acc']:.4f}")

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    model.load_state_dict(best_state)
    final = evaluate(model, data)
    logits = final["logits"]
    test_pred = logits[data.test_mask].argmax(dim=1).cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()
    labels = np.arange(len(CLASS_NAMES))
    per_class_f1 = f1_score(test_true, test_pred, labels=labels,
                            average=None, zero_division=0)
    conf = confusion_matrix(test_true, test_pred, labels=labels)

    run = {
        "model": config["model"],
        "seed": seed,
        "config": config,
        "param_count": count_parameters(model),
        "train_time": train_time,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "val_acc": final["val_acc"],
        "val_loss": final["val_loss"],
        "train_acc": final["train_acc"],
        "train_loss": final["train_loss"],
        "test_acc": final["test_acc"],
        "test_loss": final["test_loss"],
        "test_f1_macro": float(f1_score(test_true, test_pred, average="macro",
                                        zero_division=0)),
        "test_f1_weighted": float(f1_score(test_true, test_pred,
                                           average="weighted", zero_division=0)),
        "per_class_f1": per_class_f1.tolist(),
        "confusion": conf.tolist(),
        "history": history,
    }
    artifacts = {
        "state_dict": best_state,
        "all_logits": logits.detach().cpu().numpy(),
        "test_pred": test_pred,
        "test_true": test_true,
    }
    return run, artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gcn",
                        choices=["mlp", "gcn", "sage", "gat"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _, data = load_cora()
    data = data.to(device)
    cfg = vars(args).copy()
    cfg.pop("device")
    cfg.pop("seed")
    cfg["weight_decay"] = cfg.pop("weight_decay")
    run, _ = run_single(cfg, data, args.seed, device, verbose=True)
    print(json.dumps({k: v for k, v in run.items() if k != "history"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
