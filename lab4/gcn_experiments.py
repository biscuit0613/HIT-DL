"""批量实验：主对比、GCN 参数扫描、标签率与结构鲁棒性。"""

import argparse
import json
import os
import time

import numpy as np
import torch

from gcn_data import (CLASS_NAMES, dataset_stats, drop_undirected_edges,
                      label_rate_masks, load_cora)
from gcn_train import run_single

RESULTS = "results"

MODEL_CONFIGS = {
    "mlp": dict(model="mlp", hidden=16, layers=2, dropout=0.2,
                lr=0.01, weight_decay=5e-4, epochs=200),
    "gcn": dict(model="gcn", hidden=16, layers=2, dropout=0.5,
                lr=0.01, weight_decay=5e-4, epochs=200),
    "sage": dict(model="sage", hidden=16, layers=2, dropout=0.5,
                 lr=0.01, weight_decay=5e-4, epochs=200),
    "gat": dict(model="gat", hidden=8, layers=2, heads=8, dropout=0.6,
                lr=0.005, weight_decay=5e-4, epochs=200),
}


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, ensure_ascii=False, indent=2)


def summarize(runs, keys):
    summary = {}
    for key in keys:
        values = np.array([r[key] for r in runs], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "values": values.tolist(),
        }
    return summary


def print_summary(name, runs):
    accs = np.array([r["test_acc"] for r in runs])
    vals = np.array([r["val_acc"] for r in runs])
    epochs = np.array([r["best_epoch"] for r in runs])
    print(f"{name:>5s} | test {accs.mean():.4f} ± {accs.std(ddof=1):.4f} | "
          f"val {vals.mean():.4f} ± {vals.std(ddof=1):.4f} | "
          f"best epoch {epochs.mean():.0f} | params {runs[0]['param_count']}")


def run_main(data, device, seeds):
    print("\n===== 主实验：MLP / GCN / GraphSAGE / GAT =====")
    all_runs = {}
    all_logits = {}
    best_models = {}
    for model, cfg in MODEL_CONFIGS.items():
        runs = []
        states = {}
        for seed in seeds:
            run, artifacts = run_single(cfg, data, seed, device)
            runs.append(run)
            states[seed] = artifacts["state_dict"]
            all_logits[f"{model}_seed{seed}"] = artifacts["all_logits"]
        all_runs[model] = runs
        best = max(runs, key=lambda r: (r["best_val_acc"], -r["best_val_loss"]))
        best_models[model] = {
            "config": cfg,
            "seed": best["seed"],
            "state_dict": states[best["seed"]],
            "metrics": {k: v for k, v in best.items() if k != "history"},
        }
        print_summary(model, runs)

    save_json(f"{RESULTS}/main/results.json", {
        "seeds": seeds,
        "configs": MODEL_CONFIGS,
        "runs": all_runs,
        "summary": {m: summarize(r, ["test_acc", "val_acc", "train_acc",
                                     "test_f1_macro", "best_epoch",
                                     "train_time"])
                    for m, r in all_runs.items()},
    })
    np.savez_compressed(
        f"{RESULTS}/main/logits.npz",
        y=data.y.cpu().numpy(),
        train_mask=data.train_mask.cpu().numpy(),
        val_mask=data.val_mask.cpu().numpy(),
        test_mask=data.test_mask.cpu().numpy(),
        **all_logits,
    )
    torch.save(best_models, f"{RESULTS}/main/best_models.pt")
    print("主实验完成，结果已保存。")


def run_sweeps(data, device):
    print("\n===== GCN 参数扫描 =====")
    base = dict(MODEL_CONFIGS["gcn"])
    sweeps = {
        "hidden": [4, 8, 16, 32, 64, 128],
        "layers": [1, 2, 3, 4, 5],
        "dropout": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8],
        "lr": [0.001, 0.005, 0.01, 0.02, 0.05],
        "weight_decay": [0.0, 1e-5, 1e-4, 5e-4, 5e-3, 1e-2],
    }
    seeds = list(range(5))
    output = {}
    for param, values in sweeps.items():
        print(f"\n-- 扫描 {param} --")
        entries = []
        for value in values:
            cfg = dict(base)
            cfg[param] = value
            runs = [run_single(cfg, data, seed, device)[0] for seed in seeds]
            accs = np.array([r["test_acc"] for r in runs])
            vals = np.array([r["val_acc"] for r in runs])
            entries.append({
                "value": value,
                "param_count": runs[0]["param_count"],
                "runs": [{k: r[k] for k in ("seed", "test_acc", "val_acc",
                                            "test_f1_macro", "best_epoch",
                                            "train_time")} for r in runs],
                "test_mean": float(accs.mean()),
                "test_std": float(accs.std(ddof=1)),
                "val_mean": float(vals.mean()),
                "val_std": float(vals.std(ddof=1)),
            })
            print(f"  {param}={value:<8} test {accs.mean():.4f} ± "
                  f"{accs.std(ddof=1):.4f} | params {runs[0]['param_count']}")
        output[param] = {"baseline": base[param], "values": values,
                         "entries": entries}

    print("\n-- 扫描 lr × weight_decay 网格 --")
    lrs = [0.001, 0.005, 0.01, 0.02]
    wds = [0.0, 1e-4, 5e-4, 5e-3]
    grid = []
    for lr in lrs:
        row = []
        for wd in wds:
            cfg = dict(base)
            cfg["lr"], cfg["weight_decay"] = lr, wd
            runs = [run_single(cfg, data, seed, device)[0] for seed in range(3)]
            accs = np.array([r["test_acc"] for r in runs])
            row.append({"lr": lr, "weight_decay": wd,
                        "test_mean": float(accs.mean()),
                        "test_std": float(accs.std(ddof=1))})
        grid.append(row)
        print("  " + " | ".join(f"{cell['test_mean']:.4f}" for cell in row))
    output["grid_lr_wd"] = {"lrs": lrs, "weight_decays": wds, "grid": grid}

    print("\n-- 扫描 GAT 注意力头数 --")
    gat_base = dict(MODEL_CONFIGS["gat"])
    heads_values = [1, 2, 4, 8, 16]
    head_entries = []
    for heads in heads_values:
        cfg = dict(gat_base)
        cfg["heads"] = heads
        runs = [run_single(cfg, data, seed, device)[0] for seed in seeds]
        accs = np.array([r["test_acc"] for r in runs])
        head_entries.append({
            "value": heads,
            "param_count": runs[0]["param_count"],
            "runs": [{k: r[k] for k in ("seed", "test_acc", "val_acc",
                                        "test_f1_macro", "best_epoch",
                                        "train_time")} for r in runs],
            "test_mean": float(accs.mean()),
            "test_std": float(accs.std(ddof=1)),
        })
        print(f"  heads={heads:<3} test {accs.mean():.4f} ± "
              f"{accs.std(ddof=1):.4f} | params {runs[0]['param_count']}")
    output["gat_heads"] = {"baseline": 8, "values": heads_values,
                           "entries": head_entries,
                           "seeds": seeds}

    save_json(f"{RESULTS}/sweeps/results.json", {
        "base_config": base, "seeds": seeds, "sweeps": output,
    })
    print("参数扫描完成。")


def run_label_rate(data_cpu, device, seeds):
    print("\n===== 标签率实验 =====")
    rates = [1, 2, 5, 10, 20, 40]
    models = ["mlp", "gcn", "sage", "gat"]
    records = []
    for rate in rates:
        for seed in seeds:
            train_mask, val_mask, test_mask = label_rate_masks(
                data_cpu, rate, seed=100 + seed
            )
            data = data_cpu.clone()
            data.train_mask = train_mask.to(device)
            data.val_mask = val_mask.to(device)
            data.test_mask = test_mask.to(device)
            for model in models:
                cfg = dict(MODEL_CONFIGS[model])
                run, _ = run_single(cfg, data, seed, device)
                records.append({
                    "rate": rate,
                    "train_size": int(train_mask.sum()),
                    "model": model,
                    "seed": seed,
                    "test_acc": run["test_acc"],
                    "val_acc": run["val_acc"],
                    "test_f1_macro": run["test_f1_macro"],
                    "best_epoch": run["best_epoch"],
                })
        print(f"  {rate} 个/类完成")

    summary = {}
    for model in models:
        summary[model] = {}
        for rate in rates:
            runs = [r for r in records if r["model"] == model and r["rate"] == rate]
            accs = np.array([r["test_acc"] for r in runs])
            summary[model][str(rate)] = {
                "mean": float(accs.mean()),
                "std": float(accs.std(ddof=1)),
                "train_size": runs[0]["train_size"],
            }
    save_json(f"{RESULTS}/label_rate/results.json", {
        "rates": rates, "seeds": seeds, "records": records, "summary": summary,
    })
    print("标签率实验完成。")


def run_robustness(data_cpu, device, seeds):
    print("\n===== 图结构鲁棒性实验 =====")
    ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    models = ["mlp", "gcn", "sage", "gat"]
    records = []
    for ratio in ratios:
        for seed in seeds:
            data = data_cpu.clone()
            data.edge_index = drop_undirected_edges(
                data_cpu.edge_index, ratio, seed=2000 + seed
            )
            num_edges = int(data.edge_index.size(1) // 2)
            for model in models:
                cfg = dict(MODEL_CONFIGS[model])
                run, _ = run_single(cfg, data, seed, device)
                records.append({
                    "ratio": ratio,
                    "num_edges": num_edges,
                    "model": model,
                    "seed": seed,
                    "test_acc": run["test_acc"],
                    "val_acc": run["val_acc"],
                    "test_f1_macro": run["test_f1_macro"],
                })
        print(f"  删除比例 {ratio:.2f} 完成（剩余 {num_edges} 条无向边）")

    summary = {}
    for model in models:
        summary[model] = {}
        for ratio in ratios:
            runs = [r for r in records if r["model"] == model and r["ratio"] == ratio]
            accs = np.array([r["test_acc"] for r in runs])
            summary[model][str(ratio)] = {
                "mean": float(accs.mean()),
                "std": float(accs.std(ddof=1)),
                "num_edges": runs[0]["num_edges"],
            }
    save_json(f"{RESULTS}/robustness/results.json", {
        "ratios": ratios, "seeds": seeds, "records": records,
        "summary": summary,
    })
    print("鲁棒性实验完成。")


def run_normalization(data_raw, data_norm, device):
    print("\n===== 特征归一化对照 =====")
    models = ["mlp", "gcn"]
    seeds = list(range(5))
    records = []
    for tag, data in (("raw", data_raw.to(device)),
                      ("row_norm", data_norm.to(device))):
        for model in models:
            cfg = dict(MODEL_CONFIGS[model])
            runs = [run_single(cfg, data, seed, device)[0] for seed in seeds]
            accs = np.array([r["test_acc"] for r in runs])
            vals = np.array([r["val_acc"] for r in runs])
            records.append({
                "features": tag,
                "model": model,
                "test_mean": float(accs.mean()),
                "test_std": float(accs.std(ddof=1)),
                "val_mean": float(vals.mean()),
                "val_std": float(vals.std(ddof=1)),
            })
            print(f"  {tag:>8s} {model:>3s}: test {accs.mean():.4f} ± "
                  f"{accs.std(ddof=1):.4f} | val {vals.mean():.4f}")
    save_json(f"{RESULTS}/normalization/results.json",
              {"seeds": seeds, "records": records})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "main", "sweeps", "label-rate",
                                 "robustness", "normalization"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--main-seeds", type=int, default=10)
    parser.add_argument("--label-seeds", type=int, default=5)
    parser.add_argument("--robust-seeds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset, data_cpu = load_cora(normalize=True)
    _, data_raw = load_cora(normalize=False)
    data = data_cpu.to(device)
    os.makedirs(RESULTS, exist_ok=True)
    save_json(f"{RESULTS}/dataset_stats.json", dataset_stats(data_raw))

    start = time.perf_counter()
    if args.stage in ("all", "main"):
        run_main(data, device, list(range(args.main_seeds)))
    if args.stage in ("all", "sweeps"):
        run_sweeps(data, device)
    if args.stage in ("all", "label-rate"):
        run_label_rate(data_cpu, device, list(range(args.label_seeds)))
    if args.stage in ("all", "robustness"):
        run_robustness(data_cpu, device, list(range(args.robust_seeds)))
    if args.stage in ("all", "normalization"):
        run_normalization(data_raw, data_cpu, device)
    print(f"\n全部完成，用时 {time.perf_counter() - start:.1f} s，设备 {device}。")


if __name__ == "__main__":
    main()
