"""SST-2 BERT 微调入口：三种迁移策略训练、验证集选优、早停、测试集评估。"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from bert_data import build_dataloaders
from bert_model import (
    MODE_DEFAULT_EPOCHS,
    MODE_DEFAULT_LR,
    MODE_LABELS,
    build_model,
    count_params,
)

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

LAB_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_DELTA = 5e-4


def parse_args():
    parser = argparse.ArgumentParser(description="BERT SST-2 情感分类")
    parser.add_argument("--mode", choices=list(MODE_DEFAULT_LR), default="finetune")
    parser.add_argument("--data-dir", default=str(LAB_DIR / "data" / "SST-2"))
    parser.add_argument("--model-path", default=str(LAB_DIR.parent / "demo_labs" / "bert_localpath"))
    parser.add_argument("--results-dir", default=str(LAB_DIR / "results"))
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None, help="轮数上限，默认按策略 5/10/20")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_to_device(batch):
    return (
        batch["input_ids"].to(DEVICE),
        batch["attention_mask"].to(DEVICE),
        batch["token_type_ids"].to(DEVICE),
        batch["labels"].to(DEVICE),
    )


class Trainer:
    def __init__(self, model, train_loader, valid_loader, args, lr):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.args = args

        # 只把 requires_grad 的参数交给优化器：特征抽取策略下主干被冻结，
        # 若连同冻结参数一起传入，AdamW 仍会为它们维护动量状态，白占显存
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.weight_decay)
        # warmup + 线性衰减：前 10% 的步数 lr 从 0 线性升到设定值，之后线性衰减回 0。
        # 微调一开始就用满 lr 更新预训练权重容易破坏学到的表示，warmup 让更新幅度平滑放大
        total_steps = len(train_loader) * args.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps,
        )
        # 交叉熵内部含 log_softmax，直接吃模型输出的未归一化 logits
        self.loss_fn = nn.CrossEntropyLoss()
        self.history = []
        self.best_f1 = -1.0
        self.best_epoch = 0
        self.best_state = None

    def train_one_epoch(self, epoch):
        self.model.train()
        lr_start = self.optimizer.param_groups[0]["lr"]
        total_loss, total_num = 0.0, 0
        steps = len(self.train_loader)
        epoch_start = time.time()

        for step, batch in enumerate(self.train_loader, start=1):
            input_ids, attention_mask, token_type_ids, labels = batch_to_device(batch)
            self.optimizer.zero_grad()
            # 取 .logits 自己算损失，不用模型内置的 loss 输出，口径更可控；
            # attention_mask 传入后模型会在注意力中忽略 padding 位置
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits
            loss = self.loss_fn(logits, labels)
            loss.backward()
            # 按全局范数把梯度裁剪到 1.0 以内：个别 batch 的梯度可能异常大，
            # 不裁剪容易一步就把微调中的预训练权重带偏
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.args.grad_clip
            )
            self.optimizer.step()
            self.scheduler.step()  # 调度器逐 batch 走一步，与 warmup/衰减的步数口径一致

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_num += bs
            if step % self.args.log_every == 0 or step == steps:
                pct = 100.0 * step / steps
                print(
                    f"  [Epoch {epoch}] step {step:4d}/{steps} ({pct:5.1f}%) "
                    f"| loss {loss.item():.4f} | lr {self.scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        avg_loss = total_loss / max(total_num, 1)
        elapsed = time.time() - epoch_start
        print(f"[Epoch {epoch}] 平均训练 loss {avg_loss:.4f} | 用时 {elapsed:.1f}s", flush=True)
        return avg_loss, lr_start, elapsed

    @torch.no_grad()
    def evaluate(self, loader, collect_texts=False):
        # eval() 关闭 dropout；no_grad 不建反向图，显存与速度都更省
        self.model.eval()
        total_loss, total_num = 0.0, 0
        all_preds, all_labels, all_probs = [], [], []
        all_texts = []

        for batch in loader:
            input_ids, attention_mask, token_type_ids, labels = batch_to_device(batch)
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits
            loss = self.loss_fn(logits, labels)
            # 保留 softmax 概率：可视化阶段要用它看预测置信度、定位"自信的误分类"
            probs = torch.softmax(logits, dim=-1)

            total_loss += loss.item() * labels.size(0)
            total_num += labels.size(0)
            all_preds.append(probs.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            if collect_texts:
                all_texts.extend(batch["texts"])

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()
        return {
            "loss": total_loss / max(total_num, 1),
            "acc": float((preds == labels).mean()),
            "f1": float(f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)),
            "preds": preds,
            "labels": labels,
            "probs": probs,
            "texts": all_texts,
        }

    def run(self, mode_dir):
        train_start = time.time()
        no_improve = 0

        for epoch in range(1, self.args.epochs + 1):
            print(f"\n===== Epoch {epoch}/{self.args.epochs} =====", flush=True)
            train_loss, lr_start, elapsed = self.train_one_epoch(epoch)
            val = self.evaluate(self.valid_loader)
            print(
                f"[Epoch {epoch}] val loss {val['loss']:.4f} | val acc {val['acc']:.4f} "
                f"| val F1 {val['f1']:.4f}"
            )
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val["loss"],
                    "val_acc": val["acc"],
                    "val_f1": val["f1"],
                    "lr": lr_start,
                    "elapsed_s": elapsed,
                }
            )

            # 按验证集 F1 选优而不是 loss：F1 是最终要汇报的指标，更贴近任务效果；
            # 加 MIN_DELTA 阈值，过滤验证集波动造成的"假提升"
            if val["f1"] > self.best_f1 + MIN_DELTA:
                self.best_f1 = val["f1"]
                self.best_epoch = epoch
                # 最优权重克隆到 CPU 暂存：训练继续占用 GPU，结束后再 load 回来做测试
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                if not self.args.no_save:
                    torch.save(self.best_state, mode_dir / "best_model.pt")
                print(f"[Epoch {epoch}] val F1 提升，记录最优模型（best F1 = {self.best_f1:.4f}）")
                no_improve = 0
            else:
                # 连续 patience 个 epoch 验证 F1 没有实质提升就停下，避免继续训练走向过拟合
                no_improve += 1
                if self.args.patience > 0 and no_improve >= self.args.patience:
                    print(f"验证集 F1 连续 {no_improve} 个 epoch 未提升，提前停止训练")
                    break

        total_elapsed = time.time() - train_start
        return self.best_epoch, self.best_f1, total_elapsed


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = MODE_DEFAULT_LR[args.mode]
    if args.epochs is None:
        args.epochs = MODE_DEFAULT_EPOCHS[args.mode]
    set_seed(args.seed)
    # 允许 TF32 做矩阵乘：40 系显卡上提速明显，对分类精度的影响可忽略
    torch.set_float32_matmul_precision("high")

    results_dir = Path(args.results_dir)
    mode_dir = results_dir / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"设备: {DEVICE}（{device_name}）")
    print(f"策略: {args.mode}（{MODE_LABELS[args.mode]}） | lr {args.lr:.2e} | epochs {args.epochs} "
          f"| batch {args.batch_size} | max_len {args.max_len} | seed {args.seed}")

    # tokenizer 与模型加载同一份本地权重目录，词表一致
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    bundle = build_dataloaders(args, tokenizer)

    print(
        f"数据集: 训练 {bundle['n_train']} | 验证 {bundle['n_valid']} | 测试 {bundle['n_test']}"
    )
    for split, ratio in bundle["pos_ratio"].items():
        print(f"  {split} 正类占比: {ratio:.4f}")
    stats = bundle["length_stats"]
    print(
        f"训练集 token 长度: mean {stats['mean']:.1f} | p50 {stats['p50']:.0f} "
        f"| p90 {stats['p90']:.0f} | p99 {stats['p99']:.0f} | max {stats['max']} "
        f"| 截断比例 {stats['truncated_ratio']:.4f}"
    )

    (results_dir / "split_info.json").write_text(
        json.dumps(
            {
                "n_train": bundle["n_train"],
                "n_valid": bundle["n_valid"],
                "n_test": bundle["n_test"],
                "pos_ratio": bundle["pos_ratio"],
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "max_len": args.max_len,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (results_dir / "length_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model = build_model(args.model_path, args.mode)
    total_params, trainable_params = count_params(model)
    print(f"参数量: 总 {total_params:,} | 可训练 {trainable_params:,}")

    trainer = Trainer(model, bundle["train_loader"], bundle["valid_loader"], args, args.lr)
    warmup_steps = int(len(bundle["train_loader"]) * args.epochs * args.warmup_ratio)
    print(
        f"训练步数: {len(bundle['train_loader'])} 步/epoch | warmup {warmup_steps} 步 "
        f"| 梯度裁剪 {args.grad_clip}"
    )

    print("\n========== 开始训练 ==========")
    best_epoch, best_f1, total_elapsed = trainer.run(mode_dir)
    print(f"\n训练结束：best epoch {best_epoch}（val F1 = {best_f1:.4f}） | 总用时 {total_elapsed:.1f}s")

    # 先恢复验证集最优 epoch 的权重再评估：保证测试成绩来自最优模型，而不是最后一个 epoch
    model.load_state_dict(trainer.best_state)
    print("\n========== 测试集最终评估（使用 best epoch 权重） ==========")
    test = trainer.evaluate(bundle["test_loader"], collect_texts=True)
    report = classification_report(
        test["labels"], test["preds"], target_names=["Negative", "Positive"],
        digits=4, zero_division=0,
    )
    matrix = confusion_matrix(test["labels"], test["preds"])
    print(f"测试集准确率 = {test['acc']:.4f} | F1 = {test['f1']:.4f}")
    print(report)
    print("混淆矩阵（行=真实，列=预测）:")
    print(matrix)

    (mode_dir / "classification_report.txt").write_text(
        f"准确率 = {test['acc']:.4f} | F1 = {test['f1']:.4f}\n\n{report}\n混淆矩阵:\n{matrix}\n",
        encoding="utf-8",
    )

    log = {
        "config": vars(args),
        "mode": args.mode,
        "mode_label": MODE_LABELS[args.mode],
        "device": device_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "n_train": bundle["n_train"],
        "n_valid": bundle["n_valid"],
        "n_test": bundle["n_test"],
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "total_elapsed_s": total_elapsed,
        "epochs": trainer.history,
        "test": {
            "acc": test["acc"],
            "f1": test["f1"],
            "loss": test["loss"],
            "confusion_matrix": matrix.tolist(),
        },
    }
    (mode_dir / "training_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (mode_dir / "test_predictions.json").write_text(
        json.dumps(
            {
                "texts": test["texts"],
                "labels": test["labels"].tolist(),
                "preds": test["preds"].tolist(),
                "probs": test["probs"].tolist(),
                "acc": test["acc"],
                "f1": test["f1"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        f"策略: {args.mode}（{MODE_LABELS[args.mode]}） | lr {args.lr:.2e} | epochs {args.epochs} "
        f"| batch {args.batch_size} | max_len {args.max_len} | seed {args.seed}",
        f"可训练参数: {trainable_params:,} / {total_params:,}",
        "",
    ]
    for row in trainer.history:
        lines.append(
            f"Epoch {row['epoch']}: train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f} "
            f"val_f1={row['val_f1']:.4f} lr={row['lr']:.2e} 用时={row['elapsed_s']:.1f}s"
        )
    lines += [
        "",
        f"最优 epoch: {best_epoch}（val_f1={best_f1:.4f}） | 训练总用时: {total_elapsed:.1f}s",
        f"测试集: acc={test['acc']:.4f} f1={test['f1']:.4f}",
        f"混淆矩阵: {matrix.tolist()}",
    ]
    (mode_dir / "epoch_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n结果已写入 {mode_dir}/")


if __name__ == "__main__":
    main()
