# 实验三：基于 BERT 预训练模型微调的文本情感分类

SST-2 情感二分类（电影评论 positive/negative）。加载预训练
bert-base-uncased，对比三种迁移策略：全量微调、冻结特征抽取（仅训练分类头）、
从零训练，含独立验证集、学习率调度、早停、可视化与错误分析。

## 运行

所有命令在仓库根目录执行（`.venv` 在根目录）。

```bash
# 1. 训练三种策略（依次运行，每次约 4~46 分钟）
.venv/bin/python -u lab3/bert_train.py --mode finetune
.venv/bin/python -u lab3/bert_train.py --mode feature-extract
.venv/bin/python -u lab3/bert_train.py --mode scratch

# 2. 生成所有可视化（长度分布、训练曲线、策略对比、混淆矩阵、置信度、误分类样例）
.venv/bin/python -u lab3/bert_viz.py
```

训练日志、模型权重、测试预测存于 `lab3/results/{mode}/`，可视化图片存于 `lab3/figures/`。

可选参数示例：

```bash
# 快速冒烟测试（2000 条训练样本、1 个 epoch、不保存权重）
.venv/bin/python -u lab3/bert_train.py --mode finetune --train-size 2000 --epochs 1 --no-save

# 调整训练超参数
.venv/bin/python -u lab3/bert_train.py --mode finetune --lr 3e-5 --batch-size 16 --epochs 5

# 冻结特征抽取的 lr 灵敏度对照（报告表 4-4 第四行，结果存于 results/sensitivity/）
.venv/bin/python -u lab3/bert_train.py --mode feature-extract --lr 5e-4 --results-dir lab3/results/sensitivity
```

## 文件

| 文件 | 内容 |
| --- | --- |
| `bert_train.py` | 入口，训练循环、warmup+线性衰减调度、梯度裁剪、早停、测试集评估 |
| `bert_data.py` | SST-2 TSV 读取、分层三分（train/val/test）、预编码数据集、动态 padding 拼接 |
| `bert_model.py` | 三种策略的模型构建（微调/冻结特征抽取/从零）、可训练参数统计 |
| `bert_viz.py` | 长度分布、学习曲线、策略对比柱状图、混淆矩阵、置信度分布、误分类样例 |
| `results/` | 各策略训练日志 JSON、最优权重、完整训练输出、测试预测与分类报告 |
| `figures/` | 上述可视化 PNG 与误分类样例 txt |

## 数据集

SST-2（Stanford Sentiment Treebank，GLUE 子集，`data/SST-2/` 自带 TSV）：
- 原始训练集 67,349 条，分层切分 8% 作为验证集 → 训练 61,961 / 验证 5,388 / 测试 872
- 标签为 0/1 二分类（negative/positive），三集正类占比均在 51%~56%
- WordPiece 分词后长度 mean 13.3 / p90 27 / p99 44，`max_len=64` 截断率约 0.01%

## 实现亮点（超出 demo 示例的功能）

1. **训练/验证/测试三划分与模型选择**（`bert_data.py:split_train_val`）：
   demo 将 test.tsv 同时用作验证和测试；本实现分层切出 8% 验证集，
   按验证集 F1 保存最优 checkpoint 并早停（patience=3），test.tsv 仅用于最终评估。

2. **三种迁移策略对比**（`bert_model.py:build_model`）：
   全量微调（lr 2e-5）/ 冻结特征抽取（按参数名前缀冻结骨干网络，仅训练
   1,538 个分类头参数，lr 1e-3）/ 从零训练（随机初始化，lr 与微调一致保证公平），
   同一 seed、同一数据划分下对比，验证预训练权重的实际贡献。

3. **训练改进**：warmup 10% + 线性衰减学习率调度（按 batch step）、
   梯度裁剪（max_norm=1.0）、动态 padding（预编码 + collate 内按 batch
   最长补齐）替代 demo 的逐样本定长 64 填充。

4. **可视化与错误分析**（`bert_viz.py`）：训练曲线、三策略指标对比、
   混淆矩阵、正确/错误预测的置信度分布、最自信误分类样例表。

## 结果

| 策略 | 可训练参数 | 验证集最优 F1 | 测试集准确率 | 测试集 F1 |
| --- | --- | --- | --- | --- |
| 全量微调 | 109,483,778 | 0.9541 | **0.9197** | **0.9215** |
| 冻结特征抽取 | 1,538 | 0.8424 | 0.7924 | 0.8113 |
| 从零训练 | 109,483,778 | 0.9226 | 0.7878 | 0.7933 |

微调策略显著优于从零训练（测试准确率 +13.2 点）。冻结特征抽取以约 1/12 的
训练成本（225 s vs 2746 s）达到与从零训练几乎相同的测试成绩，且验证→测试
落差小得多（2.5 点 vs 12.6 点），说明预训练表示作为固定特征已蕴含可观的语言
知识，但其充分释放仍依赖微调。详细分析与误分类讨论见 `实验报告.typ`
（编译产物 `82-邴海诺.pdf`）。
