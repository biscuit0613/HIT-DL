# 实验二：Transformer 机器翻译（英语→中文）

Opus-100（en-zh 子集，采样 3 万句对，规模对齐 Multi30k）。基于 PyTorch 实现
Transformer encoder-decoder 架构，源语言英文按单词分词，目标语言中文按字符分词，
包含 BLEU 评估、beam search 解码、注意力可视化与训练日志结构化存储。

## 运行

所有命令在仓库根目录执行（`.venv` 在根目录）。

```bash
# 1. 训练 Transformer 模型（默认 30 个 epoch，训练集采样 3 万句）
.venv/bin/python -u lab2/mt_train.py --epochs 30 --batch-size 128 --train-size 30000

# 2. 生成所有可视化（学习曲线、BLEU 对比、注意力热图、样例翻译）
.venv/bin/python -u lab2/mt_viz.py
```

训练日志、词表、模型权重存于 `lab2/results/`，可视化图片存于 `lab2/figures/`。

可选参数示例：

```bash
# 快速测试（5 个 epoch，5000 句训练集）
.venv/bin/python -u lab2/mt_train.py --epochs 5 --train-size 5000

# 更大模型（d_model=512, 6 层）
.venv/bin/python -u lab2/mt_train.py --d-model 512 --enc-layers 6 --dec-layers 6
```

## 文件

| 文件 | 内容 |
| --- | --- |
| `mt_train.py` | 入口，训练循环、BLEU 评估（中文字符级）、测试集对比（greedy vs beam） |
| `mt_data.py` | Opus-100 en-zh 数据加载与采样、Vocab 构建（英文单词级/中文字符级）、DataLoader 封装 |
| `mt_model.py` | TransformerMT 模型（encoder-decoder + positional encoding） |
| `mt_decode.py` | 贪心解码、beam search 解码、带注意力权重的解码 |
| `mt_viz.py` | 生成学习曲线、BLEU 对比图、注意力热图、样例表格 |
| `results/` | 训练日志 JSON、词表、最佳模型权重、epoch 检查点、完整训练输出 |
| `figures/` | 学习曲线、BLEU 对比、注意力热图 PNG、样例翻译 txt |

## 数据集

Opus-100（`Helsinki-NLP/opus-100`，en-zh 子集，从 Hugging Face Datasets 自动下载）：
- 训练集：从原始 100 万句中随机采样 3 万句（`--train-size` 可调），规模对齐 Multi30k
- 验证集 2000 句，测试集 2000 句（官方划分）
- 源语言（src）= 英语，按空格分词；目标语言（tgt）= 中文，按字符切分（无需分词器依赖）

选择英译中而非英德互译，是因为中文翻译结果更直观、更便于报告中人工判读翻译质量。

## 实现亮点（超出 demo 示例的三处功能）

1. **Beam search 解码**（`mt_decode.py`）：实现 beam width=5 的束搜索算法，
   带长度惩罚归一化，测试集 BLEU 相比贪心解码有稳定提升。

2. **注意力可视化**（`mt_viz.py:plot_attention_heatmap`）：通过 PyTorch hook
   抓取 decoder 最后一层的交叉注意力权重，生成英文单词-中文字符对齐热图，
   直观展示 Transformer 的 attention 机制。

3. **结构化训练日志**（`results/training_log.json`）：每个 epoch 存储
   train_loss、val_loss、val_bleu、学习率、翻译样例等，支持断点续传分析与
   报告中的数据表格/曲线绘制，完整记录超参数配置与最终测试集结果。

## 结果

训练 30 epoch 后（d_model=256, nhead=8, 3 层 encoder+decoder，训练集 3 万句）的具体数值见
`results/training_log.json`、`results/train_output.txt` 与 `figures/` 下的可视化图表。
