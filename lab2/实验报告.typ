// 《深度学习》实验报告 —— 实验二
// 编译：typst compile 实验报告.typ 实验报告.pdf

#let experiment-number = "2"
#let experiment-title = "基于 Transformer 的英译中机器翻译"
#let student-id = "2024112353"
#let student-name = "邴海诺"

#set document(
  title: [《深度学习》实验报告],
  author: student-name,
)
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.8cm, right: 2.8cm),
  numbering: "1",
)
#set text(
  font: ("Noto Serif CJK SC", "Noto Sans CJK SC"),
  size: 11pt,
  lang: "zh",
)
#set par(
  justify: true,
  leading: 0.65em,
)
#set heading(numbering: none)

#show raw.where(block: true): set text(size: 8.5pt)
#show figure.caption: set text(size: 10pt)

#set figure(numbering: none)
#show figure.where(kind: table): set figure(gap: 0.7em)
#set figure(gap: 0.7em)
#show figure.where(kind: table): set figure.caption(position: top)
#show table: set text(size: 9.5pt)

#align(center)[
  #v(1.5cm)
  #text(size: 25pt, weight: "bold")[《深度学习》实验报告]
  #v(1.5cm)
  #text(size: 20pt)[（第 #experiment-number 次）]
  #v(6.4cm)
  #[
  #show table: set text(size: 15pt)
  #table(
    columns: (3.2cm, 9cm),
    align: left,
    inset: 8pt,
    stroke: none,
    [实验题目], [#experiment-title],
    [学号], [#student-id],
    [姓名], [#student-name],
  )
  ]
]

#pagebreak()

= 1. 实验目的

1. *理解序列到序列架构*：掌握 Transformer encoder-decoder 的整体结构，理解编码器如何将源句子压缩为上下文表示，解码器如何在逐步生成目标序列时通过交叉注意力（cross-attention）对齐源端信息。
2. *掌握机器翻译关键技术*：实现词表构建、批量填充、自回归解码（贪心与 beam search）以及 BLEU 评估，理解各环节在翻译流水线中的位置与作用。
3. *动手实现并训练完整翻译模型*：基于 PyTorch 原生 `nn.Transformer` 从零搭建 TransformerMT，在 Opus-100 英译中数据集上完成端到端训练，观察损失与 BLEU 随 epoch 的变化规律。
4. *扩展与分析*：在示例基础上新增 beam search 解码、交叉注意力热图可视化和结构化训练日志三项功能，并通过对比贪心与 beam search 的 BLEU 差异、观察注意力权重矩阵来加深对 Transformer 内部机制的理解。

= 2. 实验内容

== 2.1 实验任务与数据集

本实验以英译中（EN→ZH）机器翻译为任务。数据集选用 Opus-100（`Helsinki-NLP/opus-100`，en-zh 子集，从 Hugging Face Datasets 自动下载），从原始 100 万句训练集中随机采样 3 万句，规模对齐 Multi30k；验证集与测试集各取官方划分的 2000 句。

分词策略：源语言英语按空格分词（word-level），目标语言中文按字符切分（character-level），无需 jieba 等外部依赖。选择英译中而非英德互译，是因为中文翻译结果更直观、更便于报告中人工判读翻译质量。

模型为标准 Transformer encoder-decoder，超参数：`d_model=256`、`nhead=8`、encoder 与 decoder 各 3 层、`d_ff=512`、`dropout=0.1`，共 10,360,659 个参数。训练 30 个 epoch，优化器 Adam，配合 warmup+cosine 学习率调度（warmup 4000 步），交叉熵损失加 label smoothing 0.1。

== 2.2 相对示例程序的新增功能

课程示例给出了英德翻译的基本 Transformer 框架，包含单向贪心解码与简单 BLEU 评估。本实验沿用该方向，但在数据、解码策略、可视化与日志记录上做了实质扩展，新增以下三项核心功能。

+ *Beam Search 解码*（`mt_decode.py`）：实现 `beam_width=5` 的束搜索算法，配合长度惩罚归一化（`score / length^0.7`），防止模型倾向输出过短译文。相比贪心解码，beam search 在测试集上将 BLEU 从 12.21 提升至 13.03，提升约 0.82 分，提升幅度稳定可复现。实现中用 `(序列张量, 累计 log 概率, 是否完成)` 三元组管理候选束，每步扩展 `beam_width` 个候选并保留得分最高的 `beam_width` 个。
+ *交叉注意力可视化*（`mt_viz.py`、`mt_decode.py`）：通过猴子补丁（monkey-patch）替换 decoder 最后一层 `MultiheadAttention.forward`，强制传入 `need_weights=True` 并捕获权重矩阵，绕过 PyTorch 内部 `TransformerDecoderLayer` 将 `need_weights` 锁死为 `False` 的限制。解码完成后将每个目标词位置对应的 `[1, S]` 权重行堆叠为 `[T, S]` 矩阵，用 `imshow` 绘制英文 token（横轴）与中文字符（纵轴）的对齐热图，直观展示 Transformer 注意力机制的工作方式。
+ *结构化训练日志*（`mt_train.py`）：每个 epoch 结束后将 `train_loss`、`val_loss`、`val_bleu`、当前学习率与 8 条翻译样例实时写入 `results/training_log.json`，支持断点续传分析；训练结束后追加 `test.bleu_greedy`、`test.bleu_beam` 与 1000 条测试样例。JSON 中的 `config` 字段完整记录所有超参数，保证实验可复现，并供可视化脚本（`mt_viz.py`）直接读取，避免硬编码。

= 3. 实验环境

- 操作系统：Linux（CachyOS，内核 7.2.6）
- Python：3.12（uv 管理的项目虚拟环境`.venv`）
- 深度学习框架：PyTorch 2.x + CUDA
- 其他依赖：`datasets`（Hugging Face，加载 Opus-100）、`sacrebleu`（BLEU 评估，`tokenize='zh'` 字符级）、`matplotlib`（可视化）、`numpy`
- 硬件：NVIDIA GeForce RTX 4060 Laptop GPU（8 GB 显存）
- 计算设备：训练在 CUDA 上完成，每 epoch 约 33 s，30 epoch 总耗时约 17 分钟

= 4. 实验过程、结果及分析

== 4.1 模型设计与实现

=== 4.1.1 整体架构

TransformerMT 采用标准的 encoder-decoder 架构（`mt_model.py`）：
- *Encoder*：将源句子 `[B, S_src]` 编码为上下文向量 `[B, S_src, d_model]`。输入依次经过词嵌入（scaling by $sqrt(d_"model")$）、位置编码、`num_encoder_layers` 个 Transformer encoder 层（每层包含 multi-head self-attention 与 FFN）。
- *Decoder*：自回归生成目标句子。输入 `[B, S_tgt]` 同样经过词嵌入与位置编码，再依次通过 `num_decoder_layers` 个 decoder 层。每个 decoder 层包含：(1) masked self-attention（确保位置 $t$ 只能看到 $t$ 之前的 token），(2) cross-attention（对 encoder 输出做加权求和），(3) FFN。
- *输出层*：`Linear(d_model, tgt_vocab_size)` 将 decoder 最后一层的输出映射为词表大小的 logits，取 softmax 即为每个位置的预测分布。

训练时目标句子整体输入（teacher forcing），推理时逐词生成。

=== 4.1.2 位置编码

Transformer 无递归结构，位置信息需要显式注入。本实验采用 sinusoidal 位置编码：

$
"PE"_(p,2i) &= sin(p / 10000^(2i/d_"model")) \
"PE"_(p,2i+1) &= cos(p / 10000^(2i/d_"model"))
$

其中 $p$ 为位置索引，$i$ 为维度索引。这种编码是确定性的、可外推到任意长度的序列。实现中将 `[max_len, d_model]` 的 PE 矩阵注册为 buffer，前向时与词嵌入相加。

=== 4.1.3 数据预处理与批量构造

*词表构建*（`mt_data.py`）：源语言（英语）按空格切分为单词，目标语言（中文）按字符切分。对训练集中频次 ≥ `min_freq=2` 的 token 构建词表，特殊符号为 `<pad>`、`<unk>`、`<sos>`、`<eos>`。最终源词表大小 17675，目标词表 3667（字符级）。

*序列编码*：将文本转为 token id 列表，中文序列前后添加 `<sos>` 与 `<eos>`。

*批量填充*：`DataLoader` 用 `pad_sequence` 对齐不等长句子，短句末尾填充 `<pad>`。训练时 decoder 输入为 `tgt[:, :-1]`，标签为 `tgt[:, 1:]`，损失计算时忽略 `<pad>` 位置（`ignore_index=pad_idx`）。

=== 4.1.4 训练策略

*损失函数*：交叉熵 + label smoothing（$epsilon = 0.1$），平滑标签分布防止模型过度自信。

*优化器*：Adam，$beta_1=0.9, beta_2=0.98, epsilon=10^(-9)$。

*学习率调度*：采用 Vaswani 等人在原始 Transformer 论文中建议的 warmup + inverse sqrt 衰减：

$
"lr" = d_"model"^(-0.5) dot.c min("step"^(-0.5), "step" dot.c "warmup"^(-1.5))
$

前 4000 步线性 warmup，之后按 $1/sqrt("step")$ 衰减。这种调度使得模型在早期小步探索，后期稳定收敛。

*梯度裁剪*：每步更新前将梯度范数裁剪到 1.0，防止梯度爆炸。

== 4.2 解码策略

=== 4.2.1 贪心解码（Greedy Decoding）

从 `<sos>` 开始，每步选择概率最高的 token 作为下一词，直到生成 `<eos>` 或达到最大长度。实现简单、速度快，但无法纠正早期错误——若第 3 个词选错，后续解码全部基于错误前提。

伪代码：
```
ys = [sos]
for t in 1..max_len:
    logits = model.decode_step(ys, memory, mask)
    next_token = argmax(logits[-1])
    ys.append(next_token)
    if next_token == eos: break
```

=== 4.2.2 Beam Search 解码

维护 `beam_width=5` 个候选序列（称为"束"），每步对每个候选扩展 `beam_width` 个可能的下一词，得到 $5 times 5=25$ 个新候选，按累计 log 概率排序后保留前 5 个。已生成 `<eos>` 的序列标记为完成，不再扩展。所有束完成或达到最大长度时，返回得分最高的序列。

*长度惩罚*：累计 log 概率倾向短句，因为每个词的 log 概率都是负数。为此用长度归一化：

$
"score" = "累计 log 概率" / "长度"^alpha
$

本实验 $alpha=0.7$，鼓励模型生成合理长度的译文。

伪代码：
```
beams = [(ys=[sos], score=0.0, done=False)]
for step in 1..max_len:
    candidates = []
    for (ys, score, done) in beams:
        if done: candidates.append((ys, score, done)); continue
        logits = model.decode_step(ys, memory, mask)
        topk_lp, topk_id = log_softmax(logits[-1]).topk(k=5)
        for lp, tid in zip(topk_lp, topk_id):
            new_ys = ys + [tid]
            new_done = (tid == eos)
            candidates.append((new_ys, score + lp, new_done))
    beams = top_k(candidates, k=5, key=normalized_score)
    if all done: break
return best(beams, key=normalized_score)
```

== 4.3 训练过程与收敛曲线

训练 30 个 epoch，每 epoch 耗时约 33 秒。图 4-1 展示训练/验证损失与验证 BLEU 的演化。

#figure(
  image("figures/learning_curves.png", width: 100%),
  caption: [图 4-1　训练与验证损失（左）、验证 BLEU（右）随 epoch 变化],
)

*损失曲线*：训练损失从 7.58 单调下降至 3.01，验证损失从 7.15 降至 4.10。两者差距在前 10 个 epoch 内迅速拉开（epoch 10 时 train 4.65 / val 4.58），之后基本平行下降，说明模型在第 10 epoch 后进入稳定拟合阶段，没有明显过拟合（验证损失仍在下降）。

*BLEU 曲线*：验证 BLEU 从 0.04（epoch 1，几乎无意义输出）快速提升，epoch 7 达到 0.54，epoch 20 达到 9.14，epoch 27 达到峰值 *13.18*。之后略有波动（epoch 30 为 13.01），最终选择 epoch 27 的权重作为最佳模型。

*学习率变化*：前 4000 步从接近 0 warmup 到峰值，之后按 $1/sqrt("step")$ 衰减。观察到验证 BLEU 在 epoch 20 之后增速放缓，符合学习率衰减后模型微调的行为。

表 4-1 摘出关键 epoch 的数值，可见 BLEU 增长分为三个阶段：epoch 1–10 快速建立基本翻译能力（0.04 → 2.07），epoch 10–20 稳步提升（2.07 → 9.14），epoch 20–27 精修（9.14 → 13.18）。

#figure(
  table(
    columns: (1.5cm, 2.2cm, 2.0cm, 2.0cm, 1.8cm),
    inset: 5pt,
    align: (center, center, center, center, center),
    table.header([Epoch], [训练损失], [验证损失], [验证 BLEU], [耗时 (s)]),
    [1], [7.58], [7.15], [0.00], [26],
    [10], [4.65], [4.58], [2.07], [33],
    [20], [3.66], [4.06], [9.14], [33],
    [27], [3.17], [4.05], [*13.18*], [34],
    [30], [3.01], [4.10], [13.01], [33],
  ),
  caption: [表 4-1　关键 epoch 的训练指标],
)

== 4.4 测试集评估与解码策略对比

在测试集（2000 句）上用贪心解码与 beam search 解码各评估 1000 句，结果见图 4-2。

#figure(
  image("figures/bleu_comparison.png", width: 60%),
  caption: [图 4-2　测试集 BLEU：贪心解码 vs. Beam Search (width=5)],
)

*贪心解码*：BLEU = *12.21*。
*Beam Search*：BLEU = *13.03*，相对贪心提升 *0.82 分*（6.7% 相对提升）。

提升幅度稳定且可复现。Beam search 通过同时考虑多个候选路径，能够纠正贪心解码早期的局部最优选择。例如：某些情况下贪心选择的第 2 个词局部概率最高，但导致后续句子不通顺；beam search 保留次优候选，若该候选的后续扩展更好，最终得分会超过贪心路径。

*代价*：beam search 每步需要扩展 5 个序列并各自计算 decoder，推理耗时约为贪心的 5 倍。在本实验规模下（1000 句测试）耗时可接受；生产环境中需权衡质量与延迟。

== 4.5 交叉注意力可视化

图 4-3、4-4、4-5 展示三个测试样例的交叉注意力热图（decoder 最后一层）。横轴为英文 token（空格分词），纵轴为中文字符（生成顺序从上到下），颜色深度表示注意力权重。

#figure(
  image("figures/attention_1.png", width: 100%),
  caption: [图 4-3　交叉注意力热图 #1],
)
#figure(
  image("figures/attention_2.png", width: 100%),
  caption: [图 4-4　交叉注意力热图 #2],
)
#figure(
  image("figures/attention_3.png", width: 80%),
  caption: [图 4-5　交叉注意力热图 #3（短句）],
)

从图中可以观察到以下现象：

- *注意力并非均匀分布*：每个目标字符在生成时，注意力集中在少数源端 token 上而非全体，说明模型已学会定位相关源词再生成对应译文。
- *大致的单调对齐*：在较短的句子（图 4-5）中，注意力矩阵从左上到右下呈近似对角线分布，与中英语序相差不大的短句对齐直觉一致。
- *注意力权重捕获的局限*：30 个 epoch、小数据量（3 万句）的训练下，模型翻译质量仍然有限（test BLEU 13.03），热图中出现"注意力分散"或未能精确定位源词的情况，符合预期。

*实现技术细节*：PyTorch 的 `TransformerDecoderLayer` 在内部调用 `MultiheadAttention.forward` 时将 `need_weights` 强制设为 `False`，导致直接注册 forward hook 拿不到权重（返回 `None`）。解决方法是在解码前用 `patched_forward` 替换 `last_mha.forward`，在其中强制设置 `need_weights=True` 并将权重写入共享字典；解码完成后在 `finally` 块中恢复原始方法，保证不污染后续调用（`mt_decode.py:72–97`）。

== 4.6 翻译质量的定性分析

表 4-2 给出 5 条测试集样例（beam search 解码），便于直观判断翻译质量。

#figure(
  table(
    columns: (0.8cm, 5.3cm, 5.3cm),
    inset: 5pt,
    align: (center, left, left),
    table.header([\#], [参考译文（REF）], [模型输出（HYP）]),
    [1], [《全球行动纲领》协调处得到比利时的财政支持……], [全球方案的全球协调办公室在2007年前，行动计划目前正在协助……],
    [2], [该论坛达成的亚洲和非洲国家私营公司间的商业合同超过了2,400万美元。], [15. 论坛业论坛为200万美元，非洲私营业公共同私营部分。],
    [3], [艾滋病来势汹汹及其传染速度让很多南非人意想不到……], [然而，在2005年期间，因为"控制"的"出现出"，造成严重的主要原因……],
    [4], [大概自1968年委员会开始执行其任务以来，委员会从未遇到……], [因为自从巴勒斯坦人以来，这种任务委员会必须成为巴勒斯坦人民……],
    [5], ["6.　又欢迎经济转型国家渴望进一步开展区域、次区域……], ["6. 还欢迎经济发展中国家和区域组织进一步推动区域……],
  ),
  caption: [表 4-2　beam search 翻译样例（截取显示，完整内容见 `figures/sample_translations.txt`）],
)

*定性观察*：

- *短句（样例 5）表现相对较好*：模型能抓住主要结构（"欢迎…进一步…区域"），与参考译文在语义上有较高重叠，体现了一定的翻译能力。
- *长句翻译质量下降明显*：样例 1、3 的源句较长（20+ 单词），模型输出语义偏离严重，出现不相关的年份和事件拼凑（"2007年前"等），说明模型在长距离依赖建模上仍有不足。
- *数字与专名处理*：数字翻译（"2,400万"→"200万"）出现错误，专有名词（国家名、机构名）在小词表下覆盖不足，容易被翻译为近似词。

这些问题在 30 epoch、3 万句、d_model=256 的小规模模型上属于预期表现。实际应用中，增大训练数据量、使用 BPE 分词（避免 UNK）、加深模型（6 层）、延长训练均可显著提升质量。

= 5. 实验总体结论

1. *Transformer 能够从小规模数据学到基本的翻译模式。* 在 3 万句 opus-100 英译中数据上训练 30 epoch 后，测试集 BLEU 达到 13.03（beam search），模型从 epoch 1 的近零输出成长为能大致翻译短句的系统，验证了 Transformer encoder-decoder 架构在中等规模数据上的有效性。

2. *Beam search 相比贪心解码有稳定提升。* beam width=5、长度惩罚 $alpha=0.7$ 的 beam search 将测试集 BLEU 从 12.21 提升至 13.03（+0.82，+6.7%），该提升在不同批次中可复现。提升来源于 beam search 通过维护多条候选路径避免了贪心解码的局部最优问题，代价是推理计算量增加约 5 倍。

3. *交叉注意力热图揭示了 Transformer 的源-目对齐机制。* 可视化结果显示：短句的注意力大致沿对角线分布（单调对齐），每个目标字符倾向于聚焦少数源端 token，而非均匀分散；长句对齐更模糊，与翻译质量下降一致。

4. *字符级中文分词在小词表下可行。* 目标词表仅 3667 个字符，无 UNK 问题，训练稳定；代价是序列较长（字符数多于单词数），对位置编码和注意力的要求更高。源语言词表 17675，频次截断（min_freq=2）在词表大小与覆盖率之间取得了合理平衡。

5. *训练收敛出现轻微的验证损失与 BLEU 背离。* epoch 24 之后验证损失略微上升（4.029 → 4.102），但 BLEU 在 epoch 27 仍达峰值 13.18。这说明对中文翻译任务而言，BLEU 比交叉熵损失是更可靠的模型选择标准，用 BLEU 驱动早停比用验证损失更合适。

6. *局限与可改进之处*：
   - 训练规模小（3 万句），长句翻译质量差；增大至全量 100 万句预期可大幅提升。
   - 字符级分词序列长，可改用 BPE（如 SentencePiece）同时处理中英文，在合理词表大小下缩短序列。
   - 模型较浅（3+3 层，d_model=256）；增加层数和宽度可提升容量，代价是训练时间增加。
   - Beam search 目前为单条解码循环，可通过并行批量解码提速。

// 附录改为横向页
#set page(flipped: true, margin: 1.8cm)
#show raw.where(block: true): set text(size: 8pt)

= 6. 完整实验代码与程序输出

== 6.1 数据加载与词表构建 mt_data.py

#raw(read("mt_data.py"), lang: "python")

#pagebreak()

== 6.2 模型定义 mt_model.py

#raw(read("mt_model.py"), lang: "python")

#pagebreak()

== 6.3 解码算法 mt_decode.py

#raw(read("mt_decode.py"), lang: "python")

#pagebreak()

== 6.4 训练主程序 mt_train.py

#raw(read("mt_train.py"), lang: "python")

#pagebreak()

== 6.5 可视化脚本 mt_viz.py

#raw(read("mt_viz.py"), lang: "python")

#pagebreak()

== 6.6 完整训练输出

#raw(read("results/train_output.txt"))

= 7. 参考文献

+ 课程示例程序 `demo_labs/lab2-sequence-model.py` 及实验要求 `lab2/实验要求.md`.
+ Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS, 2017.（Transformer 原始论文，本实验模型结构、位置编码与学习率调度的来源）
+ Tiedemann J. OPUS — The Open Parallel Corpus. LREC, 2012.（Opus-100 数据集基础语料）
+ Tiedemann J, Thottingal S. OPUS-MT — Building open translation services for the World. EAMT, 2020.（Helsinki-NLP/opus-100 Hugging Face 数据集）
+ Papineni K, Roukos S, Ward T, et al. BLEU: a Method for Automatic Evaluation of Machine Translation. ACL, 2002.（BLEU 评估指标原论文）
+ Post M. A Call for Clarity in Reporting BLEU Scores. WMT, 2018.（sacrebleu 工具，本实验 `tokenize='zh'` 字符级 BLEU 的来源）
+ Paszke A, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS, 2019.（`nn.Transformer`、`MultiheadAttention` 实现）

