// 《深度学习》实验报告 —— 实验三
// 编译：typst compile 实验报告.typ 82-邴海诺.pdf

#let experiment-number = "3"
#let experiment-title = "基于 BERT 预训练模型微调的文本情感分类"
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

1. *理解预训练与微调范式*：使用预训练 BERT 模型，在 SST-2 情感二分类数据集上进行微调，并与“从零开始训练”的同结构 BERT 做对比，验证预训练 + 微调在 NLP 下游任务中的迁移学习优势。掌握「大规模语料预训练 + 下游任务微调」的迁移学习框架，理解预训练语言模型如何将其在大规模语料中学到的通用语言知识迁移到具体的文本分类任务。
2. *掌握预训练模型的使用方法*：学习使用 Hugging Face `transformers` 加载预训练 BERT 模型与 `AutoTokenizer`，将预训练模型适配到句子级分类任务，并完成数据编码、批量构造与训练循环的完整流程。
3. *对比不同迁移策略*：在同一数据划分与训练配置下，比较全量微调、冻结特征抽取与从零训练三种策略的测试性能，定量理解预训练带来的收益，并通过学到的分类头参数体会「特征抽取」与「微调」两种迁移路径的差异。
4. *扩展与分析*：在示例基础上新增训练/验证/测试三划分与验证集选优早停、三种迁移策略对比、学习率调度与动态 padding 等训练改进、以及完整的可视化与错误分析四项功能，并据此分析模型行为。

= 2. 实验内容

== 2.1 实验任务与数据集

本实验的任务是 SST-2（Stanford Sentiment Treebank）句子级情感二分类：给定一条英文电影评论文本，判断其情感为正面（Positive, 1）或负面（Negative, 0）。

数据集为 GLUE 基准中的 SST-2 子集，以 TSV 格式提供（`sentence\tlabel`，含表头）：

- 训练集 `train.tsv`：67,349 条（已含标签），本实验按标签分层切出 8% 作为验证集，其余用于训练；
- 测试集 `test.tsv`：872 条（官方 dev 划分，含标签），仅用于最终评估，不参与任何训练与模型选择。

文本编码使用 `bert-base-uncased` 对应的 WordPiece 分词器（词表 30,522），最大序列长度 64；文本全部小写化，标点与词间空格保留为一个 WordPiece 子词。训练集 token 长度统计：均值 13.3、p50 为 10、p90 为 27、p99 为 44，超过 64 的仅 4 条（0.0065%），说明 max_len=64 几乎不损失信息，且句子整体较短。

模型的三个策略超参数：全量微调 lr $2 times 10^(-5)$、冻结特征抽取 lr $10^(-3)$（头部参数少，需要更大的步长）、从零训练 lr $2 times 10^(-5)$（与全量微调一致，保证对比只差在初始权重）；batch size 32；轮数上限按策略的收敛速度设为 5/10/20，由 patience=3 的早停控制实际轮数；AdamW 权重衰减 0.01，warmup 比例为总步数的 10%，梯度裁剪阈值 1.0，随机种子 42。

== 2.2 相对示例程序的新增功能

课程示例（`demo_labs/lab3-bert.py`）给出了 BERT 微调的基本框架：TSV 读取、预训练/从零两个模型、朴素训练循环与最终测试。本实验在多方面做了实质扩展，新增以下四项功能：

+ *训练/验证/测试三划分与验证集选优早停*（`bert_data.py`、`bert_train.py`）：示例直接用测试集在每个 epoch 观察性能，既无法得到无偏的最终评估，也不能据其选择模型。本实验从训练集中按标签分层切出 8% 作为验证集（5,388 条），每个 epoch 结束后在验证集上计算 loss/准确率/F1，按验证集 F1（而非验证 loss）保存最优 checkpoint，并配置 patience=3 的早停；`test.tsv` 全程隔离，只在最后加载 best 权重评估一次。
+ *三种迁移策略对比*（`bert_model.py`）：示例只对比「微调 vs 从零」。本实验增加「冻结特征抽取」策略——冻结 BERT 主干全部约 1.09 亿参数、只训练 768×2+2 = 1,538 个分类头参数，构成「全量微调 / 冻结特征抽取 / 从零训练」三档对比，正好对应迁移学习中「微调（fine-tuning）」与「特征抽取（feature extraction）」两条经典路线，并结合测试性能定量说明各自特点。
+ *训练改进：学习率调度、梯度裁剪与动态 padding*（`bert_data.py`、`bert_train.py`）：示例以固定学习率训练且逐样本补齐到固定长度 64。本实验改用 warmup（10% 步数线性升温）+ 线性衰减调度，对线性层参数做范数 1.0 的梯度裁剪；数据管线改为「初始化时批量预编码 + 批内 pad 到当前 batch 最长序列（动态 padding）」，并以 attention mask 显式屏蔽补齐位。由于训练集九成序列短于 32，训练首位大幅降低。
+ *可视化与错误分析*（`bert_viz.py`）：生成训练曲线（train/val loss、val acc、val F1）、三策略测试指标对比柱状图、混淆矩阵热图、正确/错误样本的预测置信度分布，并输出测试集上「最自信」的误分类样例清单，从数据层面分析模型在哪些类型的句子上出错。

= 3. 实验环境

- 操作系统：Linux（CachyOS，内核 7.2.6）
- Python：3.12（uv 管理的项目虚拟环境 `.venv`）
- 深度学习框架：PyTorch 2.11 + CUDA（cu128 构建），启用 TF32 矩阵乘法加速
- 预训练模型：`bert-base-uncased`（本地权重，约 440 MB）
- 其他依赖：`transformers` 5.17.0（`AutoTokenizer`、`AutoModelForSequenceClassification`、线性调度器）、`scikit-learn`（分层划分、F1、分类报告、混淆矩阵）、`matplotlib`（可视化）、`numpy`
- 硬件：NVIDIA GeForce RTX 4060 Laptop GPU（8 GB 显存，单卡训练）
- 计算设备：训练在 CUDA 上完成，全量 61,961 条训练样本、batch size 32、动态 padding 下每个 epoch 1,937 步；三个策略的具体耗时见 4.4 节。

= 4. 实验过程、结果及分析

== 4.1 数据预处理与动态 padding

数据管线分三步：(1) 解析 TSV，跳过表头与非法行，校验标签只取 0/1；(2) 按标签分层切出 8% 验证集（`sklearn.train_test_split(stratify=...)`，seed=42），保证三个子集正类占比一致；(3) 用 `AutoTokenizer` 一次性批量编码全部文本（截断到 64），训练时由 collate 函数把每个 batch 补齐到该批最长序列，并同步生成 attention mask（补齐位置为 0）与全零 token_type_ids（单句任务）。

动态 padding 相比示例的固定长度补齐：本实验训练集每批实际长度约在 20–30 之间浮动（见 #figure([长度分布])），若补齐到 64，约一半的注意力计算与显存都花在 pad token 上；按批动态补齐后，等效序列长度下降近一半，训练速度明显提升，而模型可看到的有效信息（截断比例仅 0.0065%）不变。

#figure(
  image("figures/length_distribution.png", width: 82%),
  caption: [图 4-1　训练集 token 长度分布（WordPiece），max_len=64 几乎不截断],
)

三划分后的样本数与正类占比：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: center,
    [子集], [样本数], [正类占比], [说明],
    [训练], [61,961], [0.5578], [分层剩余部分],
    [验证], [5,388], [0.5579], [分层切出 8%，用于选优与早停],
    [测试], [872], [0.5092], [官方 dev 划分，仅最终评估],
  ),
  caption: [表 4-1　数据集划分统计],
)

== 4.2 模型与三种迁移策略

基础模型为 `bert-base-uncased`：12 层 Transformer encoder，隐藏层 768 维，12 个注意力头，约 1.09 亿参数。分类头为单层线性层（768→2）。三个策略共用同一 backbone 与分类头结构，仅初始化与可训练范围不同：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    [策略], [初始化], [可训练参数], [学习率], [说明],
    [全量微调], [预训练权重], [109,483,778], [$2 times 10^(-5)$], [主干+分类头全部更新],
    [冻结特征抽取], [预训练权重], [1,538], [$1 times 10^(-3)$], [仅更新分类头，主干冻结],
    [从零训练], [随机初始化], [109,483,778], [$2 times 10^(-5)$], [结构相同但无预训练知识],
  ),
  caption: [表 4-2　三种迁移策略的配置对比],
)

冻结的实现方式是遍历 `named_parameters()`，将非 `classifier` 前缀的参数 `requires_grad` 置为 False；优化器只接收 `requires_grad` 为真的参数，因此冻结策略每步的反向传播只需计算分类头的梯度。分类头参数量 768×2+2=1,538。

从对比设计的角度看，三个策略覆盖了「初始化来源（预训练权重 / 随机初始化）× 可训练范围（全部参数 / 仅分类头）」两个变量的其中三格：全量微调与冻结特征抽取使用同一份预训练权重，只差可训练范围，两者对比可以单独衡量「微调」（让整个网络适应任务）带来的收益；全量微调与从零训练的可训练范围相同，只差初始化来源，两者对比则单独衡量预训练知识本身的贡献。第四格（随机初始化 + 冻结主干）对应的随机特征不具备迁移价值，实验中不予设置。此外，从零训练刻意沿用与全量微调一致的学习率与全部超参，保证对比的唯一变量是初始权重。

配置差异也直接体现在优化开销上：冻结特征抽取的可训练参数只有全量策略的约 7.1 万分之一（1,538 对 109,483,778），AdamW 只需为分类头维护动量状态（约 12 KB），而全量策略要为 1.09 亿参数各保存两份 fp32 状态（约 0.88 GB 显存）。学习率相差 50 倍同样源于这一结构差异：只训练一个从零开始的线性分类头时，$10^(-3)$ 量级的步长足以快速收敛且没有破坏主干的风险；而全量微调更新的是已经历预训练的 1.09 亿权重，必须用 $2 times 10^(-5)$ 量级的步长缓慢调整，否则一次大步更新就会覆盖掉预训练表示。该配置差异的实测后果（训练速度相差约 4.6 倍）见 4.4 节。

== 4.3 训练策略

三个策略共享除学习率与轮数预算外的全部训练配置：随机种子 42（同一数据划分、同一训练顺序）、AdamW（权重衰减 0.01）、warmup 10% + 线性衰减调度（每步更新）、梯度裁剪范数 1.0、交叉熵损失、batch size 32。轮数上限按各策略的收敛速度设置——全量微调 5 轮、冻结特征抽取 10 轮、从零训练 20 轮：统一轮数会让收敛慢的策略欠训练（3 轮时从零模型的验证指标仍未见顶），给从零训练 4 倍于微调的预算，也排除了「它只是没训够」的解释。每个 epoch 结束后在验证集上评估，按验证集 F1 保存最优 checkpoint（min_delta 5e-4），若连续 3 个 epoch 无提升则早停（patience=3，相对保守）；训练结束后加载最优权重在测试集上评估一次。

选用验证集 F1（而非 loss）作为模型选择标准，是因为分类任务的 loss 与任务指标并不严格一致（验证 loss 最低点不一定对应 F1 最高点）；F1 同时考虑了正负两类的查准与查全，比准确率更适合衡量二分类器的综合性能。早停的引入则避免了额外轮次上的过拟合与无效计算。

== 4.4 训练过程与收敛曲线

三个策略在各自的轮数预算内训练（61,961 条训练样本，每 epoch 1,937 步）：冻结特征抽取在第 5 轮触发早停（最优为第 2 轮），全量微调跑满 5 轮（最优第 3 轮），从零训练在第 13 轮触发早停（最优第 10 轮）。逐 epoch 指标如下：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: center,
    [策略], [轮次], [train loss], [val loss], [val acc], [val F1], [训练用时],
    [全量微调], [1], [0.2727], [0.1827], [0.9399], [0.9461], [208.2 s],
    [], [2], [0.1287], [0.2107], [0.9471], [0.9525], [207.5 s],
    [], [3], [0.0835], [0.2218], [0.9490], [0.9541], [207.8 s],
    [], [4], [0.0538], [0.2503], [0.9478], [0.9535], [208.3 s],
    [], [5], [0.0352], [0.2830], [0.9488], [0.9540], [207.6 s],
    [冻结特征抽取], [1], [0.6035], [0.5022], [0.7435], [0.8028], [41.3 s],
    [], [2], [0.5058], [0.4294], [0.8174], [0.8424], [40.9 s],
    [], [3], [0.4887], [0.4533], [0.7825], [0.8278], [40.8 s],
    [], [4], [0.4875], [0.4407], [0.8014], [0.8011], [41.3 s],
    [], [5], [0.4807], [0.4124], [0.8231], [0.8424], [41.5 s],
  ),
  caption: [表 4-3　全量微调与冻结特征抽取逐 epoch 的验证集指标],
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: center,
    [策略], [轮次], [train loss], [val loss], [val acc], [val F1], [训练用时],
    [从零训练], [1], [0.6193], [0.4506], [0.7866], [0.8235], [207.6 s],
    [], [2], [0.3999], [0.3127], [0.8719], [0.8865], [207.5 s],
    [], [3], [0.2957], [0.2956], [0.8851], [0.8950], [206.8 s],
    [], [4], [0.2381], [0.2749], [0.8896], [0.9049], [206.7 s],
    [], [5], [0.2035], [0.2711], [0.9055], [0.9160], [208.1 s],
    [], [6], [0.1796], [0.2706], [0.9091], [0.9168], [207.6 s],
    [], [7], [0.1588], [0.2828], [0.9091], [0.9197], [207.8 s],
    [], [8], [0.1379], [0.3035], [0.9091], [0.9162], [207.7 s],
    [], [9], [0.1218], [0.3105], [0.9118], [0.9208], [208.0 s],
    [], [10], [0.1108], [0.3888], [0.9141], [0.9226], [207.4 s],
    [], [11], [0.0991], [0.3779], [0.9124], [0.9215], [206.7 s],
    [], [12], [0.0852], [0.4296], [0.9105], [0.9197], [205.9 s],
    [], [13], [0.0755], [0.4422], [0.9122], [0.9214], [205.9 s],
  ),
  caption: [表 4-3（续）　从零训练逐 epoch 的验证集指标],
)

#figure(
  image("figures/curves_finetune.png", width: 100%),
  caption: [图 4-2　全量微调的学习曲线],
)

#figure(
  image("figures/curves_scratch.png", width: 100%),
  caption: [图 4-3　从零训练的学习曲线],
)

#figure(
  image("figures/curves_feature-extract.png", width: 100%),
  caption: [图 4-4　冻结特征抽取的学习曲线],
)

从三组曲线可以观察到：

1. *预训练权重让模型「起跑即接近终点」*。全量微调在 epoch 1 就达到验证集准确率 0.9399、F1 0.9461——已经超过从零训练 10 个 epoch 的最优水平（F1 0.9226），并在随后各轮稳定维持在 0.95 以上。从零训练则是典型的「从零爬坡」曲线：前 5 轮 val F1 从 0.8235 升至 0.9160（第 2 轮单轮 +6.3 个百分点），直到第 10 轮才见顶，收敛所需轮数是微调的 3 倍以上。
2. *微调与从零训练都出现「验证 loss 与任务指标背离」，且都伴随过拟合迹象*。微调 train loss 从 0.2727 一路降到 0.0352，验证 loss 却从 0.1827 持续回升到 0.2830，而 val F1 在 0.9535–0.9541 高位横盘——若按验证 loss 选模型会退回 epoch 1 的权重，损失约 0.8 个 F1 点。从零训练同样如此：train loss 降至 0.0755、验证 loss 自第 6 轮起从 0.2706 回升到 0.4422，val F1 却仍在 0.92 附近缓升。这印证了 4.3 节按 F1（而非 loss）选优的动机，也说明两个模型的训练损失已明显低于其泛化水平。
3. *冻结特征抽取在 lr $10^(-3)$ 下验证指标震荡*。val F1 在 0.80–0.84 之间来回（0.8028 → 0.8424 → 0.8278 → 0.8011 → 0.8424），没有稳定的上升趋势——该步长对只有 1,538 个参数的线性头偏大，后期在最优解附近反复反弹；patience=3 于第 5 轮触发早停，选中第 2 轮的 checkpoint。它的每 epoch 用时仅约 41 秒（全量微调的约 1/5）——反向传播只经过分类头，主干仅做一次前向；冻结表示的精度上限也明显受限，val F1 始终未突破 0.85。低于该步长的灵敏度对照见 4.5 节。
4. *早停机制在两个策略上真实生效*。冻结特征抽取（第 5 轮）与从零训练（第 13 轮）分别因 val F1 连续 3 轮无实质提升而提前结束；全量微调跑满 5 轮预算（最优第 3 轮），未触发早停。三者的最优 checkpoint 分别落在第 2、10、3 轮。

== 4.5 测试集评估与策略对比

训练结束后加载各策略验证集 F1 最优的权重，在 872 条测试样本上评估：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: center,
    [策略], [测试 acc], [测试 F1], [最优验证 F1], [验证→测试落差], [训练总耗时],
    [全量微调], [0.9197], [0.9215], [0.9541], [-0.0292], [1059.3 s],
    [冻结特征抽取], [0.7924], [0.8113], [0.8424], [-0.0249], [225.0 s],
    [冻结特征抽取（lr $5 times 10^(-4)$）], [0.8016], [0.8162], [0.8500], [-0.0249], [455.7 s],
    [从零训练], [0.7878], [0.7933], [0.9226], [-0.1262], [2745.6 s],
  ),
  caption: [表 4-4　三个策略的测试集结果对比，附冻结特征抽取的 lr 灵敏度对照（验证→测试落差 = 最优 epoch 验证 acc − 测试 acc）],
)

#figure(
  image("figures/comparison.png", width: 88%),
  caption: [图 4-5　三个策略在验证/测试集上的指标对比],
)

#figure(
  image("figures/confusion_finetune.png", width: 52%),
  caption: [图 4-6　全量微调的混淆矩阵],
)

#figure(
  image("figures/confusion_scratch.png", width: 52%),
  caption: [图 4-7　从零训练的混淆矩阵],
)

结合表格与图表，得到三点主要结论：

1. *全量微调大幅领先*：测试准确率 0.9197 / F1 0.9215，比从零训练（0.7878）高 13.2 个百分点，比冻结特征抽取（0.7924）高 12.7 个百分点。最优权重在第 3 轮即取得，与该模型微调 SST-2 的公开参考结果（dev 准确率约 93.5%）处于同一水平——本实验测试集即官方 dev 的 872 条样本，约 1.5 个百分点的差距在统计涨落范围（准确率标准误约 0.9 个百分点）之内。混淆矩阵显示其两类错误均衡（37/33），没有类别偏置。
2. *冻结特征抽取以约 1/12 的成本达到与从零训练几乎相同的测试成绩*：两者准确率 0.7924 vs 0.7878、F1 0.8113 vs 0.7933，而训练耗时 225 秒 vs 2746 秒——冻结时反向传播只经过 1,538 个参数的分类头，12 层主干仅做一次前向，优化器状态也可忽略。用同一划分、同一配置单独运行一次步长减半的对照实验（仅 lr 改为 $5 times 10^(-4)$，见表 4-4）后，验证曲线相邻轮的摆动从 ±4 个百分点收窄到 1 个百分点以内、最优验证 F1 反升至 0.8500，测试 acc 提高到 0.8016——说明 1e-3 确实略偏大，但步长调整改变不了定性结论：冻结表示的精度上限明显低于微调（测试准确率始终未突破 0.82），预训练收益的充分释放仍依赖对主干权重的微调，这与「是否应该微调」的经典结论一致（Peters 等，2019）。
3. *预训练模型的泛化落差更小*：全量微调与冻结特征抽取的验证→测试落差分别只有 2.9 与 2.5 个百分点，而从零训练高达 12.6 个百分点。从零模型在训练分布相似的验证集上表现尚可（0.9141），换到官方 dev 分布（正类占比 0.5092，与训练集 0.5578 有明显差异）后性能明显下滑，说明它更多依赖训练集的表面统计规律；预训练模型则学到了更稳健的语言特征，对分布移动更不敏感。

== 4.6 错误分析与预测置信度

#figure(
  image("figures/confidence_finetune.png", width: 88%),
  caption: [图 4-8　全量微调在测试集上的预测置信度分布（对数纵轴）],
)

置信度分布显示：正确预测（802 条）高度集中在 0.95 以上；错误预测（70 条）虽然整体置信度偏低、分布更散，但仍有相当一部分「高置信度错误」。检查全量微调最自信的误分类样例（`figures/error_examples_finetune.txt` 前 5 条）：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: left,
    [置信度], [真实/预测], [文本], [错误类型],
    [0.9993], [0 → 1], [`moretti 's compelling anatomy of grief and the difficult process of adapting to loss .`], [题材沉重，字面褒义（compelling）],
    [0.9989], [1 → 0], [`as unseemly as its title suggests .`], [词汇贬义但标注为褒（语用 / 标注噪声）],
    [0.9986], [0 → 1], [`the lower your expectations , the more you 'll enjoy it .`], [反讽：字面为褒义],
    [0.9985], [0 → 1], [`it 's somewhat clumsy and too lethargically paced -- but ... offers a solid build-up , a terrific climax ...`], [让步结构：先贬后褒，句式为差评],
    [0.9983], [0 → 1], [`sit through this one , and you wo n't need a magic watch to stop time ; your dvd player will do it for you .`], [反讽：以「无聊到想快进」表达差评],
  ),
  caption: [表 4-5　全量微调最自信的 5 条误分类样例],
)

全量微调的错误集中在*反讽、让步结构与复杂否定*上：模型能正确激活字面词汇（enjoy、terrific 等褒义词），但无法理解「字面褒义、实际贬义」或「先贬后褒实为差评」的语用现象。这类错误也是人类标注容易产生分歧的类型（如 `as unseemly as its title suggests .` 字面为贬义、标注却为褒义），属于任务本身的噪声上界。

#figure(
  image("figures/confidence_scratch.png", width: 88%),
  caption: [图 4-9　从零训练在测试集上的预测置信度分布],
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: left,
    [置信度], [真实/预测], [文本], [错误类型],
    [0.9983], [0 → 1], [`determined to be fun , and bouncy , with energetic musicals , the humor did n't quite engage this adult .`], [否定结构：正面词堆叠 + did n't engage],
    [0.9983], [0 → 1], [`although huppert 's intensity and focus has a raw exhilaration about it , the piano teacher is anything but fun .`], [让步结构：anything but 否定],
    [0.9983], [0 → 1], [`outer-space buffs might love this film , but others will find its pleasures intermittent .`], [让步结构：转折后为贬义],
    [0.9983], [0 → 1], [`vera 's technical prowess ends up selling his film short ; he smoothes over hard truths even as he uncovers them .`], [习语反讽：selling short / 掩盖真相表达贬义],
    [0.9983], [0 → 1], [`sam mendes has become valedictorian at the school for soft landings and easy ways out .`], [反讽：以「优等生」讽刺其回避尖锐],
  ),
  caption: [表 4-6　从零训练最自信的 5 条误分类样例],
)

对比两组样例可以清楚看到两个模型「不会的东西」不同：从零模型最自信的 10 条错误*全部是「差评→好评」（0→1）*，集中在让步结构（love … but、although … is anything but）、否定（did n't quite engage）与习语反讽（selling his film short）——它学会了识别单个情感词，却没有学会否定、转折与语用算子；微调模型的错误清单中让步/否定结构仅剩 2/10，其余为更微妙的语用现象与标注噪声。这正是预训练价值的具体体现：BERT 在预训练语料中习得的句法结构与语义组合知识，经过微调后被大幅迁移到了情感分类任务上。

从置信度校准角度看，三个模型的输出都存在「过度自信」倾向，其中从零模型在错误样本上的置信度也接近 1.0（最高 0.9983），而冻结特征抽取的错误样本置信度上限仅约 0.91——后者由于表示未被任务适配、分类边界更「犹豫」，错误时反而更保守。

= 5. 实验总体结论

1. *预训练 + 微调范式在本任务上显著有效。* bert-base-uncased 仅微调 3 个 epoch 即在 SST-2 测试集上达到准确率 0.9197 / F1 0.9215，比相同结构从零训练 10 个 epoch 的最好水平（0.7878）高 13.2 个百分点；收敛速度的差距更突出——微调 1 个 epoch 的验证 F1（0.9461）已经超过从零模型 13 轮训练中的最优值（0.9226），说明预训练提供的不是「更多样本的等价物」，而是从零训练在同等预算下难以企及的语言先验。
2. *冻结特征抽取与从零训练测试成绩接近，但代价与稳健性完全不同，预训练收益必须通过微调释放。* 冻结全部 1.09 亿主干参数、只训练 1,538 个分类头参数时，测试准确率 0.7924，与从零训练 13 轮的结果（0.7878）几乎相同，但训练总耗时只有后者的 1/12（225 s 对 2745.6 s），且验证→测试落差小得多（2.5 对 12.6 个百分点）。也就是说，预训练权重的价值不在于「拿来做固定特征」（其精度上限明显偏低，远低于全量微调），而在于「作为优化起点被任务重塑」；不过冻结路径在算力受限时仍是比从零训练更划算的选择。
3. *预训练模型泛化更稳健。* 全量微调与冻结特征抽取的验证→测试落差分别只有 2.9 与 2.5 个百分点，而从零训练高达 12.6 个百分点——它在训练同分布的验证集上越练越好（val F1 升至 0.9226），在官方 dev 分布上却明显掉队。结合错误分析：从零模型最自信的错误全部是「差评→好评」，集中在让步结构与否定（句法-语义组合），微调模型的错误集中在反讽语用，说明预训练带来的不仅是词汇联想，更是结构与语义知识。
4. *验证集选优、学习率调度与动态 padding 等训练改进均有实际收益。* 本实验中再次观察到验证 loss 与 F1 的背离（4.4 节），按 F1 选最优 checkpoint 避免了约 0.8 个 F1 点的损失；warmup + 线性衰减使微调在小学习率下快速而稳定地收敛，梯度裁剪全程未出现 loss 尖峰；动态 padding 把训练序列长度从固定 64 降到平均 20 余个有效 token，配合 TF32 使微调每个 epoch 仅需约 3.5 分钟。早停也在两个策略上真实触发生效。
5. *局限与可改进之处*：
   - 全部实验只使用单一随机种子，多种子重复可进一步给出均值和方差；从零训练的高置信度错误全部偏向「好评」一侧，或与训练集正类占比偏高（0.5578 对测试集 0.5092）有关，可尝试按类先验重加权缓解。
   - 冻结特征抽取只尝试了单层线性分类头与两档学习率（$10^(-3)$、灵敏度实验的 $5 times 10^(-4)$），未探索「解冻最后 k 层」的折中方案。
   - 测试集仅 872 条，单个样本权重偏大，结论对个别噪声标签敏感；使用 SST-2 完整训练集（含短语级）或数据增强可作为后续工作。
   - 错误分析显示反讽类样本是主要瓶颈，可尝试引入更大预训练模型（如 RoBERTa）或领域内继续预训练（domain-adaptive pretraining）来改进。

// 附录改为横向页
#set page(flipped: true, margin: 1.8cm)
#show raw.where(block: true): set text(size: 8pt)

= 6. 完整实验代码与程序输出

== 6.1 数据加载与批构造 bert_data.py

#raw(read("bert_data.py"), lang: "python")

#pagebreak()

== 6.2 模型构建 bert_model.py

#raw(read("bert_model.py"), lang: "python")

#pagebreak()

== 6.3 训练主程序 bert_train.py

#raw(read("bert_train.py"), lang: "python")

#pagebreak()

== 6.4 可视化脚本 bert_viz.py

#raw(read("bert_viz.py"), lang: "python")

#pagebreak()

== 6.5 完整训练输出（全量微调）

#raw(read("results/finetune/train_output.txt"))

#pagebreak()

== 6.6 其余两个策略的逐 epoch 摘要与测试结果

冻结特征抽取（完整输出见 `results/feature-extract/train_output.txt`）：

#raw(read("results/feature-extract/epoch_summary.txt"))

从零训练（完整输出见 `results/scratch/train_output.txt`）：

#raw(read("results/scratch/epoch_summary.txt"))

= 7. 参考文献

+ 课程示例程序 `demo_labs/lab3-bert.py`.
+ Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019.（BERT 原始论文，本实验预训练模型的来源）
+ Socher R, Perelygin A, Wu J, et al. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. EMNLP, 2013.（SST-2 数据集原始论文）
+ Wang A, Singh A, Michael J, et al. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. ICLR, 2019.（GLUE/SST-2 数据集的调用格式）
+ Peters M E, Ruder S, Smith N A. To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks. RepL4NLP, 2019.（特征抽取与微调两种迁移策略的对比研究）
+ Loshchilov I, Hutter F. Decoupled Weight Decay Regularization. ICLR, 2019.（AdamW 优化器）
+ Hugging Face. Transformers Documentation. https://huggingface.co/docs/transformers.（`AutoTokenizer`、`AutoModelForSequenceClassification` 与线性调度器的使用文档）
