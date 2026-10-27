# 实验四：基于 GCN 与 GAT 的 Cora 引文网络节点分类

在 Cora 数据集上完成节点分类，对比 MLP（无图基线）、GCN 与 GAT；
包含多种子重复、GCN 超参数扫描、标签率实验、边删除鲁棒性实验、t-SNE、
度分组误差分析与 GAT 注意力分析。

## 运行

所有命令在 `lab4/` 目录执行（虚拟环境 `.venv` 在仓库根目录）。

```bash
# 1. 主实验：3 个模型 × 10 个随机种子（约 30 秒）
../.venv/bin/python -u gcn_experiments.py --stage main --device cuda

# 2. GCN 参数扫描 + lr×weight_decay 网格 + GAT 头数扫描（约 2 分钟）
../.venv/bin/python -u gcn_experiments.py --stage sweeps --device cuda

# 3. 标签率实验（每类 1/2/5/10/20/40 个训练节点，约 1 分钟）
../.venv/bin/python -u gcn_experiments.py --stage label-rate --device cuda

# 4. 图结构鲁棒性（删除 0~90% 无向边，约 1 分钟）
../.venv/bin/python -u gcn_experiments.py --stage robustness --device cuda

# 5. 特征归一化对照（原始二值特征 vs 按行和归一化）
../.venv/bin/python -u gcn_experiments.py --stage normalization --device cuda

# 6. 生成全部图表，并完成 t-SNE、过平滑与注意力分析
../.venv/bin/python -u gcn_viz.py
```

也可以单独跑一个模型：

```bash
../.venv/bin/python -u gcn_train.py --model gcn --seed 42 --epochs 200
../.venv/bin/python -u gcn_train.py --model gat --seed 0 --heads 8 --lr 0.005 --dropout 0.6
```

结果保存在 `lab4/results/`，图片保存在 `lab4/figures/`。

## 文件

| 文件 | 内容 |
| --- | --- |
| `gcn_data.py` | Cora 加载、去自环、特征行归一化、数据统计、标签率划分、边删除 |
| `gcn_model.py` | MLP / GCN / GAT 三种模型 |
| `gcn_train.py` | 单次训练与评估，按验证集准确率保存最优 checkpoint |
| `gcn_experiments.py` | 主实验、参数扫描、标签率、鲁棒性、归一化对照的批量入口 |
| `gcn_viz.py` | 全部图表、t-SNE、过平滑能量、GAT 注意力分析 |
| `results/` | 各实验 JSON、logits、最优模型权重、运行输出 |
| `figures/` | 报告使用的全部 PNG |

## 数据集

Cora 引文网络（PyG Planetoid 版本）：

- 2,708 个节点、5,278 条无向边（PyG 存为 10,556 条有向边）、1,433 维二值词袋特征、7 类
- 标准划分：训练 140 / 验证 500 / 测试 1,000（每类 20 个训练节点）
- 全局边同配率 0.810，节点同配率 0.825；度中位数 3，均值 3.90，最大 168
- 特征非零率约 1.27%，每个节点平均 18.2 个非零特征
- 训练前对特征按行和归一化（`x / x.sum(dim=1)`），MLP 与 GCN 均受益

## 实现亮点（超出 demo 示例的功能）

1. **三种模型统一对比与多种子评估**（`gcn_experiments.py`）：
   示例只训练一个两层 GCN，且用测试集逐 epoch 观察。本实现增加 MLP 无图基线
   与课程中介绍的 GAT，三个模型使用同一划分、同一评价协议；主实验重复 10 个种子，
   报告均值 ± 标准差，并按验证集准确率（而非测试集）保存最优 checkpoint。

2. **GCN 参数扫描与 GAT 头数扫描**（`gcn_experiments.py`）：
   单变量扫描隐藏维度、层数、dropout、学习率、权重衰减共 5 组，另做
   学习率 × 权重衰减的二维网格；GAT 扫描 1/2/4/8/16 个注意力头。所有扫描
   使用 5 个种子，报告测试与验证准确率。

3. **标签率与图结构鲁棒性实验**（`gcn_experiments.py`）：每类只给
   1/2/5/10/20/40 个标签，比较三种模型的数据效率；随机删除 10%~90% 的
   无向边后重新训练，观察图结构退化对 GNN 的影响，MLP 作为不受边影响的对照。

4. **完整的表示与误差分析**（`gcn_viz.py`）：训练曲线、类别 F1、混淆矩阵、
   按节点度分组的准确率、第一层表示的 t-SNE、GCN 逐层 Dirichlet 能量
   （过平滑）、GAT 第一层注意力的类内/类间分布与单节点注意力示意。

5. **特征归一化对照**（`gcn_experiments.py`）：在相同划分与种子下比较原始
   二值特征与行归一化特征，量化预处理对 MLP/GCN 的影响。

## 主要结果

| 模型 | 可训练参数 | 验证准确率 | 测试准确率 | 测试宏 F1 |
| --- | ---: | ---: | ---: | ---: |
| MLP | 23,063 | 0.5890 ± 0.0129 | 0.5908 ± 0.0081 | 0.5716 |
| GCN | 23,063 | 0.8004 ± 0.0072 | 0.8162 ± 0.0086 | 0.8043 |
| GAT | 92,373 | 0.8146 ± 0.0045 | 0.8232 ± 0.0084 | 0.8135 |

10 个种子、标准 Cora 划分。GAT 比 GCN 平均高 0.70 个百分点，但配对 t 检验
p=0.096，差异不显著。三个模型中 GAT 在低标签率与边删除条件下最稳定。
