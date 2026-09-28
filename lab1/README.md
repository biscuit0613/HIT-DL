# 实验一：基于网格搜索与 K 折交叉验证的 MLP 超参数优化

SST-2 情感二分类。TF-IDF 稀疏特征 + 单隐层 MLP，对 4 个超参数做网格搜索，
用重复分层 K 折交叉验证选优，最后在独立测试集上评估。

## 运行

命令都在仓库根目录执行（`.venv` 在根目录，不在 `lab1/` 里）。

```bash
# 冒烟测试，2 个组合 x 10 折，约 30 秒
.venv/bin/python -u lab1/hyperparameter_search.py --quick

# 完整实验：36 个组合 x 10 折 = 360 次训练，约 11 分钟
# --bench-combos 8 表示先用前 8 个组合做一次串行/并行耗时对比
.venv/bin/python -u lab1/hyperparameter_search.py --bench-combos 8 --jobs 4
```

报告里的数据由第二条命令产生，其完整输出存在 `output.txt`。

两个辅助脚本对应报告的两节分析，单独运行：

```bash
.venv/bin/python -u lab1/efficiency_bench.py      # 4.6 节 稀疏格式与批量大小
.venv/bin/python -u lab1/ablation_features.py     # 4.5.3 节 特征表示消融
```

常用参数：`--jobs` 并行进程数，`--device cpu|cuda|auto`，`--epochs` 轮数上限，
`--out-dir` 改写输出目录（不加会覆盖 `figures/` 和 `results/` 里的现有结果）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `hyperparameter_search.py` | 入口，编排流程与输出；最终重训与测试集评估 |
| `hs_core.py` | 输出格式、设备选择、结果汇总与落盘 |
| `hs_data.py` | SST-2 读取、折划分、折内 TF-IDF 特征流水线 |
| `hs_model.py` | MLP、单折训练循环（早停）、网格与 joblib 并行搜索 |
| `hs_viz.py` | 六张结果图 |
| `efficiency_bench.py` | 稀疏格式与批量大小的效率基准（4.6 节） |
| `ablation_features.py` | 特征表示消融对照（4.5.3 节） |
| `实验报告.typ` / `.pdf` | 报告源码与成品，`typst compile 实验报告.typ` 重新编译 |
| `output.txt` | 完整运行输出，被报告 6.2 节直接引用 |
| `figures/` `results/` | 图、CSV、最佳权重与指标 JSON |

数据集在 `../demo_labs/SST-2/`，训练集 67349 句、测试集 872 句。

## 结果

最佳组合 `hidden=128 dropout=0.5 lr=5e-3 weight_decay=1e-4`，
交叉验证 0.9190 ± 0.0014，测试集 0.8096。

交叉验证与测试集之间 11 个百分点的落差不是过拟合，而是两份数据的句长分布
不同（训练集句均 9.4 词、测试集 19.5 词），详见报告 4.5.3 节。
