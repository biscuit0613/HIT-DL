"""
基于 MindSpore 的 MLP 分类与超参数网格搜索 + K折交叉验证
超参数：隐藏层大小 (hidden_size)、丢弃率 (dropout_rate)
数据集：Iris (sklearn 内置)

本版本：
1. 每一步都打印详细状态；
2. 超参数网格精简为 4 个组合，运行更快。
"""

import time
import numpy as np
from itertools import product
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype


# ==================== 工具函数：打印分隔线 ====================
def print_sep(char="=", length=70):
    print(char * length)


def print_step(title):
    print()
    print_sep("=")
    print(f"  {title}")
    print_sep("=")


def print_substep(title):
    print()
    print_sep("-", 70)
    print(f"  >> {title}")
    print_sep("-", 70)


# ==================== 1. 数据准备 ====================
def load_and_preprocess_data():
    """加载 Iris 数据集并做标准化预处理"""
    print_step("步骤 1 / 5：加载并预处理数据")

    print("[1.1] 正在从 sklearn 加载 Iris 数据集 ...")
    iris = load_iris()
    X, y = iris.data, iris.target
    print(f"      加载完成。原始特征形状: X={X.shape}, y={y.shape}")
    print(f"      特征名称: {iris.feature_names}")
    print(f"      类别名称: {list(iris.target_names)}")
    print(f"      类别分布: {np.bincount(y)} (每类样本数)")

    print("[1.2] 正在对特征做标准化 (StandardScaler) ...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"      标准化完成。均值≈{X.mean():.4f}, 标准差≈{X.std():.4f}")

    print("[1.3] 正在转换数据类型为 float32 / int32 ...")
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    print(f"      类型转换完成。X.dtype={X.dtype}, y.dtype={y.dtype}")

    print("[1.4] 数据准备阶段结束。")
    return X, y


# ==================== 2. 可配置的 MLP 模型 ====================
class MLP(nn.Cell):
    """
    多层感知机，超参数可在实例化时指定：
    - hidden_size: 隐藏层神经元数量
    - dropout_rate: 丢弃率（0 表示不丢弃）
    """
    def __init__(self, input_size=4, hidden_size=16, num_classes=3, dropout_rate=0.0):
        super(MLP, self).__init__()
        print(f"      [MLP.__init__] 构建模型: input={input_size}, "
              f"hidden={hidden_size}, classes={num_classes}, dropout={dropout_rate}")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Dense(hidden_size, num_classes)

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==================== 3. 训练与评估 ====================
def train_and_evaluate(hidden_size, dropout_rate,
                       X_train, y_train, X_val, y_val,
                       epochs=100, lr=0.01, verbose=False):
    """
    训练一个 MLP 并在验证集上评估准确率。
    verbose=True 时打印每若干轮的损失。
    """
    if verbose:
        print(f"      [train_and_evaluate] hidden={hidden_size}, dropout={dropout_rate}, "
              f"epochs={epochs}, lr={lr}")
        print(f"      训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # 3.1 构建模型
    model = MLP(hidden_size=hidden_size, dropout_rate=dropout_rate)

    # 3.2 定义损失函数与优化器
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

    # 3.3 numpy -> Tensor
    X_train_t = Tensor(X_train, mstype.float32)
    y_train_t = Tensor(y_train, mstype.int32)
    X_val_t = Tensor(X_val, mstype.float32)
    y_val_t = Tensor(y_val, mstype.int32)

    # 3.4 定义前向与梯度函数
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    # 3.5 训练循环
    model.set_train(True)
    t_start = time.time()
    for epoch in range(epochs):
        loss = train_step(X_train_t, y_train_t)
        if verbose and (epoch + 1) % 20 == 0:
            print(f"        epoch {epoch+1:3d}/{epochs}  loss={loss.asnumpy():.4f}")
    t_end = time.time()
    if verbose:
        print(f"      训练完成，耗时 {t_end - t_start:.2f} 秒")

    # 3.6 评估
    model.set_train(False)
    logits = model(X_val_t)
    preds = ops.Argmax(axis=1)(logits)
    acc = accuracy_score(y_val, preds.asnumpy())
    if verbose:
        print(f"      验证集准确率: {acc:.4f}")

    return acc


# ==================== 4. 网格搜索 + K折交叉验证 ====================
def grid_search_cv(X, y, param_grid, n_splits=5):
    """
    对给定超参数网格执行 K 折交叉验证。
    返回：最佳参数、最佳平均分数、全部结果列表
    """
    print_step("步骤 3 / 5：网格搜索 + K折交叉验证")

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = list(product(*values))
    total_combos = len(param_combinations)

    print(f"[3.1] 参数网格: {param_grid}")
    print(f"      参数名: {keys}")
    print(f"      总组合数: {total_combos}")
    print(f"[3.2] 初始化 KFold: n_splits={n_splits}, shuffle=True, random_state=42")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    # 3.3 遍历每个参数组合
    for combo_idx, combo in enumerate(param_combinations):
        params = dict(zip(keys, combo))
        print_substep(f"组合 {combo_idx + 1}/{total_combos}  参数: {params}")

        fold_scores = []

        # 3.4 遍历每一折
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"    [折 {fold_idx + 1}/{n_splits}] "
                  f"训练样本 {len(train_idx)} 个, 验证样本 {len(val_idx)} 个")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            acc = train_and_evaluate(
                hidden_size=params['hidden_size'],
                dropout_rate=params['dropout_rate'],
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=100,
                lr=0.01,
                verbose=False   # 如需查看每轮 loss，改为 True
            )
            fold_scores.append(acc)
            print(f"        折 {fold_idx + 1} 完成，准确率 = {acc:.4f}")

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        print(f"    >> 组合 {combo_idx + 1} 汇总: 平均={mean_score:.4f}, "
              f"标准差={std_score:.4f}, 各折={[f'{s:.4f}' for s in fold_scores]}")

        results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': fold_scores
        })

    # 3.5 排序
    print_substep("所有组合执行完毕，按平均准确率降序排序")
    results.sort(key=lambda x: x['mean_score'], reverse=True)

    best = results[0]
    print(f"[3.6] 最佳组合: {best['params']}  平均分={best['mean_score']:.4f}")
    return best['params'], best['mean_score'], results


# ==================== 5. 主程序 ====================
if __name__ == "__main__":
    t_program_start = time.time()

    print_sep("=")
    print("  MindSpore MLP 超参数网格搜索 + K折交叉验证")
    print_sep("=")
    print(f"MindSpore 版本: {mindspore.__version__}")
    print(f"运行设备: {mindspore.get_context('device_target')}")

    # ---------- 步骤 1：数据 ----------
    X, y = load_and_preprocess_data()

    # ---------- 步骤 2：定义搜索空间（精简版） ----------
    print_step("步骤 2 / 5：定义超参数搜索空间（精简版）")
    param_grid = {
        'hidden_size': [8, 16],      # 只保留 2 个隐藏层大小
        'dropout_rate': [0.0, 0.1]   # 只保留 2 个丢弃率
    }
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
    total_combos = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"  总组合数: {total_combos}")
    print(f"  交叉验证折数: 5")
    print(f"  预计训练次数: {total_combos * 5}")

    # ---------- 步骤 3：网格搜索 + K折交叉验证 ----------
    t_search_start = time.time()
    best_params, best_score, all_results = grid_search_cv(X, y, param_grid, n_splits=5)
    t_search_end = time.time()
    print(f"\n[步骤 3 完成] 网格搜索总耗时: {t_search_end - t_search_start:.2f} 秒")

    # ---------- 步骤 4：输出详细结果 ----------
    print_step("步骤 4 / 5：交叉验证详细结果（按平均准确率降序）")
    for i, r in enumerate(all_results):
        print(f"\n[{i + 1}] 参数: {r['params']}")
        print(f"    平均准确率: {r['mean_score']:.4f} (+/- {r['std_score']:.4f})")
        print(f"    各折得分: {[f'{s:.4f}' for s in r['fold_scores']]}")

    # ---------- 步骤 5：最佳结果 ----------
    print_step("步骤 5 / 5：最佳超参数组合")
    print(f"  最佳参数: {best_params}")
    print(f"  最佳平均分数: {best_score:.4f}")

    t_program_end = time.time()
    print_sep("=")
    print(f"  程序运行结束，总耗时: {t_program_end - t_program_start:.2f} 秒")
    print_sep("=")