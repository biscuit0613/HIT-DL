"""三种迁移策略的模型构建：全量微调 / 冻结特征抽取 / 从零训练。"""

from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
)

# 特征抽取只训练约 1.5k 参数的分类头，需要比全量微调大得多的学习率
MODE_DEFAULT_LR = {"finetune": 2e-5, "feature-extract": 1e-3, "scratch": 2e-5}

MODE_LABELS = {
    "finetune": "全量微调",
    "feature-extract": "冻结特征抽取",
    "scratch": "从零训练",
}

# 轮数上限按各策略的收敛速度设置：微调动的是预训练权重，几轮即收敛；
# 冻结只训一个线性头，需要更多轮次；从零训练最慢，给足预算由早停决定实际轮数
MODE_DEFAULT_EPOCHS = {"finetune": 5, "feature-extract": 10, "scratch": 20}


"""
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(...)
    (encoder): BertEncoder(12层 Transformer)
    (pooler): BertPooler(...)
  )
  (dropout): Dropout(p=0.1)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
分类头在预训练时并不存在，所以无论 finetune 还是 feature-extract，from_pretrained 都会随机初始化分类头，然后从它开始学。
"""

def build_model(model_path, mode, num_labels=2):
    if mode == "scratch":
        # 从零训练：只复用预训练模型的网络结构配置（层数/维度/词表），权重全部随机初始化
        config = BertConfig.from_pretrained(model_path, num_labels=num_labels)
        return BertForSequenceClassification(config)
    
    # 加载预训练权重 pytorch_model.bin，并根据 num_labels=2 构建分类头。
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    )
    # 分类头在预训练权重里不存在，这里由 from_pretrained 随机初始化，两种策略都从它开始学
    if mode == "feature-extract":# finetune同理
        # 按参数名前缀冻结主干：embeddings/encoder 的层名都不以 classifier 开头，
        # 全部置为不可训练，等价于把 BERT 当固定的特征提取器，只训一个带 dropout 的线性分类头
        for name, param in model.named_parameters():# named_parameters() 返回 (name, param) 对，name 是参数名，param 是参数本身
            param.requires_grad = name.startswith("classifier") # name.startswith("classifier") 为 True 时，requires_grad=True，表示可训练；否则为 False，冻结。
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
