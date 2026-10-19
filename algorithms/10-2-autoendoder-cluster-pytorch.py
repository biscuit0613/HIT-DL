import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import gzip
import struct

# 设置随机种子，确保实验结果可重现
torch.manual_seed(42)

# 超参数定义
learning_rate = 0.001        # 学习率，由于使用ReLU，适当降低学习率
training_epochs = 30         # 训练轮数
batch_size = 256             # 批大小，每次训练使用的样本数量
display_step = 1             # 显示间隔，每多少轮显示一次训练信息
n_input = 784                # 输入维度，MNIST图像是28x28=784像素

# 设备配置：自动检测并使用GPU（如果可用），否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 从本地MNIST_data目录加载数据的函数
def load_mnist_local():
    """
    从本地MNIST_data目录加载MNIST数据集
    返回：训练图像、训练标签、测试图像、测试标签
    """
    data_dir = './MNIST_data'
    
    def load_images(filename):
        """
        加载MNIST图像数据文件
        参数：
            filename: 图像文件名
        返回：
            numpy数组，形状为(样本数, 784)，值在0-1之间
        """
        filepath = os.path.join(data_dir, filename)
        
        # 检查文件是否存在，如果不存在则尝试.gz压缩版本
        if not os.path.exists(filepath):
            filepath += '.gz'
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File {filename} not found in {data_dir}")
            
            # 读取.gz压缩文件
            with gzip.open(filepath, 'rb') as f:
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            # 读取未压缩文件
            with open(filepath, 'rb') as f:
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # 将数据重塑为(样本数, 784)的形状，并归一化到0-1范围
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0
    
    def load_labels(filename):
        """
        加载MNIST标签数据文件
        参数：
            filename: 标签文件名
        返回：
            numpy数组，形状为(样本数,)，包含0-9的数字标签
        """
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            filepath += '.gz'
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File {filename} not found in {data_dir}")
            
            with gzip.open(filepath, 'rb') as f:
                magic, num = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            with open(filepath, 'rb') as f:
                magic, num = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.uint8)
        
        return data.astype(np.int64)
    
    print("Loading MNIST data from local directory...")
    
    # 尝试不同的文件名组合
    file_combinations = [
        {
            'train_images': 'train-images-idx3-ubyte',
            'train_labels': 'train-labels-idx1-ubyte',
            'test_images': 't10k-images-idx3-ubyte',
            'test_labels': 't10k-labels-idx1-ubyte'
        },
        {
            'train_images': 'train-images.idx3-ubyte',
            'train_labels': 'train-labels.idx1-ubyte',
            'test_images': 't10k-images.idx3-ubyte',
            'test_labels': 't10k-labels.idx1-ubyte'
        }
    ]
    
    # 尝试不同的文件名组合，直到找到可用的文件
    for files in file_combinations:
        try:
            train_images = load_images(files['train_images'])
            train_labels = load_labels(files['train_labels'])
            test_images = load_images(files['test_images'])
            test_labels = load_labels(files['test_labels'])
            print(f"Successfully loaded data with file pattern: {files['train_images']}")
            break
        except FileNotFoundError as e:
            print(f"Failed with file pattern {files['train_images']}: {e}")
            continue
    else:
        # 如果所有文件名组合都失败，显示目录内容帮助调试
        print(f"Could not find MNIST files in {data_dir}")
        print("Available files in directory:")
        for f in os.listdir(data_dir):
            print(f"  {f}")
        raise FileNotFoundError("No valid MNIST files found")
    
    print(f"Training data shape: {train_images.shape}")
    print(f"Test data shape: {test_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    return train_images, train_labels, test_images, test_labels

# 加载数据
try:
    train_images, train_labels, test_images, test_labels = load_mnist_local()
    
    # 将numpy数组转换为PyTorch张量
    train_tensor = torch.FloatTensor(train_images)
    train_labels_tensor = torch.LongTensor(train_labels)
    test_tensor = torch.FloatTensor(test_images)
    test_labels_tensor = torch.LongTensor(test_labels)
    
    # 创建PyTorch数据集对象
    train_dataset = TensorDataset(train_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_tensor, test_labels_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please make sure the MNIST_data directory exists and contains the MNIST files")
    exit()

# 定义用于二维聚类的自编码器模型
class Autoencoder2D(nn.Module):
    """
    自编码器模型，专门用于将784维的MNIST图像压缩到2维空间进行可视化
    网络结构: 784 -> 128 -> 64 -> 10 -> 2 -> 10 -> 64 -> 128 -> 784
    
    设计思路：
    - 编码器逐步将高维数据压缩到低维表示
    - 解码器从低维表示重构原始数据
    - 瓶颈层设置为2维，便于在二维平面上可视化数据分布
    """
    def __init__(self, n_input=784):
        super(Autoencoder2D, self).__init__()
        
        # 编码器部分：将784维输入逐步压缩到2维
        self.encoder = nn.Sequential(
            # 第一层：784 -> 128
            nn.Linear(n_input, 128),
            nn.ReLU(inplace=True),  # 使用ReLU激活函数，inplace=True节省内存
            
            # 第二层：128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            
            # 第三层：64 -> 10
            nn.Linear(64, 10),
            nn.ReLU(inplace=True),
            
            # 第四层：10 -> 2 (瓶颈层)
            # 注意：最后一层不使用激活函数，直接输出2维坐标
            # 这样做的原因：避免激活函数限制输出范围，让编码可以分布在任意二维空间
            nn.Linear(10, 2),
        )
        
        # 解码器部分：从2维表示重构784维原始数据
        self.decoder = nn.Sequential(
            # 第一层：2 -> 10
            nn.Linear(2, 10),
            nn.ReLU(inplace=True),
            
            # 第二层：10 -> 64
            nn.Linear(10, 64),
            nn.ReLU(inplace=True),
            
            # 第三层：64 -> 128
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            
            # 第四层：128 -> 784 (输出层)
            nn.Linear(128, n_input),
            nn.Sigmoid()  # 输出层使用Sigmoid，确保输出在[0,1]范围内，与输入一致
        )
    
    def forward(self, x):
        """
        前向传播过程
        参数：
            x: 输入数据，形状为(batch_size, 784)
        返回：
            重构的数据，形状与输入相同
        """
        encoded = self.encoder(x)      # 编码：784 -> 2
        decoded = self.decoder(encoded)  # 解码：2 -> 784
        return decoded
    
    def encode(self, x):
        """
        仅进行编码，返回2维表示
        用于获取数据的低维嵌入，便于可视化
        参数：
            x: 输入数据
        返回：
            2维编码表示，形状为(batch_size, 2)
        """
        with torch.no_grad():  # 禁用梯度计算，节省内存
            return self.encoder(x)

# 初始化模型
model = Autoencoder2D(n_input).to(device)

# 打印模型结构
print("\n" + "="*60)
print("Autoencoder Model Structure for 2D Clustering Visualization")
print("="*60)
print("Encoder: 784 → 128 → 64 → 10 → 2")
print("Decoder: 2 → 10 → 64 → 128 → 784")
print("Activation: ReLU (except encoder output and decoder output)")
print("="*60)
print(model)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失，衡量重构图像与原始图像的差异
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 训练模型
print("\n" + "="*60)
print("Starting Training for 2D Clustering Visualization")
print("="*60)

total_batch = len(train_loader)
train_losses = []  # 记录训练损失，用于后续分析

for epoch in range(training_epochs):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    
    # 遍历每个批次
    for i, (batch_xs, _) in enumerate(train_loader):
        batch_xs = batch_xs.to(device)
        
        # 前向传播
        outputs = model(batch_xs)           # 模型预测（重构图像）
        loss = criterion(outputs, batch_xs) # 计算损失
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空之前的梯度，防止梯度累积
        loss.backward()        # 反向传播，计算当前梯度
        optimizer.step()       # 更新模型参数
        
        total_loss += loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # 按显示间隔输出训练信息
    if epoch % display_step == 0:
        print(f"Epoch: {epoch+1:04d}/{training_epochs}, Average Loss: {avg_loss:.9f}")

print("Optimization Finished!")

# 可视化训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, label='Training Loss')
plt.title('Autoencoder Training Loss (ReLU Activation)')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('training_loss_2d_clustering.png', dpi=150, bbox_inches='tight')
plt.show()

# 使用训练好的自编码器进行二维可视化
print("\n" + "="*60)
print("Generating 2D Visualization")
print("="*60)

model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 禁用梯度计算，节省内存
    # 对整个测试集进行编码，获取2维表示
    test_images_tensor = test_tensor.to(device)
    encoder_result = model.encode(test_images_tensor)
    
    # 将结果转换为numpy数组，用于绘图
    encoder_result_np = encoder_result.cpu().numpy()
    test_labels_np = test_labels_tensor.numpy()
    
    print(f"Encoded data shape: {encoder_result_np.shape}")  # 应该是(10000, 2)
    print(f"Test labels shape: {test_labels_np.shape}")      # 应该是(10000,)

# 创建二维散点图，用颜色表示数字类别
plt.figure(figsize=(12, 10))

# 创建散点图，每个点的颜色对应其真实标签
scatter = plt.scatter(encoder_result_np[:, 0],     # x坐标
                     encoder_result_np[:, 1],     # y坐标  
                     c=test_labels_np,            # 颜色根据标签值
                     cmap='tab10',                # 使用tab10色彩映射，适合10个类别
                     alpha=0.6,                   # 半透明，便于观察重叠点
                     s=8)                         # 点的大小

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Digit Class', rotation=270, labelpad=15)

# 设置图表标题和坐标轴标签
plt.title('MNIST 2D Visualization using Autoencoder with ReLU\n'
          'Each point represents a digit image compressed to 2D space', 
          fontsize=14, pad=20)
plt.xlabel('Encoded Dimension 1', fontsize=12)
plt.ylabel('Encoded Dimension 2', fontsize=12)

# 添加网格
plt.grid(True, alpha=0.3)

# 添加说明文本
plt.figtext(0.5, 0.01, 
           'Note: Similar digits tend to cluster together in the 2D space', 
           ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # 为底部文本留出空间

# 保存可视化结果
plt.savefig('mnist_2d_clustering_relu.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存训练好的模型
torch.save(model.state_dict(), 'autoencoder_2d_clustering_relu.pth')
print("\nModel saved as 'autoencoder_2d_clustering_relu.pth'")
print("Training loss plot saved as 'training_loss_2d_clustering.png'")
print("2D visualization saved as 'mnist_2d_clustering_relu.png'")

# 计算并显示最终的重构误差
model.eval()
with torch.no_grad():
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        outputs = model(data)
        test_loss += criterion(outputs, data).item()
    
    test_loss /= len(test_loader)
    print(f"\nFinal test reconstruction error: {test_loss:.6f}")

# 分析二维编码的统计特性
print("\n" + "="*60)
print("2D Encoding Statistics")
print("="*60)
print(f"Dimension 1 - Min: {encoder_result_np[:, 0].min():.3f}, "
      f"Max: {encoder_result_np[:, 0].max():.3f}, "
      f"Mean: {encoder_result_np[:, 0].mean():.3f}")
print(f"Dimension 2 - Min: {encoder_result_np[:, 1].min():.3f}, "
      f"Max: {encoder_result_np[:, 1].max():.3f}, "
      f"Mean: {encoder_result_np[:, 1].mean():.3f}")

print("\n" + "="*60)
print("Program Completed Successfully!")
print("="*60)