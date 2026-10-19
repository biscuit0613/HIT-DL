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
learning_rate = 0.01        # 学习率，控制参数更新的步长
training_epochs = 10        # 训练轮数，整个数据集被训练的次数
batch_size = 256            # 批大小，每次训练使用的样本数量
display_step = 1            # 显示间隔，每多少轮显示一次训练信息
examples_to_show = 10       # 显示的重构示例数量
n_input = 784               # 输入维度，MNIST图像是28x28=784像素

# 设备配置：自动检测并使用GPU（如果可用），否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 从本地MNIST_data目录加载数据的函数
def load_mnist_local():
    """
    从本地MNIST_data目录加载MNIST数据集
    返回：训练图像、训练标签、测试图像、测试标签
    """
    data_dir = './MNIST_data'  # MNIST数据文件所在的目录
    
    def load_images(filename):
        """
        加载MNIST图像数据文件
        参数：
            filename: 图像文件名（如'train-images-idx3-ubyte'）
        返回：
            numpy数组，形状为(样本数, 784)，值在0-1之间
        """
        filepath = os.path.join(data_dir, filename)
        
        # 检查文件是否存在，如果不存在则尝试.gz压缩版本
        if not os.path.exists(filepath):
            filepath += '.gz'  # 添加.gz扩展名
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File {filename} not found in {data_dir}")
            
            # 读取.gz压缩文件
            with gzip.open(filepath, 'rb') as f:
                # 读取文件头信息：魔数、图像数量、行数、列数
                # '>IIII'表示大端字节序的4个无符号整数
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                # 读取剩余的图像数据
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
            filename: 标签文件名（如'train-labels-idx1-ubyte'）
        返回：
            numpy数组，形状为(样本数,)，包含0-9的数字标签
        """
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            filepath += '.gz'
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File {filename} not found in {data_dir}")
            
            with gzip.open(filepath, 'rb') as f:
                # 读取文件头信息：魔数、标签数量
                magic, num = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            with open(filepath, 'rb') as f:
                magic, num = struct.unpack(">II", f.read(8))
                data = np.frombuffer(f.read(), dtype=np.uint8)
        
        return data.astype(np.int64)
    
    print("Loading MNIST data from local directory...")
    
    # 尝试不同的文件名组合，因为不同来源的MNIST数据文件名可能不同
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
            # 加载训练和测试数据
            train_images = load_images(files['train_images'])
            train_labels = load_labels(files['train_labels'])
            test_images = load_images(files['test_images'])
            test_labels = load_labels(files['test_labels'])
            print(f"Successfully loaded data with file pattern: {files['train_images']}")
            break  # 如果成功加载，跳出循环
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
    
    # 打印数据形状信息
    print(f"Training data shape: {train_images.shape}")      # 应为(60000, 784)
    print(f"Test data shape: {test_images.shape}")           # 应为(10000, 784)
    print(f"Training labels shape: {train_labels.shape}")    # 应为(60000,)
    print(f"Test labels shape: {test_labels.shape}")         # 应为(10000,)
    
    return train_images, train_labels, test_images, test_labels

# 加载数据
try:
    # 调用函数加载MNIST数据
    train_images, train_labels, test_images, test_labels = load_mnist_local()
    
    # 将numpy数组转换为PyTorch张量
    train_tensor = torch.FloatTensor(train_images)        # 训练图像张量
    train_labels_tensor = torch.LongTensor(train_labels)  # 训练标签张量
    test_tensor = torch.FloatTensor(test_images)          # 测试图像张量
    test_labels_tensor = torch.LongTensor(test_labels)    # 测试标签张量
    
    # 创建PyTorch数据集对象
    # TensorDataset将特征张量和标签张量组合成数据集
    train_dataset = TensorDataset(train_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_tensor, test_labels_tensor)
    
    # 创建数据加载器，用于批量加载数据
    # DataLoader会自动处理数据的批处理、打乱等操作
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please make sure the MNIST_data directory exists and contains the MNIST files")
    exit()

# 定义堆栈自编码器模型类
class StackedAutoencoder(nn.Module):
    """
    堆栈自编码器神经网络模型
    结构：输入层 -> 编码器 -> 瓶颈层 -> 解码器 -> 输出层
    目标：学习数据的压缩表示，并能够从压缩表示中重构原始数据
    """
    def __init__(self, n_input, n_hidden_1, n_hidden_2):
        """
        初始化自编码器
        参数：
            n_input: 输入维度（784）
            n_hidden_1: 第一隐藏层神经元数量（256）
            n_hidden_2: 第二隐藏层（瓶颈层）神经元数量（128）
        """
        super(StackedAutoencoder, self).__init__()
        
        # 编码器部分：将输入数据压缩到低维表示
        # 结构：784 -> 256 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden_1),  # 全连接层：784输入，256输出
            nn.Sigmoid(),                    # Sigmoid激活函数，将输出压缩到0-1范围
            nn.Linear(n_hidden_1, n_hidden_2), # 全连接层：256输入，128输出
            nn.Sigmoid()                     # Sigmoid激活函数
        )
        
        # 解码器部分：从低维表示重构原始数据
        # 结构：128 -> 256 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_1), # 全连接层：128输入，256输出
            nn.Sigmoid(),                    # Sigmoid激活函数
            nn.Linear(n_hidden_1, n_input),  # 全连接层：256输入，784输出
            nn.Sigmoid()                     # Sigmoid激活函数，输出在0-1范围
        )
    
    def forward(self, x):
        """
        前向传播过程
        参数：
            x: 输入数据，形状为(batch_size, 784)
        返回：
            重构的数据，形状与输入相同
        """
        encoded = self.encoder(x)    # 编码：将输入压缩到低维空间
        decoded = self.decoder(encoded)  # 解码：从低维表示重构数据
        return decoded

# 初始化模型
n_hidden_1 = 256  # 第一编码层神经元个数
n_hidden_2 = 128  # 第二编码层（瓶颈层）神经元个数

# 创建模型实例并将其移动到指定设备（GPU或CPU）
model = StackedAutoencoder(n_input, n_hidden_1, n_hidden_2).to(device)

# 打印模型结构
print("\nModel structure:")
print(model)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数，用于衡量重构图像与原始图像的差异
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器，用于更新模型参数。
#Adam（Adaptive Moment Estimation）优化器结合了动量法和RMSProp的优点。每个参数都有自适应的学习率

# 训练模型
print("\nStarting training...")
total_batch = len(train_loader)  # 计算总批次数

# 训练循环：遍历每个训练轮次
for epoch in range(training_epochs):
    model.train()  # 设置模型为训练模式（启用dropout和batch normalization）
    total_loss = 0  # 初始化累计损失
    
    # 遍历每个批次
    for i, (batch_xs, _) in enumerate(train_loader):
        # batch_xs: 当前批次的图像数据，形状为(batch_size, 784)
        # _: 标签（在自编码器中不需要使用）
        
        batch_xs = batch_xs.to(device)  # 将数据移动到指定设备
        
        # 前向传播
        outputs = model(batch_xs)       # 模型预测（重构图像）
        loss = criterion(outputs, batch_xs)  # 计算损失：重构图像与原始图像的差异
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()        # 反向传播，计算当前梯度
        optimizer.step()       # 更新模型参数
        
        total_loss += loss.item()  # 累加损失值
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    
    # 按显示间隔输出训练信息
    if epoch % display_step == 0:
        print(f"Epoch: {epoch+1:04d}/{training_epochs}, Average Loss: {avg_loss:.9f}")

print("Optimization Finished!")

# 测试模型并显示结果
model.eval()  # 设置模型为评估模式（禁用dropout和batch normalization）
with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
    # 获取测试集前examples_to_show个样本
    test_data = []
    test_label_data = []
    
    # 从测试数据加载器中收集足够的样本
    for i, (data, labels) in enumerate(test_loader):
        test_data.append(data)
        test_label_data.append(labels)
        if len(test_data) * batch_size >= examples_to_show:
            break
    
    # 合并批次数据并取前examples_to_show个样本
    test_images = torch.cat(test_data, dim=0)[:examples_to_show]
    test_labels_display = torch.cat(test_label_data, dim=0)[:examples_to_show]
    test_images = test_images.to(device)  # 移动到指定设备
    
    # 获取重构结果
    encode_decode = model(test_images)
    
    # 将张量转换为numpy数组并调整形状用于显示
    # 从(10, 784)调整为(10, 28, 28)
    original_images = test_images.cpu().numpy().reshape(-1, 28, 28)
    reconstructed_images = encode_decode.cpu().numpy().reshape(-1, 28, 28)
    
    # 显示原始图像和重构图像的对比
    # 创建2行examples_to_show列的子图
    fig, axes = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
    
    for i in range(examples_to_show):
        # 显示原始图像（第一行）
        axes[0, i].imshow(original_images[i], cmap='gray')  # 灰度图显示
        axes[0, i].axis('off')  # 关闭坐标轴
        axes[0, i].set_title(f'Label: {test_labels_display[i]}')  # 设置标题显示真实标签
        
        # 显示重构图像（第二行）
        axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')  # 设置标题
    
    # 调整子图布局
    plt.tight_layout()
    # 保存结果图像
    plt.savefig('autoencoder_results.png', dpi=150, bbox_inches='tight')
    # 显示图像
    plt.show()

# 保存训练好的模型
torch.save(model.state_dict(), 'stacked_autoencoder.pth')
print("Model saved as 'stacked_autoencoder.pth'")
print("Results saved as 'autoencoder_results.png'")

# 计算并显示最终的重构误差
model.eval()
with torch.no_grad():
    test_loss = 0
    # 在整个测试集上计算平均重构误差
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        outputs = model(data)
        test_loss += criterion(outputs, data).item()  # 累加每个批次的损失
    
    # 计算平均测试损失
    test_loss /= len(test_loader)
    print(f"Final test reconstruction error: {test_loss:.6f}")

"""
程序总结：
这个程序实现了一个堆栈自编码器，用于学习和重构MNIST手写数字图像。

主要步骤：
1. 数据加载：从本地MNIST_data目录加载MNIST数据集
2. 数据预处理：将图像数据归一化并转换为PyTorch张量
3. 模型定义：构建包含编码器和解码器的自编码器神经网络
4. 模型训练：使用均方误差损失和Adam优化器训练模型
5. 结果可视化：显示原始图像和重构图像的对比
6. 模型保存：保存训练好的模型参数

自编码器的工作原理：
- 编码器将784维的输入图像压缩到128维的瓶颈层表示
- 解码器从128维的压缩表示中重构出784维的原始图像
- 通过最小化重构误差，模型学习到了数据的重要特征表示

应用场景：
- 数据降维和特征学习
- 图像去噪
- 异常检测
- 数据压缩
"""