import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torchvision.transforms import functional as F

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),     # 将图像调整为224x224，这是ResNet的默认输入尺寸
    transforms.ToTensor(),             # 将PIL Image或numpy.ndarray转换为torch.Tensor，并缩放到[0.0, 1.0]
    transforms.Normalize(              # 标准化图像
        mean=[0.485, 0.456, 0.406],    # 在ImageNet上预训练的模型使用的均值
        std=[0.229, 0.224, 0.225]      # 在ImageNet上预训练的模型使用的标准差
    )
])

# 加载数据集
dataset = datasets.ImageFolder(root='data/CASIA2.0_revised', transform=transform)
# 划分数据集为训练集和测试集
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练的ResNet模型，并移除顶部的全连接层（fc层）
model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features  # 获取fc层的输入特征数
# 替换顶部的fc层以适应二分类问题（真实或伪造）
model.fc = nn.Linear(num_ftrs, 2)  # 2个输出，对应真实和伪造两个类别
model = model.to(device)  # 将模型移到GPU上（如果有的话）

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)  # 使用随机梯度下降优化器

num_epochs = 10  # 训练轮数

# 训练模型
for epoch in range(num_epochs):
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Loss: {running_loss / len(train_loader):.4f}')
    print(f'Accuracy: {100 * correct / total:.2f}%')

# 保存模型
torch.save(model.state_dict(), '../Model/resnet50_model_data1.pth')
print('Training complete')

