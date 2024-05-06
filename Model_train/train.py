import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 加载数据集
print("Loading datasets...")
train_real_dataset = ImageFolder('data/train', transform=transform)
train_fake_dataset = ImageFolder('data/train', transform=transform)
test_real_dataset = ImageFolder('data/valid', transform=transform)
test_fake_dataset = ImageFolder('data/valid', transform=transform)

# 创建数据加载器
batch_size = 32
train_real_loader = DataLoader(train_real_dataset, batch_size=batch_size, shuffle=True)
train_fake_loader = DataLoader(train_fake_dataset, batch_size=batch_size, shuffle=True)
test_real_loader = DataLoader(test_real_dataset, batch_size=batch_size)
test_fake_loader = DataLoader(test_fake_dataset, batch_size=batch_size)

# 定义CNN模型
class FakeImageDetector(nn.Module):
    def __init__(self):
        super(FakeImageDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # 32x32 is the output size after pooling
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型并将其移动到GPU（如果可用）
model = FakeImageDetector().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录训练和测试损失以及准确率
train_real_losses = []
train_fake_losses = []
test_real_losses = []
test_fake_losses = []
test_real_accuracies = []
test_fake_accuracies = []

# 训练模型
num_epochs = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    # 训练真实数据
    print("Training on real data...")
    model.train()
    for batch_idx, (images, labels) in enumerate(train_real_loader):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_real_losses.append(loss.item())
        print(f"\rBatch {batch_idx+1}/{len(train_real_loader)}", end='')

    # 训练伪造数据
    print("\nTraining on fake data...")
    model.train()
    for batch_idx, (images, labels) in enumerate(train_fake_loader):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_fake_losses.append(loss.item())
        print(f"\rBatch {batch_idx+1}/{len(train_fake_loader)}", end='')

    # 验证模型
    print("\nValidating...")
    model.eval()
    correct_real = 0
    total_real = 0
    correct_fake = 0
    total_fake = 0
    with torch.no_grad():
        for images, labels in test_real_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_real_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total_real += labels.size(0)
            correct_real += predicted.eq(labels).sum().item()

        for images, labels in test_fake_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_fake_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total_fake += labels.size(0)
            correct_fake += predicted.eq(labels).sum().item()

    test_real_accuracy = 100 * correct_real / total_real
    test_fake_accuracy = 100 * correct_fake / total_fake
    test_real_accuracies.append(test_real_accuracy)
    test_fake_accuracies.append(test_fake_accuracy)

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_real_losses, label='Train Real Loss')
    plt.plot(train_fake_losses, label='Train Fake Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_real_losses, label='Test Real Loss')
    plt.plot(test_fake_losses, label='Test Fake Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(test_real_accuracies, label='Test Real Accuracy')
    plt.plot(test_fake_accuracies, label='Test Fake Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 保存模型
torch.save(model.state_dict(), 'fake_image_detector.pth')
print("Model saved.")
