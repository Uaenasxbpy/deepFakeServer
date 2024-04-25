# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.forgery_detector import ForgeryDetector


# 定义数据加载器
class ForgeryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = [os.path.join(root_dir, img_name) for img_name in os.listdir(root_dir)]
        self.labels = [0] * len(self.images) if 'real' in root_dir else [1] * len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    # 数据预处理


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据你的模型输入大小调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet均值和标准差
])

# 创建数据加载器
real_dataset = ForgeryDataset(root_dir='data/real', transform=transform)
fake_dataset = ForgeryDataset(root_dir='data/fake', transform=transform)
dataloader = DataLoader([real_dataset, fake_dataset], batch_size=32, shuffle=True)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ForgeryDetector().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10  # 训练轮数
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 清零梯度缓存
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()  # 累加损失值

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

# 保存训练好的模型
torch.save(model.state_dict(), 'trained_model.pth')