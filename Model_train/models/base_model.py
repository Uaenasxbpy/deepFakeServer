# base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseCNNModel, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义全连接层
        self.fc = nn.Linear(64 * 7 * 7, num_classes)  # 假设输入图像大小是 224x224，经过两次池化后变为 7x7

    def forward(self, x):
        # 前向传播过程
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 展平特征图
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.fc(x)
        return x

    # 使用示例


num_classes = 2  # 假设伪造检测是一个二分类问题
base_model = BaseCNNModel(num_classes)

# 打印模型结构
print(base_model)