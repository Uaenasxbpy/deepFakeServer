import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# 加载训练好的模型
class FakeImageDetector(nn.Module):
    def __init__(self):
        super(FakeImageDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 卷积层1
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化层1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 卷积层2
        self.bn2 = nn.BatchNorm2d(32)  # 批归一化层2
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # 全连接层1
        self.fc2 = nn.Linear(128, 2)  # 全连接层2

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = FakeImageDetector()
model.load_state_dict(torch.load('Model_train/fake_image_detector.pth'))
model.eval()

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 预测图像真实性的函数
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加批次维度
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    confidence = torch.softmax(outputs, 1)[0][predicted[0]].item() * 100
    return predicted.item(), confidence

def get_result(image_path):
    # 提供图像路径
    # image_path = 'path_to_your_image.jpg'  # 替换为你的图像路径
    prediction, confidence = predict_image(image_path)
    # 输出预测结果和置信度
    print("预测类别:", prediction)  # 假设0表示真实，1表示伪造
    print("置信度 (%):", confidence)
    return prediction, "{:.2f}".format(confidence)


