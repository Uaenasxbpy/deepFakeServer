import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
from PIL import Image
import seaborn as sns

# 加载模型

# 加载预训练的ResNet模型，并移除顶部的全连接层（fc层）
model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features  # 获取fc层的输入特征数
# 替换顶部的fc层以适应二分类问题（真实或伪造）
model.fc = nn.Linear(num_ftrs, 2)  # 2个输出，对应真实和伪造两个类别
model.eval()  # 设置为评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 加载权重
model.load_state_dict(torch.load('resnet50_fake_detection_model.pth'))

# 数据预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),     # 将图像调整为224x224，这是ResNet的默认输入尺寸
    transforms.ToTensor(),             # 将PIL Image或numpy.ndarray转换为torch.Tensor，并缩放到[0.0, 1.0]
    transforms.Normalize(              # 标准化图像
        mean=[0.485, 0.456, 0.406],    # 在ImageNet上预训练的模型使用的均值
        std=[0.229, 0.224, 0.225]      # 在ImageNet上预训练的模型使用的标准差
    )
])

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels='fake', yticklabels='real')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    # 计算召回率
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print("Recall:", recall)

valid_dataset = datasets.ImageFolder(root='data/train', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
evaluate_model(model, valid_loader)
