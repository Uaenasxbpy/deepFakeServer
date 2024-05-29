import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torchvision.transforms import functional as F
from model_train import ModelTrainer
from model_evaluate import ModelEvaluator

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
dataset = datasets.ImageFolder(root='G:\Pycharm\Project1\deepFakeServer\Model_train\data\CASIA2.0_revised', transform=transform)
train_size = int(0.4 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练的ResNet模型，并移除顶部的全连接层（fc层）
model = resnet50(pretrained=True)

# 获取fc层的输入特征数
num_ftrs = model.fc.in_features

# 替换顶部的fc层以适应二分类问题，2个输出，对应真实和伪造两个类别
model.fc = nn.Linear(num_ftrs, 2)

# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# TODO 训练评估模型
trainer = ModelTrainer(model=model,
                       criterion=criterion,
                       optimizer=optimizer,
                       train_loader=train_loader,
                       valid_loader=test_loader,
                       device=device,
                       epoch=2)
trainer.train_model()
evaluator = ModelEvaluator(model=model,
                           test_loader=test_loader,
                           device=device)
evaluator.evaluate()
