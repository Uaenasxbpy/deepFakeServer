import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model_train import ModelTrainer
from model_evaluate import ModelEvaluator
from miniVGG_model import MiniVGG_Second

width = 224
height = 224
depth = 3
classes = 2

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置轮数
epoch = 2

# 定义输入图像的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 加载数据集
# TODO 路径需改为训练集
dataset = datasets.ImageFolder(root='G:\Pycharm\Project1\deepFakeServer\Model_train\CASIA2.0_revised\\train', transform=transform)

'''注意调整比例参数'''
train_size = int(0.2 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 导入模型
model = MiniVGG_Second(width=width,
                       height=height,
                       depth=depth,
                       classes=classes)
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 训练评估模型
trainer = ModelTrainer(model=model,
                       criterion=criterion,
                       optimizer=optimizer,
                       train_loader=train_loader,
                       valid_loader=valid_loader,
                       device=device,
                       epoch=epoch)
trainer.train_model()
# TODO 保存模型

'''注：这里的valid_loader应该替换为test_loader'''
dataset_test = datasets.ImageFolder(root='G:\Pycharm\Project1\deepFakeServer\Model_train\CASIA2.0_revised\\test', transform=transform)
test_loader= DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)
evaluator = ModelEvaluator(model=model,
                           test_loader=valid_loader,
                           device=device)
evaluator.evaluate()