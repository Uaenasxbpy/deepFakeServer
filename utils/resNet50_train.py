import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50
from model_train import ModelTrainer
from model_evaluate import ModelEvaluator

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置轮数
epoch = 20

# 数据预处理步骤
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
train_size = int(0.9 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False)

# 加载预训练的ResNet模型，并移除顶部的全连接层（fc层）
model = resnet50(pretrained=True)

# 获取fc层的输入特征数
num_ftrs = model.fc.in_features

# 替换顶部的fc层以适应二分类问题，2个输出，对应真实和伪造两个类别
model.fc = nn.Linear(num_ftrs, 2)

# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用随机梯度下降优化器
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.Adam(model.fc.parameters())

# 训练评估模型
trainer = ModelTrainer(model=model,
                       criterion=criterion,
                       optimizer=optimizer,
                       train_loader=train_loader,
                       valid_loader=valid_loader,
                       device=device,
                       epoch=epoch)
trainer.train_model()

# 保存模型
torch.save(model.state_dict(), '../Model/resnet50_model.pth')
print('Training complete')


dataset_test = datasets.ImageFolder(root='G:\Pycharm\Project1\deepFakeServer\Model_train\CASIA2.0_revised\\test', transform=transform)
test_loader= DataLoader(dataset=dataset_test, batch_size=10, shuffle=False)
evaluator = ModelEvaluator(model=model,
                           test_loader=valid_loader,
                           device=device)
evaluator.evaluate()
