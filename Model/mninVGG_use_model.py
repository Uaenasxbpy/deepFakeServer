import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.models import resnet50
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载保存的模型
# model = resnet50(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 2)
# model.load_state_dict(torch.load('Model_train/resnet50_fake_detection_model.pth', map_location=device))



class MiniVGG(nn.Module):
    def __init__(self, width, height, depth, classes):
        super(MiniVGG, self).__init__()
        self.conv1 = nn.Conv2d(depth, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1_input_size = 64 * (width // 4) * (height // 4)  # Adjusted input size
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout2(x)
        x = x.view(-1, self.fc1_input_size)  # Flatten before fully connected layer
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

width = 64
height = 64
depth = 3
classes = 2
model = MiniVGG(width, height, depth, classes).to(device)
model.load_state_dict(torch.load('Model/vgg_model_weights.pth', map_location=device))


model.eval()  # 设置模型为评估模式

# 定义输入图像的转换
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义一个函数来对新图像进行预测
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 加载图像
    image = transform(image).unsqueeze(0)           # 应用转换并添加批次维度
    image = image.to(device)                        # 将图像移至设备
    model.to(device)                                # 将模型移至设备
    with torch.no_grad():
        output = F.softmax(model(image), dim=1)     # 使用softmax获取概率
    probability, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()        # 获取预测类别索引
    probability = probability.item()                # 获取预测类别的概率
    # class_names = ['Fake', 'real']                  # 定义类别名称
    class_names = ['Cat', 'Dog']                  # 定义类别名称

    return class_names[predicted_class], probability


def get_result(image_path):
    # 提供图像路径
    # image_path = 'path_to_your_image.jpg'  # 替换为你的图像路径
    predicted_class, confidence = predict_image(image_path)
    # if predicted_class == 'Fake':
    if predicted_class == 'Cat':
        prediction = 0
    else:
        prediction = 1
    # 输出预测结果和置信度
    print("预测类别:", prediction)  # 假设0表示真实，1表示伪造
    print("置信度 (%):", confidence)
    return prediction, "{:.2f}".format(confidence * 100)

# 示例用法
# image_path = 'OIP-C.jpg'  # 替换为你的图像路径
# predicted_class, probability = predict_image(image_path)
# print(f'Predicted Class: {predicted_class}, Probability: {probability:.4f}')
