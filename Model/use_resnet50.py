from torchvision.transforms import transforms
from torchvision.models import resnet50
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Tuple

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载保存的模型
model = resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('Model/resnet50_model_data1.pth', map_location=device))

# 设置模型为评估模式
model.eval()

# 定义输入图像的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 定义一个函数来对新图像进行预测
def predict_image(image_path: str) -> Tuple[str, float]:
    """
    预测给定图像的类别和概率
    :param image_path: 图像的路径
    :return: 预测的类别名称和概率
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    model.to(device)
    with torch.no_grad():
        output = F.softmax(model(image), dim=1)
    probability, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()
    probability = probability.item()
    class_names = ['Fake', 'Real']
    return class_names[predicted_class], probability

def get_result(image_path: str) -> Tuple[str, float]:
    """
    获取图像的预测结果
    :param image_path: 图像的路径
    :return: 预测的类别名称和概率
    """
    predicted_class, probability = predict_image(image_path)
    return predicted_class, probability
