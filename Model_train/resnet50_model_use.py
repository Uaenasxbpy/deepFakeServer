import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.models import resnet50
from PIL import Image

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载保存的模型
model = resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('resnet50_fake_detection_model.pth', map_location=device))
model.eval()  # 设置模型为评估模式

# 定义输入图像的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
    class_names = ['Fake', 'real']                  # 定义类别名称
    return class_names[predicted_class], probability


def get_result(image_path):
    # 提供图像路径
    # image_path = 'path_to_your_image.jpg'  # 替换为你的图像路径
    predicted_class, confidence = predict_image(image_path)
    if predicted_class == 'real':
        prediction = 0
    else:
        prediction = 1
    # 输出预测结果和置信度
    print("预测类别:", prediction)  # 假设0表示真实，1表示伪造
    print("置信度 (%):", confidence)
    return prediction, "{:.2f}".format(confidence)

# 示例用法
# image_path = 'OIP-C.jpg'  # 替换为你的图像路径
# predicted_class, probability = predict_image(image_path)
# print(f'Predicted Class: {predicted_class}, Probability: {probability:.4f}')
