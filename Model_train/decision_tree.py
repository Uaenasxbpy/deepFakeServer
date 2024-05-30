from miniVGGModel import MiniVGG_Second
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.models import resnet50
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os

width = 224
height = 224
depth = 3
classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# VGG模型
vgg_model = MiniVGG_Second(width=width,
                           height=height,
                           depth=depth,
                           classes=classes).to(device)
vgg_model.load_state_dict(torch.load('vgg_model_weights.pth', map_location=device))

# ResNet模型
resnet_model = resnet50(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(num_ftrs, 2)
resnet_model.load_state_dict(torch.load('resnet50_fake_detection_model.pth', map_location=device))

def predict_image_vgg(image_path, model, transform, device):
    '''
    :param image_path:
    :param model:
    :param transform:
    :param device:
    :return:
    '''
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = F.softmax(model(image), dim=1)
    return output.cpu().numpy()[0]

def predict_image_resnet(image_path, model, transform, device):
    '''
    :param image_path:
    :param model:
    :param transform:
    :param device:
    :return:
    '''
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = F.softmax(model(image), dim=1)
    return output.cpu().numpy()[0]


def get_image_paths_and_labels(valid_dir):
    '''
    :param valid_dir:
    :return:
    '''
    image_paths = []
    labels = []
    class_names = ['real', 'fake']
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(valid_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg', '.tif')):
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(label)
    return image_paths, labels

# 替换为你的验证集文件夹路径
valid_dir = 'G:\Pycharm\Project1\deepFakeServer\Model_train\CASIA2.0_revised\\test'
val_image_paths, val_labels = get_image_paths_and_labels(valid_dir)


# 获取验证集上两个模型的预测概率
vgg_probs = [predict_image_vgg(p, vgg_model, transform, device) for p in val_image_paths]
resnet_probs = [predict_image_resnet(p, resnet_model, transform, device) for p in val_image_paths]

# 拼接预测概率作为逻辑回归模型的输入
X_train = np.hstack([vgg_probs, resnet_probs])
y_train = np.array(val_labels)

# 训练逻辑回归模型
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 验证逻辑回归模型
y_pred = log_reg.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f'Logistic Regression Training Accuracy: {accuracy:.4f}')


def predict_combined(image_path):
    vgg_prob = predict_image_vgg(image_path, vgg_model, transform, device)
    resnet_prob = predict_image_resnet(image_path, resnet_model, transform, device)
    combined_prob = np.hstack([vgg_prob, resnet_prob]).reshape(1, -1)
    final_pred = log_reg.predict(combined_prob)
    final_prob = log_reg.predict_proba(combined_prob)[0][final_pred[0]]
    class_names = ['Real', 'Fake']
    return class_names[final_pred[0]], final_prob

# 示例用法
image_path = 'G:\Pycharm\Project1\deepFakeServer\Model_train\OIP-C.jpg'
predicted_class, probability = predict_combined(image_path)
print(f'Predicted Class: {predicted_class}, Probability: {probability:.4f}')
