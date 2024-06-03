from Model.miniVGG_model import MiniVGG_Second
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.models import resnet50
from PIL import Image
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

class DeepFakeClassifier:
    def __init__(self):
        '''
        深度假像分类器初始化函数
        '''
        self.vgg_model_path = 'G:\Pycharm\Project1\deepFakeServer\Model\\vgg_model.pth'
        self.resnet_model_path = 'G:\Pycharm\Project1\deepFakeServer\Model\\resnet50_model.pth'
        self.valid_dir = 'G:\Pycharm\Project1\deepFakeServer\Model_train\CASIA2.0_revised\\test'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 加载VGG模型
        self.vgg_model = MiniVGG_Second(width=224, height=224, depth=3, classes=2).to(self.device)
        self.vgg_model.load_state_dict(torch.load(self.vgg_model_path, map_location=self.device))

        # 加载ResNet模型
        self.resnet_model = resnet50(pretrained=False)
        num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model.fc = torch.nn.Linear(num_ftrs, 2)
        self.resnet_model.load_state_dict(torch.load(self.resnet_model_path, map_location=self.device))
        self.resnet_model.to(self.device)

        # 获取验证集的图像路径和标签
        self.val_image_paths, self.val_labels = self.get_image_paths_and_labels()

        # 获取验证集上两个模型的预测概率
        self.vgg_probs = [self.predict_image_vgg(p) for p in self.val_image_paths]
        self.resnet_probs = [self.predict_image_resnet(p) for p in self.val_image_paths]

        # 拼接预测概率作为逻辑回归模型的输入
        self.X_train = np.hstack([self.vgg_probs, self.resnet_probs])
        self.y_train = np.array(self.val_labels)

        # 训练逻辑回归模型
        self.log_reg = LogisticRegression()
        self.log_reg.fit(self.X_train, self.y_train)

    def predict_image_vgg(self, image_path):
        '''
        预测图像的真实性/伪造性（使用VGG模型）
        :param image_path: 图像文件路径
        :return: 概率数组
        '''
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        self.vgg_model.eval()
        with torch.no_grad():
            output = F.softmax(self.vgg_model(image), dim=1)
        return output.cpu().numpy()[0]

    def predict_image_resnet(self, image_path):
        '''
        预测图像的真实性/伪造性（使用ResNet模型）
        :param image_path: 图像文件路径
        :return: 概率数组
        '''
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        self.resnet_model.eval()
        with torch.no_grad():
            output = F.softmax(self.resnet_model(image), dim=1)
        return output.cpu().numpy()[0]

    def get_image_paths_and_labels(self):
        '''
        获取图像文件路径和标签
        :return: 图像文件路径列表，标签列表
        '''
        image_paths = []
        labels = []
        class_names = ['real', 'fake']
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(self.valid_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.png', '.jpeg', '.tif')):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(label)
        return image_paths, labels

    def predict_combined(self, image_path):
        '''
        综合使用两个模型进行图像真实性/伪造性的预测
        :param image_path: 图像文件路径
        :return: 预测类别 str，预测概率  float
        '''
        vgg_prob = self.predict_image_vgg(image_path)
        resnet_prob = self.predict_image_resnet(image_path)
        combined_prob = np.hstack([vgg_prob, resnet_prob]).reshape(1, -1)
        final_pred = self.log_reg.predict(combined_prob)
        final_prob = self.log_reg.predict_proba(combined_prob)[0][final_pred[0]]
        class_names = ['Real', 'Fake']
        return class_names[final_pred[0]], final_prob

# 示例用法
'''
classifier = DeepFakeClassifier()
image_path = 'G:\Pycharm\Project1\deepFakeServer\Model_train\OIP-C.jpg'
predicted_class, probability = classifier.predict_combined(image_path)
print(f'Predicted Class: {predicted_class}, Probability: {probability:.4f}')
'''

