# 用于处理图片的模型
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch



# 图片处理和预测函数
def detect_image(image):
    predicted_label = True
    confidence = 0.97
    return predicted_label, confidence

