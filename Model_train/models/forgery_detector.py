# forgery_detector.py
import torch
import torch.nn as nn
from base_model import BaseCNNModel  # 导入基础CNN模型


class ForgeryDetector(BaseCNNModel):
    def __init__(self, num_classes=2):  # 假设伪造检测是一个二分类问题
        super(ForgeryDetector, self).__init__(num_classes)
        # 在这里，你可以选择添加更多的层或修改模型结构
        # 例如，你可以添加dropout层来防止过拟合
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 调用基础模型的前向传播
        x = super(ForgeryDetector, self).forward(x)
        # 在这里，你可以添加更多的操作，比如应用softmax等
        # x = self.dropout(x)  # 如果添加了dropout层，则在这里使用它
        return x

    # 使用示例


# 实例化伪造检测模型
model = ForgeryDetector()

# 打印模型结构
print(model)