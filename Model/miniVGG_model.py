import torch.nn as nn
import torch.nn.functional as F


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

        self.fc1_input_size = 64 * (width // 4) * (height // 4)
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

class MiniVGG_Second(nn.Module):
    def __init__(self, width, height, depth, classes):
        super(MiniVGG_Second, self).__init__()
        self.conv1 = nn.Conv2d(depth, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Increased depth
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)  # Increased depth
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)  # Increased depth
        self.bn5 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1_input_size = 128 * (width // 8) * (height // 8)  # Adjusted input size
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = self.dropout3(x)
        x = x.view(-1, self.fc1_input_size)
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
