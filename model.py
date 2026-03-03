import torch
import torch.nn as nn

# architecture model SimpleCNN
class SimpleCNN(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(64)

    self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.bn1b = nn.BatchNorm2d(64)

    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(128)

    self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.bn2b = nn.BatchNorm2d(128)

    self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(256)

    self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.bn3b = nn.BatchNorm2d(256)

    self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    self.bn4 = nn.BatchNorm2d(512)

    self.conv4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.bn4b = nn.BatchNorm2d(512)

    self.pool = nn.MaxPool2d(2, 2)
    self.relu = nn.ReLU()

    self.gap = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(0.3)
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn1b(self.conv1b(x)))
    x = self.pool(x)
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn2b(self.conv2b(x)))
    x = self.pool(x)
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.relu(self.bn3b(self.conv3b(x)))
    x = self.pool(x)
    x = self.relu(self.bn4(self.conv4(x)))
    x = self.relu(self.bn4b(self.conv4b(x)))
    x = self.pool(x)

    x = self.gap(x)             # [B, 512, 1, 1]
    x = x.view(x.size(0), -1)   # flatten -> [B, 512]
    x = self.dropout(x)
    x = self.fc(x)

    return x