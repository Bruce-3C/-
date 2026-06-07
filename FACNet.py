import torch
import torch.nn as nn
from torchvision.models import resnet18

class FACNet(nn.Module):
    def __init__(self, num_classes=6):
        super(FACNet, self).__init__()
        # 修正核心：當初訓練時並未將模型包在 self.backbone 變數中
        # 而是直接將基礎骨幹與全連接層對接到最外層
        base_model = resnet18(weights=None)
        
        # 複製 ResNet18 的所有特徵提取層
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
        # 配合 Unexpected keys 中的 "fc.weight" 與 "fc.bias"
        # 當初訓練時分類層名稱為單一的 self.fc
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x