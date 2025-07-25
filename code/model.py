import torch
import torchvision.models as models

def create_model(num_classes=101, pretrained=True):
    # 使用ResNet-18架构
    model = models.resnet18(pretrained=pretrained)
    
    # 修改最后一层全连接层
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    return model