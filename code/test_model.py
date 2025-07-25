import torch
from torch.utils.data import DataLoader
from model import create_model
from dataloader import Caltech101Dataset
import os
import argparse
from torchvision import transforms

def test_model(data_path, model_path, batch_size=64):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据集
    test_dataset = Caltech101Dataset(data_path, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 创建并加载模型
    model = create_model(num_classes=101, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 测试模型
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Caltech-101 Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to Caltech-101 dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    test_model(args.data_path, args.model_path, args.batch_size)