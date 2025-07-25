import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import create_model
from dataloader import get_caltech101_dataloaders
import os
import argparse

def train_fine_tune_model(data_path, save_path, lr=0.0001, num_epochs=200, batch_size=64):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据加载器
    train_loader, val_loader, _ = get_caltech101_dataloaders(data_path, batch_size)
    
    # 创建模型
    model = create_model(num_classes=101, pretrained=True)
    model = model.to(device)
    
    # 冻结除最后一层外的所有层
    for param in model.parameters():
        param.requires_grad = False
    
    # 只训练最后一层
    model.fc.requires_grad = True
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter()
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # 训练模型
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
    
    # 关闭TensorBoard写入器
    writer.close()
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Caltech-101 Fine-tuning')
    parser.add_argument('--data_path', type=str, required=True, help='Path to Caltech-101 dataset')
    parser.add_argument('--save_path', type=str, default='./models', help='Path to save models')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    train_fine_tune_model(args.data_path, args.save_path, args.lr, args.epochs, args.batch_size)