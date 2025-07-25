import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------------------------------
# 1. 目录划分函数（官方 30 train / rest test）
# -------------------------------------------------
def make_standard_split(root: str):
    
    src_dir = os.path.join(root, '101_ObjectCategories')
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(root, split), exist_ok=True)

    for cls in os.listdir(src_dir):
        cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = sorted(os.listdir(cls_path))
        np.random.shuffle(imgs)

        train_imgs = imgs[:30]              # 官方训练集
        rest_imgs  = imgs[30:]              # 剩余做测试
        val_imgs   = rest_imgs[:len(rest_imgs)//2]   # 再从剩余里取一半做验证
        test_imgs  = rest_imgs[len(rest_imgs)//2:]   # 另一半做测试

        for split, split_imgs in [('train', train_imgs),
                                  ('val',   val_imgs),
                                  ('test',  test_imgs)]:
            dst_cls = os.path.join(root, split, cls)
            os.makedirs(dst_cls, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(os.path.join(cls_path, img),
                             os.path.join(dst_cls, img))

# -------------------------------------------------
# 2. 数据加载函数
# -------------------------------------------------
def load_my_data(data_dir: str):

    # 0) 若尚未划分，则自动执行一次
    if not all(os.path.isdir(os.path.join(data_dir, s))
               for s in ['train', 'val', 'test']):
        print('首次运行，正在生成 Caltech-101 官方划分 …')
        make_standard_split(data_dir)
        print('划分完成！')

    # 1) 数据增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # 2) 使用 ImageFolder 直接读取三个子目录
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    # 3) 构造 DataLoader
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=32,
                      shuffle=(x in ['train', 'val']),
                      num_workers=4)
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print("Caltech-101 Classes:", class_names)
    print("Dataset sizes:", dataset_sizes)
    return dataloaders, dataset_sizes