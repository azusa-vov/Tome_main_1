import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import random

class ImageNetDataLoader:
    """ImageNet数据加载器"""
    
    def __init__(self, root_dir, split='val', image_size=224):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        
        # ImageNet标准预处理
        if split == 'val':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:  # train
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def create_dataloader(self, batch_size=32, num_samples=None, shuffle=False):
        """创建数据加载器"""
        data_path = self.root_dir / self.split
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        print(f"📁 Loading ImageNet {self.split} data from: {data_path}")
        
        # 创建数据集
        dataset = datasets.ImageFolder(data_path, transform=self.transform)
        print(f"📊 Total samples in dataset: {len(dataset)}")
        print(f"📊 Number of classes: {len(dataset.classes)}")
        
        # 如果指定了样本数量，进行采样
        if num_samples and num_samples < len(dataset):
            print(f"🎯 Sampling {num_samples} samples from {len(dataset)} total samples")
            
            # 确保每个类别都有代表性的采样
            indices = self._stratified_sampling(dataset, num_samples)
            dataset = Subset(dataset, indices)
            
            print(f"✅ Sampled dataset size: {len(dataset)}")
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=4,  # 并行加载
            pin_memory=True
        )
    
    def _stratified_sampling(self, dataset, num_samples):
        """分层采样，确保每个类别都有代表性"""
        # 获取每个样本的类别
        targets = [dataset.samples[i][1] for i in range(len(dataset))]
        
        # 计算每个类别的样本数
        samples_per_class = num_samples // len(dataset.classes)
        remainder = num_samples % len(dataset.classes)
        
        selected_indices = []
        
        # 对每个类别进行采样
        for class_idx in range(len(dataset.classes)):
            # 找到该类别的所有样本索引
            class_indices = [i for i, target in enumerate(targets) if target == class_idx]
            
            # 确定该类别要采样的数量
            samples_for_this_class = samples_per_class + (1 if class_idx < remainder else 0)
            samples_for_this_class = min(samples_for_this_class, len(class_indices))
            
            # 随机采样
            selected = random.sample(class_indices, samples_for_this_class)
            selected_indices.extend(selected)
        
        return selected_indices

def create_imagenet_dataloader(dataset_root, batch_size=32, num_samples=None, split='val'):
    """创建ImageNet数据加载器的便捷函数"""
    loader = ImageNetDataLoader(dataset_root, split=split)
    return loader.create_dataloader(batch_size=batch_size, num_samples=num_samples)
