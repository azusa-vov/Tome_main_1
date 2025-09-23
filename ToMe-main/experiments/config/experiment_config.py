from dataclasses import dataclass
from typing import List, Optional
import os
import torch
from tome.similarity_strategy import *

def create_learnable_satd_strategy(checkpoint_path: Optional[str] = None, device: str = 'cuda') -> LearnableSATDSimilarity:
    """
    创建可学习SATD策略，可选择性地加载预训练权重
    
    Args:
        checkpoint_path: 检查点文件路径，None表示使用随机初始化的Hadamard矩阵
        device: 设备
    
    Returns:
        LearnableSATDSimilarity实例
    """
    strategy = LearnableSATDSimilarity(device=device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # 根据检查点格式加载权重
            if 'h_matrix' in checkpoint:
                strategy.H_matrix.data = checkpoint['h_matrix'].to(device)
                print(f"✅ Loaded LearnableSATD H matrix from {checkpoint_path}")
                if 'val_accuracy' in checkpoint:
                    print(f"   Checkpoint validation accuracy: {checkpoint['val_accuracy']:.2f}%")
            else:
                strategy.H_matrix.data = checkpoint.to(device)
                print(f"✅ Loaded LearnableSATD H matrix from {checkpoint_path}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint {checkpoint_path}: {e}")
            print(f"   Using default Hadamard initialization")
    else:
        if checkpoint_path:
            print(f"⚠️  Checkpoint file not found: {checkpoint_path}")
        print(f"   Using default Hadamard initialization for LearnableSATD")
    
    return strategy

@dataclass
class Config:
    # 基本配置
    model_name: str = "vit_base_patch16_224"
    dataset_path: str = "/home/tangrq/datasets/imagenet"
    output_dir: str = "results"
    
    # GPU配置
    gpu_id: Optional[int] = None  # None表示自动选择，数字表示指定GPU
    min_memory_mb: int = 3072     # 最小GPU内存要求(MB)
    wait_for_gpu: bool = True     # 是否等待GPU可用
    
    # 数据配置
    batch_size: int = 32
    num_samples: int = 10000  # 用于实验的样本数，None表示使用全部数据
    use_val_set: bool = True  # 使用验证集还是训练集

    # 实验参数
    r_values: List[int] = None
    strategies: List = None
    
    # 可学习SATD配置
    learnable_satd_checkpoint: Optional[str] = None  # 从环境变量或参数获取
    
    def __post_init__(self):
        if self.r_values is None:
            self.r_values = [0, 4, 8, 12, 16]
        
        # 从环境变量获取checkpoint路径
        if self.learnable_satd_checkpoint is None:
            self.learnable_satd_checkpoint = os.environ.get('LEARNABLE_SATD_CHECKPOINT')
        
        if self.strategies is None:
            self.strategies = [
                ("Cosine", CosineSimilarity()),
                ("Euclidean", EuclideanSimilarity()), 
                ("DotProduct", DotProductSimilarity()),
                ("SAD", SADSimilarity()),
            ]

# 预设配置
def quick_imagenet_config(gpu_id: Optional[int] = None, include_learnable_satd: bool = True):
    """快速ImageNet测试配置"""
    strategies = [
        ("Cosine", CosineSimilarity()),
        ("SAD", SADSimilarity()),
        ("SATD", SATDSimilarity()),
    ]
    
    config = Config(
        gpu_id=gpu_id,
        min_memory_mb=2048,
        batch_size=64,
        num_samples=2000,  # 每个类别约2个样本
        r_values=[0, 8, 16],
        strategies=strategies
    )
    
    # 如果启用可学习SATD且有checkpoint，则添加
    if include_learnable_satd:
        device = f'cuda:{gpu_id}' if gpu_id is not None else 'cuda'
        learnable_satd = create_learnable_satd_strategy(config.learnable_satd_checkpoint, device)
        config.strategies.append(("LearnableSATD", learnable_satd))
    
    return config

def full_imagenet_config(gpu_id: Optional[int] = None, include_learnable_satd: bool = True):
    """完整ImageNet实验配置"""
    strategies = [
        ("Cosine", CosineSimilarity()),
        ("Euclidean", EuclideanSimilarity()),
        ("DotProduct", DotProductSimilarity()),
        ("SAD", SADSimilarity()),
        ("SATD", SATDSimilarity()),
    ]
    
    config = Config(
        gpu_id=gpu_id,
        min_memory_mb=4096,
        batch_size=256,
        num_samples=None,  # 使用完整的50,000样本验证集
        r_values=[0, 4, 8, 12, 16, 20, 24],
        strategies=strategies
    )
    
    # 如果启用可学习SATD且有checkpoint，则添加
    if include_learnable_satd:
        device = f'cuda:{gpu_id}' if gpu_id is not None else 'cuda'
        learnable_satd = create_learnable_satd_strategy(config.learnable_satd_checkpoint, device)
        config.strategies.append(("LearnableSATD", learnable_satd))
    
    return config

def create_custom_config_with_learnable_satd(checkpoint_path: str, **kwargs):
    """创建包含指定checkpoint的自定义配置"""
    config = Config(**kwargs)
    config.learnable_satd_checkpoint = checkpoint_path
    
    # 创建可学习SATD策略
    device = f'cuda:{config.gpu_id}' if config.gpu_id is not None else 'cuda'
    learnable_satd = create_learnable_satd_strategy(checkpoint_path, device)
    config.strategies.append(("LearnableSATD", learnable_satd))
    
    return config
