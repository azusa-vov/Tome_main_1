# 可学习SATD (Learnable SATD)

这个模块实现了可学习的SATD变换矩阵，用于优化ViT Token Merging的性能。与固定的Hadamard变换不同，这里的H矩阵是可以通过端到端训练学习的。

## 概述

传统的SATD算法使用固定的8x8 Hadamard变换矩阵。我们的想法是让这个变换矩阵变成可学习的参数，通过在ImageNet数据集上训练来优化Token Merging的性能，而不是简单地拟合余弦相似度。

### 核心思想

1. **目标**: 训练一个8x8的可学习变换矩阵H，使得使用SATD算法进行Token Merging后，ViT的图像分类准确性更高
2. **方法**: 固定预训练ViT的参数，只训练H矩阵的64个值
3. **数据**: 使用ImageNet训练集进行训练
4. **优化目标**: 最大化分类准确性，而不是拟合某种相似度度量

## 文件结构

```
tome/
├── similarity_strategy.py          # 新增LearnableSATDSimilarity类
train/
├── train_learnable_satd.py        # 训练脚本
examples/
├── test_learnable_satd.py         # 测试脚本
├── learnable_satd_workflow.py     # 完整工作流
└── LEARNABLE_SATD_README.md       # 本文档
experiments/
└── config/
    └── experiment_config.py       # 更新了配置支持
```

## 使用方法

### 1. 快速开始 (不推荐)

使用完整工作流脚本，一键训练和测试**（不推荐）**：

```bash
# 训练并测试 (使用默认参数)
python examples/learnable_satd_workflow.py --action both

# 只训练
python examples/learnable_satd_workflow.py --action train --epochs 20 --samples 20000

# 只测试 (使用已有的检查点)
python examples/learnable_satd_workflow.py --action test --checkpoint learned_satd_checkpoints/best_h_matrix.pth
```

### 2. 详细训练（推荐）

如果你想更精细地控制训练过程：

```bash
python train/train_learnable_satd.py \
    --model vit_base_patch16_224 \
    --dataset /path/to/imagenet \
    --epochs 15 \
    --batch-size 64 \
    --lr 0.001 \
    --samples 15000 \
    --r-value 8 \
    --save-dir my_learned_satd \
    --gpu-id 0
```

#### 训练参数说明

- `--model`: ViT模型名称 (默认: vit_base_patch16_224)
- `--dataset`: ImageNet数据集路径
- `--epochs`: 训练轮数 (默认: 10)
- `--batch-size`: 批次大小 (默认: 64)
- `--lr`: 学习率 (默认: 0.001)
- `--samples`: 训练样本数 (默认: 10000)
- `--r-value`: Token merging的r值 (默认: 8)
- `--save-dir`: 保存目录 (默认: learned_satd_checkpoints)
- `--gpu-id`: 指定GPU ID

### 3. 测试训练结果

```bash
python examples/test_learnable_satd.py \
    --checkpoint learned_satd_checkpoints/best_h_matrix.pth \
    --gpu-id 0
```

## 输出文件

训练过程会生成以下文件：

### 检查点文件

- `best_h_matrix.pth`: 验证集上性能最好的H矩阵
- `checkpoint_epoch_X.pth`: 各个epoch的检查点
- `training_history.json`: 训练历史记录

### 检查点内容

每个检查点包含：

```python
{
    'epoch': 训练轮数,
    'h_matrix': 学习到的8x8变换矩阵,
    'val_accuracy': 验证集准确率,
    'train_metrics': 训练指标,
    'val_metrics': 验证指标
}
```

## 技术细节

### LearnableSATDSimilarity类

```python
class LearnableSATDSimilarity(SimilarityStrategy):
    def __init__(self, device='cuda'):
        # 初始化为标准Hadamard矩阵
        hadamard_8 = torch.tensor([...], dtype=torch.float32, device=device)
        self.H_matrix = torch.nn.Parameter(hadamard_8.clone())
    
    def compute_similarity(self, a, b):
        # 使用可学习的H矩阵计算SATD相似度
        ...
```

### 训练策略

1. **模型固定**: 预训练ViT的所有参数都被冻结
2. **只训练H矩阵**: 只有8x8=64个参数需要学习
3. **端到端优化**: 直接优化分类性能，而不是相似度拟合
4. **数据增强**: 使用标准的ImageNet预处理

### 内存和计算需求

- **GPU内存**: 建议至少4GB (batch_size=64时)
- **训练时间**: 约10-30分钟/epoch (取决于样本数和GPU)
- **参数数量**: 仅64个可学习参数

## 实验建议

### 超参数调优

1. **学习率**: 推荐范围 [0.0001, 0.01]
   ```bash
   # 尝试不同学习率
   python train/train_learnable_satd.py --lr 0.0005
   python train/train_learnable_satd.py --lr 0.005
   ```

2. **训练样本数**: 更多样本通常效果更好
   ```bash
   # 使用更多样本
   python train/train_learnable_satd.py --samples 50000 --epochs 5
   ```

3. **不同r值**: 为不同的token merging强度训练不同的H矩阵
   ```bash
   # 为r=4训练
   python train/train_learnable_satd.py --r-value 4 --save-dir learned_satd_r4
   # 为r=16训练  
   python train/train_learnable_satd.py --r-value 16 --save-dir learned_satd_r16
   ```

### 性能分析

训练完成后，可以比较不同策略的性能：

```python
# 在配置中同时包含多种策略
strategies = [
    ("Cosine", CosineSimilarity()),
    ("SATD", SATDSimilarity()),           # 固定Hadamard
    ("LearnableSATD", learnable_satd),    # 学习的H矩阵
]
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   python train/train_learnable_satd.py --batch-size 32
   ```

2. **数据集路径错误**
   ```bash
   # 确认ImageNet数据集结构
   /path/to/imagenet/
   ├── train/
   │   ├── n01440764/
   │   └── ...
   └── val/
       ├── n01440764/
       └── ...
   ```

3. **训练不收敛**
   - 尝试更小的学习率 (--lr 0.0001)
   - 增加训练样本数 (--samples 20000)
   - 检查数据加载是否正确

### 调试模式

添加更详细的日志：

```bash
# 设置详细日志
export PYTHONPATH=/path/to/ToMe:$PYTHONPATH
python -u train/train_learnable_satd.py --batch-size 16 --samples 1000
```

## 理论背景

### 为什么可学习SATD有效？

1. **任务特定优化**: 与固定变换不同，可学习变换可以适应特定的视觉任务
2. **特征空间适配**: H矩阵可以学习到更适合ViT特征的变换
3. **端到端优化**: 直接优化最终性能而非中间相似度度量

### 与传统方法的区别

| 方法 | 变换矩阵 | 优化目标 | 参数数量 |
|------|----------|----------|----------|
| 传统SATD | 固定Hadamard | 通用变换 | 0 |
| 可学习SATD | 可训练矩阵 | 分类性能 | 64 |

## 未来扩展

1. **多尺度H矩阵**: 为不同的patch size设计不同的变换
2. **层级特定H矩阵**: 为ViT的不同层学习不同的变换
3. **动态H矩阵**: 根据输入内容自适应调整变换矩阵
