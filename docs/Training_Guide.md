# SATD训练指南

## 概述

本指南提供了使用SATD方法训练Vision Transformer模型的详细步骤和最佳实践。SATD (Self-Attention Token Detection) 是一种改进的token merging方法，通过优化H矩阵来提高模型效率。

## 训练准备

### 1. 环境配置

```bash
# 安装依赖
pip install torch torchvision numpy matplotlib
pip install timm  # 用于预训练的ViT模型
```

### 2. 数据准备

SATD可以应用于各种视觉任务：

- **图像分类**: ImageNet, CIFAR-10/100
- **目标检测**: COCO, Pascal VOC
- **语义分割**: ADE20K, Cityscapes

## 核心训练参数

### SATD超参数

| 参数 | 推荐值 | 作用 |
|------|--------|------|
| `similarity_threshold` | 0.7-0.9 | 控制token合并的激进程度 |
| `temperature` | 0.1-1.0 | 距离敏感度调节 |
| `alpha` | 0.5-0.9 | 余弦相似度vs距离相似度权重 |
| `lambda_diversity` | 0.1 | 多样性损失权重 |
| `lambda_structure` | 0.1 | 结构保持损失权重 |
| `lambda_h_reg` | 0.01 | H矩阵正则化权重 |

### 学习率设置

- **主模型学习率**: 1e-4 ~ 5e-4
- **H矩阵学习率**: 主模型学习率的 1/10
- **预热步数**: 总步数的 10%

## 训练策略

### 1. 渐进式训练

```python
# 阶段1: 固定threshold，训练H矩阵
phase1_threshold = 0.9  # 保守合并
train_epochs = 20

# 阶段2: 降低threshold，微调整个模型
phase2_threshold = 0.7  # 更激进合并
finetune_epochs = 10
```

### 2. 课程学习

从简单到复杂的训练顺序：

1. **预训练阶段**: 使用较高的相似性阈值
2. **中期训练**: 逐步降低阈值
3. **精调阶段**: 使用目标阈值

### 3. 自适应阈值调整

```python
def adaptive_threshold_schedule(epoch, total_epochs):
    """自适应阈值调整策略"""
    start_threshold = 0.9
    end_threshold = 0.7
    
    # 线性衰减
    threshold = start_threshold - (start_threshold - end_threshold) * (epoch / total_epochs)
    return max(threshold, end_threshold)
```

## 损失函数设计

### 1. 多任务损失

```python
total_loss = task_loss + satd_loss
```

其中：
- `task_loss`: 主任务损失（分类、检测等）
- `satd_loss`: SATD相关损失

### 2. 损失权重调度

```python
def loss_weight_schedule(epoch, total_epochs):
    """损失权重调度策略"""
    # 早期更注重任务损失，后期逐渐增加SATD损失权重
    task_weight = 1.0
    satd_weight = min(0.5, 0.1 * (epoch / total_epochs))
    
    return task_weight, satd_weight
```

## 模型集成示例

### 与ViT集成

```python
import torch.nn as nn
from satd_core import SATDTokenMerger

class SATDViT(nn.Module):
    def __init__(self, base_vit, satd_config):
        super().__init__()
        self.base_vit = base_vit
        self.satd_merger = SATDTokenMerger(**satd_config)
        
    def forward(self, x):
        # 获取patch embedding
        tokens = self.base_vit.patch_embed(x)
        
        # 在中间层应用SATD
        for i, block in enumerate(self.base_vit.blocks):
            tokens = block(tokens)
            
            # 在特定层应用token merging
            if i in [3, 6, 9]:  # 示例：在第4、7、10层应用
                tokens, info = self.satd_merger(tokens)
        
        # 最终分类
        return self.base_vit.head(tokens.mean(dim=1))
```

## 训练监控

### 关键指标

1. **Token减少比例**: 监控合并效率
2. **相似性损失**: 确保合并质量
3. **多样性损失**: 防止过度合并
4. **主任务性能**: 确保精度不下降

### 可视化工具

```python
def plot_training_metrics(metrics_dict):
    """绘制训练指标"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Token减少比例
    axes[0,0].plot(metrics_dict['reduction_ratio'])
    axes[0,0].set_title('Token Reduction Ratio')
    
    # 相似性损失
    axes[0,1].plot(metrics_dict['similarity_loss'])
    axes[0,1].set_title('Similarity Loss')
    
    # 主任务损失
    axes[1,0].plot(metrics_dict['task_loss'])
    axes[1,0].set_title('Task Loss')
    
    # 总损失
    axes[1,1].plot(metrics_dict['total_loss'])
    axes[1,1].set_title('Total Loss')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
```

## 性能优化技巧

### 1. 批处理优化

```python
# 使用检查点节省内存
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    return checkpoint(self._forward_impl, x)
```

### 2. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 动态token数量

根据输入复杂度动态调整token数量：

```python
def dynamic_token_count(image_complexity):
    """根据图像复杂度动态调整token数量"""
    if image_complexity > 0.8:
        return 196  # 保持更多token
    elif image_complexity > 0.5:
        return 144  # 中等减少
    else:
        return 100  # 大幅减少
```

## 评估与部署

### 模型评估

1. **精度评估**: 在验证集上测试主任务性能
2. **效率评估**: 测量推理时间和内存使用
3. **鲁棒性评估**: 测试不同输入下的稳定性

### 部署优化

```python
# 模型量化
import torch.quantization

model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 模型导出
torch.jit.script(model).save('satd_model.pt')
```

## 故障排除

### 常见问题

1. **过度合并**: 降低similarity_threshold
2. **合并不足**: 提高similarity_threshold或降低temperature
3. **训练不稳定**: 减小H矩阵学习率
4. **精度下降**: 增加structure_loss权重

### 调试工具

```python
def debug_satd_behavior(merger, tokens):
    """调试SATD行为"""
    with torch.no_grad():
        merged_tokens, info = merger(tokens)
        
        print(f"原始token数: {info['original_token_count']}")
        print(f"合并后token数: {info['merged_token_count']}")
        print(f"平均相似性: {info['similarity_matrix'].mean():.4f}")
        
        # 可视化相似性矩阵
        plt.imshow(info['similarity_matrix'][0].cpu())
        plt.colorbar()
        plt.title('Debug: Similarity Matrix')
        plt.show()
```

## 最佳实践总结

1. **从保守参数开始**: 使用较高threshold和较小权重
2. **渐进式训练**: 分阶段调整参数
3. **监控关键指标**: 重点关注精度和效率平衡
4. **充分验证**: 在多个数据集上测试泛化性能
5. **版本控制**: 记录所有实验配置和结果

## 进阶话题

### 1. 多尺度SATD

在不同分辨率下应用SATD，适应不同大小的目标。

### 2. 注意力引导合并

使用注意力权重指导token合并决策。

### 3. 在线学习

在推理过程中动态调整合并策略。

这些技术将在未来版本中详细介绍。