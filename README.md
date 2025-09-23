# Tome_main_1 - SATD Token Merging Framework

这是关于token merging的改进，运用了SATD (Self-Attention Token Detection) 的similarity计算方法，旨在训练H矩阵的矩阵元对于ViT模型的影响。

## 项目概述

本项目实现了一个完整的SATD框架，用于改善Vision Transformer中的token merging过程。通过学习可优化的H矩阵和改进的相似性计算方法，SATD能够更智能地合并相似token，从而在保持模型精度的同时显著提高计算效率。

## 核心特性

### 🚀 核心功能
- **改进的相似性计算**: 结合余弦相似度和欧几里得距离的混合相似性度量
- **可学习H矩阵**: 自适应学习token之间的关系权重
- **智能token合并**: 基于相似性阈值的层次化合并策略
- **多样性保持**: 防止过度合并导致的信息损失
- **结构保持**: 维护原始token的全局结构特征

### 📊 性能优势
- **计算效率**: 减少token数量，降低计算复杂度
- **精度保持**: 智能合并策略保持关键信息
- **可解释性**: 相似性计算过程透明
- **适应性**: 可应用于不同的ViT架构

## 项目结构

```
Tome_main_1/
├── README.md                 # 项目说明
├── config.yaml              # 配置文件
├── requirements.txt          # 依赖包
├── docs/                     # 文档目录
│   ├── SATD_Theory.md       # SATD理论基础
│   └── Training_Guide.md    # 训练指南
├── src/                      # 源代码
│   ├── satd_core.py         # SATD核心实现
│   └── utils.py             # 工具函数
├── examples/                 # 示例代码
│   └── basic_usage.py       # 基本使用示例
├── tests/                    # 测试代码
│   └── test_satd_core.py    # 核心功能测试
└── outputs/                  # 输出文件
    ├── similarity_matrix.png # 相似性矩阵可视化
    ├── h_matrix.png         # H矩阵可视化
    └── training_loss.png    # 训练损失曲线
```

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/azusa-vov/Tome_main_1.git
cd Tome_main_1

# 安装依赖
pip install -r requirements.txt
```

### 2. 基本使用

```python
import torch
from src.satd_core import SATDTokenMerger, SATDLoss

# 创建SATD合并器
merger = SATDTokenMerger(
    embed_dim=256,
    max_tokens=196,
    similarity_threshold=0.8,
    temperature=0.5,
    alpha=0.7
)

# 准备输入token (Batch_size, Num_tokens, Embed_dim)
tokens = torch.randn(2, 64, 256)

# 执行token合并
merged_tokens, info = merger(tokens)

print(f"原始token数量: {info['original_token_count']}")
print(f"合并后token数量: {info['merged_token_count']}")
print(f"减少比例: {info['reduction_ratio']:.2%}")
```

### 3. 运行示例

```bash
# 运行完整示例
python examples/basic_usage.py

# 运行测试
python -m pytest tests/test_satd_core.py -v
```

## SATD理论基础

### 相似性计算

SATD使用改进的混合相似性度量：

```
SATD_sim(i,j) = α * cos(h_i, h_j) + (1-α) * exp(-||h_i - h_j||_2 / τ)
```

其中：
- `α`: 余弦相似度权重 (0-1)
- `τ`: 温度参数，控制距离敏感度
- `cos(h_i, h_j)`: 余弦相似度
- `exp(-||h_i - h_j||_2 / τ)`: 距离相似度

### H矩阵优化

H矩阵学习token之间的关系权重，优化目标：
- 最大化相似token对的权重
- 最小化不相似token对的权重  
- 保持矩阵的对称性和正定性

### 损失函数

总损失包含多个组件：

```
L_total = L_similarity + λ₁ * L_diversity + λ₂ * L_structure + λ₃ * L_h_reg
```

- `L_similarity`: 相似性损失
- `L_diversity`: 多样性损失
- `L_structure`: 结构保持损失
- `L_h_reg`: H矩阵正则化损失

## 配置参数

主要超参数及推荐值：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `similarity_threshold` | 0.7-0.9 | 相似性阈值，控制合并激进程度 |
| `temperature` | 0.1-1.0 | 温度参数，控制距离敏感度 |
| `alpha` | 0.5-0.9 | 余弦相似度权重 |
| `lambda_diversity` | 0.1 | 多样性损失权重 |
| `lambda_structure` | 0.1 | 结构损失权重 |
| `lambda_h_reg` | 0.01 | H矩阵正则化权重 |

详细配置请参考 `config.yaml` 文件。

## 性能对比

与传统方法的对比：

| 方法 | 相似性计算 | 合并策略 | 可解释性 | 效率 |
|------|------------|----------|----------|------|
| 传统注意力 | 点积 | 固定 | 低 | 中等 |
| ToMe | 余弦相似度 | 贪心 | 中等 | 高 |
| **SATD** | **混合相似度 + H矩阵** | **自适应** | **高** | **高** |

## 应用场景

- **图像分类**: 减少patch token数量，提高推理速度
- **目标检测**: 优化特征金字塔，降低计算成本
- **语义分割**: 改善像素级预测的计算效率
- **多模态学习**: 实现跨模态token对齐和合并

## 可视化结果

项目生成的可视化文件展示了SATD的工作原理：

1. **相似性矩阵** (`similarity_matrix.png`): 显示token之间的相似性关系
2. **H矩阵** (`h_matrix.png`): 展示学习到的token关系权重
3. **训练曲线** (`training_loss.png`): 显示各项损失的变化趋势

## 文档

- [SATD理论基础](docs/SATD_Theory.md): 详细的数学推导和理论分析
- [训练指南](docs/Training_Guide.md): 完整的训练流程和最佳实践

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。在贡献之前，请确保：

1. 运行所有测试并确保通过
2. 遵循现有的代码风格
3. 添加适当的文档和注释

## 许可证

本项目采用MIT许可证。详细信息请参见LICENSE文件。

## 致谢

本项目基于Vision Transformer和Token Merging的相关研究，特别感谢以下工作的启发：

- Vision Transformer (ViT)
- Token Merging (ToMe)
- Self-Attention机制相关研究

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [项目Issues页面](https://github.com/azusa-vov/Tome_main_1/issues)
- Email: [联系邮箱]

---

**注**: 本项目专注于学术研究和技术创新，旨在推进Vision Transformer的效率优化技术发展。
