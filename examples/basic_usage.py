"""
SATD基本使用示例
演示如何使用SATD进行token merging
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from satd_core import SATDTokenMerger, SATDLoss


def create_sample_tokens(batch_size: int = 2, num_tokens: int = 16, embed_dim: int = 256):
    """
    创建示例token数据
    模拟ViT中的patch token
    """
    # 创建一些具有相似性的token
    base_tokens = torch.randn(batch_size, num_tokens, embed_dim)
    
    # 人工创建一些相似的token对
    for b in range(batch_size):
        # 让token 0和1相似
        base_tokens[b, 1] = base_tokens[b, 0] + 0.1 * torch.randn(embed_dim)
        # 让token 4和5相似
        base_tokens[b, 5] = base_tokens[b, 4] + 0.1 * torch.randn(embed_dim)
        # 让token 8和9相似
        base_tokens[b, 9] = base_tokens[b, 8] + 0.1 * torch.randn(embed_dim)
    
    return base_tokens


def visualize_similarity_matrix(similarity_matrix: torch.Tensor, save_path: str = None):
    """
    可视化相似性矩阵
    """
    # 取第一个batch的相似性矩阵
    sim_mat = similarity_matrix[0].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_mat, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('SATD Similarity Matrix')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"相似性矩阵图已保存到: {save_path}")
    else:
        plt.show()


def demonstrate_basic_usage():
    """
    演示SATD的基本使用
    """
    print("=" * 50)
    print("SATD Token Merging 基本使用演示")
    print("=" * 50)
    
    # 参数设置
    batch_size = 2
    num_tokens = 16
    embed_dim = 256
    max_tokens = 32
    
    # 创建示例数据
    print(f"创建示例数据: batch_size={batch_size}, num_tokens={num_tokens}, embed_dim={embed_dim}")
    tokens = create_sample_tokens(batch_size, num_tokens, embed_dim)
    print(f"原始token形状: {tokens.shape}")
    
    # 创建SATD合并器
    merger = SATDTokenMerger(
        embed_dim=embed_dim,
        max_tokens=max_tokens,
        similarity_threshold=0.7,  # 降低阈值以便看到合并效果
        temperature=0.5,
        alpha=0.7
    )
    
    # 执行token合并
    print("\n执行token合并...")
    merged_tokens, info = merger(tokens)
    
    # 显示结果
    print(f"合并后token形状: {merged_tokens.shape}")
    print(f"原始token数量: {info['original_token_count']}")
    print(f"合并后token数量: {info['merged_token_count']}")
    print(f"减少比例: {info['reduction_ratio']:.2%}")
    
    # 显示合并对信息
    for b, pairs in enumerate(info['merge_pairs']):
        if pairs:
            print(f"Batch {b} 合并对: {pairs}")
        else:
            print(f"Batch {b}: 没有找到需要合并的token对")
    
    # 可视化相似性矩阵
    print("\n生成相似性矩阵可视化...")
    visualize_similarity_matrix(info['similarity_matrix'], 
                               save_path='similarity_matrix.png')
    
    return tokens, merged_tokens, info


def demonstrate_training_process():
    """
    演示SATD的训练过程
    """
    print("\n" + "=" * 50)
    print("SATD训练过程演示")
    print("=" * 50)
    
    # 参数设置
    batch_size = 4
    num_tokens = 20
    embed_dim = 128
    max_tokens = 32
    num_epochs = 10
    
    # 创建模型和损失函数
    merger = SATDTokenMerger(
        embed_dim=embed_dim,
        max_tokens=max_tokens,
        similarity_threshold=0.6,
        temperature=0.3,
        alpha=0.8
    )
    
    loss_fn = SATDLoss(
        lambda_diversity=0.1,
        lambda_structure=0.1,
        lambda_h_reg=0.01
    )
    
    optimizer = torch.optim.Adam(merger.parameters(), lr=0.001)
    
    # 训练循环
    losses = []
    
    for epoch in range(num_epochs):
        # 创建训练数据
        tokens = create_sample_tokens(batch_size, num_tokens, embed_dim)
        
        # 创建目标相似性矩阵（示例：希望相邻token更相似）
        target_similarities = torch.zeros(batch_size, num_tokens, num_tokens)
        for b in range(batch_size):
            for i in range(num_tokens):
                for j in range(num_tokens):
                    if abs(i - j) == 1:  # 相邻token
                        target_similarities[b, i, j] = 0.8
                    elif i == j:  # 自身
                        target_similarities[b, i, j] = 1.0
                    else:
                        target_similarities[b, i, j] = 0.1
        
        # 前向传播
        merged_tokens, info = merger(tokens)
        
        # 计算损失
        loss_dict = loss_fn(
            similarity_matrix=info['similarity_matrix'],
            target_similarities=target_similarities,
            original_tokens=tokens,
            merged_tokens=merged_tokens,
            h_reg_loss=info['h_matrix_reg_loss']
        )
        
        total_loss = loss_dict['total_loss']
        losses.append(total_loss.item())
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 打印进度
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  总损失: {total_loss.item():.4f}")
            print(f"  相似性损失: {loss_dict['similarity_loss'].item():.4f}")
            print(f"  多样性损失: {loss_dict['diversity_loss'].item():.4f}")
            print(f"  结构损失: {loss_dict['structure_loss'].item():.4f}")
            print(f"  H矩阵正则化: {loss_dict['h_reg_loss'].item():.4f}")
            print(f"  Token减少比例: {info['reduction_ratio']:.2%}")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('SATD Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    print(f"\n训练损失曲线已保存到: training_loss.png")
    
    return merger, losses


def analyze_h_matrix(merger: SATDTokenMerger):
    """
    分析训练后的H矩阵
    """
    print("\n" + "=" * 50)
    print("H矩阵分析")
    print("=" * 50)
    
    # 获取H矩阵
    num_tokens = 16
    H = merger.H_matrix(num_tokens).detach().cpu().numpy()
    
    print(f"H矩阵形状: {H.shape}")
    print(f"H矩阵对角线元素平均值: {np.mean(np.diag(H)):.4f}")
    print(f"H矩阵非对角线元素平均值: {np.mean(H[~np.eye(num_tokens, dtype=bool)]):.4f}")
    
    # 可视化H矩阵
    plt.figure(figsize=(10, 8))
    vmax = np.abs(H).max()
    plt.imshow(H, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.title('Learned H Matrix')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    plt.savefig('h_matrix.png', dpi=150, bbox_inches='tight')
    print(f"H矩阵可视化已保存到: h_matrix.png")
    
    return H


def performance_comparison():
    """
    性能对比分析
    """
    print("\n" + "=" * 50)
    print("性能对比分析")
    print("=" * 50)
    
    batch_size = 8
    num_tokens = 64
    embed_dim = 512
    
    # 创建测试数据
    tokens = create_sample_tokens(batch_size, num_tokens, embed_dim)
    
    # 不同配置的SATD
    configs = [
        {'threshold': 0.5, 'temperature': 0.1, 'alpha': 0.5, 'name': 'Conservative'},
        {'threshold': 0.7, 'temperature': 0.5, 'alpha': 0.7, 'name': 'Balanced'}, 
        {'threshold': 0.9, 'temperature': 1.0, 'alpha': 0.9, 'name': 'Aggressive'}
    ]
    
    results = []
    
    for config in configs:
        merger = SATDTokenMerger(
            embed_dim=embed_dim,
            max_tokens=num_tokens,
            similarity_threshold=config['threshold'],
            temperature=config['temperature'],
            alpha=config['alpha']
        )
        
        # 测试性能
        import time
        start_time = time.time()
        
        merged_tokens, info = merger(tokens)
        
        end_time = time.time()
        
        results.append({
            'name': config['name'],
            'config': config,
            'reduction_ratio': info['reduction_ratio'],
            'processing_time': end_time - start_time,
            'merged_count': info['merged_token_count']
        })
    
    # 打印结果
    print("配置对比结果:")
    print("-" * 80)
    print(f"{'配置':<12} {'阈值':<6} {'温度':<6} {'α':<6} {'减少比例':<10} {'处理时间':<10} {'剩余token':<10}")
    print("-" * 80)
    
    for result in results:
        config = result['config']
        print(f"{result['name']:<12} {config['threshold']:<6} {config['temperature']:<6} "
              f"{config['alpha']:<6} {result['reduction_ratio']:<10.2%} "
              f"{result['processing_time']:<10.4f} {result['merged_count']:<10}")
    
    return results


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    os.chdir('outputs')
    
    try:
        # 基本使用演示
        tokens, merged_tokens, info = demonstrate_basic_usage()
        
        # 训练过程演示
        trained_merger, losses = demonstrate_training_process()
        
        # H矩阵分析
        H = analyze_h_matrix(trained_merger)
        
        # 性能对比
        results = performance_comparison()
        
        print("\n" + "=" * 50)
        print("演示完成！生成的文件:")
        print("- similarity_matrix.png: 相似性矩阵可视化")
        print("- training_loss.png: 训练损失曲线")
        print("- h_matrix.png: 学习到的H矩阵")
        print("=" * 50)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()