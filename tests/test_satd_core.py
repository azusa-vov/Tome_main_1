"""
SATD核心功能测试
"""

import torch
import pytest
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from satd_core import SATDSimilarity, HMatrix, SATDTokenMerger, SATDLoss


class TestSATDSimilarity:
    """测试SATD相似性计算"""
    
    def test_similarity_basic(self):
        """测试基本相似性计算"""
        similarity = SATDSimilarity(temperature=0.5, alpha=0.7)
        
        # 创建测试数据
        h_i = torch.randn(4, 256)
        h_j = torch.randn(4, 256)
        
        # 计算相似性
        sim = similarity(h_i, h_j)
        
        # 检查输出形状
        assert sim.shape == (4,), f"Expected shape (4,), got {sim.shape}"
        
        # 检查值范围 (相似性应该在合理范围内)
        assert torch.all(sim >= -2) and torch.all(sim <= 2), "Similarity values out of expected range"
    
    def test_similarity_identical(self):
        """测试相同向量的相似性"""
        similarity = SATDSimilarity(temperature=0.5, alpha=0.7)
        
        h = torch.randn(4, 256)
        sim = similarity(h, h)
        
        # 相同向量的相似性应该很高
        assert torch.all(sim > 0.5), "Identical vectors should have high similarity"
    
    def test_similarity_orthogonal(self):
        """测试正交向量的相似性"""
        similarity = SATDSimilarity(temperature=0.5, alpha=1.0)  # 只使用余弦相似度
        
        # 创建正交向量
        h_i = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        h_j = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        
        sim = similarity(h_i, h_j)
        
        # 正交向量的余弦相似度应该接近0
        assert torch.abs(sim[0]) < 0.1, f"Orthogonal vectors should have near-zero similarity, got {sim[0]}"


class TestHMatrix:
    """测试H矩阵"""
    
    def test_h_matrix_initialization(self):
        """测试H矩阵初始化"""
        max_tokens = 16
        embed_dim = 256
        
        h_matrix = HMatrix(max_tokens, embed_dim, symmetric=True)
        
        # 检查H矩阵形状
        H = h_matrix(max_tokens)
        assert H.shape == (max_tokens, max_tokens), f"Expected shape ({max_tokens}, {max_tokens}), got {H.shape}"
        
        # 检查对称性
        if h_matrix.symmetric:
            assert torch.allclose(H, H.T, atol=1e-6), "H matrix should be symmetric"
    
    def test_h_matrix_subset(self):
        """测试H矩阵子集"""
        max_tokens = 20
        embed_dim = 256
        
        h_matrix = HMatrix(max_tokens, embed_dim)
        
        # 测试不同大小的子集
        for num_tokens in [5, 10, 15, 20]:
            H = h_matrix(num_tokens)
            assert H.shape == (num_tokens, num_tokens), f"Expected shape ({num_tokens}, {num_tokens}), got {H.shape}"
    
    def test_h_matrix_regularization(self):
        """测试H矩阵正则化损失"""
        max_tokens = 10
        embed_dim = 256
        
        h_matrix = HMatrix(max_tokens, embed_dim)
        reg_loss = h_matrix.get_regularization_loss(max_tokens)
        
        # 正则化损失应该是标量
        assert reg_loss.dim() == 0, "Regularization loss should be a scalar"
        assert torch.isfinite(reg_loss), "Regularization loss should be finite"


class TestSATDTokenMerger:
    """测试SATD token合并器"""
    
    def test_merger_initialization(self):
        """测试合并器初始化"""
        embed_dim = 256
        max_tokens = 32
        
        merger = SATDTokenMerger(
            embed_dim=embed_dim,
            max_tokens=max_tokens,
            similarity_threshold=0.8
        )
        
        assert merger.embed_dim == embed_dim
        assert merger.similarity_threshold == 0.8
    
    def test_merger_forward(self):
        """测试合并器前向传播"""
        embed_dim = 256
        max_tokens = 32
        batch_size = 2
        num_tokens = 16
        
        merger = SATDTokenMerger(
            embed_dim=embed_dim,
            max_tokens=max_tokens,
            similarity_threshold=0.7
        )
        
        # 创建测试token
        tokens = torch.randn(batch_size, num_tokens, embed_dim)
        
        # 执行合并
        merged_tokens, info = merger(tokens)
        
        # 检查输出
        assert merged_tokens.shape[0] == batch_size, "Batch size should be preserved"
        assert merged_tokens.shape[2] == embed_dim, "Embed dimension should be preserved"
        assert merged_tokens.shape[1] <= num_tokens, "Token count should not increase"
        
        # 检查信息字典
        required_keys = ['similarity_matrix', 'merge_pairs', 'original_token_count', 
                        'merged_token_count', 'reduction_ratio', 'h_matrix_reg_loss']
        for key in required_keys:
            assert key in info, f"Missing key in info: {key}"
    
    def test_similarity_matrix_computation(self):
        """测试相似性矩阵计算"""
        embed_dim = 128
        max_tokens = 16
        batch_size = 2
        num_tokens = 8
        
        merger = SATDTokenMerger(
            embed_dim=embed_dim,
            max_tokens=max_tokens,
            similarity_threshold=0.8
        )
        
        tokens = torch.randn(batch_size, num_tokens, embed_dim)
        similarity_matrix = merger.compute_pairwise_similarity(tokens)
        
        # 检查相似性矩阵形状
        assert similarity_matrix.shape == (batch_size, num_tokens, num_tokens)
        
        # 检查对角线元素（自身相似性应该较高）
        for b in range(batch_size):
            diag_elements = torch.diag(similarity_matrix[b])
            # 注意：由于H矩阵的存在，对角线元素可能不是最大值
            assert torch.all(torch.isfinite(diag_elements)), "Diagonal elements should be finite"
    
    def test_merge_pairs_finding(self):
        """测试合并对查找"""
        embed_dim = 64
        max_tokens = 16
        batch_size = 1
        num_tokens = 4
        
        merger = SATDTokenMerger(
            embed_dim=embed_dim,
            max_tokens=max_tokens,
            similarity_threshold=0.5  # 较低阈值，更容易找到合并对
        )
        
        # 创建相似的token
        tokens = torch.randn(batch_size, num_tokens, embed_dim)
        # 让token 0和1非常相似
        tokens[0, 1] = tokens[0, 0] + 0.01 * torch.randn(embed_dim)
        
        similarity_matrix = merger.compute_pairwise_similarity(tokens)
        merge_pairs = merger.find_merge_pairs(similarity_matrix)
        
        # 应该找到至少一些合并对
        assert len(merge_pairs) == batch_size
        assert isinstance(merge_pairs[0], list)


class TestSATDLoss:
    """测试SATD损失函数"""
    
    def test_loss_initialization(self):
        """测试损失函数初始化"""
        loss_fn = SATDLoss(
            lambda_diversity=0.1,
            lambda_structure=0.1,
            lambda_h_reg=0.01
        )
        
        assert loss_fn.lambda_diversity == 0.1
        assert loss_fn.lambda_structure == 0.1
        assert loss_fn.lambda_h_reg == 0.01
    
    def test_loss_computation(self):
        """测试损失计算"""
        batch_size = 2
        num_tokens = 8
        embed_dim = 128
        
        loss_fn = SATDLoss()
        
        # 创建测试数据
        similarity_matrix = torch.rand(batch_size, num_tokens, num_tokens)
        target_similarities = torch.rand(batch_size, num_tokens, num_tokens)
        original_tokens = torch.randn(batch_size, num_tokens, embed_dim)
        merged_tokens = torch.randn(batch_size, num_tokens - 2, embed_dim)  # 假设合并了2个token
        h_reg_loss = torch.tensor(0.1)
        
        # 计算损失
        loss_dict = loss_fn(
            similarity_matrix=similarity_matrix,
            target_similarities=target_similarities,
            original_tokens=original_tokens,
            merged_tokens=merged_tokens,
            h_reg_loss=h_reg_loss
        )
        
        # 检查损失字典
        required_keys = ['total_loss', 'similarity_loss', 'diversity_loss', 
                        'structure_loss', 'h_reg_loss']
        for key in required_keys:
            assert key in loss_dict, f"Missing key in loss_dict: {key}"
            assert torch.isfinite(loss_dict[key]), f"Loss {key} should be finite"
    
    def test_individual_losses(self):
        """测试各个损失组件"""
        batch_size = 2
        num_tokens = 6
        embed_dim = 64
        
        loss_fn = SATDLoss()
        
        # 测试相似性损失
        sim_matrix = torch.rand(batch_size, num_tokens, num_tokens)
        target_sim = torch.rand(batch_size, num_tokens, num_tokens)
        sim_loss = loss_fn.similarity_loss(sim_matrix, target_sim)
        assert sim_loss >= 0, "Similarity loss should be non-negative"
        
        # 测试多样性损失
        tokens = torch.randn(batch_size, num_tokens, embed_dim)
        div_loss = loss_fn.diversity_loss(tokens)
        assert torch.isfinite(div_loss), "Diversity loss should be finite"
        
        # 测试结构损失
        original_tokens = torch.randn(batch_size, num_tokens, embed_dim)
        merged_tokens = torch.randn(batch_size, num_tokens - 1, embed_dim)
        struct_loss = loss_fn.structure_loss(original_tokens, merged_tokens)
        assert struct_loss >= 0, "Structure loss should be non-negative"


def test_integration():
    """集成测试"""
    print("运行SATD集成测试...")
    
    # 参数设置
    batch_size = 2
    num_tokens = 12
    embed_dim = 256
    max_tokens = 20
    
    # 创建模型
    merger = SATDTokenMerger(
        embed_dim=embed_dim,
        max_tokens=max_tokens,
        similarity_threshold=0.6,
        temperature=0.5,
        alpha=0.7
    )
    
    loss_fn = SATDLoss(
        lambda_diversity=0.1,
        lambda_structure=0.1,
        lambda_h_reg=0.01
    )
    
    # 创建测试数据
    tokens = torch.randn(batch_size, num_tokens, embed_dim)
    target_similarities = torch.rand(batch_size, num_tokens, num_tokens)
    
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
    
    # 检查梯度计算
    total_loss = loss_dict['total_loss']
    total_loss.backward()
    
    # 检查是否有梯度
    has_gradients = False
    for param in merger.parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            has_gradients = True
            break
    
    assert has_gradients, "Model should have gradients after backward pass"
    
    print("集成测试通过！")


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
    
    # 运行集成测试
    test_integration()