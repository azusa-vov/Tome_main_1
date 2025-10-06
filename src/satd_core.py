"""
SATD (Self-Attention Token Detection) Core Implementation
用于Vision Transformer中的token merging优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class SATDSimilarity(nn.Module):
    """
    SATD相似性计算模块
    结合余弦相似度和欧几里得距离的混合相似性度量
    """
    
    def __init__(self, temperature: float = 0.5, alpha: float = 0.7):
        """
        Args:
            temperature: 温度参数，控制距离敏感度
            alpha: 余弦相似度权重，控制两种相似性度量的平衡
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """
        计算SATD相似性
        
        Args:
            h_i: token i的特征表示 [B, D]
            h_j: token j的特征表示 [B, D]
            
        Returns:
            相似性分数 [B]
        """
        # 余弦相似度
        cos_sim = F.cosine_similarity(h_i, h_j, dim=-1)
        
        # 欧几里得距离
        euclidean_dist = torch.norm(h_i - h_j, p=2, dim=-1)
        
        # 距离相似性 (距离越小，相似性越高)
        dist_sim = torch.exp(-euclidean_dist / self.temperature)
        
        # 混合相似性
        similarity = self.alpha * cos_sim + (1 - self.alpha) * dist_sim
        
        return similarity


class HMatrix(nn.Module):
    """
    可学习的H矩阵，用于优化token之间的关系权重
    """
    
    def __init__(self, max_tokens: int, embed_dim: int, symmetric: bool = True):
        """
        Args:
            max_tokens: 最大token数量
            embed_dim: 嵌入维度
            symmetric: 是否保持矩阵对称性
        """
        super().__init__()
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim
        self.symmetric = symmetric
        
        # 初始化H矩阵
        self.H = nn.Parameter(torch.randn(max_tokens, max_tokens) * 0.1)
        
        # 如果要求对称，使用对称初始化
        if symmetric:
            with torch.no_grad():
                self.H.data = (self.H.data + self.H.data.T) / 2
                
    def forward(self, num_tokens: int) -> torch.Tensor:
        """
        获取当前使用的H矩阵子集
        
        Args:
            num_tokens: 当前token数量
            
        Returns:
            H矩阵 [num_tokens, num_tokens]
        """
        H_sub = self.H[:num_tokens, :num_tokens]
        
        if self.symmetric:
            # 保持对称性
            H_sub = (H_sub + H_sub.T) / 2
            
        return H_sub
    
    def get_regularization_loss(self, num_tokens: int) -> torch.Tensor:
        """
        计算H矩阵的正则化损失，确保矩阵性质良好
        """
        H_sub = self.forward(num_tokens)
        
        # 对角线正则化 (鼓励对角线元素为正)
        diag_loss = -torch.mean(torch.diag(H_sub))
        
        # 非对角线稀疏性正则化
        off_diag_mask = ~torch.eye(num_tokens, device=H_sub.device, dtype=bool)
        off_diag_loss = torch.mean(torch.abs(H_sub[off_diag_mask]))
        
        return diag_loss + 0.1 * off_diag_loss


class SATDTokenMerger(nn.Module):
    """
    基于SATD的token合并器
    """
    
    def __init__(self, 
                 embed_dim: int,
                 max_tokens: int,
                 similarity_threshold: float = 0.8,
                 temperature: float = 0.5,
                 alpha: float = 0.7):
        """
        Args:
            embed_dim: 嵌入维度
            max_tokens: 最大token数量
            similarity_threshold: 相似性阈值
            temperature: 温度参数
            alpha: 余弦相似度权重
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.similarity_threshold = similarity_threshold
        
        # 相似性计算模块
        self.similarity = SATDSimilarity(temperature, alpha)
        
        # H矩阵
        self.H_matrix = HMatrix(max_tokens, embed_dim)
        
    def compute_pairwise_similarity(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        计算所有token对的相似性
        
        Args:
            tokens: 输入token [B, N, D]
            
        Returns:
            相似性矩阵 [B, N, N]
        """
        B, N, D = tokens.shape
        
        # 展开为成对计算
        tokens_i = tokens.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        tokens_j = tokens.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        
        # 重塑为批量计算
        h_i = tokens_i.reshape(B * N * N, D)
        h_j = tokens_j.reshape(B * N * N, D)
        
        # 计算相似性
        similarities = self.similarity(h_i, h_j)
        
        # 重塑回矩阵形式
        similarity_matrix = similarities.reshape(B, N, N)
        
        # 应用H矩阵权重
        H = self.H_matrix(N)
        similarity_matrix = similarity_matrix * H.unsqueeze(0)
        
        return similarity_matrix
    
    def find_merge_pairs(self, similarity_matrix: torch.Tensor) -> list:
        """
        基于相似性矩阵找到需要合并的token对
        
        Args:
            similarity_matrix: 相似性矩阵 [B, N, N]
            
        Returns:
            合并对列表
        """
        B, N, _ = similarity_matrix.shape
        merge_pairs = []
        
        for b in range(B):
            sim_mat = similarity_matrix[b]
            
            # 只考虑上三角矩阵，避免重复
            mask = torch.triu(torch.ones(N, N, device=sim_mat.device), diagonal=1).bool()
            sim_upper = sim_mat[mask]
            
            # 找到超过阈值的相似对
            valid_pairs = sim_upper > self.similarity_threshold
            
            if valid_pairs.any():
                # 获取位置索引
                indices = torch.nonzero(mask, as_tuple=False)
                valid_indices = indices[valid_pairs]
                
                # 按相似性排序，优先合并最相似的
                valid_similarities = sim_upper[valid_pairs]
                sorted_indices = torch.argsort(valid_similarities, descending=True)
                
                batch_pairs = []
                for idx in sorted_indices:
                    i, j = valid_indices[idx].tolist()
                    batch_pairs.append((i, j))
                
                merge_pairs.append(batch_pairs)
            else:
                merge_pairs.append([])
                
        return merge_pairs
    
    def merge_tokens(self, tokens: torch.Tensor, merge_pairs: list) -> torch.Tensor:
        """
        执行token合并
        
        Args:
            tokens: 输入token [B, N, D]
            merge_pairs: 合并对列表
            
        Returns:
            合并后的token [B, N', D]
        """
        merged_tokens = []
        
        for b, pairs in enumerate(merge_pairs):
            batch_tokens = tokens[b]  # [N, D]
            
            if not pairs:
                # 没有需要合并的对
                merged_tokens.append(batch_tokens)
                continue
                
            # 记录哪些token已被合并
            merged_indices = set()
            new_tokens = []
            
            # 执行合并
            for i, j in pairs:
                if i not in merged_indices and j not in merged_indices:
                    # 加权平均合并 (可以根据相似性调整权重)
                    merged_token = (batch_tokens[i] + batch_tokens[j]) / 2
                    new_tokens.append(merged_token)
                    merged_indices.update([i, j])
            
            # 添加未合并的token
            for idx in range(len(batch_tokens)):
                if idx not in merged_indices:
                    new_tokens.append(batch_tokens[idx])
            
            if new_tokens:
                merged_batch = torch.stack(new_tokens)
                merged_tokens.append(merged_batch)
            else:
                merged_tokens.append(batch_tokens)
        
        # 处理不同长度的序列，使用padding
        max_len = max(t.shape[0] for t in merged_tokens)
        padded_tokens = []
        
        for t in merged_tokens:
            if t.shape[0] < max_len:
                padding = torch.zeros(max_len - t.shape[0], t.shape[1], 
                                    device=t.device, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            padded_tokens.append(t)
        
        return torch.stack(padded_tokens)
    
    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        前向传播
        
        Args:
            tokens: 输入token [B, N, D]
            
        Returns:
            merged_tokens: 合并后的token
            info: 合并信息
        """
        # 计算相似性矩阵
        similarity_matrix = self.compute_pairwise_similarity(tokens)
        
        # 找到合并对
        merge_pairs = self.find_merge_pairs(similarity_matrix)
        
        # 执行合并
        merged_tokens = self.merge_tokens(tokens, merge_pairs)
        
        # 统计信息
        original_count = tokens.shape[1]
        merged_count = merged_tokens.shape[1]
        reduction_ratio = (original_count - merged_count) / original_count
        
        info = {
            'similarity_matrix': similarity_matrix,
            'merge_pairs': merge_pairs,
            'original_token_count': original_count,
            'merged_token_count': merged_count,
            'reduction_ratio': reduction_ratio,
            'h_matrix_reg_loss': self.H_matrix.get_regularization_loss(original_count)
        }
        
        return merged_tokens, info


class SATDLoss(nn.Module):
    """
    SATD训练损失函数
    """
    
    def __init__(self, 
                 lambda_diversity: float = 0.1,
                 lambda_structure: float = 0.1,
                 lambda_h_reg: float = 0.01):
        """
        Args:
            lambda_diversity: 多样性损失权重
            lambda_structure: 结构损失权重
            lambda_h_reg: H矩阵正则化权重
        """
        super().__init__()
        self.lambda_diversity = lambda_diversity
        self.lambda_structure = lambda_structure
        self.lambda_h_reg = lambda_h_reg
        
    def similarity_loss(self, similarity_matrix: torch.Tensor, target_similarities: torch.Tensor) -> torch.Tensor:
        """
        相似性损失：鼓励相似token具有高相似度
        """
        return F.mse_loss(similarity_matrix, target_similarities)
    
    def diversity_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        多样性损失：防止所有token变得过于相似
        """
        # 计算token之间的平均相似性
        mean_token = torch.mean(tokens, dim=1, keepdim=True)  # [B, 1, D]
        diversity = torch.mean(torch.norm(tokens - mean_token, p=2, dim=-1))
        return -diversity  # 负号因为我们想最大化多样性
    
    def structure_loss(self, original_tokens: torch.Tensor, merged_tokens: torch.Tensor) -> torch.Tensor:
        """
        结构损失：保持原始token的全局结构
        """
        # 计算原始和合并后token的全局表示
        original_global = torch.mean(original_tokens, dim=1)  # [B, D]
        merged_global = torch.mean(merged_tokens, dim=1)      # [B, D]
        
        return F.mse_loss(original_global, merged_global)
    
    def forward(self, 
                similarity_matrix: torch.Tensor,
                target_similarities: torch.Tensor,
                original_tokens: torch.Tensor,
                merged_tokens: torch.Tensor,
                h_reg_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        """
        # 各项损失
        sim_loss = self.similarity_loss(similarity_matrix, target_similarities)
        div_loss = self.diversity_loss(merged_tokens)
        struct_loss = self.structure_loss(original_tokens, merged_tokens)
        
        # 总损失
        total_loss = (sim_loss + 
                     self.lambda_diversity * div_loss + 
                     self.lambda_structure * struct_loss +
                     self.lambda_h_reg * h_reg_loss)
        
        return {
            'total_loss': total_loss,
            'similarity_loss': sim_loss,
            'diversity_loss': div_loss,
            'structure_loss': struct_loss,
            'h_reg_loss': h_reg_loss
        }