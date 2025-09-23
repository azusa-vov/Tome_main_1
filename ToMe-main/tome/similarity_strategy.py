from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimilarityStrategy(ABC):
    """相似度计算策略的抽象基类"""
    
    @abstractmethod
    def compute_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        计算两组tokens之间的相似度
        Args:
            a: 形状 (batch, tokens_a, channels)
            b: 形状 (batch, tokens_b, channels) 
        Returns:
            scores: 形状 (batch, tokens_a, tokens_b)
        """
        pass
    
    def preprocess(self, metric: torch.Tensor) -> torch.Tensor:
        """预处理，默认不做任何处理"""
        return metric

class CosineSimilarity(SimilarityStrategy):
    """余弦相似度策略"""
    
    def preprocess(self, metric: torch.Tensor) -> torch.Tensor:
        # L2归一化
        return metric / metric.norm(dim=-1, keepdim=True)
    
    def compute_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 已经归一化，直接点积即为余弦相似度
        return a @ b.transpose(-1, -2)

class EuclideanSimilarity(SimilarityStrategy):
    """基于欧氏距离的相似度（距离越小相似度越高）"""
    
    def compute_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 计算所有pairs的欧氏距离
        # a: (B, N, D), b: (B, M, D)
        a_expanded = a.unsqueeze(-2)  # (B, N, 1, D)
        b_expanded = b.unsqueeze(-3)  # (B, 1, M, D)
        
        # 欧氏距离
        distances = torch.norm(a_expanded - b_expanded, dim=-1)  # (B, N, M)
        
        # 转换为相似度（距离越小相似度越高）
        return -distances

class DotProductSimilarity(SimilarityStrategy):
    """简单点积相似度"""
    
    def compute_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b.transpose(-1, -2)

class SADSimilarity(SimilarityStrategy):
    """基于绝对差的相似度（差值越小相似度越高）"""
    
    def compute_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 计算所有pairs的绝对差
        a_expanded = a.unsqueeze(-2)  # (B, N, 1, D)
        b_expanded = b.unsqueeze(-3)  # (B, 1, M, D)
        
        # 绝对差
        abs_diff = torch.abs(a_expanded - b_expanded)  # (B, N, M, D)
        
        # 求和得到距离
        distances = abs_diff.sum(dim=-1)  # (B, N, M)
        
        # 转换为相似度（差值越小相似度越高）
        return -distances
    
class SATDSimilarity(SimilarityStrategy):
    """基于简单的8x8 SATD的相似度（差值越小相似度越高）"""
    
    def __init__(self):
        # 预定义8点Hadamard变换矩阵
        self.hadamard_8 = torch.tensor([
            [ 1,  1,  1,  1,  1,  1,  1,  1],
            [ 1, -1,  1, -1,  1, -1,  1, -1],  
            [ 1,  1, -1, -1,  1,  1, -1, -1],
            [ 1, -1, -1,  1,  1, -1, -1,  1],
            [ 1,  1,  1,  1, -1, -1, -1, -1],
            [ 1, -1,  1, -1, -1,  1, -1,  1],
            [ 1,  1, -1, -1, -1, -1,  1,  1], 
            [ 1, -1, -1,  1, -1,  1,  1, -1]
        ], dtype=torch.float32)
        
    def hadamard_2d_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        对8x8矩阵执行2D Hadamard变换
        x: (..., 8, 8)
        返回: (..., 8, 8) 变换后的矩阵
        """
        device = x.device
        H = self.hadamard_8.to(device)
        
        # 2D Hadamard变换: Y = H * X * H^T
        # 先对行变换，再对列变换
        temp = torch.matmul(H, x)  # 对行变换
        result = torch.matmul(temp, H.t())  # 对列变换
        
        return result
    
    def compute_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        计算Simple SATD相似度
        a, b: (B, N, 64) 或 (B, M, 64)
        """
        # 检查维度
        assert a.shape[-1] == 64, f"Expected dim=64, got {a.shape[-1]}"
        assert b.shape[-1] == 64, f"Expected dim=64, got {b.shape[-1]}"
        
        # 重塑为8x8矩阵
        *batch_dims_a, channels_a = a.shape
        *batch_dims_b, channels_b = b.shape
        
        a_8x8 = a.view(*batch_dims_a, 8, 8)  # (..., 8, 8)
        b_8x8 = b.view(*batch_dims_b, 8, 8)  # (..., 8, 8)
        
        # 应用2D Hadamard变换
        a_transformed = self.hadamard_2d_transform(a_8x8)  # (..., 8, 8)
        b_transformed = self.hadamard_2d_transform(b_8x8)  # (..., 8, 8)
        
        # 计算所有pairs的SATD
        # a_transformed: (B, N, 8, 8), b_transformed: (B, M, 8, 8)
        a_expanded = a_transformed.unsqueeze(-3)  # (B, N, 1, 8, 8)
        b_expanded = b_transformed.unsqueeze(-4)  # (B, 1, M, 8, 8)
        
        # 计算变换域的绝对差
        abs_diff = torch.abs(a_expanded - b_expanded)  # (B, N, M, 8, 8)
        
        # 求和得到SATD值
        satd_distances = abs_diff.sum(dim=(-2, -1))  # (B, N, M)
        
        # 转换为相似度（距离越小相似度越高）
        return -satd_distances

class LearnableSATDSimilarity(nn.Module):
    """可学习的SATD相似度，使用可训练的H变换矩阵"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        # 初始化可学习的8x8变换矩阵，从标准Hadamard矩阵开始
        hadamard_8 = torch.tensor([
            [ 1,  1,  1,  1,  1,  1,  1,  1],
            [ 1, -1,  1, -1,  1, -1,  1, -1],  
            [ 1,  1, -1, -1,  1,  1, -1, -1],
            [ 1, -1, -1,  1,  1, -1, -1,  1],
            [ 1,  1,  1,  1, -1, -1, -1, -1],
            [ 1, -1,  1, -1, -1,  1, -1,  1],
            [ 1,  1, -1, -1, -1, -1,  1,  1], 
            [ 1, -1, -1,  1, -1,  1,  1, -1]
        ], dtype=torch.float32, device=device)
        
        # 创建可学习参数
        self.H_matrix = torch.nn.Parameter(hadamard_8.clone())
        self.device = device
        
    def get_parameters(self):
        """返回可学习参数，用于优化器"""
        return [self.H_matrix]
        
    def save_matrix(self, path: str):
        """保存学习到的H矩阵"""
        torch.save(self.H_matrix.detach().cpu(), path)
        
    def load_matrix(self, path: str):
        """加载预训练的H矩阵"""
        self.H_matrix.data = torch.load(path, map_location=self.device)
        
    def hadamard_2d_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用可学习的H矩阵进行2D变换
        x: (..., 8, 8)
        返回: (..., 8, 8) 变换后的矩阵
        """
        # print(f"🔧 hadamard_2d_transform被调用")
        # print(f"  - x.requires_grad: {x.requires_grad}")
        # print(f"  - H_matrix.requires_grad: {self.H_matrix.requires_grad}")
        # print(f"  - torch.is_grad_enabled(): {torch.is_grad_enabled()}")
        # 确保H矩阵在正确的设备上
        H = self.H_matrix.to(x.device)
        
        # 2D变换: Y = H * X * H^T
        temp = torch.matmul(H, x)  # 对行变换
        result = torch.matmul(temp, H.t())  # 对列变换
        # print(f"  - 变换结果: result.requires_grad: {result.requires_grad}")
        # print(f"  - 变换结果: result.grad_fn: {result.grad_fn}")
        return result
    
    def preprocess(self, metric: torch.Tensor) -> torch.Tensor:
        """预处理，默认不做任何处理"""
        return metric
    
    def compute_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        计算基于可学习H矩阵的SATD相似度
        a, b: (B, N, 64) 或 (B, M, 64)
        强制启用梯度计算
        """
        # 强制启用梯度计算，无论外部上下文如何
        with torch.enable_grad():
            # 确保输入tensor需要梯度（如果它们原本需要的话）
            if a.requires_grad or any(p.requires_grad for p in [self.H_matrix]):
                a = a.requires_grad_(True)
            if b.requires_grad or any(p.requires_grad for p in [self.H_matrix]):
                b = b.requires_grad_(True)
                
            # print(f"🔍 强制启用梯度后:")
            # print(f"  - torch.is_grad_enabled(): {torch.is_grad_enabled()}")
            # print(f"  - a.requires_grad: {a.requires_grad}")
            # print(f"  - H_matrix.requires_grad: {self.H_matrix.requires_grad}")

            # 检查维度
            assert a.shape[-1] == 64, f"Expected dim=64, got {a.shape[-1]}"
            assert b.shape[-1] == 64, f"Expected dim=64, got {b.shape[-1]}"
            
            # 重塑为8x8矩阵
            *batch_dims_a, channels_a = a.shape
            *batch_dims_b, channels_b = b.shape
            
            a_8x8 = a.view(*batch_dims_a, 8, 8)  # (..., 8, 8)
            b_8x8 = b.view(*batch_dims_b, 8, 8)  # (..., 8, 8)
            
            # 应用可学习的2D变换
            a_transformed = self.hadamard_2d_transform(a_8x8)  # (..., 8, 8)
            b_transformed = self.hadamard_2d_transform(b_8x8)  # (..., 8, 8)
            
            # print(f"  - 变换后: a_transformed.requires_grad: {a_transformed.requires_grad}")
            # print(f"  - 变换后: a_transformed.grad_fn: {a_transformed.grad_fn}")
            
            # 计算所有pairs的SATD
            # a_transformed: (B, N, 8, 8), b_transformed: (B, M, 8, 8)
            a_expanded = a_transformed.unsqueeze(-3)  # (B, N, 1, 8, 8)
            b_expanded = b_transformed.unsqueeze(-4)  # (B, 1, M, 8, 8)
            
            # 计算变换域的绝对差
            abs_diff = torch.abs(a_expanded - b_expanded)  # (B, N, M, 8, 8)
            
            # 求和得到SATD值
            satd_distances = abs_diff.sum(dim=(-2, -1))  # (B, N, M)
            
            # 转换为相似度（距离越小相似度越高）
            result = -satd_distances
            # print(f"  - 最终结果: result.requires_grad: {result.requires_grad}")
            # print(f"  - 最终结果: result.grad_fn: {result.grad_fn}")
            
            return result

class SignSimilarity(SimilarityStrategy):
    """基于符号相似度（符号相同得分为1，不同为-1）"""
    
    def compute_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 计算符号相似度
        a_sign = torch.sign(a)  # (B, N, D)
        b_sign = torch.sign(b)  # (B, M, D)
        
        # 符号相同得分为1，不同为-1
        similarity = (a_sign.unsqueeze(-2) * b_sign.unsqueeze(-3)).sum(dim=-1)  # (B, N, M)
        
        return similarity
