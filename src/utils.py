"""
SATD实用工具函数
提供配置管理、可视化、评估等辅助功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from typing import Dict, List, Tuple, Any, Optional
import time
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


class MetricsTracker:
    """
    训练指标跟踪器
    """
    
    def __init__(self):
        self.metrics = {}
        self.step_count = 0
        
    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.metrics[key].append(value)
        
        self.step_count += 1
    
    def get_latest(self, key: str) -> float:
        """获取最新的指标值"""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return 0.0
    
    def get_average(self, key: str, last_n: int = None) -> float:
        """获取平均值"""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def plot_metrics(self, save_path: str = None, figsize: Tuple[int, int] = (15, 10)):
        """绘制所有指标"""
        num_metrics = len(self.metrics)
        if num_metrics == 0:
            return
        
        # 计算子图布局
        rows = int(np.ceil(np.sqrt(num_metrics)))
        cols = int(np.ceil(num_metrics / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_metrics == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        
        for i, (key, values) in enumerate(self.metrics.items()):
            if i < len(axes):
                axes[i].plot(values)
                axes[i].set_title(key.replace('_', ' ').title())
                axes[i].grid(True)
        
        # 隐藏多余的子图
        for i in range(num_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"指标图表已保存到: {save_path}")
        else:
            plt.show()
    
    def save_metrics(self, save_path: str):
        """保存指标到文件"""
        np.savez(save_path, **self.metrics)
        print(f"指标数据已保存到: {save_path}")
    
    def load_metrics(self, load_path: str):
        """从文件加载指标"""
        data = np.load(load_path)
        self.metrics = {key: list(values) for key, values in data.items()}
        self.step_count = len(next(iter(self.metrics.values())))


def visualize_similarity_matrix(similarity_matrix: torch.Tensor, 
                               save_path: str = None,
                               title: str = "Similarity Matrix",
                               batch_idx: int = 0):
    """
    可视化相似性矩阵
    
    Args:
        similarity_matrix: 相似性矩阵 [B, N, N]
        save_path: 保存路径
        title: 图表标题
        batch_idx: 要可视化的batch索引
    """
    if similarity_matrix.dim() == 3:
        sim_mat = similarity_matrix[batch_idx].detach().cpu().numpy()
    else:
        sim_mat = similarity_matrix.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_mat, annot=False, cmap='viridis', square=True)
    plt.title(title)
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"相似性矩阵已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_h_matrix(h_matrix: torch.Tensor, 
                      save_path: str = None,
                      title: str = "H Matrix"):
    """
    可视化H矩阵
    
    Args:
        h_matrix: H矩阵 [N, N]
        save_path: 保存路径
        title: 图表标题
    """
    h_mat = h_matrix.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    # 使用diverging colormap，以0为中心
    vmax = np.abs(h_mat).max()
    sns.heatmap(h_mat, annot=False, cmap='RdBu_r', 
                vmin=-vmax, vmax=vmax, square=True, center=0)
    
    plt.title(title)
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"H矩阵已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_token_reduction(original_counts: List[int], 
                            merged_counts: List[int],
                            save_path: str = None):
    """
    可视化token减少情况
    
    Args:
        original_counts: 原始token数量列表
        merged_counts: 合并后token数量列表
        save_path: 保存路径
    """
    reduction_ratios = [(orig - merged) / orig for orig, merged in zip(original_counts, merged_counts)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Token数量对比
    x = range(len(original_counts))
    ax1.plot(x, original_counts, label='Original', marker='o')
    ax1.plot(x, merged_counts, label='Merged', marker='s')
    ax1.set_xlabel('Batch/Step')
    ax1.set_ylabel('Token Count')
    ax1.set_title('Token Count Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # 减少比例
    ax2.plot(x, reduction_ratios, label='Reduction Ratio', marker='d', color='red')
    ax2.set_xlabel('Batch/Step')
    ax2.set_ylabel('Reduction Ratio')
    ax2.set_title('Token Reduction Ratio')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Token减少可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


class PerformanceProfiler:
    """
    性能分析器
    """
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        
    def start_timer(self, name: str):
        """开始计时"""
        self.timings[name] = {'start': time.time()}
    
    def end_timer(self, name: str):
        """结束计时"""
        if name in self.timings and 'start' in self.timings[name]:
            self.timings[name]['duration'] = time.time() - self.timings[name]['start']
            return self.timings[name]['duration']
        return 0.0
    
    def record_memory(self, name: str):
        """记录内存使用"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            
            self.memory_usage[name] = {
                'allocated': memory_allocated,
                'cached': memory_cached
            }
    
    def get_timing_summary(self) -> Dict[str, float]:
        """获取计时总结"""
        summary = {}
        for name, timing in self.timings.items():
            if 'duration' in timing:
                summary[name] = timing['duration']
        return summary
    
    def print_summary(self):
        """打印性能总结"""
        print("=" * 50)
        print("性能分析总结")
        print("=" * 50)
        
        print("\n计时信息:")
        for name, timing in self.timings.items():
            if 'duration' in timing:
                print(f"  {name}: {timing['duration']:.4f}s")
        
        print("\n内存使用:")
        for name, memory in self.memory_usage.items():
            print(f"  {name}:")
            print(f"    分配内存: {memory['allocated']:.2f}GB")
            print(f"    缓存内存: {memory['cached']:.2f}GB")


def create_output_directory(base_path: str, experiment_name: str = None) -> str:
    """
    创建输出目录
    
    Args:
        base_path: 基础路径
        experiment_name: 实验名称
        
    Returns:
        创建的目录路径
    """
    if experiment_name:
        output_dir = os.path.join(base_path, experiment_name)
    else:
        # 使用时间戳作为实验名称
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_path, f"experiment_{timestamp}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def save_model_checkpoint(model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         metrics: Dict[str, float],
                         save_path: str):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        metrics: 指标字典
        save_path: 保存路径
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    print(f"模型检查点已保存到: {save_path}")


def load_model_checkpoint(model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         checkpoint_path: str) -> Tuple[int, Dict[str, float]]:
    """
    加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        
    Returns:
        (epoch, metrics)
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    
    print(f"模型检查点已从 {checkpoint_path} 加载")
    return epoch, metrics


def calculate_model_flops(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    计算模型的FLOPs (粗略估计)
    
    Args:
        model: 模型
        input_shape: 输入形状
        
    Returns:
        FLOPs估计值
    """
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # 这是一个简化的FLOPs估计
    # 实际应用中建议使用专门的工具如thop或ptflops
    num_params = count_parameters(model)
    batch_size = input_shape[0] if len(input_shape) > 0 else 1
    
    # 粗略估计: 每个参数大约需要2个操作 (乘加)
    flops_estimate = num_params * batch_size * 2
    
    return flops_estimate


def compare_models(model_configs: List[Dict[str, Any]], 
                  test_input: torch.Tensor,
                  save_path: str = None) -> Dict[str, Any]:
    """
    比较不同模型配置的性能
    
    Args:
        model_configs: 模型配置列表
        test_input: 测试输入
        save_path: 保存路径
        
    Returns:
        比较结果
    """
    from satd_core import SATDTokenMerger
    
    results = []
    profiler = PerformanceProfiler()
    
    for i, config in enumerate(model_configs):
        model_name = config.get('name', f'Model_{i}')
        
        # 创建模型
        merger = SATDTokenMerger(**config['satd_config'])
        
        # 性能测试
        profiler.start_timer(model_name)
        profiler.record_memory(f"{model_name}_before")
        
        with torch.no_grad():
            merged_tokens, info = merger(test_input)
        
        profiler.end_timer(model_name)
        profiler.record_memory(f"{model_name}_after")
        
        # 收集结果
        result = {
            'name': model_name,
            'config': config,
            'inference_time': profiler.timings[model_name]['duration'],
            'token_reduction_ratio': info['reduction_ratio'],
            'merged_token_count': info['merged_token_count'],
            'original_token_count': info['original_token_count']
        }
        
        results.append(result)
    
    # 生成比较报告
    comparison_report = {
        'results': results,
        'test_input_shape': list(test_input.shape),
        'profiler_summary': profiler.get_timing_summary()
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            yaml.dump(comparison_report, f, default_flow_style=False)
        print(f"模型比较报告已保存到: {save_path}")
    
    return comparison_report