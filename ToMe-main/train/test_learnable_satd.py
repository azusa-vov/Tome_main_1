"""
可学习SATD使用示例

这个脚本展示如何使用训练好的可学习SATD矩阵进行推理。
"""

import os
import sys
import argparse
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config.experiment_config import Config, quick_imagenet_config, full_imagenet_config
from experiments.run_experiment import run_experiment


def create_learnable_satd_config(checkpoint_path: str, gpu_id: int = None):
    """创建包含可学习SATD的配置"""
    config = quick_imagenet_config(gpu_id=gpu_id)
    
    # 添加可学习SATD策略
    if os.path.exists(checkpoint_path):
        config.add_learnable_satd_strategy(checkpoint_path)
        print(f"Added LearnableSATD strategy with checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint file not found: {checkpoint_path}")
    
    return config


def compare_strategies_with_learnable(checkpoint_path: str, gpu_id: int = None):
    """比较可学习SATD与其他策略的性能"""
    config = create_learnable_satd_config(checkpoint_path, gpu_id)
    
    print("Running experiment with learnable SATD...")
    print(f"Strategies to test: {[name for name, _ in config.strategies]}")
    print(f"R values: {config.r_values}")
    
    # 运行实验
    results = run_experiment(config)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test Learnable SATD Matrix')
    parser.add_argument('--checkpoint', required=True, help='Path to learned H matrix checkpoint')
    parser.add_argument('--gpu-id', type=int, default=None, help='GPU ID to use')
    parser.add_argument('--full-test', action='store_true', help='Run full ImageNet test instead of quick test')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if args.full_test:
        print("Running full ImageNet test...")
        config = full_imagenet_config(gpu_id=args.gpu_id)
        config.add_learnable_satd_strategy(args.checkpoint)
    else:
        print("Running quick test...")
        config = create_learnable_satd_config(args.checkpoint, args.gpu_id)
    
    # 运行实验
    results = run_experiment(config)
    
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    
    # 分析结果，特别关注可学习SATD的性能
    if 'LearnableSATD' in [name for name, _ in config.strategies]:
        print("\nLearnable SATD Performance:")
        for r in config.r_values:
            if r in results and 'LearnableSATD' in results[r]:
                acc = results[r]['LearnableSATD']['accuracy']
                time_cost = results[r]['LearnableSATD']['inference_time']
                print(f"  r={r}: Accuracy={acc:.2f}%, Time={time_cost:.3f}s")
    
    print(f"\nResults saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
