"""
可学习SATD完整工作流

这个脚本提供了完整的训练和测试可学习SATD矩阵的工作流程。
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_learnable_satd(args):
    """训练可学习SATD矩阵"""
    print("="*60)
    print("TRAINING LEARNABLE SATD MATRIX")
    print("="*60)
    
    train_script = os.path.join(os.path.dirname(__file__), '..', 'train', 'train_learnable_satd.py')
    
    cmd = [
        'python', train_script,
        '--model', args.model,
        '--dataset', args.dataset,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--samples', str(args.samples),
        '--r-value', str(args.r_value),
        '--save-dir', args.save_dir
    ]
    
    if args.gpu_id is not None:
        cmd.extend(['--gpu-id', str(args.gpu_id)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False


def test_learnable_satd(args, checkpoint_path):
    """测试训练好的可学习SATD矩阵"""
    print("="*60)
    print("TESTING LEARNABLE SATD MATRIX")
    print("="*60)
    
    test_script = os.path.join(os.path.dirname(__file__), 'test_learnable_satd.py')
    
    cmd = [
        'python', test_script,
        '--checkpoint', checkpoint_path
    ]
    
    if args.gpu_id is not None:
        cmd.extend(['--gpu-id', str(args.gpu_id)])
    
    if args.full_test:
        cmd.append('--full-test')
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Testing completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Testing failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False


def analyze_results(save_dir):
    """分析训练结果"""
    print("="*60)
    print("ANALYZING TRAINING RESULTS")
    print("="*60)
    
    # 检查训练历史
    history_file = os.path.join(save_dir, 'training_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        print("Training History Summary:")
        print(f"  Final Train Accuracy: {history['train_accuracy'][-1]:.2f}%")
        if history['val_accuracy']:
            print(f"  Final Val Accuracy: {history['val_accuracy'][-1]:.2f}%")
            print(f"  Best Val Accuracy: {max(history['val_accuracy']):.2f}%")
        
        print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
        if history['val_loss']:
            print(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
    
    # 检查最佳检查点
    best_checkpoint = os.path.join(save_dir, 'best_h_matrix.pth')
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        print(f"\nBest Checkpoint (Epoch {checkpoint['epoch']}):")
        print(f"  Validation Accuracy: {checkpoint['val_accuracy']:.2f}%")
        
        # 分析H矩阵
        import torch
        h_matrix = checkpoint['h_matrix']
        print(f"  H Matrix Stats:")
        print(f"    Shape: {h_matrix.shape}")
        print(f"    Mean: {h_matrix.mean().item():.4f}")
        print(f"    Std: {h_matrix.std().item():.4f}")
        print(f"    Min: {h_matrix.min().item():.4f}")
        print(f"    Max: {h_matrix.max().item():.4f}")
        
        return best_checkpoint
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Learnable SATD Complete Workflow')
    
    # 通用参数
    parser.add_argument('--action', choices=['train', 'test', 'both'], default='train',
                       help='Action to perform')
    parser.add_argument('--gpu-id', type=int, default=None, help='GPU ID to use')
    
    # 训练参数
    parser.add_argument('--model', default='vit_base_patch16_224', help='Model name')
    parser.add_argument('--dataset', default='/home/tangrq/datasets/imagenet', help='ImageNet dataset path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--r-value', type=int, default=8, help='Token merging r value')
    parser.add_argument('--save-dir', default='learned_satd_checkpoints', help='Save directory')
    
    # 测试参数
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path for testing')
    parser.add_argument('--full-test', action='store_true', help='Run full ImageNet test')
    
    args = parser.parse_args()
    
    success = True
    best_checkpoint = None
    
    if args.action in ['train', 'both']:
        # 训练
        success = train_learnable_satd(args)
        
        if success:
            # 分析训练结果
            best_checkpoint = analyze_results(args.save_dir)
        else:
            print("Training failed, skipping testing.")
            return
    
    if args.action in ['test', 'both']:
        # 确定要使用的检查点
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        elif best_checkpoint:
            checkpoint_path = best_checkpoint
        else:
            # 查找默认的最佳检查点
            default_checkpoint = os.path.join(args.save_dir, 'best_h_matrix.pth')
            if os.path.exists(default_checkpoint):
                checkpoint_path = default_checkpoint
            else:
                print("No checkpoint found for testing!")
                return
        
        # 测试
        success = test_learnable_satd(args, checkpoint_path)
    
    if success:
        print("\n" + "="*60)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if args.action in ['train', 'both']:
            print(f"Training results saved in: {args.save_dir}")
        
        if best_checkpoint:
            print(f"Best H matrix saved as: {best_checkpoint}")
            print(f"Use this checkpoint for inference with: --checkpoint {best_checkpoint}")
    else:
        print("\nWorkflow failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
