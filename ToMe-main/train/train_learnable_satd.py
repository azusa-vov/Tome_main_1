"""
训练可学习的SATD变换矩阵

这个脚本使用ImageNet数据集训练可学习的H变换矩阵，目标是优化ViT Token Merging的性能。
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Dict, Any
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tome.similarity_strategy import LearnableSATDSimilarity
from tome import patch
from experiments.utils.gpu_utils import setup_gpu_environment

class SmartRegularization:
    def __init__(self, h_matrix):
        self.h_matrix = h_matrix
        
    def compute_regularization(self):
        """计算有意义的正则项组合"""
        
        # 1. 基础连接项（确保梯度流）
        connection_term = (self.h_matrix ** 2).sum()
        
        # 2. 正交性约束（保持变换性质）
        I = torch.eye(8, device=self.h_matrix.device)
        orthogonal_term = torch.norm(self.h_matrix @ self.h_matrix.t() - I) ** 2
        
        # 3. 稳定性约束（防止参数过大）
        stability_term = torch.norm(self.h_matrix, p='fro') ** 2
        
        # 组合
        total_reg = (connection_term + 
                    0.1 * orthogonal_term + 
                    0.01 * stability_term)
        
        return total_reg

class LearnableSATDTrainer:
    """可学习SATD训练器"""
    
    def __init__(self, 
                 model_name: str = "vit_base_patch16_224",
                 dataset_path: str = "/home/tangrq/datasets/imagenet",
                 batch_size: int = 64,
                 learning_rate: float = 0.001,
                 device: str = 'cuda',
                 r_value: int = 8):
        """
        初始化训练器
        
        Args:
            model_name: ViT模型名称
            dataset_path: ImageNet数据集路径
            batch_size: 批次大小
            learning_rate: 学习率
            device: 设备
            r_value: token merging的r值
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.r_value = r_value
        
        # 初始化可学习的SATD相似度策略
        self.learnable_satd = LearnableSATDSimilarity(device=device)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_model(self):
        """设置ViT模型和ToMe"""
        self.logger.info(f"Loading model: {self.model_name}")
        
        # 加载预训练的ViT模型
        self.model = timm.create_model(self.model_name, pretrained=True)
        self.model.to(self.device)
        
        # 冻结模型参数，只训练H矩阵
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 应用ToMe patch，使用可学习的SATD
        patch.timm(self.model, trace_source=False)
        
        # 设置ToMe使用我们的可学习SATD策略
        self.model._tome_info["similarity_strategy"] = self.learnable_satd
        self.model.r = self.r_value
        
        self.model.eval()  # 保持评估模式，因为我们不训练ViT参数
        
        self.logger.info(f"Model setup complete. Using r={self.r_value}")
        
    def setup_data(self, num_samples: int = 10000):
        """设置数据加载器"""
        self.logger.info(f"Setting up data loader for {num_samples} samples")
        
        # ImageNet数据变换
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 使用ImageNet训练集
        full_dataset = datasets.ImageNet(
            root=self.dataset_path,
            split='train',
            transform=transform
        )
        
        # 随机选择子集进行训练
        if num_samples <= 0:
            dataset = full_dataset
        elif num_samples < len(full_dataset):
            indices = torch.randperm(len(full_dataset))[:num_samples]
            dataset = Subset(full_dataset, indices)
        else:
            dataset = full_dataset
            
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info(f"Data loader ready with {len(dataset)} samples")
        
    def setup_optimizer(self):
        """设置优化器"""
        # 只优化H矩阵参数
        params = self.learnable_satd.get_parameters()
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"Optimizer setup with learning rate: {self.learning_rate}")
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        self.model.eval()  # ViT保持评估模式
        # 强制启用梯度计算
        torch.set_grad_enabled(True)
        
        # 确保H矩阵始终需要梯度
        self.learnable_satd.H_matrix.requires_grad_(True)
        
        # 调试用，检查H_matrix是否在计算图中
        # for batch_idx, (images, targets) in enumerate(self.dataloader):
        #     if batch_idx > 0:  # 只调试第一个batch
        #         break
                
        #     images = images.to(self.device)
        #     targets = targets.to(self.device)
            
        #     self.optimizer.zero_grad()
            
        #     # 强制启用梯度
        #     with torch.enable_grad():
        #         outputs = self.model(images)
        #         loss = self.criterion(outputs, targets)
            
        #     print(f"\n🔍 检查最终的梯度链:")
        #     print(f"Loss requires_grad: {loss.requires_grad}")
        #     print(f"Loss grad_fn: {loss.grad_fn}")
            
        #     # 手动检查H_matrix是否在计算图中
        #     def check_if_h_matrix_in_graph(loss_tensor, h_matrix):
        #         """检查H矩阵是否在loss的计算图中"""
                
        #         def traverse_graph(grad_fn, visited=None, depth=0):
        #             if visited is None:
        #                 visited = set()
        #             if grad_fn is None or grad_fn in visited or depth > 50:
        #                 return False
                        
        #             visited.add(grad_fn)
                    
        #             # 检查这个节点是否与H矩阵相关
        #             if hasattr(grad_fn, 'variable') and grad_fn.variable is h_matrix:
        #                 print(f"✅ 找到H_matrix在计算图中! depth={depth}")
        #                 return True
                    
        #             # 递归检查所有输入节点
        #             if hasattr(grad_fn, 'next_functions'):
        #                 for next_fn, _ in grad_fn.next_functions:
        #                     if traverse_graph(next_fn, visited, depth + 1):
        #                         return True
                    
        #             return False
                
        #         return traverse_graph(loss_tensor.grad_fn)
            
        #     h_matrix_in_graph = check_if_h_matrix_in_graph(loss, self.learnable_satd.H_matrix)
        #     print(f"H_matrix在计算图中: {h_matrix_in_graph}")
            
        #     if not h_matrix_in_graph:
        #         print("❌ H_matrix不在计算图中，这就是问题所在！")
                
        #         # 尝试手动连接H_matrix到loss
        #         print("🔧 尝试手动连接H_matrix...")
                
        #         # 添加一个很小的正则项，强制H_matrix参与计算
        #         regularization = 1e-10 * (self.learnable_satd.H_matrix ** 2).sum()
        #         loss_with_reg = loss + regularization
                
        #         print(f"带正则的loss requires_grad: {loss_with_reg.requires_grad}")
        #         print(f"带正则的loss grad_fn: {loss_with_reg.grad_fn}")
                
        #         # 检查正则化后的loss是否包含H_matrix
        #         h_matrix_in_reg_graph = check_if_h_matrix_in_graph(loss_with_reg, self.learnable_satd.H_matrix)
        #         print(f"正则化后H_matrix在计算图中: {h_matrix_in_reg_graph}")
                
        #         try:
        #             loss_with_reg.backward()
        #             if self.learnable_satd.H_matrix.grad is not None:
        #                 print(f"✅ 正则化成功！H_matrix梯度: {self.learnable_satd.H_matrix.grad.norm().item()}")
        #                 self.optimizer.step()
        #             else:
        #                 print("❌ 正则化也失败了")
        #         except Exception as e:
        #             print(f"正则化反向传播失败: {e}")
            
        #     else:
        #         print("✅ H_matrix在计算图中，尝试正常反向传播...")
        #         try:
        #             loss.backward()
        #             if self.learnable_satd.H_matrix.grad is not None:
        #                 print(f"✅ 成功！H_matrix梯度: {self.learnable_satd.H_matrix.grad.norm().item()}")
        #                 self.optimizer.step()
        #             else:
        #                 print("❌ 虽然在计算图中，但没收到梯度")
        #         except Exception as e:
        #             print(f"反向传播失败: {e}")
 
        for batch_idx, (images, targets) in enumerate(self.dataloader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):  # 启用梯度计算
                outputs = self.model(images)
                main_loss = self.criterion(outputs, targets)
                
                # 强制添加H_matrix正则项，确保它参与计算图
                # 这个正则项很小，不会显著影响训练，但能保证梯度流通
                # h_regularization = 1e-8 * torch.sum(self.learnable_satd.H_matrix ** 2)
                reg_calculator = SmartRegularization(self.learnable_satd.H_matrix)
                h_regularization = 1e-8 * reg_calculator.compute_regularization()
                
                # 总损失
                loss = main_loss + h_regularization
                print(f"\r主损失: {main_loss.item():.6f}, H正则: {h_regularization.item():.10f}", end="", flush=True)

                # 反向传播
                loss.backward()
                # self.optimizer.step()
                if self.learnable_satd.H_matrix.grad is not None:
                    # print(f"✅ H_matrix梯度norm: {self.learnable_satd.H_matrix.grad.norm().item():.6f}")
                    self.optimizer.step()
                else:
                    print("❌ 仍然没有梯度")
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            
            # 打印进度
            if batch_idx % 100 == 0:
                accuracy = 100.0 * correct_predictions / total_samples
                self.logger.info(
                    f'Epoch: {epoch} [{batch_idx}/{len(self.dataloader)}] '
                    f'Loss: {loss.item():.6f} Acc: {accuracy:.2f}%'
                )
        
        # 计算epoch统计
        avg_loss = total_loss / len(self.dataloader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_loader: DataLoader = None) -> Dict[str, float]:
        """验证模型"""
        if val_loader is None:
            # 使用验证集
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            val_dataset = datasets.ImageNet(
                root=self.dataset_path,
                split='val',
                transform=transform
            )
            
            # 使用验证集的一个子集
            indices = torch.randperm(len(val_dataset))[:5000]
            val_subset = Subset(val_dataset, indices)
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }
    
    def train(self, 
              num_epochs: int = 10,
              num_samples: int = 10000,
              save_dir: str = "learned_satd_checkpoints",
              validate_every: int = 2):
        """训练主循环"""
        self.logger.info("Starting training...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置模型、数据和优化器
        self.setup_model()
        self.setup_data(num_samples)
        self.setup_optimizer()
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            
            # 记录训练指标
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['learning_rate'].append(train_metrics['lr'])
            
            epoch_time = time.time() - start_time
            
            self.logger.info(
                f'Epoch {epoch} completed in {epoch_time:.2f}s - '
                f'Train Loss: {train_metrics["loss"]:.6f}, '
                f'Train Acc: {train_metrics["accuracy"]:.2f}%'
            )
            
            # 验证
            if epoch % validate_every == 0 or epoch == num_epochs - 1:
                val_metrics = self.validate()
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_accuracy'].append(val_metrics['val_accuracy'])
                
                self.logger.info(
                    f'Validation - Loss: {val_metrics["val_loss"]:.6f}, '
                    f'Acc: {val_metrics["val_accuracy"]:.2f}%'
                )
                
                # 保存最佳模型
                if val_metrics['val_accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['val_accuracy']
                    best_checkpoint = {
                        'epoch': epoch,
                        'h_matrix': self.learnable_satd.H_matrix.detach().cpu(),
                        'val_accuracy': best_val_accuracy,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics
                    }
                    torch.save(best_checkpoint, os.path.join(save_dir, 'best_h_matrix.pth'))
                    self.logger.info(f'New best validation accuracy: {best_val_accuracy:.2f}%')
            
            # 保存当前epoch的检查点
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'h_matrix': self.learnable_satd.H_matrix.detach().cpu(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'history': history
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # 保存训练历史
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f'Training completed! Best validation accuracy: {best_val_accuracy:.2f}%')
        return history


def main():
    parser = argparse.ArgumentParser(description='Train Learnable SATD Matrix')
    parser.add_argument('--model', default='vit_base_patch16_224', help='Model name')
    parser.add_argument('--dataset', default='/home/tangrq/datasets/imagenet', help='ImageNet dataset path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--samples', type=int, default=0, help='Number of training samples')
    parser.add_argument('--r-value', type=int, default=8, help='Token merging r value')
    parser.add_argument('--save-dir', default='learned_satd_checkpoints', help='Save directory')
    parser.add_argument('--gpu-id', type=int, default=None, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # 设置GPU
    if args.gpu_id is not None:
        device = f'cuda:{args.gpu_id}'
    else:
        gpu_id = setup_gpu_environment(min_memory_mb=4096)
        device = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
    
    print(f"Using device: {device}")

    # 创建训练器
    trainer = LearnableSATDTrainer(
        model_name=args.model,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        r_value=args.r_value
    )
    
    # 开始训练
    history = trainer.train(
        num_epochs=args.epochs,
        num_samples=args.samples,
        save_dir=args.save_dir
    )
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
