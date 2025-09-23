import time
import torch
import timm
import tome
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from utils.data_utils import create_imagenet_dataloader
from utils.model_utils import patch_model_with_strategy
from utils.gpu_utils import setup_gpu_environment
from utils.config_utils import serialize_config
import platform
import sys

# Rich imports
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

class SimilarityComparison:
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.gpu_id = None
        self.start_time = None
        
    def setup_gpu(self):
        """设置GPU环境"""
        print("🎮 Setting up GPU environment...")
        
        self.gpu_id = setup_gpu_environment(
            gpu_id=self.config.gpu_id,
            min_memory_mb=self.config.min_memory_mb
        )
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ Using GPU {self.gpu_id}")
        
    def run(self):
        """运行相似度策略比较实验"""
        self.start_time = datetime.now()
        
        print("🚀 Starting Similarity Strategy Comparison on ImageNet")
        print(f"📊 Dataset: {self.config.dataset_path}")
        print(f"📊 Samples: {self.config.num_samples if self.config.num_samples else 'Full validation set (50K)'}")
        print(f"🕐 Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 设置GPU
        self.setup_gpu()
        
        # 准备数据
        split = 'val' if self.config.use_val_set else 'train'
        dataloader = create_imagenet_dataloader(
            self.config.dataset_path,
            batch_size=self.config.batch_size,
            num_samples=self.config.num_samples,
            split=split
        )
        
        print(f"✅ DataLoader created with {len(dataloader.dataset)} samples")

        # 对每个策略进行测试
        for strategy_name, strategy in self.config.strategies:
            print(f"\n📊 Testing {strategy_name} Strategy")
            try:
                strategy_results = self.test_strategy(strategy, dataloader)
                self.results[strategy_name] = strategy_results
                
                # 每个策略完成后清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ GPU OOM for {strategy_name}, trying smaller batch size...")
                    # 可以尝试减小batch size重试
                    self.results[strategy_name] = {"error": "GPU OOM"}
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e
            except Exception as e:
                print(f"❌ Error testing {strategy_name}: {e}")
                self.results[strategy_name] = {"error": str(e)}
        
        # 保存和显示结果
        self.save_results()
        self.print_summary()
        
        return self.results
    
    def test_strategy(self, strategy, dataloader) -> Dict:
        """测试单个策略在所有r值下的表现"""
        strategy_results = {}
        
        for r in self.config.r_values:
            print(f"  🔬 Testing r={r}")
            
            try:
                # 创建模型并应用策略
                model = self.create_model_with_strategy(strategy, r)
                
                # 运行测试
                result = self.evaluate_model(model, dataloader)
                result['r'] = r
                strategy_results[f'r_{r}'] = result
                
                # 清理模型
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"    ❌ Error at r={r}: {e}")
                strategy_results[f'r_{r}'] = {"error": str(e), "r": r}
        
        return strategy_results
    
    def create_model_with_strategy(self, strategy, r):
        """创建配置了特定策略的模型"""
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = timm.create_model(self.config.model_name, pretrained=True)
        model.eval()
        
        # 移动到指定GPU
        if torch.cuda.is_available():
            model = model.cuda(self.gpu_id)
        
        # 应用tome补丁
        tome.patch.timm(model)
        model.r = r
        
        # 设置相似度策略
        patch_model_with_strategy(model, strategy)
        
        return model
    
    def evaluate_model(self, model, dataloader) -> Dict:
        """评估模型性能"""
        try:
            # 准确度测试
            accuracy = self.test_accuracy(model, dataloader)
            # Bypassing accuracy test for faster experiments
            # accuracy = {
            #     'top1_accuracy': 0,
            #     'top5_accuracy': 0,
            #     'total_samples': 0,
            #     'gpu_id': self.gpu_id
            # }

            
            # 效率测试
            sample_batch = next(iter(dataloader))[0]  # 获取一个batch用于效率测试
            efficiency = self.test_efficiency(model, sample_batch)
            
            return {
                'accuracy': accuracy,
                'efficiency': efficiency
            }
        except Exception as e:
            return {
                'error': str(e),
                'accuracy': {'top1_accuracy': 0.0, 'top5_accuracy': 0.0, 'total_samples': 0},
                'efficiency': {'avg_time': 0.0, 'std_time': 0.0, 'throughput': 0.0, 'batch_size': 0}
            }
    
    def test_accuracy(self, model, dataloader, max_batches=None):
        """测试准确度"""
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        # 确定总批次数
        total_batches = min(max_batches or len(dataloader), len(dataloader))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Testing accuracy..."),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TextColumn("[bold green]Top1: {task.fields[top1]:.3f}"),
            TextColumn("[bold cyan]Top5: {task.fields[top5]:.3f}"),
            TimeRemainingColumn(),
            console=console,
            transient=False  # 完成后保留进度条
        ) as progress:
            
            # 创建进度任务
            task = progress.add_task(
                "accuracy", 
                total=total_batches,
                top1=0.0,
                top5=0.0
            )
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(dataloader):
                    if max_batches and i >= max_batches:
                        break
                    
                    try:
                        # 移动到指定GPU
                        if torch.cuda.is_available():
                            images = images.cuda(self.gpu_id, non_blocking=True)
                            labels = labels.cuda(self.gpu_id, non_blocking=True)
                        
                        outputs = model(images)
                        
                        # Top-1 accuracy
                        _, pred_top1 = outputs.topk(1, 1, True, True)
                        correct_top1 += pred_top1.eq(labels.view(-1, 1)).sum().item()
                        
                        # Top-5 accuracy
                        _, pred_top5 = outputs.topk(5, 1, True, True)
                        correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()
                        
                        total += labels.size(0)
                        
                        # 更新进度和实时准确度
                        current_top1 = correct_top1 / total if total > 0 else 0.0
                        current_top5 = correct_top5 / total if total > 0 else 0.0
                        
                        progress.update(
                            task, 
                            advance=1,
                            top1=current_top1,
                            top5=current_top5
                        )
                        
                        # GPU内存管理
                        if torch.cuda.is_available() and (i + 1) % 20 == 0:
                            memory_used = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
                            if memory_used > 10:
                                torch.cuda.empty_cache()
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            console.print(f"[red]❌ GPU OOM at batch {i}, stopping accuracy test[/red]")
                            break
                        else:
                            raise e
        
        top1_acc = correct_top1 / total if total > 0 else 0.0
        top5_acc = correct_top5 / total if total > 0 else 0.0
        
        console.print(f"    [green]✅ Final Results - Top-1: {top1_acc:.4f}, Top-5: {top5_acc:.4f} (GPU {self.gpu_id})[/green]")
        
        return {
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'total_samples': total,
            'gpu_id': self.gpu_id
        }
    
    def test_efficiency(self, model, sample_batch):
        """测试推理效率"""
        model.eval()
        
        # 移动到指定GPU
        if torch.cuda.is_available():
            sample_batch = sample_batch.cuda(self.gpu_id, non_blocking=True)
            torch.cuda.synchronize(self.gpu_id)
        
        # 预热
        console.print("    [yellow]🔥 Warming up...[/yellow]", end="")
        with torch.no_grad():
            for _ in range(5):
                _ = model(sample_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.gpu_id)
        console.print(" [green]Done[/green]")
        
        # 计时 - 使用简单的进度条
        num_runs = 20
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Timing inference..."),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True  # 完成后清除进度条
        ) as progress:
            
            task = progress.add_task("timing", total=num_runs)
            times = []
            
            with torch.no_grad():
                for i in range(num_runs):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.gpu_id)
                    
                    start = time.time()
                    outputs = model(sample_batch)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.gpu_id)
                    
                    times.append(time.time() - start)
                    progress.update(task, advance=1)
        
        avg_time = sum(times) / len(times)
        std_time = torch.tensor(times).std().item()
        throughput = sample_batch.size(0) / avg_time  # images/sec
        
        console.print(f"    [green]⚡ {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms, {throughput:.1f} img/s (GPU {self.gpu_id})[/green]")
        
        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'throughput': throughput,
            'batch_size': sample_batch.size(0),
            'gpu_id': self.gpu_id
        }

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        system_info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_count': torch.cuda.device_count(),
                'used_gpu_id': self.gpu_id,
            })
            
            # 获取使用的GPU信息
            if self.gpu_id is not None:
                try:
                    gpu_props = torch.cuda.get_device_properties(self.gpu_id)
                    system_info['gpu_info'] = {
                        'name': gpu_props.name,
                        'memory_total': gpu_props.total_memory // 1024**2,  # MB
                        'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                    }
                except Exception as e:
                    system_info['gpu_info'] = f"Error getting GPU info: {e}"
        
        return system_info

    def save_results(self):
        """保存结果和配置信息"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None
        
        # 构建完整的结果数据
        full_results = {
            'experiment_metadata': {
                'experiment_type': 'similarity_comparison',
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds() if duration else None,
                'duration_human': str(duration) if duration else None,
            },
            'system_info': self.get_system_info(),
            'config': serialize_config(self.config),
            'results': self.results,
            'summary_stats': self.get_summary_statistics()
        }
        
        # 保存带时间戳的详细结果
        timestamp = end_time.strftime("%Y%m%d_%H%M%S")
        detailed_file = output_dir / f'sim_comp_{timestamp}.json'
        
        with open(detailed_file, 'w') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        # 也保存一个最新版本（用于可视化）
        latest_file = output_dir / 'similarity_comparison.json'
        with open(latest_file, 'w') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        # 单独保存配置文件（方便查看）
        config_file = output_dir / f'config_{timestamp}.json'
        with open(config_file, 'w') as f:
            json.dump({
                'config': serialize_config(self.config),
                'experiment_metadata': full_results['experiment_metadata'],
                'system_info': full_results['system_info']
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed results saved to {detailed_file}")
        print(f"💾 Latest results saved to {latest_file}")
        print(f"💾 Configuration saved to {config_file}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """生成汇总统计信息"""
        stats = {
            'total_strategies': len(self.config.strategies),
            'total_r_values': len(self.config.r_values),
            'successful_experiments': 0,
            'failed_experiments': 0,
            'strategies_tested': [],
            'r_values_tested': self.config.r_values.copy()
        }
        
        for strategy_name, strategy_results in self.results.items():
            if 'error' in strategy_results:
                stats['failed_experiments'] += 1
            else:
                stats['successful_experiments'] += 1
                stats['strategies_tested'].append(strategy_name)
        
        # 计算最佳结果
        best_accuracy = {'strategy': None, 'r': None, 'accuracy': 0}
        best_efficiency = {'strategy': None, 'r': None, 'throughput': 0}
        
        for strategy_name, strategy_results in self.results.items():
            if 'error' in strategy_results:
                continue
                
            for r_key, result in strategy_results.items():
                if not r_key.startswith('r_') or 'error' in result:
                    continue
                    
                r = int(r_key.split('_')[1])
                accuracy_data = result.get('accuracy', {})
                efficiency_data = result.get('efficiency', {})
                
                top1_acc = accuracy_data.get('top1_accuracy', 0)
                throughput = efficiency_data.get('throughput', 0)
                
                if top1_acc > best_accuracy['accuracy']:
                    best_accuracy = {
                        'strategy': strategy_name,
                        'r': r,
                        'accuracy': top1_acc
                    }
                
                if throughput > best_efficiency['throughput']:
                    best_efficiency = {
                        'strategy': strategy_name,
                        'r': r,
                        'throughput': throughput
                    }
        
        stats['best_accuracy'] = best_accuracy
        stats['best_efficiency'] = best_efficiency
        
        return stats

    def print_summary(self):
        """打印结果摘要"""
        print("\n📈 RESULTS SUMMARY")
        print("=" * 80)
        
        for r in self.config.r_values:
            print(f"\nR = {r} (remove {r} tokens):")
            print("-" * 50)
            
            # 收集该r值下所有策略的结果
            r_results = []
            for strategy_name, strategy_result in self.results.items():
                r_key = f'r_{r}'
                if r_key in strategy_result and 'error' not in strategy_result[r_key]:
                    result = strategy_result[r_key]
                    
                    # 访问嵌套字典中的数值
                    accuracy_data = result.get('accuracy', {})
                    efficiency_data = result.get('efficiency', {})
                    
                    r_results.append({
                        'strategy': strategy_name,
                        'top1_accuracy': accuracy_data.get('top1_accuracy', 0.0),
                        'top5_accuracy': accuracy_data.get('top5_accuracy', 0.0),
                        'avg_time': efficiency_data.get('avg_time', 0.0),
                        'throughput': efficiency_data.get('throughput', 0.0),
                        'total_samples': accuracy_data.get('total_samples', 0)
                    })
            
            if not r_results:
                print("❌ No valid results for this r value")
                continue
            
            # 按准确度排序
            # r_results.sort(key=lambda x: x['top1_accuracy'], reverse=True)

            # 打印表头
            print(f"{'Strategy':<12} {'Top1-Acc':<10} {'Top5-Acc':<10} {'Time(s)':<10} {'Throughput(im/s)':<12} {'Samples'}")
            print("-" * 70)
            # 打印每个策略的结果
            for result in r_results:
                try:
                    print(f"{result['strategy']:<12} "
                        f"{result['top1_accuracy']:<10.4f} "
                        f"{result['top5_accuracy']:<10.4f} "
                        f"{result['avg_time']:<10.4f} "
                        f"{result['throughput']:<10.1f} "
                        f"{result['total_samples']}")
                except Exception as e:
                    print(f"Error formatting results for {result['strategy']}: {e}")
            
            print()
