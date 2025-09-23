import torch
import subprocess
import time
import os
from typing import List, Tuple, Optional

class GPUManager:
    """GPU资源管理器"""
    
    def __init__(self):
        self.available_gpus = self._get_available_gpus()
        
    def _get_available_gpus(self) -> List[int]:
        """获取所有可用GPU列表"""
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return []
        
        gpu_count = torch.cuda.device_count()
        print(f"🎮 Found {gpu_count} GPU(s)")
        
        return list(range(gpu_count))
    
    def get_gpu_memory_info(self, device_id: int) -> Tuple[float, float, float]:
        """
        获取GPU内存信息
        Returns: (used_memory_MB, total_memory_MB, usage_percent)
        """
        try:
            # 使用nvidia-ml-py3更准确，但这里用nvidia-smi作为后备
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                 '--format=csv,nounits,noheader', f'--id={device_id}'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(','))
                usage_percent = (used / total) * 100
                return used, total, usage_percent
            else:
                # 后备方案：使用PyTorch
                torch.cuda.set_device(device_id)
                used = torch.cuda.memory_allocated(device_id) / 1024 / 1024
                total = torch.cuda.get_device_properties(device_id).total_memory / 1024 / 1024
                return used, total, (used / total) * 100
                
        except Exception as e:
            print(f"⚠️ Warning: Could not get memory info for GPU {device_id}: {e}")
            return 0, 0, 0
    
    def get_gpu_utilization(self, device_id: int) -> float:
        """获取GPU利用率"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', 
                 '--format=csv,nounits,noheader', f'--id={device_id}'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                return 0.0
                
        except Exception as e:
            print(f"⚠️ Warning: Could not get utilization for GPU {device_id}: {e}")
            return 0.0
    
    def get_gpu_status(self) -> List[dict]:
        """获取所有GPU的状态信息"""
        gpu_status = []
        
        for gpu_id in self.available_gpus:
            used_mem, total_mem, mem_percent = self.get_gpu_memory_info(gpu_id)
            utilization = self.get_gpu_utilization(gpu_id)
            
            # 获取GPU名称
            try:
                gpu_name = torch.cuda.get_device_name(gpu_id)
            except:
                gpu_name = f"GPU {gpu_id}"
            
            status = {
                'id': gpu_id,
                'name': gpu_name,
                'memory_used': used_mem,
                'memory_total': total_mem,
                'memory_percent': mem_percent,
                'utilization': utilization,
                'is_available': mem_percent < 10 and utilization < 10  # 认为<10%为空闲
            }
            
            gpu_status.append(status)
        
        return gpu_status
    
    def select_best_gpu(self, min_memory_mb: int = 2048) -> Optional[int]:
        """
        自动选择最佳GPU
        Args:
            min_memory_mb: 最小所需内存(MB)
        Returns:
            GPU ID 或 None
        """
        gpu_status = self.get_gpu_status()
        
        # 过滤出满足内存要求的GPU
        suitable_gpus = [
            gpu for gpu in gpu_status 
            if (gpu['memory_total'] - gpu['memory_used']) >= min_memory_mb
        ]
        
        if not suitable_gpus:
            print(f"❌ No GPU has enough free memory ({min_memory_mb}MB required)")
            return None
        
        # 按综合得分排序（内存使用率 + 利用率）
        suitable_gpus.sort(
            key=lambda x: x['memory_percent'] + x['utilization']
        )
        
        best_gpu = suitable_gpus[0]
        print(f"🎯 Selected GPU {best_gpu['id']} ({best_gpu['name']})")
        print(f"   Memory: {best_gpu['memory_used']:.0f}/{best_gpu['memory_total']:.0f}MB "
              f"({best_gpu['memory_percent']:.1f}%)")
        print(f"   Utilization: {best_gpu['utilization']:.1f}%")
        
        return best_gpu['id']
    
    def print_gpu_status(self):
        """打印所有GPU状态"""
        gpu_status = self.get_gpu_status()
        
        print("\n🎮 GPU Status:")
        print("=" * 80)
        print(f"{'ID':<3} {'Name':<25} {'Memory Usage':<20} {'Util%':<8} {'Status'}")
        print("-" * 80)
        
        for gpu in gpu_status:
            memory_str = f"{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f}MB"
            memory_percent = f"({gpu['memory_percent']:.1f}%)"
            memory_display = f"{memory_str} {memory_percent}"
            
            status = "🟢 Available" if gpu['is_available'] else "🔴 Busy"
            
            print(f"{gpu['id']:<3} {gpu['name']:<25} {memory_display:<20} "
                  f"{gpu['utilization']:.1f}%{'':<4} {status}")
        
        print("=" * 80)
    
    def wait_for_available_gpu(self, min_memory_mb: int = 2048, 
                             check_interval: int = 30, max_wait_time: int = 1800):
        """
        等待GPU变为可用状态
        Args:
            min_memory_mb: 最小内存要求
            check_interval: 检查间隔(秒)
            max_wait_time: 最大等待时间(秒)
        """
        print(f"⏳ Waiting for available GPU (min {min_memory_mb}MB memory)...")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            gpu_id = self.select_best_gpu(min_memory_mb)
            if gpu_id is not None:
                return gpu_id
            
            print(f"⏳ No available GPU, checking again in {check_interval}s...")
            time.sleep(check_interval)
        
        raise RuntimeError(f"No GPU became available within {max_wait_time}s")

def setup_gpu_environment(gpu_id: Optional[int] = None, min_memory_mb: int = 2048):
    """
    设置GPU环境
    Args:
        gpu_id: 指定GPU ID，None表示自动选择
        min_memory_mb: 最小内存要求
    Returns:
        选中的GPU ID
    """
    gpu_manager = GPUManager()
    
    if not gpu_manager.available_gpus:
        raise RuntimeError("No CUDA GPUs available")
    
    # 显示GPU状态
    gpu_manager.print_gpu_status()
    
    # 选择GPU
    if gpu_id is not None:
        # 手动指定GPU
        if gpu_id not in gpu_manager.available_gpus:
            raise ValueError(f"GPU {gpu_id} not available")
        
        # 检查指定GPU是否有足够内存
        used_mem, total_mem, mem_percent = gpu_manager.get_gpu_memory_info(gpu_id)
        free_memory = total_mem - used_mem
        
        if free_memory < min_memory_mb:
            print(f"⚠️ Warning: GPU {gpu_id} may not have enough free memory "
                  f"({free_memory:.0f}MB free, {min_memory_mb}MB required)")
        
        selected_gpu = gpu_id
        print(f"🎯 Using manually specified GPU {gpu_id}")
        
    else:
        # 自动选择GPU
        selected_gpu = gpu_manager.select_best_gpu(min_memory_mb)
        
        if selected_gpu is None:
            # 等待GPU可用
            selected_gpu = gpu_manager.wait_for_available_gpu(min_memory_mb)
    
    # 设置CUDA设备
    torch.cuda.set_device(selected_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
    
    print(f"✅ GPU {selected_gpu} is now active")
    
    return selected_gpu
