import json
from pathlib import Path
from typing import Dict, List, Any
from utils.config_utils import compare_configs

class ResultAnalyzer:
    """实验结果分析工具"""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        self.data = data
        self.results = data['results']
        self.metadata = data.get('experiment_metadata', {})
        self.config = data.get('config', {})
        self.system_info = data.get('system_info', {})
        self.summary_stats = data.get('summary_stats', {})
    
    def print_experiment_info(self):
        """打印实验基本信息"""
        print("🔬 EXPERIMENT INFORMATION")
        print("=" * 50)
        
        # 基本信息
        if self.metadata:
            print(f"Type: {self.metadata.get('experiment_type', 'Unknown')}")
            print(f"Start Time: {self.metadata.get('start_time', 'Unknown')}")
            print(f"Duration: {self.metadata.get('duration_human', 'Unknown')}")
        else:
            print("Metadata: Not available (old format)")
        
        # 配置信息
        if self.config:
            print(f"\n📋 Configuration:")
            print(f"Model: {self.config.get('model_name', 'Unknown')}")
            print(f"Dataset: {self.config.get('dataset_path', 'Unknown')}")
            print(f"Batch Size: {self.config.get('batch_size', 'Unknown')}")
            print(f"Samples: {self.config.get('num_samples', 'All')}")
            print(f"R Values: {self.config.get('r_values', [])}")
            
            # 策略信息
            strategies = self.config.get('strategies', [])
            if strategies:
                print(f"Strategies ({len(strategies)}):")
                for strategy in strategies:
                    if isinstance(strategy, dict):
                        name = strategy.get('name', 'Unknown')
                        class_name = strategy.get('class', 'Unknown')
                        print(f"  - {name} ({class_name})")
                    else:
                        print(f"  - {strategy}")
        else:
            print("Configuration: Not available")
        
        # 系统信息
        if self.system_info:
            print(f"\n🖥️ System Information:")
            print(f"Platform: {self.system_info.get('platform', 'Unknown')}")
            print(f"PyTorch: {self.system_info.get('pytorch_version', 'Unknown')}")
            print(f"CUDA: {self.system_info.get('cuda_version', 'N/A')}")
            if 'gpu_info' in self.system_info:
                gpu_info = self.system_info['gpu_info']
                if isinstance(gpu_info, dict):
                    print(f"GPU: {gpu_info.get('name', 'Unknown')} ({gpu_info.get('memory_total', 'Unknown')}MB)")
        
        print("=" * 50)
    
    def export_config(self, output_file: str = None):
        """导出配置到单独文件"""
        if output_file is None:
            output_file = self.results_file.parent / f"config_{self.results_file.stem}.json"
        
        config_data = {
            'experiment_metadata': self.data.get('experiment_metadata', {}),
            'config': self.data.get('config', {}),
            'system_info': self.data.get('system_info', {})
        }
        
        with open(output_file, 'w') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Configuration exported to {output_file}")
        return output_file

def analyze_results(results_file: str):
    """分析结果文件的快捷函数"""
    analyzer = ResultAnalyzer(results_file)
    analyzer.print_experiment_info()
    return analyzer

def compare_experiments(results_file1: str, results_file2: str):
    """比较两个实验的配置差异"""
    with open(results_file1, 'r') as f:
        data1 = json.load(f)
    
    with open(results_file2, 'r') as f:
        data2 = json.load(f)
    
    config1 = data1.get('config', {})
    config2 = data2.get('config', {})
    
    differences = compare_configs(config1, config2)
    
    print("🔍 CONFIGURATION DIFFERENCES")
    print("=" * 40)
    
    if not differences:
        print("✅ No differences found in configurations")
    else:
        for key, diff in differences.items():
            print(f"\n{key}:")
            print(f"  File 1: {diff['in_config1']}")
            print(f"  File 2: {diff['in_config2']}")
    
    return differences
