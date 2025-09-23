import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import torch

def serialize_config(config) -> Dict[str, Any]:
    """将配置对象转换为可JSON序列化的字典"""
    config_dict = {}
    
    for key, value in config.__dict__.items():
        try:
            # 处理特殊类型
            if key == 'strategies':
                # 序列化相似度策略
                strategies_list = []
                for name, strategy in value:
                    strategy_info = {
                        'name': name,
                        'class': strategy.__class__.__name__,
                        'module': strategy.__class__.__module__,
                    }
                    
                    # 尝试获取策略参数
                    if hasattr(strategy, '__dict__'):
                        strategy_params = {}
                        for param_key, param_value in strategy.__dict__.items():
                            if isinstance(param_value, (str, int, float, bool, list, dict, type(None))):
                                strategy_params[param_key] = param_value
                            else:
                                strategy_params[param_key] = str(param_value)
                        strategy_info['parameters'] = strategy_params
                    
                    strategies_list.append(strategy_info)
                
                config_dict[key] = strategies_list
                
            elif isinstance(value, (str, int, float, bool, list, type(None))):
                # 直接可序列化的类型
                config_dict[key] = value
                
            elif isinstance(value, Path):
                # 路径对象转为字符串
                config_dict[key] = str(value)
                
            else:
                # 其他类型转为字符串表示
                config_dict[key] = str(value)
                
        except Exception as e:
            # 如果序列化失败，保存错误信息
            config_dict[key] = f"<Serialization Error: {str(e)}>"
    
    return config_dict

def load_config_from_results(results_file: str) -> Dict[str, Any]:
    """从结果文件中加载配置信息"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data.get('config', {})

def compare_configs(config1: Dict, config2: Dict) -> Dict[str, Any]:
    """比较两个配置的差异"""
    differences = {}
    
    # 检查config1中的键
    for key, value1 in config1.items():
        if key not in config2:
            differences[key] = {'in_config1': value1, 'in_config2': 'MISSING'}
        elif config2[key] != value1:
            differences[key] = {'in_config1': value1, 'in_config2': config2[key]}
    
    # 检查config2中config1没有的键
    for key, value2 in config2.items():
        if key not in config1:
            differences[key] = {'in_config1': 'MISSING', 'in_config2': value2}
    
    return differences
