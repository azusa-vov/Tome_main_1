import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

def load_results_data(results_file):
    """加载并解析结果数据，兼容新旧格式"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    metadata = data.get('experiment_metadata', {})
    config = data.get('config', {})
    system_info = data.get('system_info', {})
    summary_stats = data.get('summary_stats', {})
    
    return {
        'results': results,
        'metadata': metadata,
        'config': config,
        'system_info': system_info,
        'summary_stats': summary_stats,
        'format': 'new'
    }

def plot_results(results_file, output_dir="results"):
    """绘制实验结果图表"""
    # 加载数据
    data = load_results_data(results_file)
    results = data['results']
    config = data['config']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not results:
        print("❌ No results data found")
        return
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 为每个策略收集数据
    strategy_data = {}
    
    for strategy_name, strategy_results in results.items():
        if not isinstance(strategy_results, dict) or 'error' in strategy_results:
            print(f"⚠️ Skipping {strategy_name} due to error or invalid data")
            continue
            
        r_values = []
        top1_accuracies = []
        top5_accuracies = []
        avg_times = []
        throughputs = []
        
        # 提取数据
        for r_key, result in strategy_results.items():
            if not r_key.startswith('r_') or 'error' in result:
                continue
                
            try:
                r = int(r_key.split('_')[1])  # 从 'r_8' 提取 8
                
                # 正确访问嵌套字典
                accuracy_data = result.get('accuracy', {})
                efficiency_data = result.get('efficiency', {})
                
                if not accuracy_data or not efficiency_data:
                    print(f"⚠️ Missing data for {strategy_name} {r_key}")
                    continue
                
                top1_acc = accuracy_data.get('top1_accuracy', 0.0)
                top5_acc = accuracy_data.get('top5_accuracy', 0.0)
                avg_time = efficiency_data.get('avg_time', 0.0)
                throughput = efficiency_data.get('throughput', 0.0)
                
                if top1_acc > 0 or avg_time > 0:  # 至少有一些有效数据
                    r_values.append(r)
                    top1_accuracies.append(top1_acc)
                    top5_accuracies.append(top5_acc)
                    avg_times.append(avg_time)
                    throughputs.append(throughput)
                
            except (ValueError, KeyError, TypeError) as e:
                print(f"⚠️ Error processing {strategy_name} {r_key}: {e}")
                continue
        
        if not r_values:  # 如果没有有效数据，跳过
            print(f"⚠️ No valid data for {strategy_name}")
            continue
            
        # 按r值排序
        sorted_data = sorted(zip(r_values, top1_accuracies, top5_accuracies, avg_times, throughputs))
        if sorted_data:
            r_values, top1_accuracies, top5_accuracies, avg_times, throughputs = zip(*sorted_data)
            
            strategy_data[strategy_name] = {
                'r_values': list(r_values),
                'top1_accuracies': list(top1_accuracies),
                'top5_accuracies': list(top5_accuracies),
                'avg_times': list(avg_times),
                'throughputs': list(throughputs)
            }
    
    if not strategy_data:
        print("❌ No valid data to plot")
        return
    
    print(f"✅ Found valid data for {len(strategy_data)} strategies")
    
    # 1. Top-1准确度对比图
    ax1.set_title('Top-1 Accuracy vs Token Reduction', fontsize=14, fontweight='bold')
    for strategy_name, data in strategy_data.items():
        ax1.plot(data['r_values'], data['top1_accuracies'], 
                marker='o', linewidth=2, markersize=6, label=strategy_name)
    
    ax1.set_xlabel('R (tokens to remove)')
    ax1.set_ylabel('Top-1 Accuracy')
    if strategy_data:  # 只有在有数据时才添加图例
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 2. Top-5准确度对比图
    ax2.set_title('Top-5 Accuracy vs Token Reduction', fontsize=14, fontweight='bold')
    for strategy_name, data in strategy_data.items():
        ax2.plot(data['r_values'], data['top5_accuracies'], 
                marker='s', linewidth=2, markersize=6, label=strategy_name)
    
    ax2.set_xlabel('R (tokens to remove)')
    ax2.set_ylabel('Top-5 Accuracy')
    if strategy_data:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. 推理时间对比图
    ax3.set_title('Inference Time vs Token Reduction', fontsize=14, fontweight='bold')
    for strategy_name, data in strategy_data.items():
        ax3.plot(data['r_values'], data['avg_times'], 
                marker='^', linewidth=2, markersize=6, label=strategy_name)
    
    ax3.set_xlabel('R (tokens to remove)')
    ax3.set_ylabel('Inference Time (s)')
    if strategy_data:
        ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 吞吐量对比图
    ax4.set_title('Throughput vs Token Reduction', fontsize=14, fontweight='bold')
    for strategy_name, data in strategy_data.items():
        ax4.plot(data['r_values'], data['throughputs'], 
                marker='d', linewidth=2, markersize=6, label=strategy_name)
    
    ax4.set_xlabel('R (tokens to remove)')
    ax4.set_ylabel('Throughput (images/sec)')
    if strategy_data:
        ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = output_dir / 'similarity_comparison_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Plots saved to {plot_file}")
    
    # 显示图表（如果在交互环境中）
    try:
        plt.show()
    except Exception:
        print("📊 Plots generated (display not available in non-interactive environment)")
    
    plt.close()
    
    return strategy_data

def plot_accuracy_efficiency_tradeoff(results_file, output_dir="results"):
    """绘制准确度-效率权衡图"""
    data = load_results_data(results_file)
    results = data['results']
    
    output_dir = Path(output_dir)
    
    plt.figure(figsize=(10, 8))
    
    strategy_count = 0
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(results), 1)))
    
    for i, (strategy_name, strategy_results) in enumerate(results.items()):
        if not isinstance(strategy_results, dict) or 'error' in strategy_results:
            continue
            
        accuracies = []
        throughputs = []
        r_values = []
        
        for r_key, result in strategy_results.items():
            if not r_key.startswith('r_') or 'error' in result:
                continue
                
            try:
                r = int(r_key.split('_')[1])
                accuracy_data = result.get('accuracy', {})
                efficiency_data = result.get('efficiency', {})
                
                top1_acc = accuracy_data.get('top1_accuracy', 0.0)
                throughput = efficiency_data.get('throughput', 0.0)
                
                if top1_acc > 0 and throughput > 0:
                    accuracies.append(top1_acc)
                    throughputs.append(throughput)
                    r_values.append(r)
                    
            except (ValueError, KeyError, TypeError):
                continue
        
        if accuracies and throughputs:
            # 绘制策略曲线
            plt.plot(throughputs, accuracies, 'o-', 
                    color=colors[strategy_count], linewidth=2, markersize=8, label=strategy_name)
            
            # 为每个点添加r值标注
            for j, (acc, thr, r) in enumerate(zip(accuracies, throughputs, r_values)):
                plt.annotate(f'r={r}', (thr, acc), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            strategy_count += 1
    
    if strategy_count > 0:
        plt.xlabel('Throughput (images/sec)', fontsize=12)
        plt.ylabel('Top-1 Accuracy', fontsize=12)
        plt.title('Accuracy vs Efficiency Trade-off', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        tradeoff_file = output_dir / 'accuracy_efficiency_tradeoff.png'
        plt.savefig(tradeoff_file, dpi=300, bbox_inches='tight')
        print(f"📊 Trade-off plot saved to {tradeoff_file}")
    else:
        print("⚠️ No valid data for trade-off plot")
    
    try:
        plt.show()
    except Exception:
        pass
    
    plt.close()

def generate_summary_table(results_file, output_dir="results"):
    """生成结果汇总表"""
    data = load_results_data(results_file)
    results = data['results']
    
    output_dir = Path(output_dir)
    
    # 创建汇总表
    summary_data = []
    
    for strategy_name, strategy_results in results.items():
        if not isinstance(strategy_results, dict) or 'error' in strategy_results:
            continue
            
        for r_key, result in strategy_results.items():
            if not r_key.startswith('r_') or 'error' in result:
                continue
                
            try:
                r = int(r_key.split('_')[1])
                accuracy_data = result.get('accuracy', {})
                efficiency_data = result.get('efficiency', {})
                
                summary_data.append({
                    'Strategy': strategy_name,
                    'R': r,
                    'Top1_Acc': accuracy_data.get('top1_accuracy', 0.0),
                    'Top5_Acc': accuracy_data.get('top5_accuracy', 0.0),
                    'Avg_Time': efficiency_data.get('avg_time', 0.0),
                    'Throughput': efficiency_data.get('throughput', 0.0),
                    'Samples': accuracy_data.get('total_samples', 0)
                })
                
            except (ValueError, KeyError, TypeError) as e:
                print(f"⚠️ Error processing {strategy_name} {r_key}: {e}")
                continue
    
    if summary_data:
        try:
            import pandas as pd
            df = pd.DataFrame(summary_data)
            csv_file = output_dir / 'results_summary.csv'
            df.to_csv(csv_file, index=False)
            print(f"📄 Summary table saved to {csv_file}")
            return df
        except ImportError:
            print("⚠️ pandas not available, saving as JSON instead")
            json_file = output_dir / 'results_summary.json'
            with open(json_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"📄 Summary data saved to {json_file}")
            return summary_data
    else:
        print("❌ No valid data for summary table")
        return None

def print_experiment_summary(results_file):
    """打印实验摘要信息"""
    data = load_results_data(results_file)
    
    print("\n📋 EXPERIMENT SUMMARY")
    print("=" * 50)
    
    # 基本信息
    metadata = data['metadata']
    if metadata:
        print(f"Start Time: {metadata.get('start_time', 'Unknown')}")
        print(f"Duration: {metadata.get('duration_human', 'Unknown')}")
    
    # 配置信息
    config = data['config']
    if config:
        print(f"Model: {config.get('model_name', 'Unknown')}")
        print(f"Dataset: {config.get('dataset_path', 'Unknown')}")
        print(f"Samples: {config.get('num_samples', 'Unknown')}")
        
        # 策略信息
        strategies = config.get('strategies', [])
        if strategies:
            print(f"Strategies: {[s.get('name', 'Unknown') for s in strategies]}")
        
        r_values = config.get('r_values', [])
        if r_values:
            print(f"R Values: {r_values}")
    
    # 结果统计
    results = data['results']
    successful = sum(1 for r in results.values() if isinstance(r, dict) and 'error' not in r)
    failed = len(results) - successful
    
    print(f"Successful experiments: {successful}")
    print(f"Failed experiments: {failed}")
    
    print("=" * 50)

# 主绘图函数 - 包含所有图表
def plot_all_results(results_file, output_dir="results"):
    """生成所有图表和汇总"""
    print("📊 Generating comprehensive results analysis...")
    
    # 打印实验摘要
    print_experiment_summary(results_file)
    
    # 基本对比图
    strategy_data = plot_results(results_file, output_dir)
    
    if strategy_data:
        # 准确度-效率权衡图
        plot_accuracy_efficiency_tradeoff(results_file, output_dir)
        
        # 汇总表
        df = generate_summary_table(results_file, output_dir)
        
        print("✅ All plots and analysis completed!")
    else:
        print("⚠️ No valid data found for visualization")
    
    return strategy_data
