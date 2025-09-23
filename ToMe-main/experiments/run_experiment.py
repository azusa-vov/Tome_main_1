# run_experiment.py
import os
import sys
import argparse
from config.experiment_config import (
    quick_imagenet_config, 
    full_imagenet_config,
    Config
)
from src.similarity_comparison import SimilarityComparison
from utils.visualization import plot_all_results
from utils.gpu_utils import GPUManager
from utils.result_analyzer import analyze_results, compare_experiments

def main():
    parser = argparse.ArgumentParser(description='Run ToMe similarity strategy experiments')
    parser.add_argument('--config', choices=['quick', 'full'], default=None,
                       help='Experiment configuration')
    parser.add_argument('--gpu', type=int, default=None,
                       help='Specify GPU ID (auto-select if not provided)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to ImageNet dataset (default from config)')
    parser.add_argument('--list-gpus', action='store_true',
                       help='List available GPUs and exit')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--analyze', type=str, default=None,
                       help='Analyze existing results file')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'),
                       help='Compare two experiment results')
    
    # 可学习SATD相关参数
    parser.add_argument('--learnable-satd-checkpoint', type=str, default=None,
                       help='Path to learnable SATD checkpoint file')
    parser.add_argument('--disable-learnable-satd', action='store_true',
                       help='Disable learnable SATD even if checkpoint is available')
    
    args = parser.parse_args()
    
    # 设置环境变量以便配置文件读取
    if args.learnable_satd_checkpoint:
        os.environ['LEARNABLE_SATD_CHECKPOINT'] = args.learnable_satd_checkpoint
    
    # 分析模式
    if args.analyze:
        analyze_results(args.analyze)
        return
    
    # 比较模式
    if args.compare:
        compare_experiments(args.compare[0], args.compare[1])
        return
    
    # 列出GPU信息
    if args.list_gpus:
        gpu_manager = GPUManager()
        gpu_manager.print_gpu_status()
        return
    
    # 正常实验模式-选择配置
    include_learnable_satd = not args.disable_learnable_satd
    
    if args.config is None:
        config = Config(gpu_id=args.gpu)
        # 如果指定了checkpoint，手动添加可学习SATD
        if include_learnable_satd and args.learnable_satd_checkpoint:
            from config.experiment_config import create_learnable_satd_strategy
            device = f'cuda:{args.gpu}' if args.gpu is not None else 'cuda'
            learnable_satd = create_learnable_satd_strategy(args.learnable_satd_checkpoint, device)
            config.strategies.append(("LearnableSATD", learnable_satd))
        print("🔧 Using default configuration")
    elif args.config == "quick":
        config = quick_imagenet_config(gpu_id=args.gpu, include_learnable_satd=include_learnable_satd)
        print("🔧 Using quick configuration")
    elif args.config == "full":
        config = full_imagenet_config(gpu_id=args.gpu, include_learnable_satd=include_learnable_satd)
        print("🔧 Using full configuration")
    
    # 设置数据集路径
    if args.dataset is not None:
        config.dataset_path = args.dataset
    print(f"📁 Using dataset: {config.dataset_path}")
    
    if args.gpu is not None:
        print(f"🎯 Manually specified GPU: {args.gpu}")
    else:
        print("🎯 Auto-selecting GPU")
    
    # 显示可学习SATD状态
    learnable_satd_strategies = [name for name, _ in config.strategies if name == "LearnableSATD"]
    if learnable_satd_strategies:
        checkpoint_path = args.learnable_satd_checkpoint or os.environ.get('LEARNABLE_SATD_CHECKPOINT')
        if checkpoint_path:
            print(f"🧠 LearnableSATD enabled with checkpoint: {checkpoint_path}")
        else:
            print(f"🧠 LearnableSATD enabled with default Hadamard initialization")
    else:
        print(f"🧠 LearnableSATD disabled")
    
    print(f"📊 Strategies to test: {[name for name, _ in config.strategies]}")
    
    # 运行实验
    try:
        experiment = SimilarityComparison(config)
        results = experiment.run()
        
        # 生成图表和分析
        if not args.skip_plots:
            print("\n📊 Generating comprehensive analysis...")
            plot_all_results(f"{config.output_dir}/similarity_comparison.json", config.output_dir)
        
        print(f"\n✅ Experiment completed on GPU {experiment.gpu_id}!")
        print(f"📁 Results saved in: {config.output_dir}/")
        
        # 快速分析
        # print("\n" + "="*60)
        # analyze_results(f"{config.output_dir}/similarity_comparison.json")

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
