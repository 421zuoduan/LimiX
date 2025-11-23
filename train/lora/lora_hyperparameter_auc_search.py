import os
import sys
import argparse
import json
import itertools
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 复用已有 LoRA AUC 训练器
from train.lora.train_lora_auc import LoRATrainer, load_and_preprocess_data  # noqa: E402


def parse_list(arg: str, cast=float) -> List:
    """将逗号分隔字符串转为列表。"""
    return [cast(x) for x in arg.split(',') if x.strip() != '']


def collect_best_metrics(trainer: LoRATrainer) -> Dict[str, Any]:
    """从 trainer 的 tracker 中提取最佳指标。"""
    metrics = {}
    keys = [
        'val_loss',
        'accuracy_subset',
        'accuracy_elementwise',
        'f1_macro_elementwise',
        'f1_micro_elementwise',
        'roc_auc_macro_elementwise',
        'roc_auc_micro_elementwise',
    ]
    for key in keys:
        tracker = trainer.metric_trackers.get(key)
        if tracker is None:
            metrics[f'best_{key}'] = None
        else:
            best = tracker.get('best')
            metrics[f'best_{key}'] = float(best) if np.isfinite(best) else None
    return metrics


def main():
    parser = argparse.ArgumentParser(description="LoRA AUC 搜参")
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径 (Excel)')
    parser.add_argument('--target_cols', nargs='+', required=True, help='目标列名列表')
    parser.add_argument('--seeds', nargs='+', type=int, default=[10, 20, 30], help='随机种子')
    parser.add_argument('--output_dir', type=str, default='./results/lora_auc_search', help='输出目录')

    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=128.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')

    parser.add_argument('--auc_coeffs', type=str, default='0.005,0.01,0.02', help='AUC 权重列表，逗号分隔')
    parser.add_argument('--auc_start_epochs', type=str, default='8,12,15', help='AUC 启用轮次列表，逗号分隔')
    parser.add_argument('--auc_gammas', type=str, default='0.5,1.0', help='AUC gamma 列表，逗号分隔')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    auc_coeffs = parse_list(args.auc_coeffs, float)
    auc_starts = parse_list(args.auc_start_epochs, int)
    auc_gammas = parse_list(args.auc_gammas, float)

    # 数据载入一次复用
    X, y = load_and_preprocess_data(args.data_path, args.target_cols)

    search_space = list(itertools.product(auc_coeffs, auc_starts, auc_gammas))
    print(f"搜索组合数: {len(search_space)}")

    all_results: List[Dict[str, Any]] = []

    for idx, (auc_coeff, auc_start, auc_gamma) in enumerate(search_space, 1):
        config_name = f"aucC{auc_coeff}_start{auc_start}_g{auc_gamma}"
        config_dir = os.path.join(args.output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        print(f"\n[{idx}/{len(search_space)}] 配置: {config_name}")

        for seed in args.seeds:
            print(f"  Seed {seed}")
            trainer = LoRATrainer(
                model_path=args.model_path,
                target_layers=[10, 11],
                rank=args.rank,
                alpha=args.alpha,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                device=args.device,
                run_seed=seed,
                auc_coeff=auc_coeff,
                auc_start_epoch=auc_start,
                auc_gamma=auc_gamma
            )

            train_loader, val_loader = trainer.prepare_data(
                X, y, test_size=args.val_ratio, random_state=seed
            )

            seed_dir = os.path.join(config_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            trainer.train(train_loader, val_loader, save_dir=seed_dir)
            trainer.save_lora_weights(os.path.join(seed_dir, 'lora_weights.pth'))

            best_metrics = collect_best_metrics(trainer)
            final_metrics = trainer.val_metrics[-1] if trainer.val_metrics else {}

            record = {
                'config': config_name,
                'seed': seed,
                'auc_coeff': auc_coeff,
                'auc_start_epoch': auc_start,
                'auc_gamma': auc_gamma,
                'rank': args.rank,
                'alpha': args.alpha,
                'dropout': args.dropout,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
            }
            record.update(best_metrics)
            record.update({f'final_{k}': v for k, v in final_metrics.items()})
            all_results.append(record)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(args.output_dir, 'auc_search_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n全部结果已保存到 {results_path}")

        # 汇总：按配置聚合均值/标准差
        group_cols = ['config', 'auc_coeff', 'auc_start_epoch', 'auc_gamma']
        metric_cols = [c for c in results_df.columns if c.startswith('best_')]
        summary = results_df.groupby(group_cols)[metric_cols].agg(['mean', 'std'])
        summary_path = os.path.join(args.output_dir, 'auc_search_summary.csv')
        summary.to_csv(summary_path)
        print(f"汇总结果已保存到 {summary_path}")


if __name__ == "__main__":
    main()
