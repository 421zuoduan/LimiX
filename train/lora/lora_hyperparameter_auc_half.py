import os
import sys
import argparse
import json
import itertools
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import shutil

DEFAULT_TARGET_LAYERS = [6, 7, 8, 9, 10, 11]

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
    parser = argparse.ArgumentParser(description="LoRA AUC 搜参（上半部分 TransformerBlock）")
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径 (Excel)')
    parser.add_argument('--target_cols', nargs='+', required=True, help='目标列名列表')
    parser.add_argument('--seeds', nargs='+', type=int, default=[10, 20, 30], help='随机种子')
    parser.add_argument('--output_dir', type=str, default='./results/lora_auc_half_search', help='输出目录')

    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=128.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--learning_rates', type=str, default='5e-5,1e-4', help='学习率列表，逗号分隔')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--target_layers', nargs='+', type=int, default=DEFAULT_TARGET_LAYERS,
                        help='注入 LoRA 的 Transformer 层索引 (默认上半部分)')

    parser.add_argument('--auc_coeffs', type=str, default='0.005,0.01,0.02', help='AUC 权重列表，逗号分隔')
    parser.add_argument('--auc_start_epochs', type=str, default='8,12,15', help='AUC 启用轮次列表，逗号分隔')
    parser.add_argument('--auc_gammas', type=str, default='0.5,1.0', help='AUC gamma 列表，逗号分隔')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    auc_coeffs = parse_list(args.auc_coeffs, float)
    auc_starts = parse_list(args.auc_start_epochs, int)
    auc_gammas = parse_list(args.auc_gammas, float)
    learning_rates = parse_list(args.learning_rates, float)

    print(f"Target layers: {args.target_layers}")

    # 数据载入一次复用
    X, y = load_and_preprocess_data(args.data_path, args.target_cols)

    search_space = list(itertools.product(learning_rates, auc_coeffs, auc_starts, auc_gammas))
    print(f"搜索组合数: {len(search_space)}")

    all_results: List[Dict[str, Any]] = []
    per_config_stats: List[Dict[str, Any]] = []

    for idx, (lr, auc_coeff, auc_start, auc_gamma) in enumerate(search_space, 1):
        config_name = f"lr{lr}_aucC{auc_coeff}_start{auc_start}_g{auc_gamma}"
        config_dir = os.path.join(args.output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        print(f"\n[{idx}/{len(search_space)}] 配置: {config_name}")

        seed_records: List[Dict[str, Any]] = []

        for seed in args.seeds:
            print(f"  Seed {seed}")
            trainer = LoRATrainer(
                model_path=args.model_path,
                target_layers=args.target_layers,
                rank=args.rank,
                alpha=args.alpha,
                dropout=args.dropout,
                learning_rate=lr,
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
                'learning_rate': lr,
                'auc_coeff': auc_coeff,
                'auc_start_epoch': auc_start,
                'auc_gamma': auc_gamma,
                'target_layers': "_".join(map(str, args.target_layers)),
                'rank': args.rank,
                'alpha': args.alpha,
                'dropout': args.dropout,
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
            }
            record.update(best_metrics)
            record.update({f'final_{k}': v for k, v in final_metrics.items()})
            all_results.append(record)
            seed_records.append(record)

        # 保存该配置的 seed 级汇总（命名为 epoch_summary.csv 以对齐基础脚本习惯）
        if seed_records:
            df = pd.DataFrame(seed_records)
            df_path = os.path.join(config_dir, 'epoch_summary.csv')
            df.to_csv(df_path, index=False)
            # 记录配置级平均指标用于后续全局最佳选择
            avg_roc_auc = df['best_roc_auc_macro_elementwise'].dropna().mean() if 'best_roc_auc_macro_elementwise' in df else np.nan
            avg_val_loss = df['best_val_loss'].dropna().mean() if 'best_val_loss' in df else np.nan
            per_config_stats.append({
                'config': config_name,
                'mean_roc_auc_macro': avg_roc_auc,
                'mean_val_loss': avg_val_loss,
                'best_seed': df.sort_values(by='best_roc_auc_macro_elementwise', ascending=False).iloc[0]['seed'] if 'best_roc_auc_macro_elementwise' in df else df.iloc[0]['seed']
            })

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(args.output_dir, 'auc_search_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n全部结果已保存到 {results_path}")

        # 汇总：按配置聚合均值/标准差
        group_cols = ['config', 'learning_rate', 'auc_coeff', 'auc_start_epoch', 'auc_gamma']
        metric_cols = [c for c in results_df.columns if c.startswith('best_')]
        summary = results_df.groupby(group_cols)[metric_cols].agg(['mean', 'std'])
        summary_path = os.path.join(args.output_dir, 'auc_search_summary.csv')
        summary.to_csv(summary_path)
        print(f"汇总结果已保存到 {summary_path}")

        # 选择全局最佳（优先 mean_roc_auc_macro 最大，其次 mean_val_loss 最小）
        global_best_dir = os.path.join(args.output_dir, 'global_best_models')
        os.makedirs(global_best_dir, exist_ok=True)
        best_entry = None
        per_config_df = pd.DataFrame(per_config_stats)
        if not per_config_df.empty:
            if per_config_df['mean_roc_auc_macro'].notna().any():
                best_entry = per_config_df.loc[per_config_df['mean_roc_auc_macro'].idxmax()]
            else:
                best_entry = per_config_df.loc[per_config_df['mean_val_loss'].idxmin()]

        if best_entry is not None:
            best_config = best_entry['config']
            best_seed = int(best_entry['best_seed'])
            source_weights = os.path.join(args.output_dir, best_config, f'seed_{best_seed}', 'lora_weights.pth')
            target_weights = os.path.join(global_best_dir, f'{best_config}_best_weights.pth')
            if os.path.isfile(source_weights):
                shutil.copyfile(source_weights, target_weights)
                metadata = {
                    'config': best_config,
                    'best_seed': best_seed,
                    'mean_roc_auc_macro': float(best_entry.get('mean_roc_auc_macro')) if pd.notna(best_entry.get('mean_roc_auc_macro')) else None,
                    'mean_val_loss': float(best_entry.get('mean_val_loss')) if pd.notna(best_entry.get('mean_val_loss')) else None,
                    'weights_path': target_weights
                }
                with open(os.path.join(global_best_dir, f'{best_config}_best_metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"全局最佳权重已保存到 {target_weights}")


if __name__ == "__main__":
    main()
