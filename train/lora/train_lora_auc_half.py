import os
import sys
import argparse
import numpy as np
import pandas as pd

# Ensure project root and current script directory are on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Reuse existing AUC-aware LoRA trainer and data utilities without modifying them
import train_lora_auc as base


DEFAULT_TARGET_LAYERS = [6, 7, 8, 9, 10, 11]


def main():
    parser = argparse.ArgumentParser(description='AUC-augmented LoRA fine-tuning on upper-half Transformer blocks')
    parser.add_argument('--model_path', type=str,
                        default="/home/ruochen/models/limix/LimiX-16M.ckpt",
                        help='Path to pre-trained LimiX model')
    parser.add_argument('--data_path', type=str,
                        default="/home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx",
                        help='Path to training data')
    parser.add_argument('--target_cols', nargs='+', type=str,
                        default=["脑脊液漏", "神经功能加重", "严重并发症（死亡＋瘫痪＋再手术＋再入院）"],
                        help='Target column names')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[10, 20, 30, 40, 50],
                        help='Random seeds for experiments')
    parser.add_argument('--rank', type=int, default=16,
                        help='Rank for LoRA decomposition')
    parser.add_argument('--alpha', type=float, default=32.0,
                        help='Scaling factor for LoRA')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./results/lora_auc_half',
                        help='Output directory for results')
    parser.add_argument('--auc_coeff', type=float, default=0.01,
                        help='Weight for AUC surrogate loss (0 to disable)')
    parser.add_argument('--auc_start_epoch', type=int, default=15,
                        help='Epoch (1-based) to start applying AUC loss')
    parser.add_argument('--auc_gamma', type=float, default=1.0,
                        help='Gamma for exponential AUC loss')
    parser.add_argument('--target_layers', nargs='+', type=int,
                        default=DEFAULT_TARGET_LAYERS,
                        help='Transformer block indices to apply LoRA (default: upper half)')

    args = parser.parse_args()

    print("LoRA Fine-tuning with AUC Surrogate (Upper-half Blocks)")
    print("=" * 50)
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Target columns: {args.target_cols}")
    print(f"Random seeds: {args.seeds}")
    print(f"Target layers: {args.target_layers}")
    print(f"LoRA rank: {args.rank}")
    print(f"LoRA alpha: {args.alpha}")
    print(f"AUC coeff: {args.auc_coeff}, start epoch: {args.auc_start_epoch}, gamma: {args.auc_gamma}")
    print("=" * 50)

    os.makedirs(args.output_dir, exist_ok=True)

    X, y = base.load_and_preprocess_data(args.data_path, args.target_cols)

    all_results = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Running experiment with seed {seed}")
        print(f"{'='*60}")

        trainer = base.LoRATrainer(
            model_path=args.model_path,
            target_layers=args.target_layers,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            device=args.device,
            run_seed=seed,
            auc_coeff=args.auc_coeff,
            auc_start_epoch=args.auc_start_epoch,
            auc_gamma=args.auc_gamma
        )

        train_loader, val_loader = trainer.prepare_data(
            X, y, test_size=0.2, random_state=seed
        )

        seed_output_dir = os.path.join(args.output_dir, f'seed_{seed}')
        os.makedirs(seed_output_dir, exist_ok=True)

        trainer.train(train_loader, val_loader, save_dir=seed_output_dir)

        lora_path = os.path.join(seed_output_dir, 'lora_weights.pth')
        trainer.save_lora_weights(lora_path)

        if trainer.val_metrics:
            final_metrics = trainer.val_metrics[-1]
            final_metrics['seed'] = seed
            all_results.append(final_metrics)

            results_df = pd.DataFrame([final_metrics])
            results_path = os.path.join(seed_output_dir, 'results.csv')
            results_df.to_csv(results_path, index=False)

            print(f"Results for seed {seed}: {final_metrics}")

    if all_results:
        all_results_df = pd.DataFrame(all_results)

        target_name = "_".join(args.target_cols)
        output_path = os.path.join(args.output_dir, f'{target_name}_results.csv')
        summary_path = os.path.join(args.output_dir, f'{target_name}_summary.csv')

        os.makedirs(args.output_dir, exist_ok=True)

        all_results_df.to_csv(output_path, index=False)
        print(f"\n详细结果已保存至 {output_path}")

        numeric_cols = all_results_df.select_dtypes(include=[np.number]).columns
        summary = all_results_df[numeric_cols].agg(['mean', 'std']).round(4)

        print("\n性能指标汇总 (mean ± std):")
        for col in numeric_cols:
            if col != 'seed':
                mean_val = summary.loc['mean', col]
                std_val = summary.loc['std', col]
                mean_display = 0.0 if pd.isna(mean_val) else float(mean_val)
                std_display = 0.0 if pd.isna(std_val) else float(std_val)
                print(f"{col}: {mean_display:.4f}±{std_display:.4f}")

        summary_data = {}
        for col in numeric_cols:
            if col != 'seed':
                mean_val = summary.loc['mean', col]
                std_val = summary.loc['std', col]
                mean_display = 0.0 if pd.isna(mean_val) else float(mean_val)
                std_display = 0.0 if pd.isna(std_val) else float(std_val)
                summary_data[col] = [f"{mean_display:.4f}±{std_display:.4f}"]

        summary_data['num_seeds'] = [len(all_results)]
        summary_df = pd.DataFrame(summary_data)
        cols = ['num_seeds'] + [col for col in numeric_cols if col != 'seed']
        summary_df = summary_df[cols]

        summary_df.to_csv(summary_path, index=False)
        print(f"均值结果已保存至 {summary_path}")

    print("\nLoRA fine-tuning with AUC surrogate (upper-half) completed!")


if __name__ == "__main__":
    main()
