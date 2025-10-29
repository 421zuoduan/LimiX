import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import lora module from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lora import LoRAEnabledTransformer

# Import other modules
from utils.loading import load_model
from inference.predictor import LimiXPredictor


class MultiLabelDataset(Dataset):
    """Dataset for multi-label classification."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target matrix of shape (n_samples, n_labels)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(file_path: str, target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess data.
    
    Args:
        file_path: Path to Excel file
        target_cols: List of target column names
        
    Returns:
        X: Feature matrix
        y: Target matrix
    """
    print(f"Loading data from {file_path}")
    df = pd.read_excel(file_path)
    
    # Extract features and targets
    X = df.drop(columns=target_cols).values
    y = df[target_cols].values.astype(np.float32)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    print(f"Data shape - Features: {X.shape}, Targets: {y.shape}")
    return X, y


def run_lora_finetuning(
    model_path: str,
    data_path: str,
    target_cols: List[str],
    seeds: List[int],
    output_dir: str,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    num_epochs: int = 10
):
    """
    Run LoRA fine-tuning experiments.
    
    Args:
        model_path: Path to pre-trained LimiX model
        data_path: Path to training data
        target_cols: List of target column names
        seeds: List of random seeds
        output_dir: Output directory for results
        rank: Rank of LoRA decomposition
        alpha: Scaling factor for LoRA
        dropout: Dropout probability
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of epochs
    """
    print("LoRA Fine-tuning for LimiX Model")
    print("=" * 50)
    print(f"Model path: {model_path}")
    print(f"Data path: {data_path}")
    print(f"Target columns: {target_cols}")
    print(f"Random seeds: {seeds}")
    print(f"LoRA rank: {rank}")
    print(f"LoRA alpha: {alpha}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X, y = load_and_preprocess_data(data_path, target_cols)
    
    # Run experiments for each seed
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running experiment with seed {seed}")
        print(f"{'='*60}")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        
        print(f"Train set: {X_train.shape}, {y_train.shape}")
        print(f"Test set: {X_test.shape}, {y_test.shape}")
        
        # Create seed-specific output directory
        seed_output_dir = os.path.join(output_dir, f'seed_{seed}')
        os.makedirs(seed_output_dir, exist_ok=True)
        
        # Run multi-label LoRA fine-tuning for all targets together
        print(f"\n--- Training multi-label model for all targets ---")
        
        # Use all targets together for LoRA training
        try:
            # Import LoRA trainer
            from train_lora import LoRATrainer
            
            # Create LoRA trainer
            lora_trainer = LoRATrainer(
                model_path=model_path,
                target_layers=[10, 11],  # Last two layers
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Prepare data for LoRA training - use all targets together
            lora_train_loader, lora_val_loader = lora_trainer.prepare_data(
                X_train, y_train, test_size=0.2, random_state=seed
            )
            
            # Train LoRA model
            lora_trainer.train(lora_train_loader, lora_val_loader, save_dir=seed_output_dir)
            
            # Evaluate LoRA model
            lora_val_loss, lora_metrics = lora_trainer.evaluate(lora_val_loader)
            
            print(f"Multi-label LoRA metrics: {lora_metrics}")
            
            # Store LoRA results
            seed_results = {'seed': seed}
            for metric_name, metric_value in lora_metrics.items():
                seed_results[f'lora_{metric_name}'] = metric_value
            
        except Exception as e:
            print(f"Error in multi-label LoRA fine-tuning: {e}")
            # Use dummy LoRA metrics if training fails
            lora_metrics = {
                'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                'f1_micro': 0.0, 'roc_auc': 0.0, 'avg_precision': 0.0
            }
            
            # Store LoRA results
            seed_results = {'seed': seed}
            for metric_name, metric_value in lora_metrics.items():
                seed_results[f'lora_{metric_name}'] = metric_value
        
        # Save seed results
        seed_results_df = pd.DataFrame([seed_results])
        seed_results_path = os.path.join(seed_output_dir, 'results.csv')
        seed_results_df.to_csv(seed_results_path, index=False)
        
        all_results.append(seed_results)
        print(f"Results for seed {seed} saved to {seed_results_path}")
    
    # Save all results using format similar to multi_label_inference.py
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        
        # 将目标列名组合成文件名
        target_name = "_".join(target_cols)
        output_path = os.path.join(output_dir, f'{target_name}_results.csv')
        summary_path = os.path.join(output_dir, f'{target_name}_summary.csv')
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        all_results_df.to_csv(output_path, index=False)
        print(f"\n详细结果已保存至 {output_path}")
        
        # 计算平均值和标准差
        print("\n性能指标汇总 (mean ± std):")
        numeric_cols = all_results_df.select_dtypes(include=[np.number]).columns
        summary = all_results_df[numeric_cols].agg(['mean', 'std']).round(4)
        
        for col in numeric_cols:
            if col != 'seed':
                mean_val = summary.loc['mean', col]
                std_val = summary.loc['std', col]
                mean_display = 0.0 if pd.isna(mean_val) else float(mean_val)
                std_display = 0.0 if pd.isna(std_val) else float(std_val)
                print(f"{col}: {mean_display:.4f}±{std_display:.4f}")
        
        # 保存均值结果（mean±std），按 element-wise 与 subset 两行展示
        variant_rows = []
        variant_rows.append({'variant': 'elementwise'})
        num_seeds = len(all_results)

        def format_metric(col_name: str) -> str:
            if col_name not in summary.columns:
                return "-"
            mean_val = summary.loc['mean', col_name]
            std_val = summary.loc['std', col_name]
            mean_display = 0.0 if pd.isna(mean_val) else float(mean_val)
            std_display = 0.0 if pd.isna(std_val) else float(std_val)
            return f"{mean_display:.4f}±{std_display:.4f}"

        summary_rows = [
            {
                'group': 'Element-wise',
                'num_seeds': num_seeds,
                'accuracy': format_metric('lora_accuracy_elementwise'),
                'precision': format_metric('lora_precision_macro_elementwise'),
                'recall': format_metric('lora_recall_macro_elementwise'),
                'f1': format_metric('lora_f1_macro_elementwise'),
                'f1_micro': format_metric('lora_f1_micro_elementwise'),
                'roc_auc': format_metric('lora_roc_auc_macro_elementwise'),
                'avg_precision': format_metric('lora_avg_precision_macro_elementwise')
            },
            {
                'group': 'Subset/Samples',
                'num_seeds': num_seeds,
                'accuracy': format_metric('lora_accuracy_subset'),
                'precision': format_metric('lora_precision_samples_subset'),
                'recall': format_metric('lora_recall_samples_subset'),
                'f1': format_metric('lora_f1_samples_subset'),
                'f1_micro': '-',
                'roc_auc': format_metric('lora_roc_auc_samples_subset'),
                'avg_precision': format_metric('lora_avg_precision_samples_subset')
            }
        ]

        summary_df = pd.DataFrame(summary_rows)
        
        summary_df.to_csv(summary_path, index=False)
        print(f"均值结果已保存至 {summary_path}")
        print(f"注意：summary文件现在保存的是多个种子的指标平均值和标准差，而不是最高值")
    
    print("\nLoRA fine-tuning experiments completed!")
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for LimiX model')
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
    parser.add_argument('--output_dir', type=str, default='./results/lora',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run LoRA fine-tuning
    run_lora_finetuning(
        model_path=args.model_path,
        data_path=args.data_path,
        target_cols=args.target_cols,
        seeds=args.seeds,
        output_dir=args.output_dir,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )


if __name__ == "__main__":
    main()
