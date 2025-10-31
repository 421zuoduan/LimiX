#!/bin/bash

# 测试LoRA超参数搜索脚本
# 使用较小的参数范围进行快速测试

# 设置脚本在遇到错误时退出
set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "=========================================="
echo "测试 LoRA 超参数搜索脚本"
echo "=========================================="

# 创建一个临时的参数搜索脚本，使用较小的参数范围
cat > train/lora/test_hyperparameter_search.py << 'EOF'
import os
import sys
import argparse
import json
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
import itertools
from datetime import datetime
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
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LoRATrainer:
    """Trainer for LoRA fine-tuning of LimiX model."""
    
    def __init__(
        self,
        model_path: str,
        target_layers: List[int] = [10, 11],
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        num_epochs: int = 2,  # 减少训练轮数用于测试
        device: str = 'cuda'
    ):
        self.model_path = model_path
        self.target_layers = target_layers
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load base model
        self.base_model = load_model(model_path)
        
        # Create LoRA-enabled model
        self.model = LoRAEnabledTransformer(
            base_model=self.base_model,
            target_layers=target_layers,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        ).to(self.device)
        
        # Setup optimizer for LoRA parameters only
        self.optimizer = optim.AdamW(
            self.model.get_lora_parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        self.test_ratio = 0.2
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
    
    def prepare_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create datasets
        train_dataset = MultiLabelDataset(X_train, y_train)
        val_dataset = MultiLabelDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Prepare data for LimiX model
            batch_size = X_batch.size(0)
            n_features = X_batch.size(1)
            n_labels = y_batch.size(1)
            
            # For simplicity, we'll use a simplified approach
            total_batch_loss = 0.0
            valid_forward_passes = 0
            
            for label_idx in range(n_labels):
                # Prepare data for single label
                y_single = y_batch[:, label_idx:label_idx+1]
                
                # Create sequence dimension
                X_seq = X_batch.unsqueeze(0)  # [1, batch, features]
                y_for_model = y_single.squeeze(-1).unsqueeze(0)  # [1, batch]
                
                # Split batch into pseudo train/test segments
                eval_pos = max(1, int((1 - self.test_ratio) * X_seq.shape[1]))
                eval_pos = min(eval_pos, X_seq.shape[1] - 1)
                test_len = X_seq.shape[1] - eval_pos
                if test_len <= 0:
                    continue
                
                try:
                    # Forward pass through model with mixed precision
                    with torch.cuda.amp.autocast():
                        output = self.model(
                            x=X_seq,
                            y=y_for_model,
                            eval_pos=eval_pos,
                            task_type='cls'
                        )
                    
                    # Extract classification output
                    if isinstance(output, dict):
                        cls_output = output.get('cls_output', output)
                    else:
                        cls_output = output
                    
                    # Calculate loss
                    if cls_output.dim() > 2:
                        cls_output = cls_output.squeeze(0)
                    
                    # Retain only the logits for the pseudo test portion
                    cls_output = cls_output[:test_len]
                    y_target = y_for_model[:, eval_pos:eval_pos + test_len].squeeze(0)
                    
                    # Ensure shapes match
                    min_len = min(cls_output.shape[0], y_target.shape[0])
                    cls_output = cls_output[:min_len]
                    y_target = y_target[:min_len]
                    
                    # Ensure target is float for BCEWithLogitsLoss
                    y_target = y_target.float()
                    
                    # Collapse logits to the positive class for binary targets
                    if cls_output.shape[-1] >= 2:
                        logits = cls_output[:, 1].float()
                    else:
                        logits = cls_output.squeeze(-1).float()
                    
                    loss = self.criterion(logits, y_target)
                    
                    # Scale loss by number of labels to maintain similar magnitude
                    scaled_loss = loss / n_labels
                    
                    # Backward pass
                    scaled_loss.backward()
                    total_batch_loss += loss.item()
                    valid_forward_passes += 1
                    
                except Exception as e:
                    print(f"Error in forward pass for label {label_idx}: {e}")
                    continue
            
            # Update weights with mixed precision - only if we had valid forward passes
            if valid_forward_passes > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += total_batch_loss / n_labels
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {total_batch_loss / n_labels:.4f}")
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                batch_size = X_batch.size(0)
                n_labels = y_batch.size(1)
                
                batch_predictions = []
                batch_targets = []
                batch_loss = 0.0
                
                for label_idx in range(n_labels):
                    # Prepare data for single label
                    y_single = y_batch[:, label_idx:label_idx+1]
                    
                    # Create sequence dimension
                    X_seq = X_batch.unsqueeze(0)  # [1, batch, features]
                    y_for_model = y_single.squeeze(-1).unsqueeze(0)  # [1, batch]
                    
                    try:
                        eval_pos = max(1, min(int((1 - self.test_ratio) * X_seq.shape[1]), X_seq.shape[1] - 1))
                        with torch.cuda.amp.autocast():
                            output = self.model(
                                x=X_seq,
                                y=y_for_model,
                                eval_pos=eval_pos,
                                task_type='cls'
                            )
                        test_len = X_seq.shape[1] - eval_pos
                        if test_len <= 0:
                            continue
                            
                        # Extract classification output
                        if isinstance(output, dict):
                            cls_output = output.get('cls_output', output)
                        else:
                            cls_output = output
                        
                        if cls_output.dim() > 2:
                            cls_output = cls_output.squeeze(0)
                        
                        cls_output = cls_output[:test_len]
                        
                        # Get the target labels
                        y_target = y_for_model[:, eval_pos:eval_pos + test_len].squeeze(0)  # [test_len]
                        
                        # Convert target to float for BCEWithLogitsLoss
                        y_target = y_target.float()
                        
                        min_len = min(cls_output.shape[0], y_target.shape[0])
                        cls_output = cls_output[:min_len]
                        y_target = y_target[:min_len]
                        
                        if cls_output.shape[-1] >= 2:
                            logits = cls_output[:, 1].float()
                            pred = torch.softmax(cls_output[:, :2].float(), dim=1)[:, 1].detach().cpu().numpy()
                        else:
                            logits = cls_output.squeeze(-1).float()
                            pred = torch.sigmoid(logits).detach().cpu().numpy()
                        
                        loss = self.criterion(logits, y_target)
                        batch_loss += loss.item()
                        batch_predictions.append(pred)
                        batch_targets.append(y_target.detach().cpu().numpy())
                        
                    except Exception as e:
                        print(f"Error in evaluation for label {label_idx}: {e}")
                        # Use dummy predictions
                        batch_predictions.append(np.zeros(eval_pos if 'eval_pos' in locals() else batch_size))
                        batch_targets.append(np.zeros(eval_pos if 'eval_pos' in locals() else batch_size))
                        continue
                
                total_loss += batch_loss / n_labels
                
                # Combine predictions for all labels
                if batch_predictions:
                    min_pred_len = min(len(arr) for arr in batch_predictions)
                    preds = np.stack([arr[:min_pred_len] for arr in batch_predictions], axis=1)
                    targs = np.stack([arr[:min_pred_len] for arr in batch_targets], axis=1)
                    all_predictions.append(preds)
                    all_targets.append(targs)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        
        if all_predictions and all_targets:
            all_predictions = np.vstack(all_predictions)
            all_targets = np.vstack(all_targets)
            
            # Convert predictions to binary
            binary_predictions = (all_predictions >= 0.5).astype(int)
            
            # Calculate metrics for element-wise (per-label) and subset/sample perspectives
            subset_accuracy = accuracy_score(all_targets, binary_predictions)
            elementwise_accuracy = float((binary_predictions == all_targets).mean())

            def safe_metric(func, *args, default=np.nan, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as err:
                    print(f"{func.__name__}计算失败: {err}")
                    return default

            precision_macro = safe_metric(
                precision_score, all_targets, binary_predictions, average='macro', zero_division=0
            )
            precision_samples = safe_metric(
                precision_score, all_targets, binary_predictions, average='samples', zero_division=0
            )
            recall_macro = safe_metric(
                recall_score, all_targets, binary_predictions, average='macro', zero_division=0
            )
            recall_samples = safe_metric(
                recall_score, all_targets, binary_predictions, average='samples', zero_division=0
            )
            f1_macro = safe_metric(
                f1_score, all_targets, binary_predictions, average='macro', zero_division=0
            )
            f1_micro = safe_metric(
                f1_score, all_targets, binary_predictions, average='micro', zero_division=0
            )
            f1_samples = safe_metric(
                f1_score, all_targets, binary_predictions, average='samples', zero_division=0
            )
            roc_auc_macro = safe_metric(
                roc_auc_score, all_targets, all_predictions, average='macro'
            )
            roc_auc_samples = safe_metric(
                roc_auc_score, all_targets, all_predictions, average='samples'
            )
            avg_precision_macro = safe_metric(
                average_precision_score, all_targets, all_predictions, average='macro'
            )
            avg_precision_samples = safe_metric(
                average_precision_score, all_targets, all_predictions, average='samples'
            )

            metrics = {
                'accuracy_subset': subset_accuracy,
                'accuracy_elementwise': elementwise_accuracy,
                'precision_macro_elementwise': precision_macro,
                'precision_samples_subset': precision_samples,
                'recall_macro_elementwise': recall_macro,
                'recall_samples_subset': recall_samples,
                'f1_macro_elementwise': f1_macro,
                'f1_micro_elementwise': f1_micro,
                'f1_samples_subset': f1_samples,
                'roc_auc_macro_elementwise': roc_auc_macro,
                'roc_auc_samples_subset': roc_auc_samples,
                'avg_precision_macro_elementwise': avg_precision_macro,
                'avg_precision_samples_subset': avg_precision_samples
            }
        else:
            metrics = {
                'accuracy_subset': 0.0,
                'accuracy_elementwise': 0.0,
                'precision_macro_elementwise': 0.0,
                'precision_samples_subset': 0.0,
                'recall_macro_elementwise': 0.0,
                'recall_samples_subset': 0.0,
                'f1_macro_elementwise': 0.0,
                'f1_micro_elementwise': 0.0,
                'f1_samples_subset': 0.0,
                'roc_auc_macro_elementwise': 0.0,
                'roc_auc_samples_subset': 0.0,
                'avg_precision_macro_elementwise': 0.0,
                'avg_precision_samples_subset': 0.0
            }
        
        return avg_loss, metrics
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        save_dir: str = './checkpoints'
    ):
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            if 'accuracy_subset' in val_metrics:
                print(f"Val Accuracy (subset): {val_metrics['accuracy_subset']:.4f}")
            if 'accuracy_elementwise' in val_metrics:
                print(f"Val Accuracy (element-wise): {val_metrics['accuracy_elementwise']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, 'best_lora_model.pth')
                self.save_checkpoint(best_model_path, epoch, val_loss, val_metrics)
                print(f"Saved best model to {best_model_path}")
                
                # Also save LoRA-only weights alongside the best checkpoint
                lora_only_path = os.path.join(save_dir, 'best_lora_weights.pth')
                self.save_lora_weights(lora_only_path)
                print(f"Saved LoRA weights to {lora_only_path}")
    
    def save_checkpoint(
        self, 
        path: str, 
        epoch: int, 
        val_loss: float, 
        val_metrics: Dict[str, float]
    ):
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'target_layers': self.target_layers,
                'rank': self.rank,
                'alpha': self.alpha,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }
        }
        torch.save(checkpoint, path)
    
    def save_lora_weights(self, path: str):
        self.model.save_lora_weights(path)
        print(f"LoRA weights saved to {path}")


def load_and_preprocess_data(file_path: str, target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(file_path)
    
    # Extract features and targets
    X = df.drop(columns=target_cols).values
    y = df[target_cols].values.astype(np.float32)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    return X, y


def run_test_hyperparameter_search(
    model_path: str,
    data_path: str,
    target_cols: List[str],
    seeds: List[int],
    output_dir: str
):
    print("LoRA 测试超参数搜索 for LimiX Model")
    print("=" * 50)
    print(f"Model path: {model_path}")
    print(f"Data path: {data_path}")
    print(f"Target columns: {target_cols}")
    print(f"Random seeds: {seeds}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X, y = load_and_preprocess_data(data_path, target_cols)
    
    # 使用较小的参数范围进行测试
    param_grid = {
        'rank': [8, 16],  # 2个选项
        'dropout': [0.0, 0.1],  # 2个选项
        'learning_rate': [1e-4, 5e-4],  # 2个选项
        'batch_size': [64],  # 1个选项
        'num_epochs': [2]  # 只训练2轮用于测试
    }
    # 测试总组合数: 2 × 2 × 2 × 1 × 1 = 8个组合
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Total parameter combinations to test: {len(param_combinations)}")
    
    # Store all results
    all_search_results = []
    
    # Iterate over all parameter combinations
    for i, param_combo in enumerate(param_combinations):
        print(f"\n{'='*80}")
        print(f"Testing parameter combination {i+1}/{len(param_combinations)}")
        print(f"{'='*80}")
        
        # Create parameter dictionary
        params = dict(zip(param_names, param_combo))
        print(f"Parameters: {params}")
        
        # Run experiments for each seed with current parameters
        seed_results = []
        
        for seed in seeds[:2]:  # 只使用前2个种子进行测试
            print(f"\n--- Running with seed {seed} ---")
            
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            
            print(f"Train set: {X_train.shape}, {y_train.shape}")
            print(f"Test set: {X_test.shape}, {y_test.shape}")
            
            # Create parameter-specific output directory
            param_str = "_".join([f"{k}={v}" for k, v in params.items()])
            param_str = param_str.replace("/", "_").replace(" ", "_")
            seed_output_dir = os.path.join(output_dir, f'combo_{i+1}', f'seed_{seed}')
            os.makedirs(seed_output_dir, exist_ok=True)
            
            try:
                # Create LoRA trainer with current parameters
                rank_value = params.get('rank', 16)
                trainer = LoRATrainer(
                    model_path=model_path,
                    target_layers=[10, 11],  # Last two layers
                    rank=rank_value,
                    alpha=rank_value * 2,  # 自动设置为 rank 的两倍
                    dropout=params.get('dropout', 0.1),
                    learning_rate=params.get('learning_rate', 1e-4),
                    batch_size=params.get('batch_size', 64),
                    num_epochs=params.get('num_epochs', 2),
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # Prepare data
                train_loader, val_loader = trainer.prepare_data(
                    X_train, y_train, test_size=0.2, random_state=seed
                )
                
                # Train model
                trainer.train(train_loader, val_loader, save_dir=seed_output_dir)
                
                # Get final metrics
                if trainer.val_metrics:
                    final_metrics = trainer.val_metrics[-1]
                    final_metrics['seed'] = seed
                    seed_results.append(final_metrics)
                    
                    # Save results
                    results_df = pd.DataFrame([final_metrics])
                    results_path = os.path.join(seed_output_dir, 'results.csv')
                    results_df.to_csv(results_path, index=False)
                    
                    print(f"Results for seed {seed}: {final_metrics}")
                else:
                    print(f"No metrics available for seed {seed}")
                    
            except Exception as e:
                print(f"Error in training with seed {seed}: {e}")
                continue
        
        # Calculate average metrics across seeds for this parameter combination
        if seed_results:
            # Convert to DataFrame for easier calculation
            seed_results_df = pd.DataFrame(seed_results)
            
            # Calculate mean and std for each metric
            avg_metrics = {}
            for col in seed_results_df.columns:
                if col != 'seed':
                    avg_metrics[f'{col}_mean'] = seed_results_df[col].mean()
                    avg_metrics[f'{col}_std'] = seed_results_df[col].std()
            
            # Add parameter information
            for param_name, param_value in params.items():
                avg_metrics[param_name] = param_value
            
            # Add number of seeds
            avg_metrics['num_seeds'] = len(seed_results)
            
            # Add combination index
            avg_metrics['combo_index'] = i + 1
            
            all_search_results.append(avg_metrics)
            
            # Print summary for this parameter combination
            print(f"\n--- Summary for parameter combination {i+1} ---")
            print(f"Parameters: {params}")
            if 'accuracy_subset_mean' in avg_metrics:
                print(f"Average Subset Accuracy: {avg_metrics['accuracy_subset_mean']:.4f} ± {avg_metrics['accuracy_subset_std']:.4f}")
            if 'accuracy_elementwise_mean' in avg_metrics:
                print(f"Average Element-wise Accuracy: {avg_metrics['accuracy_elementwise_mean']:.4f} ± {avg_metrics['accuracy_elementwise_std']:.4f}")
    
    # Save all search results
    if all_search_results:
        all_results_df = pd.DataFrame(all_search_results)
        
        # 将目标列名组合成文件名
        target_name = "_".join(target_cols)
        search_results_path = os.path.join(output_dir, f'{target_name}_test_hyperparameter_search_results.csv')
        
        # 保存详细结果
        all_results_df.to_csv(search_results_path, index=False)
        print(f"\n测试超参数搜索结果已保存至 {search_results_path}")
        
        # Find best parameters based on different metrics
        best_subset_acc = all_results_df.loc[all_results_df['accuracy_subset_mean'].idxmax()]
        best_elementwise_acc = all_results_df.loc[all_results_df['accuracy_elementwise_mean'].idxmax()]
        
        print("\n" + "="*80)
        print("测试最佳参数组合")
        print("="*80)
        print(f"\n基于最佳 Subset Accuracy:")
        print(f"  参数: {dict([(k, best_subset_acc[k]) for k in param_names if k in best_subset_acc])}")
        print(f"  Subset Accuracy: {best_subset_acc['accuracy_subset_mean']:.4f} ± {best_subset_acc['accuracy_subset_std']:.4f}")
        
        print(f"\n基于最佳 Element-wise Accuracy:")
        print(f"  参数: {dict([(k, best_elementwise_acc[k]) for k in param_names if k in best_elementwise_acc])}")
        print(f"  Element-wise Accuracy: {best_elementwise_acc['accuracy_elementwise_mean']:.4f} ± {best_elementwise_acc['accuracy_elementwise_std']:.4f}")
    
    print("\nLoRA 测试超参数搜索完成!")
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='LoRA test hyperparameter search for LimiX model')
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
    parser.add_argument('--output_dir', type=str, default='./results/lora_test_hyperparameter_search',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run test hyperparameter search
    run_test_hyperparameter_search(
        model_path=args.model_path,
        data_path=args.data_path,
        target_cols=args.target_cols,
        seeds=args.seeds,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
EOF

echo "开始测试 LoRA 超参数搜索..."

# 运行测试
python train/lora/test_hyperparameter_search.py --seeds 10 20

echo "测试完成！"

# 清理测试文件
rm train/lora/test_hyperparameter_search.py

echo "=========================================="
echo "测试成功完成！"
echo "=========================================="