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
from typing import Dict, List, Tuple, Any, Optional
import warnings
import itertools
from datetime import datetime
warnings.filterwarnings('ignore')

DEFAULT_TARGET_LAYERS = [6, 7, 8, 9, 10, 11]

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


class LoRATrainer:
    """Trainer for LoRA fine-tuning of LimiX model."""
    
    def __init__(
        self,
        model_path: str,
        target_layers: List[int] = [6, 7, 8, 9, 10, 11],
        rank: int = 16,
        alpha: float = None,  # 将自动设置为 rank 的两倍
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        batch_size: int = 64,  # 增大默认 batch size
        num_epochs: int = 10,
        device: str = 'cuda',
        run_seed: Optional[int] = None
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model_path: Path to pre-trained LimiX model
            target_layers: List of layer indices to apply LoRA to
            rank: Rank of low-rank decomposition
            alpha: Scaling factor for LoRA
            dropout: Dropout probability
            learning_rate: Learning rate for LoRA parameters
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            device: Device to use for training
            run_seed: Seed identifier for the current run (stored in checkpoint metadata)
        """
        self.model_path = model_path
        self.target_layers = target_layers
        self.rank = rank
        # alpha 自动设置为 rank 的两倍
        self.alpha = alpha if alpha is not None else rank * 2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.run_seed = run_seed
        
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
        self.test_ratio = 0.2  # portion of each mini-batch reserved as pseudo test samples
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.metric_trackers = {
            'val_loss': {
                'best': float('inf'),
                'mode': 'min',
                'base_name': 'best_lora',
                'display': 'Val Loss'
            },
            'accuracy_samples': {
                'best': -float('inf'),
                'mode': 'max',
                'base_name': 'best_accuracy_samples',
                'display': 'Samples Accuracy'
            },
            'accuracy_macro': {
                'best': -float('inf'),
                'mode': 'max',
                'base_name': 'best_accuracy_macro',
                'display': 'Macro Accuracy'
            },
            'accuracy_micro': {
                'best': -float('inf'),
                'mode': 'max',
                'base_name': 'best_accuracy_micro',
                'display': 'Micro Accuracy'
            },
            'f1_macro': {
                'best': -float('inf'),
                'mode': 'max',
                'base_name': 'best_f1_macro',
                'display': 'Macro F1'
            },
            'f1_micro': {
                'best': -float('inf'),
                'mode': 'max',
                'base_name': 'best_f1_micro',
                'display': 'Micro F1'
            },
            'f1_samples': {
                'best': -float('inf'),
                'mode': 'max',
                'base_name': 'best_f1_samples',
                'display': 'Samples F1'
            },
            'roc_auc_macro': {
                'best': -float('inf'),
                'mode': 'max',
                'base_name': 'best_roc_auc_macro',
                'display': 'Macro ROC-AUC'
            },
            'roc_auc_micro': {
                'best': -float('inf'),
                'mode': 'max',
                'base_name': 'best_roc_auc_micro',
                'display': 'Micro ROC-AUC'
            },
        }
    
    def prepare_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare train and validation data loaders.
        
        Args:
            X: Feature matrix
            y: Target matrix
            test_size: Proportion of data for validation
            random_state: Random seed for splitting
            
        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
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
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
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
        """
        Evaluate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss and metrics
        """
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
            
            def safe_metric(func, *args, default=np.nan, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as err:
                    print(f"{func.__name__}计算失败: {err}")
                    return default

            # ROC-AUC 指标
            roc_auc_macro = safe_metric(
                roc_auc_score, all_targets, all_predictions, average='macro'
            )
            roc_auc_micro = safe_metric(
                roc_auc_score, all_targets, all_predictions, average='micro'
            )
            
            # F1 Score (使用0.5阈值)
            f1_macro = safe_metric(
                f1_score, all_targets, binary_predictions, average='macro', zero_division=0
            )
            f1_micro = safe_metric(
                f1_score, all_targets, binary_predictions, average='micro', zero_division=0
            )
            f1_samples = safe_metric(
                f1_score, all_targets, binary_predictions, average='samples', zero_division=0
            )
            
            # Accuracy
            accuracy_macro = (binary_predictions == all_targets).mean()
            accuracy_micro = (binary_predictions == all_targets).mean()  # 对于多标签，micro和macro计算方式相同
            accuracy_samples = (binary_predictions == all_targets).all(axis=1).mean()

            metrics = {
                'accuracy_samples': accuracy_samples,
                'accuracy_macro': accuracy_macro,
                'accuracy_micro': accuracy_micro,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'f1_samples': f1_samples,
                'roc_auc_macro': roc_auc_macro,
                'roc_auc_micro': roc_auc_micro,
            }
        else:
            metrics = {
                'accuracy_samples': 0.0,
                'accuracy_macro': 0.0,
                'accuracy_micro': 0.0,
                'f1_macro': 0.0,
                'f1_micro': 0.0,
                'f1_samples': 0.0,
                'roc_auc_macro': 0.0,
                'roc_auc_micro': 0.0,
            }
        
        return avg_loss, metrics

    def _is_finite_number(self, value) -> bool:
        if value is None:
            return False
        if isinstance(value, (float, int, np.floating, np.integer)):
            return np.isfinite(value)
        return False

    def _update_best_models(
        self,
        save_dir: str,
        epoch_idx: int,
        val_loss: float,
        val_metrics: Dict[str, float]
    ):
        if not save_dir:
            return

        metrics_to_check = {'val_loss': val_loss}
        metrics_to_check.update({
            'accuracy_samples': val_metrics.get('accuracy_samples'),
            'accuracy_macro': val_metrics.get('accuracy_macro'),
            'accuracy_micro': val_metrics.get('accuracy_micro'),
            'f1_macro': val_metrics.get('f1_macro'),
            'f1_micro': val_metrics.get('f1_micro'),
            'f1_samples': val_metrics.get('f1_samples'),
            'roc_auc_macro': val_metrics.get('roc_auc_macro'),
            'roc_auc_micro': val_metrics.get('roc_auc_micro'),
        })

        for metric_key, current_value in metrics_to_check.items():
            tracker = self.metric_trackers.get(metric_key)
            if tracker is None or not self._is_finite_number(current_value):
                continue

            better = (
                current_value < tracker['best']
                if tracker['mode'] == 'min'
                else current_value > tracker['best']
            )
            if not better:
                continue

            tracker['best'] = float(current_value)
            base_name = tracker['base_name']
            metric_label = tracker['display']

            if metric_key == 'val_loss':
                model_path = os.path.join(save_dir, f'{base_name}_model.pth')
                weights_path = os.path.join(save_dir, f'{base_name}_weights.pth')
                self.save_checkpoint(model_path, epoch_idx, val_loss, val_metrics)
                self.model.save_lora_weights(weights_path)
                meta_path = os.path.join(save_dir, f'{base_name}_metrics.json')
            else:
                model_path = None
                weights_path = os.path.join(save_dir, f'{base_name}_weights.pth')
                meta_path = os.path.join(save_dir, f'{base_name}_metrics.json')
                self.model.save_lora_weights(weights_path)

            val_loss_value = float(val_loss) if self._is_finite_number(val_loss) else None
            metrics_snapshot = {}
            for key, value in val_metrics.items():
                if self._is_finite_number(value):
                    metrics_snapshot[key] = float(value)
                else:
                    metrics_snapshot[key] = None

            metadata = {
                'metric_key': metric_key,
                'metric_display': metric_label,
                'metric_value': float(current_value),
                'epoch': int(epoch_idx + 1),
                'val_loss_at_save': val_loss_value,
                'seed': int(self.run_seed) if self.run_seed is not None else None,
                'checkpoint_path': model_path,
                'weights_path': weights_path,
                'val_metrics': metrics_snapshot
            }
            with open(meta_path, 'w', encoding='utf-8') as meta_file:
                json.dump(metadata, meta_file, ensure_ascii=False, indent=2)
            print(f"Saved new best {metric_label} ({current_value:.4f}) at epoch {epoch_idx + 1} -> {weights_path}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str = None,
        save_intermediate_epochs: List[int] = None,
        capture_epochs: List[int] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            save_intermediate_epochs: Epochs to persist checkpoints on disk
            capture_epochs: Epochs to capture metrics/weights in memory
        """
        captured_snapshots: Dict[int, Dict[str, Any]] = {}
        save_epochs = set(save_intermediate_epochs or [])
        capture_epochs = set(capture_epochs or [])
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
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
            if 'accuracy_samples' in val_metrics:
                print(f"Val Accuracy (samples): {val_metrics['accuracy_samples']:.4f}")
            if 'accuracy_macro' in val_metrics:
                print(f"Val Accuracy (macro): {val_metrics['accuracy_macro']:.4f}")
            if 'accuracy_micro' in val_metrics:
                print(f"Val Accuracy (micro): {val_metrics['accuracy_micro']:.4f}")
            if save_dir:
                self._update_best_models(save_dir, epoch, val_loss, val_metrics)
            
            current_epoch = epoch + 1
            
            # Capture snapshots for requested epochs
            if current_epoch in capture_epochs:
                captured_snapshots[current_epoch] = {
                    'val_loss': float(val_loss),
                    'val_metrics': val_metrics.copy(),
                    'lora_state_dict': self.model.get_lora_state_dict()
                }
            
            # Save intermediate checkpoints if required
            if save_dir and current_epoch in save_epochs:
                intermediate_model_path = os.path.join(save_dir, f'lora_model_epoch_{current_epoch}.pth')
                intermediate_weights_path = os.path.join(save_dir, f'lora_weights_epoch_{current_epoch}.pth')
                
                self.save_checkpoint(intermediate_model_path, epoch, val_loss, val_metrics)
                self.save_lora_weights(intermediate_weights_path)
                print(f"Saved intermediate checkpoint at epoch {current_epoch} to {intermediate_model_path}")
        
        return captured_snapshots
    
    def save_checkpoint(
        self, 
        path: str, 
        epoch: int, 
        val_loss: float, 
        val_metrics: Dict[str, float]
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            val_loss: Validation loss
            val_metrics: Validation metrics
        """
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
        """
        Save only LoRA weights.
        
        Args:
            path: Path to save LoRA weights
        """
        self.model.save_lora_weights(path)
        print(f"LoRA weights saved to {path}")


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
    df = pd.read_excel(file_path)
    
    # Extract features and targets
    X = df.drop(columns=target_cols).values
    y = df[target_cols].values.astype(np.float32)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    return X, y


def aggregate_epoch_metrics(
    metrics_by_epoch: Dict[int, List[Dict[str, float]]],
    losses_by_epoch: Dict[int, List[float]]
) -> Dict[int, Dict[str, float]]:
    """
    Aggregate metrics collected at specific epochs across seeds.
    
    Args:
        metrics_by_epoch: Mapping from epoch -> list of metric dicts (per seed)
        losses_by_epoch: Mapping from epoch -> list of validation losses (per seed)
        
    Returns:
        Dictionary mapping epoch numbers to aggregated statistics (mean/std)
    """
    aggregated_results: Dict[int, Dict[str, float]] = {}
    
    for epoch, metrics_list in metrics_by_epoch.items():
        if not metrics_list:
            continue
        
        epoch_df = pd.DataFrame(metrics_list)
        avg_metrics: Dict[str, float] = {}
        
        for col in epoch_df.columns:
            avg_metrics[f'{col}_mean'] = float(epoch_df[col].mean())
            avg_metrics[f'{col}_std'] = float(epoch_df[col].std())
        
        losses = losses_by_epoch.get(epoch, [])
        if losses:
            avg_metrics['val_loss_mean'] = float(np.mean(losses))
            avg_metrics['val_loss_std'] = float(np.std(losses))
        
        avg_metrics['num_seeds'] = len(metrics_list)
        aggregated_results[epoch] = avg_metrics
    
    return aggregated_results


def select_best_epoch_for_combination(intermediate_results: Dict[int, Dict[str, float]], primary_metric: str = 'val_loss_mean') -> Tuple[int, Dict[str, float]]:
    """
    为参数组合选择最佳epoch
    
    Args:
        intermediate_results: 字典，包含各epoch的平均指标
        primary_metric: 主要评估指标名称（默认为val_loss，越小越好）
        
    Returns:
        best_epoch: 最佳epoch编号
        best_metrics: 最佳epoch的指标
    """
    best_epoch = None
    best_score = float('inf')  # 对于loss，越小越好
    best_metrics = None
    
    for epoch, metrics in intermediate_results.items():
        # 使用val_loss作为主要评估指标（越小越好）
        current_score = metrics.get(primary_metric, float('inf'))
        
        if current_score < best_score:
            best_score = current_score
            best_epoch = epoch
            best_metrics = metrics
    
    return best_epoch, best_metrics


def update_global_best_model(
    current_best: Dict,
    new_combination_metrics: Dict[str, float],
    new_combination_params: Dict,
    new_combination_weights_record: Dict[str, Any],
    primary_metric: str = 'val_loss_mean',
    secondary_metric: str = 'accuracy_samples_mean'
) -> Tuple[bool, Dict]:
    """
    更新全局最佳模型
    
    Args:
        current_best: 当前最佳模型信息
        new_combination_metrics: 新参数组合的最佳指标
        new_combination_params: 新参数组合的参数
        new_combination_weights: 新参数组合的LoRA权重（最佳epoch）
        primary_metric: 主要评估指标名称（val_loss，越小越好）
        secondary_metric: 次要评估指标名称（accuracy，越大越好）
        
    Returns:
        should_update: 是否应该更新全局最佳模型
        updated_best: 更新后的最佳模型信息
    """
    new_weights_state = new_combination_weights_record.get('weights', {})
    new_seed = new_combination_weights_record.get('seed')

    if current_best is None:
        cloned_weights = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in new_weights_state.items()
        }
        return True, {
            'metrics': dict(new_combination_metrics),
            'params': dict(new_combination_params),
            'weights': cloned_weights,
            'combo_index': new_combination_params.get('combo_index'),
            'seed': new_seed
        }
    
    # 首先比较主要指标（val_loss，越小越好）
    current_score = current_best['metrics'].get(primary_metric, float('inf'))
    new_score = new_combination_metrics.get(primary_metric, float('inf'))
    
    if new_score < current_score:
        cloned_weights = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in new_weights_state.items()
        }
        return True, {
            'metrics': dict(new_combination_metrics),
            'params': dict(new_combination_params),
            'weights': cloned_weights,
            'combo_index': new_combination_params.get('combo_index'),
            'seed': new_seed
        }
    elif abs(new_score - current_score) < 1e-6:  # loss 相近时比较次要指标
        # 如果主要指标相同，比较次要指标（accuracy，越大越好）
        current_secondary = current_best['metrics'].get(secondary_metric, 0)
        new_secondary = new_combination_metrics.get(secondary_metric, 0)
        
        if new_secondary > current_secondary:
            cloned_weights = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in new_weights_state.items()
            }
            return True, {
                'metrics': dict(new_combination_metrics),
                'params': dict(new_combination_params),
                'weights': cloned_weights,
                'combo_index': new_combination_params.get('combo_index'),
                'seed': new_seed
            }
    
    return False, current_best


def save_global_best_model(global_best_info: Dict, output_dir: str, target_cols: List[str]):
    """
    保存全局最佳模型到统一目录
    
    Args:
        global_best_info: 全局最佳模型信息
        output_dir: 输出目录
        target_cols: 目标列名列表
    """
    # 创建全局最佳模型目录
    global_best_dir = os.path.join(output_dir, 'global_best_models')
    os.makedirs(global_best_dir, exist_ok=True)
    
    # 生成目标名称
    target_name = "_".join(target_cols)
    target_weights_path = os.path.join(global_best_dir, f'{target_name}_best_weights.pth')
    metadata_path = os.path.join(global_best_dir, f'{target_name}_best_metadata.json')
    
    try:
        torch.save(global_best_info['weights'], target_weights_path)
        print(f"全局最佳LoRA权重已保存至: {target_weights_path}")
        
        metadata = {
            'best_metrics': global_best_info['metrics'],
            'best_params': global_best_info['params'],
            'weights_path': target_weights_path,
            'combo_index': global_best_info['combo_index'],
            'seed': global_best_info.get('seed')
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"全局最佳模型元数据已保存至: {metadata_path}")
    except Exception as e:
        print(f"保存全局最佳模型时出错: {e}")


def run_hyperparameter_search(
    model_path: str,
    data_path: str,
    target_cols: List[str],
    seeds: List[int],
    output_dir: str,
    param_grid: Dict[str, List],
    target_layers: List[int] = DEFAULT_TARGET_LAYERS
):
    """
    Run hyperparameter search for LoRA fine-tuning.
    
    Args:
        model_path: Path to pre-trained LimiX model
        data_path: Path to training data
        target_cols: List of target column names
        seeds: List of random seeds
        output_dir: Output directory for results
        param_grid: Dictionary of parameter grids to search
    """
    print("LoRA Hyperparameter Search for LimiX Model")
    print("=" * 50)
    print(f"Model path: {model_path}")
    print(f"Data path: {data_path}")
    print(f"Target columns: {target_cols}")
    print(f"Random seeds: {seeds}")
    print(f"Parameter grid: {param_grid}")
    print(f"Target layers: {target_layers}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    global_best_info = None
    
    # Load data
    X, y = load_and_preprocess_data(data_path, target_cols)
    
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
        
        # Prepare directory to store aggregation summaries (lightweight)
        combination_dir = os.path.join(output_dir, f'combo_{i+1}')
        os.makedirs(combination_dir, exist_ok=True)
        
        # Determine which epochs to evaluate for this parameter combination
        num_epochs = params.get('num_epochs', 30)
        candidate_epochs = [epoch for epoch in (10, 20, 30) if epoch <= num_epochs]
        if num_epochs not in candidate_epochs:
            candidate_epochs.append(num_epochs)
        intermediate_epochs = sorted(set(candidate_epochs))
        if not intermediate_epochs:
            intermediate_epochs = [num_epochs]
        
        # Containers for metrics across seeds
        metrics_by_epoch: Dict[int, List[Dict[str, float]]] = {epoch: [] for epoch in intermediate_epochs}
        losses_by_epoch: Dict[int, List[float]] = {epoch: [] for epoch in intermediate_epochs}
        epoch_best_states: Dict[int, Dict[str, torch.Tensor]] = {}
        epoch_best_losses: Dict[int, float] = {}
        
        # Run experiments for each seed with current parameters
        for seed in seeds:
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
            
            try:
                # Create LoRA trainer with current parameters
                # alpha 自动设置为 rank 的两倍
                rank_value = params.get('rank', 16)
                trainer = LoRATrainer(
                    model_path=model_path,
                    target_layers=target_layers,
                    rank=rank_value,
                    alpha=rank_value * 2,  # alpha 设置为 rank 的两倍
                    dropout=params.get('dropout', 0.1),
                    learning_rate=params.get('learning_rate', 1e-4),
                    batch_size=params.get('batch_size', 64),  # 增大默认 batch size
                    num_epochs=params.get('num_epochs', 30),
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    run_seed=seed
                )
                
                # Prepare data
                train_loader, val_loader = trainer.prepare_data(
                    X_train, y_train, test_size=0.2, random_state=seed
                )
                
                # Train model while capturing key epochs in memory
                snapshots = trainer.train(
                    train_loader,
                    val_loader,
                    save_dir=None,
                    save_intermediate_epochs=None,
                    capture_epochs=intermediate_epochs
                )
                
                # Record final metrics for bookkeeping
                if trainer.val_metrics:
                    final_metrics = trainer.val_metrics[-1]
                    final_metrics['seed'] = seed
                    print(f"Results for seed {seed}: {final_metrics}")
                else:
                    print(f"No metrics available for seed {seed}")
                
                # Collect intermediate epoch metrics and best states
                for epoch in intermediate_epochs:
                    snapshot = snapshots.get(epoch)
                    if snapshot is None:
                        continue
                    
                    metrics_copy = snapshot['val_metrics'].copy()
                    metrics_by_epoch[epoch].append(metrics_copy)
                    losses_by_epoch[epoch].append(snapshot['val_loss'])
                    
                    best_loss = epoch_best_losses.get(epoch)
                    if best_loss is None or snapshot['val_loss'] < best_loss:
                        epoch_best_losses[epoch] = snapshot['val_loss']
                        epoch_best_states[epoch] = {
                            'weights': snapshot['lora_state_dict'],
                            'seed': trainer.run_seed,
                            'val_loss': snapshot['val_loss']
                        }
                    
            except Exception as e:
                print(f"Error in training with seed {seed}: {e}")
                continue
        
        # Aggregate metrics across seeds for each monitored epoch
        intermediate_results = aggregate_epoch_metrics(metrics_by_epoch, losses_by_epoch)
        
        if not intermediate_results:
            print(f"\n⚠️  No intermediate results found for combination {i+1}. Skipping aggregation.")
            continue
        
        best_epoch, best_epoch_metrics = select_best_epoch_for_combination(intermediate_results)
        
        if best_epoch is None or best_epoch_metrics is None:
            print(f"\n⚠️  Unable to determine best epoch for combination {i+1}. Skipping aggregation.")
            continue
        
        # Persist aggregated epoch metrics for analysis
        try:
            epoch_summary_path = os.path.join(combination_dir, 'epoch_summary.csv')
            epoch_df = pd.DataFrame.from_dict(intermediate_results, orient='index')
            epoch_df.index.name = 'epoch'
            if 'epoch' in epoch_df.columns:
                epoch_df = epoch_df.drop(columns=['epoch'])
            epoch_df.reset_index(inplace=True)
            epoch_df.to_csv(epoch_summary_path, index=False, mode='w')
        except Exception as e:
            print(f"Warning: failed to save epoch summary for combination {i+1}: {e}")
        
        # Compose summary entry for this parameter combination
        combination_summary = {
            'combo_index': i + 1,
            'best_epoch': best_epoch,
            'num_seeds': best_epoch_metrics.get('num_seeds', len(seeds))
        }
        for param_name, param_value in params.items():
            combination_summary[param_name] = param_value
        metrics_for_summary = {k: v for k, v in best_epoch_metrics.items() if k != 'epoch'}
        for metric_name, metric_value in metrics_for_summary.items():
            combination_summary[metric_name] = metric_value
        all_search_results.append(combination_summary)
        
        # Update global best model if this combination outperforms previous ones
        best_epoch_record = epoch_best_states.get(best_epoch)
        if best_epoch_record is not None:
            global_best_updated, new_global_best_info = update_global_best_model(
                global_best_info,
                best_epoch_metrics,
                {**params, 'combo_index': i + 1, 'best_epoch': best_epoch, 'best_seed': best_epoch_record.get('seed')},
                best_epoch_record
            )
            
            if global_best_updated:
                global_best_info = new_global_best_info
                save_global_best_model(global_best_info, output_dir, target_cols)
                print(f"\n🏆 NEW GLOBAL BEST MODEL FOUND! 🏆")
                print(f"Combination {i+1}, Epoch {best_epoch}")
                print(f"Val Loss: {best_epoch_metrics.get('val_loss_mean', 0):.4f}")
                print(f"Accuracy (Samples): {best_epoch_metrics.get('accuracy_samples_mean', 0):.4f}")
                print(f"Accuracy (Macro): {best_epoch_metrics.get('accuracy_macro_mean', 0):.4f}")
                print(f"Accuracy (Micro): {best_epoch_metrics.get('accuracy_micro_mean', 0):.4f}")
                print(f"F1 Score (Macro): {best_epoch_metrics.get('f1_macro_mean', 0):.4f}")
                print(f"F1 Score (Micro): {best_epoch_metrics.get('f1_micro_mean', 0):.4f}")
                print(f"F1 Score (Samples): {best_epoch_metrics.get('f1_samples_mean', 0):.4f}")
                print(f"ROC AUC (Macro): {best_epoch_metrics.get('roc_auc_macro_mean', 0):.4f}")
                print(f"ROC AUC (Micro): {best_epoch_metrics.get('roc_auc_micro_mean', 0):.4f}")
        else:
            print(f"Warning: no LoRA weights captured for best epoch {best_epoch} in combination {i+1}.")
        
        # Print summary for this parameter combination
        print(f"\n--- Summary for parameter combination {i+1} ---")
        print(f"Parameters: {params}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Val Loss: {best_epoch_metrics.get('val_loss_mean', 0):.4f} ± {best_epoch_metrics.get('val_loss_std', 0):.4f}")
        print(f"Best Accuracy (Samples): {best_epoch_metrics.get('accuracy_samples_mean', 0):.4f} ± {best_epoch_metrics.get('accuracy_samples_std', 0):.4f}")
        print(f"Best ROC AUC (Macro): {best_epoch_metrics.get('roc_auc_macro_mean', 0):.4f} ± {best_epoch_metrics.get('roc_auc_macro_std', 0):.4f}")
        if 'roc_auc_micro_mean' in best_epoch_metrics:
            print(f"Best ROC AUC (Micro): {best_epoch_metrics.get('roc_auc_micro_mean', 0):.4f} ± {best_epoch_metrics.get('roc_auc_micro_std', 0):.4f}")
        print(f"Best F1 (Macro): {best_epoch_metrics.get('f1_macro_mean', 0):.4f} ± {best_epoch_metrics.get('f1_macro_std', 0):.4f}")
        if 'f1_micro_mean' in best_epoch_metrics:
            print(f"Best F1 (Micro): {best_epoch_metrics.get('f1_micro_mean', 0):.4f} ± {best_epoch_metrics.get('f1_micro_std', 0):.4f}")
        
        # Print intermediate epoch results
        print(f"\n--- All Epochs Performance ---")
        for epoch, metrics in sorted(intermediate_results.items()):
            val_loss = metrics.get('val_loss_mean', 0)
            acc_samples = metrics.get('accuracy_samples_mean', 0)
            roc_auc = metrics.get('roc_auc_macro_mean', 0)
            roc_auc_micro = metrics.get('roc_auc_micro_mean', 0)
            f1_macro = metrics.get('f1_macro_mean', 0)
            f1_micro = metrics.get('f1_micro_mean', 0)
            marker = "👑" if epoch == best_epoch else "  "
            print(f"{marker} Epoch {epoch}: Val Loss = {val_loss:.4f} ± {metrics.get('val_loss_std', 0):.4f}, "
                  f"Accuracy (samples) = {acc_samples:.4f} ± {metrics.get('accuracy_samples_std', 0):.4f}, "
                  f"ROC AUC (macro) = {roc_auc:.4f} ± {metrics.get('roc_auc_macro_std', 0):.4f}, "
                  f"ROC AUC (micro) = {roc_auc_micro:.4f} ± {metrics.get('roc_auc_micro_std', 0):.4f}, "
                  f"F1 (macro) = {f1_macro:.4f} ± {metrics.get('f1_macro_std', 0):.4f}, "
                  f"F1 (micro) = {f1_micro:.4f} ± {metrics.get('f1_micro_std', 0):.4f}")
    
    # Save all search results
    if all_search_results:
        all_results_df = pd.DataFrame(all_search_results)
        
        # 将目标列名组合成文件名
        target_name = "_".join(target_cols)
        search_results_path = os.path.join(output_dir, f'{target_name}_hyperparameter_search_results.csv')
        
        # 保存详细结果
        all_results_df.to_csv(search_results_path, index=False, mode='w')
        print(f"\n超参数搜索结果已保存至 {search_results_path}")
        
        # Find best parameters based on different metrics
        best_samples_acc = all_results_df.loc[all_results_df['accuracy_samples_mean'].idxmax()]
        best_macro_acc = all_results_df.loc[all_results_df['accuracy_macro_mean'].idxmax()]
        best_micro_acc = all_results_df.loc[all_results_df['accuracy_micro_mean'].idxmax()]
        best_f1_macro = all_results_df.loc[all_results_df['f1_macro_mean'].idxmax()]
        best_f1_micro = None
        if 'f1_micro_mean' in all_results_df.columns:
            best_f1_micro = all_results_df.loc[all_results_df['f1_micro_mean'].idxmax()]
        best_f1_samples = all_results_df.loc[all_results_df['f1_samples_mean'].idxmax()]
        best_roc_auc = all_results_df.loc[all_results_df['roc_auc_macro_mean'].idxmax()]
        best_roc_auc_micro = None
        if 'roc_auc_micro_mean' in all_results_df.columns:
            best_roc_auc_micro = all_results_df.loc[all_results_df['roc_auc_micro_mean'].idxmax()]
        
        print("\n" + "="*80)
        print("最佳参数组合")
        print("="*80)
        print(f"\n基于最佳 Samples Accuracy:")
        print(f"  参数: {dict([(k, best_samples_acc[k]) for k in param_names if k in best_samples_acc])}")
        print(f"  Samples Accuracy: {best_samples_acc['accuracy_samples_mean']:.4f} ± {best_samples_acc['accuracy_samples_std']:.4f}")
        
        print(f"\n基于最佳 Macro Accuracy:")
        print(f"  参数: {dict([(k, best_macro_acc[k]) for k in param_names if k in best_macro_acc])}")
        print(f"  Macro Accuracy: {best_macro_acc['accuracy_macro_mean']:.4f} ± {best_macro_acc['accuracy_macro_std']:.4f}")
        
        print(f"\n基于最佳 Micro Accuracy:")
        print(f"  参数: {dict([(k, best_micro_acc[k]) for k in param_names if k in best_micro_acc])}")
        print(f"  Micro Accuracy: {best_micro_acc['accuracy_micro_mean']:.4f} ± {best_micro_acc['accuracy_micro_std']:.4f}")
        
        print(f"\n基于最佳 F1 Macro:")
        print(f"  参数: {dict([(k, best_f1_macro[k]) for k in param_names if k in best_f1_macro])}")
        print(f"  F1 Macro: {best_f1_macro['f1_macro_mean']:.4f} ± {best_f1_macro['f1_macro_std']:.4f}")
        if best_f1_micro is not None:
            print(f"\n基于最佳 F1 Micro:")
            print(f"  参数: {dict([(k, best_f1_micro[k]) for k in param_names if k in best_f1_micro])}")
            print(f"  F1 Micro: {best_f1_micro['f1_micro_mean']:.4f} ± {best_f1_micro['f1_micro_std']:.4f}")
        
        print(f"\n基于最佳 F1 Samples:")
        print(f"  参数: {dict([(k, best_f1_samples[k]) for k in param_names if k in best_f1_samples])}")
        print(f"  F1 Samples: {best_f1_samples['f1_samples_mean']:.4f} ± {best_f1_samples['f1_samples_std']:.4f}")
        
        print(f"\n基于最佳 ROC AUC:")
        print(f"  参数: {dict([(k, best_roc_auc[k]) for k in param_names if k in best_roc_auc])}")
        print(f"  ROC AUC: {best_roc_auc['roc_auc_macro_mean']:.4f} ± {best_roc_auc['roc_auc_macro_std']:.4f}")
        if best_roc_auc_micro is not None:
            print(f"\n基于最佳 ROC AUC Micro:")
            print(f"  参数: {dict([(k, best_roc_auc_micro[k]) for k in param_names if k in best_roc_auc_micro])}")
            print(f"  ROC AUC Micro: {best_roc_auc_micro['roc_auc_micro_mean']:.4f} ± {best_roc_auc_micro['roc_auc_micro_std']:.4f}")
        
        
        # Save best parameters
        best_params_path = os.path.join(output_dir, f'{target_name}_best_parameters.json')
        best_params = {
            'best_samples_accuracy': dict([(k, best_samples_acc[k]) for k in param_names + ['accuracy_samples_mean', 'accuracy_samples_std'] if k in best_samples_acc]),
            'best_macro_accuracy': dict([(k, best_macro_acc[k]) for k in param_names + ['accuracy_macro_mean', 'accuracy_macro_std'] if k in best_macro_acc]),
            'best_micro_accuracy': dict([(k, best_micro_acc[k]) for k in param_names + ['accuracy_micro_mean', 'accuracy_micro_std'] if k in best_micro_acc]),
            'best_f1_macro': dict([(k, best_f1_macro[k]) for k in param_names + ['f1_macro_mean', 'f1_macro_std'] if k in best_f1_macro]),
            'best_f1_samples': dict([(k, best_f1_samples[k]) for k in param_names + ['f1_samples_mean', 'f1_samples_std'] if k in best_f1_samples]),
            'best_roc_auc': dict([(k, best_roc_auc[k]) for k in param_names + ['roc_auc_macro_mean', 'roc_auc_macro_std'] if k in best_roc_auc]),
        }
        if best_f1_micro is not None:
            best_params['best_f1_micro'] = dict([(k, best_f1_micro[k]) for k in param_names + ['f1_micro_mean', 'f1_micro_std'] if k in best_f1_micro])
        if best_roc_auc_micro is not None:
            best_params['best_roc_auc_micro'] = dict([(k, best_roc_auc_micro[k]) for k in param_names + ['roc_auc_micro_mean', 'roc_auc_micro_std'] if k in best_roc_auc_micro])
        
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\n最佳参数已保存至 {best_params_path}")
        
        # Save global best model if found
        if global_best_info is not None:
            save_global_best_model(global_best_info, output_dir, target_cols)
            print(f"\n🏆 全局最佳模型已保存至 {output_dir}/global_best_models/")
    
    print("\nLoRA hyperparameter search completed!")
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='LoRA hyperparameter search for LimiX model (upper-half blocks)')
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
    parser.add_argument('--output_dir', type=str, default='./results/lora_hyperparameter_half',
                       help='Output directory for results')
    parser.add_argument('--target_layers', nargs='+', type=int,
                       default=DEFAULT_TARGET_LAYERS,
                       help='Transformer block indices to apply LoRA (default upper-half)')
    
    args = parser.parse_args()
    
    # Define parameter grid for search
    # param_grid = {
    #     'rank': [8, 16, 32, 64],
    #     'alpha': [16.0, 32.0, 64.0, 128.0],
    #     'dropout': [0.0, 0.1, 0.2, 0.3],
    #     'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
    #     'batch_size': [32, 64, 128],  # 增大 batch size 选项
    #     'num_epochs': [10, 20, 30]
    # }

    param_grid = {
        'rank': [64],  # 4个选项
        # alpha 设置为 rank 的两倍，不再搜索
        'dropout': [0.0],  # 3个选项，增加0.0选项
        'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3],  # 3个选项，减少5e-4
        'batch_size': [64],  # 1个选项
        'num_epochs': [40]  # 1个选项，训练30个epoch但保存10、20、30的检查点
    }
    # 总组合数: 1 × 1 × 7 × 1 × 1 = 7 个组合
    
    # Run hyperparameter search
    run_hyperparameter_search(
        model_path=args.model_path,
        data_path=args.data_path,
        target_cols=args.target_cols,
        seeds=args.seeds,
        output_dir=args.output_dir,
        param_grid=param_grid,
        target_layers=args.target_layers
    )


if __name__ == "__main__":
    main()
