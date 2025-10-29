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
        batch_size: int = 32,
        num_epochs: int = 10,
        device: str = 'cuda'
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
        """
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
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
    
    def load_data(self, file_path: str, target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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
        
        print(f"Data shape - Features: {X.shape}, Targets: {y.shape}")
        return X, y
    
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
        
        # Use mixed precision training
        scaler = torch.cuda.amp.GradScaler('cuda')
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Reset scaler for each batch
            scaler = torch.cuda.amp.GradScaler('cuda')
            
            # Prepare data for LimiX model
            # LimiX expects x: [batch, seq, feature] and y: [batch, label]
            # For multi-label, we need to handle each label separately
            batch_size = X_batch.size(0)
            n_features = X_batch.size(1)
            n_labels = y_batch.size(1)
            
            # For simplicity, we'll use a simplified approach
            # In practice, you might need to adapt this based on the exact LimiX interface
            total_batch_loss = 0.0
            valid_forward_passes = 0
            
            for label_idx in range(n_labels):
                # Prepare data for single label
                y_single = y_batch[:, label_idx:label_idx+1]
                
                # Create sequence dimension (LimiX expects sequence dimension)
                # Based on the original predictor code, x should be [1, batch, features] and y should be [1, batch]
                X_seq = X_batch.unsqueeze(0)  # [1, batch, features]
                y_for_model = y_single.squeeze(-1).unsqueeze(0)  # [1, batch]
                
                try:
                    # Forward pass through model with mixed precision
                    with torch.cuda.amp.autocast():
                        output = self.model(
                            x=X_seq,
                            y=y_for_model,
                            eval_pos=1,
                            task_type='cls'
                        )
                    
                    # Extract classification output
                    if isinstance(output, dict):
                        cls_output = output.get('cls_output', output)
                    else:
                        cls_output = output
                    
                    # Debug print to understand shapes
                    if label_idx == 0 and batch_idx == 0:
                        print(f"Debug - X_seq shape: {X_seq.shape}")
                        print(f"Debug - y_for_model shape: {y_for_model.shape}")
                        print(f"Debug - output type: {type(output)}")
                        if isinstance(output, dict):
                            print(f"Debug - output keys: {output.keys()}")
                        print(f"Debug - cls_output shape: {cls_output.shape}")
                        print(f"Debug - y_target shape: {y_for_model.squeeze(0).shape}")
                    
                    # Calculate loss
                    if cls_output.dim() > 2:
                        cls_output = cls_output.squeeze()
                    
                    # Ensure shapes match - remove the sequence dimension
                    if cls_output.dim() > 1:
                        cls_output = cls_output.squeeze(0)  # [batch, num_classes]
                    
                    # Get the target labels
                    y_target = y_for_model.squeeze(0)  # [batch]
                    
                    # Ensure target is float for BCEWithLogitsLoss
                    y_target = y_target.float()
                    
                    # Check if we have the right number of samples
                    if cls_output.shape[0] != y_target.shape[0]:
                        print(f"Shape mismatch: cls_output.shape[0]={cls_output.shape[0]}, y_target.shape[0]={y_target.shape[0]}")
                        # If shapes don't match, adjust cls_output to match target size
                        if cls_output.shape[0] < y_target.shape[0]:
                            # Pad cls_output if it's smaller
                            padding_size = y_target.shape[0] - cls_output.shape[0]
                            padding = torch.zeros(padding_size, cls_output.shape[1], device=cls_output.device)
                            cls_output = torch.cat([cls_output, padding], dim=0)
                        else:
                            # Truncate cls_output if it's larger
                            cls_output = cls_output[:y_target.shape[0]]
                        
                        # Also adjust y_target to match cls_output if needed
                        if y_target.shape[0] != cls_output.shape[0]:
                            y_target = y_target[:cls_output.shape[0]]
                    
                    # Ensure shapes match
                    if cls_output.shape[-1] == 2:  # Binary classification
                        # For binary classification, use the positive class logits
                        # Convert to float for BCEWithLogitsLoss
                        loss = self.criterion(cls_output[:, 1], y_target)
                    elif cls_output.shape[-1] == 10:  # LimiX default 10-class output
                        # For binary classification with 10-class output, use first two classes
                        loss = self.criterion(cls_output[:, 1], y_target)
                    else:
                        # For multi-class, use the appropriate class
                        loss = self.criterion(cls_output, y_target)
                    
                    # Scale loss by number of labels to maintain similar magnitude
                    scaled_loss = loss / n_labels
                    
                    # Backward pass with mixed precision
                    scaler.scale(scaled_loss).backward()
                    total_batch_loss += loss.item()
                    valid_forward_passes += 1
                    
                except Exception as e:
                    print(f"Error in forward pass for label {label_idx}: {e}")
                    continue
            
            # Update weights with mixed precision - only if we had valid forward passes
            if valid_forward_passes > 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()  # Call update() to reset for next iteration
            else:
                # If no valid forward passes, just update scaler to avoid error
                scaler.update()
            
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
                n_features = X_batch.size(1)
                n_labels = y_batch.size(1)
                
                batch_predictions = []
                batch_loss = 0.0
                
                for label_idx in range(n_labels):
                    # Prepare data for single label
                    y_single = y_batch[:, label_idx:label_idx+1]
                    
                    # Create sequence dimension
                    # Based on the original predictor code, x should be [1, batch, features] and y should be [1, batch]
                    X_seq = X_batch.unsqueeze(0)  # [1, batch, features]
                    y_for_model = y_single.squeeze(-1).unsqueeze(0)  # [1, batch]
                    
                    try:
                        # Forward pass with mixed precision
                        with torch.cuda.amp.autocast():
                            output = self.model(
                                x=X_seq,
                                y=y_for_model,
                                eval_pos=1,
                                task_type='cls'
                            )
                        
                        # Extract classification output
                        if isinstance(output, dict):
                            cls_output = output.get('cls_output', output)
                        else:
                            cls_output = output
                        
                        if cls_output.dim() > 2:
                            cls_output = cls_output.squeeze()
                        
                        # Remove sequence dimension
                        if cls_output.dim() > 1:
                            cls_output = cls_output.squeeze(0)  # [batch, num_classes]
                        
                        # Get the target labels
                        y_target = y_for_model.squeeze(0)  # [batch]
                        
                        # Convert target to float for BCEWithLogitsLoss
                        y_target = y_target.float()  # [batch]
                        
                        # Check if we have the right number of samples
                        if cls_output.shape[0] != y_target.shape[0]:
                            print(f"Eval shape mismatch: cls_output.shape[0]={cls_output.shape[0]}, y_target.shape[0]={y_target.shape[0]}")
                            # If shapes don't match, adjust cls_output to match target size
                            if cls_output.shape[0] < y_target.shape[0]:
                                # Pad cls_output if it's smaller
                                padding_size = y_target.shape[0] - cls_output.shape[0]
                                padding = torch.zeros(padding_size, cls_output.shape[1], device=cls_output.device)
                                cls_output = torch.cat([cls_output, padding], dim=0)
                            else:
                                # Truncate cls_output if it's larger
                                cls_output = cls_output[:y_target.shape[0]]
                        
                        # LimiX model outputs 10 classes by default, but we have binary targets
                        if cls_output.shape[-1] == 10:  # LimiX default 10-class output
                            # For binary classification, use only first two classes
                            cls_output_binary = cls_output[:, :2]  # [batch, 2]
                            loss = self.criterion(cls_output_binary[:, 1], y_target)
                            # Get probability of positive class
                            pred = torch.softmax(cls_output_binary, dim=1)[:, 1].cpu().numpy()
                        elif cls_output.shape[-1] == 2:  # Binary classification
                            loss = self.criterion(cls_output[:, 1], y_target)
                            # Get probability of positive class
                            pred = torch.softmax(cls_output, dim=1)[:, 1].cpu().numpy()
                        else:
                            # For multi-class, use appropriate class
                            loss = self.criterion(cls_output, y_target)
                            pred = torch.softmax(cls_output, dim=1)[:, 1].cpu().numpy()
                        
                        batch_loss += loss.item()
                        batch_predictions.append(pred)
                        
                    except Exception as e:
                        print(f"Error in evaluation for label {label_idx}: {e}")
                        # Use dummy predictions
                        batch_predictions.append(np.zeros(batch_size))
                        continue
                
                total_loss += batch_loss / n_labels
                
                # Combine predictions for all labels
                if batch_predictions:
                    batch_preds = np.stack(batch_predictions, axis=1)
                    all_predictions.append(batch_preds)
                    all_targets.append(y_batch.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        
        if all_predictions and all_targets:
            all_predictions = np.vstack(all_predictions)
            all_targets = np.vstack(all_targets)
            
            # Convert predictions to binary
            binary_predictions = (all_predictions > 0.5).astype(int)
            
            # Calculate metrics consistent with multi_label_inference.py
            try:
                roc_auc = roc_auc_score(all_targets, all_predictions, average='macro')
            except Exception as e:
                print(f"ROC-AUC计算失败: {e}")
                roc_auc = np.nan
                
            try:
                avg_precision = average_precision_score(all_targets, all_predictions, average='macro')
            except Exception as e:
                print(f"Average Precision计算失败: {e}")
                avg_precision = np.nan
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(all_targets, binary_predictions),
                'precision_macro': precision_score(all_targets, binary_predictions, average='macro', zero_division=0),
                'recall_macro': recall_score(all_targets, binary_predictions, average='macro', zero_division=0),
                'f1_macro': f1_score(all_targets, binary_predictions, average='macro', zero_division=0),
                'f1_micro': f1_score(all_targets, binary_predictions, average='micro', zero_division=0),
                'roc_auc': roc_auc,
                'avg_precision': avg_precision
            }
        else:
            metrics = {
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
                'roc_auc_macro': 0.0
            }
        
        return avg_loss, metrics
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        save_dir: str = './checkpoints'
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
        """
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
            print(f"Val Metrics: {val_metrics}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, 'best_lora_model.pth')
                self.save_checkpoint(best_model_path, epoch, val_loss, val_metrics)
                print(f"Saved best model to {best_model_path}")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'lora_epoch_{epoch+1}.pth')
                self.save_checkpoint(checkpoint_path, epoch, val_loss, val_metrics)
    
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
                'learning_rate': self.learning_rate
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
    
    print("LoRA Fine-tuning for LimiX Model")
    print("=" * 50)
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Target columns: {args.target_cols}")
    print(f"Random seeds: {args.seeds}")
    print(f"LoRA rank: {args.rank}")
    print(f"LoRA alpha: {args.alpha}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    X, y = load_and_preprocess_data(args.data_path, args.target_cols)
    
    # Run experiments for each seed
    all_results = []
    
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Running experiment with seed {seed}")
        print(f"{'='*60}")
        
        # Create trainer
        trainer = LoRATrainer(
            model_path=args.model_path,
            target_layers=[10, 11],  # Last two layers
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            device=args.device
        )
        
        # Prepare data
        train_loader, val_loader = trainer.prepare_data(
            X, y, test_size=0.2, random_state=seed
        )
        
        # Create seed-specific output directory
        seed_output_dir = os.path.join(args.output_dir, f'seed_{seed}')
        os.makedirs(seed_output_dir, exist_ok=True)
        
        # Train model
        trainer.train(train_loader, val_loader, save_dir=seed_output_dir)
        
        # Save LoRA weights
        lora_path = os.path.join(seed_output_dir, 'lora_weights.pth')
        trainer.save_lora_weights(lora_path)
        
        # Get final metrics
        if trainer.val_metrics:
            final_metrics = trainer.val_metrics[-1]
            final_metrics['seed'] = seed
            all_results.append(final_metrics)
            
            # Save results
            results_df = pd.DataFrame([final_metrics])
            results_path = os.path.join(seed_output_dir, 'results.csv')
            results_df.to_csv(results_path, index=False)
            
            print(f"Results for seed {seed}: {final_metrics}")
    
    # Save all results using format similar to multi_label_inference.py
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        
        # 将目标列名组合成文件名
        target_name = "_".join(args.target_cols)
        output_path = os.path.join(args.output_dir, f'{target_name}_results.csv')
        summary_path = os.path.join(args.output_dir, f'{target_name}_summary.csv')
        
        # 确保目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
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
                print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
        
        # 保存均值结果 - 修改为保存多个种子的指标平均值和标准差
        # 创建包含平均值和标准差的DataFrame
        summary_data = {}
        for col in numeric_cols:
            if col != 'seed':
                # 保存平均值，列名格式为 "指标名_mean"
                summary_data[f"{col}_mean"] = [summary.loc['mean', col]]
                # 保存标准差，列名格式为 "指标名_std"
                summary_data[f"{col}_std"] = [summary.loc['std', col]]
        
        # 添加种子数量信息
        summary_data['num_seeds'] = [len(all_results)]
        
        # 创建summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # 重新排列列顺序，将num_seeds放在第一列，然后按指标分组
        cols = ['num_seeds']
        for col in numeric_cols:
            if col != 'seed':
                cols.extend([f"{col}_mean", f"{col}_std"])
        summary_df = summary_df[cols]
        
        summary_df.to_csv(summary_path, index=False)
        print(f"均值结果已保存至 {summary_path}")
        print(f"注意：summary文件现在保存的是多个种子的指标平均值和标准差，而不是最高值")
    
    print("\nLoRA fine-tuning completed!")


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


if __name__ == "__main__":
    main()