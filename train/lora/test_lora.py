import os
import sys
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

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")  # Show first 3 paths

# Import lora module from current directory
from lora import LoRAEnabledTransformer

# Import loading module from utils
from utils.loading import load_model


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
    print(f"Feature columns: {list(df.drop(columns=target_cols).columns)}")
    print(f"Target columns: {target_cols}")
    
    return X, y


def test_lora_components():
    """Test LoRA components individually."""
    print("Testing LoRA components...")
    
    # Test LoRA layer
    from lora import LoRALayer
    
    # Create a simple LoRA layer
    lora_layer = LoRALayer(
        in_features=256,
        out_features=256,
        rank=16,
        alpha=32.0,
        dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(2, 10, 256)  # [batch, seq, features]
    lora_output = lora_layer(x.view(-1, 256))  # Flatten for LoRA
    lora_output = lora_output.view(2, 10, 256)  # Reshape back
    
    print(f"LoRA layer test - Input shape: {x.shape}, Output shape: {lora_output.shape}")
    
    # Test LoRA adapter
    from lora import LoRAAdapter
    
    # Create a simple linear layer to adapt
    linear_layer = nn.Linear(256, 256)
    adapter = LoRAAdapter(
        target_module=linear_layer,
        rank=16,
        alpha=32.0,
        dropout=0.1
    )
    
    print("LoRA adapter test - Created successfully")
    
    return True


def test_model_loading(model_path: str):
    """Test loading the base LimiX model."""
    print(f"Testing model loading from {model_path}...")
    
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Check if model has transformer encoder
        if hasattr(model, 'transformer_encoder'):
            print(f"Transformer encoder found: {type(model.transformer_encoder)}")
            if hasattr(model.transformer_encoder, 'layers'):
                print(f"Number of transformer layers: {len(model.transformer_encoder.layers)}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def test_lora_integration(model_path: str):
    """Test LoRA integration with LimiX model."""
    print("Testing LoRA integration with LimiX model...")
    
    try:
        # Load base model
        base_model = load_model(model_path)
        
        # Create LoRA-enabled model
        lora_model = LoRAEnabledTransformer(
            base_model=base_model,
            target_layers=[10, 11],  # Last two layers
            rank=16,
            alpha=32.0,
            dropout=0.1
        )
        
        print("LoRA-enabled model created successfully!")
        
        # Check LoRA parameters
        lora_params = lora_model.get_lora_parameters()
        base_params = lora_model.get_base_parameters()
        
        print(f"Number of LoRA parameters: {len(lora_params)}")
        print(f"Number of base parameters: {len(base_params)}")
        
        # Count total parameters
        lora_param_count = sum(p.numel() for p in lora_params)
        base_param_count = sum(p.numel() for p in base_params)
        
        print(f"LoRA parameter count: {lora_param_count:,}")
        print(f"Base parameter count: {base_param_count:,}")
        print(f"LoRA/Total ratio: {lora_param_count / (lora_param_count + base_param_count):.4f}")
        
        return lora_model
        
    except Exception as e:
        print(f"Error in LoRA integration: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_loading(data_path: str, target_cols: List[str]):
    """Test data loading and preprocessing."""
    print(f"Testing data loading from {data_path}...")
    
    try:
        X, y = load_and_preprocess_data(data_path, target_cols)
        
        # Check data shapes
        print(f"Features shape: {X.shape}")
        print(f"Targets shape: {y.shape}")
        
        # Check data types
        print(f"Features dtype: {X.dtype}")
        print(f"Targets dtype: {y.dtype}")
        
        # Check for NaN values
        print(f"Features contain NaN: {np.isnan(X).any()}")
        print(f"Targets contain NaN: {np.isnan(y).any()}")
        
        # Check target value ranges
        for i, col in enumerate(target_cols):
            unique_vals = np.unique(y[:, i])
            print(f"Target '{col}' unique values: {unique_vals}")
        
        return X, y
        
    except Exception as e:
        print(f"Error in data loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_training_loop(model, X, y, target_cols):
    """Test a simplified training loop."""
    print("Testing simplified training loop...")
    
    try:
        # Test LoRA parameter access
        lora_params = model.get_lora_parameters()
        base_params = model.get_base_parameters()
        
        print(f"LoRA parameters accessible: {len(lora_params) > 0}")
        print(f"Base parameters accessible: {len(base_params) > 0}")
        
        # Test parameter count
        lora_param_count = sum(p.numel() for p in lora_params)
        base_param_count = sum(p.numel() for p in base_params)
        
        print(f"LoRA parameter count: {lora_param_count:,}")
        print(f"Base parameter count: {base_param_count:,}")
        
        # Test weight merging/unmerging
        print("Testing weight merging...")
        try:
            model.merge_lora_weights()
            print("LoRA weights merged successfully!")
            
            model.unmerge_lora_weights()
            print("LoRA weights unmerged successfully!")
        except Exception as e:
            print(f"Error in weight merging: {e}")
        
        # Test saving/loading LoRA weights
        print("Testing LoRA weight saving/loading...")
        try:
            test_path = "./test_lora_weights.pth"
            model.save_lora_weights(test_path)
            print(f"LoRA weights saved to {test_path}")
            
            # Check if file exists
            import os
            if os.path.exists(test_path):
                print("LoRA weights file created successfully!")
                os.remove(test_path)  # Clean up
            else:
                print("LoRA weights file not created!")
        except Exception as e:
            print(f"Error in LoRA weight saving: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error in training loop test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_label_evaluation(model, X, y, target_cols):
    """Test multi-label evaluation similar to multi_label_inference.py."""
    print("Testing multi-label evaluation...")
    
    try:
        from train_lora import LoRATrainer
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create LoRA trainer for multi-label
        lora_trainer = LoRATrainer(
            model_path="/home/ruochen/models/limix/LimiX-16M.ckpt",
            target_layers=[10, 11],  # Last two layers
            rank=16,
            alpha=32.0,
            dropout=0.1,
            learning_rate=1e-4,
            batch_size=32,
            num_epochs=2,  # Just 2 epochs for testing
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Prepare data for LoRA training - use all targets together
        lora_train_loader, lora_val_loader = lora_trainer.prepare_data(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train LoRA model
        lora_trainer.train(lora_train_loader, lora_val_loader, save_dir='./test_checkpoints')
        
        # Evaluate LoRA model
        lora_val_loss, lora_metrics = lora_trainer.evaluate(lora_val_loader)
        
        print(f"Multi-label evaluation metrics: {lora_metrics}")
        
        # Check if metrics are consistent with multi_label_inference.py
        required_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_micro', 'roc_auc', 'avg_precision']
        missing_metrics = [m for m in required_metrics if m not in lora_metrics]
        
        if missing_metrics:
            print(f"Missing metrics: {missing_metrics}")
            return False
        
        print("Multi-label evaluation test passed!")
        return True
        
    except Exception as e:
        print(f"Error in multi-label evaluation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("LoRA Fine-tuning Test Suite")
    print("=" * 50)
    
    # Configuration
    model_path = "/home/ruochen/models/limix/LimiX-16M.ckpt"
    data_path = "/home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx"
    target_cols = ["脑脊液漏", "神经功能加重", "严重并发症（死亡＋瘫痪＋再手术＋再入院）"]
    
    # Test 1: LoRA components
    print("\n1. Testing LoRA components...")
    if not test_lora_components():
        print("LoRA components test failed!")
        return False
    
    # Test 2: Model loading
    print("\n2. Testing model loading...")
    base_model = test_model_loading(model_path)
    if base_model is None:
        print("Model loading test failed!")
        return False
    
    # Test 3: LoRA integration
    print("\n3. Testing LoRA integration...")
    lora_model = test_lora_integration(model_path)
    if lora_model is None:
        print("LoRA integration test failed!")
        return False
    
    # Test 4: Data loading
    print("\n4. Testing data loading...")
    X, y = test_data_loading(data_path, target_cols)
    if X is None or y is None:
        print("Data loading test failed!")
        return False
    
    # Test 5: Training loop
    print("\n5. Testing training loop...")
    if not test_training_loop(lora_model, X, y, target_cols):
        print("Training loop test failed!")
        return False
    
    # Test 6: Multi-label evaluation
    print("\n6. Testing multi-label evaluation...")
    if not test_multi_label_evaluation(lora_model, X, y, target_cols):
        print("Multi-label evaluation test failed!")
        return False
    
    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("LoRA fine-tuning system is ready for use.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)