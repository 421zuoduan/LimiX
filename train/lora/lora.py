import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from types import MethodType
import math


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for fine-tuning large language models.
    
    This implementation follows the LoRA paper: https://arxiv.org/abs/2106.09685
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = None,  # 将自动设置为 rank 的两倍
        dropout: float = 0.1,
        merge_weights: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor for LoRA
            dropout: Dropout probability
            merge_weights: Whether to merge LoRA weights with original weights
            device: Device to place the layer on
            dtype: Data type for the layer
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        # alpha 自动设置为 rank 的两倍
        self.alpha = alpha if alpha is not None else rank * 2
        self.dropout = dropout
        self.merge_weights = merge_weights
        self.scaling = self.alpha / rank
        
        # LoRA weights
        self.lora_A = nn.Parameter(
            torch.zeros((rank, in_features), device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((out_features, rank), device=device, dtype=dtype)
        )
        
        # Dropout
        self.lora_dropout = nn.Dropout(dropout)
        
        # Weight for merging
        self.weight = None  # Will be set when merging with original layer
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA weights."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LoRA layer.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        if self.merge_weights and self.weight is not None:
            # If weights are merged, just use the original linear layer
            return F.linear(x, self.weight)
        
        # LoRA forward: x @ W + x @ A^T @ B * scaling
        # Original linear transformation (will be added by the original layer)
        lora_result = self.lora_dropout(x)
        lora_result = lora_result @ self.lora_A.T
        lora_result = lora_result @ self.lora_B.T
        lora_result = lora_result * self.scaling
        
        return lora_result


class LoRAAdapter(nn.Module):
    """
    Adapter for applying LoRA to existing linear layers in a model.
    """
    def __init__(
        self,
        target_module: nn.Module,
        rank: int = 16,
        alpha: float = None,  # 将自动设置为 rank 的两倍
        dropout: float = 0.1,
        target_names: list = ["weight"],
        merge_weights: bool = False
    ):
        """
        Initialize LoRA adapter.
        
        Args:
            target_module: The target module to apply LoRA to
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor for LoRA
            dropout: Dropout probability
            target_names: Names of target parameters to apply LoRA to
            merge_weights: Whether to merge LoRA weights with original weights
        """
        super().__init__()
        self.target_module = target_module
        self.rank = rank
        # alpha 自动设置为 rank 的两倍
        self.alpha = alpha if alpha is not None else rank * 2
        self.dropout = dropout
        self.target_names = target_names
        self.merge_weights = merge_weights
        self.original_forward = target_module.forward
        self._patched_forward: Optional[MethodType] = None
        self.merged = False
        
        # Store original weights
        self.original_weights = {}
        self.lora_layers = nn.ModuleDict()
        
        # Apply LoRA to target parameters
        self._apply_lora()
        if not self.merge_weights:
            self._patch_forward()

    def _apply_lora(self):
        """Apply LoRA to target parameters."""
        for name, param in self.target_module.named_parameters():
            if any(target in name for target in self.target_names):
                if param.dim() >= 2:  # Only apply to 2D+ tensors (linear layers)
                    # Store original weights
                    self.original_weights[name] = param.data.clone()
                    
                    # Create LoRA layer
                    in_features = param.shape[1] if param.dim() == 2 else param.shape[0]
                    out_features = param.shape[0]
                    
                    lora_layer = LoRALayer(
                        in_features=in_features,
                        out_features=out_features,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                        merge_weights=self.merge_weights,
                        device=param.device,
                        dtype=param.dtype
                    )
                    
                    self.lora_layers[name] = lora_layer

    def _create_lora_forward(self) -> MethodType:
        """Create a forward method that injects LoRA contributions."""
        original_forward = self.original_forward
        lora_layers = self.lora_layers

        def lora_forward(module_self, input_tensor, *args, **kwargs):
            output = original_forward(input_tensor, *args, **kwargs)

            if not lora_layers:
                return output

            # Accumulate LoRA contributions across all adapted parameters
            lora_contribution = None
            for lora_layer in lora_layers.values():
                contribution = lora_layer(input_tensor)
                if contribution.shape != output.shape:
                    contribution = contribution.view_as(output)
                lora_contribution = contribution if lora_contribution is None else lora_contribution + contribution

            if lora_contribution is None:
                return output

            return output + lora_contribution.to(output.dtype)

        return MethodType(lora_forward, self.target_module)

    def _patch_forward(self):
        """Patch the target module forward with LoRA contributions."""
        if self.merge_weights:
            return

        self._patched_forward = self._create_lora_forward()
        self.target_module.forward = self._patched_forward

    def forward(self, *args, **kwargs):
        """Forward pass with LoRA applied."""
        # Store original forward
        if self._patched_forward is None and not self.merge_weights:
            self._patch_forward()
        return self.target_module.forward(*args, **kwargs)

    def merge_weights_to_model(self):
        """Merge LoRA weights with original weights."""
        if self.merged:
            return
        for name, lora_layer in self.lora_layers.items():
            if name in self.original_weights:
                # Compute merged weights
                original_weight = self.original_weights[name]
                lora_weight = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
                
                # Merge weights
                merged_weight = original_weight + lora_weight
                
                # Update target module's weights
                for param_name, param in self.target_module.named_parameters():
                    if param_name == name:
                        param.data.copy_(merged_weight)
                        break
        if self._patched_forward is not None:
            self.target_module.forward = self.original_forward
        self.merged = True

    def unmerge_weights_from_model(self):
        """Restore original weights."""
        for name, original_weight in self.original_weights.items():
            for param_name, param in self.target_module.named_parameters():
                if param_name == name:
                    param.data = original_weight.clone()
                    break
        if self.merged:
            self.merged = False
            self._patch_forward()

    def get_lora_parameters(self):
        """Get LoRA parameters for optimization."""
        params = []
        for lora_layer in self.lora_layers.values():
            params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return params
    
    def save_lora_weights(self, path: str):
        """Save LoRA weights."""
        lora_state_dict = {}
        for name, lora_layer in self.lora_layers.items():
            lora_state_dict[f"{name}.lora_A"] = lora_layer.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = lora_layer.lora_B.data
            lora_state_dict[f"{name}.scaling"] = lora_layer.scaling
        
        torch.save(lora_state_dict, path)
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        lora_state_dict = torch.load(path, map_location='cpu')
        
        for name, lora_layer in self.lora_layers.items():
            if f"{name}.lora_A" in lora_state_dict:
                lora_layer.lora_A.data = lora_state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in lora_state_dict:
                lora_layer.lora_B.data = lora_state_dict[f"{name}.lora_B"]
            if f"{name}.scaling" in lora_state_dict:
                lora_layer.scaling = lora_state_dict[f"{name}.scaling"]


class LoRAEnabledTransformer(nn.Module):
    """
    Transformer model with LoRA enabled for specific layers.
    """
    def __init__(
        self,
        base_model: nn.Module,
        target_layers: list = None,
        rank: int = 16,
        alpha: float = None,  # 将自动设置为 rank 的两倍
        dropout: float = 0.1
    ):
        """
        Initialize LoRA-enabled transformer.
        
        Args:
            base_model: Base transformer model
            target_layers: List of layer indices to apply LoRA to
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor for LoRA
            dropout: Dropout probability
        """
        super().__init__()
        self.base_model = base_model
        self.target_layers = target_layers or [10, 11]  # Last two layers by default
        self.rank = rank
        # alpha 自动设置为 rank 的两倍
        self.alpha = alpha if alpha is not None else rank * 2
        self.dropout = dropout
        
        # Apply LoRA to target layers
        self.lora_adapters = nn.ModuleDict()
        self._apply_lora_to_layers()
    
    def _apply_lora_to_layers(self):
        """Apply LoRA to specified transformer layers."""
        # Get transformer encoder layers
        if hasattr(self.base_model, 'transformer_encoder') and hasattr(self.base_model.transformer_encoder, 'layers'):
            layers = self.base_model.transformer_encoder.layers
            
            for layer_idx in self.target_layers:
                if layer_idx < len(layers):
                    layer = layers[layer_idx]
                    
                    # Apply LoRA to attention and MLP layers
                    for name, module in layer.named_modules():
                        if isinstance(module, nn.Linear):
                            adapter_name = f"layer_{layer_idx}_{name.replace('.', '_')}"
                            adapter = LoRAAdapter(
                                target_module=module,
                                rank=self.rank,
                                alpha=self.alpha,
                                dropout=self.dropout
                            )
                            self.lora_adapters[adapter_name] = adapter
    
    def forward(self, *args, **kwargs):
        """Forward pass with LoRA."""
        return self.base_model(*args, **kwargs)
    
    def get_lora_parameters(self):
        """Get all LoRA parameters for optimization."""
        params = []
        for adapter in self.lora_adapters.values():
            params.extend(adapter.get_lora_parameters())
        return params
    
    def get_base_parameters(self):
        """Get base model parameters (non-LoRA)."""
        params = []
        for name, param in self.base_model.named_parameters():
            # Skip parameters that have LoRA applied
            has_lora = False
            for adapter_name, adapter in self.lora_adapters.items():
                if any(target in name for target in adapter.target_names):
                    has_lora = True
                    break
            
            if not has_lora:
                params.append(param)
        
        return params
    
    def merge_lora_weights(self):
        """Merge all LoRA weights."""
        for adapter in self.lora_adapters.values():
            adapter.merge_weights_to_model()
    
    def unmerge_lora_weights(self):
        """Unmerge all LoRA weights."""
        for adapter in self.lora_adapters.values():
            adapter.unmerge_weights_from_model()
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return all LoRA weights as a state dict."""
        lora_state_dict = {}
        for adapter_name, adapter in self.lora_adapters.items():
            for name, lora_layer in adapter.lora_layers.items():
                full_name = f"{adapter_name}.{name}"
                lora_state_dict[f"{full_name}.lora_A"] = lora_layer.lora_A.detach().cpu().clone()
                lora_state_dict[f"{full_name}.lora_B"] = lora_layer.lora_B.detach().cpu().clone()
                lora_state_dict[f"{full_name}.scaling"] = torch.tensor(
                    float(lora_layer.scaling), dtype=torch.float32
                )
        return lora_state_dict
    
    def save_lora_weights(self, path: str):
        """Save all LoRA weights."""
        lora_state_dict = {}
        for adapter_name, adapter in self.lora_adapters.items():
            for name, lora_layer in adapter.lora_layers.items():
                lora_state_dict[f"{adapter_name}.{name}.lora_A"] = lora_layer.lora_A.data
                lora_state_dict[f"{adapter_name}.{name}.lora_B"] = lora_layer.lora_B.data
                lora_state_dict[f"{adapter_name}.{name}.scaling"] = lora_layer.scaling
        
        torch.save(lora_state_dict, path)
    
    def load_lora_weights(self, path: str):
        """Load all LoRA weights."""
        lora_state_dict = torch.load(path, map_location='cpu')
        
        for adapter_name, adapter in self.lora_adapters.items():
            for name, lora_layer in adapter.lora_layers.items():
                full_name = f"{adapter_name}.{name}"
                if f"{full_name}.lora_A" in lora_state_dict:
                    lora_layer.lora_A.data = lora_state_dict[f"{full_name}.lora_A"]
                if f"{full_name}.lora_B" in lora_state_dict:
                    lora_layer.lora_B.data = lora_state_dict[f"{full_name}.lora_B"]
                if f"{full_name}.scaling" in lora_state_dict:
                    lora_layer.scaling = lora_state_dict[f"{full_name}.scaling"]
