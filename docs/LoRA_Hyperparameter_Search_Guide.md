# LoRA 超参数搜索指南

本指南介绍如何使用新创建的 LoRA 超参数搜索功能来优化 LimiX 模型的微调效果。

## 概述

由于当前的 LoRA 微调效果不佳，我们创建了一套完整的超参数搜索工具，包括：

1. **完整的超参数搜索脚本** (`train/lora/lora_hyperparameter_search.py`)
2. **简化的测试脚本** (`scripts/test_lora_hyperparameter_search.sh`)
3. **Shell 包装脚本** (`scripts/run_lora_hyperparameter_search.sh`)

## 主要改进

### 1. 增大的 Batch Size

- 默认 batch size 从 32 增加到 64
- 提供更大的 batch size 选项（32, 64, 128）
- 有助于提高训练稳定性和收敛速度

### 2. 全面的参数搜索范围

超参数搜索涵盖以下参数：

- **rank**: [8, 16, 32, 64] - LoRA 低秩分解的秩
- **alpha**: 自动设置为 rank 的两倍 - LoRA 缩放因子（不再搜索）
- **dropout**: [0.0, 0.1, 0.2] - Dropout 概率
- **learning_rate**: [1e-5, 5e-5, 1e-4] - 学习率
- **batch_size**: [64] - 批次大小（已增大）
- **num_epochs**: [20] - 训练轮数
- **总组合数**: 36个组合（4 × 3 × 3 × 1 × 1），远小于100的限制

## 使用方法

### 1. 快速测试（推荐先运行）

使用较小的参数范围进行快速测试，验证脚本是否正常工作：

```bash
bash scripts/test_lora_hyperparameter_search.sh
```

这个测试脚本会：
- 使用较小的参数范围（rank: [8, 16], dropout: [0.0, 0.1], learning_rate: [1e-4, 5e-4] 等）
- alpha 自动设置为 rank 的两倍
- 只训练 2 个 epoch
- 只使用前 2 个随机种子
- 总共8个参数组合（2 × 2 × 2 × 1 × 1）
- 快速验证整个流程是否正常

### 2. 完整超参数搜索

运行完整的超参数搜索：

```bash
bash scripts/run_lora_hyperparameter_search.sh
```

或者使用自定义参数：

```bash
bash scripts/run_lora_hyperparameter_search.sh \
    --model_path /path/to/model \
    --data_path /path/to/data \
    --target_cols "目标列1 目标列2 目标列3" \
    --seeds "10 20 30 40 50" \
    --output_dir ./results/my_lora_search
```

### 3. 直接使用 Python 脚本

```bash
python train/lora/lora_hyperparameter_search.py \
    --model_path /path/to/model \
    --data_path /path/to/data \
    --target_cols 目标列1 目标列2 目标列3 \
    --seeds 10 20 30 40 50 \
    --output_dir ./results/my_lora_search
```

## 输出结果

### 1. 搜索结果文件

- `{target_name}_hyperparameter_search_results.csv`: 包含所有参数组合的详细结果
- `{target_name}_best_parameters.json`: 包含基于不同指标的最佳参数

### 2. 结果结构

每个参数组合会生成以下结构：
```
output_dir/
├── combo_1/
│   ├── seed_10/
│   │   ├── best_lora_model.pth
│   │   ├── best_lora_weights.pth
│   │   └── results.csv
│   └── seed_20/
│       └── ...
├── combo_2/
│   └── ...
└── {target_name}_hyperparameter_search_results.csv
```

### 3. 最佳参数评估

系统会自动基于以下指标评估最佳参数：
- **Subset Accuracy**: 所有标签完全正确的样本比例
- **Element-wise Accuracy**: 每个标签位置的正确率
- **F1 Macro**: 宏平均 F1 分数
- **ROC AUC**: ROC 曲线下面积

## 性能优化建议

### 1. 批次大小选择

- **小数据集**: 使用 batch_size=32 或 64
- **大数据集**: 使用 batch_size=64 或 128
- **GPU 内存限制**: 根据可用内存调整

### 2. 学习率调优

- **高学习率** (5e-4): 快速收敛，但可能不稳定
- **中等学习率** (1e-4): 平衡速度和稳定性
- **低学习率** (1e-5): 稳定但收敛慢

### 3. LoRA 参数选择

- **rank**: 控制模型容量，值越大表达能力越强
- **alpha**: 自动设置为 rank 的两倍，保持 scaling = alpha/rank = 2 的恒定值
- **dropout**: 防止过拟合，数据少时使用较高值

## 故障排除

### 1. 内存不足

如果遇到 GPU 内存不足：
- 减小 batch_size
- 减少同时训练的标签数量
- 使用梯度累积

### 2. 训练不稳定

如果训练过程不稳定：
- 降低学习率
- 增加 dropout
- 增加 num_epochs

### 3. 过拟合

如果验证集性能下降：
- 增加 dropout
- 减少 rank
- 减少训练轮数

## 示例输出

```
==========================================
LimiX LoRA超参数搜索
==========================================
模型路径: /home/ruochen/models/limix/LimiX-16M.ckpt
数据路径: /home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx
目标列: 脑脊液漏 神经功能加重 严重并发症（死亡＋瘫痪＋再手术＋再入院）
随机种子: 10 20 30 40 50
输出目录: ./results/lora_hyperparameter_search
==========================================

================================================================================
Testing parameter combination 1/2304
================================================================================
Parameters: {'rank': 8, 'dropout': 0.0, 'learning_rate': 1e-05, 'batch_size': 64, 'num_epochs': 20}

--- Running with seed 10 ---
Train set: (X, y)
Test set: (X, y)
...
```

## 注意事项

1. **计算资源**: 完整的超参数搜索可能需要大量时间和计算资源
2. **存储空间**: 确保有足够的磁盘空间存储所有模型和结果
3. **并行化**: 当前版本是串行执行，未来可以考虑并行化以提高效率

## 相关文件

- `train/lora/lora_hyperparameter_search.py`: 主要的超参数搜索脚本
- `scripts/run_lora_hyperparameter_search.sh`: Shell 包装脚本
- `scripts/test_lora_hyperparameter_search.sh`: 快速测试脚本
- `train/lora/train_lora.py`: 基础 LoRA 训练脚本（已更新默认 batch size）
- `scripts/run_full_lora_experiment.sh`: 完整 LoRA 实验脚本（已更新默认 batch size）