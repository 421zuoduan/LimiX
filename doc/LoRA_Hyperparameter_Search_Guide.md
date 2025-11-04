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
- **num_epochs**: [30] - 训练轮数（保存中间检查点）
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
│   └── epoch_summary.csv             # 记录 10/20/30 epoch 的均值与标准差
├── combo_2/
│   └── epoch_summary.csv
├── global_best_models/                    # 全局最佳模型目录
│   ├── {target_name}_best_weights.pth   # 当前最优LoRA权重（仅保留一份）
│   └── {target_name}_best_metadata.json # 对应参数与指标
└── {target_name}_hyperparameter_search_results.csv
```

**注意**：过程中不会为每个种子/epoch 单独落盘 LoRA 权重，而是仅保留聚合指标与全局最佳的一份权重文件。

### 3. 最佳参数评估

#### 评估指标

系统使用以下指标评估模型性能：

- **主要评估指标**：
  - `val_loss`: 验证损失（越小越好，作为主要选择标准）

- **准确率指标**：
  - `accuracy_subset`: 子集准确率（所有标签都预测正确才算正确）
  - `accuracy_elementwise`: 元素级准确率（每个标签单独计算准确率）

- **F1分数指标**：
  - `f1_macro_elementwise`: 宏平均F1分数
  - `f1_micro_elementwise`: 微平均F1分数
  - `f1_samples_subset`: 样本级F1分数

- **ROC AUC指标**：
  - `roc_auc_macro_elementwise`: 宏平均ROC-AUC
  - `roc_auc_micro_elementwise`: 微平均ROC-AUC
  - `roc_auc_samples_subset`: 样本级ROC-AUC

- **平均精度指标**：
  - `avg_precision_macro_elementwise`: 宏平均精度
  - `avg_precision_samples_subset`: 样本级精度

#### 最佳模型选择机制

**单个参数组合内的最佳epoch选择**：
1. 单次 30 epoch 训练，仅在第 10/20/30（以及最终 epoch）捕获指标与 LoRA 权重快照（保存在内存中）
2. 聚合所有种子在这些 epoch 上的均值与标准差
3. **基于验证损失（val_loss）选择最佳epoch**（越小越好）
4. 当 val_loss 相近时，使用 accuracy_subset 作为次要标准
5. 仅在该参数组合优于当前全局最佳时，才将对应 epoch 的 LoRA 权重写入 `global_best_models/`

**全局最佳参数组合选择**：
1. 比较各参数组合的最佳epoch性能
2. **基于验证损失（val_loss）选择全局最佳参数组合**（越小越好）
3. 当val_loss相近时，使用accuracy_subset作为次要标准
4. 当发现更好的参数组合时，自动更新全局最佳模型
5. 将最佳模型保存到统一目录 `output_dir/global_best_models/`

全局最佳目录下的 `*_best_metadata.json` 会记录触发该权重的参数组合、最佳指标以及对应的种子编号，方便回溯单次运行。

#### 评估指标选择说明

选择验证损失作为主要指标的原因：
- **直接反映训练目标**：Val loss 直接反映模型在验证集上的预测误差
- **稳定性**：相比准确率等指标，loss 通常更稳定，受阈值选择影响较小
- **数值连续性**：Loss 是连续值，更适合进行精细的比较和优化
- **避免过拟合检测**：Val loss 的上升通常表明过拟合

次要指标（accuracy_subset）用于在val_loss相近时进行最终选择，确保模型具有良好的实际预测性能。

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

### 4. 中间Epoch记录策略

- **训练30个epoch**: 在第10、20、30（以及最终epoch）记录验证指标与LoRA权重快照（内存中保存）
- **训练20个epoch**: 在第10、20（以及最终epoch）记录
- **训练少于20个epoch**: 仅记录可用的目标epoch与最终结果
- **优势**: 无需落盘多个模型文件即可比较不同epoch的性能，快速选出最优训练长度

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
Parameters: {'rank': 8, 'dropout': 0.0, 'learning_rate': 1e-05, 'batch_size': 64, 'num_epochs': 30}

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
