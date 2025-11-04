# README_MYSELF

## inference_med 模块
- **目标**：快速评估医疗表格数据，支持单标签与多标签任务。
- **接口**：`SingleLabelLimiXInference`、`MultiLabelLimiXInference` 封装了数据加载、预处理、模型推理、指标汇总。
- **指标输出**：结果保存在 `results/multi_label_predict/`，summary CSV 按 *Element-wise* 与 *Subset/Samples* 两行展示，指标统一使用 `mean±std`。
- **运行示例**：
  ```bash
  python inference_med/multi_label_inference.py \
    --model_path /path/to/LimiX-16M.ckpt \
    --file_path /path/to/data.xlsx \
    --target_cols label1 label2 label3 \
    --seeds 10 20 30 40 50
  ```
- **常见使用场景**：医疗并发症预测、多随机种子稳定性评估、与 LoRA 微调结果对比。

## train/lora 模块
- **目标**：对 LimiX 模型进行轻量 LoRA 微调，使其适配特定医疗任务。
- **核心脚本**：`train_lora.py` 定义单次训练流程，`run_lora_finetuning.py` 统一处理多种子实验，`lora_hyperparameter_search.py` 负责多参数组合搜索并自动比较不同 epoch。
- **指标输出**：超参搜索只为每组参数生成一份轻量目录 `results/lora_hyperparameter_search/combo_*/`，其中 `epoch_summary.csv` 记录 10/20/30 epoch 的多种子均值与方差；全局最佳的 LoRA 权重会覆盖式地写入 `results/lora_hyperparameter_search/global_best_models/`。
- **检查点输出**：单次训练会自动在 `best_lora_*` 系列文件中记录验证集表现最优的权重：`best_lora` 对应最优 val loss（含完整 checkpoint 与 LoRA-only 权重），`best_accuracy_subset` / `best_accuracy_elementwise`、`best_f1_macro`、`best_f1_micro`、`best_roc_auc_macro`、`best_roc_auc_micro` 会额外输出 LoRA 权重及对应的 `*_metrics.json` 说明文件（含触发 epoch、seed、当时的验证指标），方便按需挑选。
- **评估策略**：每个参数组合仅训练一次 30 epoch，同时缓存第 10/20/30 epoch；对这些 epoch 先做跨种子平均，再以验证集 loss（并在需要时用 subset accuracy 打破平手）挑出最终成绩和需保留的权重。
- **GPU 与性能注意**：虽只训练 LoRA 权重，但仍需完整前向；建议使用较大的 batch、开启 CUDA 混合精度以提升吞吐。
- **运行示例**：
  ```bash
  python train/lora/run_lora_finetuning.py \
    --model_path /path/to/LimiX-16M.ckpt \
    --data_path /path/to/data.xlsx \
    --target_cols label1 label2 label3 \
    --seeds 10 \
    --num_epochs 3 \
    --batch_size 64 \
    --output_dir results/lora/my_experiment
  ```
- **与推理对齐**：LoRA 训练和 inference_med 共享 `Element-wise` 与 `Subset/Samples` 两套指标，可直接比对微调增益。
- **调参建议**：
  - **训练配置**：针对超参搜索，`--num_epochs` 固定为 30 以一次性获得 10/20/30 epoch 表现；自定义脚本中仍可调节 epoch、批量或学习率（1e-4 ↔︎ 5e-4 / 5e-5），并配合 warmup/余弦退火。
  - **LoRA 结构**：探索更高 `rank`、`alpha` 或增加 `target_layers`（例如 [8, 9, 10, 11]），必要时仅对注意力/MLP 子层挂 LoRA。
  - **损失与样本**：对极度不平衡标签设置 `BCEWithLogitsLoss(pos_weight)`，或在 DataLoader 中上采样正例。
  - **推理配置**：测试 `cls_default_retrieval.json`、`softmax_temperature`、阈值校准等，使推理阶段更贴近 LoRA 训练。
  - **稳定性**：多随机种子取 `mean±std`；如需更稳健评估可改 K-fold 或增加验证集比例。

## 指标解读
- **统一输出字段**：当前训练日志与 CSV 会同时给出以下指标（若无有效样本则置为 0 或 `-`）：
  - `accuracy_elementwise`：标签级准确率（逐标签命中率）。
  - `accuracy_subset`：样本级准确率（所有标签全部正确才算命中）。
  - `precision_macro_elementwise` / `recall_macro_elementwise` / `f1_macro_elementwise`：标签级宏平均指标，先对每个标签计算，再取平均。
  - `precision_samples_subset` / `recall_samples_subset` / `f1_samples_subset`：样本级（subset/samples）指标，以整条样本为单位衡量。
  - `f1_micro_elementwise`：标签级微平均 F1，等价于在所有标签上累积 TP/FP/FN 后再计算 F1。
  - `roc_auc_macro_elementwise`：标签级宏平均 ROC-AUC。
  - `roc_auc_micro_elementwise`：标签级微平均 ROC-AUC，与 `f1_micro_elementwise` 一样先汇总样本再计算。
  - `roc_auc_samples_subset`：样本级 ROC-AUC。
  - `avg_precision_macro_elementwise` / `avg_precision_samples_subset`：分别对应标签级与样本级的平均精度（Average Precision）。
- **Element-wise（标签视角）**：对每个标签独立计算 Precision/Recall/F1/ROC-AUC/AP，再取宏平均；`accuracy_elementwise` 表示逐标签命中率，`f1_micro_elementwise` 与 `roc_auc_micro_elementwise` 会合并所有标签做微平均。
- **Subset/Samples（样本视角）**：把整条样本标签集合视为一个子集，计算样本级准确率（exact match）及以样本为单位的 Precision/Recall/F1/ROC-AUC/AP。
- **宏/微平均差异**：宏平均（macro）先对每个标签单独算指标再做等权平均，更关注小样本标签；微平均（micro）先累计各标签的 TP/FP/FN（或 TPR/FPR 曲线）再统一计算，样本量大的标签影响更大，反映整体性能。
- **输出统一格式**：所有指标使用 `mean±std`，便于直接写入报告或论文；缺失指标以 `-` 表示。

## 推荐工作流
1. 使用 `inference_med` 获取 LimiX 基线元素级/样本级性能。
2. 运行 `train/lora/run_lora_finetuning.py` 做任务定制；比对 `results` 与 `summary` CSV 了解微调增益。
3. 在推理阶段加载保存的 LoRA 权重（单次训练位于 `best_lora_weights.pth`，超参搜索位于 `global_best_models/{target}_best_weights.pth`）搭配基础模型，降低部署负担。
