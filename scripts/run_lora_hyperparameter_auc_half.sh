#!/bin/bash

# LoRA AUC 超参搜索脚本（上半部分 TransformerBlock 6-11）

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${MODEL_PATH:=/home/ruochen/models/limix/LimiX-16M.ckpt}"
: "${DATA_PATH:=/home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx}"
: "${TARGET_COLS:=脑脊液漏 神经功能加重 严重并发症（死亡＋瘫痪＋再手术＋再入院）}"
: "${SEEDS:=10 20 30 40 50}"
: "${OUTPUT_DIR:=./results/lora_auc_half_search}"
: "${TARGET_LAYERS:=6 7 8 9 10 11}"

# 搜索网格（逗号分隔字符串）
: "${LEARNING_RATES:=5e-5,1e-4}"
: "${AUC_COEFFS:=0.005,0.01,0.02}"
: "${AUC_START_EPOCHS:=8,12,15}"
: "${AUC_GAMMAS:=0.5,1.0}"

: "${RANK:=64}"
: "${ALPHA:=128}"
: "${DROPOUT:=0.0}"
: "${BATCH:=64}"
: "${EPOCHS:=10}"
: "${DEVICE:=cuda}"
: "${VAL_RATIO:=0.2}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

python train/lora/lora_hyperparameter_auc_half.py \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --target_cols $TARGET_COLS \
  --seeds $SEEDS \
  --output_dir "$OUTPUT_DIR" \
  --target_layers $TARGET_LAYERS \
  --learning_rates "$LEARNING_RATES" \
  --auc_coeffs "$AUC_COEFFS" \
  --auc_start_epochs "$AUC_START_EPOCHS" \
  --auc_gammas "$AUC_GAMMAS" \
  --rank $RANK \
  --alpha $ALPHA \
  --dropout $DROPOUT \
  --batch_size $BATCH \
  --num_epochs $EPOCHS \
  --device $DEVICE \
  --val_ratio $VAL_RATIO

echo "完成，结果位于 $OUTPUT_DIR"
