#!/bin/bash

# LoRA 超参搜索脚本（上半部分 TransformerBlock 6-11）

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${MODEL_PATH:=/home/ruochen/models/limix/LimiX-16M.ckpt}"
: "${DATA_PATH:=/home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx}"
: "${TARGET_COLS:=脑脊液漏 神经功能加重 严重并发症（死亡＋瘫痪＋再手术＋再入院）}"
: "${SEEDS:=10 20 30 40 50}"
: "${OUTPUT_DIR:=./results/lora_hyperparameter_half}"
: "${TARGET_LAYERS:=6 7 8 9 10 11}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

python train/lora/lora_hyperparameter_half.py \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --target_cols $TARGET_COLS \
  --seeds $SEEDS \
  --output_dir "$OUTPUT_DIR" \
  --target_layers $TARGET_LAYERS

echo "完成，结果位于 $OUTPUT_DIR"
