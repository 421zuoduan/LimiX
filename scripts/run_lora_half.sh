#!/bin/bash

# LoRA 微调脚本：上半部分 TransformerBlock (6-11 层)

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${MODEL_PATH:=/home/ruochen/models/limix/LimiX-16M.ckpt}"
: "${DATA_PATH:=/home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx}"
: "${TARGET_COLS:=脑脊液漏 神经功能加重 严重并发症（死亡＋瘫痪＋再手术＋再入院）}"
: "${SEEDS:=10 20 30 40 50}"
: "${OUTPUT_DIR:=./results/lora_half}"

: "${RANK:=16}"
: "${ALPHA:=32.0}"
: "${DROPOUT:=0.1}"
: "${LR:=1e-4}"
: "${BATCH:=32}"
: "${EPOCHS:=10}"
: "${DEVICE:=cuda}"
: "${TARGET_LAYERS:=6 7 8 9 10 11}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

python train/lora/train_lora_half.py \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --target_cols $TARGET_COLS \
  --seeds $SEEDS \
  --output_dir "$OUTPUT_DIR" \
  --rank $RANK \
  --alpha $ALPHA \
  --dropout $DROPOUT \
  --learning_rate $LR \
  --batch_size $BATCH \
  --num_epochs $EPOCHS \
  --device $DEVICE \
  --target_layers $TARGET_LAYERS

echo "完成，结果位于 $OUTPUT_DIR"
