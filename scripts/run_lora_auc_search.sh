#!/bin/bash
# 简洁的 AUC 搜参脚本，默认围绕 AUC 权重/启用轮次/gamma 做网格搜索
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${MODEL_PATH:=/home/ruochen/models/limix/LimiX-16M.ckpt}"
: "${DATA_PATH:=/home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx}"
: "${TARGET_COLS:=脑脊液漏 神经功能加重 严重并发症（死亡＋瘫痪＋再手术＋再入院）}"
: "${SEEDS:=10 20 30}"
: "${OUTPUT_DIR:=./results/lora_auc_search}"

: "${RANK:=64}"
: "${ALPHA:=128}"
: "${DROPOUT:=0.0}"
: "${LR:=5e-05}"
: "${BATCH:=64}"
: "${EPOCHS:=10}"
: "${DEVICE:=cuda}"
: "${VAL_RATIO:=0.2}"

: "${AUC_COEFFS:=0.005,0.01,0.02}"
: "${AUC_STARTS:=10,15,20}"
: "${AUC_GAMMAS:=0.5,1.0}"

cd "$ROOT"

python train/lora/lora_auc_search.py \
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
  --val_ratio $VAL_RATIO \
  --auc_coeffs "$AUC_COEFFS" \
  --auc_start_epochs "$AUC_STARTS" \
  --auc_gammas "$AUC_GAMMAS"

echo "AUC 搜参完成，结果位于 $OUTPUT_DIR"
