#!/bin/bash

# LoRA超参数搜索脚本
# 用于对LimiX模型进行LoRA微调的超参数搜索

# 设置脚本在遇到错误时退出
set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 设置默认参数
MODEL_PATH="/home/ruochen/models/limix/LimiX-16M.ckpt"
DATA_PATH="/home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx"
TARGET_COLS="脑脊液漏 神经功能加重 严重并发症（死亡＋瘫痪＋再手术＋再入院）"
SEEDS="10 20 30 40 50"
OUTPUT_DIR="./results/lora_hyperparameter_search"

# 解析命令行参数
while [ $# -gt 0 ]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --target_cols)
            TARGET_COLS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --model_path PATH        LimiX模型路径 (默认: $MODEL_PATH)"
            echo "  --data_path PATH         数据文件路径 (默认: $DATA_PATH)"
            echo "  --target_cols COLS       目标列名 (默认: $TARGET_COLS)"
            echo "  --seeds \"10 20...\"      随机种子列表 (默认: \"$SEEDS\")"
            echo "  --output_dir DIR         输出目录 (默认: $OUTPUT_DIR)"
            echo "  -h, --help             显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 打印配置信息
echo "=========================================="
echo "LimiX LoRA超参数搜索"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "目标列: $TARGET_COLS"
echo "随机种子: $SEEDS"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 记录开始时间
START_TIME=$(date +%s)
echo "超参数搜索开始时间: $(date)"

# 运行LoRA超参数搜索
python train/lora/lora_hyperparameter_search.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --target_cols $TARGET_COLS \
    --seeds $SEEDS \
    --output_dir "$OUTPUT_DIR"

# 检查搜索是否成功
if [ $? -eq 0 ]; then
    echo "✓ LoRA超参数搜索成功完成！"
    
    # 记录结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo "搜索结束时间: $(date)"
    echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    
    # 显示结果文件位置
    echo ""
    echo "结果文件位置:"
    echo "  搜索结果: $OUTPUT_DIR/*_hyperparameter_search_results.csv"
    echo "  最佳参数: $OUTPUT_DIR/*_best_parameters.json"
    
    # 显示最佳参数
    TARGET_NAME=$(echo $TARGET_COLS | tr ' ' '_')
    BEST_PARAMS_FILE="$OUTPUT_DIR/${TARGET_NAME}_best_parameters.json"
    
    if [ -f "$BEST_PARAMS_FILE" ]; then
        echo ""
        echo "最佳参数摘要:"
        python -c "
import json
with open('$BEST_PARAMS_FILE', 'r') as f:
    params = json.load(f)
    
print('基于最佳 Subset Accuracy:')
for k, v in params['best_subset_accuracy'].items():
    print(f'  {k}: {v}')
    
print()
print('基于最佳 Element-wise Accuracy:')
for k, v in params['best_elementwise_accuracy'].items():
    print(f'  {k}: {v}')
    
print()
print('基于最佳 F1 Macro:')
for k, v in params['best_f1_macro'].items():
    print(f'  {k}: {v}')
    
print()
print('基于最佳 ROC AUC:')
for k, v in params['best_roc_auc'].items():
    print(f'  {k}: {v}')
"
    fi
else
    echo "✗ LoRA超参数搜索失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "超参数搜索完成"
echo "=========================================="