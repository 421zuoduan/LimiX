#!/bin/bash

# 完整的LoRA微调实验脚本
# 用于对LimiX模型的最后两层TransformerBlock进行LoRA微调

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
OUTPUT_DIR="./results/lora"
RANK=16
ALPHA=32.0
DROPOUT=0.1
LEARNING_RATE=1e-4
BATCH_SIZE=32
NUM_EPOCHS=10
DEVICE="cuda"

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
        --rank)
            RANK="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
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
            echo "  --rank INT              LoRA秩 (默认: $RANK)"
            echo "  --alpha FLOAT           LoRA缩放因子 (默认: $ALPHA)"
            echo "  --dropout FLOAT         Dropout概率 (默认: $DROPOUT)"
            echo "  --learning_rate FLOAT   学习率 (默认: $LEARNING_RATE)"
            echo "  --batch_size INT       批次大小 (默认: $BATCH_SIZE)"
            echo "  --num_epochs INT       训练轮数 (默认: $NUM_EPOCHS)"
            echo "  --device STR            设备 (默认: $DEVICE)"
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
echo "LimiX LoRA微调实验"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "目标列: $TARGET_COLS"
echo "随机种子: $SEEDS"
echo "输出目录: $OUTPUT_DIR"
echo "LoRA配置:"
echo "  秩: $RANK"
echo "  缩放因子: $ALPHA"
echo "  Dropout: $DROPOUT"
echo "  学习率: $LEARNING_RATE"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  设备: $DEVICE"
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
echo "实验开始时间: $(date)"

# 运行LoRA微调实验
python train/lora/run_lora_finetuning.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --target_cols $TARGET_COLS \
    --seeds $SEEDS \
    --output_dir "$OUTPUT_DIR" \
    --rank $RANK \
    --alpha $ALPHA \
    --dropout $DROPOUT \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --device $DEVICE

# 检查实验是否成功
if [ $? -eq 0 ]; then
    echo "✓ LoRA微调实验成功完成！"
    
    # 记录结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo "实验结束时间: $(date)"
    echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    
    # 显示结果文件位置
    echo ""
    echo "结果文件位置:"
    echo "  详细结果: $OUTPUT_DIR/all_results.csv"
    echo "  汇总结果: $OUTPUT_DIR/summary.csv"
    
    # 显示汇总统计
    if [ -f "$OUTPUT_DIR/summary.csv" ]; then
        echo ""
        echo "结果汇总:"
        cat "$OUTPUT_DIR/summary.csv"
    fi
else
    echo "✗ LoRA微调实验失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "实验完成"
echo "=========================================="