#!/bin/bash

# 批量预测三个医疗表格的脚本
# 用于对三个文件进行single_label_predict

# 设置脚本在遇到错误时退出
set -e

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 设置默认参数
MODEL_PATH="/home/ruochen/models/limix/LimiX-16M.ckpt"
SEEDS="10 20 30 40 50"

# 解析命令行参数
while [ $# -gt 0 ]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --model_path PATH    LimiX模型路径 (默认: /home/ruochen/models/limix/LimiX-16M.ckpt)"
            echo "  --seeds \"10 20...\"   随机种子列表 (默认: \"10 20 30 40 50\")"
            echo "  -h, --help           显示此帮助信息"
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
echo "医疗表格批量预测脚本"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "随机种子: $SEEDS"
echo "项目根目录: $PROJECT_ROOT"
echo "=========================================="

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 定义表格配置
TABLE_CONFIG_1="/home/ruochen/data/medtable_术前特征_processed_naoji.xlsx,naoji"
TABLE_CONFIG_2="/home/ruochen/data/medtable_术前特征_processed_shenjing.xlsx,shenjing"
TABLE_CONFIG_3="/home/ruochen/data/medtable_术前特征_processed_yanzhong.xlsx,yanzhong"

# 对每个表格执行预测
for config in "$TABLE_CONFIG_1" "$TABLE_CONFIG_2" "$TABLE_CONFIG_3"; do
    # 分割配置获取文件路径和输出目录名
    FILE_PATH=$(echo "$config" | cut -d',' -f1)
    OUTPUT_DIR=$(echo "$config" | cut -d',' -f2)
    
    echo ""
    echo "=========================================="
    echo "开始处理表格: $(basename "$FILE_PATH")"
    echo "输出目录: results/single_label_predict/$OUTPUT_DIR"
    echo "=========================================="
    
    # 执行预测
    RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12355 python inference_med/single_label_multi_tabs_inference.py \
        --model_path "$MODEL_PATH" \
        --seeds $SEEDS
    
    if [ $? -eq 0 ]; then
        echo "✓ 成功完成 $(basename "$FILE_PATH") 的预测"
    else
        echo "✗ 处理 $(basename "$FILE_PATH") 时出错"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "所有预测任务完成！"
echo "=========================================="