#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量预测多个表格的脚本
用于预测/home/ruochen/data/medtable_术前特征_processed_naoji.xlsx,
/home/ruochen/data/medtable_术前特征_processed_shenjing.xlsx,
/home/ruochen/data/medtable_术前特征_processed_yanzhong.xlsx三个表格的结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_med.single_label_inference import SingleLabelLimiXInference
import argparse

def predict_single_tables(model_path, seeds):
    """
    预测三个单标签表格的结果
    
    Args:
        model_path: LimiX模型路径
        seeds: 随机种子列表
    """
    # 定义表格路径和对应的目标列
    table_configs = [
        {
            'file_path': '/home/ruochen/data/medtable_术前特征_processed_naoji.xlsx',
            'target_col': 'gold label',
            'output_dir': 'results/single_label_predict/naoji'
        },
        {
            'file_path': '/home/ruochen/data/medtable_术前特征_processed_shenjing.xlsx',
            'target_col': 'gold label',
            'output_dir': 'results/single_label_predict/shenjing'
        },
        {
            'file_path': '/home/ruochen/data/medtable_术前特征_processed_yanzhong.xlsx',
            'target_col': 'gold label',
            'output_dir': 'results/single_label_predict/yanzhong'
        }
    ]
    
    # 创建分析器
    analyzer = SingleLabelLimiXInference(model_path=model_path)
    
    # 对每个表格执行预测
    for config in table_configs:
        print(f"\n{'='*60}")
        print(f"开始处理表格: {os.path.basename(config['file_path'])}")
        print(f"目标列: {config['target_col']}")
        print(f"输出目录: {config['output_dir']}")
        print(f"{'='*60}")
        
        results = analyzer.run_analysis(
            file_path=config['file_path'],
            target_col=config['target_col'],
            seeds=seeds,
            output_dir=config['output_dir']
        )
        print(f"✓ 成功完成 {os.path.basename(config['file_path'])} 的预测")

def main():
    parser = argparse.ArgumentParser(description='批量预测多个单标签表格的结果')
    parser.add_argument('--model_path', type=str, 
                       default="/home/ruochen/models/limix/LimiX-16M.ckpt",
                       help='LimiX模型路径')
    parser.add_argument('--seeds', nargs='+', type=int, 
                       default=[10, 20, 30, 40, 50],
                       help='随机种子列表')
    
    args = parser.parse_args()
    
    print("开始批量预测任务")
    print(f"模型路径: {args.model_path}")
    print(f"随机种子: {args.seeds}")
    
    # 预测单标签表格
    predict_single_tables(args.model_path, args.seeds)
    
    print("\n" + "="*60)
    print("所有预测任务完成！")
    print("="*60)

if __name__ == "__main__":
    main()