import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch
from typing import Dict, List, Tuple
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.predictor import LimiXPredictor
import warnings
warnings.filterwarnings('ignore')

class SingleLabelLimiXInference:
    """
    使用LimiX进行单标签预测的完整分析流程
    """
    
    def __init__(self, model_path: str, config_path: str = "config/cls_default_noretrieval.json"):
        """
        初始化单标签推理器
        
        Args:
            model_path: LimiX模型参数路径
            config_path: 推理配置文件路径
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 读取配置文件
        with open(config_path, 'r') as f:
            self.inference_config = json.load(f)
    
    def load_data(self, file_path: str, target_col: str, feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载Excel数据
        
        Args:
            file_path: Excel文件路径
            target_col: 目标变量列名
            feature_cols: 特征列名列表，如果为None则使用所有非目标列
            
        Returns:
            X: 特征矩阵
            y: 目标向量
        """
        print(f"正在加载数据: {file_path}")
        df = pd.read_excel(file_path)
        
        # 提取目标变量，使用squeeze确保是一维的
        y = df[target_col].values.squeeze().astype(np.int64)
        
        # 对目标变量进行编码
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        
        # 提取特征
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        
        return X, y
    
    def run_single_seed(self, X: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, float]:
        """
        在单个种子下运行完整分析流程
        
        Args:
            X: 特征矩阵
            y: 目标向量
            seed: 随机种子
            
        Returns:
            性能指标字典
        """
        print(f"\n=== 种子 {seed} 开始 ===")
        
        # 划分训练集和测试集 (80:20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # 创建预测器
        clf = LimiXPredictor(
            device=self.device,
            model_path=self.model_path,
            inference_config=self.config_path,
            task_type='Classification',
            seed=seed
        )
        
        # 最后一次确保目标变量形状正确
        print(f"预测前检查: y_train形状={y_train.shape}, y_test形状={y_test.shape}")
        
        # 确保 y_train 和 y_test 是一维的
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        print(f"预测前挤压后y_train形状: {y_train.shape}, y_test形状: {y_test.shape}")
        
        pred_proba = clf.predict(X_train, y_train, X_test)
        
        # 获取预测结果
        if pred_proba.shape[1] > 1:
            # 多分类情况
            y_pred = np.argmax(pred_proba, axis=1)
            # 计算ROC-AUC需要one-hot编码
            y_test_bin = self._binarize_labels(y_test, pred_proba.shape[1])
            try:
                roc_auc = roc_auc_score(y_test_bin, pred_proba, average='macro', multi_class='ovr')
            except:
                roc_auc = np.nan
        else:
            # 二分类情况
            y_pred = (pred_proba[:, 0] >= 0.5).astype(int)
            try:
                roc_auc = roc_auc_score(y_test, pred_proba[:, 0])
            except:
                roc_auc = np.nan
        
        # 计算性能指标
        metrics = {
            'seed': seed,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'roc_auc': roc_auc
        }
        
        return metrics
    
    def _binarize_labels(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        """
        将标签二值化用于多分类ROC-AUC计算
        """
        return np.eye(n_classes)[y]
    
    def run_analysis(self, file_path: str, target_col: str, seeds: List[int] = [10, 20, 30, 40, 50], output_dir: str = None):
        """
        执行完整分析流程
        
        Args:
            file_path: Excel文件路径
            target_col: 目标变量列名
            seeds: 随机种子列表
            output_dir: 输出目录，如果为None则使用默认目录
        """
        print("开始单标签预测分析")
        print(f"模型路径: {self.model_path}")
        print(f"配置文件: {self.config_path}")
        print(f"目标变量: {target_col}")
        print(f"随机种子: {seeds}")
        
        # 加载数据
        X, y = self.load_data(file_path, target_col)
        
        # 存储所有结果
        all_metrics = []
        
        # 对每个种子执行分析
        for seed in seeds:
            metrics = self.run_single_seed(X, y, seed)
            all_metrics.append(metrics)
        
        # 汇总结果
        self._summarize_results(all_metrics, target_col, output_dir)
        
        return all_metrics
    
    def _summarize_results(self, all_metrics: List[Dict[str, float]], target_col: str, output_dir: str = None):
        """
        汇总和打印结果，并保存到指定目录
        
        Args:
            all_metrics: 所有种子的性能指标
            target_col: 目标变量列名
            output_dir: 输出目录，如果为None则使用默认目录
        """
        print("\n" + "="*50)
        print("最终结果汇总")
        print("="*50)
        
        # 转换为DataFrame便于分析
        results_df = pd.DataFrame(all_metrics)
        
        # 打印每个种子的结果
        print("\n各种子性能:")
        print(results_df.to_string(index=False))
        
        # 计算平均值和标准差
        print("\n性能指标汇总 (mean ± std):")
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        summary = results_df[numeric_cols].agg(['mean', 'std']).round(4)
        
        for col in numeric_cols:
            if col != 'seed':
                mean_val = summary.loc['mean', col]
                std_val = summary.loc['std', col]
                print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
        
        # 保存结果到指定目录
        if output_dir is None:
            output_path = f'results/single_label_predict/{target_col}_results.csv'
            summary_path = f'results/single_label_predict/{target_col}_summary.csv'
        else:
            output_path = f'{output_dir}/{target_col}_results.csv'
            summary_path = f'{output_dir}/{target_col}_summary.csv'
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存详细结果
        results_df.to_csv(output_path, index=False)
        print(f"\n详细结果已保存至 {output_path}")
        
        # 保存均值结果
        summary_df = summary.loc['mean'].to_frame().T
        # 添加种子数量信息
        summary_df['num_seeds'] = len(all_metrics)
        # 重新排列列顺序，将num_seeds放在第一列
        cols = ['num_seeds'] + [col for col in summary_df.columns if col != 'num_seeds']
        summary_df = summary_df[cols]
        
        summary_df.to_csv(summary_path, index=False)
        print(f"均值结果已保存至 {summary_path}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "/home/ruochen/models/limix/LimiX-16M.ckpt"
    DATA_PATH = "/home/ruochen/data/medtable_术前特征_processed_shenjing.xlsx"
    TARGET_COL = "gold label"
    SEEDS = [10, 20, 30, 40, 50]
    # SEEDS = [10]
    
    # 执行分析
    analyzer = SingleLabelLimiXInference(model_path=MODEL_PATH)
    results = analyzer.run_analysis(
        file_path=DATA_PATH,
        target_col=TARGET_COL,
        seeds=SEEDS
    )