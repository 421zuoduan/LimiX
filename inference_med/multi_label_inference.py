import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
import torch
from typing import Dict, List, Tuple
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.predictor import LimiXPredictor
import warnings
warnings.filterwarnings('ignore')

class MultiLabelLimiXInference:
    """
    使用LimiX进行多标签预测的完整分析流程
    """
    
    def __init__(self, model_path: str, config_path: str = "config/cls_default_noretrieval.json"):
        """
        初始化多标签推理器
        
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
    
    def load_data(self, file_path: str, target_cols: List[str], feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载Excel数据
        
        Args:
            file_path: Excel文件路径
            target_cols: 目标变量列名列表
            feature_cols: 特征列名列表，如果为None则使用所有非目标列
            
        Returns:
            X: 特征矩阵
            y: 多标签目标矩阵
        """
        print(f"正在加载数据: {file_path}")
        df = pd.read_excel(file_path)
        
        # 提取目标变量
        y = df[target_cols].values.astype(np.float32)
        
        # 提取特征
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in target_cols]
        X = df[feature_cols].values
        
        print(f"数据形状 - 特征: {X.shape}, 目标: {y.shape}")
        return X, y
    
    def run_single_seed(self, X: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, float]:
        """
        在单个种子下运行完整分析流程
        
        Args:
            X: 特征矩阵
            y: 多标签目标矩阵
            seed: 随机种子
            
        Returns:
            性能指标字典
        """
        print(f"\n=== 种子 {seed} 开始 ===")
        
        # 划分训练集和测试集 (80:20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=None
        )
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 为每个标签训练独立的分类器
        n_labels = y.shape[1]
        predictions = np.zeros((X_test.shape[0], n_labels))
        
        for i in range(n_labels):
            print(f"  训练标签 {i+1}/{n_labels}")
            
            # 创建预测器
            clf = LimiXPredictor(
                device=self.device,
                model_path=self.model_path,
                inference_config=self.config_path,
                task_type='Classification',
                seed=seed
            )
            
            # 执行预测
            pred_proba = clf.predict(X_train, y_train[:, i], X_test)
            
            # 获取正类概率 (假设二元分类)
            if pred_proba.shape[1] == 2:
                predictions[:, i] = pred_proba[:, 1]
            else:
                predictions[:, i] = pred_proba[:, 0]
        
        # 计算性能指标
        metrics = self._calculate_metrics(y_test, predictions, seed)
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, seed: int) -> Dict[str, float]:
        """
        计算多标签分类性能指标（使用sklearn标准方法）
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            seed: 随机种子
            
        Returns:
            指标字典
        """
        metrics = {'seed': seed}
        
        def safe_metric(func, *args, default=np.nan, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as err:
                print(f"{func.__name__}计算失败: {err}")
                return default
        
        # ROC-AUC 指标
        metrics['roc_auc_macro'] = safe_metric(
            roc_auc_score, y_true, y_pred_proba, average='macro'
        )
        metrics['roc_auc_micro'] = safe_metric(
            roc_auc_score, y_true, y_pred_proba, average='micro'
        )
        metrics['roc_auc_samples'] = safe_metric(
            roc_auc_score, y_true, y_pred_proba, average='samples'
        )
        
        # F1 Score (使用0.5阈值)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics['f1_macro'] = safe_metric(
            f1_score, y_true, y_pred, average='macro', zero_division=0
        )
        metrics['f1_micro'] = safe_metric(
            f1_score, y_true, y_pred, average='micro', zero_division=0
        )
        metrics['f1_samples'] = safe_metric(
            f1_score, y_true, y_pred, average='samples', zero_division=0
        )
        
        # Accuracy
        accuracy_macro = (y_pred == y_true).mean()
        accuracy_micro = (y_pred == y_true).mean()  # 对于多标签，micro和macro计算方式相同
        accuracy_samples = (y_pred == y_true).all(axis=1).mean()
        metrics['accuracy_macro'] = accuracy_macro
        metrics['accuracy_micro'] = accuracy_micro
        metrics['accuracy_samples'] = accuracy_samples
        metrics['accuracy'] = accuracy_macro  # backward compatibility
        
        return metrics
    
    def run_analysis(self, file_path: str, target_cols: List[str], seeds: List[int] = [10, 20, 30, 40, 50]):
        """
        执行完整分析流程
        
        Args:
            file_path: Excel文件路径
            target_cols: 目标变量列名列表
            seeds: 随机种子列表
        """
        print("开始多标签预测分析")
        print(f"模型路径: {self.model_path}")
        print(f"配置文件: {self.config_path}")
        print(f"目标变量: {target_cols}")
        print(f"随机种子: {seeds}")
        
        # 加载数据
        X, y = self.load_data(file_path, target_cols)
        
        # 存储所有结果
        all_metrics = []
        
        # 对每个种子执行分析
        for seed in seeds:
            metrics = self.run_single_seed(X, y, seed)
            if 'accuracy_samples' in metrics and 'accuracy_macro' in metrics:
                print(f"  Samples Accuracy: {metrics['accuracy_samples']:.4f}")
                print(f"  Macro Accuracy: {metrics['accuracy_macro']:.4f}")
            all_metrics.append(metrics)
        
        # 汇总结果
        self._summarize_results(all_metrics, target_cols)
        
        return all_metrics
    
    def _summarize_results(self, all_metrics: List[Dict[str, float]], target_cols: List[str]):
        """
        汇总和打印结果，并保存到指定目录
        
        Args:
            all_metrics: 所有种子的性能指标
            target_cols: 目标变量列名列表
        """
        print("\n" + "="*50)
        print("最终结果汇总")
        print("="*50)
        
        # 转换为DataFrame便于分析
        import pandas as pd
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
                mean_display = 0.0 if pd.isna(mean_val) else float(mean_val)
                std_display = 0.0 if pd.isna(std_val) else float(std_val)
                print(f"{col}: {mean_display:.4f}±{std_display:.4f}")
        
        # 保存结果到指定目录
        # 将目标列名组合成文件名
        target_name = "_".join(target_cols)
        output_path = f'results/multi_label_predict/{target_name}_results.csv'
        summary_path = f'results/multi_label_predict/{target_name}_summary.csv'
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存详细结果（覆盖已存在的文件）
        results_df.to_csv(output_path, index=False, mode='w')
        print(f"\n详细结果已保存至 {output_path}")
        
        # 保存均值结果（mean±std），按 Macro、Micro 与 Samples 三行展示
        num_seeds = len(all_metrics)

        def format_metric(col_name: str) -> str:
            if col_name not in summary.columns:
                return "-"
            mean_val = summary.loc['mean', col_name]
            std_val = summary.loc['std', col_name]
            mean_display = 0.0 if pd.isna(mean_val) else float(mean_val)
            std_display = 0.0 if pd.isna(std_val) else float(std_val)
            return f"{mean_display:.4f}±{std_display:.4f}"

        summary_rows = [
            {
                'group': 'Macro',
                'num_seeds': num_seeds,
                'accuracy': format_metric('accuracy_macro'),
                'f1': format_metric('f1_macro'),
                'roc_auc': format_metric('roc_auc_macro')
            },
            {
                'group': 'Micro',
                'num_seeds': num_seeds,
                'accuracy': format_metric('accuracy_micro'),
                'f1': format_metric('f1_micro'),
                'roc_auc': format_metric('roc_auc_micro')
            },
            {
                'group': 'Samples',
                'num_seeds': num_seeds,
                'accuracy': format_metric('accuracy_samples'),
                'f1': format_metric('f1_samples'),
                'roc_auc': format_metric('roc_auc_samples')
            }
        ]

        summary_df = pd.DataFrame(summary_rows)
        
        summary_df.to_csv(summary_path, index=False, mode='w')
        print(f"均值结果已保存至 {summary_path}")
        print(f"注意：summary文件现在保存的是多个种子的指标平均值和标准差，而不是最高值")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "/home/ruochen/models/limix/LimiX-16M.ckpt"
    DATA_PATH = "/home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx"
    TARGET_COLS = ["脑脊液漏", "神经功能加重", "严重并发症（死亡＋瘫痪＋再手术＋再入院）"]
    SEEDS = [10, 20, 30, 40, 50]
    
    # 执行分析
    analyzer = MultiLabelLimiXInference(model_path=MODEL_PATH)
    results = analyzer.run_analysis(
        file_path=DATA_PATH,
        target_cols=TARGET_COLS,
        seeds=SEEDS
    )
