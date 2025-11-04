import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predictor import LimiXPredictor  # noqa: E402
from inference_med.multi_label_inference import MultiLabelLimiXInference  # noqa: E402

# Add LoRA utilities to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LORA_DIR = PROJECT_ROOT / "train" / "lora"
if str(LORA_DIR) not in sys.path:
    sys.path.append(str(LORA_DIR))

from lora import LoRAEnabledTransformer  # noqa: E402

import warnings

warnings.filterwarnings("ignore")


class MultiLabelLimiXLoraInference(MultiLabelLimiXInference):
    """
    多标签推理器，支持加载LoRA微调后的LimiX模型。
    """

    def __init__(
        self,
        model_path: str,
        lora_metadata_path: str,
        config_path: str = "config/cls_default_noretrieval.json",
        lora_weights_path: Optional[str] = None,
        target_layers: Optional[List[int]] = None,
    ):
        """
        初始化LoRA多标签推理器。

        Args:
            model_path: LimiX基础模型参数路径
            lora_metadata_path: LoRA微调元数据JSON路径
            config_path: 推理配置文件路径
            lora_weights_path: LoRA权重路径，如为空则从元数据加载
            target_layers: 需要注入LoRA的Transformer层索引
        """
        super().__init__(model_path=model_path, config_path=config_path)

        self.project_root = PROJECT_ROOT
        self.lora_metadata_path = Path(lora_metadata_path).resolve()
        self.lora_metadata = self._load_lora_metadata(self.lora_metadata_path)
        self.lora_weights_path = self._resolve_lora_weights_path(
            override_path=lora_weights_path
        )
        self.lora_params = self.lora_metadata.get("best_params", {})
        self.lora_rank = int(self.lora_params.get("rank", 16))
        self.lora_dropout = float(self.lora_params.get("dropout", 0.1))
        self.lora_alpha = self.lora_params.get("alpha")  # 默认由LoRA模块处理
        metadata_target_layers = self.lora_params.get("target_layers")

        if target_layers is not None:
            self.lora_target_layers = [int(idx) for idx in target_layers]
        elif metadata_target_layers is not None:
            self.lora_target_layers = [int(idx) for idx in metadata_target_layers]
        else:
            self.lora_target_layers = [10, 11]  # 训练脚本默认值

    @staticmethod
    def _load_lora_metadata(metadata_path: Path) -> Dict:
        if not metadata_path.exists():
            raise FileNotFoundError(f"LoRA元数据文件不存在: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_lora_weights_path(self, override_path: Optional[str]) -> Path:
        """
        根据优先级解析LoRA权重路径。
        """
        candidates = []

        if override_path:
            candidates.append(Path(override_path).expanduser())
        else:
            weights_path = self.lora_metadata.get("weights_path")
            if weights_path:
                candidates.append(Path(weights_path))

        metadata_dir = self.lora_metadata_path.parent
        # 常见备选：元数据同目录内的权重文件
        if "weights_path" in self.lora_metadata:
            candidates.append(metadata_dir / Path(self.lora_metadata["weights_path"]).name)

        # 保底：直接使用元数据目录下的 *_best_weights.pth
        for file in metadata_dir.glob("*_best_weights.pth"):
            candidates.append(file)

        for candidate in candidates:
            if not candidate:
                continue
            if not candidate.is_absolute():
                # 尝试以项目根目录为基准解析
                absolute_candidate = (self.project_root / candidate).resolve()
                if absolute_candidate.exists():
                    return absolute_candidate
                # 再尝试相对元数据目录
                absolute_candidate = (self.lora_metadata_path.parent / candidate).resolve()
                if absolute_candidate.exists():
                    return absolute_candidate
            elif candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            f"无法定位LoRA权重文件，请检查元数据或手动指定路径。尝试过的路径: {candidates}"
        )

    def _attach_lora(self, predictor: LimiXPredictor) -> LimiXPredictor:
        """
        将LoRA权重注入到预测器模型中。
        """
        base_model = predictor.model
        lora_model = LoRAEnabledTransformer(
            base_model=base_model,
            target_layers=self.lora_target_layers,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        lora_model.load_lora_weights(str(self.lora_weights_path))

        # 保持设备属性，兼容下游检索流程
        if hasattr(base_model, "device"):
            lora_model.device = base_model.device

        # # 打印模型结构
        # print("\n" + "="*80)
        # print("LoRA模型结构信息")
        # print("="*80)
        # print(f"基础模型类型: {type(base_model).__name__}")
        # print(f"LoRA目标层: {self.lora_target_layers}")
        # print(f"LoRA rank: {self.lora_rank}")
        # print(f"LoRA alpha: {self.lora_alpha}")
        # print(f"LoRA dropout: {self.lora_dropout}")
        # print(f"LoRA权重路径: {self.lora_weights_path}")
        
        # print("\n完整模型结构:")
        # print(lora_model)
        
        # # 打印LoRA参数信息
        # print("\nLoRA参数信息:")
        # lora_params = []
        # for name, param in lora_model.named_parameters():
        #     if 'lora' in name.lower():
        #         lora_params.append((name, param.shape, param.requires_grad))
        
        # if lora_params:
        #     print("LoRA层参数:")
        #     for name, shape, requires_grad in lora_params:
        #         print(f"  {name}: {shape}, 训练中: {requires_grad}")
        # else:
        #     print("未找到LoRA参数")
        
        # # 打印模型总参数量
        # total_params = sum(p.numel() for p in lora_model.parameters())
        # trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        # print(f"\n模型总参数量: {total_params:,}")
        # print(f"可训练参数量: {trainable_params:,}")
        # print(f"LoRA参数占比: {trainable_params/total_params*100:.4f}%")
        # print("="*80 + "\n")

        predictor.model = lora_model
        return predictor

    def run_single_seed(self, X: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, float]:
        """
        在单个种子下执行LoRA多标签推理流程。
        """
        print(f"\n=== 种子 {seed} (LoRA) 开始 ===")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=None
        )

        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

        n_labels = y.shape[1]
        predictions = np.zeros((X_test.shape[0], n_labels), dtype=np.float32)

        for i in range(n_labels):
            print(f"  训练标签 {i + 1}/{n_labels} (LoRA)")
            clf = LimiXPredictor(
                device=self.device,
                model_path=self.model_path,
                inference_config=self.config_path,
                task_type="Classification",
                seed=seed,
            )

            clf = self._attach_lora(clf)

            pred_proba = clf.predict(X_train, y_train[:, i], X_test)

            if pred_proba.shape[1] == 2:
                predictions[:, i] = pred_proba[:, 1]
            else:
                predictions[:, i] = pred_proba[:, 0]

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
    
    def _summarize_results(self, all_metrics: List[Dict[str, float]], target_cols: List[str]):
        """
        汇总和打印结果，并保存到指定目录（LoRA版本）
        
        Args:
            all_metrics: 所有种子的性能指标
            target_cols: 目标变量列名列表
        """
        print("\n" + "="*50)
        print("最终结果汇总 (LoRA)")
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
        output_path = f'results/multi_label_predict_lora/{target_name}_results.csv'
        summary_path = f'results/multi_label_predict_lora/{target_name}_summary.csv'
        
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
        
        # 保存LoRA元数据信息
        lora_metadata_path = f'results/multi_label_predict_lora/{target_name}_lora_metadata.json'
        lora_metadata_info = {
            'lora_metadata_path': str(self.lora_metadata_path),
            'lora_weights_path': str(self.lora_weights_path),
            'lora_params': self.lora_params,
            'lora_rank': self.lora_rank,
            'lora_dropout': self.lora_dropout,
            'lora_alpha': self.lora_alpha,
            'lora_target_layers': self.lora_target_layers,
            'model_path': self.model_path,
            'config_path': self.config_path
        }
        
        with open(lora_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(lora_metadata_info, f, ensure_ascii=False, indent=2)
        print(f"LoRA元数据已保存至 {lora_metadata_path}")
    
    def run_analysis(self, file_path: str, target_cols: List[str], seeds: List[int] = [10, 20, 30, 40, 50]):
        """
        执行完整分析流程（LoRA版本）
        
        Args:
            file_path: Excel文件路径
            target_cols: 目标变量列名列表
            seeds: 随机种子列表
        """
        print("开始多标签预测分析 (LoRA)")
        print(f"模型路径: {self.model_path}")
        print(f"配置文件: {self.config_path}")
        print(f"LoRA元数据路径: {self.lora_metadata_path}")
        print(f"LoRA权重路径: {self.lora_weights_path}")
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


if __name__ == "__main__":
    # 示例路径需根据环境调整
    MODEL_PATH = "/home/ruochen/models/limix/LimiX-16M.ckpt"
    DATA_PATH = "/home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx"
    TARGET_COLS = ["脑脊液漏", "神经功能加重", "严重并发症（死亡＋瘫痪＋再手术＋再入院）"]
    LORA_METADATA = (
        "results/lora_hyperparameter_search/global_best_models/"
        "脑脊液漏_神经功能加重_严重并发症（死亡＋瘫痪＋再手术＋再入院）_best_metadata.json"
    )
    SEEDS = [10, 20, 30, 40, 50]

    analyzer = MultiLabelLimiXLoraInference(
        model_path=MODEL_PATH,
        lora_metadata_path=LORA_METADATA,
    )
    analyzer.run_analysis(
        file_path=DATA_PATH,
        target_cols=TARGET_COLS,
        seeds=SEEDS,
    )
