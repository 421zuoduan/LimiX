import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import train_test_split

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

        # 打印模型结构
        print("\n" + "="*80)
        print("LoRA模型结构信息")
        print("="*80)
        print(f"基础模型类型: {type(base_model).__name__}")
        print(f"LoRA目标层: {self.lora_target_layers}")
        print(f"LoRA rank: {self.lora_rank}")
        print(f"LoRA alpha: {self.lora_alpha}")
        print(f"LoRA dropout: {self.lora_dropout}")
        print(f"LoRA权重路径: {self.lora_weights_path}")
        
        print("\n完整模型结构:")
        print(lora_model)
        
        # 打印LoRA参数信息
        print("\nLoRA参数信息:")
        lora_params = []
        for name, param in lora_model.named_parameters():
            if 'lora' in name.lower():
                lora_params.append((name, param.shape, param.requires_grad))
        
        if lora_params:
            print("LoRA层参数:")
            for name, shape, requires_grad in lora_params:
                print(f"  {name}: {shape}, 训练中: {requires_grad}")
        else:
            print("未找到LoRA参数")
        
        # 打印模型总参数量
        total_params = sum(p.numel() for p in lora_model.parameters())
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        print(f"\n模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"LoRA参数占比: {trainable_params/total_params*100:.4f}%")
        print("="*80 + "\n")

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


if __name__ == "__main__":
    # 示例路径需根据环境调整
    MODEL_PATH = "/home/ruochen/models/limix/LimiX-16M.ckpt"
    DATA_PATH = "/home/ruochen/data/medtable_术前特征_processed-多变量预测.xlsx"
    TARGET_COLS = ["脑脊液漏", "神经功能加重", "严重并发症（死亡＋瘫痪＋再手术＋再入院）"]
    LORA_METADATA = (
        "results/lora_hyperparameter_search_64_rank/global_best_models/"
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
