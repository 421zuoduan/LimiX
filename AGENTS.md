# LimiX 智能代理系统

## 概述

LimiX是一个专为结构化数据建模设计的通用人工智能系统，采用Transformer架构优化，能够在统一的训练与推理框架下处理多种任务。作为LDM系列的首个模型，LimiX推动了表格学习从定制化流程向基础模型范式的转变。

## 核心能力

### 1. 多任务处理能力

LimiX支持以下核心任务：

- **分类任务**：支持二分类和多分类问题
- **回归任务**：连续值预测
- **缺失值插补**：自动识别和填充缺失数据
- **特征选择**：识别关键特征
- **样本选择**：选择代表性样本
- **因果推理**：探索变量间的因果关系

### 2. 医疗数据专用推理模块

项目包含专门的医疗数据处理模块(`inference_med`)，提供以下功能：

#### 单标签推理 (`SingleLabelLimiXInference`)
- 适用于单一目标变量的医疗预测任务
- 支持多种评估指标：准确率、精确率、召回率、F1分数、ROC-AUC
- 自动处理数据编码和预处理
- 支持多随机种子评估，提供结果稳定性分析

#### 多标签推理 (`MultiLabelLimiXInference`)
- 适用于多个目标变量的医疗预测任务
- 支持多标签分类场景，如并发症预测
- 提供宏平均和微平均评估指标
- 为每个标签训练独立的分类器

#### 批量表格处理 (`single_label_multi_tabs_inference`)
- 支持批量处理多个医疗数据表格
- 自动化工作流程，提高处理效率
- 统一的结果输出格式

## 技术架构

### 1. 核心模型架构

- **Transformer基础**：专为结构化数据优化的注意力机制，包含12个TransformerBlock
- **双重注意力**：同时作用于样本维度和特征维度
- **Token表示**：将特征X和目标Y映射为统一的token表示
- **多任务头**：支持回归和分类任务的专用输出模块

### 2. 推理系统

#### 基础推理器 (`LimiXPredictor`)
```python
class LimiXPredictor:
    def __init__(self, device, model_path, inference_config, 
                 task_type='Classification', mask_prediction=False, ...)
    def predict(self, x_train, y_train, x_test) -> np.ndarray
```

#### 高级推理功能
- **基于样本检索的集成推理**：提高预测准确性
- **分布式推理**：支持多GPU并行处理
- **混合精度计算**：优化内存使用和推理速度
- **动态预处理**：自适应数据特征的处理流程

### 3. 预处理管道

系统包含多种预处理组件：
- **特征过滤**：自动识别和过滤无效特征
- **分布重平衡**：调整特征分布以提高模型性能
- **分类特征编码**：智能处理类别变量
- **特征洗牌**：增强模型鲁棒性

## 使用场景

### 1. 医疗诊断辅助

- **疾病预测**：基于患者特征预测疾病风险
- **并发症评估**：预测多种可能的并发症
- **预后分析**：评估患者治疗后的可能结果

### 2. 临床决策支持

- **治疗方案推荐**：基于患者特征推荐最佳治疗方案
- **风险评估**：评估手术或治疗的风险
- **资源分配**：优化医疗资源分配

### 3. 医学研究

- **特征发现**：识别疾病相关的重要特征
- **因果分析**：探索疾病发展的因果关系
- **群体分析**：大规模患者群体的模式发现

## 性能优势

### 1. 准确性

- 在多个基准测试中达到SOTA性能
- 基于样本检索的集成推理提高预测稳定性
- 多随机种子评估确保结果可靠性

### 2. 效率

- 支持GPU加速和分布式推理
- 混合精度计算减少内存占用
- 批量处理功能提高工作效率

### 3. 通用性

- 单一模型支持多种任务类型
- 自适应预处理减少人工干预
- 灵活的配置系统适应不同场景

## 快速开始

### 1. 环境配置

```bash
# 使用Docker（推荐）
docker build --network=host -t limix/infe:v1 -f Dockerfile .

# 或手动安装
pip install python==3.12.7 torch==2.7.1
pip install flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install scikit-learn einops huggingface-hub matplotlib networkx numpy pandas scipy tqdm typing_extensions xgboost
```

### 2. 模型下载

```python
from huggingface_hub import hf_hub_download
model_file = hf_hub_download(repo_id="stableai-org/LimiX-16M", filename="LimiX-16M.ckpt", local_dir="./cache")
```

### 3. 基础使用示例

#### 分类任务
```python
from inference.predictor import LimiXPredictor

# 创建预测器
clf = LimiXPredictor(
    device='cuda', 
    model_path=model_file, 
    inference_config='config/cls_default_retrieval.json'
)

# 执行预测
prediction = clf.predict(X_train, y_train, X_test)
```

#### 医疗单标签预测
```python
from inference_med.single_label_inference import SingleLabelLimiXInference

# 创建分析器
analyzer = SingleLabelLimiXInference(model_path=MODEL_PATH)

# 执行分析
results = analyzer.run_analysis(
    file_path=DATA_PATH,
    target_col=TARGET_COL,
    seeds=[10, 20, 30, 40, 50]
)
```

#### 医疗多标签预测
```python
from inference_med.multi_label_inference import MultiLabelLimiXInference

# 创建分析器
analyzer = MultiLabelLimiXInference(model_path=MODEL_PATH)

# 执行分析
results = analyzer.run_analysis(
    file_path=DATA_PATH,
    target_cols=["并发症1", "并发症2", "并发症3"],
    seeds=[10, 20, 30, 40, 50]
)
```

## 配置选项

### 推理配置文件

| 配置文件 | 用途 | 特点 |
|---------|------|------|
| cls_default_retrieval.json | 分类任务（带检索） | 性能更好 |
| cls_default_noretrieval.json | 分类任务（无检索） | 速度更快 |
| reg_default_retrieval.json | 回归任务（带检索） | 性能更好 |
| reg_default_noretrieval.json | 回归任务（无检索） | 速度更快 |
| reg_default_noretrieval_MVI.json | 缺失值插补 | 专用配置 |

### 高级配置

- **混合精度**：`mix_precision=True`
- **异常值处理**：`outlier_remove_std=12`
- **Softmax温度**：`softmax_temperature=0.9`
- **分布式推理**：`inference_with_DDP=True`

## 扩展与定制

### 1. 自定义预处理

```python
# 生成自定义推理配置
from utils.inference_utils import generate_inference_config
generate_inference_config()
```

### 2. 批量处理脚本

```bash
# 单标签批量处理
bash scripts/single_label_inference.sh

# 多标签批量处理
bash scripts/multi_label_inference.sh --model_path /path/to/model --seeds "10 20 30 40 50"
```

### 3. 分布式推理

```bash
# 多GPU分布式推理
torchrun --nproc_per_node=8 inference_classifier.py \
    --save_name your_save_name \
    --inference_config_path path_to_config \
    --data_dir path_to_data
```

## 自述文档

- **快速查阅**：请参考 `README_MYSELF.md` 获取 inference_med 与 LoRA 微调的操作摘要、指标解读及推荐工作流。
- **LoRA 调参提示**：README 中附带训练/结构/损失/推理多方面的微调建议（epoch、batch、rank、pos_weight、检索配置等），方便快速定位性能提升方向；新增 AUC 优化版 `train/lora/train_lora_auc.py`，可通过 `scripts/run_lora_auc.sh` 启动，默认 rank=64、dropout=0.0、lr=5e-5、batch=64、`auc_start_epoch=15`，AUC 开关/力度由 `--auc_coeff/--auc_start_epoch/--auc_gamma` 控制。
- **LoRA 搜参（含 AUC）**：提供 `train/lora/lora_auc_search.py` 与 `scripts/run_lora_auc_search.sh`，默认在 AUC 权重/启用轮次/gamma 上做网格搜索，结果输出到 `./results/lora_auc_search/` 的配置子目录及汇总 CSV。
- **LoRA 结果判定**：超参搜索统一训练 30 epoch，并比较第 10/20/30 epoch 的多种子均值表现；表现最佳的参数组合只保留一份 LoRA 权重，自动覆盖写入 `global_best_models/`。AUC 版的单次训练仍在 `results/lora_auc/seed_*/` 输出 best_* 权重与指标。

## 技术支持

- **技术报告**：[LimiX_Technical_Report.pdf](https://github.com/limix-ldm/LimiX/blob/main/LimiX_Technical_Report.pdf)
- **学术论文**：[LimiX:Unleashing Structured-Data Modeling Capability for Generalist Intelligence](https://arxiv.org/abs/2509.03505)
- **数据集**：[bcco_cls](https://huggingface.co/datasets/stableai-org/bcco_cls) 和 [bcco_reg](https://huggingface.co/datasets/stableai-org/bcco_reg)

## 许可证

本仓库代码遵循 [Apache-2.0](LICENSE.txt) 协议开源，LimiX模型权重使用遵循Model License。学术研究完全开放，商业使用需获得授权。

## 引用

```bibtex
@article{LimiX,
  title={LimiX:Unleashing Structured-Data Modeling Capability for Generalist Intelligence},
  author={LimiXTeam},
  journal={arXiv preprint arXiv:2509.03505},
  year={2025}
}
