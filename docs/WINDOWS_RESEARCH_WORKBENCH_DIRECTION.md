# Windows压力感知算法研究工作台：整体方向与阶段定位

> 在看具体 R0—R8 前，先阅读 [验证 Workflow 总蓝图](VALIDATION_WORKFLOW_MASTER.md)：它把公共数据候选线、自研硬件真值线、每类结论的证据边界和最终验证包放在同一张图中。

## 1. 工作台定位

Windows工作台的目标不是提前写出一套看似完整的生产代码，而是用真实数据快速回答研究问题：

```text
这个方法能不能工作？
在哪些受试者、姿态和variation上有效？
为什么失败？
对降密度、噪声和坏点是否敏感？
是否值得进入WSL正式工程？
```

协作方式固定为：

```text
用户定方向与验收问题
    ↓
Codex拆解、写代码、测试、运行全量数据
    ↓
生成指标、图片、预测明细、模型和报告
    ↓
Codex解释结果与限制
    ↓
用户决定继续、修改或停止
```

## 2. Windows研究主线

### R0｜研究合同与数据版本

冻结输入数据、预测任务、标签、排除样本、矩阵方向、评价指标和随机种子。

最低输出：

```text
configs/labels/
configs/experiments/
outputs/reports/<run_id>_data_manifest.json
```

### R1｜全量Inventory

遍历PoPu Tactilus 5160个JSON，确认60名受试者、标签、variation、snapshot、shape、异常和唯一ID。

最低输出：

```text
data/processed/popu/popu_tactilus_inventory_v0.1.csv
outputs/reports/popu_tactilus_inventory_summary_v0.1.json
outputs/figures/popu_tactilus_label_distribution_v0.1.png
```

### R2｜Viewer与Quality

生成典型、边界和异常样本画廊，建立ACCEPT/WARN/REJECT质量门。

最低输出：

```text
outputs/figures/popu_posture_gallery_v0.1.png
outputs/figures/popu_abnormal_samples_v0.1.png
outputs/metrics/popu_quality_results_v0.1.csv
outputs/reports/popu_quality_summary_v0.1.json
```

### R3｜Body Contact Mask与Geometry

比较阈值方案，计算Centroid、COP、PCA axis与BBox，并叠加回真实压力图人工复核。

最低输出：

```text
outputs/figures/popu_mask_overlay_v0.1.png
outputs/figures/popu_geometry_overlay_v0.1.png
outputs/metrics/popu_geometry_results_v0.1.csv
```

### R4｜Feature Engineering

坚持一行等于一个PressureSample，生成可追溯、无明显标签泄漏的特征表。

最低输出：

```text
data/processed/popu/popu_tactilus_features_v0.1.csv
outputs/reports/popu_feature_schema_v0.1.json
```

### R5｜姿态Baseline与受试者隔离评价

比较规则、Logistic Regression和Random Forest；使用GroupKFold或LOSO，禁止随机按frame拆分。

最低输出：

```text
outputs/models/<run_id>_model.joblib
outputs/metrics/<run_id>_metrics.csv
outputs/metrics/<run_id>_subject_metrics.csv
outputs/metrics/<run_id>_predictions.csv
outputs/figures/<run_id>_confusion_matrix.png
outputs/figures/<run_id>_error_examples.png
```

### R6｜UNKNOWN、REJECT与错误分析

质量不合格进入REJECT；输入合格但模型置信度不足进入UNKNOWN；阈值只允许用validation subjects选择。

最低输出：

```text
outputs/metrics/<run_id>_confidence_thresholds.csv
outputs/reports/<run_id>_error_analysis.md
```

### R7｜鲁棒性与区域研究

执行Density Ablation、Fault Injection；PoPu segmentation先审计压力snapshot与区域标注对应关系，再进行身体区域原型。

最低输出：

```text
outputs/metrics/<run_id>_density.csv
outputs/metrics/<run_id>_fault.csv
outputs/figures/<run_id>_robustness.png
outputs/reports/<run_id>_region_audit.md
```

### R8｜研究冻结与WSL晋升包

当一个候选方法通过研究验收，生成供WSL工程化使用的冻结包。

```text
handoff/<candidate_id>/
├── ALGORITHM_SPEC.md
├── feature_schema.json
├── label_map.yaml
├── selected_config.yaml
├── split_manifest.json
├── reference_metrics.csv
├── reference_predictions.csv
├── reference_figures/
└── KNOWN_LIMITATIONS.md
```

## 3. 进入WSL的晋升门

只有同时满足以下条件才进入WSL：

- 输入输出定义明确；
- 数据、标签、split和随机种子可追溯；
- 自动测试通过；
- 至少完成一次受试者隔离评价；
- 保存逐样本结果和代表性错误；
- 参数已经冻结，不再依赖Notebook隐藏状态；
- 用户确认该方向值得工程化；
- 已知限制明确记录。

进入WSL后重新按工程接口实现和测试，不直接把Windows整个目录复制过去。

## 4. Windows研究工作台不负责证明的内容

- 自研MX传感器的真实性能；
- 气囊动作安全性和控制闭环；
- 整夜稳定性、舒适性或医疗效果；
- 未经同步真值验证的身体部位或姿态产品指标；
- 公开数据上的高分可以直接迁移到真实产品。
