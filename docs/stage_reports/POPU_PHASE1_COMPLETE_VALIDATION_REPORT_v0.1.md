# PoPu 压力感知算法 第一阶段完整验证总报告 v0.1

> **术语约定**：本文第一次出现的术语给出简短定义。PoPu = "Posture of Pressure-mat users"，指公开 Tactilus 64×27 压力矩阵数据集上的固定睡姿五分类研究。Small ResNet = 任务级小型残差网络。OOF = Out-of-Fold，受试者隔离交叉验证下每条样本在未参与训练的折得到的预测。WAR = Wrong Action Rate，拒识规则接受但实际错误的占比。OOF = Out-of-Fold 预测。CV = 交叉验证。LOSO = Leave-One-Subject-Out。

---

## 1. 封面信息与执行摘要

| 项 | 内容 |
|---|---|
| 报告版本 | v0.1 |
| 报告日期 | 2026-08-21 |
| 研究问题 | PoPu Tactilus 64×27 压力矩阵数据上的固定睡姿五分类（`empty / supine / prone / left / right`） |
| 验证阶段 | P1（数据盘点）→ P7（软件鲁棒性）主阶段及其子阶段（含 P5.2-A/B/C 神经网络子阶段） |
| 证据范围 | 仓库现有 P1–P7 阶段报告 + 机器可读 JSON/CSV + 外部 AutoDL 受治理证据包（仅引用既有哈希，不重新解压） |
| Git 提交基线 | `ed363a1`（"feat: add PoPu P7 Full evidence re-verification"） |
| 全量证据包 SHA-256 | `cbaffa74878b149e546a42826ae373442c62683af890362684f80963e7fddda1`（P7 Full 归档，672,702,773 字节，2,163 文件） |

**执行摘要（先给结论）**：

1. **做了什么**：把 P1–P7 主阶段及其子阶段串成一个从数据治理到软件扰动鲁棒性的完整闭环，覆盖盘点与质量门、几何与掩膜冻结、无标签特征工程、传统模型横向比较、神经网络公平比较（含 CPU/CUDA Smoke、Mini 筛选、Full 比较）、`UNKNOWN/REJECT` 拒识与错误分析、温度校准与三模型一致性、软件扰动鲁棒性。
2. **得到什么**：在 PoPu 公开 60 受试者 × 5,100 固定姿态记录上，Small ResNet 在受试者分组重复交叉验证（3 repeats × 5 folds，pool-first 聚合）下达到 record macro-F1 ≈ 0.9866、balanced accuracy ≈ 0.9866、最差受试者 macro-F1 ≈ 0.8825。P6 单模型拒识（`threshold=0.94`）在全部 15,018 record 上 coverage 95.10%、accepted accuracy 99.56%、WAR 0.4194%。P6.1 校准 + 一致性规则（`T=0.75`, `threshold=0.5`, `require_unanimous=true`）在 20 名独立受试者上 coverage 96.76%、accepted accuracy 99.49%、WAR 0.489%。
3. **不能说明什么**：P6 受试者公平性未通过（`p6_final_acceptance=false`，最差受试者 coverage 87.06%、accepted accuracy 91.89%、WAR 7.06%）；P7 在 10% 高斯噪声与 4×4 降密度下 macro-F1 降至 0.668 / 0.682，模型明显失效且 `UNKNOWN/REJECT` 无法挽救（65.6% 错误为高置信错判）。
4. **下一步是什么**：下一阶段是 SLP Adapter（[S0/S0.2 阶段报告](S0_SLP_FULL_INVENTORY_AND_ANNOTATION_BOUNDARY_v0.1.md)），而非继续在 PoPu 上调参。PoPu 仅提供工程方法、协议与代码资产，不提供可直接迁移的性能承诺或拒识阈值。
5. **PoPu 阶段边界**：本报告确认"软件算法 + 软件扰动"验证闭环完成；硬件鲁棒性、舒适性、整夜睡眠、闭环控制等产品/安全结论均**不在 PoPu 证据范围内**。

---

## 2. 研究问题与总体思路

### 2.1 任务定义

PoPu Tactilus 公开数据集由 60 名受试者、5,160 条 JSON 记录组成，每条记录统一为 `64×27` 压力矩阵并附 10 个 snapshot。任务是在每条记录 10 个 snapshot 聚合后的 record 粒度上做五分类：

- `empty`（空床）作为独立类别，不视作"姿态 0"；
- `supine / prone / left / right` 四种人体卧姿。

### 2.2 为什么采用 P1–P7 主阶段及其子阶段路径

| 阶段 | 解决问题 | 不可省略的原因 |
|---|---|---|
| P1 数据盘点 | 是否所有 JSON 可解析、结构一致、能否区分有标签固定姿态与未标注序列 | 后续所有分析的前提：必须先固定"输入是什么" |
| P2 质量门 | 是否存在结构性坏数据、统计异常样本是否应删除 | 删除前必须先量化、被审阅的样本不能静默丢 |
| P3 几何/Mask | 接触区域与几何量的工程计算是否可重复 | 后续特征工程的输入；策略选择必须可回退 |
| P3.1 Mask 冻结 | 三种 Mask 策略如何选 | 单策略的 bbox 容易被弱信号拉大 |
| P3.2 区域审计 | COCO 标注能否一对一监督身体区域 | 不可伪造配对 |
| P4a 无标签特征 | 71 维特征列是否可复现生成、是否泄漏标签 | 后续模型的输入契约 |
| P5/P5.1 传统模型 | 71 维特征下谁是受试者隔离最强基线 | 任何神经网络都必须先在公平 CV 下赢过传统基线 |
| P5.2-A/B/C 神经网络 | 是否存在显著更强的候选模型 | 必须有 CPU+CUDA 通路、最小可行性、Full 公平比较三道关 |
| P6 拒识 | 在置信度阈值下，coverage/accepted_accuracy/WAR/公平性如何权衡 | 不能只看总体，必须看最差受试者 |
| P6.1 校准+一致性 | 温度 T 与三模型一致是否改善公平性 | 单点 P6 不能闭环，必须给出有界改善 |
| P7 鲁棒性 | 14 类软件扰动下模型是否退化、规则是否还能兜底 | 公平性 OK 不代表产品闭环，必须有扰动证据 |

### 2.3 明确边界

> **本报告和 PoPu 全部阶段**仅验证五分类睡姿**软件**研究闭环，**不是**：
> - 产品安全、临床安全、医疗器械合规；
> - 舒适度评估；
> - 硬件故障（如连接器接触不良、ESD、单元老化、零点漂移）验证；
> - 整夜睡眠连续过程评估；
> - 闭环控制（自动调节床体、警报、临床干预）验证。

---

## 3. 数据、标签与质量治理

### 3.1 数据规模与结构（P1）

来源：[P1_POPU_TACTILUS_INVENTORY_v0.1.md](P1_POPU_TACTILUS_INVENTORY_v0.1.md)、[popu_tactilus_inventory_summary_v0.1.json](../../outputs/reports/popu_tactilus_inventory_summary_v0.1.json)。

| 项 | 实际值 |
|---|---:|
| JSON 记录数 | 5,160 |
| 受试者数 | 60 |
| 统一传感矩阵形状 | `64×27` |
| 结构错误 | 0 |
| 重复 `sample_id` | 0 |
| `OK` 记录 | 5,100 |
| `WARN` 记录 | 60（皆为 `others.json`） |
| 固定姿态有标签 snapshot | 51,000 |
| `others.json` 未标注 snapshot | 35,247 |

有标签记录分布：`supine=1,260 / prone=1,260 / left=1,260 / right=1,260 / empty=60` 条 JSON 记录，对应每姿态 `12,600` snapshot（empty 仅 600）。`others.json` 共 60 条，每条含 341–914 snapshot，源 JSON 缺 `position`/`variation` 字段，**不混入固定姿态监督训练池**。

### 3.2 记录级质量门（P2）

来源：[P2_POPU_TACTILUS_QUALITY_GATE_v0.1.md](P2_POPU_TACTILUS_QUALITY_GATE_v0.1.md)、[popu_tactilus_quality_summary_v0.1.json](../../outputs/reports/popu_tactilus_quality_summary_v0.1.json)。

| 状态 | 记录数 | 含义 |
|---|---:|---|
| `ACCEPT` | 5,006 | 同姿态分布内无统计异常 |
| `WARN` | 94 | 同姿态内 robust z > 4.5 的统计异常候选（不是确认的坏样本） |
| `REJECT` | 0 | 无读回失败或结构性无效 |
| `EXCLUDED` | 60 | `others.json`（无固定姿态标签） |

94 条 WARN 的触发原因：总信号 54、帧间变异 40、活跃单元数 1。**未在 P2 阶段删除**，只进入"全量"与"仅 ACCEPT"双口径敏感性分析（后续 P5.2 实际为 ACCEPT-only，见 §4）。

### 3.3 标签体系与受试者隔离

- 标签类别固定为 `empty / supine / prone / left / right` 五类，顺序在 P5.2-C 协议中显式冻结为 `["empty", "supine", "prone", "left", "right"]`。
- 受试者编号从 1 到 60，**每一受试者的所有记录只进入同一数据划分**（P2 决策）。
- `others.json` 保留 Inventory 但**不参与**任何监督训练。

### 3.4 P3.2 区域标注审计

来源：[P3_2_POPU_SEGMENTATION_ALIGNMENT_AUDIT_v0.1.md](P3_2_POPU_SEGMENTATION_ALIGNMENT_AUDIT_v0.1.md)。

| 项目 | 结果 |
|---|---:|
| 发现 COCO 文件 | 1,730 |
| 结构错误 | 0 |
| `AMBIGUOUS_TACTILUS_CANDIDATES`（人体系 1:3） | 1,670 |
| `ONE_TO_ONE_CANDIDATE`（仅空床） | 60 |
| 可用于逐记录人体区域监督的候选 | **0** |

**结论**：在当前公开文件结构下，1,670 份人体 COCO 标注对应同一受试者/姿态/variation 下 3 条 Tactilus 压力记录；没有任何独立元数据证明应选 `_0/_1/_2`。**P4b 区域监督继续 HOLD**，本阶段仅冻结"全局粗几何"的 `largest_component` Mask。

> **不确定项**：COCO 图与压力记录在采集时间上的同步未由现有证据验证；标"未在现有证据中验证"。

---

## 4. 方法演进与关键决策

### 4.1 P1–P7 阶段总表

| 阶段 | 要解决的问题 | 方法 / 协议 | 关键产出 | 结论 / 决策 |
|---|---|---|---|---|
| **P1** 数据盘点 | JSON/矩阵是否一致 | `inventory_popu.py` 全量遍历 | 5,160 条记录登记 | COMPLETE；带用途分流（others.json 隔离） |
| **P2** 质量门 | 是否删除统计异常 | 每姿态 MAD robust z > 4.5 | ACCEPT 5,006 / WARN 94 / REJECT 0 | PARTIAL；WARN 暂保留，进入双口径分析 |
| **P3** 几何/Mask | Mask/几何能否稳定 | P2 代表帧 + 50 分位阈值 + 连通域过滤 | v0.1 Mask + 几何 CSV/JSON | PARTIAL；Mask 候选未冻结 |
| **P3.1** Mask 冻结 | 选哪种 Mask | 三策略全量比较：`relative_filtered` / `largest_component` / `relative_closed` | `largest_component` v0.2 冻结规则 | COMPLETE；冻结"主接触区/粗几何"输入 |
| **P3.2** 区域审计 | COCO 能否监督身体区域 | 1,730 份 COCO 配对基数审计 | `AMBIGUOUS=1,670` / `ONE_TO_ONE=60`（皆 empty） | COMPLETE；区域监督 HOLD |
| **P4a** 无标签特征 | 71 维特征是否可复现 | 71 特征列 + 受试者隔离 | 51,000 行逐 snapshot 特征表 | COMPLETE；primary 50,060 / warn 940 |
| **P5** 传统首轮 | 71 维 + 受试者隔离下基线 | held-out 12 受试者 + GroupKFold | `logreg` macro-F1 test 0.9466 | FIRST_ROUND_BASELINE；未冻结 |
| **P5.1** 传统比较 | 7 候选 + 重复 CV + 消融合 | repeated subject-grouped CV（5×3 seeds [11,22,33]）+ 特征消融 | `calibrated_linear_svm` record macro-F1 0.9452 | CANDIDATE_FROZEN（仅传统侧） |
| **P5.2-A** Smoke | CPU+CUDA 通路 | RTX 4090 CUDA R02 | smoke pass；SUCCEEDED | CPU_CUDA_SMOKE_PASS |
| **P5.2-B** Mini | 三 NN 可行性 | 6 受试者 / seed=42 / `["1".."4"]` 训练、`["5","6"]` 验证 | 三候选 `verdict=proceed` | MINI_ACCEPTED；不形成排名 |
| **P5.2-C** Full | NN vs SVM 终极比较 | 3×5=15 fold，stage A 选 epoch + stage B refit | `small_resnet` record macro-F1 0.9866 | SMALL_RESNET_ACCEPTED（PoPu 研究候选族） |
| **P6** 拒识 | `UNKNOWN/REJECT` 与高置信错判 | P5.2-C Full OOF record 概率 + `max_probability ≥ 0.94` | repeat 2 coverage 96.90%、acc 99.61%、WAR 0.380% | 总体 operating point 找到，但 **公平性未通过** |
| **P6.1** 校准+一致性 | T 与三模型一致能否改善 | 40 开发受试者选 T，20 评估；T=0.75, threshold=0.5, require_unanimous=true | 评估集 coverage 96.76%、acc 99.49%、WAR 0.489% | 改善但最差受试者 coverage 仍 < 80%；研究候选保留 |
| **P7** 软件鲁棒性 | 14 类扰动下模型是否退化 | 15 fold × 14 condition × 5 seed = 1,050 condition-seed + 15 clean | clean macro-F1 0.9866；10% 噪声 0.668 / 4×4 stride 0.682 | mild OK、severe FAIL；UNKNOWN/REJECT 不能挽回高置信错判 |

### 4.2 关键决策的逻辑链

- **传统基线为何不能直接冻结**：P5 仅是单次 held-out，P5.1 用 repeated subject-grouped CV 才让候选稳定到 0.005 margin；SVM 与 logreg 在 record macro-F1 上差 0.0028，按固定 tie-break（balanced accuracy）才得出 `calibrated_linear_svm`。
- **Small ResNet 何时成为候选**：仅在 P5.2-C Full（45 训练单元、SHA-verified 证据包、Reviewer 独立重算）后才被接受。它**不是"最终部署模型"**——决定仅冻结"候选架构族与研究路线"。
- **`largest_component` Mask 的边界**：可能丢弃真实但与主躯干分离的手臂或腿部接触；P3.1 报告明确写到"不得解释为身体部位分割"。
- **UNKNOWN/REJECT 阈值的来源**：`0.94` 来自 P6 验证（开发 repeat 0/1）；`0.5 + require_unanimous=true` 来自 P6.1 校准后规则的 `rules[1]` 分支（**不是** `rules[0]` 的 0.75）。两者**均从证据包 pinned rule block 加载**，CLI 不暴露覆写 flag（P7 Round 2 强化）。

---

## 5. 核心实验结果

### 5.1 P5.1 传统模型横向比较（record-level，primary cohort，3 repeats mean ± std）

来源：[P5_1_POPU_GROUPED_MODEL_COMPARISON_v0.1.md](P5_1_POPU_GROUPED_MODEL_COMPARISON_v0.1.md)、[popu_model_comparison_p5_1_summary_v0.1.json](../../outputs/reports/popu_model_comparison_p5_1_summary_v0.1.json)。

| 模型 | record macro-F1 | record bal-acc | record acc | 最差受试者 acc |
|---|---:|---:|---:|---:|
| dummy（基线下限） | 0.2058 ± 0.0005 | 0.2084 | 0.2578 | 0.2000（subj 60） |
| centroid（模板） | 0.7851 ± 0.0004 | 0.7856 | 0.7349 | 0.4086（subj 46） |
| knn | 0.8855 ± 0.0014 | 0.8858 | 0.8588 | 0.6549（subj 60） |
| random_forest | 0.9318 ± 0.0010 | 0.9321 | 0.9161 | 0.5882（subj 38） |
| extra_trees | 0.9369 ± 0.0003 | 0.9372 | 0.9224 | 0.6549（subj 38） |
| logistic_regression | 0.9424 ± 0.0010 | 0.9418 | 0.9296 | 0.6706（subj 17） |
| **calibrated_linear_svm** | **0.9452 ± 0.0022** | **0.9429** | **0.9353** | **0.6784（subj 17）** |

选择规则：margin 0.005 内视为并列 → SVM 与 logreg 在 0.0028 差距内并列 → tie-break 1（balanced accuracy）→ SVM 胜出。特征消融（top-2 × 5 组）：

| 特征组 | n | logreg | svm |
|---|---:|---:|---:|
| intensity_only | 14 | 0.5282 | 0.5201 |
| mask_geometry_only | 21 | 0.8926 | 0.8887 |
| grid_zones_only | 36 | 0.8233 | 0.8253 |
| intensity_geometry | 35 | 0.9039 | 0.9020 |
| all | 71 | 0.9424 | 0.9452 |

**结论**：几何/形状承载主要判别力（单用即 ~0.89），强度与网格分区必须叠加到 71 列才能达 0.945。

### 5.2 P5.2-C 神经网络 Full 公平比较（record-level，primary cohort，3 repeats）

来源：[P5_2_C_POPU_NEURAL_FULL_RESULTS_v0.1.md](P5_2_C_POPU_NEURAL_FULL_RESULTS_v0.1.md)、[popu_neural_full_p5_2_c_reviewer_acceptance_v0.1.json](../../outputs/reports/popu_neural_full_p5_2_c_reviewer_acceptance_v0.1.json)。

| 模型 | macro-F1 mean | macro-F1 std | balanced acc | 最差受试者 macro-F1 |
|---|---:|---:|---:|---:|
| calibrated_linear_svm | 0.945168 | 0.002169 | 0.942864 | 0.740556 |
| matrix_mlp | 0.974183 | 0.001077 | 0.974814 | 0.755706 |
| tiny_cnn | 0.978906 | 0.000351 | 0.978885 | 0.820635 |
| **small_resnet** | **0.986649** | **0.002832** | **0.986636** | **0.882483** |

small_resnet 相对 tiny_cnn record macro-F1 提升 0.007743（高于 margin 0.005），相对 SVM 提升 0.041481。Reviewer 在 SHA-verified 证据包上独立重算结果一致，45/45 训练单元完整，checkpoint reload 一致性、OOF 覆盖率（450,540 snapshot + 45,054 record）全部满足。

**含义**：Small ResNet 是 PoPu 研究的"架构族候选"，但**不能直接迁移到 SLP、PressurePose、自采硬件或产品**。

### 5.3 P6 `UNKNOWN/REJECT`（Small ResNet Full OOF，15,018 record）

来源：[P6_POPU_REJECT_RESULTS_v0.1.md](P6_POPU_REJECT_RESULTS_v0.1.md)、[outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/summary.json](../../outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/summary.json)。

冻结规则：`max_probability ≥ 0.94` 接受，否则 `UNKNOWN/REJECT`；阈值由开发（repeat 0/1）选择，repeat 2 独立评估。

| 数据 | threshold | coverage | accepted_accuracy | WAR |
|---|---:|---:|---:|---:|
| 开发（repeat 0/1） | 0.94 | 0.9420 | 0.9953 | 0.0044 |
| 独立评估（repeat 2） | 0.94 | 0.9690 | 0.9961 | 0.0038 |
| 全部 3 repeats | 0.94 | 0.9510 | 0.9956 | 0.0042 |

**公平性门槛**（机器可读）：最差受试者 coverage ≥ 80%、accepted accuracy ≥ 95%、WAR ≤ 5%、类间 coverage 差 ≤ 10pp。repeat 2 评估**未通过**：

| 项 | 实际 | 门槛 | 通过 |
|---|---:|---:|:---:|
| 最差受试者 coverage | 0.8471（≥ 0.8） | ≥ 0.80 | ✓ |
| **最差受试者 accepted accuracy** | **0.9189**（subj 15） | ≥ 0.95 | ✗ |
| **最差受试者 WAR** | **0.0706**（subj 15） | ≤ 0.05 | ✗ |
| 类间 coverage 差 | 0.0462 | ≤ 0.10 | ✓ |

机器判定 `p6_final_acceptance = false`（来源：[outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/summary.json](../../outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/summary.json)）。

主要高置信错误（全部 repeats，`max_probability ≥ 0.90`）：`left→prone` 29、`prone→supine` 12、`right→prone` 9、`left→supine` 7；全 248 个错误中 83 个为高置信错判。

### 5.4 P6.1 温度校准 + 三模型一致性（Small ResNet Full OOF，独立 20 受试者）

来源：[P6_1_POPU_CALIBRATION_ENSEMBLE_RESULTS_v0.1.md](P6_1_POPU_CALIBRATION_ENSEMBLE_RESULTS_v0.1.md)、[outputs/analysis/EXP-P6.1-POPU-CALIBRATION-20260820-R01/summary.json](../../outputs/analysis/EXP-P6.1-POPU-CALIBRATION-20260820-R01/summary.json)。

- 温度：开发受试者 NLL 最优 `T=0.75`（整体锐化概率）。
- 跨 repeat 错误复现：24 条记录在 3 个 repeat 中持续错判（如受试者 31 `left→prone` 6 条）。
- 独立 20 受试者结果：

| 规则 | coverage | accepted_acc | WAR | 最差覆盖率 | 最差 acc | 最高 WAR |
|---|---:|---:|---:|---:|---:|---:|
| 校准后三模型概率平均 | 0.9756 | 0.9937 | 0.0061 | 0.7742 | 0.9459 | 0.0471 |
| **校准平均 + 三模型一致** | **0.9676** | **0.9949** | **0.0049** | **0.7419** | **0.9634** | **0.0353** |

`require_unanimous=true` 把 borderline 预测拒掉，最低受试者 WAR 降到 3.53%，但**最差受试者 coverage 74.19%**仍低于 80% 门槛，机器判定仍 `false`。受试者 15 改善：coverage 83.53%、accepted accuracy 97.18%、WAR 2.35%。

### 5.5 P7 Full 软件鲁棒性（15 fold × 14 condition × 5 seed）

来源：[P7_POPU_SOFTWARE_ROBUSTNESS_FULL_RESULTS_v0.1.md](P7_POPU_SOFTWARE_ROBUSTNESS_FULL_RESULTS_v0.1.md)、[outputs/analysis/EXP-P7-FULL-ANALYSIS-20260821-R01/condition_metrics.csv](../../outputs/analysis/EXP-P7-FULL-ANALYSIS-20260821-R01/condition_metrics.csv)。

P6/P6.1 规则来自 Full 证据包 pinned rule block（SHA 验证，CLI 不暴露覆写 flag）：
- P6 single：`threshold=0.94`（source SHA `af9ec5d7...`）
- P6.1 ensemble：`T=0.75`, `threshold=0.5`, `require_unanimous=true`（source SHA `d8b191ba...`，从 `rules[1]` 加载）

Clean 基线（15-fold OOF stitched，n=15,018）：

| 指标 | 值 |
|---|---:|
| Accuracy | 0.9835 |
| Balanced accuracy | 0.9866 |
| Macro-F1（5 类） | **0.9866** |
| P6 coverage | 0.9510 |
| P6 accepted accuracy | 0.9956 |
| P6 WAR | 0.0044 |
| P6.1 coverage | 0.9728 |
| P6.1 accepted accuracy | 0.9959 |
| P6.1 WAR | 0.0041 |

14-condition 表（5-seed mean；Δ vs clean macro-F1）：

| Condition | macro-F1 | Δ macro-F1 | balanced acc | P6 cov | P6 acc | P6 WAR | P6.1 cov | P6.1 WAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean（参考） | 0.986644 | 0.000000 | 0.986636 | 0.9510 | 0.9956 | 0.0044 | 0.9728 | 0.0041 |
| density_stride_2_2 | 0.961567 | -0.025077 | 0.960488 | 0.8894 | 0.9846 | 0.0137 | 0.9115 | 0.0112 |
| **density_stride_4_4** | **0.682021** | **-0.304623** | 0.711571 | 0.6789 | 0.7513 | 0.1689 | 0.6338 | 0.1518 |
| noise_p95_0.01 | 0.986479 | -0.000165 | 0.986460 | 0.9512 | 0.9951 | 0.0046 | 0.9724 | 0.0041 |
| **noise_p95_0.05** | **0.938365** | **-0.048279** | 0.944590 | 0.8601 | 0.9676 | 0.0279 | 0.8832 | 0.0223 |
| **noise_p95_0.10** | **0.668383** | **-0.318261** | 0.680153 | 0.6900 | 0.6608 | 0.2340 | 0.6443 | 0.2018 |
| bad_cell_0.01 | 0.985889 | -0.000755 | 0.985895 | 0.9483 | 0.9952 | 0.0046 | 0.9710 | 0.0047 |
| bad_cell_0.05 | 0.979669 | -0.006975 | 0.979712 | 0.9269 | 0.9932 | 0.0063 | 0.9573 | 0.0072 |
| bad_cell_0.10 | 0.953275 | -0.033369 | 0.953180 | 0.8533 | 0.9809 | 0.0162 | 0.8913 | 0.0105 |
| bad_rows_1 | 0.984866 | -0.001778 | 0.984845 | 0.9471 | 0.9943 | 0.0054 | 0.9694 | 0.0050 |
| bad_rows_2 | 0.983804 | -0.002840 | 0.983774 | 0.9420 | 0.9942 | 0.0054 | 0.9661 | 0.0052 |
| bad_rows_4 | 0.980032 | -0.006612 | 0.980017 | 0.9302 | 0.9930 | 0.0065 | 0.9581 | 0.0064 |
| bad_columns_1 | 0.982826 | -0.003818 | 0.982818 | 0.9409 | 0.9940 | 0.0056 | 0.9648 | 0.0056 |
| bad_columns_2 | 0.969736 | -0.016908 | 0.969744 | 0.9078 | 0.9881 | 0.0104 | 0.9318 | 0.0090 |
| bad_columns_4 | 0.955444 | -0.031200 | 0.955298 | 0.8741 | 0.9827 | 0.0149 | 0.8990 | 0.0120 |

按扰动强度分层（[P7 Full §11](../../docs/stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_FULL_RESULTS_v0.1.md)）：

| 强度档 | 条件 | 平均 Δ macro-F1 | 平均 P6 WAR | 判定 |
|---|---|---:|---:|---|
| 轻度（Δ F1 < 0.05） | noise_0.01、bad_cell_0.01/0.05、bad_rows_1/2/4、bad_columns_1/2/4、density_stride_2_2 | -0.005 至 -0.025 | < 0.02 | 可接受；P6 coverage ≥ 88%、accepted acc ≥ 0.98 |
| 中度（0.05 ≤ Δ F1 < 0.20） | bad_cell_0.10、bad_columns_4、noise_0.05 | -0.03 至 -0.05 | 0.01 – 0.03 | P6 仍安全，P6.1 coverage 略低 |
| **严重（Δ F1 ≥ 0.20）** | **noise_0.10、density_stride_4_4** | **-0.30 至 -0.32** | **0.17 – 0.23** | **模型明显失效；P6/P6.1 都不能挽回** |

10% 噪声下逐类（seed 701，10,058 records）：`empty F1=1.0 / supine F1=0.568 / prone F1=0.854 / left F1=0.377 / right F1=0.541`。主导错误方向（全部 seeds，raw-error 29,672 行）：

| y_true | y_pred | 计数 |
|---|---|---:|
| left | supine | 13,603 |
| right | supine | 10,970 |
| prone | supine | 3,788 |

**关键负面结论**：在 10% 噪声下 65.6% raw-error 是高置信（≥0.90）错误；`UNKNOWN/REJECT` 是"低置信度或不一致"规则，**对"高置信但错误"的分布漂移无能为力**。

---

## 6. 错误、拒识与鲁棒性分析

### 6.1 主要混淆与高置信错误（clean / mild 域）

P5.1 SVM 15,018 records 混淆（pooled over repeats）：

| y_true | y_pred | 计数 |
|---|---|---:|
| left | right | 163 |
| right | left | 131 |
| prone | supine | 108 |
| prone | left | 128 |
| left | prone | 79 |
| supine | prone | 62 |
| supine | left | 79 |

**形态对称性是物理可分性问题**（left↔right、supine↔prone），不是工程缺陷。P5.2-C 仍存在 248 个 clean 错误，其中 83 个为 `max_probability ≥ 0.90`。

### 6.2 公平性未闭环

- P6 机器判定：`p6_final_acceptance = false`。
- repeat 2 受试者 15：coverage 87.06%、accepted accuracy 91.89%、WAR 7.06%。
- P6.1 改善但仍未通过最差受试者 coverage 门槛（74.19% < 80%）。
- 跨 repeat 持续错判：24 条 record 在 3 repeats 全错。

### 6.3 严重扰动下模型失效

- 10% 高斯噪声（p95 比例）：macro-F1 0.6683、P6 WAR 23.4%、P6.1 WAR 20.2%。
- 4×4 降密度重建：macro-F1 0.6820、P6 WAR 16.9%、P6.1 WAR 15.2%。
- 主导错误：`left/right → supine`（模型退化为"全猜 supine"）。
- 65.6% 的 raw-error 为高置信（≥0.90）。

### 6.4 raw error / wrong action / WAR 的严格区别

- **Raw error row**（`error_cases.csv`）：所有 `y_true ≠ y_pred` 的样本，与规则无关。
- **Wrong action**（`summary.json` → `wrong_action_n`）：模型错且**规则未拒掉**的样本；按规则被拒绝的错误样本**不计入** wrong action。
- **WAR** = `wrong_action_n / total_n`，是 wrong action 的**率**，**不等于** raw-error 率。

混淆示例（10% 噪声，全部 seeds）：

| 量 | 值 |
|---|---:|
| Raw error rows | 29,672 |
| High-confidence raw error rows | 19,473（65.6%） |
| P6 wrong-action rate（5-seed mean） | 0.2340 |
| P6.1 wrong-action rate（5-seed mean） | 0.2018 |

### 6.5 受试者差异

P7 clean 基线 4-criterion 最差受试者：

| 准则 | 受试者 | n | WAR | accuracy | coverage | accepted_accuracy |
|---|---|---:|---:|---:|---:|---:|
| by_wrong_action_rate | 46 | 93 | 0.0753 | 0.8925 | 0.8710 | 0.9136 |
| by_coverage | 58 | 246 | 0.0000 | 0.9472 | 0.8089 | 1.0000 |
| by_accepted_accuracy | 46 | 93 | 0.0753 | 0.8925 | 0.8710 | 0.9136 |
| by_raw_accuracy | 31 | 255 | 0.0588 | 0.8588 | 0.8157 | 0.9279 |

10% 噪声下 subject 60 在 5/5 seeds 都是 by_WAR 最差（WAR 0.60）。**注意**：研究证据报告个体差异**不等于**鼓励设计按受试者定制的阈值（见 §7 边界）。

---

## 7. 最终结论与边界

### 7.1 PoPu 已完成的闭环（软件验证）

PoPu 第一阶段已建立如下软件验证闭环：
- 数据治理（盘点 + 质量门 + 受试者隔离）；
- 几何/Mask 冻结（`largest_component` v0.2）；
- 无标签特征工程（71 列 v0.1）；
- 传统基线（`calibrated_linear_svm` 冻结为研究候选）；
- 神经网络公平比较（Small ResNet 作为研究候选族）；
- `UNKNOWN/REJECT` 总体操作点（`0.94`）找到但**公平性未通过**；
- 温度校准 + 三模型一致性改善部分指标但**仍未闭环**；
- 14 类软件扰动下的鲁棒性刻画（含 clean/严重 4 档分层）。

### 7.2 已支持 / 不支持的声明

| 声明 | PoPu 证据支持？ | 依据 |
|---|:---:|---|
| 在公开 PoPu 受试者隔离 5 分类上，Small ResNet 优于 `calibrated_linear_svm`、`matrix_mlp`、`tiny_cnn` | ✓ | P5.2-C Full 公平比较 |
| PoPu 上总体 record macro-F1 达 ~0.987 | ✓ | P5.2-C、P7 Full clean |
| `max_probability ≥ 0.94` 在总体上有 ~95% coverage / ~99.6% accepted accuracy / ~0.42% WAR | ✓ | P6 / P7 Full clean |
| 校准 + 三模型一致进一步降低 WAR | ✓ | P6.1 评估集 |
| **PoPu 受试者公平性已通过产品门槛** | **✗** | `p6_final_acceptance=false`；受试者 15 WAR 7.06%、acc 91.89% |
| **软件扰动下模型鲁棒到硬件级** | **✗** | 10% 噪声 / 4×4 降密度下 macro-F1 < 0.7，P6/P6.1 WAR > 15% |
| **可作为 SLP、PressurePose 或自采硬件的部署模型** | **✗** | PoPu 是公开 Tactilus 64×27 数据集；SLP 是 RGB/IR 深度模态 |
| **可作为产品安全/舒适性/整夜睡眠/闭环控制证据** | **✗** | 全部任务边界外 |
| **高置信但错误的预测已被拒识规则捕获** | **✗** | 65.6% raw-error 是高置信；P6/P6.1 不能挽回 |

### 7.3 阈值与规则的可迁移性边界

- `0.94`、`T=0.75`、`threshold=0.5`、`require_unanimous=true` **仅是 PoPu 研究操作点**。
- 它们从 P5.2-C Full 证据包 pinned rule block 加载（SHA 验证）；CLI 不暴露覆写 flag（P7 Round 2 fail-closed）。
- **不能直接迁移**到 SLP Adapter、PressurePose、自采硬件或产品部署阈值——每个新数据集需独立验证不确定性与拒识行为。

### 7.4 "PoPu 阶段完成"的措辞边界

- **可以说**：PoPu 软件算法与软件扰动验证闭环完成；研究候选族、研究协议、测试体系、证据归档方法就绪。
- **不可说**：PoPu 阶段完成 = 产品完成 / 部署通过 / 硬件验证通过 / 安全通过 / 临床证据 / 整夜睡眠证据。

---

## 8. 产出与可复用资产

> **本地工作台产物声明**：`outputs/` 与 `data/processed/` 中**大部分路径被 `.gitignore` 排除**，属于**本地工作台产物，未随代码仓库上传**。**不要上传原始数据、压缩包、OOF 全量文件或大模型文件**到代码仓库。下列链接仅在本地工作树中可访问。

| 类别 | 可复用资产 | 路径 | 用途 | 复用限制 |
|---|---|---|---|---|
| 数据盘点/质量 | Inventory CSV + Summary JSON | [popu_tactilus_inventory_v0.1.csv](../../data/processed/popu/popu_tactilus_inventory_v0.1.csv), [popu_tactilus_inventory_summary_v0.1.json](../../outputs/reports/popu_tactilus_inventory_summary_v0.1.json) | 受试者与记录登记、姿态分布、others.json 隔离 | 不能当作"已清洗"训练数据；P2 仍可能有 WARN |
| 质量门 | 逐记录质量结果 + Summary | [popu_tactilus_quality_results_v0.1.csv](../../outputs/metrics/popu_tactilus_quality_results_v0.1.csv), [popu_tactilus_quality_summary_v0.1.json](../../outputs/reports/popu_tactilus_quality_summary_v0.1.json) | 后续阶段输入；cohort 过滤（primary = ACCEPT） | SHA-256 `9d3398a5…3e954c`；下游必须校验 |
| 几何/Mask | 冻结规则 + v0.2 输出 | [popu_geometry_frozen_v0.2.json](../../configs/experiments/popu_geometry_frozen_v0.2.json)（如存）, [popu_geometry_results_v0.2.csv](../../outputs/metrics/popu_geometry_results_v0.2.csv), [popu_geometry_summary_v0.2.json](../../outputs/reports/popu_geometry_summary_v0.2.json) | 几何特征输入；P3.1 选定的 `largest_component` 策略 | 不能解释为身体部位分割 |
| 区域审计 | 配对基数审计 | [popu_segmentation_alignment_audit_v0.1.csv](../../outputs/metrics/popu_segmentation_alignment_audit_v0.1.csv), [popu_segmentation_alignment_summary_v0.1.json](../../outputs/reports/popu_segmentation_alignment_summary_v0.1.json) | 区域监督 HOLD 的证据 | 不允许用于逐记录人体区域监督 |
| 特征 | 51,000 行逐 snapshot 特征表 | [popu_features_p4a_v0.1.csv](../../data/processed/popu/popu_features_p4a_v0.1.csv)（71 特征列），[popu_features_p4a_summary_v0.1.json](../../outputs/reports/popu_features_p4a_summary_v0.1.json) | 模型输入；primary=50,060 / warn=940 / excluded=60 | 22 个空床 NaN 主轴需显式处理 |
| 模型代码 | 数据/模型/训练/评估 | `src/topper_perception/neural/data.py` `models.py` `training.py` `dataset.py` `checkpoint.py` | 受试者隔离训练与评估 | 不直接迁移到 SLP；需重新写数据接口 |
| 模型协议 | 冻结配置 + 协议文档 | [popu_neural_full_v0.1.json](../../configs/experiments/popu_neural_full_v0.1.json), [P5_2_C_POPU_NEURAL_FULL_PROTOCOL_v0.1.md](P5_2_C_POPU_NEURAL_FULL_PROTOCOL_v0.1.md) | 受试者隔离 OOF + 两阶段 epoch 选择 | 不包含硬件/产品条件 |
| 传统候选 | joblib + 元数据 | `outputs/models/popu_research_candidate_p5_1_v0.1.joblib`（16,097 B） + `popu_research_candidate_p5_1_v0.1.metadata.json` | 71 特征基线；保留为对照 | 不能重训/覆盖/重冻结 |
| 神经网络候选 | P5.2-C Full 证据包 | `EXP-P5.2-C-FULL-COMPARISON-20260820-R01-FINAL.tar.gz`（SHA-256 `131e6fd6…b5b`，约 198 MB） | 45 训练单元 + checkpoints + OOF | 受治理外部证据包，不入 Git |
| 拒识分析 | Summary + 错误案例 + 高置信错误 | [EXP-P6-POPU-REJECT-20260820-R01](../../outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/) | P6 阈值与公平性证据 | 阈值 `0.94` 是 PoPu 操作点 |
| 校准与一致性 | Summary + 温度网格 + 评估 | [EXP-P6.1-POPU-CALIBRATION-20260820-R01](../../outputs/analysis/EXP-P6.1-POPU-CALIBRATION-20260820-R01/) | P6.1 改善但不闭环 | rule 从 `rules[1]` 加载（threshold=0.5, require_unanimous=true） |
| 鲁棒性 | Full 分析产物 + condition metrics | [EXP-P7-FULL-ANALYSIS-20260821-R01](../../outputs/analysis/EXP-P7-FULL-ANALYSIS-20260821-R01/), 原始 `EXP-P7-FULL-20260820-R02.tar.gz`（SHA-256 `cbaffa74878b149e546a42826ae373442c62683af890362684f80963e7fddda1`，672,702,773 字节） | 14 condition × 5 seed × 15 fold | 仅软件扰动，非硬件失效 |
| 鲁棒性配置 | Frozen P7 config | [popu_p7_robustness_v0.1.json](../../configs/analysis/popu_p7_robustness_v0.1.json) | 14 条件与 5 seeds 冻结 | 不暴露覆写 P6/P6.1 阈值 flag |
| 测试 | pytest 全量 | `tests/test_neural_*.py` 等 | 80%+ 覆盖 | mock 数据，不读完整 PoPu |
| 阶段报告 | P1–P7 markdown | `docs/stage_reports/P*.md` | 完整证据链 | 仅作历史证据，不可改写 |
| 受治理外部数据 | PoPu Tactilus 原始数据 | 本地只读 PoPu 数据副本，原始数据不入仓（P1） | 仅读；不复制入仓 | 隐私/受试者边界；本报告不暴露 |

---

## 9. 从 PoPu 到 SLP 的交接建议

下一阶段**不是**继续在 PoPu 上调参，而是 SLP Adapter（参考 [S0_SLP_FULL_INVENTORY_AND_ANNOTATION_BOUNDARY_v0.1.md](S0_SLP_FULL_INVENTORY_AND_ANNOTATION_BOUNDARY_v0.1.md) 与 [S0_2_SLP_TWO_PHASE_ROUTE_AND_AGENT_HANDOFF_v0.1.md](S0_2_SLP_TWO_PHASE_ROUTE_AND_AGENT_HANDOFF_v0.1.md)）。PoPu 能且只能提供以下三类资产：

1. **工程方法**：数据读取、Mask/几何、特征工程、训练循环、checkpoint/resume、JSON 严格序列化（`allow_nan=False`）、SHA 锁定证据包等模式。
2. **协议**：受试者隔离分组 CV、pool-first-then-metric、5-seed mean/std/worst、最差受试者多准则分析、pinned rule block 加载。
3. **代码资产**：模块化的 model/training/checkpoint/evaluation/runner，可在新数据集上重新挂接数据加载与 label schema。

PoPu **不能提供**：
- 直接可迁移的性能承诺；
- 直接可迁移的拒识阈值（`0.94`、`0.5`）；
- 硬件失效、舒适性、整夜、闭环结论。

SLP Adapter 阶段建议顺序：
1. 数据/标签治理（帧索引、重复帧检查）；
2. 跨模态同步与标定审计；
3. 各单模态基线；
4. 受控融合（冻结编码器 → 全量微调）；
5. 受试者隔离评估（沿用 PoPu 的 repeated subject-grouped CV 协议）。

---

## 10. 可复现性与证据索引

> **本地工作台产物声明**：`outputs/` 与 `data/processed/` 中**大部分路径被 `.gitignore` 排除**，属于**本地工作台产物，未随代码仓库上传**。**不要上传原始数据、压缩包、OOF 全量文件或大模型文件**到代码仓库。下列链接与归档哈希仅在本地工作树中可访问。

### 10.1 核心 EXP-ID 与协议

| EXP-ID | 阶段 | 关键产出 | 证据位置 |
|---|---|---|---|
| EXP-P5.2-C-FULL-COMPARISON-20260820-R01 | P5.2-C | 45 单元 Full 比较证据包 | SHA `131e6fd6…b5b`，本地 `EXP-P5.2-C-FULL-COMPARISON-20260820-R01-FINAL.tar.gz`（[P5_2_C_POPU_NEURAL_FULL_RESULTS_v0.1.md](P5_2_C_POPU_NEURAL_FULL_RESULTS_v0.1.md)） |
| EXP-P6-POPU-REJECT-20260820-R01 | P6 | 拒识与错误分析 | [outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/](../../outputs/analysis/EXP-P6-POPU-REJECT-20260820-R01/) |
| EXP-P6.1-POPU-CALIBRATION-20260820-R01 | P6.1 | 校准+一致性 | [outputs/analysis/EXP-P6.1-POPU-CALIBRATION-20260820-R01/](../../outputs/analysis/EXP-P6.1-POPU-CALIBRATION-20260820-R01/) |
| EXP-P7-CONFIG-VALIDATION-20260820-R01 | P7 配置验证 | canonical SHA + Smoke 校验 | [outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/](../../outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/) |
| **EXP-P7-FULL-20260820-R02** | P7 Full | **完整软件扰动 sweep** | 归档 `EXP-P7-FULL-20260820-R02.tar.gz`，**SHA-256 `cbaffa74878b149e546a42826ae373442c62683af890362684f80963e7fddda1`**，大小 672,702,773 字节，2,163 文件 |
| EXP-P7-FULL-ANALYSIS-20260821-R01 | P7 复现分析 | 8 个机读产物 | [outputs/analysis/EXP-P7-FULL-ANALYSIS-20260821-R01/](../../outputs/analysis/EXP-P7-FULL-ANALYSIS-20260821-R01/) |

### 10.2 主要配置路径

| 配置 | 路径 |
|---|---|
| 几何冻结 v0.2 | [popu_geometry_frozen_v0.2.json](../../configs/experiments/popu_geometry_frozen_v0.2.json) |
| 特征 v0.1 | [popu_features_p4a_v0.1.json](../../configs/experiments/popu_features_p4a_v0.1.json) |
| 传统基线 v0.1 | [popu_baseline_p5_v0.1.json](../../configs/experiments/popu_baseline_p5_v0.1.json) |
| 传统比较 v0.1 | [popu_model_comparison_p5_1_v0.1.json](../../configs/experiments/popu_model_comparison_p5_1_v0.1.json) |
| 神经网络 Mini v0.1 | [popu_neural_mini_v0.1.json](../../configs/experiments/popu_neural_mini_v0.1.json) |
| 神经网络 Full v0.1 | [popu_neural_full_v0.1.json](../../configs/experiments/popu_neural_full_v0.1.json) |
| P6 拒识 | [popu_p6_reject_v0.1.json](../../configs/analysis/popu_p6_reject_v0.1.json) |
| P6.1 校准 | [popu_p6_1_calibration_v0.1.json](../../configs/analysis/popu_p6_1_calibration_v0.1.json) |
| P7 鲁棒性 | [popu_p7_robustness_v0.1.json](../../configs/analysis/popu_p7_robustness_v0.1.json) |

### 10.3 关键测试

- `tests/test_neural_p7_runner.py`（60 个，含 Round 3+4 回归）
- `tests/test_neural_p7_robustness.py`（5 个）
- `tests/test_neural_p6_reject.py`
- `tests/test_neural_p6_1.py`
- `tests/test_neural_full_splits.py`
- `tests/test_neural_full_protocol.py`
- `tests/test_neural_full_runner.py`
- `tests/test_neural_p7_full_analysis.py`（54 单元 + 1 选做集成）
- `tests/test_experiment_artifacts.py`

最近一次全量测试基线：184 passed（[P7_RESULTS §3](../../docs/stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_RESULTS_v0.1.md)），529 passed（[P7_FULL §15](../../docs/stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_FULL_RESULTS_v0.1.md)）。

### 10.4 Git 提交基线

`ed363a1`（"feat: add PoPu P7 Full evidence re-verification"）。

### 10.5 P7 Full 归档 SHA-256

`cbaffa74878b149e546a42826ae373442c62683af890362684f80963e7fddda1`（EXP-P7-FULL-20260820-R02.tar.gz，672,702,773 字节，2,163 文件）。本报告**不重新解压或重算**该归档；引用现有哈希与现成分析产物。

---

## 11. 已知限制与未验证项

> 以下内容**未在 PoPu 现有证据中验证**，不应出现在 PoPu 阶段结论中。

1. 自研传感器的硬件密度、量程、漂移、串扰、饱和、单元失效。
2. 身体部位的真值监督（P3.2 区域监督继续 HOLD）。
3. 整夜睡眠或长时连续过程的稳定性。
4. 临床安全、舒适度、闭环控制。
5. SLP / PressurePose / PMD / TIP 等其他数据集的迁移（PoPu 数据是公开 Tactilus 64×27，与上述模态不同）。
6. 三模型一致性规则的 5-repeats ensemble 在 P7 中需三 repeat 才能形成，单 repeat 路径在 P7 Smoke 中触发 `ensemble_error` 并回退到 P6 single。
7. Per-subject 阈值——研究证据**报告**受试者差异，**不推荐**针对单受试者定阈值。

---

## 12. 报告级数字一致性自检

下列关键数字均能在源报告中找到；任何数字不在源报告中的，已标注"未在现有证据中验证"或避免写入。

- 5,160 / 60 / 64×27：见 P1 §4
- 5,006 / 94 / 0 / 60：见 P2 §4
- 1,730 / 1,670 / 60 / 0：见 P3.2 §4
- 51,000 / 50,060 / 940 / 22：见 P4a §5
- 0.9466：见 P5 §5.1（logreg held-out test macro-F1）
- 0.9452 / 0.9424 / 0.0028：见 P5.1 §5.1
- 0.9866 / 0.9866 / 0.8825：见 P5.2-C §4
- 0.94 / 0.9510 / 0.9956 / 0.0042：见 P6 §3 + 全量 JSON
- 0.75 / 0.5 / true / 0.9676 / 0.9949 / 0.0049：见 P6.1 §4
- 0.9866 / 0.6684 / 0.6820 / 0.9384：见 P7 Full §3 + §5
- 65.6%：见 P7 Full §7.2

---

## 13. 报告完成时的自检

| 检查项 | 状态 |
|---|---|
| 所有引用路径与配置路径已在仓库存在或属外部归档（仅引用哈希） | 已核对 |
| 所有关键数字可在源报告或机器可读 JSON 中追溯 | 已核对（§12） |
| 不暴露任何密码、Token、个人信息或原始受试者可识别数据 | 已核对 |
| 未修改任何 P1–P7 阶段报告、算法代码或实验结果 | 已核对 |
| 未重新解压或重算 EXP-P7-FULL-20260820-R02.tar.gz | 已核对（仅引用 SHA） |
| 本报告撰写时尚未提交 Git | 已核对（仅新增本文） |

## 14. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-21 | 首次撰写 v0.1：基于现有 P1–P7 报告 + 机器可读 JSON/CSV + 已记录的 EXP-P7-FULL-20260820-R02 归档 SHA；不重训、不重跑 P7、不改已有代码/结果、不提交 Git。 |