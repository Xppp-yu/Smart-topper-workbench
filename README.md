# Smart Topper Windows Research Workbench

这是智能顶垫压力感知算法的 **Windows全量研究、快速实验与结果解释工作台**。

项目位置：

```text
E:\TeamProjects\smarttopper-team-workbench
```

公共数据位置：

```text
E:\TeamProjects\datasets\smart-topper
```

正式工程位置：

```text
/home/charles/projects/smart-topper-pressure-baseline
```

## 当前状态

`WINDOWS-RESEARCH-WORKBENCH / POPU-P6.1-COMPLETE / P7-PROTOCOL-READY / SLP-S0-COMPLETE`

已经完成（均有真实产物与阶段报告）：

- Windows Python 3.12.13 独立 `.venv`；`uv.lock` 依赖锁；
- Windows本地PoPu与PMD数据副本；
- 环境与数据路径健康检查；真实PoPu Tactilus JSON读取；单帧64×27二维热力图；五状态共同色标总览；
- P1 全量盘点、P2 质量门、P3 首版接触 Mask/Geometry 都已有可追溯输出；
- P3.1 Mask 候选策略比较（冻结 `largest_component`）与 P3.2 COCO 区域标注对齐审计（区域监督 HOLD）均已真实运行并记录结论；
- P4a 无标签逐 snapshot 特征表（51,000 行 × 71 特征）已首轮真实运行；
- P5 受试者隔离五分类 Baseline（dummy/logreg/rf/knn）已首轮真实运行；`logreg` 为历史首轮领先候选（primary test macro-F1 0.9466），未冻结；
- P5.1 repeated subject-grouped CV 横向比较已真实运行并冻结 `calibrated_linear_svm` 为 **传统模型候选**（PoPu research candidate；7 候选 × top-2 特征消融 × 全量 OOF/record/逐受试者产物；record macro-F1 0.9452，logreg 0.9424 在 margin 内统计平局、由 tie-break 1 胜出；16,097 B，独立重载 smoke OK）。该候选是已冻结的传统模型，不是已覆盖 CNN 的总体最优模型。

P1/R1 的全量 Inventory 已执行。其实现逐个读取 JSON，只在内存中保留当前文件与最终的紧凑清单，不会把 5,160 个记录或所有压力矩阵整体载入内存。

P5.2-A CNN 训练底座与 CPU/CUDA Smoke、P5.2-B Mini 筛选和 P5.2-C Full 公平比较均已完成。P5.2-C 在 RTX 4090 上完成 3 模型 × 3 repeats × 5 folds 共 45 个训练单元，Reviewer 独立复核后接受 `small_resnet` 为 PoPu 固定睡姿五分类总体研究候选模型族（record macro-F1 `0.986649 ± 0.002832`）；P5.1 `calibrated_linear_svm` 保留为传统模型对照。P6/P6.1 UNKNOWN/REJECT、错误复核、温度校准与三模型一致性模拟已完成：总体拒识风险下降，但存在 24 条跨 repeat 持续错判，一致性规则以覆盖率换取更低 WAR，因此只保留为研究候选。P7 降密度、噪声、坏点、坏行坏列协议与扰动代码已就绪，模型扰动推理尚未运行。PoPu 结果仍不等于外部数据集、自研硬件或产品验证。

SLP 已进入两阶段连续开发：S0 全量结构 Inventory 已完成（109 名受试者、1,941 个模态组合、1,939 完整、2 个 `simLab cover2/depthRaw` 组合 quarantine）。SLP 只有 RGB/IR 原始人工 14 节点，不含腰、臀等像素级身体区域真值。区域线按“节点几何 R0 → OpenCV 预标注 R1 → 人工复核 R2 → 双审共识 R3”推进；只有 R2/R3 可作为默认区域训练参考。

## 新的固定定位

### Windows负责研究

- 用户决定问题、方向、优先级和是否继续；
- Codex把问题转为可运行实验，编写代码并执行；
- 使用真实全量数据生成CSV、JSON、图片、模型和报告；
- 同时交付结果解释、失败原因、结论边界和下一步建议；
- 允许快速尝试多个算法版本，但必须保存配置、测试和证据。

### WSL负责工程化与正式交付

- 只接收Windows中已经验证并冻结的候选方案；
- 建立稳定的 `PressureSample`、Adapter、Quality、Geometry、Features和Model接口；
- 提供严格配置、单元测试、Replay、CLI/API和硬件Adapter；
- 支持后续项目调用和团队正式交付；
- 不允许Windows试验代码自动覆盖WSL工程。

详细规则见：

- [项目总状态与阶段入口](docs/PROJECT_STATUS.md)
- [Codex × Claude Code 协作约定（四层角色 / TASK-ID / EXP-ID）](COLLABORATION_WORKFLOW.md)
- [实验治理与远程 GPU 执行方案](docs/EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md)
- [验证 Workflow 总蓝图：有什么、缺什么、怎么验证、能得到什么](docs/VALIDATION_WORKFLOW_MASTER.md)
- [阶段记录与报告约定](docs/STAGE_REPORTING_CONVENTION.md)
- [Windows整体方向与研究阶段](docs/WINDOWS_RESEARCH_WORKBENCH_DIRECTION.md)
- [文件夹结构与数据流](docs/FOLDER_STRUCTURE_AND_DATA_FLOW.md)
- [团队文件接收与验收清单](docs/TEAM_INTAKE_CHECKLIST.md)
- [PoPu二维热力图说明](docs/POPU_HEATMAP_CONTRACT.md)
- [SLP 两阶段连续开发总计划](docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md)
- [SLP Agent 连续开发任务清单](docs/SLP_AGENT_TASK_BACKLOG_v0.1.md)
- [SLP S0 全量 Inventory 与标注边界](docs/stage_reports/S0_SLP_FULL_INVENTORY_AND_ANNOTATION_BOUNDARY_v0.1.md)
- [私密 GitHub 与多 Agent 交接清单](docs/PRIVATE_GITHUB_AND_AGENT_HANDOFF_CHECKLIST_v0.1.md)

## 环境

```text
Python：3.12.13
虚拟环境：.venv
依赖管理：uv
VS Code解释器：.venv\Scripts\python.exe
```

初始化或恢复环境：

```powershell
cd E:\TeamProjects\smarttopper-team-workbench
uv sync --python 3.12
```

健康检查：

```powershell
uv run python scripts\run_healthcheck.py --config configs\paths.local.json
```

自动测试：

```powershell
uv run pytest -q
```

P1/R1 全量 PoPu Tactilus Inventory（会写入 CSV、JSON 和 PNG；已执行，重跑会覆盖同名 v0.1 产物）：

```powershell
uv run python scripts\inventory_popu.py --config configs\paths.local.json
```

如需生成逐文件 SHA-256 冻结清单，可额外加 `--include-sha256`；这会增加 I/O 时间，不是首次数据结构盘点的必需项。

P2/R2 质量门与样本画廊（读取 P1 Inventory，逐条处理有标签 JSON；已产生质量 CSV、汇总 JSON 和两张复核图；重跑会覆盖同名 v0.1 产物）：

```powershell
uv run python scripts\quality_popu.py
```

`WARN` 仅表示相对于同姿态统计分布的异常候选，必须看异常样本图后才能认定为问题；无固定姿态标签的 `others.json` 会被保留为 `EXCLUDED`，不会混入训练数据。

P3/R3 接触 Mask 与 Geometry（读取 P2 质量结果，保留 `ACCEPT` 与 `WARN`，输出逐记录几何 CSV、汇总 JSON 和两张叠加图）：

```powershell
uv run python scripts\geometry_popu.py
```

该 Mask 是可重复的相对信号边界，不能称为真实解剖部位分割、标定压力边界或产品接触面积。

P3.2/R3.2 PoPu 身体区域标注—压力记录对齐审计（只读 COCO 文件与 Tactilus 文件名；不会训练模型，也不会把有歧义的文件当作帧级真值）：

```powershell
uv run python scripts\audit_popu_segmentation.py --config configs\paths.local.json
```

它会输出每份 COCO 标注对应的候选 Tactilus 记录数、画布/类别/多边形坐标检查和明确的 `ONE_TO_ONE_CANDIDATE` / `AMBIGUOUS_TACTILUS_CANDIDATES` 等状态。只有有文档可追溯的逐记录、逐帧配对规则通过审计后，才允许把区域标注用于监督训练。

P3.1/R3.1 Mask 候选策略比较（相对阈值过滤、仅最大连通域、相对阈值后闭运算；会逐记录比较同一记录内相邻帧稳定性）：

```powershell
uv run python scripts\compare_popu_mask_strategies.py
```

该结果只能用于筛选稳定的 Geometry 输入规则；稳定不等于解剖正确，仍须配合叠加图和 P3.2 标签对齐审计人工复核。

P4a/R4a 无标签特征表（读取 P1/P2 结果，逐 snapshot 生成 71 列数值特征；特征与标签/追溯列严格分离；`others.json` 只进 EXCLUDED manifest）：

```powershell
uv run python scripts\features_popu.py
```

P5/R5 受试者隔离姿态 Baseline（读取 P4a 特征表；12 个 held-out 受试者 + 开发集 GroupKFold 选型；dummy/logreg/rf/knn；primary 与 combined 双口径）：

```powershell
uv run python scripts\baseline_popu.py
```

注意：P5 v0.1 对每个候选模型各评估了一次 held-out test，模型选择未读取任何 test 分数；`logreg` 仅为历史首轮领先候选。

P5.1/R5.1 受试者分组横向比较与候选冻结（读取 P4a 特征表；7 候选 × repeated subject-grouped CV（5 折 × 3 repeats，group=subject_id）× top-2 特征消融；冻结 winner=`calibrated_linear_svm` 为 `popu_research_candidate_p5_1_v0.1`）：

```powershell
uv run python scripts\model_comparison_popu.py --config configs\experiments\popu_model_comparison_p5_1_v0.1.json
```

注意：结果为 PoPu 公开数据上的**研究候选**，不是产品模型、未经外部验证；候选以 `FrozenClassOrderClassifier` 包裹以保证冻结标签顺序，独立 joblib 重载后 predict/predict_proba 冒烟通过；完整结果见 [P5.1 阶段报告](docs/stage_reports/P5_1_POPU_GROUPED_MODEL_COMPARISON_v0.1.md)。

## PoPu二维热力图

单张左卧图：

```powershell
uv run python scripts\preview_popu.py --subject 1 --posture left --variation 1 --frame-index 0
```

五状态总览：

```powershell
uv run python scripts\preview_popu.py --subject 1 --variation 1 --frame-index 0 --overview
```

输出进入 `outputs/figures/`，PNG旁边同时保存JSON元数据。

## 每次研究任务的标准交付

```text
代码
+ 运行命令
+ 自动测试
+ 全量或明确范围的真实数据结果
+ 图、CSV、JSON或模型
+ 结果解释
+ 结论边界
+ 下一步决策建议
```

任何“完成”都必须由真实输出和测试支持；只有代码文件、空目录或计划文档不能算完成。

## 研究边界

- PoPu和PMD是公开数据研究，不等于自研传感器验证。
- 软件降采样不等于真实硬件密度测试。
- Replay不等于气囊物理控制闭环。
- 身体接触Mask不等于Head/Torso/Arm/Leg身体部位分割。
- PMD和PoPu不能逐行配对为同一个监督样本。
