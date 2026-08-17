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

`WINDOWS-RESEARCH-WORKBENCH / POPU-P3-PARTIAL`

已经完成：

- Windows Python 3.12.13 独立 `.venv`；
- `uv.lock` 依赖锁；
- Windows本地PoPu与PMD数据副本；
- 环境与数据路径健康检查；
- 真实PoPu Tactilus JSON读取；
- 单帧64×27二维热力图；
- 五状态共同色标总览；
- P1 全量盘点、P2 质量门、P3 首版接触 Mask/Geometry 都已有可追溯输出；
- P3.1 Mask 候选比较与 P3.2 COCO 区域标注对齐审计已具备代码和单元测试，但尚未生成真实数据输出。

P1/R1 的全量 Inventory 已执行。其实现逐个读取 JSON，只在内存中保留当前文件与最终的紧凑清单，不会把 5,160 个记录或所有压力矩阵整体载入内存。

尚未完成：P3.1 Mask 策略冻结、P3.2 标注配对审计、特征表、姿态/部位 Baseline、UNKNOWN/REJECT、Density、Fault、区域算法和最终研究报告。

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
- [验证 Workflow 总蓝图：有什么、缺什么、怎么验证、能得到什么](docs/VALIDATION_WORKFLOW_MASTER.md)
- [阶段记录与报告约定](docs/STAGE_REPORTING_CONVENTION.md)
- [Windows整体方向与研究阶段](docs/WINDOWS_RESEARCH_WORKBENCH_DIRECTION.md)
- [文件夹结构与数据流](docs/FOLDER_STRUCTURE_AND_DATA_FLOW.md)
- [团队文件接收与验收清单](docs/TEAM_INTAKE_CHECKLIST.md)
- [PoPu二维热力图说明](docs/POPU_HEATMAP_CONTRACT.md)

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

P1/R1 全量 PoPu Tactilus Inventory（会写入 CSV、JSON 和 PNG；本次尚未执行）：

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
