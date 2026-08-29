# 实验治理与远程 GPU 执行方案 v0.1

最后更新：2026-08-19
文档状态：`PROPOSAL_FOR_IMPLEMENTATION`
适用范围：Windows 研究工作台的 PoPu、SLP2022、PressurePose 及后续自研同步数据研究。

> 本文是待实施方案，不代表对应代码、实验或云端环境已经完成。它不改写 P0-P5.1 的历史结果，也不授权把公开数据结论外推为产品验证。

## 1. 决策摘要

当前流程应从“一个 Agent 负责设计、开发、全量计算、自审和总结”调整为四层职责：

1. **Controller（Codex）**：定义研究问题、变量、评价协议、Gate 和单张任务单。
2. **Coding Agent（Claude Code）**：读代码、实现、调试、单元测试和小数据 Smoke Test；通过后即停止，不陪跑全量实验，不自行给出最终研究结论。
3. **Experiment Runner（本地计算机或租用服务器）**：基于冻结 Git SHA 与配置后台执行 Mini/Full Run，保存日志、指标、预测、图和 checkpoint。
4. **Reviewer（Codex）**：只读复核配置、数据版本、指标、关键图和失败样本，给出 `ACCEPT / ITERATE / STOP / INVALID`。

研究路线同步调整为：

```text
P5.1 传统模型候选（已完成，保留）
    -> P5.2 PoPu 神经网络公平比较
    -> 冻结 PoPu 总体候选
    -> P6 UNKNOWN/REJECT
    -> P7 软件鲁棒性
    -> PoPu 参考验证包
    -> SLP/PressurePose Adapter 与各自任务线
```

P5.1 的 `calibrated_linear_svm` 是**已冻结的传统模型候选**，不是已经覆盖 CNN 的总体最优模型。P6 在 P5.2 完成总体候选选择前保持等待，避免先为一个可能被替换的模型冻结阈值。

## 2. 为什么该调整是科学的

### 2.1 保留既有证据，不倒推改写

- P5.1 的 repeated subject-grouped CV、特征消融、预测明细和冻结模型继续有效。
- P5.2 是前瞻性新增的模型族比较，不修改 P5.1 数值和历史报告。
- PoPu 没有独立、从未参与研发决策的最终测试集，因此 P5.2 只能称为开发期受试者分组复核，不能包装成最终外部验证。
- SLP/PressurePose 接入后分别建立 Adapter、Manifest、标签与坐标合同；不同数据集分别评价，不逐行拼接成同一受试者样本。

### 2.2 防止开发集、测试集和 Agent 决策相互污染

- 模型选择只读开发期 OOF 或验证结果，不读取被定义为最终 test 的结果。
- SLP 等后续数据在调参前冻结 test subjects；test 只在候选与协议冻结后运行一次。
- Coding Agent 不根据 Full Run 结果不断改代码；下一轮修改必须由 Reviewer 形成新的 `EXP-ID` 和任务单。
- Reviewer 与实现者分离。Codex 同时担任 Controller 和 Reviewer，属于流程内独立复核，不等同于论文级盲审；关键产品或发布结论仍应增加人工/第二审阅者。

## 3. 当前事实基线

截至 2026-08-19 的只读检查：

| 项目 | 当前事实 |
|---|---|
| Windows 工作台 | `E:\TeamProjects\smarttopper-team-workbench`，Git 工作区干净 |
| 本地 GPU | NVIDIA GeForce RTX 4060 Laptop，8188 MiB 显存 |
| SLP2022 | 130,749 个文件，22.22 GiB，`PRESENT_NOT_INTEGRATED` |
| PressurePose | 161 个文件，21.93 GiB，`PRESENT_NOT_INTEGRATED` |
| 两套后续数据合计 | 约 44.15 GiB 原始文件；尚未计入缓存、分片、checkpoint 和实验产物 |
| 当前 Python 依赖 | NumPy、pandas、scikit-learn；尚未引入 PyTorch |
| 共享数据边界 | `E:\TeamProjects\datasets\smart-topper` 只读；派生数据写入工作台 `data/processed/` 或 `outputs/` |

SLP 的 13 万个小文件意味着后续速度可能同时受 CPU 解码、文件系统和数据上传影响，不能只升级 GPU。正式上云前应生成可追溯的版本化 Manifest，并在不改变原始数据的前提下建立训练分片或缓存。

## 4. TASK-ID 与 EXP-ID

### 4.1 TASK-ID：开发任务

示例：`TASK-P5.2-A-CNN-SCAFFOLD-v0.1`

任务单必须包含：

- 目标、非目标和允许修改的目录；
- 输入合同、数据子集和配置；
- 需要新增/修改的模块；
- Unit Test、错误测试、真实小样本 Smoke Test；
- 禁止执行的 Mini/Full 命令；
- 交付文件、Git commit 和已知限制。

### 4.2 EXP-ID：不可变计算实验

建议格式：`EXP-P5.2-<MODEL>-<SCOPE>-<YYYYMMDD>-RNN`，例如：

```text
EXP-P5.2-TINYCNN-SMOKE-20260819-R01
EXP-P5.2-TINYCNN-MINI-20260820-R01
EXP-P5.2-TINYCNN-FULL-20260822-R01
```

同一 `EXP-ID` 的 Git SHA、resolved config、数据 Manifest 和 split Manifest 一旦进入 `QUEUED` 不再修改。参数变化必须创建新 `EXP-ID`。

## 5. 实验生命周期

```text
DRAFT
  -> CODE_READY
  -> SMOKE_PASS
  -> QUEUED
  -> RUNNING
  -> SUCCEEDED | FAILED
  -> REVIEWED
  -> ACCEPTED | REJECTED | INVALID
```

关键 Gate：

| Gate | 必须满足 |
|---|---|
| `CODE_READY` | 单元测试通过；配置可解析；无全量结果声明 |
| `SMOKE_PASS` | 100-1000 样本或 1-2 个受试者，1 epoch；前向、反向、保存、加载和指标链路可运行 |
| `QUEUED` | Git SHA、数据版本、split、随机种子、硬件要求和预算上限冻结 |
| `SUCCEEDED` | 退出码 0、预期产物齐全、无 NaN/数据泄漏/样本重复等完整性失败 |
| `ACCEPTED` | Reviewer 已完成可重复性、方法、指标、错误案例和限制复核 |

`FAILED` 不是研究失败结论。只有日志与错误证据；修复后建立新的 TASK/EXP，不覆盖原记录。

## 6. 建议的增量仓库结构

以下为目标结构，由 Claude Code 分批实施；不移动现有稳定模块，不重写 P0-P5.1 历史：

```text
docs/
  EXPERIMENT_GOVERNANCE.md
configs/
  experiments/
    schema/
    <EXP-ID>.yaml
src/topper_perception/
  experiments/
    contracts.py
    runner.py
    artifacts.py
    registry.py
scripts/
  run_experiment.py
tests/
  test_experiment_contracts.py
  test_experiment_runner.py
outputs/experiments/                 # Git ignore
  <EXP-ID>/
    status.json
    resolved_config.yaml
    manifest.json
    logs/
    metrics.json
    metrics_by_subject.csv
    predictions.parquet
    plots/
    checkpoints/
    DONE.json | FAILED.json
```

`manifest.json` 至少记录：`experiment_id`、`git_commit`、`git_dirty`、`config_sha256`、`data_manifest_sha256`、`split_sha256`、`model_version`、Python/依赖版本、CUDA/GPU、CPU/RAM、开始结束时间、随机种子和命令行。上述 identity 必须由产物自身携带；外部终端记录或聊天记录不能替代。任一字段缺失、漂移或无法复算时，不得进入正式 Mini/Full。

第一版只使用 Git + YAML/JSON + Python Runner + 文件产物注册，不引入 Airflow、Kubernetes 或大型 MLOps 平台。实验数量和机器数量显著增长后，再评估 MLflow/Optuna。

## 7. PoPu P5.2 的执行分层

### P5.2-A：CNN 训练底座与 Smoke

候选范围：

- P5.1 `calibrated_linear_svm`：冻结对照，不重训改写；
- 原始压力矩阵 MLP：非卷积神经网络对照；
- TinyCNN：主要轻量 CNN 候选；
- Small ResNet：容量稍高的 CNN 候选；
- CNN + 71 个工程特征：仅在前述模型显示明确缺口时追加。

通过条件：受试者切分隔离、train-fold normalization、标签映射、左右翻转标签交换、checkpoint/resume、CPU 与 CUDA Smoke、模型重载预测全部通过。此阶段不跑 Full CV。

### P5.2-B：Mini 筛选

- 只用开发受试者的固定子集；
- 3-5 epochs，固定种子和早停规则；
- 用于排除明显不可行的架构/输入方案，不形成最终排名；
- 只有通过资源、稳定性和方向性 Gate 的候选进入 Full。

### P5.2-C：Full 公平比较

- 与 P5.1 使用相同的受试者隔离原则、记录聚合和主指标；
- 主指标为 record macro-F1；同时报告 repeated splits 波动、最差受试者、逐类别、校准、参数量、推理时间和训练成本；
- 若神经网络相对 SVM 的提升小于 0.005，且稳定性/难例没有实质改善，优先保留更简单的 SVM；
- 只有 Reviewer 接受后才冻结 PoPu 总体候选并进入 P6。

## 8. 本地与远程算力分工

| 任务 | 建议执行位置 |
|---|---|
| 代码、单元测试、格式检查 | 本地 CPU |
| 传统 ML、单折冒烟、小型特征处理 | 本地 CPU |
| CNN Smoke、单模型 Mini Run | 本地 RTX 4060 Laptop 8GB |
| PoPu CNN Full CV、多个候选或超参筛选 | 远程 24GB GPU |
| SLP 多模态、PressurePose 3D/较大模型 | 远程 48GB GPU |
| 80GB A100/H100 | 仅在 48GB OOM、吞吐收益经测算或多 GPU 分布式确有必要时 |

Mixed Precision 是 Runner 的配置项，不是默认结论：支持时优先 BF16；不支持时使用 FP16 + GradScaler。它可以降低显存和提高吞吐，但不能修复小文件 I/O、数据泄漏或错误实验设计。

## 9. 服务器配置建议

### 9.1 推荐策略：保存两套模板，不长期绑定一台机器

**模板 A：日常全量训练，优先使用**

| 资源 | 建议 |
|---|---|
| GPU | RTX 4090 24GB；价格/性能优先 |
| CPU | 8-16 vCPU；若平台的 4090 套餐 CPU 较少，必须先使用数据分片降低小文件压力 |
| RAM | 48-64GB |
| 临时 NVMe | 300-500GB |
| 持久卷 | 200GB 起步；同时缓存 SLP/PressurePose 时建议 500GB |
| 适用 | PoPu CNN、压力矩阵模型、较小 SLP 单模态/双模态实验 |

**模板 B：覆盖后续重任务的稳妥档**

| 资源 | 建议 |
|---|---|
| GPU | L40S 48GB；预算更紧且能接受较慢训练时可用 RTX A6000 48GB |
| CPU | 16 vCPU |
| RAM | 64-128GB，推荐约 96GB |
| 本地/网络存储 | 500GB 起步，需要保留多模态缓存和多个 checkpoint 时 1TB |
| 适用 | SLP RGB/IR/depth/pressure 融合、PressurePose 3D、较大 batch、较大超参搜索 |

当前决定先使用 AutoDL。首次实例选 **RTX 4090 24GB + 至少 8 vCPU（优先 12-16）+ 至少 32GB RAM（优先 64GB）**；PoPu 阶段先用平台默认免费数据盘。仅在显存或多模态任务触发升级条件时，再从 AutoDL 当时的算力市场选择 32GB/48GB/80GB 档，不提前长期占用高价实例。

### 9.2 何时升级 24GB -> 48GB

满足任一条件才升级：

- 合理 batch、AMP、gradient accumulation 后仍发生 CUDA OOM；
- 多模态模型需要同时驻留多个 backbone 或高分辨率张量；
- 3D/体素/大热图任务的单样本显存显著增大；
- 24GB 上单次 Full Run 的预计时间导致实验周转不可接受，并且 48GB 实测吞吐可以抵消价格差。

不因“模型叫 CNN/神经网络”自动升级 A100/H100。

## 10. 当前价格快照与预算

以下只用于 2026-08-19 的 AutoDL 预算估算，创建实例前必须在算力市场重新确认。AutoDL 官网当前展示的参考价包括：RTX 3090 24GB `¥1.32/h`、RTX 4090 24GB `¥1.88/h`、RTX 5090 32GB `¥2.78/h`、A800 80GB `¥4.98/h`、H20 96GB `¥7.58/h`。不同地区、主机、库存和会员折扣可能不同。

| GPU | 参考价/小时 | 100 GPU 小时约成本 | 当前用途 |
|---|---:|---:|---|
| RTX 3090 24GB | ¥1.32 | ¥132 | 低价兼容/非紧急实验 |
| RTX 4090 24GB | ¥1.88 | ¥188 | 当前 PoPu CNN 默认选择 |
| RTX 5090 32GB | ¥2.78 | ¥278 | 24GB OOM 或吞吐实测值得升级时 |
| A800 80GB | ¥4.98 | ¥498 | 48GB 以下仍无法容纳的重任务 |
| H20 96GB | ¥7.58 | ¥758 | 当前阶段不默认使用 |

这不是性能或可用性保证。首次不为“可能将来需要”购买 80GB/96GB 卡；先用 4090 测量显存、每 epoch 时间和数据加载利用率，再按 Gate 升级。

AutoDL 主机通常提供免费 50GB 数据盘；付费扩容数据盘即使实例关机也继续按日计费，官网文档给出的会员参考价约为 `¥0.0066/GB/日`，实际以主机页面为准。PoPu 阶段不扩容；SLP/PressurePose 分片确定后再扩到满足数据、缓存和 checkpoint 的容量。重要产物必须同步回本地，不能把实例本地盘当永久备份。

首轮预算建议：

- 账户首次充值/硬上限：¥100；
- 先做 5-10 小时 PoPu P5.2 远程试跑，测出每折/每 epoch 实际时间和显存；
- 根据测量结果计算 Full CV 预算，再决定继续 4090 还是升级显存档；
- 单个实验必须配置预计最长时长、checkpoint、失败退出和完成后关机/终止提醒。

## 11. 云端数据与执行方式

```text
本地只读 raw datasets
    -> 版本化 Manifest + 校验哈希
    -> 可追溯训练分片/缓存
    -> 一次上传到 AutoDL 数据盘
    -> 拉取指定 Git SHA / 容器镜像
    -> run_experiment.py --config <EXP-ID>.yaml
    -> 后台训练 + checkpoint + status
    -> 同步 metrics/plots/predictions/checkpoints 回本地
    -> Codex 只读 Review
    -> 关闭实例，按保留策略清理付费数据盘和云端缓存
```

要求：

- 用锁定版本的 Docker/CUDA/PyTorch 环境或等价可复现实例；
- 原始数据不写回、不原地清洗；
- 不为每次实验重新上传 13 万个 SLP 小文件；
- 训练过程与 Agent 会话解耦，使用后台进程/作业状态，不让 Claude Code 等待；
- checkpoint 必须支持中断恢复，日志必须实时落盘；
- 云端上传前复核每个数据集的许可、再分发和非商业使用约束；未来自研数据必须使用私有、加密存储和最小权限。

## 12. Claude Code 实施批次

每个批次单独任务、单独提交、单独验收。不得一次把 A-C 合并成巨型任务。

### Batch A：文档和状态口径修订

目标：把本文决策同步到现有导航，但不写 Runner、不引入 PyTorch、不运行实验。

允许修改：

- `COLLABORATION_WORKFLOW.md`
- `docs/PROJECT_STATUS.md`
- `docs/POPU_REFERENCE_PIPELINE_ROADMAP.md`
- `docs/VALIDATION_WORKFLOW_MASTER.md`
- `README.md`

必须完成：

- 将 P5.1 标为传统模型候选已冻结；
- 新增 P5.2-A/B/C 与进入 P6 的 Gate；
- 写明四层角色、TASK-ID/EXP-ID 和 Agent 不跑 Full 的边界；
- 保留所有 P0-P5.1 历史数值和报告链接；
- 修正文档互相矛盾处。

验收：`git diff --check`；本地链接/状态一致性检查；仅文档提交；工作区干净。

### Batch B：最小 Experiment Runner

目标：实现配置解析、状态机、Manifest、产物目录和一个不依赖深度学习的 dummy/smoke 实验。

限制：

- 不移动现有 `features/models/evaluation` 模块；
- 不改 P5.1 模型与结果；
- 不实现远程平台专有 API；
- 不运行 PoPu/SLP/PressurePose Full。

验收：Unit Test + 临时目录 Smoke；同一 config 可复现；脏工作区拒绝进入 `QUEUED`；失败能生成 `FAILED.json`；成功能生成完整 Manifest 和 `DONE.json`。

### Batch C：P5.2-A CNN Scaffold

目标：引入 PyTorch、PoPu matrix Dataset/DataLoader、MLP/TinyCNN/Small ResNet 注册、训练与评估接口，以及 1 epoch 小真实数据 Smoke。

限制：不做 Mini/Full、不调最终超参、不冻结候选、不进入 P6。

验收：受试者隔离、fold 内 normalization、增强标签变换、CPU/CUDA Smoke、checkpoint/resume、模型独立重载、固定 seed 最小复现。

Batch C 通过后，由 Controller 另行签发 P5.2-B 的 EXP 配置；Runner 负责计算，Claude Code 不等待结果。

## 13. 本方案的停止条件

以下任一情况发生时停止推进并先修复协议：

- 数据集许可不允许当前云端存储/处理方式；
- split 或归一化存在受试者泄漏；
- EXP 无法绑定 Git SHA、数据 Manifest 或 resolved config；
- 产物被覆盖，或同一 EXP-ID 对应多套参数；
- Full Run 在 Smoke/Mini 未通过前启动；
- Agent 根据 test 结果反复调参；
- 公开数据结论被写成自研硬件、舒适性或闭环效果验证。

## 14. 参考来源

- AutoDL 当前 GPU 参考价格（部署前重新核对）：https://www.autodl.com/
- AutoDL 实例与付费数据盘计费：https://www.autodl.com/docs/price/
- AutoDL 本地数据盘：https://www.autodl.com/docs/local_disk/
- AutoDL 无卡模式与自动关机：https://www.autodl.com/docs/save_money/
- AutoDL 基础镜像版本：https://www.autodl.com/docs/base_config/
- NVIDIA RTX 4090 规格（24GB）：https://www.nvidia.com/en-eu/geforce/graphics-cards/40-series/rtx-4090/
- NVIDIA RTX A6000 规格（48GB ECC）：https://www.nvidia.com/en-gb/products/workstations/rtx-a6000/
- NVIDIA L40S 规格（48GB）：https://www.nvidia.com/es-la/data-center/l40s/
