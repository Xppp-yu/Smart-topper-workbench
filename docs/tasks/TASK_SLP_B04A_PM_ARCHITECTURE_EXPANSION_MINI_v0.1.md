# TASK-SLP-B04A-PM-ARCHITECTURE-EXPANSION-MINI-v0.1

状态：`DOCUMENTATION_REVIEW_PENDING / IMPLEMENTATION_NOT_STARTED`

## Objective

在不改写 B04 历史协议和结果的前提下，扩大 SLP8 PM-only 区域分割的架构假设覆盖范围，并通过前瞻性冻结的 Mini Gate 最多保留 1–2 个候选供 B07 Full 协议使用。

## Why now

B04 只比较了 B03 延续的极简 TinyFCN 与一个正式分割候选 SmallUNet。R05 证明 SmallUNet 可学习且 TinyFCN 不可行，但不能据此声称 SmallUNet 已经过充分架构比较。B07 若只运行一个候选，只能称为泛化/稳定性确认，不能称为有充分候选池的架构公平比较。

## Prerequisites

- B01 冻结训练表和 TEST 防泄漏合同保持不变；
- B02 非学习 baseline `0.205644` 保持历史值，不重新估计；
- B04 R05 结果保持不变；SmallUNet 作为 incumbent；TinyFCN 不再进入候选池；
- 本任务的协议必须先经 Reviewer 接受，之后才能另开实现任务。

## Candidate hypotheses

| 候选 | 架构假设 | 当前状态 |
|---|---|---|
| `slp8_small_unet_v0.1` | 经典 encoder-decoder + skip connection | incumbent，已有 B04 实现与结果 |
| ResUNet-lite | residual CNN 是否改善特征学习和训练稳定性 | `NOT_IMPLEMENTED` |
| DeepLabV3+-lite | atrous/multi-scale context 是否改善不同身体区域尺度 | `NOT_IMPLEMENTED` |
| SegFormer-B0 | global attention 是否改善整体空间关系 | `DEFERRED_UNLESS_FAIRNESS_CONTRACT_FROZEN` |

SegFormer-B0 只有在运行前明确冻结单通道输入适配、是否使用预训练权重、输入 resize、增强、参数/显存档位和优化策略后才可纳入。若无法与 CNN 候选形成可解释比较，则本轮标记 `DEFERRED`，不得临时加入。

## Data boundary

- 数据：B01 TRAIN 3,645 / VAL 450；TRAIN 81 subjects / VAL 10 subjects；
- TEST rows、labels、onehot 和类别统计：`0 loaded`；
- setting/cover：danaLab / uncover；
- pressure semantics：`raw_pmarray_response`，不是 kPa；
- reference provenance：`V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`；
- source review status：`NOT_REVIEWED`。

## Protocol items to freeze before implementation

1. 候选最终名单、模型版本和参数/资源 tier；
2. 输入适配、初始化/预训练策略和增强边界；
3. 固定训练预算、优化器、学习率策略、early stopping 和预注册 seeds；
4. Macro IoU baseline margin、class-collapse、worst-subject 和 per-region guardrail 的精确定义；
5. near-tie margin 与最多 1–2 个候选的选择顺序；
6. checkpoint/resume/reload、运行 identity、产物和停止条件；
7. Reviewer 独立重算范围。

在上述项目冻结前，本文中的示例数值不得被当作已冻结实验配置。

## Required Gate families

### Hard fail-closed gates

- subject overlap、manifest 或 frozen hash mismatch；
- 任何 TEST 读取；
- NaN/Inf、OOM、训练参数未变化或未达到最小 epoch；
- 明确 class collapse；
- checkpoint reload 数值或 prediction hash 不一致；
- 参数、wall time、总时间或 CUDA peak 超出冻结预算；
- 运行产物缺少 identity 字段。

### Performance eligibility

- fixed foreground Macro IoU 必须超过 B02 baseline `0.205644` 加预注册绝对 margin；
- 全部 8 个区域必须报告 IoU/Dice/precision/recall 与预测支持；
- worst-subject、每个 posture、centroid error 必须报告；
- class-collapse 和 worst-subject 的数值阈值必须在看到新结果前冻结。

### Selection

通过硬 Gate 的候选按预注册 seeds 的汇总 Macro IoU 排序。near-tie 时依次比较 worst-subject、最差区域、seed 波动、推理时间/显存/参数量；仍接近时优先简单模型。最多保留 1–2 个候选。Mini 只做筛选，不宣布最终冠军。

## Experiment identity hard gate

每个正式产物必须内嵌并可复算：

```text
experiment_id
git_commit
git_dirty
config_sha256
data_manifest_sha256
split_sha256
model_version
```

外部终端记录或聊天记录不能补代缺失字段。

## Staged execution

1. `B04A-PROTOCOL`：只写协议/配置/schema/测试合同，不写模型，不运行研究计算；
2. `B04A-IMPLEMENTATION-SMOKE`：实现候选和单元测试，只做 CPU/最小 CUDA Smoke；
3. `B04A-MINI-RUN`：Owner 单独授权后由 Experiment Runner 执行真实 GPU Mini；
4. `B04A-REVIEW`：Codex 独立检查产物、指标、失败案例和 identity；
5. Reviewer 接受后才解锁 B07。

## Files allowed to change in the documentation stage

- `README.md`
- `COLLABORATION_WORKFLOW.md`
- `docs/PROJECT_STATUS.md`
- `docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md`
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`
- `docs/VALIDATION_WORKFLOW_MASTER.md`
- `docs/EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md`
- `docs/tasks/`
- `docs/stage_reports/`
- `docs/deliverables/`

## Out of scope for this documentation stage

- 模型代码、配置实现、测试代码；
- 数据处理、Smoke、Mini、Full、TEST；
- GPU/云端操作；
- 修改 B04 数值或历史 EXP-ID；
- commit、push、PR 或 merge（除非 Owner 后续明确授权）。

## Acceptance criteria for this documentation stage

- B04A 在 README、PROJECT_STATUS、总计划和 Backlog 中依赖一致；
- B07 明确为 `BLOCKED_BY_B04A`；
- B08/B09 不得越过 B04A/B07；
- TEST 被拆为候选与协议冻结后的一次性独立 Gate；
- B04 Protocol 与 R05 Results 有独立入口；
- `git diff --check` 和文档链接检查通过；
- 明确记录所有研究计算为 `NOT RUN`。

## Prohibited conclusions

- 文档就绪不等于候选实现、Smoke、Mini 或 Full 就绪；
- B04A 不能改写 B04 历史结果；
- Mini 不能证明最佳架构；
- SLP8 reference 不是人工像素级、医学、皮肤界面应力、硬件或产品 GT；
- 不得形成舒适性、整夜稳定性或气囊闭环结论。
