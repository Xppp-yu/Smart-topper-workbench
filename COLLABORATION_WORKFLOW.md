# Smart Topper 多 Agent 协作制度 v0.2

状态：`ACTIVE`

适用范围：PoPu、SLP、PressurePose、自采数据及其后续工程化任务。

## 1. 核心原则

> Owner 负责方向和最终授权；网页 GPT 负责战略讨论、任务设计和 GitHub 层二审；Claude Code 负责本地实现；Codex 负责本地真实状态、任务控制和阶段验收；Experiment Runner 负责冻结实验；GitHub 保存已提交、已推送的共享基线。

任何角色都不能把自己看不到的状态当成已验证事实。尤其要始终区分：

- **GitHub 状态**：最后一次已经 commit + push、协作者都能读取的共享基线；
- **本地状态**：GitHub 基线，加上未提交修改、untracked 文件、ignored outputs、raw data、运行中的任务和实验。

因此：`GitHub is the shared baseline, not the local latest state.`

## 2. 角色与权限边界

| 角色 | 主要职责 | 不应承担或不得自行决定 |
|---|---|---|
| Owner | 决定方向、优先级、预算、是否运行 Mini/Full、是否合并，以及最终 `ACCEPT / ITERATE / STOP` | 不需要亲自管理全部代码细节 |
| 网页 GPT | 读取 GitHub 基线；讨论研究路线；起草 `TASK-ID`；审查已推送的 commit、diff、协议和脱敏证据；提供第二视角 | 不得声称掌握本地 dirty worktree、ignored outputs、raw data 或正在运行的任务；不得把 GitHub 二审当成本地验收 |
| Claude Code | 按单个 `TASK-ID` 在指定工作树中实现、调试、测试、Smoke 和交付 | 不扩展任务范围；不自行宣布研究成功；不擅自运行 Mini/Full；未获明确授权不 commit/push |
| Codex | 读取本地真实仓库、未提交代码、原始数据、outputs 和运行状态；核对任务冲突；复跑测试；检查证据；阶段验收；必要时做定点修复；形成正式提交 | 不把路线讨论当作实验结果；不因单元测试通过而批准 Full；不替 Owner 作最终产品授权 |
| Experiment Runner | 按冻结的 Git SHA、resolved config、data/split manifest 和 `EXP-ID` 执行 Mini/Full，完整保存状态和产物 | 不边跑边改代码、参数或数据；不从运行成功直接推导研究结论 |
| GitHub | 保存已确认的代码、协议、任务、测试和脱敏证据，作为团队交接中枢 | 不是 raw data 仓库，也不是本地实时状态的完整镜像 |
| 人工/第二 Reviewer | 对 Region Reference、关键标签、产品结论和发布结论做最终复核 | 不能被自动指标或单一 Agent 完全替代 |

## 3. 本地状态快照

网页 GPT 起草新任务前，应获得以下最小快照：

```text
Branch:
HEAD:
Dirty:
Untracked:
Active TASK:
Running jobs:
Relevant outputs:
Ahead/behind GitHub:
```

仓库提供以下命令生成快照；无法自动判断的字段必须显式填写或保留 `UNSET`，不得猜测：

```powershell
uv run python scripts\project_status_snapshot.py `
  --active-task TASK-SLP-A02-CONTENT-QA-v0.1 `
  --running-jobs "none" `
  --relevant-output "outputs/reports/slp_content_qa_v0.1.json"
```

快照只用于交接，不代表任务已验收。

## 4. 标准开发流程

```text
Owner 提出目标并决定优先级
  -> 网页 GPT 读取 GitHub，讨论方案并起草 TASK-ID
  -> Codex 核对本地状态、证据、冲突和任务边界
  -> Claude Code 本地实现、测试、Smoke、handoff
  -> Codex 读取本地代码/raw data/ignored outputs，复核并给出阶段结论
  -> 形成边界清楚的 commit，并 push GitHub
  -> 网页 GPT 对已推送基线做 Second Review
  -> Owner 决定 ACCEPT / ITERATE / STOP
```

### TASK-ID 最低合同

每张任务单至少包含：

- 目标与明确非目标；
- 允许修改的目录和禁止触碰的文件；
- 输入、输出、数据子集和真值来源；
- 必须新增或修改的模块；
- Unit、错误路径和 Smoke 验证；
- 禁止运行的 Mini/Full 命令；
- 交付文件、已知限制和 Reviewer checklist。

Claude Code 的 handoff 必须报告：`TASK-ID`、修改文件、实际运行命令、测试结果、生成产物、已知失败、禁止结论、Git 状态。没有真实运行的步骤必须写成 `NOT RUN`。

## 5. 冻结实验流程

正式 Mini/Full 只能在代码验收后进入 Runner：

```text
代码和协议验收
  -> 冻结 Git SHA + resolved config + data manifest + split manifest + EXP-ID
  -> Owner 明确授权预算与运行范围
  -> Runner 执行并写 status / logs / metrics / predictions / DONE 或 FAILED
  -> Codex 复核本地真实证据
  -> 网页 GPT 做 GitHub 层研究二审
  -> Owner 最终授权
```

同一 `EXP-ID` 进入 `QUEUED` 后不得修改 SHA、配置、manifest 或 split；任何变化都必须创建新 `EXP-ID`。Smoke/Mini 通过不等于 Full 成功，运行成功也不等于研究结论成立。

## 6. 并行开发规则

PoPu 和 SLP 可以并行，但必须满足以下任一条件：

1. 使用不同 branch + 不同 worktree；或
2. 使用不同 `TASK-ID`，并在任务单中声明互不重叠的文件边界。

推荐：

```text
worktree/popu  -> PoPu TASK-ID
worktree/slp   -> SLP TASK-ID
```

禁止两个 Coding Agent 在同一工作树中同时修改相同文件。发现未归属的 dirty/untracked 文件时，先保留并查明所有者，不得顺手覆盖、暂存或提交。

## 7. Git 与 GitHub 规则

- 私密 GitHub 只承载代码、通用配置、测试、协议、任务和脱敏证据；
- 不提交凭据、`configs/paths.local.json`、raw RGB/IR/depth/pressure、数据压缩包、大模型 checkpoint 或敏感样本；
- 每次只暂存当前 `TASK-ID` 已确认的精确路径，禁止 `git add .`、`git add -A`；
- 本地 Review 在 commit/push 前完成；网页 GPT 的 Second Review 只针对已推送的 GitHub 基线；
- 每个大阶段使用独立提交；提交信息应包含任务或阶段语义；
- GitHub 上看不到的 outputs，应以摘要、manifest、hash、指标和必要脱敏图表交接，不伪装成仓库内证据。

## 8. SLP 当前执行顺序

SLP 路线分为两条（见 Backlog）：

```text
# 路线 A：SLP8 pressure-only GT（A09R 建立）
A09R  SLP8 GT 合同校准（当前项目参考 GT）
  -> B01  训练表冻结（DONE_WITH_LIMITATIONS）
  -> B02  非学习基线（DONE_WITH_LIMITATIONS）
  -> B03  神经网络 Smoke（DONE_WITH_LIMITATIONS）
  -> B04  TinyFCN / SmallUNet Mini（DONE_WITH_LIMITATIONS）
  -> B04A 受控架构扩展 Mini（先协议与实现，运行需另行授权）
  -> B07  Full 协议冻结（BLOCKED_BY_B04A）

# 路线 B：OpenCV/人工复核（历史路线，已改为 HOLD）
A09 Region Schema Review  [SUPERSEDED_BY_A09R]
  -> A10 R0 Geometry Seed  [HOLD]
  -> A11 OpenCV Refinement [HOLD]
  -> A12 R1 Pseudo-label Export  [HOLD]
  -> A13-A17 Human Review and Region Reference Freeze  [HOLD]
  -> B01+ Frozen R2/R3 Dataset Training and Evaluation  [HOLD]
```

真值边界（A09R Owner 决策，2026-08-24；B01 数据合同补充，2026-08-24）：

- **SLP 8-region pressure-only GT**（`SLP_8Region_Pressure_VAL_v1.1`，4,590 samples，102 danaLab，uncover only）是当前 `PROJECT_ACCEPTED_REFERENCE_GT`，用于 SLP8 pressure-only 区域分割训练和评估。provenance=`V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`，`source_review_status=NOT_REVIEWED`。**不是人工像素级语义 mask；不是医学/皮肤界面应力/产品 GT**；不得改写 provenance/review 字段；不得称为 kPa。
- **B01（`TASK-SLP-B01-SLP8-TRAINING-TABLE-FREEZE-v0.1`）** 基于 A09R 合同和 A06 subject split 冻结了 pressure-only 8-region 训练/验证/测试数据入口（`slp8_training_tables_v0.1`，3645/450/495，subject overlap = 0）。B01 模块 `src/topper_perception/io/slp8_training_table_freeze.py` 提供 TEST 防泄漏合同：开发态下 `compute_class_stats(ml_split="test")` 直接抛 `TestLeakageError`；只有 `enable_test_access(purpose="final_evaluation")` 显式开启才允许读取 TEST label/onehot 或计算 TEST 类别统计；任何其它 purpose 名称均被拒绝。 数据卡与顶层 freeze manifest 均记录该合同。
- RGB/IR 14 关节点是 `J0`（原始人工关节真值）；homography 映射是派生 `J1`。
- 10-region polygon 路线（`slp_region_annotation_v0.1`，R0–R3 tiers）是历史内部治理合同，**不是当前训练合同**，不得与 8-region 数据混用。
- 仅 danaLab/uncover，不得外推到产品、硬件、舒适性、医疗或气囊控制。

## 9. 状态与结论口径

- 只有代码、任务书、配置模板或测试：最多 `READY_TO_RUN`；
- Smoke 通过：只能证明最小执行链路；
- 真实数据运行且产物完成：可进入 Reviewer 验收；
- 缺少真值或配对：审计任务可标记完成，但训练路线保持 `HOLD/BLOCKED`；
- 公开数据结果不外推为自研硬件、舒适性、医疗效果、整夜稳定性或气囊闭环验证。

若本文件与单次聊天摘要冲突，以已提交的本文件、对应 `TASK-ID` 和冻结实验协议为仓库治理基线；Owner 的新决定需通过后续版本化修改进入仓库。

实时进度只在 `docs/PROJECT_STATUS.md` 维护；本文件只保存稳定治理规则与依赖顺序。路线文档和 Backlog 不应复制易漂移的“一句话当前状态”。
