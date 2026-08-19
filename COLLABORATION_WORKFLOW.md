# Codex × Claude Code 协作约定

## 1. 目的

本项目采用“Codex 负责研究与验收闭环，Claude Code 负责较大实现，Experiment Runner 负责后台计算”的协作方式。目标是提高开发速度，同时避免代码一次性生成后缺少方向、证据、复用边界和阶段记录，也避免 Agent 陪跑全量实验后自行给出研究结论。

## 2. 角色分工（四层）

当前流程从“一个 Agent 负责设计、开发、全量计算、自审和总结”调整为四层职责。完整依据见 [docs/EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md](docs/EXPERIMENT_GOVERNANCE_AND_GPU_EXECUTION_PLAN_v0.1.md)。

### Controller（Codex）：定义与验收

- 定义研究问题、变量、评价协议、Gate 和单张任务单（`TASK-ID`）；
- 判断当前数据、证据、缺口和阶段位置；
- 给 Coding Agent 编写明确任务书，包括输入、输出、禁止范围和验收标准；
- 审阅提交范围、代码结构、配置、测试和生成产物；
- 更新阶段报告与项目状态，验收后完成 Git 提交；
- 明确区分 `COMPLETE`、`READY_TO_RUN`、`HOLD` 和 `BLOCKED`。

### Coding Agent（Claude Code）：实现与冒烟

- 读代码、实现、调试、单元测试和小数据 Smoke Test；
- 按任务书编写新模块、脚本、配置和测试；
- 承担较大功能修改、重构以及新研究路线的代码实现；
- 保持模块可复用，避免把逻辑全部写成一次性脚本；
- 报告修改文件、测试结果、尚未运行的命令和已知限制；
- **通过 Smoke Test 后即停止，不陪跑 Mini/Full 实验，不自行给出最终研究结论。**

### Experiment Runner（本地计算机或租用服务器）：后台计算

- 基于冻结 Git SHA 与 resolved config 后台执行 Mini/Full Run；
- 保存日志、指标、预测、图和 checkpoint；
- 训练过程与 Agent 会话解耦，不让 Claude Code 等待；
- 记录 `manifest.json`、`status.json` 与 `DONE.json` / `FAILED.json` 产物。

### Reviewer（Codex）：只读复核

- 只读复核配置、数据版本、split、指标、关键图和失败样本；
- 给出 `ACCEPT / ITERATE / STOP / INVALID`；
- 下一轮修改必须由 Reviewer 形成新的 `EXP-ID` 和任务单。

> Codex 同时担任 Controller 和 Reviewer，属于流程内独立复核，不等同于论文级盲审；关键产品或发布结论仍应增加人工/第二审阅者。

## 3. TASK-ID 与 EXP-ID

### TASK-ID：开发任务

示例：`TASK-P5.2-A-CNN-SCAFFOLD-v0.1`。任务单必须包含：目标与非目标、允许修改目录、输入合同、数据子集、配置、需新增/修改模块、Unit/错误/Smoke Test、禁止执行的 Mini/Full 命令、交付文件、Git commit 和已知限制。

### EXP-ID：不可变计算实验

建议格式 `EXP-P5.2-<MODEL>-<SCOPE>-<YYYYMMDD>-RNN`。同一 `EXP-ID` 的 Git SHA、resolved config、数据 Manifest 和 split Manifest 一旦进入 `QUEUED` 不再修改；参数变化必须创建新 `EXP-ID`。

## 4. 标准协作流程

1. 用户提出目标或研究问题。
2. Codex（Controller）检查当前仓库、数据证据和项目看板，确定任务位置并签发 `TASK-ID` 任务书：
   - 目标与非目标；
   - 输入数据与配置；
   - 要修改或新增的模块；
   - 最低测试与输出；
   - 禁止推进的后续阶段；
   - 验收条件。
3. Claude Code（Coding Agent）完成较大代码实现、单元测试与小数据 Smoke，并提交交付清单。
4. Experiment Runner 基于冻结 SHA 与配置后台执行 Mini/Full Run。
5. Codex（Reviewer）审查 diff、复跑测试、复核配置/指标/图/失败样本，给出 `ACCEPT / ITERATE / STOP / INVALID`。
6. 如果只有定位明确的小问题，由 Codex 定点修复并补回归测试；较大问题重新签发任务单交回 Claude Code。
7. Codex 写阶段报告、更新 `docs/PROJECT_STATUS.md`，确认边界后完成 Git 提交。

## 5. 每次交付必须包含

- 修改和新增文件清单；
- Git 提交 ID 或明确说明尚未提交；
- 已运行的测试命令和结果；
- 尚未运行的全量命令；
- 真实输出路径；
- 已知限制和不能得出的结论；
- 工作区是否干净。

只有代码、空目录、配置模板或单元测试时，阶段最多标为 `READY_TO_RUN`。只有真实数据运行、产物检查和阶段报告都完成后，才可标为 `COMPLETE`。得到“无法使用”或“缺少真值”也可以完成审计任务，但相应训练路线必须保持 `HOLD/BLOCKED`。Smoke/Mini 通过不等于 Full 结论。

## 6. 数据与版本边界

- 外部原始数据、`outputs/` 生成物和 `configs/paths.local.json` 不进入 Git；
- 代码、通用配置、测试和阶段报告进入 Git；
- 不静默填补缺失值、伪造标签或强行建立样本配对；
- 公开数据结果只支持研究链路和算法候选，不外推为自研硬件或产品验证；
- 每个大阶段尽量使用独立提交，保证可回退、可比较和可交接。

## 7. 当前交接点

截至 2026-08-19：P0–P5.2-A 已完成。P5.2-A 的 CPU 与 RTX 4090 CUDA Smoke 均已通过，训练、checkpoint/resume/reload 和固定 seed 最小复现链路成立；Smoke 指标不用于模型排名。P5.1 的 `calibrated_linear_svm` 继续保留为**传统模型候选**（record macro-F1 0.9452），不是已经覆盖 CNN 的总体最优模型。P5.2-B Mini 筛选的协议、代码、版本化配置与测试已实现（`runner_type=popu_neural_mini`，冻结配置 `configs/experiments/popu_neural_mini_v0.1.json`，协议报告 `docs/stage_reports/P5_2_B_POPU_NEURAL_MINI_PROTOCOL_v0.1.md`）；Reviewer 首轮复核返回 `REVIEW_NEEDS_FIX`，已按 6 项要求新增修复提交，并追加数据 Manifest SHA-256 哈希校验（读取前校验、写入 manifest.json，不重写历史）；本任务**未运行**真实 Mini/Full。下一交接点为需 Reviewer 复核该修复提交、配置冻结并 Controller 授权后，才由 Experiment Runner 在 AutoDL 执行 P5.2-B Mini Run（P2 质量 manifest 不在 Git bundle 中，AutoDL 部署须单独上传并先校验 SHA-256）。P5.2-C 完成总体候选选择并经 Reviewer 接受后才放行 P6 `UNKNOWN/REJECT`。P4b 人体区域监督因 PoPu 标注无法唯一配对而继续 HOLD。
