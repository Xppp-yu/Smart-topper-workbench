# TASK-SLP-B09T-FINAL-TEST-PROTOCOL-FREEZE-v0.1

状态：`READY_FOR_INDEPENDENT_REVIEW / TEST_DENIED / EXECUTION_NOT_AUTHORIZED`

## 1. Purpose

在不读取 TEST 的前提下冻结 B09T 唯一一次最终评价合同：输入 checkpoint、TEST 结构计数、
ensemble 规则、指标、失败终态、证据输出和 anti-adaptation 规则。本任务不是 TEST 授权。

## 2. Allowed changes

- `configs/experiments/slp8_pm_b09t_final_test_protocol_v0.1.json`
- `scripts/validate_slp8_b09t_protocol.py`
- `tests/test_slp8_b09t_protocol.py`
- 本任务、对应阶段报告、`docs/PROJECT_STATUS.md`、`docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`

允许把以上精确文件提交并推送到独立 `codex/` review branch；独立审查通过前不得合并
`main`，也不得开始读取 TEST。

## 3. Frozen decisions

- 输入仅为已审计 B11F R02 的 seeds 42/123/2026 final checkpoint。
- primary prediction 是三 hard mask 的逐像素 plurality vote：有 2/3 majority 时取多数类；
  三者全异时预先固定取 seed 42 hard prediction，并报告该分支的像素数与比例。该规则只补全
  原 majority 合同的未定义分支，不允许根据 TEST 改 tie-break。
- primary metric 是 pooled fixed foreground（classes 1..8）macro IoU；空类固定计 0。
- 同时报 pooled Dice、background、逐 region、逐 subject、worst subject 和 pixel accuracy。
- 3/3 unanimous reject 仅作 secondary research readout；不是概率、OOD 或安全机制。
- 只允许 B01 冻结 TEST：11 subjects / 495 samples；必须保持 subject isolation。
- TEST 结果不得用于调参、换模型、换阈值、换指标或常规重跑。

## 4. Authorization and execution boundary

本阶段 `test_authorized=false`、`load_test=false`、`execution_authorized=false`。validator
不得导入或调用 TEST opt-in API。后续 evaluator 实现也只能先做 synthetic/no-TEST 测试。

真正执行前必须另立授权包，由 Owner 精确冻结 EXP-ID、runner SHA、protocol SHA、三个
checkpoint SHA、B01 SHA、environment fingerprint、预算和命令，并明确一次性
`purpose="final_evaluation"` 授权。失败或协议缺陷必须新任务并做 TEST pollution review。

## 5. Acceptance

- validator 正例通过、关键字段漂移反例 fail closed。
- Markdown links、py_compile、`git diff --check` 通过。
- TEST rows/labels/onehot/predictions/metrics 均为 `NOT READ / NOT RUN`。

下一 Gate：

`B09T_PROTOCOL_INDEPENDENT_REVIEW / TEST_DENIED / EXECUTION_NOT_AUTHORIZED`
