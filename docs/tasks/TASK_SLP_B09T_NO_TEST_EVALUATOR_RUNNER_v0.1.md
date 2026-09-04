# TASK-SLP-B09T-NO-TEST-EVALUATOR-RUNNER-v0.1

状态：`IMPLEMENTED / READY_FOR_INDEPENDENT_REVIEW / TEST_DENIED / EXECUTION_NOT_AUTHORIZED`

## Purpose

在已接受的 B09T 协议上实现纯 hard-prediction evaluator、无真实 TEST 路径的 CLI 和
synthetic smoke。验证 plurality/tie-break、固定指标与输出 wiring，但不加载 B01 TEST，
不加载真实 checkpoint，不运行 GPU。

## Allowed changes

- `src/topper_perception/evaluation/slp8_b09t_evaluator.py`
- `scripts/run_slp8_b09t_evaluator.py`
- `tests/test_slp8_b09t_evaluator.py`
- 本任务、阶段报告、项目状态与 SLP backlog。

允许提交并推送以上精确文件到独立 `codex/` review branch；复审 ACCEPT 前不得合并。

## Required behavior

- seed 顺序严格为 42/123/2026。
- 2/3 majority；三者全异固定 seed 42 tie-break，并报告触发数与比例。
- unanimous secondary 对任何分歧输出 `UNKNOWN_REGION=-1`。
- 复用 fixed foreground classes 1..8、空类计 0 的指标实现。
- CLI 只有 `--validate-only` 和 `--synthetic-smoke`；不存在正式执行模式。
- synthetic 只接受 `SMOKE-B09T-*`，拒绝既有输出且不覆盖。
- 源码不得引用 TEST opt-in/B01 loader/CUDA/formal run flag。

## Prohibited

- 读取 TEST rows/labels/onehot、生成 TEST predictions/metrics。
- 调用 `enable_test_access` 或设置 `load_test=True`。
- 加载真实 B11F checkpoint 或启动 GPU。
- 把 synthetic 指标当研究结果。

下一 Gate：

`B09T_NO_TEST_EVALUATOR_RUNNER_INDEPENDENT_REVIEW / TEST_DENIED / EXECUTION_NOT_AUTHORIZED`
