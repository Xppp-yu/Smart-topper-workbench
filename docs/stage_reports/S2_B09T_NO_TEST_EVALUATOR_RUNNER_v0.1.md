# S2 B09T No-TEST Evaluator/Runner v0.1

TASK-ID：`TASK-SLP-B09T-NO-TEST-EVALUATOR-RUNNER-v0.1`

状态：`READY_FOR_INDEPENDENT_REVIEW / TEST_DENIED / EXECUTION_NOT_AUTHORIZED`

## Outcome

已实现纯 NumPy hard-prediction ensemble/evaluation、无正式执行入口的 CLI 与 synthetic
wiring smoke。三模型全异分支、unanimous reject、pooled fixed-foreground metrics、逐
region/subject 和 fail-closed CLI 均有直接回归。

## Verification

```text
uv run python -m pytest tests/test_slp8_b09t_evaluator.py tests/test_slp8_b09t_protocol.py tests/test_check_markdown_links.py -q
36 passed

uv run python scripts/run_slp8_b09t_evaluator.py --validate-only
B09T_EVALUATOR_VALIDATION_PASSED TEST=0 GPU_NOT_RUN EXECUTION_NOT_AUTHORIZED

uv run python scripts/run_slp8_b09t_evaluator.py --synthetic-smoke --experiment-id SMOKE-B09T-20260904-R01 --output-dir outputs/analysis/b09t_synthetic_smoke_r01
B09T_SYNTHETIC_SMOKE_PASSED TEST=0 GPU_NOT_RUN EXECUTION_NOT_AUTHORIZED

uv run python scripts/validate_slp8_b09t_protocol.py configs/experiments/slp8_pm_b09t_final_test_protocol_v0.1.json
B09T_PROTOCOL_VALIDATION_PASSED TEST_DENIED EXECUTION_NOT_AUTHORIZED

uv run python -m py_compile src/topper_perception/evaluation/slp8_b09t_evaluator.py scripts/run_slp8_b09t_evaluator.py
PASS

git diff --check
PASS
```

Synthetic output is ignored and contains exactly one `synthetic_summary.json`: 2 synthetic samples,
32,256 pixels, one deliberately exercised three-way-disagreement pixel, `test_access=false`,
`test_rows=0`, `gpu_run=false`. Its metric values are wiring evidence only.

## Boundaries

### Verified

- no-TEST evaluator/runner 的投票、指标和 synthetic 输出 wiring。
- CLI 不存在 real run mode，并拒绝正式 EXP-ID、既有输出和 validate-only 输出参数。
- 源码不引用 B01 loader、TEST opt-in、CUDA 或正式授权 flag。

### Inferred

- 纯数组 evaluator 与后续真实 inference 输出之间接口明确；真实 runner 仍须独立实现和验证。

### Unverified

- B01 TEST、真实 checkpoint、CUDA、正式 TEST metrics 全部 `NOT RUN`。

### Limitations

- synthetic 数值仅证明 wiring，不构成模型性能证据。
- 本任务不授权 TEST，不创建正式 EXP-ID。

## Next Gate

`B09T_NO_TEST_EVALUATOR_RUNNER_INDEPENDENT_REVIEW / TEST_DENIED / EXECUTION_NOT_AUTHORIZED`
