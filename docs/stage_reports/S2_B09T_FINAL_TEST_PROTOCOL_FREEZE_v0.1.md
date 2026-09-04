# S2 B09T Final TEST Protocol Freeze v0.1

TASK-ID：`TASK-SLP-B09T-FINAL-TEST-PROTOCOL-FREEZE-v0.1`

状态：`READY_FOR_INDEPENDENT_REVIEW / TEST_DENIED / EXECUTION_NOT_AUTHORIZED`

## Outcome

B09T 的一次性评价口径已冻结为机器可验证配置。该配置绑定三个已审计 B11F checkpoint，
固定 plurality primary（2/3 取多数，三者全异固定取 seed 42 并审计比例）、foreground 1..8 空类计零的 pooled macro IoU，以及完整 secondary
readout。协议禁止根据 TEST 做任何候选、阈值、参数或指标适配。

## R02 P1 remediation

PR #29 首轮独立审查为 `ITERATE`：原 validator 只检查部分字段的格式或长度，九类合法格式
漂移可穿透；原 majority 合同也未定义三模型全异像素。本轮修复后：

- canonical protocol content SHA-256 固定为
  `479d311cfa549d5851f9fc16bee28f1d16a4c2fe5796b6d59b68e39be1bc2690`；任一未声明字段、
  列表删减或值替换均 fail closed。
- runner、B01、candidate、B11F EXP-ID 和三个 checkpoint 继续逐项给出可读错误。
- 2/3 相同时取 majority；三者全异时固定取 seed 42 hard prediction，并强制报告该分支
  像素数和比例。该 tie-break 在 TEST 前声明，不允许按结果更换。

## Verification

```text
uv run python -m pytest tests/test_slp8_b09t_protocol.py -q
21 passed

uv run python -m pytest tests/test_slp8_b09t_protocol.py tests/test_check_markdown_links.py -q
27 passed

uv run python scripts/validate_slp8_b09t_protocol.py configs/experiments/slp8_pm_b09t_final_test_protocol_v0.1.json
B09T_PROTOCOL_VALIDATION_PASSED TEST_DENIED EXECUTION_NOT_AUTHORIZED

uv run python -m py_compile scripts/validate_slp8_b09t_protocol.py
PASS

git diff --check
PASS
```

## Boundaries

### Verified

- 协议字段与 fail-closed validator 已实现。
- TEST 与 execution authorization 在配置中均为 strict false。
- validator 源码不调用 `enable_test_access(...)`，也不设置 `load_test=True`。
- validator 对整份 canonical protocol identity 做 SHA-256 固定，并逐项锁定 runner、B01、
  candidate、EXP-ID、checkpoint 与三模型投票合同。

### Inferred

- 复用 B09 fixed-foreground pooled confusion 语义，可以保持开发期与最终评价的指标可比性；
  仍需后续 evaluator 实现审查确认实际调用路径一致。

### Unverified

- 真实 checkpoint inference、TEST 加载、495 样本预测、指标和运行资源全部 `NOT RUN`。
- evaluator、runner、环境 preflight 和正式授权包尚未实现。

### Limitations

- 本阶段只有协议，不构成 TEST 授权或最终性能结果。
- GT 仍为 pressure-only、danaLab/uncover、`source_review_status=NOT_REVIEWED`。
- 不得外推到 covered、产品、硬件、舒适性、医疗、整夜或气囊控制。

## Next Gate

`B09T_PROTOCOL_INDEPENDENT_REVIEW / TEST_DENIED / EXECUTION_NOT_AUTHORIZED`
