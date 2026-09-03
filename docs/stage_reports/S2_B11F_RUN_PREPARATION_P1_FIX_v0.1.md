# S2 B11F 运行准备 P1 修复 v0.1

状态：`FIXES_COMPLETE / READY_FOR_RELEASE_BINDING / GPU_NOT_AUTHORIZED / TEST_DENIED`

TASK-ID：`TASK-SLP-B11F-RUN-PREPARATION-P1-FIX-v0.1`

## 修复内容

- 新增 no-training environment preflight payload，输出环境对象及 canonical SHA-256。
- 正式/恢复入口必须显式传入 Owner-authorized environment fingerprint；首次启动在创建
  output directory 前核验，恢复时同时由 identity 和原 `environment.json` 复核。
- config 新增并 fail-closed 固定 `max_total_wall_seconds=2700`。
- 新增原子 `budget.json`：首次启动冻结 UTC start 与不可变 deadline；elapsed/remaining
  从同一 deadline 重算，停机间隔保守计入，恢复不重置预算。
- 每 batch、epoch、completion 和 DONE 前检查预算；STOPPED/FAILED/DONE 携带 budget
  core/file SHA 与摘要；seed checkpoint identity 绑定 budget core。
- 新增缺失/错误环境授权、授权漂移、budget core 篡改、deadline 重置、预算耗尽、时钟
  回退及 max-wall 配置漂移反例。

## 实际验证

```text
uv run python -m pytest tests/test_slp8_region_final_fit.py -q
40 passed

uv run python -m pytest tests/test_slp8_region_final_fit.py \
  tests/test_slp8_b11_candidate_freeze.py tests/test_slp8_region_full.py -q
123 passed

uv run python scripts/validate_slp8_b11f_final_fit_preparation.py \
  configs/experiments/slp8_pm_final_development_fit_v0.1.json
PASS; TEST=0; GPU_NOT_AUTHORIZED

uv run python scripts/run_slp8_region_final_fit.py --validate-only
PASS; TEST=0; GPU_NOT_AUTHORIZED

py_compile: PASS
git diff --check: PASS
GPU/CUDA preflight: NOT RUN
GPU final fit: NOT RUN
TEST: 0
```

## Verified

- 环境 fingerprint 缺失或不匹配时，在 output directory 创建和数据加载之前拒绝。
- resume authorization 环境漂移、budget core/deadline 篡改、过期与时钟回退均在数据
  加载前拒绝。
- 两次 STOPPED→resume 的测试保持相同 start、deadline 和 budget core，elapsed 从首次
  启动累计；没有重新获得 2,700 秒。
- 原有 deterministic/RNG/checkpoint/terminal/TEST=0 回归保持通过。

## Inferred

- 固定 UTC deadline 是保守的 EXP wall budget：中断后的停机时间也消耗预算，从而关闭
  通过重启刷新 timeout 的路径。

## Unverified

- AutoDL no-training environment fingerprint 与 Owner authorization 记录的实际生成和复核。
- RTX 4090 上每 batch budget 更新、45 分钟 SIGINT/SIGKILL 边界和跨进程恢复。
- 三个正式 checkpoint、DONE bundle 和 B09T TEST 性能。

## Limitations

- UTC 时钟前跳会保守缩短预算；时钟回退直接 fail closed。
- budget 在 batch 边界内部检查，外部 `timeout` 仍作为硬停止补充。
- 本轮 synthetic CPU 测试不构成 GPU、性能、产品、硬件或安全证据。

## Next Gate

形成新的 clean/pushed runner release SHA，回填运行准备合同后执行独立只读复审；当前
保持 `B11F_RUN_PREPARATION_P1_FIX / GPU_NOT_AUTHORIZED / TEST_DENIED`。
