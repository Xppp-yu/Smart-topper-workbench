# S2 B11F SLP8 最终开发集拟合运行准备 v0.1

状态：`R01_FAILED_BEFORE_TRAINING_LOOP / AUTHORIZATION_CONSUMED / RUNTIME_FIX_REVIEW_REQUIRED / GPU_NOT_AUTHORIZED / TEST_DENIED`

TASK-ID：`TASK-SLP-B11F-FINAL-DEVELOPMENT-FIT-RUN-PREPARATION-v0.1`

## 交付

- 冻结 proposed EXP-ID：`EXP-SLP-B11F-PM-FINAL-FIT-20260903-AUTODL-R01`。
- R01 独立审查结论为 `ITERATE`；两项 P1 环境授权绑定与累计恢复预算已修复。
- 绑定已推送 runner SHA `9af268fa168207a269abbef22e522ac04fd6b6c5`。
- 绑定 config、candidate contract、B01 freeze manifest 与 A06 split hash。
- 冻结 RTX 4090、8,192 MiB peak、至少 1 GiB 可用空间和 runner 内部 2,700 秒
  EXP-level UTC deadline；恢复不会重置。
- AutoDL no-training preflight 已在 RTX 4090 上完成并审查 `ACCEPT`；未训练、TEST=0。
- Owner 于 `2026-09-03T18:40:09+08:00` 授权冻结三 seed final fit；R01 在首个 training
  batch 前因 class-weight NumPy→Torch 接口错误失败，该授权已消耗，不得用于恢复或新 run。

## 实际本地核验

```text
git push origin main
f166925..9af268f  main -> main

git fetch origin main
runner release == 9af268fa168207a269abbef22e522ac04fd6b6c5
ahead/behind = 0/0
worktree clean

Proposed EXP-ID output exists: false
B01 freeze manifest exists: true
B01 freeze manifest SHA-256:
42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04

Git/LF config SHA-256:
a6590d6f068644d98fa5340ec3d4a2e02171b529ec22ab092efb54a298925a43
Git/LF candidate SHA-256:
34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b

uv run python scripts/validate_slp8_b11f_final_fit_preparation.py \
  configs/experiments/slp8_pm_final_development_fit_v0.1.json
PASS; TEST=0; GPU_NOT_AUTHORIZED

uv run python scripts/run_slp8_region_final_fit.py --validate-only
PASS; TEST=0; GPU_NOT_AUTHORIZED

uv run python -m pytest tests/test_slp8_region_final_fit.py -q
40 passed

uv run python -m pytest tests/test_slp8_region_final_fit.py \
  tests/test_slp8_b11_candidate_freeze.py tests/test_slp8_region_full.py -q
123 passed

uv run python -m pytest tests/test_check_markdown_links.py -q
6 passed

git diff --check
PASS

three bash command blocks: syntax PASS (`bash -n`)

GPU/CUDA preflight: PASS（后续 AutoDL transcript）
GPU final fit: FAILED_BEFORE_FIRST_TRAINING_BATCH
TEST: 0
```

## AutoDL no-training preflight

```text
code transfer: complete-history local Git bundle
bundle main: 2ac18c8102600bc3504c1f5a9a15ada964990b7e
bundle SHA-256: 654bba6e6aa10063e3caa5ac6d6b9ae7810249330cc3b54439d29bb0f39b0b50
runner checkout: 9af268fa168207a269abbef22e522ac04fd6b6c5
GPU: NVIDIA GeForce RTX 4090
torch/cuda/cuDNN: 2.13.0+cu130 / 13.0 / 92000
environment fingerprint: a5a9342b18d00b614355e63ce056a7edd92dd80358d8aead5ef6e8e0ba045669
run_mode: cuda_determinism_unverified
validator: PASS
validate-only: PASS
environment-preflight: PASS
gpu_training_run: false
test_access: false
test_rows: 0
formal EXP output before/after: ABSENT
final exit: 0
review: ACCEPT; no P0/P1/P2
```

## Verified

- R05 implementation review 已 `ACCEPT`；P1 fix release 已推送。
- 首次运行要求 Owner-authorized environment fingerprint，并在 output 创建前核验；
  fingerprint 进入 run、seed、checkpoint 与 terminal identity。
- `budget.json` 固定首次 UTC start、deadline 与 core SHA；恢复、停机与重复 resume 都沿用
  同一 deadline，预算耗尽/篡改/时钟回退在数据加载前 fail closed。
- 本地 B01 freeze manifest 与 dataset root 存在；仅计算 freeze 文件 hash，未加载 TEST
  rows、labels、onehot、统计或预测。
- proposed EXP-ID 的本地正式输出目录不存在。
- AutoDL preflight 绑定的 config/candidate/B01 hash 全部一致；正式 EXP 输出前后均不存在。
- preflight 与正式入口使用同一 `_environment_record()` canonical fingerprint 算法。
- 配置继续保持 `execution_authorized=false`；Owner 授权仅允许冻结正式命令以显式
  `--run-authorized` 启动该精确 EXP-ID，不形成通用或 TEST 授权。

## Inferred

- 既有 B09 RTX 4090 unit 预算支持把 45 分钟作为三 seed final-fit 的 proposed total
  ceiling；真实耗时仍必须由 AutoDL 运行测量，不能由历史 unit 直接推定。

## Unverified

- 三 seed 真实训练的 wall time、peak CUDA、STOPPED/resume 和 DONE 路径。
- 三个 `final.pt` 的实际 SHA、独立重载与集成推理。
- B09T TEST 性能。

## Limitations

- AutoDL 无法在线 fetch GitHub；代码通过完整历史 Git bundle 传输并以 SHA-256 绑定，
  `origin/main` 在 AutoDL 指向该 bundle。此偏差已在 preflight review 中披露并接受。
- `run_mode=cuda_determinism_unverified` 如实表示尚未由真实训练验证端到端 CUDA 重现性。
- Windows CRLF candidate hash 只作诊断，不能替代 Linux Git/LF runtime hash。
- 45 分钟已由 config 与 runner 内部固定为 2,700 秒 EXP wall budget；UTC 前跳会保守
  缩短预算，外部 timeout 仍提供硬停止补充。
- final fit 不保留验证集，training loss 不是 validation/test performance。
- GT 仍是 danaLab/uncover、自动接受且 `NOT_REVIEWED` 的 pressure-only 参考标签；不构成
  cover、自研硬件、产品、舒适性、医疗、整夜或气囊控制验证。

## Next Gate

`B11F_RUNTIME_FIX_REVIEW / GPU_NOT_AUTHORIZED / TEST_DENIED`
