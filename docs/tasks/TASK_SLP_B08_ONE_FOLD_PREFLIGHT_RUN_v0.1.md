# TASK-SLP-B08-ONE-FOLD-PREFLIGHT-RUN-v0.1

状态：`R01_FAILED / R02_FAILED / R03_PREFLIGHT_PASSED / B08_ACCEPTED / TEST_DENIED`

## R01 现场结果（2026-08-31）

- Owner 已明确授权 R01 在 AutoDL RTX 4090 上执行一个
  `slp8_resunet_lite_v0.1 / fold_1 / seed 42` unit；30-unit Full 与 TEST 未授权。
- clean runner SHA、protocol/fold/freeze hashes、Python 3.12、CUDA 与 RTX 4090
  环境检查通过；validate-only 得到 `units=30, TEST=0`。
- R01 在第一个 epoch 前失败，未进入 GPU 训练。根因是 real B01 loader 返回
  `RegionSample`，而 real-path synthetic guard 错误调用 `r.get(...)`，触发
  `AttributeError`。
- R01 runner 的异常发生在 `train_one_unit()` 返回之前，外层没有兜底捕获，因而
  未按合同写根级 `FAILED.json`。R01 output 必须原样保留，禁止覆盖或沿用该 EXP-ID。
- 本地 Round 7 已修复类型合同与异常 terminal 兜底；R02 runner 已冻结并执行，
  结果见下节。

## R02 现场结果（2026-08-31）

- R02 使用 clean `e0ba25a9aa0b33be971327ca398d822f7c7d1c8a`；bundle、
  protocol/fold hashes、CUDA 和 validate-only (`units=30, TEST=0`) 均通过。
- 在首个 TRAIN batch 的 loss forward 失败；PyTorch `2.13.0+cu130` 的 CUDA
  NLLLoss2d kernel 在 strict deterministic 模式下无实现。没有 optimizer step。
- R02 写出唯一根级 `FAILED.json` 与一致 manifest，证明 Round 7 terminal 修复
  有效；但随后 stack 空 OOF，载体 error 被二次 `ValueError` 覆盖。现场 traceback
  保留第一根因。R02 output 必须原样保留，禁止覆盖或沿用 EXP-ID。
- Round 8 本地改用 strict deterministic-compatible 的数学等价 weighted mean
  cross entropy；FAILED 不写 OOF，DONE 强制精确 OOF coverage。

## 目标

在 B08 runner 已接受后，仅执行一个真实 development fold/candidate/seed，
测量 wall time、peak CUDA memory，并验证 best-checkpoint reload 与完整 identity
carriers。该任务不是 B09 Full，也不产生候选排名。

## R03 最终结果（2026-09-01）

- Owner 明确授权 R03 单 unit；clean runner
  `02fb364902736a64ee8708440f0dd0bdddf860bc`，RTX 4090，Torch
  `2.13.0+cu130`，strict deterministic random-tensor model/AdamW probe 通过。
- 唯一终态 `DONE.json`；与 `preflight_manifest.json` 完全一致；状态
  `PREFLIGHT_PASSED`。
- TRAIN 3,240 samples / 72 subjects；VAL 855 samples / 19 subjects；fold_1、
  seed 42、ResUNet-lite；best epoch 22。
- wall `155.32885087199975 s <= 900 s`；peak CUDA `368.764416 MiB <= 8192 MiB`。
- best checkpoint SHA-256：
  `51db02cf6e26a11cef27b6627437cedaafe269357ca214d4662b7941a57c2506`；
  checkpoint reload prediction hash 一致。
- OOF 855 unique samples / 19 subjects，predictions/targets coverage 完整；TEST=0。
- R01/R02/R03 archive 已下载并在 Windows 本地只读复核；archive SHA-256：
  `55aa5a4c708a1ba9b2f9ee89a8d05d937e61f4edcd02bdf83b1f5600b478a298`；
  `LOCAL_ARCHIVE_AUDIT_PASSED`。archive/checkpoint/OOF 不提交 Git。

## 冻结运行身份

- historical failed EXP-ID：`EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R01`
- failed runner Git SHA：`5af426039ae41209af7929bc9319a0657e5f92b4`
- proposed retry EXP-ID：`EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R02`
- retry runner Git SHA：`e0ba25a9aa0b33be971327ca398d822f7c7d1c8a`
- proposed R03 EXP-ID：`EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R03`
- R03 runner Git SHA：`02fb364902736a64ee8708440f0dd0bdddf860bc`
- candidate：`slp8_resunet_lite_v0.1`（两候选中参数量较大）
- fold：`fold_1`（19 VAL subjects / 855 VAL samples，最大 fold）
- seed：`42`
- epochs：`30`（B07 frozen；不得缩短或调参）
- expected device：RTX 4090 CUDA；serial one-unit only
- budget：`<=15 min`、`<=8192 MiB peak CUDA`
- TEST：`load_test=False`，rows/labels/onehot/statistics/predictions/metrics 全为 0

## 输入哈希

- committed protocol SHA-256：`98314e70590094496418c0c8a43bb8b62497841a9b2437b9306f3d247e382c83`
- committed fold manifest SHA-256：`0ac344c9bb89cc71757c796096a8e2c63e8b4bb1cf9eeea2cab875fd2add8b2b`
- B01 freeze manifest SHA-256：`42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04`
- A06 split SHA-256：`024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706`

运行现场必须重新计算并精确匹配上述哈希；任一漂移 fail closed。

## R01 历史运行命令（不得重跑）

```bash
python scripts/run_slp8_region_full.py \
  --config configs/experiments/slp8_pm_full_protocol_v0.1.json \
  --output-dir outputs/experiments/EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R01 \
  --b01-freeze-dir /root/autodl-tmp/data/processed/slp8_training_tables_v0.1 \
  --dataset-root /root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1 \
  --one-fold-preflight \
  --candidate slp8_resunet_lite_v0.1 \
  --fold-id fold_1 \
  --seed 42 \
  --device cuda \
  --batch-size 16 \
  --max-epochs 30 \
  --experiment-id EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R01 \
  --run-authorized
```

## R02 Preflight Gate

R02 命令必须在新 runner SHA 形成后单独冻结。运行前必须确认：checkout 为该
clean SHA；`git status --porcelain` 为空；
CUDA/driver/PyTorch 可用；真实输入存在且 hash 匹配；output directory 不存在；
无其他训练进程。运行后必须存在唯一 DONE 或 FAILED、preflight manifest、
best/last checkpoint，且 reload hash、wall/CUDA budget、TEST=0 全部可审计。

## R02 历史运行命令（不得重跑）

仅在 checkout 为 clean `e0ba25a9aa0b33be971327ca398d822f7c7d1c8a`
且 Owner 重新授权 R02 后运行：

```bash
uv run python scripts/run_slp8_region_full.py \
  --config configs/experiments/slp8_pm_full_protocol_v0.1.json \
  --output-dir /root/autodl-tmp/outputs/EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R02 \
  --b01-freeze-dir /root/autodl-tmp/data/processed/slp8_training_tables_v0.1 \
  --dataset-root /root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1 \
  --experiment-id EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R02 \
  --one-fold-preflight \
  --candidate slp8_resunet_lite_v0.1 \
  --fold-id fold_1 \
  --seed 42 \
  --device cuda \
  --max-epochs 30 \
  --batch-size 16 \
  --run-authorized
```

## 禁止

- R01/R02 已失败封存；R03 已完成并封存，三者均不得重跑或覆盖。
- 不得改 candidate/fold/seed/epochs/batch size/budget 后沿用同一 EXP-ID。
- 不得运行 30-unit Full，不得访问 TEST，不得把 preflight 指标用于排名。
- 不得覆盖已有 output；失败必须保留并使用新 EXP-ID 重试。

## R03 历史运行命令（已完成，不得重跑）

该命令当时仅在 checkout 为 clean `02fb364902736a64ee8708440f0dd0bdddf860bc`、
AutoDL 随机张量 strict-CUDA primitive probe 通过且 Owner 明确授权后运行：

```bash
uv run python scripts/run_slp8_region_full.py \
  --config configs/experiments/slp8_pm_full_protocol_v0.1.json \
  --output-dir /root/autodl-tmp/outputs/EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R03 \
  --b01-freeze-dir /root/autodl-tmp/data/processed/slp8_training_tables_v0.1 \
  --dataset-root /root/autodl-tmp/datasets/SLP_8Region_Pressure_VAL_v1.1 \
  --experiment-id EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R03 \
  --one-fold-preflight \
  --candidate slp8_resunet_lite_v0.1 \
  --fold-id fold_1 \
  --seed 42 \
  --device cuda \
  --max-epochs 30 \
  --batch-size 16 \
  --run-authorized
```

## 当前本地盘点

- Windows 真实 B01 freeze 与 dataset root 存在，仅作只读链路核对。
- 本机为 RTX 4060 Laptop 8188 MiB，不是本任务冻结的 RTX 4090 运行环境，
  因此不在本机执行正式 preflight。
- R01 preflight：`FAILED_BEFORE_FIRST_EPOCH`；实际 GPU training `NOT STARTED`。
- R02：`FAILED_AT_FIRST_LOSS_FORWARD / ZERO_OPTIMIZER_STEPS`。
- R03：`PREFLIGHT_PASSED / DONE / TEST=0`；本地 archive 审计通过。
- 30-unit Full、TEST：`NOT RUN / NOT AUTHORIZED`。

## 下一 Gate

B08 runner 与真实 one-fold preflight 已验收。下一 Gate 是单独编写和复核 B09
Full 运行准备记录；30-unit Full 仍需 Owner 明确授权，不能从 B08 ACCEPT 推断。
TEST 继续禁止。
