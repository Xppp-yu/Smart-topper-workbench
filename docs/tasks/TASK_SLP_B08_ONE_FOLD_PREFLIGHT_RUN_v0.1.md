# TASK-SLP-B08-ONE-FOLD-PREFLIGHT-RUN-v0.1

状态：`PREPARATION_READY_FOR_REVIEW / GPU_NOT_AUTHORIZED / TEST_DENIED`

## 目标

在 B08 runner 已接受后，仅执行一个真实 development fold/candidate/seed，
测量 wall time、peak CUDA memory，并验证 best-checkpoint reload 与完整 identity
carriers。该任务不是 B09 Full，也不产生候选排名。

## 冻结运行身份

- proposed EXP-ID：`EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R01`
- runner Git SHA：`5af426039ae41209af7929bc9319a0657e5f92b4`
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

## 唯一运行命令模板

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

## Preflight Gate

运行前必须确认：checkout 为上述 clean SHA；`git status --porcelain` 为空；
CUDA/driver/PyTorch 可用；真实输入存在且 hash 匹配；output directory 不存在；
无其他训练进程。运行后必须存在唯一 DONE 或 FAILED、preflight manifest、
best/last checkpoint，且 reload hash、wall/CUDA budget、TEST=0 全部可审计。

## 禁止

- 当前任务尚未获得 Owner GPU 授权，不得执行上述命令。
- 不得改 candidate/fold/seed/epochs/batch size/budget 后沿用同一 EXP-ID。
- 不得运行 30-unit Full，不得访问 TEST，不得把 preflight 指标用于排名。
- 不得覆盖已有 output；失败必须保留并使用新 EXP-ID 重试。

## 当前本地盘点

- Windows 真实 B01 freeze 与 dataset root 存在，仅作只读链路核对。
- 本机为 RTX 4060 Laptop 8188 MiB，不是本任务冻结的 RTX 4090 运行环境，
  因此不在本机执行正式 preflight。
- GPU preflight、Full、TEST：`NOT RUN`。

## 下一 Gate

Codex Review 本准备记录；接受后形成单独 preparation commit。随后由 Owner
明确决定是否授权该 EXP-ID 在 RTX 4090 执行。授权不得由本文件状态推断。
