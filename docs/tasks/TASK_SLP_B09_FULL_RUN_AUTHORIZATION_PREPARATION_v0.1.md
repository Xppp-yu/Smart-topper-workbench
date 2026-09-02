# TASK-SLP-B09-FULL-RUN-AUTHORIZATION-PREPARATION-v0.1

状态：`OWNER_AUTHORIZED / READY_FOR_AUTODL_PREFLIGHT / TEST_DENIED`

## 1. 目的

为 SLP8 PM-only B09 的 30-unit Full 公平比较冻结一次可审计的运行授权包。
本文件记录待 Owner 决策的对象；它本身不构成运行授权。

本治理任务由 Owner 的“下一步推进”指令启动，只允许修改并提交以下文档：

- `docs/PROJECT_STATUS.md`
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`
- `docs/tasks/TASK_SLP_B09_FULL_RUN_PREPARATION_v0.1.md`
- `docs/tasks/TASK_SLP_B09_FULL_RUNNER_CLI_BRIDGE_v0.1.md`
- 本文件

允许形成一个 docs-only commit 并合入本地主线；不授权 GPU、Full、TEST、实验目录
写入或远端训练。push 仍需在提交后确认主线干净且只含上述治理文件。

## 2. 已满足的前置 Gate

- B07 Full 协议：`PROTOCOL_ACCEPTED / COMPUTE_NOT_RUN / TEST_DENIED`。
- B08 one-fold：`ACCEPT / R03_ONE_FOLD_PREFLIGHT_PASSED / TEST_DENIED`。
- B09 运行准备：`B09_RUN_PREPARATION_ACCEPTED`。
- B09 CLI bridge：`FULL_RUNNER_CLI_BRIDGE_ACCEPTED`，合入并推送
  `main@8b3ebdaa021405790b6137bff581acc490d8a024`。
- 独立复核：bridge + run-preparation + runner `172 passed`；Markdown links
  `6 passed`；validator `80 OK / 0 ERR`；`py_compile` PASS；GPU `NOT RUN`；TEST=0。

## 3. 待授权的冻结对象

| 字段 | 冻结值 |
|---|---|
| TASK-ID | `TASK-SLP-B09-FULL-RUN-AUTHORIZATION-PREPARATION-v0.1` |
| EXP-ID | `EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01` |
| Runner Git SHA | `8b3ebdaa021405790b6137bff581acc490d8a024` |
| Git dirty | 必须为 `false` |
| B07 protocol SHA-256 | `98314e70590094496418c0c8a43bb8b62497841a9b2437b9306f3d247e382c83` |
| B07 fold manifest SHA-256 | `0ac344c9bb89cc71757c796096a8e2c63e8b4bb1cf9eeea2cab875fd2add8b2b` |
| B01 freeze manifest SHA-256 | `42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04` |
| A06 split SHA-256 | `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706` |
| Candidates | `slp8_deeplabv3plus_lite_v0.1`, `slp8_resunet_lite_v0.1` |
| Folds | `fold_1`–`fold_5` |
| Seeds | `42`, `123`, `2026` |
| Units | `2 × 5 × 3 = 30` |
| Data | 91 development subjects / 4,095 TRAIN+VAL samples |
| TEST | denied; `load_test=False`; all TEST carriers must remain zero |

## 4. 预算与停止条件

- 每 unit wall time 上限：15 分钟。
- 每 candidate wall time 上限：225 分钟。
- 实验 total wall time 上限：450 分钟。
- peak CUDA memory 上限：8192 MiB。
- 任一 OOM、NaN、split leak、identity/hash 漂移、TEST 非零、checkpoint reload
  不一致、OOF coverage 不完整或预算超限，必须 fail-closed 并保留 FAILED/STOPPED
  证据；不得修改同一 EXP-ID 后重跑。

## 5. 授权边界

Owner 若选择 `AUTHORIZE`，授权范围仅为上述冻结 SHA/EXP-ID/30 units/budget 的
AutoDL RTX 4090 TRAIN+VAL Full。任何字段变化都必须新建 EXP-ID 并重新授权。

本授权不包括：

- B01 TEST 读取、预测、指标或 `enable_test_access()`；
- B09T 最终 TEST；
- 修改候选、fold、seed、训练参数、选择规则或预算；
- 覆盖 B08 或任何既有实验产物；
- 将公开数据结果外推到产品、硬件、舒适性、医疗、整夜或气囊控制。

## 6. Owner 决策栏

- 当前决定：`AUTHORIZE`（Owner 于 2026-09-02 明确回复“授权”）。
- 可选决定：`AUTHORIZE / ITERATE / STOP`。
- 授权范围严格限定为 §3–§5 冻结对象；TEST 仍未授权。

## 7. 运行前 Gate

Experiment Runner 在任何写入或训练前必须验证：clean checkout 精确等于冻结 SHA；
三份 SHA-256 与 A06 split 三方绑定一致；AutoDL CUDA 环境通过；目标 EXP-ID 目录
不存在或是同 identity 的非终态可恢复目录；validator 为 `80 OK / 0 ERR`；
Owner 决策为明确的 `AUTHORIZE`。

## 8. 当前结论与下一 Gate

- Verified：协议、runner、bridge、TEST=0 合同与本地复核通过。
- Unverified：30-unit wall/peak、完整 OOF、候选选择结果；因为 Full 尚未运行。
- 当前 Gate：`OWNER_AUTHORIZED / READY_FOR_AUTODL_PREFLIGHT / TEST_DENIED`。
- 下一 Gate：AutoDL clean checkout、CUDA、输入 hash 与 validator 全部通过后，由 Runner
  以冻结 EXP-ID 执行；任一预检失败则保持停止。
