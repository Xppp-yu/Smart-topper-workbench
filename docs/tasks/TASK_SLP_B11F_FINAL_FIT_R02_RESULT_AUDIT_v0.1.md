# TASK-SLP-B11F-FINAL-FIT-R02-RESULT-AUDIT-v0.1

状态：`ACCEPTED / DOCUMENTATION_CLOSURE / TEST_DENIED`

## 1. Purpose

对已获 Owner 精确授权并在 AutoDL 完成的 B11F final-development-fit R02 做独立只读审计，
登记三组 final checkpoint、终态、身份、预算和 TEST carrier。任务只允许提交脱敏摘要与哈希；
不得提交 checkpoint、证据压缩包、凭据或机器本地路径。

## 2. Frozen run

- EXP-ID：`EXP-SLP-B11F-PM-FINAL-FIT-20260904-AUTODL-R02`
- Runner：`a6a5d8e6f8db003149169ee48f71d6e41e445a80`
- Authorization-package release：`21bda4e0bdf6fde2691a254593957e6350187540`
- Authorized environment fingerprint：`a5a9342b18d00b614355e63ce056a7edd92dd80358d8aead5ef6e8e0ba045669`
- Launcher SHA-256：`0dea035c0af16b39617138177cdf441eb447463d55d90a06516b72dede5ade75`
- Evidence archive SHA-256：`a5a98916b79dca55366d6df6a8cd19df375fdb775b6e30ea775b8578cde70dad`
- Budget：首次启动起连续 `2700` 秒，停机计入且不可重置。
- TEST：`DENIED`，rows/labels/onehot 均须为 `0`。

## 3. Required review

1. 根状态只能有 `DONE.json`，不得并存 `RUNNING/STOPPED/FAILED`。
2. `DONE.json`、`environment.json`、`budget.json`、三个 `complete.json` 和六个 checkpoint
   的身份必须一致。
3. 三个 seed 必须为 42/123/2026，fixed epochs 必须为 15/20/12。
4. 三个 final checkpoint 的文件 SHA、reload carrier、模型与 optimizer state 必须完整。
5. checkpoint tensor 必须可加载且全部为有限数值。
6. config/candidate/B01/environment/budget SHA 与授权值必须一致。
7. TEST carrier 必须全部保持 0；不得执行 B09T 或读取 TEST。
8. 记录无法由原始证据支持的结论，不把 training loss 当 validation/TEST 指标。

## 4. Allowed repository changes

- `docs/tasks/TASK_SLP_B11F_FINAL_FIT_R02_RESULT_AUDIT_v0.1.md`
- `docs/stage_reports/S2_B11F_FINAL_FIT_R02_RESULTS_AND_AUDIT_v0.1.md`
- `docs/PROJECT_STATUS.md`
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`

允许对以上精确文件做 documentation-only commit 和 push。禁止修改训练代码、配置、候选合同、
数据或原始运行证据。

## 5. Exit gate

只有全部 checkpoint 和 carrier 独立审计通过，才能将 B11F 标记为 accepted，并进入 B09T
协议起草。B09T 执行与任何 TEST 读取仍需另一份一次性 Owner 精确授权。
