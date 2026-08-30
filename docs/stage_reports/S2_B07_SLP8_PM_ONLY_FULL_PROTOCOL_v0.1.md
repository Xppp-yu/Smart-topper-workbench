# S2 B07 — SLP8 PM-only Full Protocol v0.1

TASK-ID：`TASK-SLP-B07-FULL-PROTOCOL-FREEZE-v0.1`

状态：`PROTOCOL_ACCEPTED / COMPUTE_NOT_RUN / TEST_DENIED`

## 执行摘要

B04A R03 已由 Owner 接受为 `ACCEPT_WITH_LIMITATIONS`，因此 B07 前置 Gate
解除。本阶段只冻结开发期 Full 协议，不实现或运行训练。

## 冻结设计

| 项目 | 决定 |
|---|---|
| 候选 | DeepLabV3+-lite、ResUNet-lite |
| 开发池 | B01 TRAIN+VAL，91 subjects / 4,095 samples |
| folds | 固定 5-fold subject CV，19/18/18/18/18 subjects |
| seeds | 42、123、2026 |
| 总单元 | 2 candidates × 5 folds × 3 seeds = 30 |
| primary | 每 seed 拼接 4,095 OOF 后计算 foreground macro IoU，再跨 seed 算术均值 |
| 选择 | 选 1；差 `<0.02` 时按最低 subject IoU、参数量、版本名破同分 |
| 训练合同 | 完整继承 B04A，不做 post-Mini tuning |
| TEST | 0 rows / labels / onehot / predictions / metrics |
| 预算 | 15 min/unit，225 min/candidate，450 min total，8192 MiB peak |

Fold 1 为 19 subjects/855 samples；Fold 2–5 各 18 subjects/810 samples。
每个 subject 在每 seed 恰好进入一次 OOF，任何缺失或重复均 FAILED。

## 身份与产物边界

所有 JSON/CSV/checkpoint/log 必须携带同一冻结 identity：EXP-ID、clean Git
SHA、config/data/A06/fold SHA、model、candidate、fold、seed。任何漂移必须
在写训练产物前或恢复时 fail closed。

## 已验证、推断、未验证与限制

已验证：B01 manifest 提供 81 TRAIN + 10 VAL 开发 subjects；5 folds 精确
覆盖 91 subjects；B04A R03 晋级候选及指标来自不可变本地证据。

推断：15 min/unit 预算以 R03 约 4.3 min/seed 实测为基础并留出约 3.5 倍
余量；B08 一折预检必须重新测量，若超限只能版本化修改协议，不能边跑边改。

未验证：Full runner、resume、OOF merge、GPU one-fold 和 30-unit Full 均
`NOT RUN`。

协议验证：B07 validator + failure injection + Markdown links 共 18 passed；
正式 validator 输出 `TEST=0 folds=5 candidates=2 seeds=3 units=30`；JSON parse、
`py_compile` 与 `git diff --check` 通过。

限制：数据仅 danaLab/uncover，GT 为自动接受且 `NOT_REVIEWED`；不能外推
为人工像素级、医学、产品、舒适性、硬件或气囊控制证据。

## 下一 Gate

协议 validator 与独立审查已经通过，可进入 B08 Runner + one-fold preflight；
B09 Full 与 B09T TEST 继续分别阻塞。
