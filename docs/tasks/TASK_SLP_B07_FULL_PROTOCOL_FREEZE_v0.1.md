# TASK-SLP-B07-FULL-PROTOCOL-FREEZE-v0.1

状态：`PROTOCOL_ACCEPTED / B08_NOT_STARTED / FULL_NOT_AUTHORIZED / TEST_DENIED`

## 目标

在 Owner 接受 B04A R03 `ACCEPT_WITH_LIMITATIONS` 后，冻结 SLP8 PM-only
开发期 Full 的候选、subject folds、训练合同、指标、选择规则、预算、身份
载体、停止条件和 TEST=0 边界。

## 前置证据

- B04A R03：9/9 units DONE，TEST=0，9/9 reload 一致。
- 晋级候选：DeepLabV3+-lite、ResUNet-lite。
- 原始 budget carrier 缺陷已披露；Reviewer 从 per-seed 证据重算资源，
  writer 修复已合入 `main@ae24d96`。
- B01 freeze/A06 split 不变；TEST 11 subjects / 495 rows 仅允许结构计数。

## 冻结决定

1. 候选仅为 `slp8_deeplabv3plus_lite_v0.1` 与
   `slp8_resunet_lite_v0.1`。
2. 将原 TRAIN+VAL 的 91 个开发受试者合并为 development pool；按排序 ID
   modulo 5 冻结 5-fold，19/18/18/18/18 subjects。
3. 3 seeds `[42,123,2026]`；2×5×3=30 个 fold-seed units；任一失败不允许
   静默丢弃。
4. B04A 的 optimizer/loss/epoch/augmentation 合同原样继承，不根据 Mini
   结果调参；class weights 和任何 preprocessing 仅在 fold-TRAIN 拟合。
5. 每 seed 拼接 5 folds 得到 4,095 条完整 OOF 后计算 primary；禁止把
   不等大小 fold 的简单均值当 primary。
6. Full 最终冻结 1 个开发期研究候选；primary 差 `<0.02` 时依次比较
   跨 seeds 最低 subject IoU、参数量、model_version。
7. TEST rows/labels/onehot/statistics/predictions/metrics 全部为 0；任何访问
   fail closed。

## 允许修改

- `configs/experiments/slp8_pm_full_folds_v0.1.json`
- `configs/experiments/slp8_pm_full_protocol_v0.1.json`
- `scripts/validate_b07_protocol.py`
- `tests/test_b07_protocol_validator.py`
- 本任务单、对应 stage report、项目状态和 Backlog 的窄范围更新

## 非目标

- 不实现 Full runner、checkpoint 或 resume。
- 不做 synthetic smoke、一折预检、Full、TEST 或 GPU 运行。
- 不创建正式 EXP-ID，不授权 AutoDL。
- 不修改 B04A 候选、结果或原始证据。

## 验收

- fold manifest 精确覆盖 91 subjects 一次，0 duplicate，0 TEST。
- 5 fold 样本总计每 seed 4,095；train/val subject overlap=0。
- 2 candidates × 5 folds × 3 seeds = 30 units。
- 配置、fold SHA、资源乘法、OOF primary、identity 与 TEST=0 可机器校验。
- JSON parse、validator tests、Markdown links、`git diff --check` 通过。

## 下一 Gate

协议已经 Codex 独立复核接受；下一任务为 `TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1`。
B08 仍需单独实现与验收；B09 Full 和 B09T TEST 均未授权。
