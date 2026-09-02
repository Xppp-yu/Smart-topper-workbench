# TASK-SLP-B09-UNIT-IDENTITY-CARRIER-FIX-AND-R01-ADJUDICATION-v0.1

状态：`COMPLETE / OWNER_ACCEPTED_R01_WITH_LIMITATIONS / TEST_DENIED`

## 目标

1. 只读审计 B09 R01 原始 30-unit TRAIN+VAL Full 证据包。
2. 修复 `units/*/complete.json.identity` 漏写冻结 `git_dirty` 的实现缺陷。
3. 增加覆盖全部 30 个 unit carrier 的回归测试。
4. 基于原始证据与修复后的验证结果裁决 R01 是可带限制接受，还是必须使用新 EXP-ID 重跑。

## 输入与冻结边界

- 原始 EXP-ID：`EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01`
- 原始 runner Git SHA：`8b3ebdaa021405790b6137bff581acc490d8a024`
- 原始证据归档 SHA-256：`68156598a47ae65ba33d26f4005f9d9fdc8ec67ff24d43ffd605c19847ca5918`
- 运行范围：SLP8 pressure-only，TRAIN+VAL，2 candidates × 5 folds × 3 seeds。
- TEST：`DENIED`；不得调用 `enable_test_access`，不得读取 TEST label/onehot，测试与审计均保持 `TEST=0`。

## 允许修改

- `src/topper_perception/neural/slp8_region_full.py`
- `tests/test_slp8_region_full.py`
- `tests/test_b09_full_runner_cli_bridge.py`
- 本任务文档、B09 R01 阶段报告、项目状态与 backlog 治理文档。

## 禁止事项

- 不修改、回填、重打包或覆盖 R01 原始证据归档及解包内容。
- 不复用 R01 EXP-ID 执行新运行。
- 不启动 Mini/Full GPU，不执行 TEST。
- 不把运行完成等同于无条件研究验收。
- 完成独立验证后允许形成一个仅含本任务精确文件的本地 commit；未获后续明确指令
  不 push、不合入 `main`。

## 验收条件

- 原始归档 hash 与结构核验通过，并保存只读 audit-only 结果。
- 新生成的所有 unit `complete.json.identity.git_dirty` 与冻结 run identity 严格一致。
- 缺失或漂移的 unit identity 在 resume/audit 路径 fail closed。
- 定向测试、B08/B09 相关回归、validator、`py_compile`、`git diff --check` 通过。
- 阶段报告明确区分：原始事实、修复后代码能力、R01 永久限制、是否需要 R02。

## 当前 Gate

`TASK-SLP-B10-UNKNOWN-REJECT-READY-TO-DRAFT / TEST_DENIED`
