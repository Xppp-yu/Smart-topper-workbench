# TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1

状态：`ACCEPTED / COMMITTED_AND_PUSHED_AT_8b3ebda / GPU_FULL_NOT_AUTHORIZED / TEST_DENIED`

## 1. 任务目标

在 B09 运行准备提交 `2d02fbcbcff9d4a8249bdb7fbd8551955f612f7b`
之上，为已经实现的 `run_full()` 增加唯一、受治理的 30-unit real-B01 CLI
入口，并补齐 B09 audit 已确认缺失的少量 identity / TEST=0 / OOF coverage
carrier。

本任务只接通和治理现有能力，不重写训练、fold、OOF、checkpoint、resume、预算
或候选选择算法。

## 2. 起点与工作树

- TASK-ID：`TASK-SLP-B09-FULL-RUNNER-CLI-BRIDGE-v0.1`
- 工作树：`E:\TeamProjects\smarttopper-team-workbench-b09-cli-bridge`
- 分支：`codex/task-slp-b09-full-runner-cli-bridge-v0.1`
- 起点：`2d02fbcbcff9d4a8249bdb7fbd8551955f612f7b`
- 当前 Full Gate：`GPU_FULL_NOT_AUTHORIZED / TEST_DENIED`
- 实现任务不得创建或启动任何正式 `EXP-SLP-B09-*`。

## 3. 已验证的现有能力（禁止重复实现）

`src/topper_perception/neural/slp8_region_full.py` 已经具备：

- 2 candidates × 5 folds × 3 seeds = 30 unique units；
- real B01 TRAIN/VAL loading，且 `load_test=False`；
- fold-TRAIN-only preprocessing / class weights；
- per-unit checkpoint、resume、`complete.json` 和 budget state；
- unit-level `oof/unit_oof.npz`；
- seed OOF merge、pooled metrics、逐受试者指标和候选决策；
- `budget_report.json` 的四个冻结上限、per-unit/per-candidate wall 与
  per-candidate peak；
- 每 unit `complete.json.result.peak_cuda_mb`；
- 所有 planned units terminal 后通过 `write_terminal_state()` 写入唯一
  `DONE.json` / `FAILED.json` / `STOPPED.json`。

不得把上述项目列为新实现成果，也不得创建平行 writer 或旁路 runner。

## 4. 必须实现的 CLI bridge

### 4.1 新入口

在 `scripts/run_slp8_region_full.py` 增加：

```text
--run-full
```

要求：

1. `--run-full` 与 `--one-fold-preflight` 严格互斥。
2. `--run-full` 不得与 `--validate-only`、`--no-write` 或
   `--synthetic-cpu-smoke` 混用。
3. 必须同时提供：
   - `--config`
   - `--output-dir`
   - `--b01-freeze-dir`
   - `--dataset-root`
   - `--experiment-id`
   - `--run-authorized`
4. EXP-ID 必须严格匹配：

   ```text
   ^EXP-SLP-B09-PM-FULL-30-UNIT-\d{8}-AUTODL-R\d{2}$
   ```

5. 必须使用 clean、可解析的当前 Git commit；`git_dirty=True` 时在训练和
   output-dir 创建前拒绝。
6. 必须通过 `load_frozen_full_protocol()` 和 `build_full_config()` 构造配置，
   然后只调用现有 `run_full(full_config)`；禁止复制 30-unit 循环。
7. real Full 必须严格使用 B07 冻结的 max_epochs、min_epochs、patience、batch
   size、候选、seeds、folds 与 budget。CLI override 与冻结值不一致时必须拒绝。
8. 正式入口必须使用 real mode：`synthetic_mode=False`、`no_write_mode=False`。
9. 未授权、参数不完整、identity 不匹配、协议漂移或 output 冲突时必须非零退出，
   且不得训练、不得读取 TEST、不得创建新的实验目录。
10. 保持现有 validate-only、synthetic smoke 和 one-fold preflight 行为不变。

### 4.2 Resume / terminal 边界

- 不得用 `--force` 或删除产物实现 resume。
- 已有 `DONE.json` / `FAILED.json` / `STOPPED.json` 的 terminal 实验不得重新
  启动。
- 非 terminal 的中断目录只能通过现有完整 identity、unit `complete.json`、
  checkpoint 和 budget-state 校验继续。
- identity 不一致必须 fail-closed；完成 unit 不得覆盖或重训。
- 不得为测试方便削弱 `write_unit_complete_atomic()`、budget state 或 terminal
  互斥合同。

## 5. 必须补齐的 artifact carriers

只补齐以下已验证缺口：

1. 顶层 `status.json` 增加并核对：
   - `config_sha256`
   - `data_manifest_sha256`
   - `fold_manifest_sha256`
   - `split_sha256`
2. `input_manifest_hashes.json` 增加严格类型的：
   - `test_access: false`
   - `test_rows: 0`
   - `test_labels: 0`
   - `test_onehot: 0`
   - `test_predictions: 0`
   - `test_metrics: 0`
3. terminal JSON 的 DONE/FAILED/STOPPED payload 增加完整 frozen identity：
   - `status` / `terminal_state`
   - `experiment_id`
   - `git_commit`
   - `git_dirty`
   - 四个 frozen hashes
4. `candidates/<candidate>/candidate_decision.json` 的每个 seed block 增加
   `total_subjects`，并与现有 `SeedOOFResult.total_subjects` 一致。

不得重复新增：

- DONE writer；
- unit OOF；
- 四个 budget limits；
- per-unit `peak_cuda_mb`；
- 新的 OOF 格式或第二套 terminal 机制。

## 6. Validator 对齐

更新 `scripts/validate_b09_full_run_preparation.py`：

- preparation 静态检查从“确认 CLI 仍拒绝 Full”切换为验证
  `--run-full`、授权门、互斥门和 `run_full()` 唯一 dispatch；
- audit-only 必须继续 fail-closed 校验完整 carrier；
- 保留 A06 三方绑定、精确 30-unit key、budget 重算、两位小数 writer precision
  和完整 TEST=0 六 carrier；
- validator 自身不得训练、不得创建 output-dir、不得读取 TEST。

## 7. 允许修改

- `scripts/run_slp8_region_full.py`
- `src/topper_perception/neural/slp8_region_full.py`
- `scripts/validate_b09_full_run_preparation.py`
- `tests/test_slp8_region_full.py`
- `tests/test_b09_full_run_preparation.py`
- 可新增 `tests/test_b09_full_runner_cli_bridge.py`
- 本任务单
- `docs/PROJECT_STATUS.md` 与 `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` 的窄范围更新

需要修改其它文件时必须先停下说明，不得自行扩大范围。

## 8. 明确禁止

- 不运行 30-unit Full、one-fold GPU preflight 或任何 Mini；
- 不访问、加载、统计、预测或评估 TEST；
- 不调用 `enable_test_access()`；
- 不修改 B07 protocol / fold manifest；
- 不修改 B01 freeze / A06 split / raw data；
- 不改变候选、seed、fold、损失、优化器、超参数、early stopping 或选择规则；
- 不创建正式 EXP-ID output；
- 不覆盖任何已有实验；
- 不使用 `git add .` / `git add -A`；
- 未经 Codex 独立 Review 不 commit、不 push、不 merge。

## 9. 必须测试

至少覆盖：

1. `--run-full` + 完整授权参数只 dispatch 一次 `run_full()`。
2. 未给 `--run-authorized` 拒绝，且 `run_full()` call count = 0。
3. 缺 dataset root / freeze dir / EXP-ID 任一项均拒绝。
4. 非 B09 EXP-ID、synthetic sentinel、脏 Git 均拒绝。
5. `--run-full` 与 one-fold / validate / no-write / synthetic 任一混用均拒绝。
6. 冻结训练参数漂移拒绝。
7. 现有 one-fold preflight 定向回归通过且不 dispatch Full。
8. current real writer fixture 产生 B09 audit 所需全部 carriers。
9. DONE/FAILED/STOPPED identity 与 terminal 互斥。
10. TEST 六 carrier 严格为 false / int 0；不存在未知 TEST carrier。
11. candidate seed `total_subjects == 91`（完整 5-fold seed OOF）。
12. interruption/resume 不重训完成 unit，不覆盖 complete/checkpoint/budget。

必须实际运行：

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path

E:\TeamProjects\smarttopper-team-workbench\.venv\Scripts\python.exe `
  -m pytest tests/test_b09_full_runner_cli_bridge.py `
  tests/test_b09_full_run_preparation.py -q

E:\TeamProjects\smarttopper-team-workbench\.venv\Scripts\python.exe `
  -m pytest tests/test_slp8_region_full.py -q

E:\TeamProjects\smarttopper-team-workbench\.venv\Scripts\python.exe `
  -m pytest tests/test_check_markdown_links.py -q

E:\TeamProjects\smarttopper-team-workbench\.venv\Scripts\python.exe `
  -m py_compile scripts/run_slp8_region_full.py `
  src/topper_perception/neural/slp8_region_full.py `
  scripts/validate_b09_full_run_preparation.py

git diff --check
```

如果没有新增独立 bridge 测试文件，应调整第一条命令并在交付中解释覆盖位置。

## 10. 交付与 Gate

交付必须包含：

- TASK-ID、branch、base HEAD；
- 修改文件与每项合同对应位置；
- 实际命令和逐项结果；
- 未运行项明确写 `NOT RUN`；
- no-GPU / TEST=0 证据；
- artifact schema before/after 对照；
- verified / inferred / unverified / limitations；
- prohibited conclusions；
- staged / commit / push 状态；
- 当前 Gate 与下一 Gate。

实现交付后的 Gate：

```text
READY_FOR_CODE_REVIEW / GPU_FULL_NOT_AUTHORIZED / TEST_DENIED
```

Codex 已独立验收并将 bridge 合入 `main@8b3ebda`，B09 运行准备已升级为：

```text
B09_RUN_PREPARATION_ACCEPTED / FULL_RUNNER_CLI_BRIDGE_ACCEPTED /
GPU_FULL_NOT_AUTHORIZED / TEST_DENIED
```

独立验收结果：bridge + run-preparation + runner `172 passed`，Markdown links
`6 passed`，validator `80 OK / 0 ERR`，`py_compile` 通过；无 GPU、TEST=0。
之后仍需 Owner 以独立 EXP-ID、冻结 Git SHA 和预算明确授权，才能实际运行
30-unit Full。
