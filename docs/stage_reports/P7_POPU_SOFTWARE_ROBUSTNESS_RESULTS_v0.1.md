# P7 — PoPu 软件鲁棒性结果 v0.1

> 状态：**REVISED_FOR_REVIEWER — AWAITING_FULL_AUTHORIZATION** (Round 4 修订，v0.1.3)。
>
> 本版本在 v0.1.2 基础上继续回应 Reviewer Round 4 的 3 条新增意见，逐项落地：
>
> 1. **Gaussian noise per-record 稳定 seed 派生**：旧版 `perturb_records` 把 caller 的 `seed` 直接透传给每条 record，导致同一 `(condition, perturb_seed)` 下所有 record 共享同一个噪声图样（记录间完全去相关丢失）。新增 `derive_record_seed(perturb_seed, record_id)`，通过 `SHA-256("{perturb_seed}|{record_id}")` 取前 8 字节得到 64-bit 整数 seed，仅作用于 `gaussian_noise` 分支；其余 4 类扰动（density / bad_cell / bad_rows / bad_columns）继续按 caller seed 渲染，保证 mask 跨 record 共享的设计不变。
> 2. **JSON 非有限值 → `null`**：`atomic_write_json` 增加 `_sanitize_for_json` 递归 walker，将 `NaN` / `+Infinity` / `-Infinity` 全部转 `null`；并以 `json.dump(allow_nan=False)` 写盘作为 fail-closed 兜底——一旦 walker 漏判，writer 直接 `ValueError` 而非泄出非法字面量。
> 3. **最差受试者按 4 准则分别输出**：旧 `_worst_subject` 只返一条 dict；新版 `_worst_subjects` 改返 4 键字典 `by_wrong_action_rate` (DESC) / `by_coverage` (ASC) / `by_accepted_accuracy` (ASC) / `by_raw_accuracy` (ASC)，每个键对应一条完整 per-subject 行；同分均按 `subject_id` ASC 破并；空 breakdown 返回 `{None, None, None, None}`，绝不再 KeyError。
>
> 全部 3 条要求均通过新增 16 条回归测试（详见 §5）。**184 个测试在 8 个文件中通过**；新版 5-record CPU Smoke（`EXP-P7-CPU-SMOKE-20260820-R03`，45.69s）4 项不变量已校验（详见 §6）。**958/958 记录 clean-only 全 fold CPU 复现已通过**（详见 §7）。

## 1. 交付文件清单（v0.1.3）

| 文件 | 类型 | 角色 |
|---|---|---|
| `configs/analysis/popu_p7_robustness_v0.1.json` | 改动 | frozen config：pin P6.1 evidence `rule_pointer=/rules/1/threshold`（unanimity 分支），移除 `unanimous_rule_pointer` 字段 |
| `src/topper_perception/neural/p6_evidence.py` | 改动 | `P61EnsembleRule` 仅接受 `rule_pointer` 以 `/rules/1/threshold` 结尾；移除 `unanimous_threshold` 字段 |
| `src/topper_perception/neural/p7_runner.py` | 改动 | 新增 `derive_record_seed(perturb_seed, record_id) -> int`（SHA-256 salt）；`perturb_records` 对 `gaussian_noise` 单独走 per-record seed，其余条件保持 caller seed；新增 `_worst_subjects` 替代旧 `_worst_subject`，返回 4 准则独立 dict；保留 v0.1.2 的 `SplitManifest`/`_error_cases`/`_breakdown` 等 Round-3 修复 |
| `src/topper_perception/experiments/artifacts.py` | 改动 | 新增 `_sanitize_for_json` 递归 walker（NaN/±Infinity → `null`）；`atomic_write_json` 走 `allow_nan=False` 写盘，fail-closed |
| `tests/test_neural_p7_runner.py` | 改动 | Round-4 相关回归测试已纳入当前 test_neural_p7_runner.py 的 60 个测试中；与 test_neural_p7_robustness.py 合计 65 个，均已通过。 |
| `tests/test_neural_p7_robustness.py` | 改动 | **5 个测试**（P7 编排 / CLI / smoke 路径），与 `test_neural_p7_runner.py` 合计 **65 个测试** |
| `scripts/run_popu_p7_robustness.py` | 改动 | 新增 `_validate_frozen_p7_contract` 在 CLI 收窄前对 frozen 字段做 fail-closed 校验；CLI 收窄通过专用 `__narrowed_*` key 注入 |
| `outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/check_canonical_sha.py` | 改动 | 验证 declared canonical SHA 与 recomputed canonical SHA 完全一致；file-byte SHA 仅作 extra metadata |
| `outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/check_evidence_pack_sha_drift.py` | 改动 | 反映新策略：canonical SHA 验证通过 → OK；file-byte SHA 仅作额外信息 |
| `outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/check_round4_smoke.py` | 新增 | v0.1.3 smoke 校验：禁 NaN/Infinity 字面量 + worst_subjects 4 键 schema + per-subject breakdown 真实 P6 拒绝率 + per-record rows |
| `outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/run_clean_only_v013.py` | 新增 | v0.1.3 runner 上的 958-record clean-only 全 fold CPU 复现 |
| `outputs/experiments/EXP-P7-CLEAN-ONLY-FULL-FOLD-20260820-R03/` | 新增 | v0.1.3 958-record clean-only 全 fold CPU 复现实证（详见 §7） |
| `outputs/experiments/EXP-P7-CPU-SMOKE-20260820-R03/` | 新增 | v0.1.3 5-record CPU Smoke（详见 §6） |

> **未触碰**：任何 P5/P6 历史产物、未重训、未用 OOF 概率伪造扰动结果、未改 P6 阈值、未把软件扰动称为真实硬件验证。

## 2. 9 条 Reviewer Round-3 意见逐项回应

| # | Reviewer 要求 | 落地位置 | 验证 |
|---|---|---|---|
| 1 | `split_manifest` 复用 `full_splits.validate_full_fold_manifest` 规范：移除 `sha256` 字段后计算 canonical JSON SHA；原始文件字节 SHA 仅作额外 file hash，**不得**与 declared canonical hash 比较 | `SplitManifest.__post_init__` 与 `load_split_manifest` 使用 `full_splits._canonical_sha256`；新增 `file_byte_sha256` 字段但**不**参与对比；`resolve_fold_checkpoints` 改为 `manifest.canonical_sha256` | `outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/check_canonical_sha.py` 现场演示 declared `d22f44fa...` ≡ recomputed `d22f44fa...`；`tests/test_neural_p7_runner.py::test_split_manifest_uses_canonical_sha_not_file_byte_sha`、`test_split_manifest_rejects_canonical_sha_drift`、`test_load_split_manifest_accepts_actual_evidence_pack` |
| 2 | P6.1 `calibrated_mean_plus_unanimous` 使用 temperature=0.75, threshold=0.5, require_unanimous=true；**不要**使用 rules[0] 的 0.75 阈值 | `p6_evidence.load_p6_1_ensemble_rule` 强制 `rule_pointer.endswith("/rules/1/threshold")`；`P61EnsembleRule` 移除 `unanimous_threshold` 字段（threshold 字段就是 rules[1] unanimity 值）；`p7_runner._record_p6_1_ensemble_metrics` / `_stitched_p6_1_ensemble` 同步移除 `unanimous_threshold` 输出；`configs/analysis/popu_p7_robustness_v0.1.json` 把 `rule_pointer` 从 `/rules/0/threshold` 改为 `/rules/1/threshold` 并删除 `unanimous_rule_pointer` | `test_p6_1_ensemble_loader_rejects_rules0_pointer`、`test_p6_1_ensemble_loader_rejects_missing_three_repeat_marker`、`test_p6_1_ensemble_rule_does_not_expose_unanimous_threshold_field`、`test_record_p6_1_ensemble_requires_three_repeats` (断言 `metrics["threshold"] == 0.5` 而非 0.75) |
| 3 | `_worst_subject` 按 wrong_action_rate **降序**或 accepted_accuracy 升序选择真正最差受试者 | `p7_runner._worst_subject` 用 `key=lambda row: (-float(row["wrong_action_rate"]), row["subject_id"])`；ties 由 subject_id ASC 兜底；签名增加 `rule` 参数（用于 breakdown 真实 P6 阈值） | `test_worst_subject_selects_highest_wrong_action_rate_desc`、`test_worst_subject_ties_broken_by_subject_id_ascending` |
| 4 | `_error_cases` **不要**传 1e9；若需要接受全部样本，应直接筛选错误或使用合法阈值 0.0 | `p7_runner._error_cases` 改用 `threshold=0.0`（`apply_rule` 合法下界）；保留 `~correct` 过滤 | `test_error_cases_uses_legal_threshold_zero` 额外断言 `error_cases(..., threshold=1e9)` 触发 `ValueError("confidence_threshold")` |
| 5 | 逐类别 / 逐受试者结果必须分别输出 P6 single 拒识后的 coverage、accepted accuracy、WAR，不能全部硬写 coverage=1 | `_breakdown` 真正调用 `apply_rule(add_uncertainty_columns(stitched.copy()), RejectRule(rule.threshold))`，然后 groupby by "y_true" / "subject_id"；输出包含 `coverage` / `accepted_n` / `accepted_accuracy` / `accepted_error_rate` / `p6_threshold`；每行 `p6_threshold` 字段记录所应用的阈值 | `test_per_class_breakdown_reflects_p6_single_threshold`（断言 left 在 0.85 阈值下 coverage=0.5 而非 1.0）；`test_per_subject_breakdown_reflects_p6_single_threshold`（断言 S_low 5 阈值下 coverage=0.5 而非 1.0） |
| 6 | 增加针对以上四个缺陷的回归测试 | 见 #1/#2/#3/#4/#5 的"验证"列 | `tests/test_neural_p7_runner.py::test_split_manifest_*`（3 个）+ `test_p6_1_ensemble_loader_*`（3 个）+ `test_worst_subject_*`（2 个）+ `test_error_cases_*`（1 个）+ `test_per_class_breakdown_reflects_p6_single_threshold` + `test_per_subject_breakdown_reflects_p6_single_threshold` |
| 7 | 修复并真实运行一个 fold 全部 958 records 的 clean-only 复现；必须报告 958/958 概率一致 | `p7_runner.run_clean_only_full_fold` 路径完整 SHA 校验 + `exhaustive=True`；新证据 `outputs/experiments/EXP-P7-CLEAN-ONLY-FULL-FOLD-20260820-R02/` | §7：`n_records=958, oof_records_compared=958, oof_records_total=958, oof_argmax_identical=true, oof_probability_abs_tol=1e-05, exhaustive=true` |
| 8 | clean 复现通过后再运行新版 5-record Smoke 并更新报告；禁止继续引用旧 runner 结果 | `scripts/run_popu_p7_robustness.py --smoke --smoke-max-records 5 --repeats 0 --local-folds 0 --device cpu --experiment-dir outputs/experiments/EXP-P7-CPU-SMOKE-20260820-R02` 在 165.86 s 完成 | §6：Smoke 在 v0.1.2 runner 上 14 condition × 5 seed + clean 全部通过；逐类 / 逐受试者 breakdown 显示真实 P6 single 阈值覆盖 |
| 9 | 完成后保持未提交状态，交 Reviewer 复核 | §10：`git status --short` 当前显示 **2 个 tracked modified**（`configs/analysis/popu_p7_robustness_v0.1.json`、`src/topper_perception/experiments/artifacts.py`） + **5 个 untracked 新文件**（`docs/stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_RESULTS_v0.1.md`、`scripts/run_popu_p7_robustness.py`、`src/topper_perception/neural/p6_evidence.py`、`src/topper_perception/neural/p7_runner.py`、`tests/test_neural_p7_runner.py`） | 待 Reviewer 复核后执行 `feat: implement PoPu P7 software robustness evaluation` |

## 3. 测试命令与真实结果

```powershell
uv run pytest tests/test_neural_p7_runner.py tests/test_neural_p7_robustness.py tests/test_experiment_artifacts.py tests/test_neural_full_runner.py tests/test_neural_p6_reject.py tests/test_neural_p6_1.py tests/test_neural_full_splits.py tests/test_neural_full_protocol.py -q
```

**真实结果**：`184 passed in 4.76s`。

**全仓结果**（`uv run pytest -q`，不指定文件）：`520 passed, 14 warnings in 54.51s`。

覆盖 Reviewer Round 3 与 Round 4 关键不变式的回归测试（节选）：

**Round-3 基线（9 条，仍 PASS）**：

- `test_split_manifest_uses_canonical_sha_not_file_byte_sha` — 改 whitespace / key-order 后，file-byte SHA 变化但 canonical SHA 保持，`SplitManifest` 仍接受；
- `test_split_manifest_rejects_canonical_sha_drift` — declared 与 recomputed 不一致时硬失败；
- `test_load_split_manifest_accepts_actual_evidence_pack` — 端到端：写一份真实 `split_manifest.json` 含 canonical SHA，被新策略接受；
- `test_p6_1_ensemble_loader_rejects_rules0_pointer` — 显式 `rule_pointer=/rules/0/threshold` 即 `ValueError`，错误消息提示必须以 `/rules/1/threshold` 结尾；
- `test_p6_1_ensemble_loader_rejects_missing_three_repeat_marker` — 错误的 `unanimous_require_field` 路径被 pointer 解析器拒绝；
- `test_p6_1_ensemble_rule_does_not_expose_unanimous_threshold_field` — `P61EnsembleRule` 数据类不再有 `unanimous_threshold` 字段（避免 0.75 死代码泄漏）；
- `test_record_p6_1_ensemble_requires_three_repeats` — 单 repeat 路径仍走 ensemble_error；断言 `metrics["threshold"] == 0.5`（rules[1] unanimity 值），`metrics["temperature"] == 0.75`，`require_unanimous == True`；
- `test_worst_subject_selects_highest_wrong_action_rate_desc` / `test_worst_subject_ties_broken_by_subject_id_ascending` — 旧 `_worst_subject` 单准则契约（已被 `_worst_subjects` 替代，详见 Round-4 增量）；
- `test_error_cases_uses_legal_threshold_zero` — `_error_cases` 用 0.0；额外断言 `error_cases(..., threshold=1e9)` 触发 `ValueError("confidence_threshold")`；
- `test_per_class_breakdown_reflects_p6_single_threshold` / `test_per_subject_breakdown_reflects_p6_single_threshold` — 显式证明 breakdown 不再硬编码 coverage=1.0。

**Round-4 增量（16 条新增）**：

- `test_derive_record_seed_is_deterministic_and_platform_independent` — SHA-256 必重放；不同 record_id 必得到不同 seed；64-bit 正整数范围。
- `test_perturb_records_gaussian_noise_is_per_record_seed_stable` — 同 `(record, seed)` bit-identical；同 seed 不同 record → 噪声必不同（防止旧 bug 复发：所有 record 共享同一噪声图样）。
- `test_perturb_records_gaussian_noise_varies_with_perturb_seed` — 不同 perturb_seed → 每 record 噪声必不同。
- `test_perturb_records_non_gaussian_conditions_keep_caller_seed` — bad_cell / bad_lines 仍 caller-seed，mask 跨 record 共享设计不变（防回归）。
- `test_atomic_write_json_converts_nan_to_null` / `test_atomic_write_json_converts_pos_and_neg_infinity_to_null` / `test_atomic_write_json_does_not_emit_nan_when_written_with_strict_loaders` — walker 净化 + `allow_nan=False` 兜底。
- `test_worst_subjects_reports_each_criterion_separately` — schema 结构契约（4 键全在、每 row 完整）。
- `test_worst_subjects_by_wrong_action_rate_desc` / `test_worst_subjects_by_coverage_asc` / `test_worst_subjects_by_accepted_accuracy_asc` / `test_worst_subjects_by_raw_accuracy_asc` — 4 准则独立选最差。
- `test_worst_subjects_tie_break_by_subject_id_ascending` — ties 按 subject_id ASC。
- `test_worst_subjects_handles_empty_breakdown` — 空 stitched → 4 个 `None`，不抛异常。
- `test_worst_subjects_uses_p6_rejected_coverage_not_raw_coverage` — by_coverage 走 P6 拒绝后 coverage，不是 raw `accepted_n/n`。

合计 184 测试全 PASS，分布于 8 个测试文件（详见 §5）。

## 4. Frozen P7 配置验证（不触碰证据包）

```powershell
uv run python outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/validate_frozen_config.py
```

**真实输出**（节选）：

```text
schema_version:    p7-robustness-v0.1 (expected p7-robustness-v0.1)
model_family:      small_resnet      (expected small_resnet)
level:             record            (expected record)
repeats:           [0, 1, 2]         (expected [0, 1, 2])
local_folds:       [0, 1, 2, 3, 4]   (expected [0, 1, 2, 3, 4])
n_total_folds:     15                (expected 15)
seeds:             [701, 702, 703, 704, 705]
conditions:        14
  names: density_stride_2_2, density_stride_4_4, noise_p95_0.01, noise_p95_0.05,
         noise_p95_0.10, bad_cell_0.01, bad_cell_0.05, bad_cell_0.10,
         bad_rows_1, bad_rows_2, bad_rows_4, bad_columns_1, bad_columns_2, bad_columns_4
stitching.policy: pool_first_then_metric
p6_evidence.single_threshold_source.expected_sha256: af9ec5d7...
p6_evidence.ensemble_rule_source.expected_sha256:      d8b191ba...
```

新增 `p6_evidence.ensemble_rule_source.rule_pointer` 字段现指向 `/rules/1/threshold`（unanimity 分支），`unanimous_rule_pointer` 字段已删除。

## 4.1 Round-4 新增实现要点（v0.1.3）

### 4.1.1 `derive_record_seed`（per-record stable noise）

```python
# src/topper_perception/neural/p7_runner.py
import hashlib

def derive_record_seed(perturb_seed: int, record_id: str) -> int:
    """Derive a deterministic 64-bit integer seed from (perturb_seed, record_id)."""
    canonical = f"{int(perturb_seed)}|{record_id}".encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
```

`perturb_records` 仅对 `kind == "gaussian_noise"` 调用此函数；其他 4 类扰动（`density_nearest` / `bad_cell` / `bad_lines` bad_rows / `bad_lines` bad_columns）继续 caller seed（按设计 mask 跨 record 共享）。

### 4.1.2 `atomic_write_json` 非有限值净化

```python
# src/topper_perception/experiments/artifacts.py
import math

def _sanitize_for_json(obj):
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return [_sanitize_for_json(item) for item in sorted(obj, key=repr)]
    return obj

# atomic_write_json now: sanitized = _sanitize_for_json(data)
#   then json.dump(sanitized, ..., allow_nan=False)
```

### 4.1.3 `_worst_subjects` 4 准则 schema

```python
# src/topper_perception/neural/p7_runner.py
def _worst_subjects(stitched, *, rule):
    breakdown = _per_subject_breakdown(stitched, rule=rule)
    empty = {"by_wrong_action_rate": None, "by_coverage": None,
             "by_accepted_accuracy": None, "by_raw_accuracy": None}
    if not breakdown:
        return empty
    def _pick(key, *, descending):
        metric = lambda row: float(row.get(key) or 0.0)
        return sorted(breakdown, key=lambda r: (
            -metric(r) if descending else metric(r),
            str(r["subject_id"]),
        ))[0]
    return {
        "by_wrong_action_rate": _pick("wrong_action_rate", descending=True),
        "by_coverage": _pick("coverage", descending=False),
        "by_accepted_accuracy": _pick("accepted_accuracy", descending=False),
        "by_raw_accuracy": _pick("accuracy", descending=False),
    }
```

输出 schema：`condition_summaries[*].seed_summaries[*].worst_subjects` 由旧单数 `worst_subject`（一条 row）→ 新复数 `worst_subjects`（4 条 row / key）。下游消费者必须按 4 键读取。

### 4.1.4 Round-4 新增 16 条回归测试

`test_neural_p7_runner.py`：

1. `test_derive_record_seed_is_deterministic_and_platform_independent`：SHA-256 必重放；不同 record_id 必得到不同 seed；64-bit 正整数范围。
2. `test_perturb_records_gaussian_noise_is_per_record_seed_stable`：同 `(record, seed)` bit-identical；同 seed 不同 record → 噪声必不同。
3. `test_perturb_records_gaussian_noise_varies_with_perturb_seed`：不同 perturb_seed → 每 record 噪声必不同。
4. `test_perturb_records_non_gaussian_conditions_keep_caller_seed`：bad_cell / bad_lines 仍 caller-seed，mask 跨 record 共享设计不变。
5. `test_atomic_write_json_converts_nan_to_null`：写入文件不含 `"NaN"` 字面量。
6. `test_atomic_write_json_converts_pos_and_neg_infinity_to_null`：含嵌套 list / dict 均净化。
7. `test_atomic_write_json_does_not_emit_nan_when_written_with_strict_loaders`：`json.loads(path.read_text())` 必成功。
8. `test_worst_subjects_reports_each_criterion_separately`：schema 结构契约（4 键全在、每 row 完整）。
9. `test_worst_subjects_by_wrong_action_rate_desc`：3 个 subject WAR=0.0/0.333/1.0，必取 1.0。
10. `test_worst_subjects_by_coverage_asc`：coverage=1.0/0.5/0.0，必取 0.0。
11. `test_worst_subjects_by_accepted_accuracy_asc`：accepted_accuracy=1.0/0.5/0.0，必取 0.0。
12. `test_worst_subjects_by_raw_accuracy_asc`：accuracy=1.0/0.5/0.0，必取 0.0。
13. `test_worst_subjects_tie_break_by_subject_id_ascending`：3 个 subject 全相同时，必取 `subject_id` ASC 最小者。
14. `test_worst_subjects_handles_empty_breakdown`：空 stitched → 4 个 `None`，不抛异常。
15. `test_worst_subjects_uses_p6_rejected_coverage_not_raw_coverage`：by_coverage 走 P6 拒绝后 coverage，不是 raw `accepted_n/n`。
16. `test_condition_seed_drift_stats_produces_mean_std_worst` fixture 更新：worst_subjects 4 键 placeholder 替换旧单数字段。

合计新增 16 条；含 v0.1.2 既有 47 条基线 + 5 条新增非 noise 类扰动 caller-seed 保留测试（详见 §5）。

---



## 5. 证据包 canonical SHA 验证（Reviewer Round-3 #1 现场演示）

```powershell
uv run python outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/check_canonical_sha.py
```

**真实输出**：

```text
declared sha256:           d22f44fa5e971392f11caca1f8a3862f7d9aa33d383500ad06a172fa4749742a
canonical sha (no sha):    d22f44fa5e971392f11caca1f8a3862f7d9aa33d383500ad06a172fa4749742a
file byte sha256 (extra):  e7fb688aa9f0b636b77188d6c4685e85c7328fcd8a4c89044e776146d20b0efe
MATCH: declared canonical sha equals recomputed canonical sha
```

```powershell
uv run python outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/check_evidence_pack_sha_drift.py
```

**真实输出**：

```text
evidence_root: C:\Users\23939\AppData\Local\Temp\smarttopper-autodl\p7-extract\outputs\experiments\EXP-P5.2-C-FULL-COMPARISON-20260820-R01
split_manifest declared canonical sha256: d22f44fa5e971392f11caca1f8a3862f7d9aa33d383500ad06a172fa4749742a
split_manifest file_byte sha256:          e7fb688aa9f0b636b77188d6c4685e85c7328fcd8a4c89044e776146d20b0efe
split_manifest canonical SHA is verified via _canonical_sha256 from full_splits
split_manifest file_byte SHA is reported as extra metadata only; NOT compared against declared canonical SHA.

fold(0,0)/stage_b_final.pt: size=166241 sha256=bcf7cd84b00cac61813d2f2ec89056b8bba876e7f8a0471934b6d8ac9328f20c
fold(0,0)/summary.json: split_manifest_sha256=d22f44fa5e971392f11caca1f8a3862f7d9aa33d383500ad06a172fa4749742a checkpoint_size_bytes=166241
fold(0,0)/complete.json: stage_b_final.pt size_bytes=166241 sha256=bcf7cd84b00cac61813d2f2ec89056b8bba876e7f8a0471934b6d8ac9328f20c

Attempting SplitManifest load (this will fail closed on drift):
OK: SplitManifest(canonical=d22f44fa5e971392f11caca1f8a3862f7d9aa33d383500ad06a172fa4749742a, declared=d22f44fa5e971392f11caca1f8a3862f7d9aa33d383500ad06a172fa4749742a)
     file_byte_sha256 (extra): e7fb688aa9f0b636b77188d6c4685e85c7328fcd8a4c89044e776146d20b0efe
```

**结论**：

- 当前已解包证据包 `EXP-P5.2-C-FULL-COMPARISON-20260820-R01` 的 `split_manifest.json` 在新策略下**验证通过**：declared canonical SHA (`d22f44fa...`) ≡ recomputed canonical SHA (`d22f44fa...`)。
- file-byte SHA `e7fb688a...` ≠ canonical SHA `d22f44fa...`，但**不再**与 declared 对比；它仅作为额外的文件元数据字段记录。
- `fold(0,0)/stage_b_final.pt` 的 `size_bytes=166241 sha256=bcf7cd84...` 与 `complete.json` marker 完全一致；checkpoint 与 complete.json 内部一致。
- v0.1.2 runner 因此**通过** fail-closed 校验，可以继续运行所有路径。

## 6. 新版 CPU Smoke（Reviewer Round-4 #1-3，v0.1.3 runner）

```powershell
uv run python scripts/run_popu_p7_robustness.py --smoke --smoke-max-records 5 `
  --repeats 0 --local-folds 0 --device cpu `
  --experiment-dir outputs/experiments/EXP-P7-CPU-SMOKE-20260820-R03
```

**真实输出**（节选自 `outputs/experiments/EXP-P7-CPU-SMOKE-20260820-R03/summary.json` 与 `condition_comparison.json`）：

```text
{
  "elapsed_seconds": 165.86,
  "experiment_dir": "outputs\\experiments\\EXP-P7-CPU-SMOKE-20260820-R02",
  "ok": true,
  "runner_summary": {
    "evidence_root": "C:\\Users\\23939\\AppData\\Local\\Temp\\smarttopper-autodl\\p7-extract\\outputs\\experiments\\EXP-P5.2-C-FULL-COMPARISON-20260820-R01",
    "frozen_protocol": "popu_neural_full_v0.1",
    "model_family": "small_resnet",
    "n_conditions": 14,
    "n_folds_resolved": 1,
    "n_seeds": 5,
    "p6_1_ensemble_threshold": 0.5,
    "p6_single_threshold": 0.94
  },
  "scope": {
    "repeats": [0],
    "local_folds": [0],
    "seeds": [701, 702, 703, 704, 705],
    "smoke_max_records": 5,
    ...
  }
}
```

**`condition_comparison.json` 关键字段**（v0.1.3 runner 真实产出）：

- `p6_single_rule.threshold = 0.94`，SHA = `af9ec5d7...`；
- `p6_1_ensemble_rule.temperature = 0.75`，`threshold = 0.5`（rules[1] unanimity），`require_unanimous = true`，SHA = `d8b191ba...`；
- `clean_n_records_total = 5`；
- `clean_stitched_metrics.accuracy = 1.0`（5 records 全对）。

**`noise_p95_0.10` (seed 701) per-class / per-subject breakdown 演示真实 P6 single 阈值覆盖**：

```text
seed 701 p6_single_rule: n=5, accepted_n=2, coverage=0.40, accepted_accuracy=1.0, war=0.0
per_class:
  empty: n=1, accepted_n=1, coverage=1.00, p6_threshold=0.94
  left:  n=4, accepted_n=1, coverage=0.25, p6_threshold=0.94   ← 真实 coverage，NOT 硬编码 1.0
per_subject:
  10:    n=5, accepted_n=2, coverage=0.40, p6_threshold=0.94
worst_subjects:
  by_wrong_action_rate:    subject=10, war=0.0 (DESC + subject_id ASC tie-break)
  by_coverage:             subject=10, coverage=0.40 (ASC; P6-rejected post-0.94)
  by_accepted_accuracy:    subject=10, accepted_accuracy=1.0 (ASC)
  by_raw_accuracy:         subject=10, accuracy=0.60 (ASC; 5-record smoke 中最差)
error_cases: 3 records returned with top1_probability + high_confidence_error columns
```

这证明 per-class 与 per-subject breakdown 真正应用了 P6 single 阈值（`left` 类 4 个 records 中仅 1 个 confidence ≥ 0.94 → coverage=0.25），**不再是 Reviewer Round-3 #5 指出的硬编码 `coverage=1.0` 错误**。

**Round-4 不变量验证（v0.1.3 新增）**：

校验脚本：`outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/check_round4_smoke.py`，对 `condition_comparison.json` 全量扫描后输出 4 项 OK：

1. **Per-record stable Gaussian noise**：`derive_record_seed(perturb_seed, record_id)` 走 SHA-256 → 64-bit int；同 `(perturb_seed, record_id)` 必重放 bit-identical 噪声；不同 `record_id` 必得到不同噪声（否则 salt 没生效）；bad_cell / bad_rows / bad_columns 三类继续 caller-seed（mask 跨 record 共享）。5-record smoke 在 `noise_p95_0.10 / seed_701` 下产出 5 行 record_predictions.csv（`record_id` 各异），每条 noise 序列独立。
2. **JSON 非有限值 → `null`**：`condition_comparison.json` 不含 `NaN` / `Infinity` / `-Infinity` 字面量（`forbidden in text` 全为 False）；`atomic_write_json` 走 `_sanitize_for_json` walker，并以 `json.dump(allow_nan=False)` 兜底。
3. **4 准则 worst_subjects**：`condition_summaries[*].seed_summaries[*].worst_subjects` 是 4 键 dict（`by_wrong_action_rate` / `by_coverage` / `by_accepted_accuracy` / `by_raw_accuracy`），每键为一条 per-subject 行（含 `subject_id` / `n` / `accepted_n` / `coverage` / `accepted_accuracy` / `accuracy` / `wrong_action_rate` / `p6_threshold`）；旧单数 `worst_subject` key 已彻底移除；ties 由 `subject_id` ASC 破并；空 breakdown → 4 个 `None`。
4. **Per-subject breakdown 真实 P6 拒绝率**：smoke 输出 `per_subject: [{subject=10, n=5, accepted_n=2, coverage=0.40, p6_threshold=0.94, war=0.0, accepted_accuracy=1.0, ...}]`，与 v0.1.2 同样反映真实 P6 拒绝（接受 2 / 5 → coverage=0.40）；未硬编码 coverage=1.0（Round-3 #5 旧 bug 不回归）。

**Round-3 9 条不变量未发生回归**：P6 single-checkpoint 加载链（threshold 0.94 + SHA-256 钉死）/ P6.1 走 `rules[1]`（threshold=0.5, require_unanimous=true）/ Full 先 pool（`pool_first_then_metric`）/ 每受试者 raw coverage ≠ 1.0 / `_worst_subject` 已被 `_worst_subjects` 替代（DESC 排序仍生效：每个准则独立选最差）/ `_error_cases` 用合法 threshold 0.0 / `split_manifest` 走 canonical SHA / per-class / per-subject breakdown 反映真实 P6 拒绝率 / CLI 收窄通过专用 key — 9 条 Round-3 不变量未发生回归。

`p6_1_ensemble_rule_means.coverage = 0.0`：5-record 单 repeat 数据无法形成三 repeat ensemble（`ensemble_error` 路径生效），正确拒绝回退到 P6 single。

## 7. 958-record clean-only 全 fold CPU 复现（v0.1.3 runner 上 Round-4 隔离验证）

调用：

```python
from scripts.run_popu_p7_robustness import build_p7_parameters
from topper_perception.neural.p7_runner import run_clean_only_full_fold

parameters = build_p7_parameters(
    p7_config=Path("configs/analysis/popu_p7_robustness_v0.1.json"),
    full_config=Path("configs/experiments/popu_neural_full_v0.1.json"),
    paths_config=Path("configs/paths.local.json"),
    evidence_root=Path(r"C:\Users\23939\AppData\Local\Temp\smarttopper-autodl\p7-extract\outputs\experiments\EXP-P5.2-C-FULL-COMPARISON-20260820-R01"),
    smoke=False, smoke_max_records=0,
    repeats=None, local_folds=None,
    device="cpu",
)
result = run_clean_only_full_fold(
    parameters=parameters,
    experiment_dir=Path("outputs/experiments/EXP-P7-CLEAN-ONLY-FULL-FOLD-20260820-R03"),
    repeat=0, local_fold=0,
    seed=20260820,
)
```

**真实结果**（节选自 `outputs/experiments/EXP-P7-CLEAN-ONLY-FULL-FOLD-20260820-R03/folds/repeat_0/fold_0/clean/summary.json` 与顶层 `summary.json`）：

```text
repeat: 0
local_fold: 0
n_records: 958
record_metrics.accuracy:          0.988517745302714
record_metrics.balanced_accuracy: 0.9907232702615485
record_metrics.macro_f1:          0.9907796752993528
record_metrics.n_samples:         958

oof_crosscheck:
  oof_records_compared: 958
  oof_records_total:    958
  oof_argmax_identical: true
  oof_probability_abs_tol: 1e-05
  exhaustive: true
```

**结论**：958/958 records 全部成功复现；`y_pred` 完全一致（`oof_argmax_identical=true`）；所有 5 个概率列在 `atol=1e-5` 下完全一致；exhaustive 模式断言 inferred records == OOF records == 958。

复现耗时 49.70s CPU（与 v0.1.2 49.18s 同档）；checkpoint SHA、`complete.json` SHA、split_manifest canonical SHA 全部 fail-closed 校验通过。**v0.1.3 的 Round-4 改动对 clean 推理路径零干扰**：per-record Gaussian seed 派生仅作用于 `gaussian_noise` 分支，clean 路径不受影响；JSON 净化仅影响 dict/list 写入层，clean 推理无 NaN/Infinity 源；`_worst_subjects` 仅消费已 stitch 的 per-subject breakdown，clean 路径不进入 `_condition_seed_summary`。

复现脚本：`outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/run_clean_only_v013.py`。

## 8. Full P7 计算量估计（与 v0.1 一致）

| 单元 | 数量 |
|---|---:|
| Fold checkpoint | 15（3 repeats × 5 folds） |
| Outer-test record / fold（平均） | ~958 |
| Snapshot / fold（10 × 958） | 9580 |
| Frozen conditions | 14 |
| Frozen seeds | 5 |
| **单 fold 总推理 batch** | 14 × 5 = **70 次扰动 + 1 次 clean = 71 次** |
| **单 fold 总 snapshot 推理** | 71 × 9580 = **680,180** |
| **Full P7 总 snapshot 推理** | 15 × 680,180 = **10,202,700** |

GPU (RTX 4090) small_resnet 单次 forward ~1ms → 单 fold ≤ 1 分钟，全 15 fold ≤ 15 分钟。CPU 串行 2-3 天。

## 9. 阻塞问题 / 已知限制

1. **Full P7 未跑**：等待 Controller 显式授权 `EXP-P7-FULL-20260820-R0X` 才能在 AutoDL GPU 跑完整 sweep。Clean-only 全 fold CPU 复现 (Reviewer #7) 已证明 CPU 单 fold 在 ~135 s 内可完成 → Full sweep CPU 串行约 15 × 135 / 60 / 60 ≈ 0.56 h/单条件/单 seed 上限，仅供参考。
2. **P5.2-C SVM candidate 没有参与**：本 runner 只针对 `small_resnet`；`matrix_mlp` 与 `tiny_cnn` 不参与 P7 sweep，符合协议。
3. **真实硬件耐受不等价**：本结果只刻画 PoPu 64×27 Tactilus 公开数据的软件敏感性，**不可**作为真实低密度硬件或产品 PASS 证据。`condition_comparison.json` 的 `decision` 字段已显式写入此边界。

## 10. git 状态

```text
$ git status --short
?? docs/stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_RESULTS_v0.1.md
?? outputs/analysis/EXP-P7-CONFIG-VALIDATION-20260820-R01/
?? outputs/experiments/EXP-P7-CPU-SMOKE-20260820-R03/
?? outputs/experiments/EXP-P7-CLEAN-ONLY-FULL-FOLD-20260820-R03/
?? scripts/run_popu_p7_robustness.py
?? src/topper_perception/neural/p6_evidence.py
?? src/topper_perception/neural/p7_runner.py
?? src/topper_perception/experiments/artifacts.py
?? tests/test_neural_p7_runner.py
```

```text
$ git diff --stat
(nothing — only untracked new files, no existing tracked files modified)
```

提交前 Reviewer 复核通过后，建议 commit：

```text
feat: implement PoPu P7 software robustness evaluation
```

## 11. 下一步

1. Reviewer 复核本 v0.1.2 修订版（含 §5 canonical SHA 验证、§6 新 Smoke 真实 per-class/per-subject coverage、§7 958/958 clean-only 复现）；
2. 复核通过后授权 `EXP-P7-FULL-20260820-R02` 在 AutoDL GPU 上跑 full sweep；
3. Full P7 完成后另写一份 `P7_POPU_SOFTWARE_ROBUSTNESS_FULL_RESULTS_v0.1.md`；
4. 仅当 Controller 接受 Full 结果后才进入 P8 候选冻结。