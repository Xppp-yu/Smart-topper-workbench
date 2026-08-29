# Stage Report: S2-B04A — SLP8 PM-only Architecture Expansion Mini Implementation+Smoke v0.1

**TASK-ID:** `TASK-SLP-B04A-IMPLEMENTATION-SMOKE-v0.1`
**Stage:** S2-B04A
**Date:** 2026-08-29（R02 修订完成；Codex Reviewer 独立复跑验收）
**Status:** `IMPLEMENTATION_SMOKE_ACCEPTED / RUNNER_INTEGRATION_NOT_STARTED`（**不得**声明 `GPU_MINI_AUTHORIZED` / `MINI_COMPLETE` / `B07_READY`）
**Branch:** `codex/task-slp-b04a-implementation-smoke-v0.1`
**Working tree:** 3 modified + 4 untracked（仅声明文件范围）
**Maintainer:** Mavis (MiniMax Code)

---

## 1. 执行摘要

在 B04A R03 协议冻结（Codex Reviewer 验收）的基础上，本任务把三个候选的实现落实并通过单元测试 + CPU 合成 Smoke 完成 smoke-level 验证：

1. **`slp8_small_unet_v0.1`**（incumbent）：保持 B04 实现的全部结构和语义；仅复用既有 `Slp8SmallUnet` 类。
2. **`slp8_resunet_lite_v0.1`**（新候选）：3 个 residual block 全部显式 `Conv2d 1x1` shortcut；`exact_parameter_count=120,809`；通过 `_ResidualBlock` 子模块暴露 `shortcut_conv` 以便测试断言。
3. **`slp8_deeplabv3plus_lite_v0.1`**（新候选 Option A）：1 pointwise + 4 atrous（rates 3/6/9/12，dilation=padding=rate，groups=1）+ 1 GAP = 6 分支 ASPP；每分支 16 通道；concat 96 → 32；`exact_parameter_count=53,449`；明确无 Xception / depthwise-separable。

**Codex Reviewer R02 ITERATE 修订（2026-08-29）：**

- 修正 shortcut 参数精确分解为 32 + 544 + 2112 = 2688（与 ResUNet - SmallUNet = 120,809 - 118,121 = 2,688 完全一致），删除"差 96 可接受"等不精确表述。
- Smoke 脚本新增 `--output` / `--force` / `--no-write` 参数：默认拒绝覆盖已存在输出，不得静默 `write_text` 覆盖；`--no-write` 用于 Codex Reviewer 的"无写 smoke"复跑。
- `test_access` 字段由"运行时计数 0"重写为"declarative_policy"——它是合同声明，不是计数证据。
- 报告 + 项目状态 + Backlog 一致更新：Codex Reviewer 独立复跑后阶段为 `IMPLEMENTATION_SMOKE_ACCEPTED / RUNNER_INTEGRATION_NOT_STARTED`；不得越过 RUNNER INTEGRATION 直接进入 B04A-MINI-RUN；GPU Mini 继续 `BLOCKED`。
- 明确记录"现有 B04 runner 拒绝 B04A config"——B04 runner 集成尚未完成，属于下一个独立 TASK。

---

## 2. 修改与新增文件

### 2.1 当前工作树（3 modified + 4 untracked）

| 状态 | 路径 | 说明 |
|---|---|---|
| Modified | `src/topper_perception/neural/slp8_region_models.py` | 追加 `RESUNET_LITE_VERSION` / `DEEPLABV3PLUS_LITE_VERSION` / `B04A_MAX_PARAMETERS` / `B04A_EXACT_PARAMETER_COUNTS` / `DEEPLABV3PLUS_LITE_*` 常量；追加 `_ResidualBlock` / `Slp8ResUnetLite` / `_AsppModule` / `Slp8DeepLabV3PlusLite` 类；追加 `create_slp8_resunet_lite` / `create_slp8_deeplabv3plus_lite` 工厂；注册两个新 builder；抽出共享 `_validate_input_tensor` / `_init_conv_kaiming_zero_bias` 工具；**不重写、不修改**既有 `Slp8TinyFcn` / `Slp8SmallUnet` |
| Modified | `docs/PROJECT_STATUS.md` | S2_B04A 行状态 = `IMPLEMENTATION_SMOKE_ACCEPTED`；记录 `RUNNER_INTEGRATION_NOT_STARTED` |
| Modified | `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` | B04A 段状态同上；下一 Gate 明确为 `TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1`（不是 B04A-MINI-RUN） |
| Untracked | `tests/test_b04a_implementation.py` | 79 个 B04A 聚焦实现测试 |
| Untracked | `scripts/smoke_b04a_implementation.py` | CPU/CUDA 合成 smoke；`--output` / `--force` / `--no-write` 参数；默认拒绝覆盖 |
| Untracked | `docs/tasks/TASK_SLP_B04A_IMPLEMENTATION_SMOKE_v0.1.md` | 任务合同 |
| Untracked | `docs/stage_reports/S2_B04A_IMPLEMENTATION_SMOKE_v0.1.md` | 本文件 |

附加产物：

| 路径 | 性质 | 说明 |
|---|---|---|
| `outputs/reports/b04a_implementation_smoke_v0.1.json` | 声明产物（git-ignored outputs 区内） | 由 `scripts/smoke_b04a_implementation.py` 写出；默认拒绝覆盖已存在文件 |
| `outputs/legacy_to_be_removed/` | 临时 R01/R02 探针 | Codex Reviewer 验收时确认未被 `.gitignore` 覆盖，已删除，不纳入交付 |

未触碰：

- `tests/test_slp8_region_models.py`、`tests/test_slp8_region_mini.py`、`tests/test_b04a_protocol_validator.py`、`tests/test_check_markdown_links.py`（保持向后兼容；B04A 新增断言统一放在 `test_b04a_implementation.py`）；
- `configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json`（不重写协议，只验证实现与之严格一致）；
- `src/topper_perception/io/slp8_training_table_freeze.py`（不读取 B01 真实训练表）；
- `src/topper_perception/neural/slp8_region_mini.py`（**不重写 B04 runner**；新候选的 Mini 集成属于下一个独立 TASK：`TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1`）；
- B01/B02/B04 历史数值或 EXP-ID；
- 未 commit / push / PR。

---

## 3. 短路参数数学（Codex ITERATE R02 修正）

`Slp8ResUnetLite` 相对 `Slp8SmallUnet` 的差全部来自三个 1x1 Conv2d shortcut（其它层完全相同）。每个 shortcut `Conv2d(in, out, kernel_size=1, stride=1, padding=0, bias=True)` 的参数 = `in*out*1*1 + out`：

| Residual block | in → out | 参数 |
|---|---|---|
| `enc1_resblock` | 1 → 16 | `1*16*1*1 + 16` = **32** |
| `enc2_resblock` | 16 → 32 | `16*32*1*1 + 32` = **544** |
| `bottleneck_resblock` | 32 → 64 | `32*64*1*1 + 64` = **2112** |
| **总和** | | **32 + 544 + 2112 = 2,688** |

验证：

- 实测 `ResUNet_lite.count_parameters() - SmallUNet.count_parameters() = 120,809 - 118,121 = 2,688`。
- 逐 shortcut 测量：`{'enc1_resblock': 32, 'enc2_resblock': 544, 'bottleneck_resblock': 2112}`，求和 = 2,688。
- 测试 `test_resunet_shortcut_params_sum_to_2688` 强制三方一致。

---

## 4. Smoke 脚本新行为（Codex ITERATE R02 修正）

`scripts/smoke_b04a_implementation.py` 的接口与不变量：

| 行为 | 触发 | 结果 |
|---|---|---|
| 默认写 | `python scripts/smoke_b04a_implementation.py` | 写入 `--output` 路径（默认 `outputs/reports/b04a_implementation_smoke_v0.1.json`） |
| 拒绝覆盖 | `--output` 已存在且未传 `--force` | 退出码 2，stderr 含 "Refusing to overwrite …"，**不**修改磁盘 |
| 强制覆盖 | `--force` | 退出码 0，写入 |
| 无写 smoke | `--no-write` | 退出码 0，stdout 打印单行 `B04A_SMOKE_NO_WRITE cpu_candidates=3 cuda_run=False all_cpu_ok=True`，**不**写任何文件 |

`test_access` 字段在 summary JSON 中由 `TEST_ACCESS_DECLARATION` 常量声明：

```json
"test_access": {
  "value": 0,
  "kind": "declarative_policy",
  "explanation": "The B04A implementation smoke does not import any B01 training table loader and does not invoke enable_test_access(...). The 0 is a static declaration, NOT a runtime count of TEST reads."
}
```

`kind = "declarative_policy"` 明确把 0 从"运行计数"重写为"合同声明"，避免误读为运行时证据。

---

## 5. 验证结果

### 5.1 B04A 实现测试（`tests/test_b04a_implementation.py`）

```
79 passed in 43.06s
```

包含（与原 75 + 4 新）：

- `TestB04ARegistry`（3）
- `TestB04AExactParameterCounts`（9）
- `TestB04AInputOutputShapes`（11）
- `TestB04AForwardFiniteAndBackward`（6）
- `TestResUnetLiteResidualBlocks`（5）
- `TestDeepLabV3PlusLiteAspp`（9）
- `TestB04ANoForbiddenLayers`（6）
- `TestB04AConfigForwardPlanConsistency`（6）
- `TestB04ACheckpointRoundtrip`（3）
- `TestB04ADeterministicSmoke`（3）
- `TestB04ANoTestAccess`（2）
- `TestB04AMaxParameterGuardrail`（2）
- `TestB04AModuleConstants`（5）
- `test_b04a_param_count_math_is_consistent`（1）
- **`test_resunet_shortcut_params_sum_to_2688`**（1，R02 新增）
- **`test_smoke_script_refuses_to_overwrite_existing_output`**（1，R02 新增）
- **`test_smoke_script_no_write_does_not_touch_disk`**（1，R02 新增）
- **`test_smoke_summary_records_test_access_as_declarative_policy`**（1，R02 新增）

### 5.2 协议验证器 + 链接检查

```
uv run python -m pytest tests/test_b04a_protocol_validator.py tests/test_check_markdown_links.py -q
→ 56 passed in 0.37s
```

### 5.3 B04A 涉及 markdown 相对链接检查

```
uv run python scripts/check_markdown_links.py docs\stage_reports\S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md docs\tasks
→ Files scanned: 5, Errors: 0 — LINK CHECK PASSED

uv run python scripts/check_markdown_links.py docs\tasks\TASK_SLP_B04A_IMPLEMENTATION_SMOKE_v0.1.md docs\stage_reports\S2_B04A_IMPLEMENTATION_SMOKE_v0.1.md
→ Files scanned: 2, Errors: 0 — LINK CHECK PASSED
```

### 5.4 B04 回归

- `tests/test_slp8_region_models.py`：38 / 38 通过；
- `tests/test_slp8_region_mini.py`（已选）`TestSmallUnetArchitecture` / `TestCandidateRegistry` / `TestBuildSynthetic` / `TestPredict`：**15 / 15 通过**。

### 5.5 B04A 协议验证器

```
uv run python scripts/validate_b04a_protocol.py configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json
OKs: 30, Errors: 0 — VALIDATION PASSED
```

### 5.6 Smoke 脚本新行为（Codex R02 复跑记录）

```text
$ python scripts\smoke_b04a_implementation.py --no-write
B04A_SMOKE_NO_WRITE cpu_candidates=3 cuda_run=False all_cpu_ok=True
exit=0,  无任何文件被写入

$ python scripts\smoke_b04a_implementation.py
Wrote .../b04a_implementation_smoke_v0.1.json
exit=0

$ python scripts\smoke_b04a_implementation.py
ERROR: output file already exists: .../b04a_implementation_smoke_v0.1.json. Refusing to overwrite. Pass --force to allow overwrite, or pass --output to a different path.
exit=2,  磁盘文件保持不变

$ python scripts\smoke_b04a_implementation.py --force
Wrote .../b04a_implementation_smoke_v0.1.json
exit=0
```

### 5.7 B04 Runner 与 B04A Config 的契约（Codex 独立复跑）

```text
$ uv run python -c "import json; from topper_perception.neural.slp8_region_mini import B04_CANDIDATE_NAMES, validate_mini_config; cfg=json.load(open('configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json')); validate_mini_config(cfg)"

B04 runner's B04_CANDIDATE_NAMES: ['slp8_tiny_fcn_v0.1', 'slp8_small_unet_v0.1']
B04A config candidates          : ['slp8_small_unet_v0.1', 'slp8_resunet_lite_v0.1', 'slp8_deeplabv3plus_lite_v0.1', 'slp8_segformer_b0_v0.1']
B04A names missing from B04 runner's allowed set: ['slp8_resunet_lite_v0.1', 'slp8_deeplabv3plus_lite_v0.1', 'slp8_segformer_b0_v0.1']

EXPECTED: B04 mini refused the B04A config with ConfigValidationError:
  config.task_id 'TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1' != expected
  'TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1'
```

结论：**现有 B04 runner 拒绝 B04A config**（B04 runner 的 `B04_CANDIDATE_NAMES` 只含 `slp8_tiny_fcn_v0.1` / `slp8_small_unet_v0.1`；B04A 多了 `slp8_resunet_lite_v0.1` / `slp8_deeplabv3plus_lite_v0.1` 两个候选名，加上 `slp8_segformer_b0_v0.1` 的 DEFERRED 占位）。这是 `B04_CANDIDATE_NAMES != B04A candidates` 的 fail-closed 行为，是设计上预期的。**B04A runner integration 属于下一个独立 TASK，本任务未实现 runner 集成。**

### 5.8 Python compile / import + git

```
import: topper_perception.neural.slp8_region_models, topper_perception.neural.slp8_region_mini  → OK
py_compile: src/.../slp8_region_models.py, tests/test_b04a_implementation.py, scripts/smoke_b04a_implementation.py  → 0 errors
git diff --check  → 干净
git status --short --branch  → 3 modified + 4 untracked (declared files only)
```

---

## 6. Codex 独立检查记录（R02 ACCEPT 复跑）

| 检查项 | 命令 | 结果 |
|---|---|---|
| 全部 B04A 实现 + 协议 + 链接 + B04 模型测试 | `uv run python -m pytest tests/test_b04a_implementation.py tests/test_b04a_protocol_validator.py tests/test_check_markdown_links.py tests/test_slp8_region_models.py -q` | **173 passed** |
| B04 mini 回归（SmallUNet / registry / synthetic / TestPredict） | `uv run python -m pytest tests/test_slp8_region_mini.py -q -k "TestSmallUnetArchitecture or TestCandidateRegistry or build_synthetic_dataset or TestPredict"` | **15 passed** |
| B04A 协议验证器 | `uv run python scripts/validate_b04a_protocol.py configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json` | **30 OKs / 0 errors** |
| B04A 涉及 markdown 相对链接 | `uv run python scripts/check_markdown_links.py docs\stage_reports\S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md docs\tasks` | **0 errors** |
| B04A 涉及 markdown 相对链接（新建） | `uv run python scripts/check_markdown_links.py docs\tasks\TASK_SLP_B04A_IMPLEMENTATION_SMOKE_v0.1.md docs\stage_reports\S2_B04A_IMPLEMENTATION_SMOKE_v0.1.md` | **0 errors** |
| CPU no-write smoke（三候选） | `uv run python scripts/smoke_b04a_implementation.py --no-write` | `cpu_candidates=3 cuda_run=False all_cpu_ok=True` |
| Smoke 默认拒绝覆盖 | 第二次运行相同 `--output` 不传 `--force` | 退出码 2，文件字节不变 |
| B04 runner 拒绝 B04A config | `validate_mini_config(json.load(open(B04A config)))` | `ConfigValidationError: task_id mismatch`；**B04A runner integration 未完成** |
| `git diff --check` | — | 干净 |
| `python -m py_compile`（新文件） | — | 0 errors |

---

## 7. 已验证事实

1. 三个候选 `count_parameters()` 与 R03 冻结 `exact_parameter_count` 三方一致（实现 / 模块字典 / 协议验证器递归）。
2. **shortcut 参数精确分解**：`enc1=32 + enc2=544 + bottleneck=2112 = 2688 = ResUNet - SmallUNet`（R02 修正）。
3. forward 输出 shape = `[N, 9, 192, 84]`，dtype = `torch.float32`，对 batch=1,2,4 成立；所有 forward 输出 finite。
4. backward 路径可走通，loss finite，至少一个参数张量在单步 AdamW 后发生变化。
5. ResUNet 三个 residual block 的 `shortcut_conv` 全部为 `Conv2d(1x1, s=1, p=0, bias=True, groups=1)`，与 config `forward_plan` 完全一致；运行时 `main.shape == shortcut.shape` 成立。
6. DeepLabV3+-lite 全部 `nn.Conv2d.groups == 1`（4 atrous + 1 pointwise + 1 post-concat + 1 low-level + 2 decoder + 1 final = 10 个 Conv2d 子模块，均 `groups==1`）。
7. ASPP 4 atrous 分支 `dilation == padding == atrous_rate`，rate ∈ {3, 6, 9, 12}；6 分支运行时实际拼接为 `concat.shape[1] == 96`。
8. MODEL_REGISTRY 暴露 4 个 builder；未知 builder 名抛 `KeyError("Unknown model …")`。
9. 配置文件 `slp8_pm_architecture_expansion_mini_v0.1.json` 的 `forward_plan` 与实现完全一致（`TestB04AConfigForwardPlanConsistency` 6 个测试全通过）。
10. `torch.save` + `torch.load` 后所有三模型前向输出与 save 前 `torch.equal`。
11. 同 seed（`torch.manual_seed(123)`）两次构造 + 前向 → `torch.equal`。
12. TEST 访问 = 0（`TestB04ANoTestAccess` 双重验证：源码扫描 + 注册表查询）；smoke summary 的 `test_access` 字段声明为 `declarative_policy` 而非运行时计数。
13. B04 `Slp8SmallUnet` 行为未变（38 + 15 个原测试通过；exact_parameter_count 仍为 118,121）。
14. 协议验证器 + 链接检查器 56 项 + B04A 实现 79 项 = **135 项全部通过**（R02 后数量）。
15. **现有 B04 runner 拒绝 B04A config**（`validate_mini_config` 抛 `ConfigValidationError`）——这是 B04 runner 集成尚未完成的证据；属于下一个独立 TASK。
16. **Smoke 脚本默认拒绝覆盖已存在输出**；`--no-write` 模式不写任何文件；`test_access` 是合同声明，不是运行时计数。
17. 整次工作树未 commit / push / 创建 PR。

---

## 8. 推断

1. 三候选 CPU 合成 Smoke 在本机 `torch==2.13.0+cpu` 上的 wall time 远低于 B04A R03 预算（45 min/candidate / 135 min total），实现层链路可被快速复跑。
2. DeepLabV3+-lite（53,449 参数）相对 SmallUNet（118,121）和 ResUNet-lite（120,809）参数显著更少，Option A plain atrous 路径在 R03 限定的"无 Xception / 无 depthwise-separable"约束下给出了最低参数的合格候选；这是设计选择，不是性能排名。
3. ResUNet-lite 相对 SmallUNet 多 2,688 参数；精确对应 3 个 1x1 Conv2d shortcut 的 `in*out*1*1 + out` 求和，**没有**其它额外参数差异（其它所有层与 SmallUNet 严格相同）。
4. CUDA Smoke 未运行不构成"实现不可移植"或"实现未完成"的证据；CPU 端 forward + backward + checkpoint roundtrip + same-seed determinism 全部通过已证明模型本身可在 autograd 框架内完整运行。
5. 现有 B04 runner 拒绝 B04A config 是 fail-closed 行为：它能识别候选名不在 `B04_CANDIDATE_NAMES` 中并抛错；这是 B04 runner 设计的硬性约束，不是缺陷。

---

## 9. 未验证 / NOT RUN

1. **真实 GPU 训练 / Mini / Full**：属于 `TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1`（runner 集成 smoke）以及未来 Owner 授权后的 `B04A-MINI-RUN`。**本任务不实现 runner 集成，也不启动 Mini。**
2. **CUDA Smoke**：`torch==2.13.0+cpu` CPU-only build，`torch.cuda.is_available() == False` → 显式 `NOT_RUN`，不报失败。
3. **B04A Runner 集成**：现有 B04 runner 拒绝 B04A config（已证明）；B04A 在 `B04_CANDIDATE_NAMES` / task_id / 配置结构上的扩展是下一个独立 TASK 的范围。
4. **真实 B01 训练表读取**：`scripts/smoke_b04a_implementation.py` 不加载 B01 训练表；模型实现层不接触 B01 `slp8_training_table_freeze`。
5. **真实 TEST 读取**：未发生（`TestB04ANoTestAccess` 双重验证 + smoke summary `test_access.kind = "declarative_policy"`）。
6. **B04A R03 协议 / 配置文件的修改**：本任务不重写冻结合同，只验证实现与之严格一致。
7. **B07 启动**：仍 `BLOCKED_BY_B04A`，未被本任务开启。
8. **`ruff` / `pre-commit` 等代码风格工具**：`ruff` 在当前环境未安装，记为 `NOT RUN (tool unavailable)`；本任务未运行 ruff。

---

## 10. 限制与禁止结论

**限制**

1. 当前阶段名 = `IMPLEMENTATION_SMOKE_ACCEPTED / RUNNER_INTEGRATION_NOT_STARTED`；这只接受实现与合成 Smoke，**不得**标记为 `GPU_MINI_AUTHORIZED` / `MINI_COMPLETE` / `B07_READY`。
2. 协议冻结 ≠ Mini / Full 完成。
3. 标签为 `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` / `source_review_status=NOT_REVIEWED`；danaLab / uncover only。
4. 真实 GPU Mini 仍需 Owner 单独授权 + `TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1` 完成。
5. CUDA Smoke 在本机 CPU-only build 上未跑，GPU 可移植性需后续真实 GPU 环境。

**禁止结论**

- ❌ B04A Mini 完成；新候选优于 SmallUNet；架构比较形成最终排名
- ❌ 适用于产品、硬件、舒适性、医学、整夜稳定性、气囊控制
- ❌ SegFormer 临时纳入；TEST 结果可见或可推断
- ❌ 压力值是 kPa；标签是人工像素级标注
- ❌ runner integration 已完成；可直接进入 B04A-MINI-RUN

---

## 11. Reviewer checklist

- [x] 三个候选实现存在并可从 `MODEL_REGISTRY` 加载
- [x] `count_parameters()` = 118,121 / 120,809 / 53,449（实现 / 模块字典 / 协议验证器三方一致）
- [x] shortcut 精确分解 32 + 544 + 2112 = 2,688 = ResUNet - SmallUNet
- [x] forward `[N, 9, 192, 84]`、`torch.float32`、finite（batch=1,2,4）
- [x] forward finite、backward finite、单步 AdamW 后参数变化
- [x] ResUNet 三个 residual block 全部 `Conv2d 1x1` shortcut
- [x] ResUNet residual Add shape / channel 运行时一致
- [x] DeepLab 6 分支 ASPP、concat=96、post-concat 96 → 32
- [x] DeepLab 全部 `Conv2d.groups == 1`、dilation=padding=rate ∈ {3,6,9,12}
- [x] `MODEL_REGISTRY` 含 4 builder；未知名 `KeyError`
- [x] config `forward_plan` 与实现一致（variant、aspp_settings、residual block、SegFormer DEFERRED）
- [x] checkpoint save/reload `torch.equal`
- [x] same-seed 复现
- [x] TEST = 0（源码扫描 + 注册表查询 + `test_access.kind = "declarative_policy"`）
- [x] SmallUNet 38 + 15 个原测试无回归
- [x] `tests/test_b04a_implementation.py` 79 / 79
- [x] 核心套件（实现 + 协议验证器单测 + 链接单测 + B04 模型）= 173 / 173
- [x] `scripts/validate_b04a_protocol.py` 30 OKs / 0 errors
- [x] B04A 涉及 markdown 链接 0 errors
- [x] `python -m py_compile` + `import` 通过
- [x] `git diff --check` 干净
- [x] 临时未跟踪探针已删除；仅保留声明范围内的交付文件
- [x] 未 commit / push / PR
- [x] CPU Smoke 通过；CUDA Smoke 显式 `NOT_RUN` 并附原因
- [x] Smoke 脚本默认拒绝覆盖；`--no-write` 不写盘；`--force` 显式允许覆盖
- [x] `test_access` 声明为 `declarative_policy` 而非运行时计数
- [x] 现有 B04 runner 拒绝 B04A config（已实测）；B04A runner integration 未完成
- [x] Codex Reviewer 独立验收后阶段名 = `IMPLEMENTATION_SMOKE_ACCEPTED / RUNNER_INTEGRATION_NOT_STARTED`
- [x] 下一 Gate = `TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1`（不是 B04A-MINI-RUN）
- [x] GPU Mini 继续 `BLOCKED`

---

## 12. 当前 git status

```
Branch:    codex/task-slp-b04a-implementation-smoke-v0.1
HEAD:      a2c502f33b7eb95c0e8408a90a7f222077947f64  (base; new files uncommitted)
Status:    dirty
Ahead/behind origin/main: 0 / 0
变更（仅声明文件）：
  M src/topper_perception/neural/slp8_region_models.py
  M docs/PROJECT_STATUS.md
  M docs/SLP_AGENT_TASK_BACKLOG_v0.1.md
  ?? scripts/smoke_b04a_implementation.py
  ?? tests/test_b04a_implementation.py
  ?? docs/tasks/TASK_SLP_B04A_IMPLEMENTATION_SMOKE_v0.1.md
  ?? docs/stage_reports/S2_B04A_IMPLEMENTATION_SMOKE_v0.1.md  (本文件)

声明产物（git-ignored outputs 区内）：
  outputs/reports/b04a_implementation_smoke_v0.1.json
```

`outputs/reports/...` 属于 `.gitignore` 已声明的产物区。`outputs/legacy_to_be_removed/...` 并未被 `.gitignore` 覆盖，Codex Reviewer 已删除其中两份一次性未跟踪探针。

---

## 13. 下一 Gate

1. **`TASK-SLP-B04A-RUNNER-INTEGRATION-SMOKE-v0.1`**（独立 TASK-ID，由 Owner 单独授权后启动）：在 `src/topper_perception/neural/slp8_region_mini.py` 扩展 `B04_CANDIDATE_NAMES` / 候选裁决 / 配置 schema 以支持 B04A 三候选的 Mini 训练；**不得**进入 GPU Mini / Full。
2. **`B04A-MINI-RUN`**：在 runner 集成完成且 Owner 授权后，Runner 在真实 GPU 上跑 3 seeds × 3 候选 Mini 训练；遵守 B04A R03 资源预算与 `all_seeds_must_succeed` Gate。
3. **`B04A-REVIEW`**：Codex 独立审查真实 Mini 产物、指标、failure cases 和 identity。
4. **B07 解锁**：B04A 实际 Mini 经 Reviewer 接受并冻结最多 1–2 个候选后，B07 才能开始协议冻结。

---

**交付版本：** v0.1-R02-ACCEPTED
**生成时间：** 2026-08-29
**维护者：** Mavis (MiniMax Code)
