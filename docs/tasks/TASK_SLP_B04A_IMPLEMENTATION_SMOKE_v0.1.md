# TASK-SLP-B04A-IMPLEMENTATION-SMOKE-v0.1

**Status:** `IMPLEMENTATION_SMOKE_ACCEPTED`
**Date:** 2026-08-29
**Branch:** `codex/task-slp-b04a-implementation-smoke-v0.1`
**Stage:** S2-B04A (implementation + CPU/CUDA Smoke)

## Objective

实现并验证 B04A R03 协议冻结的三个 SLP8 PM-only 区域分割候选：

1. `slp8_small_unet_v0.1`（incumbent，不修改结构与语义）— exact 118,121
2. `slp8_resunet_lite_v0.1`（新候选，3 个 residual block，全部 1x1 Conv2d shortcut）— exact 120,809
3. `slp8_deeplabv3plus_lite_v0.1`（新候选，Option A plain atrous，6 分支 ASPP）— exact 53,449

只做模型实现、注册、配置合同接入、单元测试和 CPU/最小合成 Smoke。**不运行真实 GPU Mini、不读取真实 TEST、不启动 B07。**

## Why now

B04A R03 已通过 Codex Reviewer 验收，但模型本身尚未实现。本任务把"协议冻结的形状"落实为"可加载、可前向、可反向、可 checkpoint、可 same-seed 复现"的实现，并通过单元测试 + CPU 合成 Smoke 证明实现满足 R03 合同。

## Prerequisites

- B01 训练表冻结合同保持不变（A06 SHA `024f5abe...`、freeze manifest SHA `3c789995...`）；
- B02 baseline `0.205644` 历史值不变；
- B04 R05 结果（SmallUNet `0.439625` / TinyFCN `0.051631`）保持不变；
- SegFormer-B0 继续 `DEFERRED`；
- B04A 协议（`TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1` R03 验收版）保持不变；
- B07 仍 `BLOCKED_BY_B04A`；
- 本任务不修改 B01/B02/B04 任何历史数值或 EXP-ID。

## Hard implementation contract

### 1. `slp8_small_unet_v0.1`（incumbent）

- 不改变 `Slp8SmallUnet` 的现有结构和语义；
- exact parameter count 必须为 **118,121**（保持 B04 R05 行为）；
- 输出必须为 `[N, 9, 192, 84]`；
- 沿用 B04 已注册的 `create_slp8_small_unet` 工厂与 `Slp8SmallUnet` 类，不复制不重构；
- 原有 `tests/test_slp8_region_models.py` 和 `tests/test_slp8_region_mini.py` 中针对 SmallUNet 的断言不得回归。

### 2. `slp8_resunet_lite_v0.1`

严格按 R03 forward_plan 实现：

- exact parameter count = **120,809**（由 `scripts/validate_b04a_protocol.py` 递归计算并匹配）；
- 三个 residual block（`enc1_resblock`、`enc2_resblock`、`bottleneck_resblock`）；
- 所有通道变化的 shortcut 显式为 `Conv2d(1x1, stride=1, padding=0, bias=True)`；
- 禁止 Identity shortcut 用于通道不匹配；
- residual Add 两侧 `main_output_channels == shortcut_output_channels` 且 `main_output_shape == shortcut_output_shape`；
- decoder 使用冻结的 bilinear interpolation、concat、channel widths（与 SmallUNet 共享结构）；
- 输出 `[N, 9, 192, 84]`。

### 3. `slp8_deeplabv3plus_lite_v0.1`

严格实现 R03 Option A（plain atrous Conv2d）：

- `variant = "option_A_plain_atrous_Conv2d"`；
- exact parameter count = **53,449**；
- 禁止 Xception / depthwise-separable；
- 所有 Conv2d `groups=1`（已由验证器强制）；
- ASPP 4 atrous 分支 `dilation=atrous_rate`、`padding=atrous_rate`、`groups=1`；
- 1×1 pointwise 分支 + 4 atrous 分支 + GAP 分支 = 6 分支；
- 每分支输出 16 通道；
- ASPP concat = 96 通道；
- post-concat `Conv2d 96 → 32 → ReLU`；
- decoder：low-level projection（`Conv2d 16→16`）+ bilinear recovery + Concat 48 通道 + 两层 `Conv2d 48→32→32`；
- 输出 `[N, 9, 192, 84]`。

### 4. 公共合同

- 输入 `[N, 1, 192, 84]`，模型输入 `torch.float32`；
- `n_classes = 9`；
- 不使用预训练权重；
- augmentation = `none`；
- 初始化严格遵守 `Kaiming normal (fan_out, relu)` + 零 bias；
- 不修改阈值、seeds、训练预算、Gate、候选名单、SegFormer 状态；
- SegFormer 继续 `DEFERRED`，不实现；
- 不复制或新建另一套不兼容的训练链路；
- 复用现有 B04 model registry、checkpoint、determinism、runner 基础设施。

## Required tests

下列测试在 `tests/test_b04a_implementation.py` 中实现并全部通过（详见阶段报告）：

1. **精确参数量**：SmallUNet=118,121 / ResUNet-lite=120,809 / DeepLabV3+-lite=53,449（由 `count_parameters()` 实测 + 与 `B04A_EXACT_PARAMETER_COUNTS` 字典对照 + 与 `validate_b04a_protocol.py` 推导对照，三方一致）；
2. **输入/输出 shape**：所有 batch size（1、2、4）均输出 `[N, 9, 192, 84]`；
3. **batch size 1 和 >1**；
4. **forward finite**：`torch.isfinite(y).all()` 对所有 batch size 成立；
5. **backward 后参数发生变化**：单步 AdamW 后至少一个参数张量发生更改；
6. **ResUNet residual Add shape/channel**：`main_output_channels == shortcut_output_channels`、`main_output_shape == shortcut_output_shape`，且运行时 `main.shape == shortcut.shape`；
7. **ResUNet shortcut 是 1×1 Conv2d**：`kernel_size=(1,1)`、`stride=(1,1)`、`padding=(0,0)`、`bias=True`、`groups=1`；
8. **DeepLab ASPP 六分支** + concat = 96 通道（运行时实际拼接 6 个 16 通道张量后 `concat.shape[1] == 96`）；
9. **DeepLab dilation/padding/groups**：所有 Conv2d `groups=1`；4 atrous 分支 `dilation == padding == atrous_rate` ∈ {3,6,9,12}；
10. **模型 registry 版本名**：`MODEL_REGISTRY` 暴露 `slp8_small_unet_v0.1` / `slp8_resunet_lite_v0.1` / `slp8_deeplabv3plus_lite_v0.1`；未知名 `KeyError`；
11. **config forward_plan 一致性**：`configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json` 中三候选的 `forward_plan` 与实现结构完全一致（exact_parameter_count、residual block、ASPP 配置、variant 标签、SegFormer DEFERRED 全部由 `TestB04AConfigForwardPlanConsistency` 验证）；
12. **checkpoint save/reload 一致性**：`torch.save` + `torch.load` 后 `torch.equal(y1, y2)` 对所有三个模型成立；
13. **deterministic same-seed CPU 合成 Smoke**：相同 seed 两次构造 + 前向得到 `torch.equal`；
14. **TEST=0**：`slp8_region_models.py` 不导入 B01 `slp8_training_table_freeze` / `enable_test_access` / `TestLeakageError` / `compute_class_stats` / `slp8_8region_pressure_dataset`；smoke 脚本不调用任何 B01 训练表接口；
15. **SmallUNet 回归**：38 个 `tests/test_slp8_region_models.py` 测试 + 15 个 `tests/test_slp8_region_mini.py`（SmallUNet / 候选 registry / build_synthetic_dataset / TestPredict）全部通过。

## Smoke scope

- **CPU Smoke**（必跑）：3 候选分别 forward + backward + 一步 AdamW + checkpoint roundtrip + same-seed determinism；
- **CUDA Smoke**（条件）：`torch.cuda.is_available() == False`（CPU-only `torch==2.13.0+cpu` build）→ 显式记录 `NOT_RUN` 并附原因，**不得报失败**；
- **不读取 B01 真实 freeze tables**（除已加载的 `config_sha256` / `data_manifest_sha256` 字符串之外）；
- **不运行完整 epoch 训练**；
- **不运行真实 GPU Mini**；
- **不生成正式 EXP-ID 研究结果**。

## Files allowed to change in this implementation stage

- `src/topper_perception/neural/slp8_region_models.py`（追加 ResUNet-lite / DeepLabV3+-lite + 注册；不重写已有类）；
- `tests/test_b04a_implementation.py`（新增）；
- `scripts/smoke_b04a_implementation.py`（新增）；
- `docs/tasks/TASK_SLP_B04A_IMPLEMENTATION_SMOKE_v0.1.md`（本文件）；
- `docs/stage_reports/S2_B04A_IMPLEMENTATION_SMOKE_v0.1.md`（新增）；
- `docs/PROJECT_STATUS.md`（更新 S2_B04A 行）；
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`（更新 B04A 条目）。

## Out of scope

- 修改 `tests/test_slp8_region_models.py` 与 `tests/test_slp8_region_mini.py` 的现有断言（保持向后兼容；新增 B04A 测试只追加到 `tests/test_b04a_implementation.py`）；
- 修改 B01/B02/B04 任何历史数值或 EXP-ID；
- 修改协议（`TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1`）或配置（`slp8_pm_architecture_expansion_mini_v0.1.json`）——本任务不重写冻结合同，只验证实现与之严格一致；
- 启动 B07；
- 运行真实 GPU Mini；
- commit / push / PR（除非 Owner 后续明确授权）。

## Acceptance criteria

- 三个候选 `count_parameters()` 与 R03 冻结的 `exact_parameter_count` 完全一致（118,121 / 120,809 / 53,449）；
- `tests/test_b04a_implementation.py` 75 个单元测试全部通过；
- `tests/test_slp8_region_models.py` 38 个原有单元测试全部通过（无回归）；
- `tests/test_slp8_region_mini.py` 的 15 个 SmallUNet / 候选 registry / build_synthetic / TestPredict 单元测试全部通过；
- `scripts/validate_b04a_protocol.py` 对 B04A 配置文件返回 30 OKs / 0 errors；
- `tests/test_b04a_protocol_validator.py` + `tests/test_check_markdown_links.py` 共 56 个测试全部通过；
- B04A 涉及的 markdown 文档相对链接检查 `0 errors`；
- `scripts/smoke_b04a_implementation.py` 在 CPU 上对三候选完成 forward + backward + checkpoint roundtrip + same-seed determinism；CUDA Smoke 显式记录 `NOT_RUN`（CPU-only torch build）；
- `outputs/reports/b04a_implementation_smoke_v0.1.json` 写出；
- `git diff --check` 干净；
- `git status --short --branch` 列出受控变更：`M src/topper_perception/neural/slp8_region_models.py`、`?? scripts/smoke_b04a_implementation.py`、`?? tests/test_b04a_implementation.py`，以及新增的 `docs/tasks/`、`docs/stage_reports/` 文档；
- 全部数据合同（A06 SHA、freeze manifest SHA、provenance、source_review_status、uncover、danaLab、`raw_pmarray_response`、TEST=0）保持不变；
- 协议验证器 + 链接检查器 = 56 项已运行 + 75 项 B04A 实现测试 = 131 项新测试全部通过；
- 整次工作树未 commit / 未 push / 未创建 PR。

## Prohibited conclusions

- ❌ 文档/实现就绪不等于 Mini / Full 候选完成；
- ❌ B04A 不能改写 B04 历史结果；
- ❌ 不能宣称任何候选优于另一个候选；
- ❌ 任何 Mini/Full 的相对性能排名都未在此阶段产生；
- ❌ 不外推到产品、硬件、舒适性、医学、整夜稳定性或气囊控制；
- ❌ SegFormer 不被纳入；
- ❌ TEST 结果不可见也不可推断；
- ❌ 压力值不是 kPa；
- ❌ 标签不是人工像素级标注；
- ✅ Codex Reviewer 可在独立验收后标记 `IMPLEMENTATION_SMOKE_ACCEPTED`；该状态不得解释为 `RUNNER_INTEGRATION_COMPLETE` / `GPU_MINI_AUTHORIZED` / `MINI_COMPLETE` / `B07_READY`。
