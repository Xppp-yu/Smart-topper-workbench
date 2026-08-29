# TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1

**状态：** `PROTOCOL_ACCEPTED / IMPLEMENTATION_NOT_STARTED`（R03，2026-08-29；Codex Reviewer 已验收）

## Objective

冻结 SLP8 PM-only 受控架构扩展 Mini（B04A）的完整协议。**不实现模型，不运行训练/Smoke/Mini/Full，不使用 GPU，不读取 TEST。**

## Why now

B04 只比较了 TinyFCN 与 SmallUNet，SmallUNet 在 B04 R05 达到 0.439625，但该比较不足以支持广泛的架构选择。B04A 引入 residual CNN（ResUNet-lite）和 atrous/multi-scale CNN（DeepLabV3+-lite），在受控 Mini 中最多保留 1–2 个候选，为 B07 Full 协议提供更全面的架构假设覆盖。

## Prerequisites

- B01 冻结训练表（SHA `024f5abe...`）保持不变
- B02 非学习 baseline `0.205644` 保持历史值
- B04 R05 结果保持不变：SmallUNet `0.439625`（FEASIBLE）、TinyFCN `0.051631`（NOT_FEASIBLE）
- B07 保持 `BLOCKED_BY_B04A`

## Candidate hypotheses

| 候选 | 架构假设 | 角色 | 状态 | exact_parameter_count | 增强 |
|---|---|---|---|---|---|
| `slp8_small_unet_v0.1` | 经典 encoder-decoder + skip | incumbent | B04 实现 | 118,121 | `none` |
| `slp8_resunet_lite_v0.1` | Residual CNN（1x1 Conv2d shortcut） | new_candidate | NOT_IMPLEMENTED | 120,809 | `none` |
| `slp8_deeplabv3plus_lite_v0.1` | Atrous / multi-scale（Option A: plain atrous） | new_candidate | NOT_IMPLEMENTED | 53,449 | `none` |
| `slp8_segformer_b0_v0.1` | Transformer global attention | **DEFERRED** | NOT_IMPLEMENTED | N/A | N/A |

**训练合同一致性（R02 关键约束）：**所有三个有效候选使用完全相同的训练合同。

**R03 关键约束：**
- ResUNet 三个 residual block 全部 shortcut = `Conv2d 1x1, k=1, s=1, p=0, bias=true`，无 Identity / extra conv。
- DeepLabV3+-lite 选 **Option A**（plain atrous Conv2d），无 Xception / depthwise-separable。
- `exact_parameter_count` 由验证器递归计算 Conv2d 参数并匹配。
- 验证器有 5 大类 R03 检查（residual Add / Concat / ASPP / depthwise groups / exact param count）+ 至少 1 个故障注入测试。

**SegFormer-B0 纳入条件（全部必须满足）：**
1. 单通道（1-ch）输入适配方案明确冻结
2. 输入 resize/padding 策略明确冻结
3. 预训练权重使用政策与 CNN 候选等价处理
4. 增强策略与 CNN 候选一致
5. Optimizer 和 lr policy 作为候选定义的一部分冻结
6. 参数/显存/wall-time tier 独立明确指定
7. Augmentation 与预训练 augmentation 解耦

**DEFERRED 意味着：**不得在看到 Mini 结果后临时加入。

## Protocol items frozen (R03)

1. **候选名称、版本、架构假设、resource tier**：精确机器可读定义。
2. **三个候选训练合同一致**：seeds、optimizer、early stopping、max_epochs、augmentation 全部相同。
3. **三个候选 `exact_parameter_count`**：118,121 / 120,809 / 53,449，由验证器递归计算并匹配。
4. **单通道输入形状 `[192, 84]`**：`pressure_storage_dtype = float64`，`model_input_dtype = float32`，转换位置明确。
5. **初始化 `Kaiming normal (fan_out, relu)`、无预训练**：精确冻结。
6. **增强范围**：**所有候选 `none`**。
7. **Loss `CrossEntropyLoss` + TRAIN-only `1/sqrt(pixel_ratio)` class weights**：mean 排除 background，clip [0.25, 4.0]，background weight = 1.0。
8. **Optimizer `AdamW(lr=0.001, wd=0.0001)`、无 scheduler**。
9. **max_epochs=30、min_epochs=5、patience=4**。
10. **Seeds `[42, 123, 2026]`、汇总方式 `macro_iou_mean_across_seeds`**。
11. **Feasibility threshold `= B02 0.205644 + margin 0.15 = 0.355644`，比较符 `>=`**。
12. **Class collapse 定义**（per seed → candidate 级联 INFEASIBLE）。
13. **Worst-subject guardrail**（per seed → candidate 级联 INFEASIBLE；前瞻性 policy floor，无 B02/B04 数学推导）。
14. **Per-region guardrail**（per seed → candidate 级联 INFEASIBLE）。
15. **Primary metrics 不含 background**；background 单独报告。
16. **Per-region/posture/subject/centroid 指标完整报告**。
17. **Near-tie margin `|diff| < 0.02` → 优先更简单模型**。
18. **决策规则 0/1/2/3_feasible 全部定义**；3_feasible → Top 2 by macro_iou_mean。
19. **最多 1–2 个候选晋级**。
20. **参数上限**：SmallUNet 150,000；新候选 300,000。
21. **Peak CUDA memory 8,192 MB；wall time 45 min/candidate（3 seeds 累计）+ 135 min total**；resume 保留 candidate-level 和 total-level 累计。
22. **Checkpoint/resume/reload determinism**：B04 Runner 基础设施继承。
23. **产物 identity 字段按文件类型分别精确冻结**。
24. **失败/停止/完成状态**：DONE/STOPPED/FAILED，精确冻结。
25. **Reviewer 独立复算范围**：精确冻结。
26. **架构完全冻结**（R03）：
    - **ResUNet-lite**: 每个 residual block 显式 `Conv2d 1x1, k=1, s=1, p=0, bias=true` shortcut；`main_output_channels == shortcut_output_channels`；`main_output_shape == shortcut_output_shape`。
    - **DeepLabV3+-lite (Option A)**: `variant: option_A_plain_atrous_Conv2d`；所有 Conv2d `groups=1`；ASPP 4 个 atrous 分支 `dilation=atrous_rate, padding=atrous_rate`；6 分支全部 16 ch；Concat 96 ch；post_concat 96→32。
27. **验证器 + 测试**：27 R02 + 23 R03 = 50 个验证器测试 + 6 链接检查 = 56 个全部通过。

## Gate families

### Hard fail-closed gates (per seed)

- subject overlap、manifest/split/hash mismatch
- 任何 TEST 读取（rows/labels/onehot）
- NaN/Inf、OOM、参数未更新、未达到 min_epochs
- Class collapse（零预测）
- Worst-subject IoU < 0.20（前瞻性 floor）
- Per-region IoU < 0.05（任一前景区）
- Checkpoint reload max_abs_diff > 0
- Prediction hash mismatch after reload
- 超过参数/显存/wall-time 预算
- 产物缺少 identity 字段

### Candidate-level effect

- **all_seeds_must_succeed = true**：任一 seed 触发 hard gate → 整个 candidate INFEASIBLE
- 禁止只对剩余成功 seed 求均值
- 3 seeds 全部成功 → macro_iou_mean = 三者均值

### Performance eligibility

- `val_fixed_foreground_macro_iou >= 0.355644`（primary，含 0）
- 全部 8 个前景区 IoU/Dice/precision/recall + support
- worst-subject IoU >= 0.20
- 每前景区 IoU >= 0.05

### Selection

- 跨 seed macro_iou_mean 排序
- 最多 3_feasible → 取 Top 2（near-tie 优先更简单）
- 最多保留 2 个候选

## Files allowed to change

- `docs/tasks/TASK_SLP_B04A_PROTOCOL_FREEZE_v0.1.md`（本文件）
- `docs/stage_reports/S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md`
- `docs/deliverables/SLP/B04A_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md`
- `configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json`
- `scripts/validate_b04a_protocol.py`
- `tests/test_b04a_protocol_validator.py`
- `scripts/check_markdown_links.py`
- `tests/test_check_markdown_links.py`
- `docs/PROJECT_STATUS.md`（S2_B04A 行）
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`（B04A 条目）
- `docs/deliverables/README.md`（索引和最新报告）

## Out of scope

- 模型代码、配置实现、测试代码（在 B04A-IMPLEMENTATION-SMOKE 任务中）
- 数据处理、Smoke、Mini、Full、TEST
- GPU/云端操作
- 修改 B01/B02/B04 数值或历史 EXP-ID
- commit、push、PR（除非 Owner 后续明确授权）

## Expected artifacts

1. `docs/tasks/TASK_SLP_B04A_PROTOCOL_FREEZE_v0.1.md`（本文件）
2. `docs/stage_reports/S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md`
3. `docs/deliverables/SLP/B04A_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md`
4. `configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json`
5. `scripts/validate_b04a_protocol.py`（正式合同验证器）
6. `tests/test_b04a_protocol_validator.py`（50 单元测试）
7. `scripts/check_markdown_links.py`（Markdown 相对链接检查器）
8. `tests/test_check_markdown_links.py`（6 单元测试）
9. 更新 `docs/PROJECT_STATUS.md`
10. 更新 `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`
11. 更新 `docs/deliverables/README.md`

## Acceptance criteria

- 所有 27 项协议要素精确冻结，机器可读，无歧义
- 无 Owner 待决事项；R03 已通过 Codex Reviewer Gate
- JSON parse 验证通过
- 配置必填字段完整
- 与 B01/B02/B04 历史值一致性检查通过
- 三个候选训练合同完全一致（除架构外）
- 失败 seed 严格级联到 candidate INFEASIBLE
- 架构完全冻结（每层细节）
- identity 字段按文件类型分别定义
- SegFormer DEFERRED 明确记录
- exact_parameter_count 由验证器递归计算并匹配
- ResUNet shortcut 显式 1x1 Conv2d（无 Identity / extra conv 模糊）
- DeepLab 选 Option A（无 Xception / depthwise-separable）
- 验证器 5 大类 R03 检查（每类至少 1 个故障注入测试）
- 50 个验证器单元测试 + 6 链接检查 = 56 个全部通过
- Markdown 相对链接全部有效
- git status --short --branch 记录 dirty 状态
- 所有研究计算记录为 NOT RUN

## Prohibited conclusions

- 文档就绪不等于候选实现、Smoke、Mini 或 Full 就绪
- B04A 不能改写 B04 历史结果
- Mini 不能证明最佳架构
- SLP8 reference 不是人工像素级、医学、皮肤界面应力、硬件或产品 GT
- 不得形成舒适性、整夜稳定性或气囊闭环结论
