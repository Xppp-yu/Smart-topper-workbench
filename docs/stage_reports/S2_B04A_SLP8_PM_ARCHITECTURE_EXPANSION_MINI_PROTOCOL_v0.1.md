# Stage Report: S2-B04A — SLP8 PM-only Architecture Expansion Mini Protocol v0.1 (R03)

**TASK-ID:** `TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1`
**Stage:** S2-B04A
**日期:** 2026-08-29（R03：Codex Reviewer ITERATE 后第二轮修订）
**状态:** `PROTOCOL_ACCEPTED / IMPLEMENTATION_NOT_STARTED`
**Branch:** `codex/task-slp-b04a-protocol-freeze-v0.1`
**Git:** `ffab5763cc55d07968602b5413c6e8d2c9eaa56d`
**工作树状态:** dirty（新增/修改文件未 commit）

---

## 1. 执行摘要

R03 在 R02 基础上做以下关键修订（来自 Codex Reviewer ITERATE 反馈）：

1. **ResUNet 第一残差块消除歧义**：shortcut 显式冻结为 `Conv2d 1x1, in=1, out=16, stride=1, padding=0, bias=true`；删除所有 "Identity" 和 "extra conv for now" 模糊描述；ResUNet 所有 residual Add 两侧 shape/channels 完全一致。
2. **DeepLab 二选一** → 选 **Option A（普通 atrous Conv2d）**：删除 "depthwise-separable" 和 "Xception" 全部声称；明确每个分支的 `groups=1`、`dilation=atrous_rate`、`padding=atrous_rate`；post_concat 输入 6 × 16 = 96。
3. **参数量由冻结层表机械推导**：`exact_parameter_count` 字段为 3 个候选均加入；验证器递归计算 Conv2d 参数并对比。
   - SmallUNet：`exact_parameter_count=118121`（B04 R05 历史值）
   - ResUNet-lite：`exact_parameter_count=120809`（与 Reviewer 给出的参考值一致）
   - DeepLabV3+-lite：`exact_parameter_count=53449`（Option A 重算）
4. **验证器新增** 5 大类 R03 检查（每类至少 1 个故障注入测试）：
   - residual Add shape/channel consistency
   - Concat result_channels consistency
   - ASPP branch output / post_concat 通道一致性
   - depthwise groups consistency
   - exact parameter count consistency
5. **删除** R01 临时验证脚本：已移动到 `outputs/legacy_to_be_removed/`（git-ignored 区域）；不应再被引用。
6. **区分"已运行的"与"未运行的"**：协议验证器 56 项测试（27 R02 + 23 R03 单元测试 + 6 链接检查）已全部运行通过；**模型实现、模型 Smoke、模型 unit tests 全部 NOT RUN**，待 B04A-IMPLEMENTATION-SMOKE 任务。

---

## 2. 冻结的协议要素

| 项目 | 状态 | 说明 |
|---|---|---|
| 候选名单与架构假设 | ✅ 冻结 | SmallUNet (incumbent) / ResUNet-lite (1x1 shortcut) / DeepLabV3+-lite (Option A) |
| 三个候选训练合同一致性 | ✅ 冻结 | 全部 seeds=[42,123,2026]、AdamW lr=0.001、max_epochs=30、patience=4、aug=none |
| 三个候选 `exact_parameter_count` | ✅ 冻结 | 118,121 / 120,809 / 53,449，验证器递归计算并匹配 |
| 输入形状与归一化 | ✅ 冻结 | `[192, 84]`，float32 输入，raw passthrough，dtype 转换位置明确 |
| 初始化与预训练 | ✅ 冻结 | 无预训练，Kaiming init |
| 增强策略 | ✅ 冻结 | 全部候选 `augmentation_policy = none` |
| Loss 与 class weights | ✅ 冻结 | CrossEntropy + TRAIN-only `1/sqrt(pixel_ratio)`，mean 排除 background，clip [0.25, 4.0] |
| Optimizer 与学习率 | ✅ 冻结 | AdamW，lr=0.001，weight_decay=0.0001，无 scheduler |
| 训练预算 | ✅ 冻结 | max 30 epoch，min 5 epoch，patience 4 |
| Seeds 与汇总 | ✅ 冻结 | seeds=[42, 123, 2026]；必须 3 seeds 全部可行 |
| 合格阈值 | ✅ 冻结 | B02=0.205644 + margin=0.15 = 0.355644（`>=`） |
| Class collapse 定义 | ✅ 冻结 | per_seed → candidate 级联 INFEASIBLE |
| Worst-subject guardrail | ✅ 冻结 | < 0.20 → INFEASIBLE（前瞻性 policy） |
| Per-region guardrail | ✅ 冻结 | < 0.05 → INFEASIBLE（per seed，候选级联） |
| 决策规则 | ✅ 冻结 | 0/1/2/3_feasible 全部定义；3_feasible → Top 2 |
| Near-tie margin | ✅ 冻结 | 跨 seed macro IoU 均值差 < 0.02 → 优先更简单模型 |
| 最多晋级候选数 | ✅ 冻结 | 1–2 个 |
| 资源预算 | ✅ 冻结 | 单候选 45 min（3 seeds 累计）/ 8192 MB CUDA；总 135 min；candidate+total 累计 |
| Checkpoint/resume/reload | ✅ 冻结 | B04 Runner 基础设施继承；candidate_seconds + total_seconds 必须保存恢复 |
| 产物 identity 格式 | ✅ 冻结 | JSON top-level、CSV 旁 `.identity.json`、checkpoint `identity` dict、log 首行 JSON |
| TEST 禁令 | ✅ 冻结 | TEST rows=0, labels=0, onehot=0；任何 TEST 读取 → fail-closed |
| 架构完全冻结 | ✅ 冻结 | 每层 channels/kernel/stride/padding/residual projection/ASPP dilation rates 列出 |

### SegFormer-B0 纳入/延期决定

**决定：DEFERRED。**

SegFormer-B0（~3.7M 参数，默认使用 ImageNet 预训练权重）与 CNN 候选（均从头训练）存在根本性公平性不对称。协议明确规定了 7 项必须预先解决的前置公平性项目。**DEFERRED，不得在看到 Mini 结果后临时纳入。**

---

## 3. 候选架构（完整冻结，每层细节）

### 3.1 SmallUNet（incumbent）

- **架构假设**：经典 encoder-decoder + skip connection
- **exact_parameter_count**: 118,121
- **每层细节**：见配置 `architecture_freeze.forward_plan` 完整列出
- **B04 R05 结果**：VAL macro IoU 0.439625（FEASIBLE）

### 3.2 ResUNet-lite（new candidate）

- **架构假设**：Residual CNN — encoder 内每个 block 增加 1×1 Conv2d shortcut
- **exact_parameter_count**: 120,809
- **关键冻结点**：
  - 3 个 residual block（enc1, enc2, bottleneck）
  - 每个 block 显式定义 `main_output_channels` 和 `shortcut_output_channels`
  - **ResUNet 不使用 Identity shortcut**；全部为 1x1 Conv2d
  - 第一个 block 通道不匹配（1→16）：1x1 Conv2d 投影
  - 其他通道不匹配（16→32, 32→64）：1x1 Conv2d 投影
  - Decoder: concat skip + bilinear recovery（与 SmallUNet 相同的 decoder 结构）
- **每层细节**：见配置

### 3.3 DeepLabV3+-lite（new candidate, Option A）

- **架构假设**：Atrous / multi-scale context — 普通（non-depthwise-separable）atrous Conv2d
- **exact_parameter_count**: 53,449
- **关键冻结点（R03 选 Option A）**：
  - 6 个 ASPP 分支（branch_id 0–5），每分支 16 ch
    - branch 0: 1x1 pointwise (in=32, out=16, k=1, s=1, p=0, groups=1)
    - branches 1–4: 3x3 atrous (rates=[3,6,9,12])：每分支 `dilation=atrous_rate`、`padding=atrous_rate`、`groups=1`
    - branch 5: GAP → 1x1 → BilinearInterpolate
  - Concat: 6 × 16 = 96 ch
  - post_concat: 1x1 Conv 96→32 → ReLU
  - Decoder: low-level features (1x1 Conv 16→16) + bilinear upsample + Concat 48 ch + 两层 3x3 Conv
  - Final: 1x1 Conv 32→9
- **明确不包含**：Xception、depthwise-separable、grouped conv (groups != 1 and groups != in_channels)
- **每层细节**：见配置

### 3.4 SegFormer-B0（DEFERRED）

- **状态**：DEFERRED
- **延期理由**：ImageNet 预训练权重与从头训练的 CNN 候选存在根本性公平性不对称

---

## 4. 数据合同

| 字段 | 值 |
|---|---|
| TRAIN samples / subjects | 3,645 / 81 |
| VAL samples / subjects | 450 / 10 |
| TEST samples / subjects | **0 / 0**（loader 永不读取 TEST rows） |
| Pressure storage dtype | float64 (B01 freeze) |
| Pressure model input dtype | float32 (after conversion) |
| Label dtype | int64 |
| raw_semantics | `raw_pmarray_response`（NOT kPa） |
| Provenance | `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` |
| source_review_status | `NOT_REVIEWED` |
| A06 split SHA | `024f5abe...`（B01 合同，不变） |
| B01 freeze manifest core SHA | `3c789995...`（B01 合同，不变） |

---

## 5. 核心 Gate 数值

| 项目 | 值 |
|---|---|
| Feasibility threshold | `val_fixed_foreground_macro_iou >= 0.355644`（B02 0.205644 + 0.15） |
| Class collapse | 任意前景类零 VAL 预测（per seed）→ INFEASIBLE |
| Worst-subject floor | < 0.20 (per seed) → INFEASIBLE（前瞻性 policy floor） |
| Per-region floor | < 0.05 (per seed) → INFEASIBLE |
| all_seeds_must_succeed | `true`（任一 seed 失败 → 整个 candidate INFEASIBLE） |
| 3_feasible 规则 | Top 2 by macro_iou_mean |
| Near-tie margin | 0.02 |

---

## 6. 资源预算（累计语义明确）

```
per_candidate_wall_minutes: 45  (3 seeds 累计)
total_wall_minutes:         135 (3 candidates × 45, serial)
max_peak_cuda_mb:           8192
candidates_serial:          true
seeds_serial_within_candidate: true
checkpoint_resume_keeps_accumulators:
  candidate_level: 保存 candidate_seconds_consumed，恢复不双计
  total_level:     保存 total_seconds_consumed，恢复不双计
```

---

## 7. 验证结果（R03）

| 验证项 | 结果 |
|---|---|
| JSON parse | ✅ 通过 |
| B04A 合同验证器 | ✅ 28+ OKs, 0 errors |
| 协议验证器单元测试 | ✅ **56 passed**（27 R02 + 23 R03 + 6 链接检查） |
| Markdown 链接检查 | ✅ 0 errors on 3 modified docs + 3 B04A docs |
| B01 A06 SHA `024f5abe...` | ✅ 一致 |
| B01 freeze manifest SHA `3c789995...` | ✅ 一致 |
| B02 baseline `0.205644` | ✅ 不变 |
| 有效阈值 `0.355644` | ✅ 算术一致（`>=`） |
| 三个候选 augmentation_policy | ✅ 全部 `none` |
| 三个候选 augmentation blocks | ✅ byte-identical |
| 三个候选 seeds | ✅ 全部 `[42, 123, 2026]` |
| 失败 seed 处理 | ✅ `all_seeds_must_succeed=true` + 显式 `forbidden` |
| `3_feasible` 决策规则 | ✅ 存在 |
| 资源预算乘法 | ✅ 45 × 3 = 135 |
| `pressure_storage_dtype` vs `model_input_dtype` | ✅ float64 / float32 |
| 背景排除 class-weight mean | ✅ `background_excluded_from_mean=true` |
| 背景不进 primary | ✅ `background_included_in_macro_average=false` |
| ResUNet shortcut 显式 1x1 Conv2d (1→16) | ✅ 全部 3 个 residual block 显式列出 |
| ResUNet 残差 Add shape/channel 一致 | ✅ 3 个 block 全部一致 |
| DeepLab Option A 显式声明 | ✅ `variant: option_A_plain_atrous_Conv2d` |
| DeepLab 无 depthwise | ✅ 所有 Conv2d `groups=1` |
| DeepLab ASPP dilation == atrous_rate | ✅ 4 个 atrous 分支一致 |
| DeepLab post_concat 96 → 32 | ✅ 一致 |
| exact_parameter_count SmallUNet | ✅ 118,121（与 B04 R05 一致） |
| exact_parameter_count ResUNet | ✅ 120,809（验证器递归计算匹配） |
| exact_parameter_count DeepLab | ✅ 53,449（Option A 重算） |
| SegFormer DEFERRED | ✅ |
| Worst-subject 0.20 无"50% of B02" | ✅ 前瞻性 policy floor |
| Identity carrier format per file type | ✅ 4 种格式分别定义 |
| TEST=0 | ✅ 两层配置均声明 |
| `git diff --check` | ✅ clean |
| `git status --short --branch` | dirty（按预期） |

---

## 8. 已验证事实

1. B01 数据合同不变（A06 SHA、freeze manifest SHA）
2. B02 baseline `0.205644` 历史值不变
3. B04 R05 SmallUNet `0.439625` 历史结果不变
4. 三个候选训练合同完全一致（除架构外）
5. 失败 seed 严格级联到 candidate INFEASIBLE
6. 资源预算累计语义明确
7. 架构完全冻结（每层细节在配置中列出）
8. Identity 字段按文件类型分别定义
9. 背景不进 primary metrics
10. ResUNet shortcut 显式 1x1 Conv2d，无 Identity / extra conv 模糊描述
11. DeepLabV3+-lite 选 Option A（plain atrous），无 Xception / depthwise-separable
12. 三个候选 `exact_parameter_count` 由验证器递归计算并匹配
13. 协议验证器 + 链接检查器 = 56 个测试全部通过
14. R01 临时验证脚本已移至 `outputs/legacy_to_be_removed/`（git-ignored 区域）

---

## 9. 推断

1. ResUNet 120,809 参数量与 Reviewer 给出的参考值完全一致，验证了 1x1 shortcut 的精确性
2. DeepLab 53,449 参数量远低于 300k 上限（Option A 比 Option B depthwise-separable 少约 4x 参数）
3. 三个候选的 max_parameters 仍合理（SmallUNet 150k、新候选 300k），且都未被超过

---

## 10. 未验证事项（NOT RUN）

> **以下为 NOT RUN**——属于 B04A-IMPLEMENTATION-SMOKE 任务范围：
>
> - 模型实现（ResUNet-lite、DeepLabV3+-lite 代码）
> - 模型的单元测试
> - CPU / CUDA Smoke
> - 实际参数量（虽然 `exact_parameter_count` 已由 validator 机械推导）
> - 训练稳定性、peak CUDA memory、wall time 实际值
> - 任何 GPU 计算、Mini、Full
> - TEST 读取

---

## 11. 限制

1. 协议冻结不构成实验结果
2. 标签为 `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`，`NOT_REVIEWED`
3. danaLab/uncover only，结果不外推
4. 参数上限 300k 是前瞻性决策
5. 架构冻结在 B04A 实现/Smoke 阶段生效

---

## 12. 禁止结论

- ❌ B04A Mini 结果证明最佳架构
- ❌ 新候选优于 SmallUNet incumbent
- ❌ 架构比较形成最终排名
- ❌ 适用于产品、硬件、舒适性、医疗、整夜稳定性或气囊控制
- ❌ SegFormer 在看到 Mini 结果后可被纳入
- ❌ TEST 结果可见或可推断
- ❌ 压力值为 kPa
- ❌ 标签为人工像素级标注

---

## 13. Reviewer Checklist

- [x] 候选名单、版本、架构假设全部精确冻结
- [x] 三个候选训练合同完全一致
- [x] ResUNet 三个 residual block 全部显式 1x1 Conv2d shortcut；无 Identity / extra conv 模糊
- [x] ResUNet 所有 residual Add 两侧 shape/channels 完全一致
- [x] DeepLabV3+-lite 选 Option A（plain atrous）；无 Xception / depthwise-separable
- [x] DeepLab ASPP dilation/padding 与 atrous_rate 一致；6 分支全部 16 ch；post_concat 96→32
- [x] 三个候选 `exact_parameter_count` 由 validator 机械推导并匹配
- [x] 输入形状、dtype、归一化、resize 政策精确冻结
- [x] `pressure_storage_dtype` (float64) 与 `model_input_dtype` (float32) 分别冻结
- [x] 增强策略精确冻结（所有候选 = none）
- [x] Loss、class weights、background 处理精确冻结
- [x] Optimizer、scheduler、early stopping 精确冻结
- [x] Seeds=[42, 123, 2026] 和汇总方式精确冻结
- [x] 阈值=0.355644 精确冻结（`>=`）
- [x] Class collapse / worst-subject / per-region guardrails 精确冻结
- [x] 0/1/2/3_feasible 决策规则全部定义
- [x] Per-region/posture/subject/centroid 指标完整列出
- [x] Near-tie margin=0.02 精确冻结
- [x] 最多 1–2 个候选选择顺序精确冻结
- [x] 参数上限精确冻结
- [x] Peak CUDA memory=8,192 MB 精确冻结
- [x] Wall time 累计语义精确冻结
- [x] Checkpoint/resume/reload 合同精确冻结
- [x] 产物 identity 字段按文件类型分别精确冻结
- [x] 失败/停止/完成状态定义精确冻结
- [x] Reviewer 独立复算范围完整列出
- [x] ResUNet 架构完全冻结（每层 channels/kernel/stride/padding/residual projection）
- [x] DeepLabV3+-lite 架构完全冻结（ASPP rates、low-level decoder）
- [x] SegFormer 明确 DEFERRED
- [x] TEST=0 禁用在配置中明确声明
- [x] B01 split SHA、freeze manifest SHA 与 B01 合同一致
- [x] B02 baseline 0.205644 历史值不变
- [x] 无 Owner 待决事项；R03 已通过 Codex Reviewer Gate

---

## 14. 验证器扩展性测试（56 项）

R03 验证器单元测试包括以下 23 项 R03 新增检查（每项有故障注入）：

| 类别 | 测试数 | 测试名称 |
|---|---|---|
| Residual Add shape/channel | 9 | test_residual_shortcut_must_be_conv2d_not_identity, ..._must_be_1x1_kernel, ..._padding_must_be_zero, ..._stride_must_be_one, ..._bias_must_be_true, ..._add_channels_mismatch_caught, ..._add_shape_mismatch_caught, ..._in_channels_must_match_block_input, ..._projection_text_must_not_say_identity_or_extra |
| Concat result_channels | 2 | test_concat_result_channels_must_ge_with_channels, test_aspp_concat_result_must_equal_sum_of_branches |
| ASPP branch / post_concat | 6 | test_aspp_branch_output_channels_must_match_per_branch_setting, test_aspp_expected_post_concat_must_match_computed, test_aspp_post_concat_in_channels_must_match_expected, test_aspp_atrous_dilation_must_match_branch_rate, test_aspp_atrous_padding_must_match_branch_rate, test_deeplabv3plus_aspp_atrous_rates_required |
| Depthwise groups | 2 | test_depthwise_groups_must_equal_in_channels_caught, test_arbitrary_grouped_conv_caught |
| Exact parameter count | 4 | test_exact_parameter_count_must_be_present, test_exact_parameter_count_must_match_computed, test_resunet_exact_parameter_count_must_be_120809, test_deeplab_exact_parameter_count_must_be_53449 |

合计 27 R02 测试 + 23 R03 测试 + 6 链接检查 = **56 项全部通过**。

---

## 15. 当前 git status

```
Branch:   codex/task-slp-b04a-protocol-freeze-v0.1
HEAD:     ffab5763cc55d07968602b5413c6e8d2c9eaa56d
Status:   dirty
Ahead/behind origin/main: 0 / 0
```

---

## 16. Codex Reviewer 验收与下一 Gate

Reviewer verdict：`ACCEPT`（2026-08-29）。Codex 独立重跑 56 项协议/链接测试，复算三候选参数量，执行协议验证器、Markdown 相对链接检查和 `git diff --check`，均通过。`ruff` 在当前环境不可用，记为 `NOT RUN (tool unavailable)`；本任务未运行模型实现、模型测试、Smoke、GPU Mini、Full 或 TEST。

1. **B04A-IMPLEMENTATION-SMOKE**（独立 TASK-ID）：实现 ResUNet-lite 和 DeepLabV3+-lite 模型注册表 + CPU/最小 CUDA Smoke 合同；**不得运行真实 GPU Mini**。
2. **Owner 授权 GPU Mini**：协议和实现验收后，由 Owner 单独授权。
3. **B04A-REVIEW**：Codex 独立审查 GPU Mini 产物、指标和 identity。
4. **B07 解锁**：B04A 实际 Mini 经 Reviewer 接受并冻结最多 1–2 个候选后，B07 才能开始协议冻结。

---

**交付版本：** v0.1-R03
**生成时间：** 2026-08-29
**维护者：** Mavis (MiniMax Code)
