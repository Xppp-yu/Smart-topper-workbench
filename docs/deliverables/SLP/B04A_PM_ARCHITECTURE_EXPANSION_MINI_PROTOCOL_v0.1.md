# B04A 交付说明：SLP8 PM-only Architecture Expansion Mini Protocol v0.1 (R03)

**TASK-ID:** `TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1`
**Stage:** S2-B04A
**日期:** 2026-08-29（R03 修订轮）
**状态:** `PROTOCOL_ACCEPTED / IMPLEMENTATION_NOT_STARTED`
**Git:** `ffab5763cc55d07968602b5413c6e8d2c9eaa56d`

> 本文件是 B04A 协议冻结（R03）的交付说明。完整协议内容见 [../../stage_reports/S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md](../../stage_reports/S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md)。

---

## 1. 协议冻结摘要（R03）

R03 在 R02 基础上做以下关键修订（Codex Reviewer ITERATE 反馈）：

1. **ResUNet 第一残差块消除歧义**：shortcut 显式冻结为 `Conv2d 1x1, in=1, out=16, stride=1, padding=0, bias=true`；删除 "Identity" / "extra conv for now" 模糊描述。
2. **DeepLab 选 Option A**（普通 atrous Conv2d）：删除 Xception / depthwise-separable 声称；明确 `groups=1` 和 `dilation=atrous_rate`。
3. **exact_parameter_count 字段**：3 个候选均加入；验证器递归计算并匹配：
   - SmallUNet: **118,121**（B04 R05 一致）
   - ResUNet-lite: **120,809**（与 Reviewer 给出的参考值完全一致）
   - DeepLabV3+-lite: **53,449**（Option A 重算）
4. **验证器新增 5 大类 R03 检查**（每类至少 1 个故障注入测试）：
   - residual Add shape/channel 一致性
   - Concat result_channels 一致性
   - ASPP branch output / post_concat 通道一致性
   - depthwise groups 一致性
   - exact parameter count 一致性
5. **删除 R01 临时脚本**：已移至 `outputs/legacy_to_be_removed/`（git-ignored 区域）
6. **区分"已运行"与"未运行"**：
   - **已运行**：协议验证器 56 项单元测试（27 R02 + 23 R03 + 6 链接检查）
   - **NOT RUN**：模型实现、模型 Smoke、模型 unit tests、GPU Mini、Full、TEST

---

## 2. 冻结的关键数值

| 项目 | 值 |
|---|---|
| 候选 | SmallUNet v0.1 (incumbent) / ResUNet-lite v0.1 / DeepLabV3+-lite v0.1 |
| SegFormer-B0 | DEFERRED |
| 合格阈值 | B02 0.205644 + margin 0.15 = **0.355644**（`>=`） |
| Seeds | [42, 123, 2026]（全部必须成功） |
| Augmentation | 所有候选 = `none` |
| Batch size | 16 |
| Optimizer | AdamW, lr=0.001, wd=0.0001 |
| Max epochs | 30 |
| Min epochs | 5 |
| Early stopping | patience=4 |
| exact_parameter_count | 118,121 / 120,809 / 53,449 |
| 参数上限 | SmallUNet: 150,000；新候选: 300,000 |
| Peak CUDA memory | 8,192 MB |
| 单 candidate wall time | 45 min（3 seeds 累计） |
| 总 wall time | 135 min（3 candidates × 45 min） |
| 失败 seed 处理 | 整个 candidate INFEASIBLE（`all_seeds_must_succeed=true`） |
| `3_feasible` 规则 | Top 2 by macro_iou_mean |
| Class collapse | 零预测 → INFEASIBLE |
| Worst-subject floor | < 0.20 → INFEASIBLE（前瞻性 policy） |
| Per-region floor | < 0.05 → INFEASIBLE |
| Near-tie margin | 0.02 |
| 最多晋级 | 1–2 个候选 |
| Class weight mean | 排除 background |
| Primary metrics | 不含 background |
| Pressure storage dtype | float64 |
| Model input dtype | float32 |
| Dtype 转换位置 | normalization 之后、augmentation 之前 |
| Identity 字段格式 | JSON top-level / CSV `.identity.json` 旁 / checkpoint `identity` dict / log 首行 JSON |

---

## 3. 候选架构（完整冻结）

### 3.1 slp8_small_unet_v0.1（incumbent）

- **架构假设**：经典 encoder-decoder + skip connection
- **来源**：B04 protocol；slp8_region_models.Slp8SmallUnet
- **exact_parameter_count**: 118,121
- **增强**：`none`
- **B04 R05 结果**：VAL macro IoU 0.439625（FEASIBLE）
- **架构冻结**：见 stage report §3.1 + 配置文件 `forward_plan`

### 3.2 slp8_resunet_lite_v0.1（新增候选，R03 修订）

- **架构假设**：Residual CNN — encoder 内每个 block 增加 1x1 Conv2d shortcut
- **来源**：B04A protocol / NOT_IMPLEMENTED
- **exact_parameter_count**: 120,809
- **增强**：`none`
- **架构冻结（完整）**：
  - 3 个 residual block（enc1, enc2, bottleneck）
  - **每个 block shortcut 显式为 `Conv2d 1x1, k=1, s=1, p=0, bias=true`**
  - **不使用 Identity shortcut**；不使用 "extra conv for now"
  - 第一个 block 通道不匹配（1→16）：1x1 Conv2d 投影（in=1, out=16, k=1, s=1, p=0, bias=true）
  - 其他通道不匹配（16→32, 32→64）：1x1 Conv2d 投影
  - 通道匹配：1x1 Conv2d 仍（无 identity 路径）
  - Decoder: concat skip + bilinear recovery
- **每层细节**：见 stage report §3.2 + 配置文件 `forward_plan`

### 3.3 slp8_deeplabv3plus_lite_v0.1（新增候选，R03 选 Option A）

- **架构假设**：Atrous / multi-scale context — **普通（non-depthwise-separable）atrous Conv2d**
- **来源**：B04A protocol / NOT_IMPLEMENTED
- **exact_parameter_count**: 53,449
- **增强**：`none`
- **架构冻结（完整，R03 Option A）**：
  - **variant**: `option_A_plain_atrous_Conv2d`
  - **不包含**：Xception、depthwise-separable、grouped conv
  - stem: 1→16 k3 + 16→16 k3（low-level features source）
  - down1: 16→32 k3 s2（stride 2 downsample）
  - ASPP: 6 分支，每分支 16 ch
    - branch 0: 1x1 pointwise (in=32, out=16, k=1, s=1, p=0, groups=1)
    - branches 1–4: 3x3 atrous (rates=[3,6,9,12])，`dilation=atrous_rate`, `padding=atrous_rate`, `groups=1`
    - branch 5: GAP → 1x1 → BilinearInterpolate
  - Concat: 6 × 16 = 96 ch
  - post_concat: 1x1 Conv 96→32 → ReLU
  - Decoder: low-level features (1x1 Conv 16→16) + bilinear upsample + Concat 48 ch + 两层 3x3 Conv
  - Final: 1x1 Conv 32→9
- **每层细节**：见 stage report §3.3 + 配置文件 `forward_plan`

### 3.4 slp8_segformer_b0_v0.1（DEFERRED）

- **状态**：DEFERRED
- **延期理由**：ImageNet 预训练权重与从头训练的 CNN 候选存在根本性公平性不对称

---

## 4. 数据合同

| 字段 | 值 |
|---|---|
| TRAIN | 3,645 samples / 81 subjects |
| VAL | 450 samples / 10 subjects |
| TEST | **0 / 0**（永不读取） |
| Split SHA | `024f5abe...`（B01 合同不变） |
| Freeze manifest SHA | `3c789995...`（B01 合同不变） |
| Provenance | `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` |
| Review status | `NOT_REVIEWED` |
| Setting / Cover | danaLab / uncover |
| Storage dtype | float64（压力），int64（标签） |
| Model input dtype | float32（在 normalization 之后转换） |

---

## 5. 协议文件索引

```
configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json
  — 机器可读的冻结实验配置（含每层架构冻结 + exact_parameter_count）

docs/stage_reports/S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md
  — 完整协议阶段报告（含所有精确定义）

docs/tasks/TASK_SLP_B04A_PROTOCOL_FREEZE_v0.1.md
  — 协议冻结任务合同

scripts/validate_b04a_protocol.py
  — 正式合同验证器（含 R02 + R03 共 11+ 类检查）

tests/test_b04a_protocol_validator.py
  — 验证器单元测试（27 R02 + 23 R03 = 50 个）

scripts/check_markdown_links.py
  — Markdown 相对链接检查器

tests/test_check_markdown_links.py
  — 链接检查器单元测试（6 个）

合计 56 个测试，全部通过
```

---

## 6. 已运行 vs NOT RUN

### 已运行（PASS）

- 协议验证器（`validate_b04a_protocol.py`）：28+ 项检查，0 errors
- 协议验证器单元测试：50 项（27 R02 + 23 R03），全部通过
- 链接检查器单元测试：6 项，全部通过
- JSON parse：通过
- git diff --check：clean
- git status --short --branch：dirty（按预期）

### NOT RUN

- 模型实现（ResUNet-lite、DeepLabV3+-lite 代码）
- 模型的单元测试
- CPU / CUDA Smoke
- 实际参数量（虽然 `exact_parameter_count` 已由 validator 机械推导；需在实现时验证）
- 训练稳定性、peak CUDA memory、wall time 实际值
- 任何 GPU 计算、Mini、Full
- TEST 读取

---

## 7. 禁止结论

> 1. Mini 结果证明最佳架构
> 2. 新候选优于 SmallUNet incumbent
> 3. 架构比较形成最终排名
> 4. 适用于产品、硬件、舒适性、医疗、整夜稳定性或气囊控制
> 5. SegFormer 在看到 Mini 结果后可被纳入
> 6. TEST 结果可见或可推断
> 7. 压力值为 kPa
> 8. 标签为人工像素级标注

---

## 8. 下一步

| 阶段 | 状态 | 说明 |
|---|---|---|
| B04A Protocol | ✅ PROTOCOL_ACCEPTED | R03 已通过 Codex Reviewer Gate |
| B04A Implementation + Smoke | NEXT_ALLOWED | 独立 TASK；不得运行真实 GPU Mini |
| B04A GPU Mini | BLOCKED（等待 Owner 授权） | 独立 TASK |
| B04A Review | BLOCKED（等待 GPU Mini 完成） | Codex |
| B07 Protocol | BLOCKED_BY_B04A | — |

---

**交付版本：** v0.1-R03
**生成时间：** 2026-08-29
**维护者：** Mavis (MiniMax Code)
