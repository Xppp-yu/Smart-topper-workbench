# S1_A09_SLP_REGION_ONTOLOGY_CONTRACT_FREEZE_v0.1

> **⚠️ SUPERSEDED BY**: `S1_A09R_SLP8_GT_CONTRACT_REALIGN_v0.1.md`
> This report is kept for historical reference only. The A09R document supersedes all findings here.

**TASK-ID**: `TASK-SLP-A09-CONTRACT-FREEZE-CORRECTION-v0.1`
**Branch**: `codex/task-slp-a09-contract-freeze-v0.1`
**HEAD**: `6f39a8bf4caf4c506ae509f3bd9aa8ee7b329368`
**Date**: 2026-08-24
**Status**: COMPLETE — Reviewer Gate A09 Frozen

---

## 1. 阶段目标

冻结 SLP 粗区域词表（Region Ontology v0.1）、label tier 语义、review 状态字段和机器可验证 JSON Schema v0.1，为 A10 Geometry Region Seeder 提供稳定合同。

本阶段**不生成任何 R0/R1/R2/R3 真实标注**，不启动 A10，不生成训练数据。

---

## 2. 已验证 / 已推断 / 尚未验证

### 已验证

| 结论 | 证据 |
|---|---|
| 10 区词表已冻结 | schema `region_id` enum 包含全部 10 区 |
| 76 个 Draft 2020-12 validator 测试全部通过 | `uv run pytest -q tests/test_slp_region_annotation_schema.py` |
| Schema 自身通过 Draft 2020-12 metaschema 检查 | `test_schema_passes_draft_2020_12_metaschema` PASS |
| tier/source/status 兼容关系已嵌入 schema allOf | 5 个互斥 if/then 块 |
| R0 final_polygon 可为 null；R1/R2/R3 必须有 | schema R0 rule + R1/R2/R3 rule |
| R2/R3 必须有 non-empty reviewer_id + ISO datetime | allOf R2/R3 rule |
| 未知 region/tier/source/status/rejected/extra 字段全部被 schema 拒绝 | 负面测试 100% 覆盖 |
| SHA256 hex 格式强制（64 char） | `pattern: ^[A-Fa-f0-9]{64}$` |
| polygon 至少 3 点，点为 [number, number] | `$defs/polygon` |
| provenance.source_artifacts 非空 + generator + created_at 必填 | provenance required + minItems |
| quality_flags / reason_codes 唯一性强制 | `uniqueItems: true` |

### 已推断

- Draft 2020-12 原生无法检测 Python float NaN/±Inf（`type: number` 接受），通过 Python-level `_validate_strict()` 补充拦截。
- `format: date-time` 与 `"type": ["string", "null"]` 在 jsonschema 4.26 中对 `null` 值崩溃；`reviewed_at` 和 `provenance.created_at` 的日期合法性通过 Python-level `datetime.fromisoformat()` 在 `_validate_strict()` 中检查。
- `jsonschema` 安装为 `dev` 依赖，不影响生产环境。

### 尚未验证

- A10 Geometry Region Seeder（BLOCKED_BY_A09，依赖本阶段冻结合同）
- R0/R1 真实区域生成逻辑（A10/A11 任务）
- 人工复核 GUI 和流程（A13 BLOCKED_BY_A09_A12）
- R2/R3 真实标注和 Pilot（A14–A17）
- 词表在 SLP 全体帧上的覆盖率（无标注数据，需 Pilot 后统计）

---

## 3. 冻结合同

### 3.1 权威 10 区词表

```
head_neck
shoulder_left
shoulder_right
thorax_back
abdomen_waist
pelvis_hip
thigh_left
thigh_right
lower_leg_foot_left
lower_leg_foot_right
```

**粗语义定义（床面俯视图，非精确解剖分割）：**

| region_id | 粗语义 | 边界说明 |
|---|---|---|
| `head_neck` | 头部 + 颈部的床面投影 | 以肩部以上节点几何为上界；不包含肩膀 |
| `shoulder_left` | 左侧肩/上背部 | 人体左外侧；left/right 以受试者自身左右为准，非观察者视角 |
| `shoulder_right` | 右侧肩/上背部 | 同上 |
| `thorax_back` | 胸背区（肩以下、腰以上） | 沿躯干轴分段；cover 条件下可能包含被子轮廓 |
| `abdomen_waist` | 腰腹区 | **不是腰椎定位**，是床面投影的粗腰腹区域 |
| `pelvis_hip` | 骨盆/髋区 | **不等于精确臀部分割**；不以臀部肌肉解剖为界 |
| `thigh_left` | 左侧大腿 | 髋部到膝部的外侧投影 |
| `thigh_right` | 右侧大腿 | 同上 |
| `lower_leg_foot_left` | 左小腿 + 足部 | 膝部到足尖 |
| `lower_leg_foot_right` | 右小腿 + 足部 | 同上 |

**左右侧约定**：以受试者自身左右为准（即图像右→左为 left）。A06/A08 中 body_axis 的 left/right 定义与此一致。

**遮挡 / 重叠规则**：
- 侧卧时 `shoulder_left` 和 `shoulder_right` 可能重叠，标签允许 `reason_codes: ["occlusion"]`。
- cover1/cover2 条件下可见轮廓属于被褥与体表叠加，不等于真实体表边界；`reason_codes: ["blanket_contour"]` 触发复核。
- 未接触床面的区域（如侧卧时悬空的下肢）可标注但需在 `quality_flags` 中标记。

**pelvis_hip ≠ 精确臀部**：`pelvis_hip` 是髋关节投影的粗区域代理，不是臀部肌肉解剖分割。若业务必须拆出精确臀部，必须增加人工标注规则和不确定性字段（A13 人工复核 GUI 中实现），本 schema 暂不支持。

**abdomen_waist ≠ 腰椎定位**：`abdomen_waist` 是床面投影粗腰腹区域，不是腰椎解剖定位，不得用于腰椎角度或脊柱曲度分析。

**cover 条件被子轮廓说明**：cover1/cover2 条件下 RGB/IR/Depth 可见轮廓属于被褥与人体叠加轮廓，不得直接作为体表边界真值。A12 OpenCV Refiner 应输出 `blanket_contour` quality flag，促发人工复核。

### 3.2 旧版路线文档词表说明

`docs/SLP_RESEARCH_AND_REGION_ANNOTATION_ROUTE_v0.1.md` 第 5.1 节列出的区域词表（`head`、`chest_upper_torso`、`left/right_upper_arm` 等）是首轮历史建议，**不是当前机器合同**。

当前机器合同以本报告第 3.1 节的 10 区词表为准，与 `SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md` 第 4.5 节（`head_neck`、`shoulder_left/right`、`thorax_back`、`abdomen_waist`、`pelvis_hip`、`thigh_left/right`、`lower_leg_foot_left/right`）完全一致。

### 3.3 Label Tier 语义

| Tier | 名称 | 来源 | 是否可训练 |
|---|---|---|---|
| R0 | Geometry Seed | 节点 + 人体轴 + 体型先验生成 polygon | 否（伪标签） |
| R1 | OpenCV Proposal | R0 + 前景/边界细化 | 否（伪标签） |
| R2 | Human Reviewed Reference | 人工 accepted/edited，QC 通过 | **是（默认训练标签）** |
| R3 | Double-reviewed Consensus | 双审/仲裁后高可信子集 | **是（优先评价）** |

**训练默认标签限制**：训练代码默认**只接受 R2/R3**。若研究 R0/R1 弱监督，必须使用不同 EXP-ID 并明确标记，不得混入默认训练集。

### 3.4 Tier / Source / Status 兼容关系（已嵌入 schema allOf）

| Tier | label_source | review_status 允许值 | final_polygon | reviewer_id / reviewed_at |
|---|---|---|---|---|
| R0 | `joint_geometry` | `pending`, `uncertain`, `rejected` | 可为 `null` | 可为 `null`（rejected 例外，见下） |
| R1 | `opencv_refined` | `pending`, `uncertain`, `rejected` | **必须为 polygon** | 可为 `null`（rejected 例外，见下） |
| R2 | `human_accepted` 或 `human_edited` | `accepted`, `edited` | **必须为 polygon** | **必须 non-null + ISO datetime** |
| R3 | `human_consensus` | `accepted`, `adjudicated` | **必须为 polygon** | **必须 non-null + ISO datetime** |

**人工结论（accepted/edited/rejected/adjudicated）必须有 reviewer_id + reviewed_at**：schema `allOf` 块"Human conclusions"规则强制所有含 `accepted/edited/rejected/adjudicated` 状态的记录必须包含 reviewer 标识和时间。

### 3.5 Review 字段说明

| 字段 | 说明 |
|---|---|
| `reviewer_id` | 复核者标识；R2/R3/rejected 必须 non-null non-empty string |
| `reviewed_at` | ISO 8601 datetime（R2/R3/rejected 必须合法 datetime） |
| `reason_codes` | `alignment` / `joint_error` / `occlusion` / `blanket_contour` / `region_ambiguity` / `polygon_topology` / `low_image_quality` / `other`；去重 |
| `quality_flags` | 任意非空字符串；去重；用于记录额外质量标记 |

---

## 4. Schema 技术说明

### 4.1 文件

`configs/annotations/slp_region_annotation_v0.1.schema.json`

### 4.2 关键设计决策

1. **`final_polygon` 不在顶层 `required`**：R0 允许 `null`；通过 R1/R2/R3 专属 `allOf` rule 要求 polygon。
2. **Tier/Source/Status 兼容性用 5 个独立 `allOf` if/then 块**：互不重叠，独立验证。
3. **`reviewed_at` 不使用 `format: date-time`**：与 `"type": ["string", "null"]` 组合时 jsonschema 4.26 对 `null` 值崩溃。日期合法性在 Python-level `_validate_strict()` 中通过 `datetime.fromisoformat()` 验证。
4. **NaN/±Inf 不使用 schema 原生检测**：Draft 2020-12 `type: number` 接受 NaN/Inf；在 Python-level `_validate_strict()` 中通过 `math.isfinite()` 拦截。
5. **provenance `additionalProperties: false`**：拒绝未知字段。

### 4.3 测试覆盖

```
uv run pytest -q tests/test_slp_region_annotation_schema.py
76 passed in 0.32s
```

| 测试类别 | 数量 |
|---|---|
| Schema 结构（existing 4 tests） | 4 |
| Draft 2020-12 metaschema 合规 | 1 |
| 正向：每 tier 合法实例 | 10 |
| 负向：未知 region/tier/source/status | 4 |
| 负向：tier/source 不匹配 | 8 |
| 负向：status 不匹配 | 6 |
| 负向：R1/R2/R3 缺 final_polygon | 3 |
| 负向：R2/R3 缺 reviewer | 4 |
| 负向：人工结论缺 reviewer/time | 5 |
| 负向：extra fields | 1 |
| 负向：SHA256 格式 | 3 |
| 负向：polygon 结构 | 4 |
| 负向：NaN/±Inf | 3 |
| 负向：provenance 缺失字段 | 4 |
| 负向：ISO datetime 格式 | 3 |
| 负向：reason_codes / quality_flags | 2 |
| 负向：sample_id / subject_id 格式 | 3 |
| 负向：frame_index 范围 | 2 |
| 负向：confidence 范围 | 2 |
| **合计** | **76** |

---

## 5. 禁止结论（Prohibited Conclusions）

1. 不得将 R0/R1 伪标签称为 ground truth 或训练真值。
2. 不得将 `pelvis_hip` 用于精确臀部分割或臀部肌肉分析。
3. 不得将 `abdomen_waist` 用于腰椎定位或脊柱曲度分析。
4. 不得将 cover 条件下的 OpenCV 轮廓直接作为体表真值。
5. 不得将 SLP 区域标签作为自研顶垫产品验证结论。
6. 不得将 `head_neck` 细分为 head 和 neck 单独区域（本版不支持）。
7. 不得将侧卧时 `shoulder_left` / `shoulder_right` 重叠视为标注错误（允许 occluded 标记）。

---

## 6. 已知限制

| 限制 | 影响 | 缓解 |
|---|---|---|
| `pelvis_hip` ≠ 精确臀部 | 臀部专项分析不准确 | Pilot 人工复核时判断是否需要拆出精确标注 |
| cover1/cover2 轮廓含被子 | cover 区域边界偏大 | A13 复核工具使用 `blanket_contour` reason code |
| JSON Schema 无法原生检测 NaN/±Inf | 数据处理层需使用 `_validate_strict()` | Python-level 拦截已在测试中验证 |
| 日期格式非 schema 原生验证 | 需使用 `_validate_strict()` | Python-level 拦截已在测试中验证 |
| jsonschema 为 dev 依赖 | 非运行时强制检查 | CI 中运行 pytest；生产环境数据建议使用 `_validate_strict()` |

---

## 7. 下一 Gate

**Gate A09（当前）** ✅ 已完成。

**Gate A10（下一步）**：`TASK-SLP-A10-REGION-SEEDER` 可启动 — 使用本冻结合同的 10 区词表和 tier 语义实现 R0 几何播种器。

前置条件（A09 Freeze 完成）：
- ✅ Schema v0.1 冻结，76 测试全部通过
- ✅ 10 区词表权威定义已记录
- ✅ tier/source/status 兼容关系已嵌入 schema
- ✅ R0/R1 不是默认训练标签已明确
- ✅ pelvis_hip 和 abdomen_waist 语义限制已记录

**本阶段不开通**：
- A11（OpenCV Foreground — BLOCKED_BY_A04_A10）
- A12（Region Boundary Refiner — BLOCKED_BY_A11）
- A13（人工复核工具 — BLOCKED_BY_A09_A12）
- A17（Region Reference Freeze — BLOCKED_BY_A16）

---

## 8. 文件变更摘要

| 文件 | 变更 |
|---|---|
| `configs/annotations/slp_region_annotation_v0.1.schema.json` | 重写：移除 `final_polygon` 顶层 required；新增 5 个 `allOf` tier/source/status 兼容块；移除 `format: date-time` 以避免 nullable 崩溃 |
| `tests/test_slp_region_annotation_schema.py` | 重写：从 4 个静态检查扩展到 76 个 Draft 2020-12 实例验证测试；新增 `_validate_strict()` 处理 NaN/Inf 和日期 Python-level 检查 |
| `pyproject.toml` | 新增 `jsonschema>=4` 为 dev 依赖 |
| `docs/stage_reports/S1_A09_SLP_REGION_ONTOLOGY_CONTRACT_FREEZE_v0.1.md` | 新增：本阶段报告 |

---

## 9. Reviewer Checklist

- [x] Schema 通过 Draft 2020-12 metaschema 检查
- [x] 每 tier 至少一个合法实例测试通过
- [x] 每项交叉约束至少一个失败实例测试通过
- [x] `uv run pytest -q tests/test_slp_region_annotation_schema.py` → 76 passed
- [x] 禁止结论已列出，无违反
- [x] pelvis_hip ≠ 精确臀部已记录
- [x] abdomen_waist ≠ 腰椎定位已记录
- [x] cover 条件下被子轮廓已说明
- [x] 旧版路线文档不同词表已标注为历史建议
- [x] R0/R1 不是默认训练标签已明确
- [x] 未生成任何 R0/R1/R2/R3 真实标签
- [x] git diff --check 未运行（禁止自行 commit）

---

*本报告由 Claude Code（Mavis）生成，Codex Reviewer 验收后更新 `docs/PROJECT_STATUS.md`。*
