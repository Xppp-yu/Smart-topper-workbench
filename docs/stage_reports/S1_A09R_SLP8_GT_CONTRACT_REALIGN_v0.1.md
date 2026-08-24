# S1_A09R_SLP8_GT_CONTRACT_REALIGN_v0.1

**TASK-ID**: `TASK-SLP-A09R-SLP8-GT-CONTRACT-REALIGN-v0.1`
**Branch**: `codex/task-slp-a09-contract-freeze-v0.1`
**HEAD**: `6f39a8bf4caf4c506ae509f3bd9aa8ee7b329368`
**Date**: 2026-08-24
**Status**: COMPLETE_WITH_LIMITATIONS — Codex Reviewer Accepted

---

## 1. Owner 决策及生效范围

### 决策

Owner（2026-08-24）正式接受 `SLP_8Region_Pressure_VAL_v1.1` 作为本项目 **SLP8 pressure-only 区域分割训练和评估的 PROJECT_ACCEPTED_REFERENCE_GT**。

数据集路径：
```
E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\SLP_8Region_Pressure_VAL_v1.1
```

### 数据合同

| 字段 | 值 |
|---|---|
| 样本数 | 4,590 |
| 受试者数 | 102 danaLab |
| 每 subject 帧数 | 45（静态 pose） |
| 姿态分布 | SUPINE 1,530 / LEFT 1,530 / RIGHT 1,530 |
| X（压力）shape | (192, 84)，dtype=float64 |
| Y（区域标签）shape | (192, 84)，dtype=uint8 |
| Label IDs | 0 = BACKGROUND，1 = HEAD_NECK，2 = SHOULDER，3 = THORAX_BACK，4 = LUMBAR_WAIST，5 = PELVIS_HIP，6 = ARM，7 = THIGH，8 = LOWER_LEG_FOOT |
| 背景像素占比 | ~70.4%（全数据集均值） |
| annotation_provenance | `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` |
| source_review_status | `NOT_REVIEWED` |

### 必须同时记录的限制

1. 项目已接受为训练评估 GT ✅
2. 来源是自动 Corrected Support（`V221_CORRECTED_SUPPORT_AUTO_ACCEPTED`）✅
3. 不是人工逐帧 semantic mask（`NOT_REVIEWED`）✅
4. 不是医学、皮肤界面应力或产品真值 ✅
5. 不得修改或伪造原有 provenance/review 字段 ✅

---

## 2. 新旧 Ontology 对比

### 8-region SLP8 GT（当前训练合同）

| 字段 | 值 |
|---|---|
| schema | `configs/annotations/slp_8region_pressure_gt_v1.1.schema.json` |
| region 词表 | 8 区：HEAD_NECK / SHOULDER / THORAX_BACK / LUMBAR_WAIST / PELVIS_HIP / ARM / THIGH / LOWER_LEG_FOOT |
| 数据类型 | pressure matrix → semantic mask（uint8） |
| 区域合并 | SHOULDER（左右合并），THIGH（左右合并），LOWER_LEG_FOOT（左右合并），包含 ARM |
| 命名 | LUMBAR_WAIST（非 abdomen_waist） |
| Provenance | V221_CORRECTED_SUPPORT_AUTO_ACCEPTED |
| review_status | NOT_REVIEWED |
| 适用任务 | SLP8 pressure-only 区域分割训练 |

### 10-region polygon 路线（历史治理合同）

| 字段 | 值 |
|---|---|
| schema | `configs/annotations/slp_region_annotation_v0.1.schema.json` |
| region 词表 | 10 区：head_neck / shoulder_left/right / thorax_back / abdomen_waist / pelvis_hip / thigh_left/right / lower_leg_foot_left/right |
| 数据类型 | RGB/IR/Depth → polygon（JSON） |
| 区域拆分 | SHOULDER（左右分）、THIGH（左右分）、LOWER_LEG_FOOT（左右分），无 ARM |
| Provenance | R0=joint_geometry / R1=opencv_refined / R2=human_accepted/human_edited / R3=human_consensus |
| review_status | pending / accepted / edited / rejected / uncertain / adjudicated |
| 适用任务 | **历史内部治理**，不再是当前训练入口 |

### 禁止混用

两者**不得互相映射**。仓库训练入口必须使用 8-region adapter；不得把 10-region schema 或 R0-R3 tier 标签注入当前训练数据流。

---

## 3. GT 来源与项目接受状态区别

| 来源 | annotation_provenance | source_review_status | 用途 |
|---|---|---|---|
| V221 自动 Corrected Support | `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` | `NOT_REVIEWED` | **当前项目训练 GT**（Owner 接受） |
| 人工像素级标注 | `HUMAN_PIXEL_ANNOTATED` | `ACCEPTED` | 未使用 |
| OpenCV 自动分割 | `OPencv_AUTO_SEGMENTED` | `NOT_REVIEWED` | 不可用 |

---

## 4. 数据合同摘要

### 4.1 文件结构

```
SLP_8Region_Pressure_VAL_v1.1/
├── manifest/
│   ├── val_manifest.csv          # 主索引（4,590 rows）
│   ├── class_schema.json          # 类别合同
│   └── dataset_summary.json       # 汇总统计
├── qa/
│   └── dataset_summary.json       # 同上
└── samples/
    └── danaLab_{subject_id}_{cover}_{frame}/  # 如 danaLab_00001_uncover_000001/
        ├── pressure.npy           # (192, 84), float64
        ├── region_label.npy       # (192, 84), uint8
        ├── region_onehot.npy      # (9, 192, 84), uint8
        └── points.csv             # 冗余兼容表达
```

### 4.2 类别合同（`slp8-v2.2.1-canonical-export-v1.1`）

| ID | 名称 | 描述 |
|---|---|---|
| 0 | BACKGROUND | 未接触床面区域 |
| 1 | HEAD_NECK | 头部 + 颈部投影 |
| 2 | SHOULDER | 双肩合并区 |
| 3 | THORAX_BACK | 胸背区 |
| 4 | LUMBAR_WAIST | 腰腹区 |
| 5 | PELVIS_HIP | 骨盆/髋区 |
| 6 | ARM | 手臂区 |
| 7 | THIGH | 大腿合并区 |
| 8 | LOWER_LEG_FOOT | 小腿 + 足部合并区 |

**注意**：`LUMBAR_WAIST` ≠ `abdomen_waist`（10-region 路线词表）；`SHOULDER` 是左右合并的，不是 `shoulder_left`/`shoulder_right`；`ARM` 不在 10-region 路线中。

### 4.3 Adapter 关键约束

- 数据路径只能通过 CLI/config/local path 注入；**禁止提交 E:\ 或 D:\ 绝对路径**到代码和正式配置。
- Pressure 保持 raw PMarray response semantics；**不得在 adapter 内转换为 kPa**。
- Normalization 仅可在 TRAIN subjects 上 fit；**不在 adapter 内对全 102 人预先 fit**。
- 原始数据不修改；不写回 dataset 目录。
- 每样本加载时 fail-closed 检查：shape、dtype、finite、label range、onehot roundtrip。

---

## 5. 真实数据验证结果

**命令**：
```bash
uv run python scripts/validate_slp_8region_pressure_dataset.py \
  --dataset-root "E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\SLP_8Region_Pressure_VAL_v1.1" \
  --split-manifest "E:\TeamProjects\smarttopper-slp-a06\data\processed\slp\slp_subject_split_v0.1.json"
```

**结果**：实现者最终运行 `23.2s`；Codex Reviewer 最终修复后独立复跑 `148.3s`。两次均为 **ALL CHECKS PASSED（4,590 samples，0 failures）**。耗时受文件系统缓存与并发磁盘占用影响，不作为质量指标。

| 检查项 | 结果 |
|---|---|
| Manifest 行数 = 4,590 | PASS |
| Unique sample_ids = 4,590 | PASS |
| 无重复 sample_id | PASS |
| 唯一 subject 数 = 102 | PASS |
| Per-posture 分布 = SUPINE/LEFT/RIGHT 各 1,530 | PASS |
| 每 subject 45 帧 | PASS |
| 路径 containment（无 D:\ escapes）| PASS（13,770 条路径，0 escapes） |
| pressure_npy 列无绝对路径 | PASS |
| 压力 SHA256 spot-check（9/9） | PASS |
| 全量 pressure finite / dtype / shape / SHA256（4,590/4,590） | PASS |
| 全量 label range / dtype / shape（4,590/4,590） | PASS |
| 全量 one-hot binary / exclusivity / argmax roundtrip（4,590/4,590） | PASS |
| 全量 points.csv containment + is_file（4,590/4,590） | PASS |
| dataset_summary.json consistency | PASS |
| class_schema.json consistency（version/names/IDs） | PASS |
| A06 split SHA256 = `024f5abe` | PASS |
| 102 主体全部在 A06 split 中 | PASS |
| 0 个主体缺失（vs A06） | PASS |
| 0 个主体多余（vs A06） | PASS |
| Per-A06-split 样本数 = train 3,645 / val 450 / test 495 | PASS |
| Primary subject split 完整性（quarantine 不重分配 primary） | PASS |

---

## 6. A06 Split 兼容性结果

| 指标 | 值 |
|---|---|
| A06 split SHA256 | `024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706` |
| 102 主体全部兼容 | ✅ |
| Train subjects | 81 × 45 = 3,645 samples |
| Val subjects | 10 × 45 = 450 samples |
| Test subjects | 11 × 45 = 495 samples |
| 主体跨 split 重叠 | 0 |
| Quarantine 主体差异（simLab 隔离）| 不影响 danaLab primary |

---

## 7. 已验证

| 结论 | 证据 |
|---|---|
| 4,590 样本存在且结构完整 | manifest CSV 4,591 行（含 header）|
| 102 danaLab subjects × 45 frames | manifest 结构验证 |
| SUPINE/LEFT/RIGHT 各 1,530 | manifest posture 计数 |
| 路径全为相对路径且在 dataset root 内 | 13,770 条路径，0 escapes |
| pressure shape (192,84) float64 finite | 全量 4,590/4,590 PASS |
| region_label shape (192,84) uint8 range [0,8] | 全量 4,590/4,590 PASS |
| onehot shape (9,192,84) uint8、binary、互斥、roundtrip | 全量 4,590/4,590 PASS |
| SHA256 与 manifest 声明一致 | 全量 4,590/4,590 PASS |
| class_schema.json 与实际 class IDs 一致 | class_schema.json 9 classes |
| A06 split 完全兼容（102 主体，3645/450/495）| validator A06 section |
| adapter 可只读加载样本 | 66 dataset tests PASS |
| 回归测试 | 221 passed，1 skipped |
| governance 文档已更新 | 6 个治理文档已修订 |
| 旧 10-region 路线标记为历史 | AGENTS.md + backlog + plan |
| A10-A17 路线标记为 HOLD/SUPERSEDED | backlog 已更新 |
| B01 改为 READY | backlog 已更新 |

---

## 8. 合理推断

| 推断 | 依据 |
|---|---|
| 13,770 条路径全为 relative（CSV 列检查）| manifest CSV 结构 |
| A06 split 对 danaLab 主体分配稳定（SHA256 frozen）| A06 manifest freeze |
| 背景像素比 ~70.4% 是合理压力传感器空白区 | dataset_summary.json |

---

## 9. 未验证

| 项目 | 原因 |
|---|---|
| cover1/cover2 区域边界质量 | 当前数据集仅含 uncover |
| 区域标签的解剖学正确性 | 需要人工复核（A13 HOLD）；V221 自动来源 NOT_REVIEWED |
| simLab 数据不在本 GT 中 | 本 GT 仅含 danaLab |
| B01 训练表实际生成 | A09R 已验收；B01 尚未执行 |

---

## 10. 限制

| 限制 | 影响 | 缓解 |
|---|---|---|
| source_review_status = NOT_REVIEWED | 区域边界非人工逐帧验证 | 仅用于研究；产品需独立验证 |
| 仅 uncover | cover1/cover2 区域分割无 GT | A10-A17 HOLD；未来可选重新打开 |
| 无 ARM 的左右拆分 | 精细手臂区域分析受限 | 10-region 路线保留（HOLD）|
| PELVIS_HIP 是粗区域 | 非精确臀部分割 | 仅粗区域分析有效 |
| 无 simLab 数据 | 跨域泛化无法在本 GT 上验证 | A06 split 中 simLab 在 TEST held-out |
| V221 自动来源 | 不是医学级解剖真值 | 仅研究用；产品需独立临床验证 |

---

## 11. 下一 Gate

**Gate A09R**：`COMPLETE_WITH_LIMITATIONS` — Codex Reviewer 已验收。

**Gate B01（下一步）**：`TASK-SLP-B01` 已从 `BLOCKED_BY_A17` 改为 `READY`。

B01 入口条件：
- ✅ 8-region GT 数据合同冻结
- ✅ Adapter 可读（66 dataset tests PASS）
- ✅ A06 split 兼容性验证通过（3645/450/495，0 overlap）
- ✅ 全量 validator ALL CHECKS PASSED（4590/4590，0 failures）

**本阶段不开通**（A10-A17 路线已改为 HOLD/SUPERSEDED_FOR_CURRENT_SLP8_GT）：
- A10（A09R 后可重新打开，但不再是 B01 前置）
- A11–A17

---

## 12. 禁止结论

1. 不得将 V221 自动来源改称为人工像素级标注。
2. 不得将 SLP8 pressure GT 称为医学、皮肤界面应力或产品真值。
3. 不得将 8-region 数据与 10-region polygon 路线混用或映射。
4. 不得将 SHOULDER（合并）称为左右分肩区域；不得将 THIGH（合并）称为左右分大腿。
5. 不得将 LUMBAR_WAIST 用于腰椎定位或脊柱曲度分析。
6. 不得将 PELVIS_HIP 用于精确臀部分割。
7. 不得将 uncover-only 数据外推到 cover1/cover2 条件。
8. 不得将 R0/R1 伪标签混入当前训练数据流。
9. 不得将本 GT 用于自研顶垫产品验证。
10. 不得将 A10-A17 HOLD 路线误标记为 DONE 或 READY。

---

## 13. 文件变更摘要

| 文件 | 变更 |
|---|---|
| `configs/annotations/slp_8region_pressure_gt_v1.1.schema.json` | 新增：8-region GT 机器合同 |
| `src/topper_perception/io/slp_8region_pressure_dataset.py` | 新增：adapter（CSV 读取、fail-closed 验证、路径 containment） |
| `scripts/validate_slp_8region_pressure_dataset.py` | 新增：全量 validator（6 节，ALL PASS） |
| `tests/test_slp_8region_pressure_dataset.py` | 新增：66 单元测试（adapter、class schema、filtering、path、onehot required） |
| `AGENTS.md` | 更新：SLP truth boundary 增加 SLP8 GT 说明，标记 10-region 为历史 |
| `docs/PROJECT_STATUS.md` | 更新：SLP section 新增 S1_A09R 行；数据边界新增 SLP8 GT 说明；S4 改为 HOLD |
| `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` | 更新：A09 改为 SUPERSEDED_BY_A09R；A09R DONE_WITH_LIMITATIONS；A10-A16 改为 HOLD；A17 SUPERSEDED；B01 改为 READY |
| `docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md` | 更新：关键决策注记 Q4 栏更新为 SLP8 GT；关键更正注记增加 Owner 接受说明 |
| `COLLABORATION_WORKFLOW.md` | 更新：当前 SLP8 路线和真值边界 |
| `docs/VALIDATION_WORKFLOW_MASTER.md` | 更新：SLP8 已接入；PoPu HOLD 与 SLP8 B01 分离 |
| `docs/stage_reports/S1_A09_SLP_REGION_ONTOLOGY_CONTRACT_FREEZE_v0.1.md` | 历史 A09 报告标记为 SUPERSEDED_BY_A09R |
| `configs/annotations/slp_region_annotation_v0.1.schema.json` | 更新：移除 final_polygon 顶层 required；新增 5 个 allOf tier/source/status 块（R0 null polygon，R1/R2/R3 必填） |
| `tests/test_slp_region_annotation_schema.py` | 更新：从 4 个静态检查扩展到 76 个 Draft 2020-12 实例验证测试 |
| `pyproject.toml` | 更新：新增 jsonschema dev 依赖 |
| `uv.lock` | 自动更新 |
| `docs/stage_reports/S1_A09R_SLP8_GT_CONTRACT_REALIGN_v0.1.md` | 新增：本报告 |

---

## 14. Reviewer Checklist

- [x] Owner 决策已记录（annotation_provenance、source_review_status、禁止改写）
- [x] 8-region GT 合同与实际 class_schema.json 一致
- [x] 10-region 合同不再被误用为训练入口（已标记历史）
- [x] Adapter 可以只读加载真实样本
- [x] 全量 validator 通过（4590 samples，0 failures；Codex Reviewer 独立复跑）
- [x] A06 split 兼容性验证通过（102 主体，3645/450/495，0 overlap）
- [x] 66 dataset tests PASS
- [x] 76 Draft 2020-12 schema 测试 PASS（来自 A09）
- [x] 221 regression tests PASS（6.27s）
- [x] 287 tests 总套件 PASS（66 dataset + 221 regression；另 1 skipped）
- [x] git diff --check PASS（working-copy CRLF 警告为 autocrlf 非 whitespace 错误）
- [x] 未修改 raw dataset
- [x] 未运行 GPU/Mini/Full 训练
- [x] Reviewer 验收后创建本任务 commit；push NOT RUN
- [x] 禁止结论已列出，无违反

---

*本报告由 Claude Code（Mavis）生成并由 Codex Reviewer 独立复核、定点收尾。Owner 决策编号：SLP-8REGION-GT-ACCEPT-20260824。*
