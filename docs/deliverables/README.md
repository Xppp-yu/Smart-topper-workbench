# SLP 项目交付物索引

本文档索引 SLP 项目所有阶段交付物。

## 阶段索引

| Stage | TASK-ID | 名称 | 状态 | 交付物 |
|-------|---------|------|------|--------|
| S2-B01 | TASK-SLP-B01-... | Training Table Freeze | DONE | [B01 Stage Report](../stage_reports/S2_B01_SLP8_TRAINING_TABLE_FREEZE_v0.1.md) |
| S2-B02 | TASK-SLP-B02-... | Non-Learning Region Baseline | DONE | [B02 Stage Report](../stage_reports/S2_B02_SLP8_NON_LEARNING_REGION_BASELINE_v0.2.md) |
| S2-B03 | TASK-SLP-B03-... | PM-only Region Smoke | DONE | [B03 交付说明](SLP/B03_PM_ONLY_REGION_SMOKE_v0.1.md) |
| S2-B04 | TASK-SLP-B04-... | PM-only Region Mini | DONE_WITH_LIMITATIONS | [B04 Protocol/Runner](SLP/B04_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md)；[B04 R05 Results](SLP/B04_PM_ONLY_REGION_MINI_RESULTS_v0.1.md) |
| S2-B04A | TASK-SLP-B04A-PROTOCOL-FREEZE-v0.1 | PM-only Architecture Expansion Mini Protocol | PROTOCOL_ACCEPTED / IMPLEMENTATION_NOT_STARTED（R03） | [B04A Protocol](../stage_reports/S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md)；[B04A 交付说明](SLP/B04A_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md)；[B04A 任务合同](../tasks/TASK_SLP_B04A_PROTOCOL_FREEZE_v0.1.md)；[配置](../../configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json)；3 候选：SmallUNet(118,121)/ResUNet-lite(120,809,1x1 Conv2d)/DeepLabV3+-lite(53,449,Option A)；SegFormer-B0 DEFERRED；阈值 0.355644；Reviewer 56 tests PASS |

## 最新 Stage Report

- [S2-B04A Protocol (R03)](../stage_reports/S2_B04A_SLP8_PM_ARCHITECTURE_EXPANSION_MINI_PROTOCOL_v0.1.md) ← NEW
- [S2-B04 Protocol/Runner](../stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md)
- [S2-B04 R05 Results](../stage_reports/S2_B04_SLP8_PM_ONLY_REGION_MINI_RESULTS_v0.1.md)
- [S2-B03 Stage Report](../stage_reports/S2_B03_SLP8_PM_ONLY_REGION_SMOKE_v0.1.md)

## 运行说明

各阶段运行说明请参考对应交付文档。

---

**更新时间:** 2026-08-29（B04A 协议冻结）
**维护者:** Smart Topper Team
