# TASK-POPU-P7-RECOVERED-ARCHIVE-REGISTER-v0.1

状态：`ACCEPTED_WITH_LIMITATIONS`

## Objective

登记 P7 原始临时归档消失后的当前证据保管状态，保留原 tar 的历史验证结论，同时为由已验收解压树重新封装的恢复包建立独立 identity。

## Allowed changes

- `docs/PROJECT_STATUS.md`
- `docs/stage_reports/P7_POPU_SOFTWARE_ROBUSTNESS_FULL_RESULTS_v0.1.md`
- `docs/stage_reports/POPU_PHASE1_COMPLETE_VALIDATION_REPORT_v0.1.md`
- `docs/stage_reports/P7_RECOVERED_ARCHIVE_REGISTER_v0.1.md`
- 本任务卡

## Prohibited

- 修改 P7 指标、anchor、模型、配置或结论；
- 把恢复包称为原 tar；
- 用恢复包的新 SHA 覆盖原始冻结 SHA；
- 放宽原始归档 hash gate；
- 重跑 P7、模型推理或任何 GPU 实验；
- 将大归档提交 Git。

## Acceptance criteria

- 明确原始 Temp 路径当前不存在；
- 保留原始归档在 2026-08-21 已验证的历史事实；
- 恢复包路径使用符号化本地证据根，不提交机器绝对路径；
- 恢复包文件数、大小和 SHA-256 完整登记；
- 明确恢复包不是原 tar，不能替换历史 identity；
- `git diff --check` 和文档链接检查通过。

## Reviewer result

- 日期：2026-08-29；
- verdict：`ACCEPT`；
- 本次只接受证据保管登记；
- P7 研究结论无变化，分析重跑为 `NOT RUN`。
