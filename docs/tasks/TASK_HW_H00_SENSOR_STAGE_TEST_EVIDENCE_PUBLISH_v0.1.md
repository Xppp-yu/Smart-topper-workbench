# TASK-HW-H00-SENSOR-STAGE-TEST-EVIDENCE-PUBLISH-v0.1

状态：`IMPLEMENTED / REVIEW_PENDING`

## Objective

把本地 `outputs/sensor_validation/` 中已有的自采 32×32 压力传感器阶段性报告整理为可进入私有 GitHub 的脱敏证据快照，同时保持原始 CSV/XLS 和生成产物在 Git 外。

## Allowed changes

- `.gitignore`
- `docs/PROJECT_STATUS.md`
- `docs/tasks/TASK_HW_H00_SENSOR_STAGE_TEST_EVIDENCE_PUBLISH_v0.1.md`
- `docs/stage_reports/H0_SELF_COLLECTED_SENSOR_STAGE_TEST_v0.1.md`
- `docs/evidence/sensor_validation/*.csv`

## Prohibited

- 提交原始压力 CSV/XLS、DOCX、PNG 或本机路径；
- 修改原始数据或本地 `outputs/sensor_validation/` 产物；
- 把 mapped ADC 称为 kPa、N、kg 或物理标定压力；
- 宣称传感器精度、硬件可靠性、人体、舒适性、医疗、整夜或控制安全验证完成。

## Acceptance criteria

- Git 只包含脱敏 Markdown 和小型聚合 CSV；
- 不包含绝对路径、账号、凭据或原始样本；
- 报告区分已有证据快照与本次实际验证；
- `outputs/sensor_validation/` 明确保持为本地生成目录；
- `git diff --check`、文档链接与 CSV 对照检查通过；
- 结束时记录 `NOT RUN` 的独立原始数据复算。

## Reviewer checklist

- 三份入库 CSV 与本地摘要逐行一致；
- 原始数据和二进制生成物未暂存；
- 证据边界和下一 Gate 清楚；
- 本任务不改变 SLP B04A 的当前优先级和授权状态。
