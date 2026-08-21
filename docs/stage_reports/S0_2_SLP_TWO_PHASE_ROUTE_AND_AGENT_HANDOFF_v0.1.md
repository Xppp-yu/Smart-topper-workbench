# S0.2 — SLP 两阶段路线与多 Agent 交接冻结 v0.1

## 1. 结论

**状态：PLANNING_COMPLETE — READY_FOR_CONTINUOUS_DEVELOPMENT。**

已审阅队长《智能睡眠顶垫支撑控制——感知层算法开发清单 V1.0》，并将其拆分为 SLP 可验证任务和必须留给自研压力/气囊数据的任务。SLP 后续采用两阶段路线：

1. Region Reference 形成前：配对、坐标、节点、人体轴、OpenCV 预标注和人工复核；
2. Region Reference 形成后：区域训练集、单模态、遮盖压力测试、有限融合、Full 和拒识。

阶段边界冻结为 `SLP Region Reference v1.0` 经人工复核和 Reviewer 接受，而不是 OpenCV 自动输出。R0/R1 是伪标签；R2/R3 才是默认训练参考。

## 2. 队长清单的任务边界

SLP 主要承担人体位置、方向、姿态、节点、粗区域和遮盖鲁棒性。区域绝对载荷、空床基线、坏点/饱和、稳定时序、异常支撑状态、动作收益和回退策略不能由 SLP 完整验证，必须进入自研传感器与气囊同步实验路线。

## 3. 新增与更新入口

- `docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md`
- `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`
- `configs/annotations/slp_region_annotation_v0.1.schema.json`
- `tests/test_slp_region_annotation_schema.py`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/PRIVATE_GITHUB_AND_AGENT_HANDOFF_CHECKLIST_v0.1.md`
- `docs/evidence/slp_s0_inventory_summary_v0.1.json`
- README、PROJECT_STATUS 和首轮 SLP 路线的交叉入口已更新。

## 4. OpenCV 状态

- 当前环境 `opencv_available=False`，尚未安装或实现 OpenCV 标注器。
- 这是有意的依赖 Gate：A03/A04 坐标与 homography 未通过前，不生成真实区域 proposal。
- TASK-SLP-A11 将加入单一可选 `opencv-python-headless` 依赖；若确需 contrib 功能再替换，禁止同时安装普通版和 contrib 版。
- OpenCV 只产生 R1 proposal；人工复核工具产生 R2/R3。

## 5. 私密 GitHub 准备度

- 当前分支 `main`，尚无 remote。
- 原始数据、local path config、outputs/processed 已被忽略。
- 候选上传集合 181 个文件、1,333,100 bytes，最大单文件 108,691 bytes。
- 当前工作区高信号 secret pattern 扫描无命中。
- `gitleaks` 未安装，Git 历史级 secret scan 仍是上传前 Gate。
- 当前变更尚未提交，也未推送。

## 6. 验证

```text
定向：8 passed in 0.89s
全仓：528 passed, 14 warnings in 36.78s
```

14 条均来自既有 joblib/NumPy shape DeprecationWarning，没有 SLP 新增测试失败。

## 7. 下一批可并行任务

- TASK-SLP-A01：许可与数据版本；
- TASK-SLP-A02：内容解码与数值 QA；
- TASK-SLP-A03：Frame Master Index；
- TASK-SLP-A09：Region schema/ontology Reviewer 审查。

A04 依赖 A03；A10/A11 OpenCV 线依赖坐标、人体轴和 schema，不应越级。

## 8. 尚未验证

- SLP 许可的明确商业/再分发条款；
- 全量内容可解码和数值范围；
- homography 方向和往返误差；
- OpenCV 预标注准确率和人工复核成本；
- Region Reference v1.0；
- 任何 SLP 区域模型结果。

## 9. 不能得出的结论

- 路线和 schema 完成不等于 OpenCV 已实现；
- OpenCV proposal 不等于区域真值；
- SLP 区域参考不等于自研顶垫产品验证；
- 私密 GitHub 准备度检查不等于已经创建或连接远程仓库。
