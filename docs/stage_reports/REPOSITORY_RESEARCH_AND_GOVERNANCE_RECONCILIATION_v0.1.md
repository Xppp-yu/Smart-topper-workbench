# Repository Research and Governance Reconciliation v0.1

TASK-ID：`TASK-REPO-SLP-STATUS-AND-ROUTE-RECONCILIATION-v0.1`

状态：`DOCUMENTATION_IMPLEMENTED / REVIEW_PENDING`

## Objective

收敛远端 `main @ acf57c1` 上的状态漂移，前瞻性插入 B04A 受控架构扩展，并补齐 Full 前的运行 identity 与 TEST 使用边界。本任务只修改文档，不启动任何研究计算。

## Verified before editing

- B04 R05 已完成：TRAIN 3,645 / VAL 450 / TEST 0；SmallUNet `0.439625`，TinyFCN `0.051631`；
- B04 的实际 FEASIBLE 判定主要由 fixed foreground Macro IoU `>=0.205644` 决定；per-region 与 worst-subject 有报告但不是冻结晋级 Gate；
- SmallUNet 8 区 IoU 均非零，worst-subject Macro IoU `0.308241`，checkpoint reload `max_abs_diff=0.0`；
- B04 R05 归档 SHA-256 已验证，但产物没有内嵌 Git commit；
- PoPu P7 Full 软件扰动报告已记录 14 conditions × 5 seeds × 15 folds 并支持软件扰动阶段收口；
- README、PROJECT_STATUS、SLP 总计划和 Backlog 存在不同步状态；
- A11 的 OpenCV optional/headless 说明与当前基础依赖 `opencv-python` 不一致。

## Owner route decision implemented

```text
B04 historical Mini (complete)
  -> B04A controlled architecture expansion
     -> protocol review
     -> implementation and Smoke
     -> separately authorised GPU Mini
     -> independent review, keep at most 1-2 candidates
  -> B07 Full protocol freeze
  -> B08 runner and one-fold preflight
  -> B09 development Full with TEST=0
  -> frozen candidate/protocol
  -> B09T one-time final TEST
```

## Documentation changes

- README 改为导航摘要，并指向 PROJECT_STATUS 作为唯一实时状态；
- PROJECT_STATUS 修正 PoPu P7，并加入 B04A/B07 状态；
- SLP 总计划和 Backlog 新增 B04A、增强 Gate、分阶段执行和 TEST 独立 Gate；
- 实验治理要求正式产物自带 Git/config/data/split/model identity；
- B04 Protocol 与 R05 Results 分开提供入口；
- A11 OpenCV 依赖矛盾被记录为未来重新打开前的决策项；
- 协作角色边界不因聊天中的能力描述而改变。

## Verified

- 本任务基于最新远端 `main @ acf57c1` 的独立 worktree；
- 原工作区的未跟踪 `outputs/sensor_validation/` 未被读取、移动、修改或暂存；
- B04 和 PoPu P7 数值来自已提交阶段报告与本地已验证证据记录；
- 所有修改限定为 Markdown 文档。

## Inferred

- 扩大架构假设覆盖范围会提高 B07/B09 的研究解释力；具体模型优劣仍必须由未来预注册实验回答。

## Unverified / NOT RUN

- B04A 模型实现：`NOT RUN / NOT IMPLEMENTED`；
- CPU/CUDA Smoke：`NOT RUN`；
- B04A GPU Mini：`NOT RUN`；
- B07/B08/B09：`NOT RUN`；
- TEST：`NOT READ / NOT RUN`；
- 独立两次 GPU byte-identical 复现：`NOT RUN`。

## Limitations

- B04A 的精确超参数、seed、margin、class-collapse 阈值、资源 tier 和候选最终名单仍需在独立协议任务中冻结；
- SegFormer-B0 是否进入本轮取决于单通道输入和预训练公平性合同；
- SLP8 标签仍为 `NOT_REVIEWED` 自动接受参考，不是人工像素级或产品 GT。

## Next gate

Reviewer 先验收本次文档一致性；之后只允许启动 B04A 协议冻结任务。协议获接受前不得实现候选，真实 GPU Mini 仍需 Owner 另行授权。
