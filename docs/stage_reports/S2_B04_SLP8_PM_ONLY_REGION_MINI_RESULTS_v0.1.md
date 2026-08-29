# Stage Results: S2-B04 — SLP8 PM-only Region Mini R05

TASK-ID：`TASK-SLP-B04-PM-ONLY-REGION-MINI-PROTOCOL-AND-RUNNER-v0.1`

EXP-ID：`EXP-SLP-B04-PM-REGION-MINI-20260828-AUTODL-R05`

状态：`DONE_WITH_LIMITATIONS`

本文件只记录 R05 真实运行结果。冻结协议、Runner 和 R03/R04 历史见 [B04 Protocol](S2_B04_SLP8_PM_ONLY_REGION_MINI_PROTOCOL_v0.1.md)。

## Input and environment

| 项目 | 值 |
|---|---|
| 运行现场源码记录 | `72fbe67`（未内嵌于产物） |
| 数据 | B01 TRAIN 3,645 / VAL 450 / TEST 0 |
| 受试者 | TRAIN 81 / VAL 10，交集 0 |
| 数据边界 | danaLab / uncover / raw PMarray response |
| 参考标签 | `V221_CORRECTED_SUPPORT_AUTO_ACCEPTED` / `NOT_REVIEWED` |
| GPU | NVIDIA GeForce RTX 4090 |
| Python / PyTorch | 3.12.3 / 2.8.0+cu128 |
| Seed / epoch / batch | 42 / max 20 / 16 |
| Gate | VAL fixed foreground Macro IoU `>=0.205644` |

## Results

| 候选 | Macro IoU | Macro Dice | VAL loss | Worst-subject IoU | 决策 |
|---|---:|---:|---:|---:|---|
| TinyFCN | 0.051631 | 0.089858 | 1.659858 | 0.048360 | `NOT_FEASIBLE` |
| SmallUNet | 0.439625 | 0.607810 | 0.732006 | 0.308241 | `FEASIBLE` |

SmallUNet 的 8 个前景区域 IoU 为 `0.339576–0.523164`，没有零预测区域；pixel accuracy `0.841269`，归一化质心误差均值 `0.042190`。3/3,600 条质心记录因 GT 区域缺失被显式标为无效。

两候选 checkpoint reload 均为 `max_abs_diff=0.0` 且 prediction hash 一致。总 wall time `231.18 s`，峰值 CUDA 显存 `362.99 MiB`，预算状态均为 `ok`，0 candidate failed。

## Failures preserved and fixes

| 现场问题 | 修复提交 |
|---|---|
| 真实入口缺少 subject isolation helper import | `f3fb7d9` |
| CUDA 严格确定性下二维 NLL loss 不受支持 | `c4ebc5d` |
| 参数变化审计混用 CPU/CUDA tensor | `762f44e` |
| GT 区域缺失时质心函数解包 None | `72fbe67` |

R01–R04 失败产物保留；R05 使用新 EXP-ID，没有覆盖历史证据。

## Archive

- 本地 ignored archive：`<WORKBENCH>/outputs/evidence_archives/SLP_B04_R05/EXP-SLP-B04-PM-REGION-MINI-20260828-AUTODL-R05.tar.gz`
- SHA-256：`57885db25dba04a3f9d82666b47dbcc85f030f9842a0c20764d20133ead87c19`
- Windows Reviewer 已独立重算并确认匹配。

## Verified

- B01 TRAIN/VAL 合同、subject isolation、TEST=0、CUDA 执行、训练、指标、reload、预算和互斥终态均有产物证据；
- SmallUNet 达到 B04 历史 Gate；TinyFCN 未达到。

## Inferred

- SmallUNet 适合作为 B04A incumbent；这不是最终架构排名。

## Unverified

- 独立两次真实 GPU byte-identical 复现；
- TEST、cover1/cover2、其他 setting、自研硬件和产品环境。

## Limitations and next gate

- 产物未内嵌 Git commit，后续 Runner 必须补齐 identity；
- 标签不是人工像素级或产品 GT；
- B04 历史完成不再直接解锁 B07。Owner 后续决定先执行 B04A，B07 状态由此改为 `BLOCKED_BY_B04A`。
