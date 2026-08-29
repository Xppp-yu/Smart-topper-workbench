# Stage Report: PoPu P7 Recovered Archive Register v0.1

**TASK-ID:** `TASK-POPU-P7-RECOVERED-ARCHIVE-REGISTER-v0.1`

**状态:** `ACCEPTED_WITH_LIMITATIONS`

## Objective

记录 `EXP-P7-FULL-20260820-R02` 的当前本地证据保管状态，不改写 2026-08-21 对原始 tar 的历史验收。

## Historical source

原 tar 在 2026-08-21 验证时：

- 文件名：`EXP-P7-FULL-20260820-R02.tar.gz`；
- 大小：`672,702,773` bytes；
- SHA-256：`cbaffa74878b149e546a42826ae373442c62683af890362684f80963e7fddda1`；
- 文件数：`2,163`；
- 验证结果：原 tar 当时字节哈希和归档内证据均通过 Reviewer 复核。

该 tar 原位于 Windows Temp，2026-08-29 检查时已不存在。路径消失不追溯性地否定当时完成的哈希验证，但意味着当前不能再从原始 tar 字节重新执行 opt-in integration。

## Recovery archive

由此前已验收的解压目录重新封装：

```text
<LOCAL_EVIDENCE_ROOT>/popu/
  EXP-P7-FULL-20260820-R02-recovered-extract-20260829.tar.gz
```

| 字段 | 值 |
|---|---|
| 解压树文件数 | `2,163` |
| 恢复包文件数 | `2,163` |
| 恢复包大小 | `672,497,970` bytes |
| 恢复包 SHA-256 | `19e03e5665aa7af559b4b15e21b7e989e3e947654de686a60fd2cede4c1a4f8b` |
| tar 列表可读 | PASS |

恢复包的新 SHA 和大小与原 tar 不同，这是重新封装导致的预期结果。它是当前证据保管副本，不是原始运行归档。

## Verified

- 恢复包存在且可完整列出；
- 源解压树和恢复包均包含 2,163 个文件；
- 恢复包 SHA-256 已在封装后计算；
- 原始 Temp 路径当前不存在；
- 恢复包未加入 Git，Git 只保存脱敏 identity。

## Inferred

- 重新封装足以保存当前解压证据树，便于后续人工恢复和文件级审计。

## Unverified / NOT RUN

- 恢复包与原 tar 的 byte-identical 一致性：不成立且不宣称；
- 原 tar 当前 SHA 重算：原文件不存在；
- P7 分析、anchor 和指标重新计算：`NOT RUN`；
- 模型训练、推理和 GPU：`NOT RUN`。

## Limitations

- 恢复包不能替换冻结的原 tar SHA；
- 当前 opt-in integration test 仍绑定原始临时路径与原始 hash gate；
- 若未来需要从恢复包重新分析，必须新建 recovery-specific TASK/EXP，不能静默放宽旧合同。

## Next gate

P7 证据保管登记完成。PoPu P7 仍保持 `COMPLETE — SOFTWARE_PERTURBATION_ONLY`，不解锁任何硬件或产品结论。当前项目主线返回 SLP B04A Protocol Freeze。
