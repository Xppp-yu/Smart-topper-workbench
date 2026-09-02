# S2 B09 SLP8 PM-only Full R01 结果、审计与 unit identity 修复 v0.1

状态：`OWNER_ACCEPTED_WITH_LIMITATIONS / B09_COMPLETE / TEST_DENIED`

TASK-ID：`TASK-SLP-B09-UNIT-IDENTITY-CARRIER-FIX-AND-R01-ADJUDICATION-v0.1`

## 1. 运行对象

- EXP-ID：`EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01`
- Runner：`8b3ebdaa021405790b6137bff581acc490d8a024`
- 环境：AutoDL RTX 4090
- 数据范围：91 development subjects / 4,095 TRAIN+VAL samples
- 设计：2 candidates × 5 subject-isolated folds × 3 seeds = 30 units
- TEST：0；未授权、未读取、未评价。

原始证据归档：

- 文件：`EXP-SLP-B09-PM-FULL-30-UNIT-20260901-AUTODL-R01-ORIGINAL.tar.gz`
- 大小：147,939,524 bytes
- SHA-256：`68156598a47ae65ba33d26f4005f9d9fdc8ec67ff24d43ffd605c19847ca5918`
- 归档可读，共 271 entries；本轮仅在 ignored `outputs/` 下解包审计，未修改或重新打包原始证据。

## 2. 已验证运行结果

- terminal：`DONE`
- units：30 DONE / 0 FAILED / 0 STOPPED
- budget：`budget_ok=true`
- total wall：3,322.39 s
- 每个 candidate × seed 的 OOF subject coverage：91
- TEST carriers：`test_access=false`，其余五个 TEST 计数均为严格整数 0

冻结选择规则给出的结果：

| candidate | mean pooled IoU | mean pooled Dice | mean worst-subject IoU | decision |
|---|---:|---:|---:|---|
| `slp8_deeplabv3plus_lite_v0.1` | 0.494134 | 0.657759 | 0.350911 | WINNER |
| `slp8_resunet_lite_v0.1` | 0.456795 | 0.622978 | 0.311755 | ELIMINATED |

IoU lead 为 `0.037338`，超过冻结的 `0.02` margin，因此开发期 winner 是
`slp8_deeplabv3plus_lite_v0.1`。

## 3. 原始证据审计失败

在冻结 runner SHA 上执行只读 `--audit-only` 得到：

```text
summary: 27 OK / 30 ERR
B09_RUN_PREPARATION_VALIDATION_FAILED
```

30 个错误全部是同一项：每个 `units/*/complete.json.identity` 缺少
`git_dirty=false`。run-level `DONE.json`、`manifest.json`、`status.json` 和
`resume_identity.json` 均记录相同 Git SHA，且 run-level `git_dirty=false`；每个
unit 也记录正确 `git_commit`。因此这是 unit completion carrier 的字段遗漏，不是
训练、fold、OOF、预算、TEST 或模型选择计算错误。

## 4. 根因与修复

`run_full()` 构造 `unit_expected_identity` 时继承了 run identity，但 run identity
本身没有 `git_dirty`；随后该对象原样写入 `complete.json`。修复是在同一冻结点
显式加入：

```python
"git_dirty": config.git_dirty
```

回归测试现在对生成的 30 个 `complete.json` 逐个验证 `git_commit` 与严格布尔
`git_dirty`，并保留第二次运行零重训、carrier 字节不变的 resume 合同。

## 5. 验证

- 定向 30-unit carrier + resume：`2 passed`
- B08/B09 runner、CLI bridge、run preparation、Markdown links：`178 passed`
- GPU Full：`NOT RUN`（本任务禁止）
- TEST：`0`

首次在新工作树执行 `uv run pytest` 因该新 `.venv` 未安装 PyTorch而 collection
失败；随后使用主工作树兼容环境（PyTorch `2.12.1+cu126`）并将 `PYTHONPATH`
指向本任务工作树，以上测试全部通过。该环境问题不属于代码回归。

## 6. 裁决

Codex 建议：`ACCEPT_WITH_LIMITATIONS`，不要求 R02 重跑。Owner 于
2026-09-02 确认该裁决，R01 正式按永久披露的 provenance 限制接受。

理由：

1. 缺陷只影响 unit provenance carrier 的一个冗余字段，不进入训练、OOF、指标或
   winner 计算。
2. run-level 多个独立 carrier 均固定 `8b3ebda` 且 `git_dirty=false`，原始归档
   hash 已冻结，足以把 30 个 unit 绑定到同一次 clean run，但 unit 文件单独拿出时
   不能自证 clean，必须永久披露。
3. TEST=0、fold/manifest/hash、30-unit 完整性、OOF coverage、预算与 winner 规则
   均通过审计；重新消耗 GPU 不会修复原始证据，只会生成新的 R02 证据。
4. writer 已修复并由生成式 30-unit 回归覆盖，未来运行不会重复该缺陷。

若 Owner 或外部审查要求“每个 unit 单文件必须独立通过原协议”，则只能创建新
EXP-ID 运行 R02；不得修补 R01 原始 JSON。否则 R01 可作为带永久 provenance
限制的开发期 Full 证据，并进入 B10。

## 7. 结论边界

### Verified

- 原始归档 hash、可读性、271 entries。
- 30/30 DONE、TEST=0、OOF subject coverage、预算和冻结 winner 结果。
- audit-only 的 30 个错误均为 unit `identity.git_dirty` 缺失。
- 修复后相关测试 178 passed。

### Inferred

- 由于 run-level clean Git carriers 与全部 unit Git SHA 一致，unit 缺字段没有改变
  数值计算；这是由代码路径和载体一致性支持的推断，不等于修改了原始证据。

### Unverified

- 未执行第二次真实 GPU Full。
- 未执行任何 TEST evaluation。
- 未验证 cover、自研硬件、舒适性、医疗、整夜或气囊控制效果。

### Limitations

- R01 的 30 个原始 unit `complete.json` 永久缺少 `identity.git_dirty`，原始
  `--audit-only` 将永久保持 `27 OK / 30 ERR`。
- SLP8 reference GT 为 pressure-only、uncover、`source_review_status=NOT_REVIEWED`，
  不是人工像素级、医学或产品 GT。

### Next Gate

`TASK-SLP-B10-UNKNOWN-REJECT-READY-TO-DRAFT / TEST_DENIED`
