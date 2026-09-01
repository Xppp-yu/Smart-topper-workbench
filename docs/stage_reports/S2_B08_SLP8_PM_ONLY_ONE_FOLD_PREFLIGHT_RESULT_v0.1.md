# S2_B08 — SLP8 PM-only RTX 4090 one-fold preflight 结果

**TASK-ID**: `TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1`
**EXP-ID**: `EXP-SLP-B08-PM-FULL-ONE-FOLD-PREFLIGHT-20260831-AUTODL-R03`
**状态**: `ACCEPT / PREFLIGHT_PASSED / TEST_DENIED`

## 1. 冻结身份

- runner Git SHA：`02fb364902736a64ee8708440f0dd0bdddf860bc`，dirty=false。
- candidate/fold/seed：`slp8_resunet_lite_v0.1 / fold_1 / 42`。
- max epochs/batch：`30 / 16`；best epoch：`22`。
- device：NVIDIA GeForce RTX 4090；Torch `2.13.0+cu130`；strict deterministic。
- protocol SHA：`98314e70590094496418c0c8a43bb8b62497841a9b2437b9306f3d247e382c83`。
- fold SHA：`0ac344c9bb89cc71757c796096a8e2c63e8b4bb1cf9eeea2cab875fd2add8b2b`。
- freeze SHA：`42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04`。
- split SHA：`024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706`。

## 2. 结果与预算

| 项目 | 结果 | Gate |
|---|---:|---|
| TRAIN | 3,240 samples / 72 subjects | PASS |
| VAL OOF | 855 unique samples / 19 subjects | PASS |
| wall | 155.32885087199975 s | PASS (`<=900 s`) |
| peak CUDA | 368.764416 MiB | PASS (`<=8192 MiB`) |
| best checkpoint reload | prediction hash consistent | PASS |
| terminal | exactly `DONE.json` | PASS |
| TEST | 0 | PASS |

Best checkpoint SHA-256：
`51db02cf6e26a11cef27b6627437cedaafe269357ca214d4662b7941a57c2506`。

## 3. 本地归档与独立审计

- Archive：`B08_ONE_FOLD_PREFLIGHT_R01_R02_R03_EVIDENCE_20260901.tar.gz`。
- Archive SHA-256：
  `55aa5a4c708a1ba9b2f9ee89a8d05d937e61f4edcd02bdf83b1f5600b478a298`；
  sidecar 与本地重算一致。
- 本地只读审计：DONE==manifest；identity/counts/budgets/reload/TEST gates 通过；
  best checkpoint SHA 匹配；OOF predictions/targets 均 855，sample IDs 855 unique，
  subjects 19；结果 `LOCAL_ARCHIVE_AUDIT_PASSED`。
- Archive、checkpoint、OOF 保存在本地 evidence 目录，不提交 Git。

## 4. 失败历史与限制

- R01：真实 `RegionSample` 被误作 mapping，在首 epoch 前失败；无规范 root terminal。
- R02：strict CUDA NLLLoss2d 在首 loss forward 失败，zero optimizer steps；FAILED
  terminal 生效，但 error 被空 OOF stack 二次异常覆盖。
- R03 证明当前 runner 的一个 fold/candidate/seed 链路可运行，不是 30-unit Full，
  不产生候选比较、最终模型或 TEST 结论。
- SLP8 reference GT 为 pressure-derived project reference，非人工像素级/医学/产品 GT。

## 5. Verified / Inferred / Unverified / 下一 Gate

- **Verified**：上述 identity、counts、terminal、resource、reload、OOF、archive 与
  TEST=0。
- **Inferred**：同 runner 可调度 B07 的其他 29 units；尚未实际运行，不能视为通过。
- **Unverified**：30-unit Full 的总 wall/resource、跨 fold/seed OOF、候选公平比较、
  最差受试者和任何 TEST 指标。
- **下一 Gate**：B09 Full 运行准备记录与 Owner 独立授权。当前 Full/TEST 均未授权。
