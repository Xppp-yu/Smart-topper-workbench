# S2 B11 SLP8 研究候选冻结 v0.1

状态：`CANDIDATE_CONTRACT_ACCEPTED / FINAL_FIT_NOT_RUN / TEST_DENIED`

TASK-ID：`TASK-SLP-B11-RESEARCH-CANDIDATE-FREEZE-v0.1`

## 冻结结论

研究候选冻结为 `slp8_pm_research_candidate_v0.1`，模型族为
`slp8_deeplabv3plus_lite_v0.1`。开发期依据是 B09 91-subject TRAIN+VAL OOF：
mean pooled IoU `0.494134`、Dice `0.657759`、mean worst-subject IoU `0.350911`。

输入仅为 danaLab/uncover 单帧 raw PMarray response；输出为 192×84、background +
8 regions。它不是人工像素级、医学或产品 GT。

## 最终开发集拟合合同

本阶段仅冻结、未运行：

| seed | B09 五折 best epochs | 固定 final-fit epoch（中位数） |
|---:|---|---:|
| 42 | 19, 10, 15, 7, 22 | 15 |
| 123 | 15, 25, 20, 20, 8 | 20 |
| 2026 | 19, 7, 14, 12, 8 | 12 |

未来 B11F 独立 GPU 任务在全部 91 development subjects / 4,095 samples 上训练三个
模型。primary prediction 为三模型逐像素多数票；可选研究拒识仅在 3/3 hard mask
一致时输出 region，否则 `UNKNOWN_REGION`。

## 限制

- B10 证明 3/3 一致区域仍有 8.7916% pixel error，且 55.4078% 的原始错误为三
  seed 一致错判；该规则不是安全机制。
- 没有 logits/probabilities，概率校准与 OOD detection 均冻结为 `NOT_AVAILABLE`。
- B09 R01 unit `complete.json` 漏 `git_dirty` 的 provenance 限制永久保留。
- cover、自研硬件、舒适性、医疗、整夜、气囊控制均未验证。

## 验证

- candidate validator：`1 OK / 0 ERR`
- candidate tests + Markdown links：`9 passed`
- `py_compile`：PASS
- GPU / final fit：`NOT RUN`
- TEST：0

## Verified

- B09/B10 证据 hash、winner、开发指标、三 seed 和按中位数冻结的 epoch。
- 配置对 TEST、hash、seed/epoch、inference semantics 和限制 fail closed。

## Inferred

- 三个全开发集模型预期比单一 fold checkpoint 更适合作为最终研究候选载体；真实
  final-fit checkpoint 尚未生成，不能视为已验证性能。

## Unverified

- B11F 最终拟合的成功、重载一致性和资源预算。
- B09T 一次性 TEST 性能。

## Next Gate

`B11F_FINAL_DEVELOPMENT_FIT_PREPARATION / GPU_NOT_AUTHORIZED / TEST_DENIED`
