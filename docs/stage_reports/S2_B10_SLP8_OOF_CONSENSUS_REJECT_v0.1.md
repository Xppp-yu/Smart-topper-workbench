# S2 B10 SLP8 OOF seed-consensus 拒识与错误分析 v0.1

状态：`ACCEPT_WITH_LIMITATIONS / HARD-CONSENSUS-ONLY / TEST_DENIED`

TASK-ID：`TASK-SLP-B10-OOF-CONSENSUS-REJECT-v0.1`

## 结果

对 B09 winner `slp8_deeplabv3plus_lite_v0.1` 的 seeds 42/123/2026、4,095 个
TRAIN+VAL OOF hard masks 做逐像素多数票与 3/3 一致性分析：

| 指标 | 结果 |
|---|---:|
| 全部像素多数票错误率 | 13.8641% |
| 3/3 一致像素覆盖率 | 87.3766% |
| 3/3 一致像素错误率 | 8.7916% |
| 前景像素 3/3 一致覆盖率 | 73.7753% |
| 拒绝分歧像素捕获的原始错误比例 | 44.5922% |
| 仍留在 3/3 一致区的原始错误比例 | 55.4078% |

最低一致覆盖受试者为 `00017`：全部像素覆盖 79.3958%，前景覆盖 61.5894%，
多数票错误率 20.4074%。多数票错误率最高的三个受试者为 `00017`、`00007`、
`00063`。最高的 3/3 一致错判样本为 `SLP:danaLab:00043:uncover:000034`，
错判 4,500 / 16,128 pixels（27.9018%）。

## 解释

seed 分歧具有一定拒识价值：拒绝 12.6234% 的像素可捕获 44.5922% 的多数票错误。
但它不能作为安全机制，因为 55.4078% 的错误仍是三 seed 一致错判。尤其前景覆盖
仅 73.7753%，说明简单的 3/3 门槛会拒绝较多区域像素。

## 证据

ignored outputs：`outputs/analysis/slp_b10_consensus_r01/`

- `summary.json`
- `per_subject.csv`（91 subjects）
- `high_consensus_errors.csv`（top 50 samples）

三个输入 OOF SHA-256 已写入 `summary.json`。分析过程不读取 raw PM、checkpoint 或
TEST，不运行训练/GPU。

## 验证

- B10 手算与 target-drift fail-closed：2 passed
- Markdown links：6 passed
- `py_compile`：PASS
- TEST：0

## 边界

### Verified

- 三 seed sample 顺序、targets、candidate 和 shape 一致。
- 上述 coverage/error 指标及逐受试者、高一致性错误清单。

### Inferred

- seed 分歧可作为有限的 epistemic proxy；它不是经过校准的概率不确定性。

### Unverified

- softmax/logit confidence、概率校准、alignment confidence、真实 OOD、cover、TEST。

### Limitations

- B09 OOF 不含 logits/probabilities，无法诚实完成概率阈值选择。
- 3/3 一致不代表正确，超过一半的原始错误仍为三 seed 一致错误。

### Next Gate

`B11_RESEARCH_CANDIDATE_FREEZE_READY_TO_DRAFT / TEST_DENIED`
