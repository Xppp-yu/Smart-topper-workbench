# TASK-SLP-B10-OOF-CONSENSUS-REJECT-v0.1

状态：`COMPLETE / ACCEPT_WITH_LIMITATIONS / TRAIN_VAL_OOF_ONLY / TEST_DENIED`

## 目标

基于 B09 已接受的 winner DeepLabV3+-lite 三 seed hard-mask OOF，执行无需重新训练的
一致性拒识与错误分析：逐像素多数票、3/3 一致覆盖率、被拒像素错误捕获率、逐受试者
coverage-risk 与高一致性错判。

## 输入与边界

- 输入仅为 B09 R01 原始归档内三个 winner OOF NPZ。
- 三个 NPZ 必须逐字节语义一致地携带相同 sample 顺序、targets 与 4,095 samples。
- subject 从受治理 sample-id `SLP:danaLab:<subject>:uncover:<frame>` 解析。
- 不读取 checkpoint、raw PM、TEST 或非 winner OOF。
- OOF 未保存 logits/probabilities，因此本任务不得声称概率校准、softmax confidence 或
  alignment/OOD 检测；输出名为 `seed_consensus`，不是 model confidence。

## 允许修改

- `scripts/analyze_slp8_b10_consensus.py`
- `tests/test_slp8_b10_consensus.py`
- 本任务与阶段报告、项目状态、SLP backlog。

## 输出

- ignored output 下的 JSON 总结、逐受试者 CSV、高一致性错误 CSV。
- 原始证据只读，输出目录不得与 B09 EXP-ID 重叠。

## 验收

- 对 seed/sample/target/candidate/shape 漂移 fail closed。
- 多数票和 3/3 一致性指标有手算回归测试。
- 真实 B09 OOF 分析完成并记录 hash、指标、限制。
- TEST=0；GPU/训练均 `NOT RUN`。
- 验收通过后允许形成一个仅含本任务精确文件的本地 commit；合并与 push 需在
  Reviewer 接受后执行。

## 下一 Gate

`B10_REVIEW -> B11_CANDIDATE_FREEZE / TEST_DENIED`
