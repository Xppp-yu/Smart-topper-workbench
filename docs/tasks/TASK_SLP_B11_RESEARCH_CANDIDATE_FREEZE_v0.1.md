# TASK-SLP-B11-RESEARCH-CANDIDATE-FREEZE-v0.1

状态：`ACCEPT / CANDIDATE_CONTRACT_FROZEN / GPU_NOT_AUTHORIZED / TEST_DENIED`

## 目标

把 B09/B10 的开发期证据冻结为一个可审计的 SLP8 pressure-only 研究候选合同，
明确模型族、输入输出、最终开发集拟合计划、拒识语义、适用限制和 B09T 前置 Gate。

## 允许修改

- `configs/experiments/slp8_pm_research_candidate_v0.1.json`
- `scripts/validate_slp8_b11_candidate_freeze.py`
- `tests/test_slp8_b11_candidate_freeze.py`
- 本任务、阶段报告、项目状态与 SLP backlog。

## 冻结边界

- winner family：`slp8_deeplabv3plus_lite_v0.1`。
- 输入：单帧 SLP PMarray response，pressure-only，danaLab/uncover。
- 输出：192×84 的 9-class mask（background + 8 regions）。
- 开发证据：B09 91 subjects TRAIN+VAL OOF；TEST=0。
- 最终拟合：未来独立 GPU 任务在全部 91 development subjects 上训练 seeds
  42/123/2026，固定 epochs 15/20/12（各 seed 的 B09 五折 best-epoch 中位数）。
- 可选拒识：三模型 hard mask 3/3 一致；输出 `UNKNOWN_REGION`。它不是概率置信度，
  不得称为校准、OOD 或安全机制。

## 禁止事项

- 本任务不训练、不生成最终 checkpoint、不运行 GPU、不读取 TEST。
- 不把交叉验证 fold checkpoint 冒充最终拟合模型。
- 不宣称产品、硬件、舒适性、医疗、整夜或气囊控制有效。
- B09T 仍需最终拟合任务完成、独立审计和 Owner 单独授权。

## 验收

- validator 对模型、hash、seed/epoch、TEST、接口和限制 fail closed。
- 定向测试、Markdown links、`py_compile`、`git diff --check` 通过。
- 验收后允许精确提交、快进合并并 push。

## 下一 Gate

`B11F_FINAL_DEVELOPMENT_FIT_PREPARATION / GPU_NOT_AUTHORIZED / TEST_DENIED`
