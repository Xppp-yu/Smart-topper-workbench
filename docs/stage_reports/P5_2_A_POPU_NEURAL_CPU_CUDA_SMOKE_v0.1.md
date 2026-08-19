# P5.2-A/R5.2-A — PoPu 神经网络 CPU/CUDA Smoke 报告 v0.1

## 1. 结论

**状态：COMPLETE — CPU_CUDA_SMOKE_PASS。**

PoPu 原始压力矩阵的 MatrixMLP、TinyCNN、SmallResNet 已在受治理的 CPU 与 NVIDIA RTX 4090 CUDA Smoke 中完成最小真实数据训练。受试者隔离、仅训练折 normalization、左右翻转标签交换、checkpoint/resume、续训参数变化、独立重载预测一致和固定 seed 最小复现均通过。

本阶段只证明训练底座和 GPU 执行链路可用。1 epoch / 2 受试者 / 1,000 样本的验证准确率不得用于模型排名、总体候选冻结或产品效果声明。本阶段没有运行 P5.2-B Mini 或 P5.2-C Full。

## 2. 验收实验

| 项 | CPU Smoke | CUDA Smoke（正式验收） |
|---|---|---|
| EXP-ID | `EXP-P5.2-A2-CPU-SMOKE-20260819-R02` | `EXP-P5.2-A2-CUDA-SMOKE-20260819-R02` |
| Git SHA | `a65438b` | `5803f5cdbd0bf1b08e7435d3999c3c8eefcb8e08` |
| Git dirty | `false` | `false` |
| scope | `smoke` | `smoke` |
| 数据 | PoPu Tactilus，subject 1 训练、subject 2 验证 | 同 CPU Smoke |
| 样本 | 1,000 selected；train 1,004（含 502 增强）；val 498 | 同 CPU Smoke |
| 训练 | 1 epoch；MatrixMLP / TinyCNN / SmallResNet | 同 CPU Smoke |
| 结果 | CPU 通路通过 | `SUCCEEDED`，约 7.0 秒，CUDA 通路通过 |

CUDA R02 运行环境：

- GPU：`NVIDIA GeForce RTX 4090`，24,564 MB；
- NVIDIA driver：`595.58.03`；
- Python：`3.12.3`；
- PyTorch：`2.8.0+cu128`；PyTorch CUDA runtime：`12.8`；cuDNN：`91002`；
- `cuda.available=true`，`device_count=1`；
- 自动测试：远端 `271 passed`，本地同提交 `271 passed`。

## 3. CUDA R02 产物与完整性

本地只读实验目录：`outputs/experiments/EXP-P5.2-A2-CUDA-SMOKE-20260819-R02/`（Git ignored），共 17 个文件、12,557,839 bytes，包括：

- [提交配置](../../outputs/experiments/EXP-P5.2-A2-CUDA-SMOKE-20260819-R02/submitted_config.json)、[resolved config](../../outputs/experiments/EXP-P5.2-A2-CUDA-SMOKE-20260819-R02/resolved_config.json)；
- [manifest](../../outputs/experiments/EXP-P5.2-A2-CUDA-SMOKE-20260819-R02/manifest.json)、[status](../../outputs/experiments/EXP-P5.2-A2-CUDA-SMOKE-20260819-R02/status.json)、[DONE](../../outputs/experiments/EXP-P5.2-A2-CUDA-SMOKE-20260819-R02/DONE.json)；
- [metrics](../../outputs/experiments/EXP-P5.2-A2-CUDA-SMOKE-20260819-R02/metrics.json)、训练日志、三模型预测和六个 latest/best checkpoints。

远端证据包与下载后的本地文件 SHA256 一致：

```text
48d7f05d54f2b5358bbdc411be07dbbdc356795494065262df1e63418652dda8
EXP-P5.2-A2-CUDA-SMOKE-20260819-R02.tar.gz
```

本地独立读取断言通过：`state=SUCCEEDED`、`scope=smoke`、Git SHA/clean 状态匹配、`device=cuda`、CUDA available、selected samples 1,000、固定 seed 复现、GPU/driver/PyTorch CUDA/cuDNN 元数据完整。

## 4. Smoke 观测值（禁止排名）

| 模型 | 参数量 | val accuracy |
|---|---:|---:|
| MatrixMLP | 476,165 | 0.8012 |
| TinyCNN | 6,085 | 0.4900 |
| SmallResNet | 10,165 | 0.6566 |

这些数值来自单次、单 seed、1 epoch、两个受试者的通路测试，不满足模型比较要求。不得据此选择 MatrixMLP 或淘汰 CNN。

## 5. 历史与治理说明

- CPU R01 因全局截断造成类别偏斜，被 Reviewer 否决并仅保留为历史 Smoke；CPU R02 为接受版本。
- CUDA R01 的训练与 CUDA 通路成功，但 `manifest.json` 中 `cuda=null`，环境证据不满足治理要求；不改写 R01。修复环境采集并新增回归测试后，以新 Git SHA 和新 EXP-ID 执行 CUDA R02。
- CUDA R02 的配置、实验目录和证据包均独立保存，没有覆盖 R01。
- 下一阶段只能是另行签发、冻结配置并授权的 P5.2-B Mini。Coding Agent 在此停止，不自动运行 Mini/Full。

## 6. 不能得出的结论

- 尚未得出 CNN、MLP 或 P5.1 SVM 谁是 PoPu 总体最优候选；
- 尚未完成 repeated subject-grouped 公平比较、概率校准、最差受试者和逐类别稳定性评价；
- 尚未验证 SLP、PressurePose、自采传感器、具体人体部位、整夜睡眠或产品闭环；
- 公开 PoPu Smoke 不构成自研硬件、舒适性或临床效果证据。
