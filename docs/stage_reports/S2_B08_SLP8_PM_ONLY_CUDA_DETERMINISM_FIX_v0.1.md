# S2_B08 — CUDA deterministic loss 与失败 OOF Round 8 修复

**TASK-ID**: `TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1`
**状态**: `REVIEW_ACCEPTED / RUNNER_COMMITTED / R03_NOT_RUN / TEST_DENIED`

## 1. R02 现场根因

R02 在 RTX 4090、PyTorch `2.13.0+cu130`、strict deterministic 模式下进入
首个 TRAIN batch，在 `F.cross_entropy()` 的 CUDA NLLLoss2d forward 失败：

```text
RuntimeError: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation
```

失败发生在首个 loss forward，没有 optimizer step。第一异常后代码仍尝试
`np.stack()` 空 OOF，外层最终写出了唯一 FAILED terminal，但 error 字段被二次
`ValueError` 覆盖。

## 2. 修改

- 新增 `deterministic_cross_entropy_2d()`：保持冻结 CrossEntropyLoss 的 weighted
  mean 数学语义，使用 log-softmax、one-hot、elementwise multiply 和 reduction；
  不关闭 strict deterministic，不启用 warn-only。
- TRAIN weighted loss 与 VAL unweighted loss 统一改用该等价实现。
- 仅 `status == DONE` 才允许写真实 OOF；FAILED/STOPPED/INTERRUPTED 不 stack。
- DONE 必须精确覆盖全部 VAL predictions/targets，否则转 FAILED。

## 3. 验证

- float64 CPU weighted/unweighted value 与 gradient 对照 PyTorch CrossEntropyLoss：
  `rtol=atol=1e-12` 通过。
- 原始训练异常保留、无空 OOF；nominal DONE 空 OOF fail closed：通过。
- Round 8 定向：`4 passed`。
- B08 全套：`80 passed`；Markdown links：`6 passed`。
- 本机 RTX 4060 Laptop、Torch `2.12.1+cu126`、strict deterministic 下，冻结
  ResUNet、真实 `(1,192,84)` 输入形状、随机张量一次 AdamW step 通过；输出
  `(2,9,192,84)`、finite loss、peak 44.5 MB、deterministic=True。
- 同环境原生 `F.cross_entropy` 可复现与 AutoDL 相同的 deterministic RuntimeError，
  证明单纯降级至 Torch 2.12 不能解决。

## 4. 边界与下一 Gate

- primitive probe 不读取 SLP TRAIN/VAL/TEST，不是 Mini/Full 实验。
- Torch `2.13.0+cu130` + RTX 4090 尚未验证新 primitive；R03 正式运行前必须
  在 AutoDL 先执行随机张量 CUDA probe。
- R01/R02 必须保留；R03、Full、TEST 均未运行。
- runner 修复提交：`02fb364902736a64ee8708440f0dd0bdddf860bc`。
- 下一 Gate：推送 baseline、制作 R03 bundle；AutoDL primitive probe 通过后，
  Owner 再单独授权 R03。
