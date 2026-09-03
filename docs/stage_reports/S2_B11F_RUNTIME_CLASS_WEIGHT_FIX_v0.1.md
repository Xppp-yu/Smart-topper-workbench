# S2 B11F runtime class-weight 修复 v0.1

状态：`FIXES_COMPLETE / READY_FOR_INDEPENDENT_REVIEW / FAILED_EXP_IMMUTABLE / GPU_NOT_AUTHORIZED / TEST_DENIED`

TASK-ID：`TASK-SLP-B11F-RUNTIME-CLASS-WEIGHT-FIX-v0.1`

## 观察到的失败

首次已授权 GPU final fit 在创建受治理输出后、进入训练 batch 前失败：

```text
AttributeError: 'numpy.ndarray' object has no attribute 'to'
```

`class_weights_to_tensor()` 的公开合同返回 NumPy array；B09/B04A 路径显式
`torch.from_numpy(...).to(device).to(torch.float32)`，B11F 遗漏该转换。测试曾把 helper
mock 为 Torch tensor，因此未覆盖真实接口。

## 边界

- 失败 EXP-ID：`EXP-SLP-B11F-PM-FINAL-FIT-20260903-AUTODL-R01`，必须永久保留且禁止恢复。
- 训练 batch：`0`（根据 traceback 位置；待根 terminal 载体复核）。
- GPU final fit：`FAILED_BEFORE_TRAINING_LOOP`。
- TEST：`0`。

## 修复

- 与 B09/B04A 一致，使用 `torch.from_numpy(...)` 后转入目标 device 和
  `torch.float32`。
- B11F 运行路径测试的 class-weight mock 改为真实 `np.ndarray`；旧 `.to(device)` 实现
  会被这些运行路径直接捕获。
- B11F 定向：`40 passed`。
- B11F+B11+B08/B09 联合：`123 passed`。
- Markdown links：`6 passed`。
- validator、validate-only、`py_compile`、`git diff --check`：PASS；TEST=0，GPU NOT RUN。

## 尚未验证

- AutoDL `FAILED.json`、environment/budget hash 与唯一根 terminal。
- clean runner release、新 bundle 与 R02 preflight。
- 真实 GPU final fit、checkpoint 与 TEST 性能。

## 下一 Gate

`B11F_RUNTIME_FIX_REVIEW / GPU_NOT_AUTHORIZED / TEST_DENIED`
