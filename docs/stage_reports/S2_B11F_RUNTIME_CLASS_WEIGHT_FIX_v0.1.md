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
- 训练 batch：`0`（traceback 位于 weight tensor conversion，早于 DataLoader/training loop）。
- GPU final fit：`FAILED_BEFORE_TRAINING_LOOP`。
- TEST：`0`。

## AutoDL 失败终态审计

原始 machine-readable 载体、root inventory、SHA-256 manifest 与 operator traceback 已
版本化至 [`docs/evidence/b11f_r01_failure`](../evidence/b11f_r01_failure/README.md)。归档在
AutoDL 与本地的 SHA-256 均为
`ab5d97a2d9a6cf703e36efc1dc9815f2fdb4ff80fc819654c6fe318277731c98`；包内七个文件经
本地独立重算全部匹配 `SHA256SUMS`。

- 根状态唯一：`FAILED.json=EXISTS`；`RUNNING/STOPPED/DONE=ABSENT`。
- `test_access=false`；`test_rows/test_labels/test_onehot=0`。
- `environment_hash_match=true`；持久化与 observed environment SHA-256 均为
  `feb853112e6acffd736f351b2ac8d13daeb8ec99698765b6dccf0ab1c2635021`。
- `budget.state=FAILED`；固定 2,700 秒预算在约 `2.035854` 秒时封存，remaining 约
  `2697.964146` 秒；budget core SHA-256 为
  `66ef679b0b246e68aeb8593fe297b382f2e8872cf1a1a6212fee33cbff2a7b25`。
- terminal error 与 traceback 一致：
  `AttributeError: 'numpy.ndarray' object has no attribute 'to'`。

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

- clean runner release、新 bundle 与 R02 preflight。
- 真实 GPU final fit、checkpoint 与 TEST 性能。

## 下一 Gate

独立复审已于 2026-09-03 `ACCEPT`：P0/P1/P2 均为 0；R01 唯一 FAILED、首 batch
前 traceback、environment/budget/hash/identity/预算数学关系、TEST=0 carriers 与仅限
NumPy→Torch 的 runtime diff 均复核通过。证据清单补齐后 9/9 文件 SHA-256 匹配。

`B11F_PRODUCTION_WIRING_SMOKE_PREPARATION / GPU_NOT_AUTHORIZED / TEST_DENIED`
