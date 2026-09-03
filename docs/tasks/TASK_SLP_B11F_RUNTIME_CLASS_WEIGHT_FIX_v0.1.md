# TASK-SLP-B11F-RUNTIME-CLASS-WEIGHT-FIX-v0.1

状态：`FIXES_COMPLETE / READY_FOR_INDEPENDENT_REVIEW / FAILED_EXP_IMMUTABLE / GPU_NOT_AUTHORIZED / TEST_DENIED`

## 目标

修复 B11F 首次真实 CUDA 启动暴露的 class-weight NumPy→Torch 接口错误，并补足能在
CPU 回归中复现真实返回类型的反例。不得恢复或覆盖失败 EXP-ID
`EXP-SLP-B11F-PM-FINAL-FIT-20260903-AUTODL-R01`。

## 失败证据

- runner：`9af268fa168207a269abbef22e522ac04fd6b6c5`。
- preflight environment fingerprint：
  `a5a9342b18d00b614355e63ce056a7edd92dd80358d8aead5ef6e8e0ba045669`。
- 异常：`AttributeError: 'numpy.ndarray' object has no attribute 'to'`。
- 位置：`slp8_region_final_fit.py` 的 class-weight device conversion。
- 训练循环未开始；TEST=0。AutoDL 原始 `FAILED.json`、environment/budget carriers、
  root marker 清单、文件 SHA 清单与 operator traceback 已同步到
  [`docs/evidence/b11f_r01_failure`](../evidence/b11f_r01_failure/README.md)，等待独立复审。

## 允许修改

- `src/topper_perception/neural/slp8_region_final_fit.py`
- `tests/test_slp8_region_final_fit.py`
- 本任务、对应阶段报告、B11F 运行准备记录、backlog 与 `PROJECT_STATUS.md`

禁止修改冻结 config、candidate、B01 freeze、既有失败输出或任何 TEST 数据。

## 验收

- 与 B09 冻结路径一致，显式将 class-weight NumPy array 转为 float32 Torch tensor 后再
  送入目标 device。
- 回归 mock 使用真实 `np.ndarray` 类型；旧实现必须被该回归捕获。
- B11F 定向与联合回归、validator、validate-only、`py_compile`、Markdown links 和
  `git diff --check` 全部通过。
- 形成新 clean/pushed runner SHA、新 Git bundle、新 EXP-ID 和重新独立 preflight；旧
  R01 绝不 resume/reuse。

## 下一 Gate

`B11F_RUNTIME_FIX_EVIDENCE_READY_FOR_REVIEW / GPU_NOT_AUTHORIZED / TEST_DENIED`
