# TASK-SLP-B11F-RUN-PREPARATION-P1-FIX-v0.1

状态：`FIXES_COMPLETE / READY_FOR_INDEPENDENT_REVIEW_R02 / GPU_NOT_AUTHORIZED / TEST_DENIED`

## 目标

修复 B11F 运行准备独立审查发现的两项 P1：

1. 首次启动必须在任何输出或训练前，把实际执行环境绑定到 Owner 明确授权的环境
   fingerprint；缺失、格式错误或不匹配均 fail closed。
2. 45 分钟限制必须成为 runner 内部持久化的 EXP 级累计 wall budget；恢复不得重置
   deadline 或重新获得预算，篡改、时钟回退和预算耗尽均 fail closed。

## 允许修改

- `configs/experiments/slp8_pm_final_development_fit_v0.1.json`
- `src/topper_perception/neural/slp8_region_final_fit.py`
- `scripts/run_slp8_region_final_fit.py`
- `scripts/validate_slp8_b11f_final_fit_preparation.py`
- `tests/test_slp8_region_final_fit.py`
- B11F 本修复任务、修复阶段报告、运行准备任务/报告、`PROJECT_STATUS.md` 与 SLP backlog

允许在测试和独立复审准备完成后形成并推送受限提交。禁止修改 raw dataset、B01
freeze、B11 candidate、既有实验证据或任何其他任务文件。

## 必须实现

- 提供 no-training environment fingerprint 输出模式；只收集环境，不加载数据或训练。
- 正式与恢复入口必须显式接收 64-char authorized environment fingerprint。
- 首次启动在创建 output directory 前比较实际 fingerprint；不匹配时零输出。
- authorization fingerprint 必须进入 run/seed/checkpoint/terminal identity；恢复同时校验
  原 environment 文件及 Owner authorization fingerprint。
- config 固定 `max_total_wall_seconds=2700`，validator 对漂移 fail closed。
- `budget.json` 原子持久化首次启动时间、固定 deadline、累计 elapsed/remaining 和状态；
  RUNNING、STOPPED、FAILED、DONE 必须携带可审计的 budget path/hash/摘要。
- 每 batch、每 epoch、恢复前和 DONE 前检查 EXP budget；恢复沿用原 deadline，绝不重置。
- 增加缺授权 fingerprint、首次环境不匹配、恢复 authorization 漂移、budget 篡改、恢复
  deadline 重置、预算耗尽及零输出顺序的反例测试。

## 禁止事项

- 不运行 AutoDL/CUDA preflight 或 GPU final fit。
- 不读取 TEST rows/labels/onehot/statistics/predictions，不调用 `enable_test_access()`。
- 不创建 proposed EXP-ID 正式输出，不标记 `OWNER_AUTHORIZED` 或 `QUEUED`。
- 不把 training loss、synthetic CPU 结果或环境采集当成性能证据。

## 下一 Gate

修复 release：`9af268fa168207a269abbef22e522ac04fd6b6c5`；下一 Gate：
`B11F_RUN_PREPARATION_INDEPENDENT_REVIEW_R02 / GPU_NOT_AUTHORIZED / TEST_DENIED`
