# S2 B11F Production-Wiring Smoke v0.1

## 1. 结论

TASK-ID：`TASK-SLP-B11F-PRODUCTION-WIRING-SMOKE-v0.1`

本地 CPU 真实数据 smoke R07 已 `PASS`。它验证了 R01 class-weight runtime 修复之后，
B11F 的真实 B01 development metadata、生产 statistics helper、NumPy→Torch 权重转换、
真实 Dataset/DataLoader、DeepLabV3+-lite、加权 deterministic loss、backward 和一次 AdamW
step 可以串联执行。

这不是正式 final fit，不授权 AutoDL/GPU，不创建正式 EXP-ID 或 checkpoint，也没有读取 TEST。
R07 技术复审结论为 `ACCEPT`；因实现与复审均由同一 Codex 完成，人员独立性限制保留。

## 2. 基线与变更

- 基线：`86415948afed4aa30e9cced2bfabbacba03aed5e`
- 分支：`codex/task-slp-b11f-production-wiring-smoke-v0.1`
- 新增：
  - `scripts/smoke_slp8_b11f_production_wiring.py`
  - `tests/test_slp8_b11f_production_wiring.py`
  - `docs/tasks/TASK_SLP_B11F_PRODUCTION_WIRING_SMOKE_v0.1.md`
  - 本报告
- 更新：`docs/PROJECT_STATUS.md`、`docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`
- production runner、optimizer、loss、seed、epochs、模型与冻结 config/candidate/B01 均未修改。

## 3. 有效 R07 证据

- ignored summary：`outputs/analysis/b11f_production_wiring_smoke_20260904_r07.json`
- committed sanitized copy：`docs/evidence/b11f_production_wiring_smoke/R07_summary.json`
- summary SHA-256：`65dd6723d5a59aa36bac666d3b73f0b5d69408e2140581b74cf16af8e699b95a`
- scope：`LOCAL_CPU_REAL_DATA_ONE_BATCH_ONLY`
- 完整 development metadata：4,095 samples / 91 subjects，仅 TRAIN+VAL。
- statistics smoke 子集：确定性的前 128 个真实 development samples；不是正式统计估计。
- class weights：真实 `numpy.ndarray`、`float64`、shape `[9]`、class ID 顺序 0..8；
  经 `torch.from_numpy(...).to("cpu").to(torch.float32)` 成为 CPU float32 tensor。
- input/label/logits：`[1,1,192,84]` float32 / `[1,192,84]` int64 /
  `[1,9,192,84]`。
- loss：有限；值 `25.50458335876465` 仅为 smoke diagnostic，不是 validation/research metric。
- gradients：28 个有限梯度 tensor。
- optimizer：冻结 AdamW、lr=0.001、weight decay=0.0001；恰好 1 step；28 个参数 tensor 改变。
- frozen production batch size：16；本地 smoke microbatch size：1。
- `formal_experiment_id=null`、`checkpoint_created=false`、`resume_used=false`。
- `test_access=false`、TEST rows/labels/onehot 均为 0。
- `gpu_training_run=false`、device=CPU、`CUDA_VISIBLE_DEVICES=-1`、AutoDL 未连接。

`determinism.run_mode="cpu_synthetic_reproducible"` 是既有 helper 对所有 CPU 进程的历史命名；
本 summary 的 `scope`、真实 B01 路径及 cardinality 明确表明本轮不是 synthetic 数据。

## 4. 开发期 fail-closed 轨迹

- R01：Windows 上空字符串屏蔽 CUDA 无效，在数据加载前拒绝；无 output。
- R02：完整 4,095 样本范围的进程异常退出；当时尚无细分阶段标记，无法独立定位；无 output。
- R03：同一完整 statistics 范围再次异常退出；无 output，未到训练 step。
- R04：microbatch label 断言仍误用 production batch size=16，forward 前拒绝；无 output。
- R05：真实 forward/backward/step 完成后，summary 阶段误用 rows-hash helper 处理文件路径而拒绝；
  无 output。
- R06：修正 smoke 自身断言和文件 hash 调用后完整 PASS；首轮复审发现 output publish
  仍存在 TOCTOU 覆盖竞争，结论 `ITERATE`。
- R07：改用 atomic hard-link no-clobber publish；竞争者插入目标文件时拒绝且保留 sentinel，
  同路径重跑退出 1 且 summary hash 不变；真实 CPU smoke 完整 PASS。

这些编号均为本地 `SMOKE-*`，不属于正式 `EXP-SLP-B11F-*`，也不恢复/覆盖失败的 AutoDL R01。

## 5. 实际验证

```text
uv run python -m pytest tests/test_slp8_b11f_production_wiring.py -q
7 passed

uv run python -m pytest tests/test_slp8_b11f_production_wiring.py tests/test_slp8_region_final_fit.py tests/test_slp8_b11_candidate_freeze.py tests/test_slp8_region_full.py -q
130 passed

uv run python scripts/smoke_slp8_b11f_production_wiring.py ...r07.json
PASS / TEST=0 / GPU_NOT_RUN / AUTODL_NOT_CONNECTED

uv run python scripts/validate_slp8_b11f_final_fit_preparation.py configs/experiments/slp8_pm_final_development_fit_v0.1.json
PASS / TEST=0 / GPU_NOT_AUTHORIZED

uv run python scripts/run_slp8_region_final_fit.py --validate-only
PASS / TEST=0 / GPU_NOT_AUTHORIZED

uv run python -m pytest tests/test_check_markdown_links.py -q
PASS

uv run python -m py_compile scripts/smoke_slp8_b11f_production_wiring.py
PASS

git diff --check
PASS
```

## 6. Reviewer checklist

1. CPU isolation是否在 import torch 前设置为 `CUDA_VISIBLE_DEVICES=-1`，且所有 tensor/model 明确在 CPU。
2. 是否加载完整 B01 development metadata 并保持 `_test_rows is None`，且 summary TEST carriers 全为 0。
3. statistics 是否明确仅为 128 个真实样本的 wiring smoke，不冒充正式 4,095 样本统计。
4. NumPy→Torch 转换是否与修复后的生产路径逐字一致，九类顺序/dtype/device 是否严格。
5. 是否只有一个真实 microbatch、一次 optimizer step，且无 epoch loop/checkpoint/resume/formal EXP-ID。
6. optimizer/loss/model/seed/正式 batch/epochs 等冻结合同是否未被修改。
7. output 是否原子写入、拒绝覆盖、无绝对数据路径、无 NaN/Inf。
8. R01--R06 的开发期轨迹是否准确披露，且提交内有效证据是否只指向 R07。

## 7. R07 技术复审

- Verdict：`ACCEPT`
- P0：0
- P1：0；R06 的 TOCTOU 覆盖竞争已由 atomic hard-link no-clobber、竞争插入反例和
  同路径重跑拒绝关闭。
- P2：0；R07 脱敏 summary 已 byte-identical 纳入 Git 证据并由 `SHA256SUMS` 绑定。
- 复审重新执行联合回归 130 passed、validator、validate-only、Markdown links、py_compile、
  evidence hash 和 diff check，全部通过。
- 限制：实现者与复审者是同一 Codex，本结论是技术复审 ACCEPT，不声称人员独立性。

## 8. Verified / inferred / unverified / limitations / next Gate

Verified：本地 CPU 上，128 样本真实 statistics 子集与一个真实 microbatch 的完整生产接线通过；
class-weight ndarray 转换、loss/backward/一次 AdamW step、summary carrier 均通过；TEST=0；GPU NOT RUN。

Inferred：该 smoke 能直接覆盖 AutoDL R01 暴露的 NumPy→Torch 接线缺陷，但不能推出正式三 seed
训练必然完成。

Unverified：完整 4,095 样本 statistics 的数值/耗时/资源；正式 batch=16；真实 CUDA；15/20/12
epochs；45 分钟预算与跨进程恢复；三个最终 checkpoint。

Limitations：本地 CPU microbatch=1；statistics 仅用 128 样本；既有 determinism helper 的 CPU
run-mode 名称含 `synthetic`，但实际输入为真实 B01 数据。R07 脱敏 summary 已纳入提交证据。

Next Gate：
`B11F_AUTODL_NO_TRAINING_PREFLIGHT_PREPARATION / AUTODL_NOT_AUTHORIZED / GPU_NOT_AUTHORIZED / TEST_DENIED`。
Owner 已于 2026-09-04 授权 R07 closure commit/push；该授权不包含 AutoDL 连接或执行。
形成并推送新 SHA 后，只能另行准备 no-training preflight 授权材料。仍不得直接运行 GPU，
TEST 仍需未来另一份一次性 Owner 授权。
