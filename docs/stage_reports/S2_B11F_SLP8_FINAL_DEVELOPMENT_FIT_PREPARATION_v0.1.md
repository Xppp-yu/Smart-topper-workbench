# S2 B11F SLP8 最终开发集拟合准备 v0.1

状态：`ACCEPT / IMPLEMENTATION_PREPARATION_COMPLETE / GPU_NOT_AUTHORIZED / TEST_DENIED`

TASK-ID：`TASK-SLP-B11F-FINAL-DEVELOPMENT-FIT-PREPARATION-v0.1`

## 交付

- 新冻结配置 `slp8_pm_final_development_fit_v0.1.json`：唯一候选
  DeepLabV3+-lite，91 subjects / 4,095 TRAIN+VAL samples，seeds 42/123/2026，
  fixed epochs 15/20/12。
- 新 final-fit 核心与 CLI：真实入口要求 Owner `--run-authorized`、独立 B11F
  EXP-ID、clean 40-char Git SHA、全新输出目录和 CUDA。
- B01 仅以 `load_test=False` 加载，且断言 `_test_rows is None`；开发池只接受
  TRAIN+VAL，精确核验 4,095 samples / 91 subjects。
- 每 seed 使用固定 epoch；最后一个 epoch 的 training loss 仅作运行诊断，明确不
  是 validation/test performance。
- 每个最终 checkpoint 原子写入完整 identity 与 TEST=0 carrier；独立重载后对固定
  audit batch 的 hard predictions 做逐元素一致性检查。
- 成功只在三个模型全部完成后写 `DONE.json`；异常写 `FAILED.json`；已存在输出目录
  fail closed，禁止覆盖证据。

## 独立审查 ITERATE 与修复中项目

独立审查正确发现初版不能进入运行准备：optimizer 错用 Adam、B01 TRAIN/VAL 未与
freeze manifest 绑定、CUDA deterministic 原语不完整、candidate hash 重复读取，且
KeyboardInterrupt 未形成可恢复终态。修复将固定并验证 AdamW 与全部训练超参数；验证
TRAIN/VAL manifest hash、3,645/450、81/10 subjects、唯一 sample、来源元数据、onehot
和禁止 TEST-like 路径；复用 B09 `apply_settings()`；在 dispatch 单次冻结 candidate
hash；逐 epoch 原子保存 `last.pt`，并支持相同 identity 下从 `STOPPED.json` 显式恢复。
环境、peak CUDA memory、checkpoint SHA 与 DONE 前的三 checkpoint 重审计也纳入载体。

## 实际验证

```text
uv run python -m pytest tests/test_slp8_region_final_fit.py -q
31 passed（R05 独立只读复审）

uv run python -m pytest tests/test_slp8_region_final_fit.py \
  tests/test_slp8_b11_candidate_freeze.py tests/test_slp8_region_full.py -q
114 passed

R04 新增五项故障/恢复定向测试：5 passed
R05 新增三项恢复诊断/严格 JSON/最终 peak 测试：3 passed

uv run python -m pytest tests/test_check_markdown_links.py -q
6 passed

uv run python scripts/validate_slp8_b11f_final_fit_preparation.py \
  configs/experiments/slp8_pm_final_development_fit_v0.1.json
PASS (fail-closed protocol and execution-plan validation)

uv run python scripts/run_slp8_region_final_fit.py --validate-only
B11F_FINAL_FIT_PREPARATION_VALIDATION_PASSED TEST=0 GPU_NOT_AUTHORIZED

py_compile: PASS
git diff --check: PASS
GPU final fit: NOT RUN
TEST: 0
```

## Verified

- 初版独立审查结论为 `ITERATE`；GPU 与 TEST 均未运行。
- 修复后新增 optimizer/hyperparameter 漂移、TEST-like 路径、STOPPED terminal 与单次
  candidate hash carrier 覆盖；第二轮再补 explicit RUNNING resume 拒绝、STOPPED→RUNNING
  状态迁移、RNG checkpoint/restore、环境持久化与 CUDA 初始化顺序覆盖。第三轮补
  environment SHA 的 STOPPED/resume/DONE 绑定、篡改拒绝、完整中断恢复至 DONE，以及
  与不中断训练 checkpoint 参数逐元素一致的回归。R04 将环境 preflight 扩展到遗留
  RUNNING 恢复，在加载训练数据前拒绝环境篡改；拒绝覆盖缺失 completion carrier 的既有
  `final.pt`；根状态以 RUNNING 内容原子更新后 rename 到终态，消除 RUNNING/DONE/FAILED
  并存窗口；并增加真实 DataLoader shuffle RNG、CUDA 调用顺序及上述故障注入回归。
  R05 将最后 epoch loss、累计 wall time 与历史 CUDA peak 写入 `last.pt` 并在恢复时
  fail-closed 校验；显存上限改为逐 batch 且在 reload audit 后再次核验；completion 与
  DONE 写入前拒绝 NaN/Infinity 等非严格 JSON 数值。
- R05 独立只读复审结论为 `ACCEPT`，未发现 P0/P1/P2；有限 final-epoch resume 可生成
  严格 JSON DONE 并保留 loss/wall/peak，NaN resume 会 fail closed 到唯一 FAILED。

## Inferred

- 该实现具备进入运行准备和后续 AutoDL no-training preflight 的条件；尚不能由本地
  单元测试推断 RTX 4090 上三个正式 checkpoint 能完成。

## Unverified

- 真实 B01 全开发池的端到端 CUDA 训练、峰值显存、耗时和恢复行为。
- 三个 final checkpoint 的实际 hash、重载结果和集成推理。
- B09T 一次性 TEST 性能。

## Limitations

- final fit 没有开发集留出，因此不产生新的泛化分数；B09 OOF 仍是开发期性能依据。
- audit batch 只验证保存/重载一致性，不是性能评估。
- GT 仍为 danaLab/uncover、自动接受且 `NOT_REVIEWED` 的 pressure-only 参考标签。
- 不构成 cover、自研硬件、产品、舒适性、医疗、整夜或气囊控制验证。

## Next Gate

`B11F_FINAL_DEVELOPMENT_FIT_RUN_PREPARATION / GPU_NOT_AUTHORIZED / TEST_DENIED`
