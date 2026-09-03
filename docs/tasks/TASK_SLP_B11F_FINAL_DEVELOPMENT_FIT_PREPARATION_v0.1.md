# TASK-SLP-B11F-FINAL-DEVELOPMENT-FIT-PREPARATION-v0.1

状态：`ACCEPT / IMPLEMENTATION_PREPARATION_COMPLETE / GPU_NOT_AUTHORIZED / TEST_DENIED`

## 目标

实现并验证 B11 冻结研究候选的最终开发集拟合入口。入口只读取 B01 的
TRAIN+VAL 开发池（91 subjects / 4,095 samples），为 DeepLabV3+-lite 按冻结的
seed/epoch 训练三个最终 checkpoint，并在任何 TEST 访问发生前 fail closed。

## 允许修改

- `configs/experiments/slp8_pm_final_development_fit_v0.1.json`
- `src/topper_perception/neural/slp8_region_final_fit.py`
- `scripts/run_slp8_region_final_fit.py`
- `scripts/validate_slp8_b11f_final_fit_preparation.py`
- `tests/test_slp8_region_final_fit.py`
- 本任务、B11F 阶段报告、项目状态与 SLP backlog

## 冻结输入

- candidate contract：`slp8_pm_research_candidate_v0.1`
- model：`slp8_deeplabv3plus_lite_v0.1`
- development pool：B01 TRAIN+VAL，仅 danaLab/uncover
- seeds/epochs：42→15、123→20、2026→12
- optimizer/lr/weight decay/batch size：继承 B09 冻结训练原语
- TEST：`load_test=False` 且 `_test_rows is None`

## 必须实现

1. validate-only 为零写入，且不加载训练数组或 TEST。
2. 真实运行必须同时提供独立 EXP-ID 与 `--run-authorized`。
3. dispatch 前冻结 clean 40-char Git SHA、配置 hash、candidate hash、B01 freeze hash。
4. 全开发池 normalization 与 class weights 只由 TRAIN+VAL 计算。
5. 每 seed 固定 epoch，不使用开发集早停，也不产生伪造的 final-fit 验证分数。
6. checkpoint 原子写入并携带完整身份、seed、fixed epochs、TEST=0 字段。
7. checkpoint 独立重载，固定 audit batch 的 hard prediction 必须逐元素一致。
8. 已完成 EXP-ID/seed 不覆盖；失败写 `FAILED.json`，成功写 `DONE.json`。
9. 三个 checkpoint 全部完成并审计通过前，不得放行 B09T。

## 本任务禁止事项

- 不运行 GPU final fit，不生成正式 checkpoint。
- 不调用 `enable_test_access`，不以任何方式读取 TEST rows/labels/onehot。
- 不运行 B09T，不报告 TEST 分数。
- 不把训练集 audit prediction 当作性能评价。
- 不宣称产品、硬件、舒适性、医疗、整夜或气囊控制效果。
- 未经后续任务授权，不 commit 或 push。

## 验收

- 定向测试覆盖配置漂移、TEST 注入、脏 Git、授权、输出碰撞、身份、重载与终态。
- synthetic CPU smoke 只能证明执行链路，不能生成研究指标或替代 GPU final fit。
- `py_compile`、Markdown links、`git diff --check` 通过。
- 阶段报告必须以 Verified / Inferred / Unverified / Limitations / Next Gate 收尾。

## 下一 Gate

当前：`B11F_FINAL_DEVELOPMENT_FIT_RUN_PREPARATION / GPU_NOT_AUTHORIZED / TEST_DENIED`。

R05 独立只读复审已 `ACCEPT`。下一步必须先形成 clean release SHA，并在独立运行准备
记录中绑定全新 EXP-ID、预算和执行环境；这不构成 GPU 授权。

2026-09-03 Owner 后续指令“下一步推进”授权本任务验收收口与本地提交；未授权 push、
GPU final fit 或 TEST。
