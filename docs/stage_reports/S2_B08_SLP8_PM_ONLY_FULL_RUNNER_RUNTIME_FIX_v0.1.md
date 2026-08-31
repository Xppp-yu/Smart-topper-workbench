# S2_B08 — one-fold real runtime Round 7 修复

**TASK-ID**: `TASK-SLP-B08-FULL-RUNNER-AND-ONE-FOLD-PREFLIGHT-v0.1`
**状态**: `REVIEW_ACCEPTED / RUNNER_COMMITTED / R02_NOT_RUN / TEST_DENIED`

## 1. 现场证据与根因

R01 使用 clean `5af426039ae41209af7929bc9319a0657e5f92b4`、冻结
candidate/fold/seed/30 epochs 和 RTX 4090 启动。环境、输入 hashes、CUDA 与
validate-only 均通过；runner 在第一个 epoch 前抛出：

```text
AttributeError: 'RegionSample' object has no attribute 'get'
```

真实 loader 的合同是 `Sequence[RegionSample]`，但 real-path synthetic guard
按 synthetic dictionary 调用了 `.get()`。异常还发生在 one-fold 外层结果处理
之前，导致 R01 没有根级 `FAILED.json`。

## 2. 修改

- `src/topper_perception/neural/slp8_region_full.py`
  - real TRAIN/VAL 类型改为 `Mapping | RegionSample` 的显式联合合同；
  - 新增 fail-closed `_validate_real_region_records()`；真实路径使用
    `record.sample_id`，拒绝 mapping 与 `SYNTH_` ID。
- `scripts/run_slp8_region_full.py`
  - 在调用训练前冻结完整 identity；
  - 捕获普通 `train_one_unit()` 异常，写一致的 `preflight_manifest.json` 与唯一
    根级 `FAILED.json`，保留 `test_access=false`；不捕获 `KeyboardInterrupt`。
- `tests/test_slp8_region_full.py`
  - 新增生产 `RegionSample` 合同与 synthetic contamination 回归；
  - 新增训练异常写根级唯一 FAILED terminal 回归。

## 3. 实际验证

| 检查 | 结果 |
|---|---|
| Round 7 + one-fold 定向 | `3 passed, 73 deselected` |
| B08 完整测试 | `76 passed` |
| B04/B04A core/models/links | `308 passed, 1 failed`；唯一失败是旧测试硬编码本机 CUDA 不可用，但当前复用环境实际 `cuda_run=True`，与本次差异无关 |
| `py_compile`（实现、CLI、测试） | PASS |
| `git diff --check` | PASS（仅 Windows LF→CRLF 提示） |

## 4. 产物与 Git

- R01 AutoDL output：由 Owner 侧保留；本地未访问、未修改、未删除。
- 本轮没有生成模型 checkpoint 或指标。
- runner 修复提交：`e0ba25a9aa0b33be971327ca398d822f7c7d1c8a`。
- 本治理记录随后的 docs commit 不改变冻结 runner SHA；R02 bundle 尚待制作。

## 5. Verified / Inferred / Unverified

- **Verified**：真实类型错误由现场 traceback 精确定位；两项修复和 76 个 B08
  测试通过；普通训练异常现在写唯一 FAILED terminal。
- **Inferred**：修复后的 real path 将越过原第 1303 行 guard；尚未在 RTX 4090
  上重新执行，因此不能声称真实训练已开始或完成。
- **Unverified**：R02 wall time、peak CUDA、checkpoint reload、真实 OOF 与指标。

## 6. 限制、禁止结论与下一 Gate

- R01 是失败证据，不可覆盖，也不能伪装为规范 terminal 完整的成功试验。
- R02、30-unit Full、TEST 均未运行；不得作候选排名或模型效果结论。
- 下一 Gate：推送包含 runner 与治理记录的 baseline、制作 R02 bundle，再由 Owner
  单独授权 R02。
