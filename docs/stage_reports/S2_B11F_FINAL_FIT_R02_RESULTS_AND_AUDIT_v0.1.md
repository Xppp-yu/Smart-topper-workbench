# S2 B11F Final-fit R02 Results and Audit v0.1

TASK-ID：`TASK-SLP-B11F-FINAL-FIT-R02-RESULT-AUDIT-v0.1`

Verdict：`ACCEPT`（P0/P1/P2 = `0/0/0`）

## 1. Run identity and evidence

- EXP-ID：`EXP-SLP-B11F-PM-FINAL-FIT-20260904-AUTODL-R02`
- Runner：`a6a5d8e6f8db003149169ee48f71d6e41e445a80`
- Authorization-package release：`21bda4e0bdf6fde2691a254593957e6350187540`
- Config SHA-256：`a6590d6f068644d98fa5340ec3d4a2e02171b529ec22ab092efb54a298925a43`
- Candidate Git-blob SHA-256：`34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b`
- B01 freeze-manifest SHA-256：`42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04`
- Authorized environment fingerprint：`a5a9342b18d00b614355e63ce056a7edd92dd80358d8aead5ef6e8e0ba045669`
- Launcher SHA-256：`0dea035c0af16b39617138177cdf441eb447463d55d90a06516b72dede5ade75`
- Evidence archive SHA-256：`a5a98916b79dca55366d6df6a8cd19df375fdb775b6e30ea775b8578cde70dad`

证据压缩包与 checkpoint 保留在 ignored local outputs，不提交 Git。

## 2. Terminal and budget audit

- 根终态唯一为 `DONE.json`；`RUNNING.json`、`STOPPED.json`、`FAILED.json` 均不存在。
- `models_complete/models_required = 3/3`。
- `environment.json` 文件 SHA 与 DONE carrier 一致；canonical fingerprint 独立重算后与授权值一致。
- `budget.json` 文件 SHA 和 budget-core SHA 均与 DONE carrier 一致。
- 首次启动至 deadline 精确为 `2700.0` 秒；终态 `DONE`。
- 实际累计 wall time 约 `279.819` 秒，终态剩余约 `2420.181` 秒；无预算重置、超时或时钟回退证据。
- TEST carrier：`test_access=false`，rows/labels/onehot=`0/0/0`。

## 3. Checkpoint audit

| Seed | Fixed epochs | Last training loss | Wall seconds | Peak CUDA MiB | Final SHA-256 | Reload |
|---:|---:|---:|---:|---:|---|---|
| 42 | 15 | 0.5575269840 | 89.511 | 330.730 | `633aed4a25aa2cfc42208ef3c610a78aed3569acf0d75fddd47361623e655af3` | match |
| 123 | 20 | 0.5199020513 | 118.530 | 330.951 | `e63415455816ea14dbbec4c54e9fd3c6c2f48be08de96fc8e60d2e1e94f7ffd5` | match |
| 2026 | 12 | 0.5658103683 | 69.644 | 330.951 | `1ce88a9b1b4797bd158795f3e796e3682970ba881e76d7fc8759f70bb2c7578f` | match |

三个 `complete.json` 与 DONE 的对应 result 完全一致。三个 `final.pt` 和三个 `last.pt`
均可在 CPU 上加载；每个 model state 有 28 个 tensor entry、53,449 个参数，全部为有限数值，
optimizer state 均存在。`last.pt` 还包含 epoch、budget、elapsed wall、peak CUDA、RNG 和
training-loss resume state。证据 manifest 中 12 个实验文件全部通过 SHA-256 校验。

## 4. Commands actually run

```text
Get-FileHash -Algorithm SHA256 outputs/analysis/b11f-final-r02-evidence.tar.gz
PASS; matches the transferred .sha256 record

archive member listing followed by extraction into a fresh ignored audit directory
PASS; no archive path traversal found

sha256sum -c ../b11f-final-r02-files.sha256
12/12 OK

strict JSON parse of environment/budget/DONE and three complete carriers
6/6 PASS; no NaN/Infinity constants

torch.load(..., map_location="cpu", weights_only=False) for three final.pt and three last.pt
6/6 PASS; model tensors finite; optimizer state present

Get-FileHash -Algorithm SHA256 launch_slp8_b11f_final_fit_r02.sh
PASS; 0dea035c0af16b39617138177cdf441eb447463d55d90a06516b72dede5ade75

git status --short --branch
## main...origin/main; clean at review start

git diff --check
PASS
```

## 5. Evidence interpretation

The archived `b11f-final-r02.exitcode` contains `1`, but it is not the training pipeline exit code.
After the pipeline had completed, the operator ran `screen -r b11f-final-r02` from an already attached
screen; that command returned 1, and a later `PIPESTATUS` capture recorded that unrelated status.
The file is preserved as an invalid operator capture and was not overwritten. Success is instead
established by the atomic unique DONE terminal, 3/3 complete carriers, checkpoint hashes and reload
checks. The transcript is empty because the runner emitted no stdout; it is not used as primary evidence.

## 6. Boundaries

### Verified

- Exact frozen identity, unique DONE terminal, continuous budget and three completed seeds.
- Three final and three resumable checkpoints are internally consistent, loadable and finite.
- TEST remained inaccessible and all TEST carriers remained zero.
- Actual peak CUDA use remained below the 8,192 MiB ceiling.

### Inferred

- The combined atomic terminal, per-seed carriers and checkpoint audit establish successful completion
  despite the invalid post-run exitcode capture.

### Unverified

- Bitwise reproducibility on another CUDA/driver/GPU environment.
- Interruption followed by a real cross-process resume; no resume was required in this successful run.
- Any TEST performance, product performance, comfort, medical, overnight or closed-loop control claim.

### Limitations

- Run mode records `cuda_determinism_unverified`; no cross-environment bitwise determinism claim is allowed.
- Reported loss is training loss, not validation or TEST performance.
- The source GT remains pressure-only, danaLab/uncover and `source_review_status=NOT_REVIEWED`.
- The empty transcript and invalid operator exitcode capture are permanent provenance limitations.

## 7. Next Gate

`B11F_FINAL_FIT_R02_ACCEPTED / THREE_CHECKPOINTS_AUDITED / B09T_PROTOCOL_DRAFT_READY / TEST_DENIED`

B09T may now be designed and independently reviewed. It must not run or read TEST until a separate,
one-time Owner authorization freezes the exact evaluator, checkpoint hashes, environment, command and
reporting rules.
