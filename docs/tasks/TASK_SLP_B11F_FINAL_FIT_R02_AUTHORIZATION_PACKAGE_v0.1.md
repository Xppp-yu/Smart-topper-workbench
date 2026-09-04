# TASK-SLP-B11F-FINAL-FIT-R02-AUTHORIZATION-PACKAGE-v0.1

状态：`READY_FOR_OWNER_AUTHORIZATION / GPU_NOT_AUTHORIZED / TEST_DENIED`

## 1. Purpose

Freeze one exact AutoDL B11F final-development-fit R02 launch and resume wrapper after the accepted
R03 no-training preflight. This task prepares and validates commands only. It does not execute GPU
training or access TEST.

## 2. Exact identity

- EXP-ID: `EXP-SLP-B11F-PM-FINAL-FIT-20260904-AUTODL-R02`
- Runner SHA: `a6a5d8e6f8db003149169ee48f71d6e41e445a80`
- Release SHA containing this package: `PENDING_MERGE`
- Bundle SHA-256: `5e9d855397face954cac18e3dbadb26449129f828f77d45412b3c4f30d8e6bb2`
- Config SHA-256: `a6590d6f068644d98fa5340ec3d4a2e02171b529ec22ab092efb54a298925a43`
- Candidate Git-blob SHA-256: `34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b`
- B01 freeze-manifest SHA-256: `42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04`
- Authorized environment fingerprint: `a5a9342b18d00b614355e63ce056a7edd92dd80358d8aead5ef6e8e0ba045669`
- Launcher LF payload SHA-256: `0dea035c0af16b39617138177cdf441eb447463d55d90a06516b72dede5ade75`
- Wall budget: `2700` seconds from the first launch deadline; downtime counts and resume cannot reset it.
- TEST: denied; expected rows/labels/onehot remain `0/0/0`.

The historical R01 EXP-ID is consumed and forbidden. R02 preflight attempts R01/R02 and their
checkouts/transcripts are also preserved and must not be reused. The accepted R03 preflight checkout
is reused intentionally so the locked `.venv` and observed environment are identical.

## 3. Exact wrapper

Tracked source: `scripts/launch_slp8_b11f_final_fit_r02.sh`

Transfer payload: `outputs/analysis/launch_slp8_b11f_final_fit_r02.lf.sh` (ignored, LF bytes)

Remote path: `/root/autodl-tmp/launch_slp8_b11f_final_fit_r02.sh`

First launch, only after exact Owner authorization:

```bash
bash /root/autodl-tmp/launch_slp8_b11f_final_fit_r02.sh run
```

Resume is separately authorized only when R02 has exactly one nonterminal root state and its original
budget remains valid:

```bash
bash /root/autodl-tmp/launch_slp8_b11f_final_fit_r02.sh resume
```

The wrapper refuses identity/hash/path drift, an existing output on first launch, DONE/FAILED resume,
ambiguous RUNNING+STOPPED state, missing nonterminal state, expired/increased budget, and environment
drift. It never uses `--force`, never resets the deadline, and always selects `uv run --extra neural`.

## 4. Authorization boundary

Owner authorization must repeat the exact EXP-ID, runner SHA, launcher SHA, environment fingerprint,
2700-second budget, `run` mode and TEST denial. Authorization for `run` does not authorize `resume`.
Any failure consumes this EXP-ID and requires evidence review; it must not be overwritten or retried.

Current Gate:

`OWNER_AUTHORIZATION_FOR_EXACT_B11F_FINAL_FIT_R02 / GPU_NOT_AUTHORIZED / TEST_DENIED`
