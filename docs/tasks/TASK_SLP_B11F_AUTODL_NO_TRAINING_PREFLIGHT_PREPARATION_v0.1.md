# TASK-SLP-B11F-AUTODL-NO-TRAINING-PREFLIGHT-PREPARATION-v0.1

## 1. Objective

Prepare, validate and review the exact transfer bundle, immutable hashes and shell commands for a
future AutoDL environment-only preflight after B11F production-wiring smoke R07 acceptance.

This task does not authorize connecting to AutoDL or executing the script. It does not reserve an
EXP-ID, create a formal experiment output, load a training batch, train, resume, create a checkpoint
or access TEST.

## 2. Frozen baseline

- Runner Git SHA: `a6a5d8e6f8db003149169ee48f71d6e41e445a80`
- Config SHA-256: `a6590d6f068644d98fa5340ec3d4a2e02171b529ec22ab092efb54a298925a43`
- Candidate SHA-256: `839c9482c69cf34d3c91c3acb3c7a36cb4d199117d0d6eb2ceb7906bac52b994`
- B01 freeze manifest SHA-256: `42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04`
- Complete-history bundle: `outputs/analysis/smarttopper-b11f-main-a6a5d8e.bundle`
- Bundle SHA-256: `5e9d855397face954cac18e3dbadb26449129f828f77d45412b3c4f30d8e6bb2`
- Bundle size: `2,150,560 bytes`
- Preflight script SHA-256: `e9cdf8240cf5e2fe10020adb5325657b9ef53b1907f079b05edb10cc87cd85bb`
- Formal EXP-ID: `NONE / NOT RESERVED`

The previous R01 EXP-ID, authorization, environment fingerprint and bundle are consumed and must not
be reused or resumed.

## 3. Allowed changes

- `configs/experiments/slp8_b11f_autodl_no_training_preflight_v0.1.json`
- `scripts/preflight_slp8_b11f_autodl_no_training.sh`
- `scripts/validate_slp8_b11f_autodl_preflight_preparation.py`
- `tests/test_slp8_b11f_autodl_preflight_preparation.py`
- this task and its stage report
- narrow B11F entries in `docs/PROJECT_STATUS.md` and `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md`
- ignored bundle under `outputs/analysis/`

Production runner/config/candidate/B01 artifacts and raw datasets are read-only.

## 4. Exact future preflight

The exact prepared command is:

```bash
bash /root/autodl-tmp/preflight_slp8_b11f_autodl_no_training.sh
```

The uploaded script must be byte-identical to
`scripts/preflight_slp8_b11f_autodl_no_training.sh`. The Owner authorization must bind the future
release commit, runner SHA, bundle SHA, script SHA and the four input identities before the command
may be executed.

The script may create only a fresh checkout at
`/root/autodl-tmp/smarttopper-b11f-preflight-a6a5d8e`. It may inspect GPU identity and collect the
canonical environment fingerprint. It may not create anything under `outputs/experiments/` or call
any formal run/resume argument.

## 5. Pass evidence required from AutoDL

- complete transcript and every command exit status;
- exact clean detached checkout and bundle origin/ref;
- bundle/config/candidate/B01 SHA matches;
- GPU/PyTorch/CUDA/cuDNN environment payload;
- canonical `environment_fingerprint_sha256` from the production helper;
- validator and validate-only markers with TEST=0;
- formal experiment output absent before and after;
- final marker `B11F_AUTODL_NO_TRAINING_PREFLIGHT_PASSED TEST=0 TRAINING_NOT_STARTED`.

The transcript must then receive a separate read-only review. A successful preflight does not itself
authorize final fit.

## 6. Local verification commands

```bash
git bundle verify outputs/analysis/smarttopper-b11f-main-a6a5d8e.bundle
uv run python scripts/validate_slp8_b11f_autodl_preflight_preparation.py
uv run python -m pytest tests/test_slp8_b11f_autodl_preflight_preparation.py -q
uv run python -m pytest tests/test_slp8_b11f_autodl_preflight_preparation.py tests/test_slp8_b11f_production_wiring.py tests/test_slp8_region_final_fit.py tests/test_slp8_b11_candidate_freeze.py tests/test_slp8_region_full.py -q
bash -n scripts/preflight_slp8_b11f_autodl_no_training.sh
uv run python -m pytest tests/test_check_markdown_links.py -q
uv run python -m py_compile scripts/validate_slp8_b11f_autodl_preflight_preparation.py
git diff --check
git status --short --branch
```

## 7. Current Gate

The Owner authorized closure commit/push after direct technical review on `2026-09-04`. This
authorization does not permit an AutoDL connection, preflight execution, GPU training or TEST
access.

`OWNER_AUTHORIZATION_FOR_EXACT_AUTODL_NO_TRAINING_PREFLIGHT / AUTODL_NOT_AUTHORIZED / GPU_NOT_AUTHORIZED / TEST_DENIED`
