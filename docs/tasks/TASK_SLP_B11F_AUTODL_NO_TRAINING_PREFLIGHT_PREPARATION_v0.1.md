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
- Candidate Git blob SHA-256: `34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b`
- B01 freeze manifest SHA-256: `42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04`
- Complete-history bundle: `outputs/analysis/smarttopper-b11f-main-a6a5d8e.bundle`
- Bundle SHA-256: `5e9d855397face954cac18e3dbadb26449129f828f77d45412b3c4f30d8e6bb2`
- Bundle size: `2,150,560 bytes`
- R03 preflight script SHA-256: `e9fd730e8c8cbeb3f13508c8adb1a5765bebfd0e676a884322c29387ed27519c`
- R03 remote script: `/root/autodl-tmp/preflight_slp8_b11f_autodl_no_training_r03.sh`
- Formal EXP-ID: `NONE / NOT RESERVED`

The previous final-fit R01 EXP-ID, authorization and environment fingerprint are consumed and must
not be reused or resumed. AutoDL no-training preflight R01 also failed closed before environment
collection because its Windows working-tree candidate SHA did not equal the Linux Git blob SHA; its
transcript, script and checkout must not be overwritten or reused. R02 passed the corrected hash gate
but failed before environment collection because plain `uv run` omitted the declared `neural` extra;
its transcript, exitcode, script and checkout are also consumed and preserved. R03 must select the
locked neural dependency set with `uv run --extra neural` on every Python invocation.

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
bash /root/autodl-tmp/preflight_slp8_b11f_autodl_no_training_r03.sh
```

The uploaded script must be byte-identical to
`scripts/preflight_slp8_b11f_autodl_no_training.sh`. The Owner authorization must bind the future
release commit, runner SHA, bundle SHA, script SHA and the four input identities before the command
may be executed.

The script may create only a fresh checkout at
`/root/autodl-tmp/smarttopper-b11f-preflight-a6a5d8e-r03`. It may inspect GPU identity and collect the
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

`OWNER_AUTHORIZATION_FOR_EXACT_AUTODL_NO_TRAINING_PREFLIGHT_R03 / AUTODL_R03_NOT_AUTHORIZED / GPU_NOT_AUTHORIZED / TEST_DENIED`
