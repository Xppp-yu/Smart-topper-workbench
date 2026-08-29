# TASK-SLP-B04A-MINI-RUN-v0.1

**Status:** `RUN_PREPARATION_READY_FOR_REVIEW / GPU_MINI_NOT_AUTHORIZED`
**Stage:** S2-B04A (real TRAIN/VAL GPU Mini)
**Date:** 2026-08-30
**Proposed EXP-ID:** `EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-LOCAL-R01`

> This task is the executable contract for the real B04A Mini. Preparing or
> accepting this document does not authorize the run. The Owner must separately
> approve the final EXP-ID, clean Git SHA, machine, budget, and the
> `--run-authorized` invocation after all preflight gates pass.

## Objective

Execute the frozen B04A pressure-only architecture expansion Mini on the B01
TRAIN/VAL contract, using exactly three candidates and three registered seeds,
then preserve a complete immutable artifact bundle for independent review. The
Mini may advance at most two candidates to B07 protocol design; it does not
select a final champion.

## Prerequisites

- B04A protocol, implementation, CPU/CUDA-capable implementation smoke, and
  runner integration are accepted.
- The run starts from a clean, committed Git baseline. The final SHA is recorded
  only after this preparation task is reviewed and released; `236f14e` is the
  preparation parent, not the final run SHA.
- Frozen config:
  `configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json`.
- B01 freeze bundle is restored read-only and validates against its embedded
  identity. The available transfer archive is
  `E:\TeamProjects\autodl-transfer\slp8_training_tables_v0.1.tar.gz`, expected
  archive SHA-256
  `23B32395238130437C1EC1B0771FBC793B8BE74578B0A1ACB2CA237C1913269A`;
  its B01 freeze manifest core SHA-256 must remain
  `3c78999551580fc46ce15229e053798b5e4c9464a5bab27e05130cb319090b1e`.
- SLP8 dataset root exists at
  `E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\SLP_8Region_Pressure_VAL_v1.1`.
- A CUDA-enabled PyTorch environment passes the no-write CUDA smoke. The
  repository's currently observed `uv` environment is CPU-only and therefore
  is not sufficient for the real run until the user's terminal environment is
  verified.
- Owner separately authorizes the final command, EXP-ID, and compute budget.

## Frozen run contract

| Item | Frozen value |
|---|---|
| Candidates | `slp8_small_unet_v0.1`, `slp8_resunet_lite_v0.1`, `slp8_deeplabv3plus_lite_v0.1` |
| Deferred/forbidden | SegFormer-B0 remains `DEFERRED`; TinyFCN is forbidden |
| Seeds | `42`, `123`, `2026` |
| Data | B01 TRAIN 3,645 / VAL 450 / TEST 0; danaLab/uncover only |
| Device | CUDA |
| Training | batch size 16; max 30 epochs; min 5 epochs; AdamW; lr 0.001; weight decay 0.0001; no scheduler; no augmentation; AMP disabled |
| Determinism | deterministic algorithms; cuDNN deterministic; benchmark false; one CPU thread |
| Feasibility | fixed foreground Macro IoU >= 0.355644 |
| Hard guards | no class collapse; worst-subject Macro IoU >= 0.20; every foreground region IoU >= 0.05; all seeds must succeed |
| Budget | 45 cumulative minutes per candidate; 135 cumulative minutes total; peak CUDA <= 8192 MB |
| Advancement | 0/1/2/3-feasible frozen rule; at most two advance; near-tie margin 0.02 with frozen tiebreak |
| TEST | no TEST rows, labels, onehot, class statistics, or `enable_test_access` call |

No value in this table may be changed after observing Mini output. A necessary
change creates a new TASK-ID/config version/EXP-ID and leaves the original run
auditable.

## Files and data boundaries

### May be written by the Runner

- `outputs/experiments/<EXP-ID>/` only.
- Restored B01 artifacts under `data/processed/slp8_training_tables_v0.1/`
  are inputs and must remain unchanged during training.

### Must remain read-only

- `E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\SLP_8Region_Pressure_VAL_v1.1`.
- `configs/experiments/slp8_pm_architecture_expansion_mini_v0.1.json` after
  final config hash is recorded.
- Any prior B04/B04A output directory.

### Out of scope

- Editing model, Runner, optimizer, Gate, candidate, seed, or data code.
- Running B07/Full or reading TEST.
- Adding SegFormer-B0 after seeing CNN results.
- Overwriting or deleting an existing output directory.
- Commit, push, merge, cloud upload, or package installation without separate
  authorization applicable to that action.

## Preflight procedure

Run these commands from a clean released workbench in PowerShell. They do not
authorize or start real training.

```powershell
$B04ARepo = "E:\TeamProjects\smarttopper-team-workbench"
$B04AArchive = "E:\TeamProjects\autodl-transfer\slp8_training_tables_v0.1.tar.gz"
$B04AFreezeParent = Join-Path $B04ARepo "data\processed"
$B04AFreezeDir = Join-Path $B04AFreezeParent "slp8_training_tables_v0.1"
$B04ADataRoot = "E:\TeamProjects\datasets\smart-topper\SLP2022\SLP\SLP_8Region_Pressure_VAL_v1.1"
$B04AConfig = Join-Path $B04ARepo "configs\experiments\slp8_pm_architecture_expansion_mini_v0.1.json"
$B04APreflight = Join-Path $B04ARepo "outputs\preflight\B04A_MINI_20260830_R01"

git -C $B04ARepo status --short --branch
git -C $B04ARepo rev-parse HEAD
git -C $B04ARepo rev-list --left-right --count origin/main...HEAD

Test-Path -LiteralPath $B04AArchive
Test-Path -LiteralPath $B04ADataRoot
(Get-FileHash -Algorithm SHA256 -LiteralPath $B04AArchive).Hash

New-Item -ItemType Directory -Force -Path $B04AFreezeParent | Out-Null
if (Test-Path -LiteralPath $B04AFreezeDir) {
    throw "Refusing to extract over existing B01 freeze directory: $B04AFreezeDir"
}
tar -xzf $B04AArchive -C $B04AFreezeParent

Set-Location $B04ARepo
uv run python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
uv run python scripts\validate_b04a_protocol.py $B04AConfig
uv run python scripts\smoke_b04a_implementation.py --no-write
uv run python scripts\run_slp8_region_mini.py --config $B04AConfig --output-dir $B04APreflight --validate-config
```

Preflight passes only when all of the following are true:

- Git is clean, committed, and synchronized with the intended released branch.
- The archive hash exactly matches the value above and extraction creates the
  expected B01 directory without overwriting anything.
- `torch.cuda.is_available()` is `True`; CUDA smoke is actually executed and
  passes, not reported as `NOT RUN`.
- Protocol validation reports `30 OKs / 0 errors`.
- Validate-only produces a successful B04A identity bundle without reading B01
  or TEST.
- The final real output directory does not exist.

If the current `uv` environment remains CPU-only, stop. Installing or replacing
PyTorch is an environment task and must not be improvised inside the frozen
experiment run.

## Owner authorization record

Before the real command is entered, record all fields:

```text
Owner authorization: PENDING
Final EXP-ID: PENDING (proposed: EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-LOCAL-R01)
Final Git SHA: PENDING
Git dirty: must be false
Config SHA-256: PENDING
B01 data/freeze manifest SHA-256: PENDING
B01 freeze manifest core SHA-256: expected 3c78999551580fc46ce15229e053798b5e4c9464a5bab27e05130cb319090b1e
A06 split SHA-256: expected 024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706
Machine/GPU: PENDING
Peak CUDA budget: 8192 MB
Total wall budget: 135 cumulative minutes
TEST access: denied / 0
```

## Real run command template

Do not execute this block until the authorization record is complete.

```powershell
$B04AExpId = "EXP-SLP-B04A-PM-ARCH-EXPANSION-MINI-20260830-LOCAL-R01"
$B04AOutput = Join-Path $B04ARepo "outputs\experiments\$B04AExpId"

if (Test-Path -LiteralPath $B04AOutput) {
    throw "Refusing to reuse existing EXP-ID/output: $B04AOutput"
}

uv run python scripts\run_slp8_region_mini.py `
  --config $B04AConfig `
  --output-dir $B04AOutput `
  --b01-freeze-dir $B04AFreezeDir `
  --dataset-root $B04ADataRoot `
  --run-authorized
```

For an interrupted non-terminal run, use the same EXP-ID and identity only:

```powershell
uv run python scripts\run_slp8_region_mini.py `
  --config $B04AConfig `
  --output-dir $B04AOutput `
  --b01-freeze-dir $B04AFreezeDir `
  --dataset-root $B04ADataRoot `
  --resume-from $B04AOutput `
  --run-authorized
```

Never resume a `DONE` run and never use `--force` for a real Mini.

## Expected artifacts

The output bundle must contain the frozen protocol's complete artifact list,
including `manifest.json`, `resolved_config.json`, identity/hash files,
environment, status, epoch and aggregate metrics, per-region/per-subject/
per-posture evidence, centroid errors, confusion matrix, predictions manifest,
candidate decision, reload consistency, budget report, logs, all nine seed
checkpoint pairs, and exactly one terminal file: `DONE.json`, `FAILED.json`, or
`STOPPED.json`.

## Fail-closed conditions

- Dirty or uncommitted Git baseline; wrong released SHA; identity/hash drift.
- Archive/freeze/data path missing, archive hash mismatch, or attempted overwrite.
- CUDA unavailable; CUDA smoke `NOT RUN`; peak CUDA or wall budget exceeded.
- Any TEST access or TEST-derived statistic.
- Candidate/seed/config mismatch; partial-seed mean; missing identity carrier.
- Subject overlap, non-finite value, OOM, class collapse, worst-subject or
  per-region guard failure, checkpoint/reload mismatch.
- Existing output directory or multiple terminal files.

A fail-closed run preserves evidence and stops. It is not silently repaired or
relabelled as a completed experiment.

## Acceptance criteria

- Owner authorization was recorded before the real command.
- The immutable EXP identity agrees across every artifact carrier.
- TRAIN=3,645 / VAL=450 / TEST=0 and subject isolation are preserved.
- All three candidates and all three seeds were attempted under the frozen
  budget; candidate feasibility obeys `all_seeds_must_succeed=true`.
- Reload consistency and output completeness pass.
- Reviewer independently recomputes the decision and records
  `ACCEPT / ITERATE / STOP / INVALID`.
- Only after Reviewer acceptance may at most two candidates enter B07.

## Prohibited conclusions

- Mini identifies the universally best segmentation architecture.
- A successful process exit equals Reviewer acceptance.
- B07 is ready before independent review.
- Results apply to cover conditions, TEST subjects, self-developed hardware,
  calibrated physical pressure, comfort, medical outcomes, overnight use, or
  airbag control.

## Reviewer checklist

- [ ] Authorization timestamp precedes run start.
- [ ] Final Git/config/data/split/model identity is complete and reproducible.
- [ ] Raw data and B01 freeze inputs were not modified.
- [ ] TEST rows/labels/onehot/statistics loaded = 0.
- [ ] Three candidates x three seeds are complete or fail-closed with evidence.
- [ ] Metrics and guardrails are independently recomputed from saved outputs.
- [ ] Checkpoint reload predictions agree.
- [ ] Candidate decision and near-tie topology match the frozen R03 rule.
- [ ] Resource budget and terminal-state exclusivity pass.
- [ ] Limitations and prohibited conclusions are retained.

## Next Gate

Current Gate remains `GPU_MINI_NOT_AUTHORIZED`. After preparation review,
successful preflight, and explicit Owner authorization, the Experiment Runner
may enter `QUEUED` and then `RUNNING`. B07 remains `BLOCKED_BY_B04A` until the
completed Mini bundle is independently accepted and at most two candidates are
frozen.
