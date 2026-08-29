"""Minimal CPU/CUDA synthetic smoke for the B04A implementation stage.

This script exercises the three B04A candidates
(``slp8_small_unet_v0.1``, ``slp8_resunet_lite_v0.1``,
``slp8_deeplabv3plus_lite_v0.1``) end-to-end on synthetic data. It
performs:

  1. Build each model.
  2. Forward pass on CPU (mandatory).
  3. Backward pass on CPU (mandatory) — one AdamW step.
  4. Checkpoint save and reload.
  5. Same-seed determinism.
  6. CUDA smoke (forward + backward) if and only if a CUDA device is
     available; otherwise the CUDA phase is reported as ``NOT RUN`` and
     the script does NOT mark the run as failed.

It writes a single JSON summary to the path given by ``--output``
(default: ``outputs/reports/b04a_implementation_smoke_v0.1.json``).

It never imports or reads B01 training tables, never accesses TEST, and
never runs an epoch of real training.

Output policy
-------------

* If ``--output`` points to an existing file, the script REFUSES to
  overwrite it and exits non-zero with a clear error.  This prevents
  silent ``write_text`` clobbering of historical artifacts.
* Pass ``--force`` to allow overwrite (use with care; the smoke summary
  is a fresh declaration, not an historical artifact).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Make src/ importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topper_perception.neural.slp8_region_models import (  # noqa: E402
    B04A_EXACT_PARAMETER_COUNTS,
    DEEPLABV3PLUS_LITE_VERSION,
    INPUT_SHAPE,
    N_CLASSES,
    RESUNET_LITE_VERSION,
    SMALL_UNET_VERSION,
    Slp8DeepLabV3PlusLite,
    Slp8ResUnetLite,
    Slp8SmallUnet,
)


# Tuple of (model class, version string) for the three B04A candidates.
CANDIDATES: tuple[tuple[type[nn.Module], str], ...] = (
    (Slp8SmallUnet, SMALL_UNET_VERSION),
    (Slp8ResUnetLite, RESUNET_LITE_VERSION),
    (Slp8DeepLabV3PlusLite, DEEPLABV3PLUS_LITE_VERSION),
)


# Declarative policy recorded in the JSON summary.
#
# This is NOT a runtime counter of how many times TEST was read; TEST
# is never read by this script.  It is a static declaration that the
# script's smoke path does not invoke any B01 TEST access contract.
# The 0 is therefore a contract, not a measured quantity.
TEST_ACCESS_DECLARATION: dict[str, Any] = {
    "value": 0,
    "kind": "declarative_policy",
    "explanation": (
        "The B04A implementation smoke does not import any B01 training "
        "table loader and does not invoke enable_test_access(...). The 0 "
        "is a static declaration, NOT a runtime count of TEST reads."
    ),
}


def _build_synthetic_batch(batch_size: int = 2, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(
        batch_size, 1, INPUT_SHAPE[0], INPUT_SHAPE[1], generator=g, dtype=torch.float32
    )
    labels = torch.randint(
        0, N_CLASSES, (batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1]),
        generator=g, dtype=torch.long,
    )
    return x, labels


def _smoke_one(
    cls: type[nn.Module], version: str, device: torch.device
) -> dict[str, Any]:
    """Run CPU/CUDA forward+backward smoke for one candidate."""

    result: dict[str, Any] = {
        "version": version,
        "device": str(device),
        "param_count": B04A_EXACT_PARAMETER_COUNTS[version],
        "forward_shape": None,
        "forward_finite": None,
        "backward_finite": None,
        "checkpoint_roundtrip_equal": None,
    }

    torch.manual_seed(42)
    model = cls().to(device)
    x, labels = _build_synthetic_batch(2, seed=42)
    x = x.to(device)
    labels = labels.to(device)

    # Forward
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    forward_ms = (time.perf_counter() - t0) * 1000.0
    result["forward_shape"] = list(y.shape)
    result["forward_finite"] = bool(torch.isfinite(y).all().item())
    result["forward_ms"] = forward_ms

    # Backward
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    y = model(x)
    loss = loss_fn(
        y.reshape(y.shape[0], N_CLASSES, -1),
        labels.reshape(2, -1),
    )
    loss.backward()
    optimizer.step()
    result["backward_finite"] = bool(
        all(
            (p.grad is not None) and torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.requires_grad
        )
    )
    result["loss_value"] = float(loss.item())

    # Checkpoint roundtrip
    model.eval()
    with torch.no_grad():
        y_before = model(x)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model2 = cls()
    model2.load_state_dict(state)
    model2.eval()
    with torch.no_grad():
        y_after = model2(x.cpu())
    result["checkpoint_roundtrip_equal"] = bool(torch.equal(y_before.cpu(), y_after))

    # Same-seed determinism
    torch.manual_seed(123)
    model3 = cls()
    model3.eval()
    with torch.no_grad():
        y_a = model3(x.cpu())
    torch.manual_seed(123)
    model4 = cls()
    model4.eval()
    with torch.no_grad():
        y_b = model4(x.cpu())
    result["deterministic_same_seed"] = bool(torch.equal(y_a, y_b))

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "b04a_implementation_smoke_v0.1.json",
        help=(
            "Path to write the JSON summary. "
            "Refuses to overwrite an existing file unless --force is set."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting the output file (use with care).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Run the smoke pipeline but DO NOT write any file.  Used by "
            "Codex Reviewer to record a no-write smoke run; in --no-write "
            "mode the script prints a one-line summary to stdout instead."
        ),
    )
    args = parser.parse_args(argv)

    cpu = torch.device("cpu")
    cuda_available = bool(torch.cuda.is_available())
    cuda_device: torch.device | None = (
        torch.device("cuda") if cuda_available else None
    )

    results: list[dict[str, Any]] = []
    for cls, version in CANDIDATES:
        # CPU smoke (mandatory).
        cpu_result = _smoke_one(cls, version, cpu)
        results.append({**cpu_result, "phase": "cpu"})

    # CUDA smoke (optional; only if available).
    if cuda_available and cuda_device is not None:
        for cls, version in CANDIDATES:
            cuda_result = _smoke_one(cls, version, cuda_device)
            results.append({**cuda_result, "phase": "cuda"})
    else:
        results.append({
            "phase": "cuda",
            "version": "all",
            "status": "NOT_RUN",
            "reason": (
                "torch.cuda.is_available() is False on this host; "
                "B04A R03 explicitly permits recording CUDA Smoke as "
                "NOT RUN when CUDA is unavailable. The CPU smoke is "
                "mandatory and has been recorded for every candidate."
            ),
        })

    payload: dict[str, Any] = {
        "task_id": "TASK-SLP-B04A-IMPLEMENTATION-SMOKE-v0.1",
        "stage": "S2_B04A",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "candidates": [
            {
                "version": v,
                "exact_parameter_count": B04A_EXACT_PARAMETER_COUNTS[v],
            }
            for _, v in CANDIDATES
        ],
        "test_access": TEST_ACCESS_DECLARATION,
        "results": results,
        "notes": [
            "B01 training tables, TEST rows, and TEST labels are NEVER "
            "loaded by this smoke script.",
            "CUDA smoke is recorded as NOT_RUN when the host's torch "
            "build is CPU-only; CPU smoke is mandatory and is run for "
            "every B04A candidate.",
        ],
    }

    if args.no_write:
        # Print a one-line summary and exit 0; nothing is written to disk.
        n_cpu = sum(1 for r in results if r.get("phase") == "cpu")
        all_cpu_ok = all(
            r.get("forward_finite") and r.get("backward_finite")
            and r.get("checkpoint_roundtrip_equal") and r.get("deterministic_same_seed")
            for r in results
            if r.get("phase") == "cpu"
        )
        print(
            f"B04A_SMOKE_NO_WRITE cpu_candidates={n_cpu} "
            f"cuda_run={cuda_available} all_cpu_ok={all_cpu_ok}"
        )
        return 0

    out_path: Path = args.output
    if out_path.exists() and not args.force:
        print(
            f"ERROR: output file already exists: {out_path}. "
            f"Refusing to overwrite. Pass --force to allow overwrite, or "
            f"pass --output to a different path.",
            file=sys.stderr,
        )
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
