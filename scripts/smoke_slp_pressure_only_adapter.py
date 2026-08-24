#!/usr/bin/env python3
"""Real-data Smoke Test for SLP Pressure-only Adapter.

This script performs a real-data smoke test on the pressure-only adapter
using actual SLP PM PNG files from the dataset.

Usage:
    uv run python scripts/smoke_slp_pressure_only_adapter.py \
        --canonical-jsonl "path/to/slp_canonical_samples_v0.1.jsonl" \
        --split-manifest "path/to/slp_subject_split_v0.1.json" \
        --slp-root "path/to/SLP" \
        --sample-count 3

Requirements:
- Reads TRAIN split only (default).
- At least 3 non-quarantine danaLab PM frames.
- Outputs sample_id, split, source URI, shape, dtype, min/max.
- Verifies visual_modalities_loaded=False.
- Non-zero exit code if any frame violates contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topper_perception.io.slp_pressure_only_adapter import (
    SlpPressureOnlyAdapter,
    load_a05_canonical_samples_jsonl,
    load_a06_split_manifest,
    DataSplit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-data smoke test for SLP pressure-only adapter."
    )
    parser.add_argument(
        "--canonical-jsonl",
        type=str,
        required=True,
        help="Path to A05 canonical samples JSONL file.",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        required=True,
        help="Path to A06 split manifest JSON file.",
    )
    parser.add_argument(
        "--slp-root",
        type=str,
        required=True,
        help="Path to SLP dataset root directory.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="Minimum number of samples to read (default: 3).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Validate input paths
    canonical_path = Path(args.canonical_jsonl)
    split_path = Path(args.split_manifest)
    slp_root = Path(args.slp_root)

    if not canonical_path.exists():
        print(f"ERROR: Canonical JSONL not found: {canonical_path}", file=sys.stderr)
        return 1
    if not split_path.exists():
        print(f"ERROR: Split manifest not found: {split_path}", file=sys.stderr)
        return 1
    if not slp_root.exists():
        print(f"ERROR: SLP root not found: {slp_root}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("SLP Pressure-only Adapter - Real Data Smoke Test")
    print("=" * 70)
    print(f"Canonical JSONL: {canonical_path}")
    print(f"Split manifest:  {split_path}")
    print(f"SLP root:        {slp_root}")
    print(f"Sample count:    {args.sample_count}")
    print()

    # Load data
    print("Loading A05 canonical samples...")
    try:
        samples = load_a05_canonical_samples_jsonl(canonical_path)
        print(f"  Loaded {len(samples)} canonical samples")
    except Exception as e:
        print(f"ERROR: Failed to load canonical samples: {e}", file=sys.stderr)
        return 1

    print("Loading A06 split manifest...")
    try:
        split_manifest = load_a06_split_manifest(split_path)
        print(f"  Loaded {len(split_manifest)} subject splits")
    except Exception as e:
        print(f"ERROR: Failed to load split manifest: {e}", file=sys.stderr)
        return 1

    # Create adapter
    print("Creating adapter with load_pressure_data=True...")
    try:
        adapter = SlpPressureOnlyAdapter(
            canonical_samples=samples,
            split_manifest=split_manifest,
            slp_root=slp_root,
            load_pressure_data=True,
        )
    except Exception as e:
        print(f"ERROR: Failed to create adapter: {e}", file=sys.stderr)
        return 1

    # Iterate TRAIN split, non-quarantine
    print()
    print("Reading TRAIN split (non-quarantine) samples...")
    print("-" * 70)

    samples_read = 0
    errors = []
    non_danaLab_skipped = 0

    for sample in adapter.iter_samples(
        include_quarantine=False,
        split=DataSplit.TRAIN,
    ):
        # Filter to danaLab only (simLab has no real PM data)
        if sample.setting != "danaLab":
            non_danaLab_skipped += 1
            continue

        # Check provenance contract
        assert sample.provenance.visual_modalities_loaded is False, (
            f"FAIL: visual_modalities_loaded should be False, got True for {sample.sample_id}"
        )
        assert sample.provenance.model_input_tensor_modalities == ("PM",), (
            f"FAIL: model_input_tensor_modalities should be ('PM',), "
            f"got {sample.provenance.model_input_tensor_modalities} for {sample.sample_id}"
        )

        # Validate PM contract
        pm = sample.pressure_map
        contract_errors = []

        # Check shape (192, 84)
        if pm.shape != (192, 84):
            contract_errors.append(
                f"shape={pm.shape} (expected (192, 84))"
            )

        # Check dtype
        if pm.dtype != np.float32:
            contract_errors.append(
                f"dtype={pm.dtype} (expected float32)"
            )

        # Check value range [0, 1]
        min_val, max_val = pm.min(), pm.max()
        if min_val < 0.0 or max_val > 1.0:
            contract_errors.append(
                f"range=[{min_val:.6f}, {max_val:.6f}] (expected [0.0, 1.0])"
            )

        # Check finite
        if not np.all(np.isfinite(pm)):
            contract_errors.append("contains NaN or Inf values")

        if contract_errors:
            errors.append((sample.sample_id, contract_errors))
            print(f"  CONTRACT VIOLATION: {sample.sample_id}")
            for err in contract_errors:
                print(f"    - {err}")
            print()
            continue

        # Print success
        print(f"  sample_id:    {sample.sample_id}")
        print(f"  split:       {sample.split}")
        print(f"  source_uri:  {sample.pressure_map_uri}")
        print(f"  shape:       {pm.shape}")
        print(f"  dtype:       {pm.dtype}")
        print(f"  min/max:     [{min_val:.6f}, {max_val:.6f}]")
        print(f"  quarantine:  {sample.quarantine}")
        print()

        samples_read += 1

        if samples_read >= args.sample_count:
            break

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Samples read:      {samples_read}")
    print(f"  danaLab skipped:   {non_danaLab_skipped}")
    print(f"  Contract errors:  {len(errors)}")

    if non_danaLab_skipped > 0:
        print()
        print(f"  NOTE: {non_danaLab_skipped} non-danaLab samples were skipped")
        print("        (simLab has no real PM data in SLP dataset)")

    if errors:
        print()
        print("CONTRACT VIOLATIONS DETECTED:")
        for sample_id, errs in errors:
            print(f"  {sample_id}:")
            for err in errs:
                print(f"    - {err}")
        return 1

    if samples_read < args.sample_count:
        print()
        print(f"ERROR: Only read {samples_read} samples, requested {args.sample_count}")
        print("       (not enough TRAIN split samples available)")
        return 1

    print()
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
