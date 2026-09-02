"""B10 seed-consensus rejection analysis for B09 TRAIN+VAL OOF hard masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _subject(sample_id: str) -> str:
    parts = sample_id.split(":")
    if len(parts) != 5 or parts[:2] != ["SLP", "danaLab"] or parts[3] != "uncover":
        raise ValueError(f"invalid governed sample_id: {sample_id!r}")
    return parts[2]


def analyze(paths: list[Path]) -> tuple[dict, list[dict], list[dict]]:
    if len(paths) != 3:
        raise ValueError("exactly three seed OOF files are required")
    arrays = [np.load(p, allow_pickle=True) for p in paths]
    required = {"predictions", "targets", "sample_ids", "candidate", "seed"}
    for path, z in zip(paths, arrays):
        if not required.issubset(z.files):
            raise ValueError(f"{path}: missing keys {sorted(required - set(z.files))}")
    candidates = {str(z["candidate"].item()) for z in arrays}
    seeds = {int(z["seed"].item()) for z in arrays}
    if len(candidates) != 1 or len(seeds) != 3:
        raise ValueError("candidate must match and seeds must be unique")
    sample_ids = arrays[0]["sample_ids"].astype(str)
    target = arrays[0]["targets"]
    if target.shape[0] != len(sample_ids):
        raise ValueError("sample/target length mismatch")
    for z in arrays[1:]:
        if not np.array_equal(sample_ids, z["sample_ids"].astype(str)):
            raise ValueError("sample order mismatch across seeds")
        if not np.array_equal(target, z["targets"]):
            raise ValueError("target mismatch across seeds")
    pred = np.stack([z["predictions"] for z in arrays], axis=0)
    if pred.shape[1:] != target.shape:
        raise ValueError("prediction/target shape mismatch")
    classes = np.arange(int(max(pred.max(), target.max())) + 1)
    counts = np.stack([(pred == c).sum(axis=0) for c in classes], axis=0)
    majority = counts.argmax(axis=0)
    unanimous = counts.max(axis=0) == 3
    correct = majority == target
    foreground = target != 0

    def metrics(mask: np.ndarray) -> dict:
        denom = int(mask.sum())
        return {
            "pixels": denom,
            "coverage": float(mask.mean()),
            "error_rate": float((~correct & mask).sum() / denom) if denom else None,
        }

    raw_errors = ~correct
    summary = {
        "candidate": next(iter(candidates)),
        "seeds": sorted(seeds),
        "samples": int(len(sample_ids)),
        "test_access": False,
        "confidence_semantics": "three_seed_hard_prediction_consensus_not_probability",
        "all_pixels": metrics(np.ones_like(unanimous, dtype=bool)),
        "unanimous_pixels": metrics(unanimous),
        "foreground_unanimous_coverage": float((unanimous & foreground).sum() / foreground.sum()),
        "rejected_error_capture_rate": float((raw_errors & ~unanimous).sum() / raw_errors.sum()),
        "unanimous_error_fraction_of_raw_errors": float((raw_errors & unanimous).sum() / raw_errors.sum()),
        "input_sha256": {p.name: _sha256(p) for p in paths},
    }
    per_subject = []
    error_rows = []
    subjects = np.array([_subject(x) for x in sample_ids])
    for subject in sorted(set(subjects)):
        sm = subjects == subject
        m = np.broadcast_to(sm[:, None, None], unanimous.shape)
        fg = foreground & m
        err = raw_errors & m
        per_subject.append({
            "subject_id": subject,
            "samples": int(sm.sum()),
            "unanimous_coverage": float(unanimous[m].mean()),
            "foreground_unanimous_coverage": float((unanimous & fg).sum() / fg.sum()) if fg.any() else None,
            "majority_error_rate": float(raw_errors[m].mean()),
            "unanimous_error_rate": float((err & unanimous).sum() / unanimous[m].sum()),
        })
    sample_error = (raw_errors & unanimous).reshape(len(sample_ids), -1).sum(axis=1)
    total = np.prod(target.shape[1:])
    for i in np.argsort(sample_error)[-50:][::-1]:
        error_rows.append({"sample_id": sample_ids[i], "subject_id": subjects[i],
                           "unanimous_wrong_pixels": int(sample_error[i]),
                           "unanimous_wrong_fraction": float(sample_error[i] / total)})
    return summary, per_subject, error_rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--oof", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    summary, subjects, errors = analyze(args.oof)
    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("per_subject.csv", subjects), ("high_consensus_errors.csv", errors)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
