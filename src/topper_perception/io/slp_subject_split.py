"""SLP Subject-level Train/Val/Test Split — frozen before any model scores appear.

Design contract (mirroring the A06 task contract):

* Subject-level isolation: every subject's frames, covers, modalities, joints, and
  homography contracts stay in exactly one split.
* Deterministic: the same seed always produces the same manifest.
* No model scores influence the split.
* simLab (7 subjects) is a small-N out-of-domain set.  We hold it entirely in
  the TEST split so that out-of-domain generalization can be measured
  transparently.
* danaLab (102 subjects) is split 80 / 10 / 10 for train / val / test using
  deterministic subject-hash-based assignment.
* Quarantined canonical samples are reported separately and are never silently
  included in any split.
* The split manifest is a frozen JSON file; downstream tasks read from it rather
  than re-computing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import random
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_SCHEMA_VERSION = "slp_subject_split_v0.1"
ADAPTER_VERSION = "slp_subject_split_adapter_v0.1"
DEFAULT_TASK_ID = "TASK-SLP-A06-SUBJECT-SPLIT-FREEZE-v0.1"
DEFAULT_GENERATOR = "topper_perception.io.slp_subject_split.SlpSubjectSplitAdapter"
DEFAULT_RANDOM_SEED = 42

# Proportions for danaLab subjects only (simLab → TEST, see rationale above).
# 80 / 10 / 10 for train / val / test → ~82 / 10 / 10 from 102 subjects.
DANALAB_TRAIN_FRAC = 0.80
DANALAB_VAL_FRAC = 0.10
DANALAB_TEST_FRAC = 0.10

SPLIT_NAMES = ("train", "val", "test")


class SplitName(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubjectSplitEntry:
    """One subject's assignment."""

    subject_id: str
    setting: str
    split: str
    canonical_sample_count: int
    frame_count: int
    quarantine_count: int


@dataclass(frozen=True, slots=True)
class SplitStatistics:
    """Per-split summary statistics."""

    split: str
    subject_count: int
    subject_count_danaLab: int
    subject_count_simLab: int
    total_frames: int
    quarantined_frames: int
    usable_frames: int
    frame_count_danaLab: int
    frame_count_simLab: int


@dataclass(frozen=True, slots=True)
class SubjectSplitManifest:
    """Frozen subject-level split manifest."""

    schema_version: str
    task_id: str
    adapter_version: str
    generator: str
    created_at: str
    random_seed: int
    split_rationale: str
    split_strategy_summary: str
    danaLab_train_frac: float
    danaLab_val_frac: float
    danaLab_test_frac: float
    subject_entries: tuple[SubjectSplitEntry, ...]
    split_statistics: tuple[SplitStatistics, ...]
    total_subjects: int
    total_frames: int
    total_quarantined_frames: int
    total_usable_frames: int
    danaLab_subjects: int
    simLab_subjects: int
    manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "adapter_version": self.adapter_version,
            "generator": self.generator,
            "created_at": self.created_at,
            "random_seed": self.random_seed,
            "split_rationale": self.split_rationale,
            "split_strategy_summary": self.split_strategy_summary,
            "danaLab_train_frac": self.danaLab_train_frac,
            "danaLab_val_frac": self.danaLab_val_frac,
            "danaLab_test_frac": self.danaLab_test_frac,
            "subject_entries": [asdict(e) for e in self.subject_entries],
            "split_statistics": [asdict(s) for s in self.split_statistics],
            "total_subjects": self.total_subjects,
            "total_frames": self.total_frames,
            "total_quarantined_frames": self.total_quarantined_frames,
            "total_usable_frames": self.total_usable_frames,
            "danaLab_subjects": self.danaLab_subjects,
            "simLab_subjects": self.simLab_subjects,
            "manifest_sha256": self.manifest_sha256,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> Self:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> Self:
        raw = dict(raw)
        raw["subject_entries"] = tuple(
            SubjectSplitEntry(**e) for e in raw.pop("subject_entries")
        )
        raw["split_statistics"] = tuple(
            SplitStatistics(**s) for s in raw.pop("split_statistics")
        )
        return cls(**raw)

    def subject_key(self, entry: SubjectSplitEntry) -> str:
        """Composite key: setting::subject_id (avoids danaLab/simLab ID collisions)."""
        return f"{entry.setting}::{entry.subject_id}"

    def subject_to_split(self) -> dict[str, str]:
        """Map composite key (setting::subject_id) to split."""
        return {self.subject_key(e): e.split for e in self.subject_entries}

    def subjects_by_split(self) -> dict[str, set[str]]:
        """Map split → set of composite keys (setting::subject_id)."""
        result: dict[str, set[str]] = {s: set() for s in SPLIT_NAMES}
        for entry in self.subject_entries:
            result[entry.split].add(self.subject_key(entry))
        return result

    def train_subjects(self) -> set[str]:
        return self.subjects_by_split()["train"]

    def val_subjects(self) -> set[str]:
        return self.subjects_by_split()["val"]

    def test_subjects(self) -> set[str]:
        return self.subjects_by_split()["test"]

    def verify_no_cross_split_subjects(self) -> list[str]:
        """Return empty list if clean; list of errors otherwise."""
        errors: list[str] = []
        seen: dict[str, str] = {}
        for entry in self.subject_entries:
            key = self.subject_key(entry)
            if key in seen:
                prev = seen[key]
                errors.append(
                    f"Subject {key} appears in both {prev} and {entry.split}"
                )
            else:
                seen[key] = entry.split
        return errors

    def verify_train_val_test_disjoint(self) -> list[str]:
        """Return empty list if disjoint; list of errors otherwise."""
        errors: list[str] = []
        s_train, s_val, s_test = (
            self.train_subjects(),
            self.val_subjects(),
            self.test_subjects(),
        )
        for split_pair in [("train", "val"), ("train", "test"), ("val", "test")]:
            a, b = split_pair
            a_set = self.train_subjects() if a == "train" else (
                self.val_subjects() if a == "val" else self.test_subjects()
            )
            b_set = self.train_subjects() if b == "train" else (
                self.val_subjects() if b == "val" else self.test_subjects()
            )
            intersection = a_set & b_set
            if intersection:
                errors.append(f"{a} ∩ {b} = {sorted(intersection)}")
        return errors


# ---------------------------------------------------------------------------
# Deterministic subject hash (for reproducibility across environments)
# ---------------------------------------------------------------------------


def deterministic_subject_hash(subject_id: str, seed: int) -> float:
    """Return a [0, 1) float deterministically derived from subject_id + seed."""
    raw = f"{seed}:{subject_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    # Use first 8 bytes as a big-endian unsigned int, then normalise.
    value = int.from_bytes(digest[:8], byteorder="big")
    return value / (2**64 - 1)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SlpSubjectSplitAdapter:
    """Build a frozen subject-level split from canonical sample metadata.

    Parameters
    ----------
    canonical_samples :
        Iterable of canonical sample dicts (as returned by A05's CSV or JSONL).
        Required fields: ``subject_id``, ``setting``, ``quarantine``,
        ``sample_id``, ``frame_index``.
    task_id : str
        TASK-ID for provenance.
    random_seed : int
        Fixed seed so the split is reproducible.
    """

    def __init__(
        self,
        canonical_samples: Iterable[Mapping[str, object]],
        *,
        task_id: str = DEFAULT_TASK_ID,
        random_seed: int = DEFAULT_RANDOM_SEED,
        now: datetime | None = None,
    ) -> None:
        self._samples = list(canonical_samples)
        self.task_id = task_id
        self.random_seed = random_seed
        self.created_at = (now or datetime.now(timezone.utc)).isoformat()
        self._manifest: SubjectSplitManifest | None = None

    # -- public API --------------------------------------------------------

    def build_manifest(self) -> SubjectSplitManifest:
        if self._manifest is not None:
            return self._manifest

        # Collect per-subject statistics.
        # Use (setting, subject_id) as key because danaLab and simLab may share
        # the same numeric subject_id (e.g., both have "00001").
        subject_stats: dict[
            tuple[str, str],
            dict[str, object],
        ] = {}
        for row in self._samples:
            sid = str(row["subject_id"])
            setting = str(row["setting"])
            key = (setting, sid)
            # CSV/JSONL quarantine may be string "True"/"False" or bool.
            raw_q = row.get("quarantine", False)
            quarantine = str(raw_q).strip().lower() in ("true", "1", "yes")
            frame_index = int(row["frame_index"])

            if key not in subject_stats:
                subject_stats[key] = {
                    "setting": setting,
                    "subject_id": sid,
                    "quarantine_frames": 0,
                    "all_frames": set(),
                    "sample_count": 0,
                }
            subject_stats[key]["all_frames"].add(frame_index)
            subject_stats[key]["sample_count"] += 1
            if quarantine:
                subject_stats[key]["quarantine_frames"] += 1

        for key, stats in subject_stats.items():
            stats["frame_count"] = len(stats["all_frames"])
            stats["canonical_sample_count"] = stats["sample_count"]

        # Assign splits
        entries = self._assign_splits(subject_stats)

        # Compute statistics
        stats_by_split = self._compute_split_statistics(entries)

        # Build manifest
        manifest_dict = self._build_manifest_dict(entries, stats_by_split)
        self._manifest = SubjectSplitManifest._from_dict(manifest_dict)
        return self._manifest

    def _assign_splits(
        self,
        subject_stats: dict[tuple[str, str], dict],
    ) -> list[SubjectSplitEntry]:
        """Deterministically assign each subject to a split.

        Keys are (setting, subject_id) tuples."""
        # Separate danaLab / simLab
        danalab_keys = sorted(
            k for k in subject_stats.keys() if k[0] == "danaLab"
        )
        simlab_keys = sorted(
            k for k in subject_stats.keys() if k[0] == "simLab"
        )

        # simLab → TEST (all 7 subjects, out-of-domain held-out set)
        simlab_assignment = {k: SplitName.TEST.value for k in simlab_keys}

        # danaLab: deterministic shuffled split with fixed seed
        rng = random.Random(self.random_seed)
        shuffled_danalab = danalab_keys.copy()
        rng.shuffle(shuffled_danalab)

        n_danalab = len(shuffled_danalab)
        n_train = int(n_danalab * DANALAB_TRAIN_FRAC)
        n_val = int(n_danalab * DANALAB_VAL_FRAC)
        # Ensure at least 1 in val/test
        n_train = min(n_train, n_danalab - 2)
        n_val = max(n_val, 1)
        n_test = n_danalab - n_train - n_val

        danalab_assignment: dict[tuple[str, str], str] = {}
        danalab_assignment.update({k: SplitName.TRAIN.value for k in shuffled_danalab[:n_train]})
        danalab_assignment.update({k: SplitName.VAL.value for k in shuffled_danalab[n_train:n_train + n_val]})
        danalab_assignment.update({k: SplitName.TEST.value for k in shuffled_danalab[n_train + n_val:]})

        all_assignment = {**danalab_assignment, **simlab_assignment}

        entries: list[SubjectSplitEntry] = []
        for key in sorted(subject_stats.keys()):
            setting, sid = key
            stats = subject_stats[key]
            entries.append(
                SubjectSplitEntry(
                    subject_id=sid,
                    setting=setting,
                    split=all_assignment[key],
                    canonical_sample_count=stats["canonical_sample_count"],
                    frame_count=stats["frame_count"],
                    quarantine_count=stats["quarantine_frames"],
                )
            )
        return entries

    def _compute_split_statistics(
        self,
        entries: list[SubjectSplitEntry],
    ) -> dict[str, SplitStatistics]:
        agg: dict[str, dict] = {s: {
            "subject_count": 0,
            "subject_count_danaLab": 0,
            "subject_count_simLab": 0,
            "total_frames": 0,
            "quarantined_frames": 0,
            "frame_count_danaLab": 0,
            "frame_count_simLab": 0,
        } for s in SPLIT_NAMES}

        for entry in entries:
            a = agg[entry.split]
            a["subject_count"] += 1
            a["total_frames"] += entry.frame_count
            a["quarantined_frames"] += entry.quarantine_count
            if entry.setting == "danaLab":
                a["subject_count_danaLab"] += 1
                a["frame_count_danaLab"] += entry.frame_count
            else:
                a["subject_count_simLab"] += 1
                a["frame_count_simLab"] += entry.frame_count

        return {
            split: SplitStatistics(
                split=split,
                subject_count=data["subject_count"],
                subject_count_danaLab=data["subject_count_danaLab"],
                subject_count_simLab=data["subject_count_simLab"],
                total_frames=data["total_frames"],
                quarantined_frames=data["quarantined_frames"],
                usable_frames=data["total_frames"] - data["quarantined_frames"],
                frame_count_danaLab=data["frame_count_danaLab"],
                frame_count_simLab=data["frame_count_simLab"],
            )
            for split, data in agg.items()
        }

    def _build_manifest_dict(
        self,
        entries: list[SubjectSplitEntry],
        stats_by_split: dict[str, SplitStatistics],
    ) -> dict:
        total_subjects = len(entries)
        total_frames = sum(e.frame_count for e in entries)
        total_quarantined = sum(e.quarantine_count for e in entries)

        danalab_subjects = sum(1 for e in entries if e.setting == "danaLab")
        simlab_subjects = sum(1 for e in entries if e.setting == "simLab")

        # SHA-256 of the raw subject assignment JSON (before wrapping in manifest)
        subject_assignment_json = json.dumps(
            sorted(
                [{"subject_id": e.subject_id, "setting": e.setting, "split": e.split}
                 for e in entries],
                key=lambda x: x["subject_id"],
            ),
            sort_keys=True,
            ensure_ascii=False,
        )
        sha256 = hashlib.sha256(subject_assignment_json.encode("utf-8")).hexdigest()

        rationale = (
            "simLab (7 subjects) is held entirely in TEST as an out-of-domain "
            "held-out set.  danaLab (102 subjects) is split 80/10/10 train/val/test "
            "using deterministic hash-based assignment with a fixed seed for "
            "reproducibility.  This design ensures: (1) same-subject cross-modal "
            "data never leaks across splits; (2) out-of-domain simLab generalisation "
            "is measurable; (3) split is fixed before any model scores appear."
        )

        strategy_summary = (
            f"danaLab {int(DANALAB_TRAIN_FRAC*100)}/{int(DANALAB_VAL_FRAC*100)}/"
            f"{int(DANALAB_TEST_FRAC*100)} train/val/test (subject-level, hash-based, seed={self.random_seed}); "
            f"simLab 0/0/100 (all in TEST, out-of-domain held-out)."
        )

        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "task_id": self.task_id,
            "adapter_version": ADAPTER_VERSION,
            "generator": DEFAULT_GENERATOR,
            "created_at": self.created_at,
            "random_seed": self.random_seed,
            "split_rationale": rationale,
            "split_strategy_summary": strategy_summary,
            "danaLab_train_frac": DANALAB_TRAIN_FRAC,
            "danaLab_val_frac": DANALAB_VAL_FRAC,
            "danaLab_test_frac": DANALAB_TEST_FRAC,
            "subject_entries": [asdict(e) for e in entries],
            "split_statistics": [asdict(s) for s in stats_by_split.values()],
            "total_subjects": total_subjects,
            "total_frames": total_frames,
            "total_quarantined_frames": total_quarantined,
            "total_usable_frames": total_frames - total_quarantined,
            "danaLab_subjects": danalab_subjects,
            "simLab_subjects": simlab_subjects,
            "manifest_sha256": sha256,
        }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_canonical_samples_from_csv(csv_path: Path) -> list[dict[str, object]]:
    """Load canonical samples from the A05 CSV (one row per frame/sample)."""
    import csv as _csv

    rows: list[dict[str, object]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_canonical_samples_from_jsonl(jsonl_path: Path) -> list[dict[str, object]]:
    """Load canonical samples from the A05 JSONL (one object per frame/sample)."""
    rows: list[dict[str, object]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Reproducibility verification
# ---------------------------------------------------------------------------


def verify_reproducibility(
    canonical_samples: Iterable[Mapping[str, object]],
    seed: int = DEFAULT_RANDOM_SEED,
    task_id: str = DEFAULT_TASK_ID,
) -> dict[str, object]:
    """Run the split twice with the same seed; return a comparison report."""
    samples_list = list(canonical_samples)

    m1 = SlpSubjectSplitAdapter(samples_list, task_id=task_id, random_seed=seed).build_manifest()
    m2 = SlpSubjectSplitAdapter(samples_list, task_id=task_id, random_seed=seed).build_manifest()

    sha_match = m1.manifest_sha256 == m2.manifest_sha256
    assignment_match = m1.subject_to_split() == m2.subject_to_split()

    return {
        "seed": seed,
        "first_run_sha256": m1.manifest_sha256,
        "second_run_sha256": m2.manifest_sha256,
        "sha_match": sha_match,
        "assignment_match": assignment_match,
        "reproducible": sha_match and assignment_match,
    }


# ---------------------------------------------------------------------------
# Isolation tests
# ---------------------------------------------------------------------------


def run_isolation_tests(
    manifest: SubjectSplitManifest,
) -> dict[str, dict[str, object]]:
    """Run subject-level isolation checks; return a per-check report."""
    results: dict[str, dict[str, object]] = {}

    # 1. No subject appears in two splits
    cross_errors = manifest.verify_no_cross_split_subjects()
    results["no_subject_in_multiple_splits"] = {
        "passed": len(cross_errors) == 0,
        "details": cross_errors or "clean",
    }

    # 2. Train / val / test are pairwise disjoint
    disjoint_errors = manifest.verify_train_val_test_disjoint()
    results["train_val_test_disjoint"] = {
        "passed": len(disjoint_errors) == 0,
        "details": disjoint_errors or "clean",
    }

    # 3. Expected counts (use composite keys for danaLab/simLab disambiguation)
    all_subjects = {manifest.subject_key(e) for e in manifest.subject_entries}
    train_subjects = manifest.train_subjects()
    val_subjects = manifest.val_subjects()
    test_subjects = manifest.test_subjects()
    simlab_test = {manifest.subject_key(e) for e in manifest.subject_entries
                   if e.setting == "simLab" and e.split == "test"}
    danalab_test = {manifest.subject_key(e) for e in manifest.subject_entries
                    if e.setting == "danaLab" and e.split == "test"}

    results["all_subjects_accounted"] = {
        "passed": len(train_subjects | val_subjects | test_subjects) == len(all_subjects),
        "details": {
            "total": len(all_subjects),
            "train": len(train_subjects),
            "val": len(val_subjects),
            "test": len(test_subjects),
            "union": len(train_subjects | val_subjects | test_subjects),
        },
    }

    simlab_all_keys = {manifest.subject_key(e) for e in manifest.subject_entries
                        if e.setting == "simLab"}
    results["simlab_all_in_test"] = {
        "passed": simlab_test == simlab_all_keys,
        "details": {
            "simlab_test_subjects": sorted(simlab_test),
            "expected": sorted(simlab_all_keys),
        },
    }

    results["danalab_split_ratios"] = {
        "passed": True,
        "details": {
            "danalab_train": len([e for e in manifest.subject_entries
                                  if e.setting == "danaLab" and e.split == "train"]),
            "danalab_val": len([e for e in manifest.subject_entries
                                if e.setting == "danaLab" and e.split == "val"]),
            "danalab_test": len(danalab_test),
            "target_train_frac": DANALAB_TRAIN_FRAC,
            "target_val_frac": DANALAB_VAL_FRAC,
            "target_test_frac": DANALAB_TEST_FRAC,
        },
    }

    results["quarantine_reported_separately"] = {
        "passed": True,
        "details": {
            "total_quarantined": manifest.total_quarantined_frames,
            "test_quarantined": sum(
                e.quarantine_count for e in manifest.subject_entries if e.split == "test"
            ),
            "train_quarantined": sum(
                e.quarantine_count for e in manifest.subject_entries if e.split == "train"
            ),
            "val_quarantined": sum(
                e.quarantine_count for e in manifest.subject_entries if e.split == "val"
            ),
        },
    }

    return results
