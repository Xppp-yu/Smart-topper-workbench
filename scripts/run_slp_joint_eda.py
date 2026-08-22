"""Run SLP Joint Occlusion and Quality EDA on real SLP data.

This script:
1. Loads A05 canonical samples and A06 split manifest.
2. Runs J0 (original) and J1 (homography-derived) joint EDA.
3. Produces: JSON summary, per-joint QA CSV, bone segment CSV,
   anomaly CSV, group-level stats CSV, and small visualizations.

Design contract:
- J0 and J1 are reported separately. J1 is never mixed into J0 GT statistics.
- Usable and quarantined samples are reported separately.
- danaLab and simLab are always reported in separate buckets.
- The A06 frozen split is used as-is; this script does not modify it.
- No region ground truth is generated.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.io.slp_joint_eda import (
    ADAPTER_VERSION,
    DEFAULT_TASK_ID,
    JointEdaResult,
    build_group_summaries,
    run_j0_eda,
    run_j1_eda_from_csv,
    result_to_dict,
    write_anomaly_csv,
    write_bone_segment_csv,
    write_group_stats_csv,
    write_joint_qa_csv,
    write_joint_scatter_plot,
    write_occlusion_heatmap,
)
from topper_perception.io.slp_subject_split import load_canonical_samples_from_csv


PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]

# Default paths (relative to project root).
DEFAULT_A05_CSV = PROJECT_ROOT_PATH / "data/processed/slp/slp_canonical_samples_v0.1.csv"
DEFAULT_A06_JSON = PROJECT_ROOT_PATH / "data/processed/slp/slp_subject_split_v0.1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT_PATH / "outputs/reports/slp_joint_eda_v0.1"
DEFAULT_CSV_DIR = PROJECT_ROOT_PATH / "outputs/analysis/slp_joint_eda_v0.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slp-root",
        type=Path,
        required=True,
        help="SLP data root (e.g. E:/TeamProjects/datasets/smart-topper/SLP2022/SLP).",
    )
    parser.add_argument(
        "--a05-csv",
        type=Path,
        default=DEFAULT_A05_CSV,
        help="A05 canonical samples CSV.",
    )
    parser.add_argument(
        "--a06-json",
        type=Path,
        default=DEFAULT_A06_JSON,
        help="A06 subject split manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON summary output.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="Directory for CSV QA outputs.",
    )
    parser.add_argument(
        "--jump-threshold-px",
        type=float,
        default=100.0,
        help="Frame-to-frame joint displacement threshold for anomaly detection (px).",
    )
    parser.add_argument(
        "--bone-zscore-threshold",
        type=float,
        default=4.0,
        help="Z-score threshold for anomalous bone length detection.",
    )
    parser.add_argument(
        "--skip-j1",
        action="store_true",
        help="Skip J1 (homography-derived joint) EDA.",
    )
    parser.add_argument(
        "--task-id",
        default=DEFAULT_TASK_ID,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # 1. Load A05 canonical samples.
    if not args.a05_csv.is_file():
        print(
            f"error: A05 canonical samples CSV not found at {args.a05_csv}\n"
            "Run scripts/build_slp_canonical_samples.py first.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading A05 canonical samples from {args.a05_csv} ...")
    canonical_samples = load_canonical_samples_from_csv(args.a05_csv)
    print(f"  Loaded {len(canonical_samples)} canonical samples.")

    # Count quarantined.
    q_count = sum(
        1 for s in canonical_samples
        if str(s.get("quarantine", "False")).strip().lower() in ("true", "1", "yes")
    )
    print(f"  Quarantined: {q_count}")

    # 2. Load A06 split manifest (optional).
    split_manifest: dict | None = None
    if args.a06_json.is_file():
        print(f"Loading A06 split manifest from {args.a06_json} ...")
        split_manifest = json.loads(args.a06_json.read_text(encoding="utf-8"))
        print(f"  Split manifest loaded: {len(split_manifest.get('subject_entries', []))} entries.")
    else:
        print(f"A06 split manifest not found at {args.a06_json}; split grouping skipped.")

    # 3. Resolve SLP root.
    slp_root = args.slp_root.resolve()
    if not slp_root.exists():
        print(f"error: SLP root does not exist: {slp_root}", file=sys.stderr)
        return 2

    # 4. Run J0 EDA.
    print("\nRunning J0 (original joint) EDA ...")
    j0_result = run_j0_eda(
        canonical_samples,
        slp_root,
        task_id=args.task_id,
        jump_threshold_px=args.jump_threshold_px,
        bone_zscore_threshold=args.bone_zscore_threshold,
    )
    print(
        f"  J0: {j0_result.usable_frames} usable frames, "
        f"{j0_result.quarantined_frames} quarantined, "
        f"{len(j0_result.anomalies)} anomalies detected."
    )

    # 5. Run J1 EDA (if not skipped).
    j1_result: JointEdaResult | None = None
    if not args.skip_j1:
        print("\nRunning J1 (homography-derived joint) EDA ...")
        j1_result = run_j1_eda_from_csv(
            canonical_samples,
            slp_root,
            task_id=args.task_id,
            jump_threshold_px=args.jump_threshold_px,
            bone_zscore_threshold=args.bone_zscore_threshold,
        )
        if j1_result:
            print(
                f"  J1: {j1_result.usable_frames} usable frames, "
                f"{j1_result.quarantined_frames} quarantined, "
                f"{len(j1_result.anomalies)} anomalies detected."
            )
        else:
            print("  J1: No J1 joints could be generated (homography matrices missing or direction unresolved).")

    # 6. Build group summaries.
    print("\nBuilding group-level summaries ...")
    group_summaries = build_group_summaries(
        canonical_samples,
        j0_result,
        slp_root,
        split_manifest,
    )
    print(f"  {len(group_summaries)} groups computed.")

    # 7. Write outputs.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary.
    summary: dict[str, object] = {
        "task_id": args.task_id,
        "adapter_version": ADAPTER_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slp_root": str(slp_root),
        "a05_csv": str(args.a05_csv),
        "a06_json": str(args.a06_json),
        "jump_threshold_px": args.jump_threshold_px,
        "bone_zscore_threshold": args.bone_zscore_threshold,
        "canonical_sample_count": len(canonical_samples),
        "quarantined_count": q_count,
        "j0": result_to_dict(j0_result),
        "j1": result_to_dict(j1_result) if j1_result else None,
        "group_summaries": group_summaries,
        "joint_names": [
            "head_cervical", "neck_c7",
            "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder", "left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle",
            "left_hip", "left_knee", "left_ankle",
        ],
        "bone_segments": [
            {"start": a, "end": b, "start_name": na, "end_name": nb}
            for (a, b), na, nb in zip(
                [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (1, 5), (5, 6), (6, 7),
                    (2, 8), (8, 9), (9, 10),
                    (5, 11), (11, 12), (12, 13),
                ],
                [
                    "head_cervical", "neck_c7",
                    "right_shoulder", "right_elbow", "right_wrist",
                    "left_shoulder", "left_elbow", "left_wrist",
                    "right_hip", "right_knee", "right_ankle",
                    "left_hip", "left_knee", "left_ankle",
                ],
                [
                    "neck_c7",
                    "right_shoulder", "right_elbow", "right_wrist",
                    "left_shoulder", "left_elbow", "left_wrist",
                    "right_hip", "right_knee", "right_ankle",
                    "left_hip", "left_knee", "left_ankle",
                    "left_ankle",
                ],
            )
        ],
    }

    summary_path = args.output_dir / "slp_joint_eda_summary_v0.1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSummary written to {summary_path}")

    # Per-joint QA CSV.
    j0_qa_path = args.csv_dir / "slp_joint_qa_j0_v0.1.csv"
    write_joint_qa_csv(j0_result, j0_qa_path)
    print(f"J0 per-joint QA CSV written to {j0_qa_path}")

    if j1_result:
        j1_qa_path = args.csv_dir / "slp_joint_qa_j1_v0.1.csv"
        write_joint_qa_csv(j1_result, j1_qa_path)
        print(f"J1 per-joint QA CSV written to {j1_qa_path}")

    # Bone segment CSV.
    results_for_bone = [j0_result]
    if j1_result:
        results_for_bone.append(j1_result)
    bone_csv_path = args.csv_dir / "slp_bone_segment_stats_v0.1.csv"
    write_bone_segment_csv(results_for_bone, bone_csv_path)
    print(f"Bone segment stats CSV written to {bone_csv_path}")

    # Anomaly CSV.
    all_anomalies = j0_result.anomalies.copy()
    if j1_result:
        all_anomalies.extend(j1_result.anomalies)
    if all_anomalies:
        anomaly_csv_path = args.csv_dir / "slp_joint_anomalies_v0.1.csv"
        write_anomaly_csv(all_anomalies, anomaly_csv_path)
        print(f"Anomaly cases CSV written to {anomaly_csv_path}")

    # Group-level stats CSV.
    group_csv_path = args.csv_dir / "slp_joint_group_stats_v0.1.csv"
    write_group_stats_csv(group_summaries, group_csv_path)
    print(f"Group-level stats CSV written to {group_csv_path}")

    # Small visualizations.
    scatter_path = args.output_dir / "slp_joint_scatter_j0_v0.1.png"
    write_joint_scatter_plot(j0_result, scatter_path, title="J0 Joint Coordinates (RGB pixel space)")
    print(f"J0 scatter plot written to {scatter_path}")

    heatmap_path = args.output_dir / "slp_joint_occlusion_heatmap_j0_v0.1.png"
    write_occlusion_heatmap(j0_result.per_joint, heatmap_path, title="J0 Per-Joint Occlusion Rate")
    print(f"J0 occlusion heatmap written to {heatmap_path}")

    if j1_result:
        scatter_j1_path = args.output_dir / "slp_joint_scatter_j1_v0.1.png"
        write_joint_scatter_plot(j1_result, scatter_j1_path, title="J1 Joint Coordinates (PM space)")
        print(f"J1 scatter plot written to {scatter_j1_path}")

        heatmap_j1_path = args.output_dir / "slp_joint_occlusion_heatmap_j1_v0.1.png"
        write_occlusion_heatmap(j1_result.per_joint, heatmap_j1_path, title="J1 Per-Joint Occlusion Rate")
        print(f"J1 occlusion heatmap written to {heatmap_j1_path}")

    print("\n" + "=" * 60)
    print("EDA Summary")
    print("=" * 60)
    print(f"Canonical samples: {len(canonical_samples)}")
    print(f"  Usable: {len(canonical_samples) - q_count}")
    print(f"  Quarantined: {q_count}")
    print(f"J0 usable frames: {j0_result.usable_frames}")
    print(f"J0 anomalies: {len(j0_result.anomalies)}")
    print(f"J1 usable frames: {j1_result.usable_frames if j1_result else 0}")
    print(f"J1 anomalies: {len(j1_result.anomalies) if j1_result else 0}")
    print(f"Groups: {len(group_summaries)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
