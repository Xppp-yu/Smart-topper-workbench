"""Restore real PoPu Tactilus snapshots and export two-dimensional heatmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topper_perception.healthcheck import load_path_config
from topper_perception.io.popu import POPU_POSTURES, select_tactilus_frame
from topper_perception.visualization import (
    render_posture_overview,
    render_pressure_heatmap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if args.overview:
        filename = f"popu_subject_{int(args.subject):03d}_posture_overview.png"
    else:
        filename = (
            f"popu_subject_{int(args.subject):03d}_{args.posture}_"
            f"v{args.variation}_record_{args.record_index:02d}_frame_{args.frame_index:03d}.png"
        )
    return PROJECT_ROOT / "outputs" / "figures" / filename


def _write_metadata(output_path: Path, payload: object) -> Path:
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paths.local.json",
    )
    parser.add_argument("--subject", default="1")
    parser.add_argument("--posture", choices=POPU_POSTURES, default="left")
    parser.add_argument("--variation", default="1")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_path_config(args.config)
    if "popu_data" not in paths:
        raise KeyError(f"popu_data is missing from path config: {args.config}")
    popu_root = paths["popu_data"]
    output_path = _output_path(args)

    if args.overview:
        frames = [
            select_tactilus_frame(
                popu_root,
                subject_id=args.subject,
                posture=posture,
                variation=args.variation,
                record_index=args.record_index,
                frame_index=args.frame_index,
            )
            for posture in POPU_POSTURES
        ]
        render_posture_overview(frames, output_path)
        metadata_path = _write_metadata(
            output_path,
            {"figure": str(output_path), "frames": [frame.metadata() for frame in frames]},
        )
        print(f"Rendered overview: {output_path}")
    else:
        frame = select_tactilus_frame(
            popu_root,
            subject_id=args.subject,
            posture=args.posture,
            variation=args.variation,
            record_index=args.record_index,
            frame_index=args.frame_index,
        )
        render_pressure_heatmap(frame, output_path)
        metadata_path = _write_metadata(
            output_path,
            {"figure": str(output_path), **frame.metadata()},
        )
        print(json.dumps(frame.metadata(), ensure_ascii=False, indent=2))
        print(f"Rendered heatmap: {output_path}")

    print(f"Wrote metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

