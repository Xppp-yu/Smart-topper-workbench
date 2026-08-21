"""Render deterministic SLP homography overlay samples for manual review.

This script intentionally creates inspection artifacts only. It does not decide
homography direction automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from topper_perception.io.slp_homography_audit import select_fixed_danalab_subjects
from topper_perception.io.slp_inventory import resolve_slp_root



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/A04_overlay_samples"))
    parser.add_argument("--count", type=int, default=6)
    return parser.parse_args()



def draw_points(image: np.ndarray, points: np.ndarray, label: str) -> np.ndarray:
    canvas = image.copy()
    for point in points:
        x, y = np.round(point).astype(int)
        cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
    cv2.putText(canvas, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return canvas



def main() -> None:
    args = parse_args()
    slp_root = resolve_slp_root(args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)

    subjects = select_fixed_danalab_subjects(slp_root, count=args.count)
    if not subjects:
        raise SystemExit("no danaLab subjects found")

    for subject in subjects:
        rgb = subject / "RGB" / "uncover" / "image_000001.png"
        if not rgb.exists():
            continue
        image = cv2.imread(str(rgb))
        if image is None:
            continue
        result = draw_points(image, np.empty((0, 2)), "A04 overlay placeholder")
        cv2.imwrite(str(args.output / f"{subject.name}_rgb_overlay.png"), result)

    print(f"written={args.output}")


if __name__ == "__main__":
    main()
