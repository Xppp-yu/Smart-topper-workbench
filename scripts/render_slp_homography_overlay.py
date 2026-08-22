"""Render fixed SLP RGB / IR / Depth / PM overlays for manual direction review.

The script produces visual artefacts only. It does not pick a semantic
direction. Each overlay panel records the assumption it was rendered under
(``H maps source -> PM``), which is the contract documented in the SLP
README and is the only contract consistent with the A04 audit evidence:

* 109/109 danaLab RGB ``align_PTr_RGB.npy`` matrices land J0 RGB joints
  inside the PM grid 99.3 % of the time on average (direct), and 0 % of the
  time when applied to RGB joints as an inverse mapping.
* 102/102 danaLab IR matrices behave analogously (direct 99.3 %, inverse 0 %).
* All 327 matrices pass the invertibility / round-trip math tests.

The overlays therefore test the visual counterpart of the audit evidence.
If a future review finds that the rendered overlay disagrees with the
assumed direction, the audit code path must be revisited; the overlay
script itself does not change the dataset contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from topper_perception.geometry.slp_homography import (
    apply_homography,
    in_bounds_mask,
)
from topper_perception.io.slp_homography_audit import (
    load_joint_xy,
    read_png_dimensions,
    select_fixed_danalab_subjects,
)
from topper_perception.io.slp_inventory import resolve_slp_root


JOINT_NAMES = (
    "Right ankle",
    "Right knee",
    "Right hip",
    "Left hip",
    "Left knee",
    "Left ankle",
    "Right wrist",
    "Right elbow",
    "Right shoulder",
    "Left shoulder",
    "Left elbow",
    "Left wrist",
    "Thorax",
    "Head top",
)
SKELETON_EDGES = (
    (0, 1), (1, 2), (2, 12), (3, 4), (4, 5), (12, 3),
    (6, 7), (7, 8), (8, 12), (9, 12), (9, 10), (10, 11),
    (12, 13),
)

SOURCE_COLOR_RGB = (255, 64, 64)      # blue tint for J0 RGB
SOURCE_COLOR_IR = (64, 200, 255)        # yellow tint for J0 IR
PROJECTED_COLOR_RGB = (0, 0, 255)     # pure red for H_RGB(RGB_joints)
PROJECTED_COLOR_IR = (0, 255, 255)    # pure yellow for H_IR(IR_joints)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SLP homography overlays.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subject-count", type=int, default=6)
    parser.add_argument("--frame-index", type=int, default=1)
    parser.add_argument("--cover", type=str, default="uncover")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/figures/A04_overlay_samples")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/reports/slp_homography_overlay_manifest_v0.1.json"),
    )
    return parser.parse_args()


def load_image_or_none(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return image if image is not None else None


def draw_joints(
    canvas: np.ndarray,
    points: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int = 4,
    edges: bool = True,
) -> None:
    if points.size == 0:
        return
    finite = np.isfinite(points).all(axis=1)
    for i, (x, y) in enumerate(np.round(points).astype(int)):
        if not finite[i]:
            continue
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            cv2.circle(canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    if edges:
        for a, b in SKELETON_EDGES:
            if not (finite[a] and finite[b]):
                continue
            ax, ay = np.round(points[a]).astype(int)
            bx, by = np.round(points[b]).astype(int)
            if (
                0 <= ax < canvas.shape[1]
                and 0 <= ay < canvas.shape[0]
                and 0 <= bx < canvas.shape[1]
                and 0 <= by < canvas.shape[0]
            ):
                cv2.line(canvas, (ax, ay), (bx, by), color, 1, lineType=cv2.LINE_AA)


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        label,
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    return canvas


def composite_panels(
    panels: dict[str, np.ndarray],
    target_size: tuple[int, int],
) -> np.ndarray:
    resized: list[np.ndarray] = []
    for panel in panels.values():
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        resized.append(
            cv2.resize(panel, target_size, interpolation=cv2.INTER_AREA)
        )
    top = np.concatenate(resized[:2], axis=1)
    bottom = np.concatenate(resized[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def overlay_for_subject(
    subject_dir: Path,
    *,
    frame_index: int,
    cover_condition: str,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "subject_id": subject_dir.name,
        "frame_index": frame_index,
        "cover_condition": cover_condition,
        "modalities": {},
        "homography_projection_stats": {},
        "errors": [],
    }

    panels: dict[str, np.ndarray] = {}
    joint_data: dict[str, np.ndarray] = {}
    homographies: dict[str, np.ndarray] = {}

    image_specs = (
        ("RGB", subject_dir / "RGB" / cover_condition / f"image_{frame_index:06d}.png"),
        ("IR", subject_dir / "IR" / cover_condition / f"image_{frame_index:06d}.png"),
        (
            "depth",
            subject_dir / "depth" / cover_condition / f"image_{frame_index:06d}.png",
        ),
        ("PM", subject_dir / "PM" / cover_condition / f"image_{frame_index:06d}.png"),
    )

    for modality, path in image_specs:
        if not path.is_file():
            summary["errors"].append(f"missing_{modality}_image")
            continue
        image = load_image_or_none(path)
        if image is None:
            summary["errors"].append(f"unreadable_{modality}_image")
            continue
        panels[modality] = image
        try:
            w, h = read_png_dimensions(path)
            summary["modalities"][modality] = {
                "uri": path.name,
                "width": int(w),
                "height": int(h),
            }
        except ValueError as exc:
            summary["errors"].append(str(exc))

    pm_dims = summary["modalities"].get("PM")
    if pm_dims is None:
        summary["errors"].append("missing_pm_reference_dimensions")
        return summary

    pm_width = int(pm_dims["width"])
    pm_height = int(pm_dims["height"])

    for modality in ("RGB", "IR"):
        joints_path = subject_dir / f"joints_gt_{modality}.mat"
        if joints_path.is_file():
            try:
                joint_data[modality] = load_joint_xy(subject_dir, modality)[frame_index - 1]
            except ValueError as exc:
                summary["errors"].append(f"invalid_joints_gt_{modality}: {exc}")
        else:
            summary["errors"].append(f"missing_joints_gt_{modality}")

        matrix_path = subject_dir / f"align_PTr_{modality}.npy"
        if matrix_path.is_file():
            homographies[modality] = np.load(matrix_path, allow_pickle=False)

    for modality, joints in joint_data.items():
        if modality not in panels:
            continue
        draw_joints(
            panels[modality],
            joints,
            color=SOURCE_COLOR_RGB if modality == "RGB" else SOURCE_COLOR_IR,
        )
        if modality in homographies:
            projected = apply_homography(joints, homographies[modality])
            in_bounds = bool(
                in_bounds_mask(projected, width=pm_width, height=pm_height).mean()
            )
            summary["homography_projection_stats"][modality] = {
                "joint_count": int(joints.shape[0]),
                "in_bounds_count": int(in_bounds_mask(projected, width=pm_width, height=pm_height).sum()),
                "direct_in_bounds_rate": float(
                    in_bounds_mask(projected, width=pm_width, height=pm_height).mean()
                ),
                "matrix_uri": f"align_PTr_{modality}.npy",
            }
            if "PM" in panels:
                draw_joints(
                    panels["PM"],
                    projected,
                    color=PROJECTED_COLOR_RGB if modality == "RGB" else PROJECTED_COLOR_IR,
                )

    summary["direction_assumption_for_rendering"] = {
        "source_to_pm": "H = align_PTr_<modality>.npy maps source -> PM (per SLP README)",
        "automatic_direction_selection": False,
        "reviewer_required_to_confirm": True,
    }
    summary["joint_count_per_modality"] = {
        modality: int(joints.shape[0])
        for modality, joints in joint_data.items()
    }
    return summary


def main() -> None:
    args = parse_args()
    slp_root = resolve_slp_root(args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    subjects = select_fixed_danalab_subjects(slp_root, count=args.subject_count)
    if not subjects:
        raise SystemExit("no danaLab subjects found")

    manifest_subjects: list[dict[str, object]] = []

    for subject in subjects:
        overlay = overlay_for_subject(
            subject,
            frame_index=args.frame_index,
            cover_condition=args.cover,
        )
        overlay["slp_root_relative_uri"] = subject.relative_to(slp_root).as_posix()
        manifest_subjects.append(overlay)

        if not {"RGB", "IR", "depth", "PM"}.issubset(overlay.get("modalities", {}).keys()):
            continue

        panels = {modality: None for modality in ("RGB", "IR", "depth", "PM")}
        for modality, path in (
            ("RGB", subject / "RGB" / args.cover / f"image_{args.frame_index:06d}.png"),
            ("IR", subject / "IR" / args.cover / f"image_{args.frame_index:06d}.png"),
            (
                "depth",
                subject / "depth" / args.cover / f"image_{args.frame_index:06d}.png",
            ),
            ("PM", subject / "PM" / args.cover / f"image_{args.frame_index:06d}.png"),
        ):
            image = load_image_or_none(path)
            if image is None:
                continue
            panels[modality] = image

        if any(value is None for value in panels.values()):
            continue

        joint_data = {}
        homographies = {}
        for modality in ("RGB", "IR"):
            joints_path = subject / f"joints_gt_{modality}.mat"
            if joints_path.is_file():
                joint_data[modality] = load_joint_xy(subject, modality)[args.frame_index - 1]
            matrix_path = subject / f"align_PTr_{modality}.npy"
            if matrix_path.is_file():
                homographies[modality] = np.load(matrix_path, allow_pickle=False)

        pm_image = panels["PM"]
        if pm_image.ndim == 2:
            pm_image = cv2.cvtColor(pm_image, cv2.COLOR_GRAY2BGR)
        target_size = (pm_image.shape[1], pm_image.shape[0])

        labeled = {
            "RGB": label_panel(panels["RGB"], "RGB  | J0"),
            "IR": label_panel(panels["IR"], "IR   | J0"),
            "depth": label_panel(panels["depth"], "depth (no J0)"),
            "PM": label_panel(pm_image, "PM   | H_RGB, H_IR -> PM (assumed)"),
        }

        draw_joints(
            labeled["RGB"],
            joint_data.get("RGB", np.empty((0, 2))),
            color=SOURCE_COLOR_RGB,
        )
        draw_joints(
            labeled["IR"],
            joint_data.get("IR", np.empty((0, 2))),
            color=SOURCE_COLOR_IR,
        )

        for modality, color in (("RGB", PROJECTED_COLOR_RGB), ("IR", PROJECTED_COLOR_IR)):
            if modality in joint_data and modality in homographies:
                projected = apply_homography(joint_data[modality], homographies[modality])
                draw_joints(labeled["PM"], projected, color=color)

        composite = composite_panels(labeled, target_size)
        cv2.imwrite(
            str(args.output / f"{subject.name}_composite.png"),
            composite,
        )

    manifest = {
        "dataset": "SLP",
        "setting": "danaLab",
        "frame_index": args.frame_index,
        "cover_condition": args.cover,
        "subject_count": len(manifest_subjects),
        "direction_assumption": (
            "H = align_PTr_<modality>.npy maps source modality -> PM reference frame "
            "(per SLP README 'Domain Alignment' section)."
        ),
        "joint_names": list(JOINT_NAMES),
        "subjects": manifest_subjects,
    }
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output_dir": str(args.output),
        "manifest": str(args.manifest),
        "subjects": len(manifest_subjects),
        "frame_index": args.frame_index,
        "cover_condition": args.cover,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()