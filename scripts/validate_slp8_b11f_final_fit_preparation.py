"""Fail-closed B11F preparation validator; performs no training or writes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topper_perception.neural.slp8_region_final_fit import build_plan, load_protocol


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("config", type=Path); args = p.parse_args()
    try:
        protocol = load_protocol(args.config.resolve(), ROOT)
        plan = build_plan(protocol)
        if plan != ((42, 15), (123, 20), (2026, 12)):
            raise RuntimeError("execution plan mismatch")
        print("summary: PASS (fail-closed protocol and execution-plan validation)")
        print("B11F_FINAL_FIT_PREPARATION_VALIDATION_PASSED TEST=0 GPU_NOT_AUTHORIZED")
        return 0
    except Exception as exc:
        print(f"ERR: {exc}"); print("B11F_FINAL_FIT_PREPARATION_VALIDATION_FAILED"); return 1


if __name__ == "__main__": raise SystemExit(main())
