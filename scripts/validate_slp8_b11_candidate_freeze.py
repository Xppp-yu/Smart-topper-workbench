"""Fail-closed validator for the B11 SLP8 research-candidate freeze."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXPECTED = {
    "candidate_id": "slp8_pm_research_candidate_v0.1",
    "model_family": "slp8_deeplabv3plus_lite_v0.1",
    "archive_sha256": "68156598a47ae65ba33d26f4005f9d9fdc8ec67ff24d43ffd605c19847ca5918",
    "summary_sha256": "548bb1a798bc49dcd6a591197b4abf61e2dd2c7733ac51e8070519e9b90a166a",
}


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"unreadable config: {exc}"]
    if d.get("candidate_id") != EXPECTED["candidate_id"]: errors.append("candidate_id mismatch")
    if d.get("model_family") != EXPECTED["model_family"]: errors.append("model_family mismatch")
    ev = d.get("development_evidence", {})
    con = d.get("consensus_evidence", {})
    fit = d.get("final_development_fit", {})
    inf = d.get("inference_contract", {})
    if ev.get("archive_sha256") != EXPECTED["archive_sha256"]: errors.append("archive hash mismatch")
    if con.get("summary_sha256") != EXPECTED["summary_sha256"]: errors.append("consensus hash mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(ev.get("runner_git_commit", ""))): errors.append("invalid runner SHA")
    if ev.get("test_access") is not False: errors.append("TEST must be strict false")
    if fit.get("status") != "NOT_RUN_SEPARATE_GPU_GATE": errors.append("final fit must remain NOT_RUN")
    if fit.get("seeds") != [42, 123, 2026]: errors.append("seed set mismatch")
    if fit.get("fixed_epochs_by_seed") != {"42": 15, "123": 20, "2026": 12}: errors.append("epoch freeze mismatch")
    if fit.get("models_required") != 3: errors.append("three final models required")
    if inf.get("primary_prediction") != "per_pixel_majority_vote_across_three_seed_models": errors.append("primary inference mismatch")
    if inf.get("probability_calibration") != "NOT_AVAILABLE": errors.append("probability calibration must remain unavailable")
    if inf.get("ood_detection") != "NOT_AVAILABLE": errors.append("OOD must remain unavailable")
    if d.get("test_gate") != "B09T_SEPARATE_ONE_TIME_OWNER_AUTHORIZATION_REQUIRED": errors.append("TEST gate mismatch")
    if len(d.get("prohibited_conclusions", [])) < 6: errors.append("prohibited conclusions incomplete")
    return errors


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("config", type=Path); args = p.parse_args()
    errors = validate(args.config)
    for error in errors: print(f"ERR: {error}")
    print(f"summary: {0 if errors else 1} OK / {len(errors)} ERR")
    return 1 if errors else 0


if __name__ == "__main__": raise SystemExit(main())
