from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "launch_slp8_b11f_final_fit_r02.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_launch_identity_and_boundaries_are_exact():
    source = _source()
    assert "EXP-SLP-B11F-PM-FINAL-FIT-20260904-AUTODL-R02" in source
    assert "a6a5d8e6f8db003149169ee48f71d6e41e445a80" in source
    assert "5e9d855397face954cac18e3dbadb26449129f828f77d45412b3c4f30d8e6bb2" in source
    assert "a6590d6f068644d98fa5340ec3d4a2e02171b529ec22ab092efb54a298925a43" in source
    assert "34f0fcf45d07920b99b7baf6d595f61297f086ff3187c9ec9b3bd69400b2cd4b" in source
    assert "42e3cbec9def2d735dc02de3343b8dbf830960f2c9ff2ca16b90c3f46dcf3e04" in source
    assert "a5a9342b18d00b614355e63ce056a7edd92dd80358d8aead5ef6e8e0ba045669" in source
    assert "B11F_MAX_SECONDS=2700" in source
    assert "uv run --extra neural python" in source
    assert "--run-authorized" in source
    assert "--resume-authorized" in source
    assert "load_test" not in source
    assert "TEST" not in source
    assert "--force" not in source


def test_first_run_refuses_existing_output_before_runner_dispatch():
    source = _source()
    run_guard = source.index('test ! -e "$B11F_OUTPUT"')
    first_dispatch = source.index("scripts/run_slp8_region_final_fit.py")
    assert run_guard < first_dispatch


def test_resume_is_fail_closed_and_uses_original_deadline():
    source = _source()
    assert 'test ! -e "$B11F_OUTPUT/DONE.json"' in source
    assert 'test ! -e "$B11F_OUTPUT/FAILED.json"' in source
    assert 'budget["deadline_utc_epoch_seconds"]' in source
    assert 'if not 1 <= remaining <= maximum:' in source
    assert '"${B11F_REMAINING_SECONDS}s"' in source
    assert re.search(r'"\$\{B11F_COMMON_ARGS\[@\]\}" --resume-authorized', source)
