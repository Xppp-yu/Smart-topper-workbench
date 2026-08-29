"""Unit tests for the Markdown relative-link checker."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_markdown_links import check_file  # noqa: E402


def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_good_link_passes(tmp_path: Path) -> None:
    target = _write(tmp_path, "docs/target.md", "# target")
    md = _write(tmp_path, "docs/source.md", "see [t](target.md)")
    errs = check_file(md, tmp_path)
    assert not errs, errs


def test_broken_link_caught(tmp_path: Path) -> None:
    md = _write(tmp_path, "docs/source.md", "see [t](missing.md)")
    errs = check_file(md, tmp_path)
    assert errs, "expected an error for missing.md"
    assert "missing.md" in errs[0]


def test_relative_parent_link(tmp_path: Path) -> None:
    target = _write(tmp_path, "configs/x.json", "{}")
    md = _write(tmp_path, "docs/sub/source.md", "see [c](../../configs/x.json)")
    errs = check_file(md, tmp_path)
    assert not errs, errs


def test_external_url_ignored(tmp_path: Path) -> None:
    md = _write(tmp_path, "docs/source.md", "see [a](https://example.com)")
    errs = check_file(md, tmp_path)
    assert not errs, errs


def test_anchor_only_ignored(tmp_path: Path) -> None:
    md = _write(tmp_path, "docs/source.md", "see [a](#section)")
    errs = check_file(md, tmp_path)
    assert not errs, errs


def test_link_escaping_repo_root_caught(tmp_path: Path) -> None:
    # Repo root is tmp_path; make a link that escapes it.
    md = _write(tmp_path, "docs/source.md", "see [a](../../etc/passwd)")
    errs = check_file(md, tmp_path)
    assert any("escapes repo root" in e for e in errs), errs
