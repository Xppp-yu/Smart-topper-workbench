"""Markdown Relative-Link Checker (R02).

Walks a set of Markdown files (or a directory recursively) and verifies
that every relative Markdown link target resolves to an existing file
on disk.

The check is intentionally local-only (no network), since the B04A
documents only reference files inside the repository.

Usage:
    python scripts/check_markdown_links.py <path> [<path> ...]
    python scripts/check_markdown_links.py docs/

Returns exit 0 on PASS, exit 1 on any missing or malformed link.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable


# A simple markdown link matcher:
#   [label](relative/path.md)
#   [label](relative/path.md#fragment)
#   [label](relative/path.md "title")
LINK_RE = re.compile(
    r"""
    \[(?P<label>[^\]]+)\]
    \(
        (?P<target>[^)\s]+)
        (?:\s+\"[^\"]*\")?
    \)
    """,
    re.VERBOSE,
)


def _iter_markdown_files(paths: Iterable[Path], root: Path) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = p.resolve()
        if p.is_file() and p.suffix.lower() == ".md":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(q.resolve() for q in p.rglob("*.md")))
        else:
            print(f"WARN: skipping non-existent path {p}")
    return out


def _strip_fragment_and_query(target: str) -> str:
    if "#" in target:
        target = target.split("#", 1)[0]
    if "?" in target:
        target = target.split("?", 1)[0]
    return target


def check_file(md_path: Path, repo_root: Path) -> list[str]:
    """Return a list of broken-link error messages for the given file."""
    errors: list[str] = []
    try:
        text = md_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = md_path.read_text(encoding="gbk")
        except Exception as e:  # pragma: no cover
            return [f"{md_path}: could not read ({e})"]

    for m in LINK_RE.finditer(text):
        target = m.group("target")
        label = m.group("label")

        # Skip absolute URLs, mailto, anchors-only, and code-block-link-style
        if target.startswith(("http://", "https://", "mailto:", "tel:")):
            continue
        if not target:
            continue

        # Strip the fragment / query for the existence check
        clean = _strip_fragment_and_query(target)
        if not clean:
            # pure anchor link; nothing to check on disk
            continue

        # Resolve relative to the markdown file
        candidate = (md_path.parent / clean).resolve()
        md_path_abs = md_path.resolve()
        try:
            rel_md = md_path_abs.relative_to(repo_root.resolve())
        except ValueError:
            rel_md = md_path_abs  # shouldn't happen given the caller

        try:
            candidate.relative_to(repo_root.resolve())
        except ValueError:
            # The link escapes the repo root; that's a violation
            errors.append(
                f"{rel_md}: link {target!r} ({label!r}) "
                f"escapes repo root: {candidate}"
            )
            continue

        if not candidate.exists():
            errors.append(
                f"{rel_md}: link {target!r} ({label!r}) "
                f"does not resolve to an existing file: {candidate}"
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("Usage: check_markdown_links.py <path> [<path> ...]")
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    targets = [Path(a) for a in argv]
    files = _iter_markdown_files(targets, repo_root)
    if not files:
        print(f"No markdown files found in: {targets}")
        return 2

    all_errors: list[str] = []
    for f in files:
        all_errors.extend(check_file(f, repo_root))

    print(f"=== Markdown Relative-Link Check ===")
    print(f"Files scanned: {len(files)}")
    print(f"Errors: {len(all_errors)}")
    for e in all_errors:
        print(f"  ! {e}")
    if all_errors:
        print("\nLINK CHECK FAILED")
        return 1
    print("\nLINK CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
