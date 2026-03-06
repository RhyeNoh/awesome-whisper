#!/usr/bin/env python3
"""
Scan all .txt files under /workspace/data and print any file content that contains
one or more of: /  \  (  {  [

Usage:
  python scan_txt_special_chars.py
  python scan_txt_special_chars.py --root /workspace/data
  python scan_txt_special_chars.py --glob "**/*.txt"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


NEEDLES = ["/", "\\", "(", "{", "["]


def iter_txt_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")
    return sorted(root.glob(pattern))


def file_contains_any(path: Path, needles: list[str]) -> bool:
    # Read as UTF-8, but tolerate weird bytes.
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(ch in text for ch in needles)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print .txt files containing certain characters.")
    parser.add_argument("--root", type=Path, default=Path("/workspace/data"), help="Root directory to scan.")
    parser.add_argument("--glob", dest="glob_pattern", default="**/*.txt", help="Glob pattern relative to root.")
    args = parser.parse_args()

    try:
        txt_files = iter_txt_files(args.root, args.glob_pattern)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not txt_files:
        print(f"No .txt files found under {args.root} using pattern '{args.glob_pattern}'.")
        return 0

    matches = 0
    for p in txt_files:
        try:
            if file_contains_any(p, NEEDLES):
                matches += 1
                content = p.read_text(encoding="utf-8", errors="replace")
                print("=" * 100)
                print(f"FILE: {p}")
                print(f"MATCHED_CHARS: {[ch for ch in NEEDLES if ch in content]}")
                print("-" * 100)
                print(content.rstrip("\n"))
                print()
        except Exception as e:
            print(f"WARNING: Failed to read {p}: {e}", file=sys.stderr)

    print("=" * 100)
    print(f"Done. Matched {matches} file(s) out of {len(txt_files)} scanned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

