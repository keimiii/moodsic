#!/usr/bin/env python3
"""
Setup script to vendor EmoNet (code + optional pretrained weights) into this repo
without modifications, preserving license and attribution.

It downloads the face-analysis/emonet GitHub repository as a zip (pinned branch)
and copies the following into models/emonet/ by default:
  - LICENSE.txt
  - README.md
  - emonet/ (package directory)
  - pretrained/ (checkpoints, if present in repo)
Optionally, demo.py and demo_video.py can be included.

You can also pass one or more --weights-url to fetch checkpoints if the upstream
repo does not include them in the zip (or if you prefer specific assets).

Usage examples:
  python scripts/emonet_setup.py
  python scripts/emonet_setup.py --branch main --include-demos
  python scripts/emonet_setup.py --weights-url https://example.com/emonet_8.pth --weights-url https://example.com/emonet_5.pth
  python scripts/emonet_setup.py --dest models/emonet --force

Notes:
- This project uses EmoNet under CC BY-NC-ND 4.0. We distribute unmodified code
  and (optionally) unmodified checkpoints with attribution. Do not fine-tune or
  modify weights for distribution.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

REPO_OWNER = "face-analysis"
REPO_NAME = "emonet"
DEFAULT_BRANCHES = ["main", "master"]
DEFAULT_DEST = Path("models/emonet")

COPY_DIRS = ["emonet", "pretrained"]
COPY_FILES = ["LICENSE.txt", "README.md"]
OPTIONAL_FILES = ["demo.py", "demo_video.py", "test.py"]

NOTICE_TEXT = """This directory vendors third-party code from EmoNet
(https://github.com/face-analysis/emonet), used under the
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
(CC BY-NC-ND 4.0) license. No modifications have been made to the vendored
files. See LICENSE.txt for details. Copyright belongs to the original authors.
"""

def _merge_tree(src: Path, dst: Path, force: bool = False) -> None:
    """Merge directory tree src into dst. If force=True, overwrite conflicts.
    If force=False, keep existing files and skip conflicts.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _merge_tree(item, target, force=force)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if force:
                    target.unlink()
                else:
                    # Skip existing file when not forcing
                    continue
            target.write_bytes(item.read_bytes())

def _flatten_nested_emonet(dest: Path, force: bool = False) -> None:
    """If vendored layout is models/emonet/emonet/..., flatten so only one
    'emonet' package exists with this structure:
      - models/emonet/__init__.py
      - models/emonet/models/emonet.py (+ __init__.py)
      - models/emonet/{evaluation.py, metrics.py, data_augmentation.py}
      - models/emonet/data/
    """
    nested = dest / "emonet"
    if not nested.exists() or not nested.is_dir():
        return

    print("Flattening nested emonet/ package into", dest)

    # 1) Move emonet/models -> dest/models
    src_models = nested / "models"
    if src_models.exists():
        _merge_tree(src_models, dest / "models", force=force)

    # 2) Move top-level modules
    for fname in ["evaluation.py", "metrics.py", "data_augmentation.py"]:
        src = nested / fname
        if src.exists():
            dst = dest / fname
            if dst.exists() and force:
                dst.unlink()
            if not dst.exists():
                dst.write_bytes(src.read_bytes())

    # 3) Move data directory
    src_data = nested / "data"
    if src_data.exists():
        _merge_tree(src_data, dest / "data", force=force)

    # 4) __init__.py -> dest/__init__.py (keep upstream version if forcing)
    src_init = nested / "__init__.py"
    if src_init.exists():
        dst_init = dest / "__init__.py"
        if dst_init.exists() and force:
            dst_init.unlink()
        if not dst_init.exists():
            dst_init.write_bytes(src_init.read_bytes())

    # 5) Cleanup: remove nested folder
    import shutil
    shutil.rmtree(nested, ignore_errors=True)
    print("Flatten complete.")


def _download_repo_zip(branch: str) -> bytes:
    url = f"https://codeload.github.com/{REPO_OWNER}/{REPO_NAME}/zip/refs/heads/{branch}"
    try:
        with urlopen(url, timeout=60) as resp:
            return resp.read()
    except HTTPError as e:
        raise RuntimeError(f"Failed to download {url}: {e}")
    except URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e}")


def _find_root_prefix(z: zipfile.ZipFile) -> str:
    # Most GitHub zips have a single top-level folder like emonet-main/
    names = z.namelist()
    if not names:
        raise RuntimeError("Zip archive is empty")
    prefix = names[0].split("/")[0]
    if not prefix:
        # Fallback: compute common prefix
        prefix = os.path.commonprefix(names).split("/")[0]
    return prefix


def _extract_members(z: zipfile.ZipFile, members: list[str], temp_dir: Path) -> None:
    for m in members:
        for zi in z.infolist():
            if zi.filename.rstrip("/").endswith(m):
                z.extract(zi, path=temp_dir)


def _safe_copy(src: Path, dst: Path, force: bool = False) -> None:
    if src.is_dir():
        if dst.exists():
            if force:
                shutil.rmtree(dst)
            else:
                raise FileExistsError(f"Destination exists: {dst} (use --force to overwrite)")
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not force:
            raise FileExistsError(f"Destination exists: {dst} (use --force to overwrite)")
        shutil.copy2(src, dst)


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=120) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:
        raise RuntimeError(f"Failed to download weights from {url}: {e}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Vendor EmoNet into models/emonet")
    p.add_argument("--branch", default=None, help="Git branch to download (default: try main then master)")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination directory (default: models/emonet)")
    p.add_argument("--include-demos", action="store_true", help="Include demo.py and demo_video.py")
    p.add_argument("--force", action="store_true", help="Overwrite destination files if they exist")
    p.add_argument("--weights-url", action="append", default=[], help="Additional checkpoint URL(s) to download into pretrained/")

    args = p.parse_args(argv)

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    branches = [args.branch] if args.branch else DEFAULT_BRANCHES

    zip_bytes: bytes | None = None
    last_error: Exception | None = None
    for br in branches:
        try:
            print(f"Downloading {REPO_OWNER}/{REPO_NAME}@{br}...")
            zip_bytes = _download_repo_zip(br)
            branch = br
            break
        except Exception as e:
            print(f"  Failed for {br}: {e}")
            last_error = e
            continue
    if zip_bytes is None:
        raise SystemExit(f"Could not download repo zip from any branch {branches}: {last_error}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            root_prefix = _find_root_prefix(zf)
            # Extract entire zip; we'll copy only what we need
            zf.extractall(td_path)

        repo_root = td_path / root_prefix

        # Copy license and readme
        for fname in COPY_FILES:
            src = repo_root / fname
            if src.exists():
                _safe_copy(src, dest / fname, force=args.force)
                print(f"Copied {fname}")
            else:
                print(f"Warning: missing {fname} in upstream")

        # Copy package and pretrained dirs
        for dname in COPY_DIRS:
            src = repo_root / dname
            if src.exists():
                _safe_copy(src, dest / dname, force=args.force)
                print(f"Copied {dname}/")
            else:
                print(f"Note: {dname}/ not found in upstream (skipping)")

    # Flatten nested emonet/ if present
    _flatten_nested_emonet(dest, force=args.force)

        # Optionally copy demos
        if args.include_demos:
            for fname in OPTIONAL_FILES:
                src = repo_root / fname
                if src.exists():
                    _safe_copy(src, dest / fname, force=args.force)
                    print(f"Copied {fname}")

    # Write NOTICE
    notice_path = dest / "NOTICE.txt"
    notice_path.write_text(NOTICE_TEXT, encoding="utf-8")
    print(f"Wrote {notice_path}")

    # Download additional weights if requested
    weights_dir = dest / "pretrained"
    if args.weights_url:
        weights_dir.mkdir(parents=True, exist_ok=True)
        for url in args.weights_url:
            filename = url.split("/")[-1] or "checkpoint.pth"
            out = weights_dir / filename
            print(f"Downloading weights: {url} -> {out}")
            _download_file(url, out)

    # Summarize
    copied = []
    for path in [*(dest / d for d in COPY_DIRS if d != "emonet"), *(dest / f for f in COPY_FILES)]:
        if path.exists():
            copied.append(path)
    weights = list(weights_dir.glob("**/*")) if weights_dir.exists() else []

    print("\nSummary:")
    for pth in copied:
        print(f"  - {pth}")
    if weights:
        print("  - Weights:")
        for w in weights:
            if w.is_file():
                print(f"      * {w.relative_to(dest)} ({w.stat().st_size} bytes)")
    else:
        print("  - No weights found in pretrained/. You may use --weights-url to fetch specific files if upstream does not include them in the zip.")

    print("\nDone. Ensure your runtime loads EmoNet checkpoints from models/emonet/pretrained/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
