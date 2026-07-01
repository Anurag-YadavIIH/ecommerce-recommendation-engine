"""
Download the MovieLens 'ml-latest-small' dataset.

The dataset is already bundled in this repo under data/movielens/, so you do NOT
need to run this. It's here in case you want a fresh copy, a different size, or
you deleted the CSVs.

Usage:
    python data/download_data.py
    python data/download_data.py --size full   # the big ~25M-rating version

Source: GroupLens Research, https://grouplens.org/datasets/movielens/
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "full": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
}

# GitHub mirror used as a fallback for the small dataset (works behind some
# firewalls that block grouplens.org).
MIRROR = "https://raw.githubusercontent.com/vinicius-msouza/ml-latest-small/master/"
MIRROR_FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]

DEST = Path(__file__).resolve().parent / "movielens"


def _from_official(size: str) -> bool:
    try:
        print(f"Downloading '{size}' dataset from GroupLens ...")
        req = urllib.request.Request(URLS[size], headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for member in zf.namelist():
                if member.endswith(".csv"):
                    name = Path(member).name
                    DEST.mkdir(parents=True, exist_ok=True)
                    (DEST / name).write_bytes(zf.read(member))
                    print(f"  saved {name}")
        return True
    except Exception as exc:
        print(f"  official source failed ({exc}); trying mirror ...")
        return False


def _from_mirror() -> bool:
    try:
        DEST.mkdir(parents=True, exist_ok=True)
        for f in MIRROR_FILES:
            req = urllib.request.Request(MIRROR + f, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=60).read()
            (DEST / f).write_bytes(data)
            print(f"  saved {f}")
        return True
    except Exception as exc:
        print(f"  mirror failed ({exc})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "full"], default="small")
    args = parser.parse_args()

    ok = _from_official(args.size)
    if not ok and args.size == "small":
        ok = _from_mirror()

    if ok:
        print(f"\nDone. Files are in {DEST}")
    else:
        print("\nCould not download the dataset. Check your internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
