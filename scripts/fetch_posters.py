"""Download movie posters referenced in data/movies.parquet.

Usage:
    uv run python scripts/fetch_posters.py [--input data/movies.parquet] [--out data/posters]

Posters come from image.tmdb.org (no API key needed for the CDN). Already-downloaded
files are skipped, so re-running resumes where it left off.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

WORKERS = 20
MAX_RETRIES = 3


def download_one(url: str, dest: Path, session: requests.Session) -> tuple[Path, bool, str | None]:
    if dest.exists() and dest.stat().st_size > 0:
        return dest, True, None
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", "2")))
                continue
            if r.status_code >= 500:
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
            dest.write_bytes(r.content)
            return dest, True, None
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                return dest, False, str(e)
            time.sleep(1 + attempt)
    return dest, False, "exhausted retries"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/movies.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/posters"))
    args = ap.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found. Run fetch_movies.py first.", file=sys.stderr)
        return 1

    df = pd.read_parquet(args.input)
    df = df[df["poster_url"].notna()].copy()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(row.poster_url, args.out / f"{row.id}.jpg") for row in df.itertuples(index=False)]
    todo = [(u, d) for u, d in jobs if not (d.exists() and d.stat().st_size > 0)]
    print(f"{len(jobs) - len(todo)} already present, {len(todo)} to download")
    if not todo:
        return 0

    session = requests.Session()
    failures: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, u, d, session): d for u, d in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="posters"):
            dest, ok, err = fut.result()
            if not ok:
                failures.append((dest, err or "unknown"))

    if failures:
        print(f"{len(failures)} failures:", file=sys.stderr)
        for dest, err in failures[:10]:
            print(f"  {dest.name}: {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
