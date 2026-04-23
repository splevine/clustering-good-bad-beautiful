"""Fetch movies from TMDB and write a parquet.

Two modes:
- Default: single /discover/movie sort_by=vote_count.desc. Capped at 10,000
  films by TMDB's 500-page pagination limit.
- --by-year: iterate through every primary_release_year and paginate within
  each, which bypasses the global page cap. Slower but can reach 100K+ films.

Usage:
    uv run python scripts/fetch_movies.py                                  # top 5K, fast
    uv run python scripts/fetch_movies.py --n 10000                        # top 10K (hard cap)
    uv run python scripts/fetch_movies.py --by-year --n 100000 --out data/movies_100k.parquet

Requires a TMDB v4 Read Access Token in the TMDB_BEARER_TOKEN env var
(or in a .env file in the repo root). Get one free at
https://www.themoviedb.org/settings/api.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.exceptions import HTTPError
from tqdm import tqdm

TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
PAGE_SIZE = 20
TMDB_PAGE_CAP = 500  # TMDB's hard pagination limit per /discover query


def tmdb_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
    )
    return s


def get_genre_map(s: requests.Session) -> dict[int, str]:
    r = s.get(f"{TMDB_BASE}/genre/movie/list", params={"language": "en-US"}, timeout=15)
    r.raise_for_status()
    return {g["id"]: g["name"] for g in r.json()["genres"]}


def fetch_page(s: requests.Session, page: int, extra_params: dict | None = None,
               max_retries: int = 4) -> list[dict] | None:
    """Fetch one page. Returns the list of results, or None if TMDB refused
    (400 = page cap) or we gave up after retries. Retries on 429 and 5xx."""
    params = {
        "sort_by": "vote_count.desc",
        "include_adult": "false",
        "language": "en-US",
        "page": page,
    }
    if extra_params:
        params.update(extra_params)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = s.get(f"{TMDB_BASE}/discover/movie", params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", "2")))
                continue
            if 500 <= r.status_code < 600:
                # Transient server-side error; back off and retry.
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 400:
                return None  # pagination cap
            r.raise_for_status()
            return r.json().get("results", [])
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    tqdm.write(f"giving up on page {page} params={extra_params}: {last_err}")
    return []  # skip this page, don't crash the whole run


def normalize(row: dict, genre_map: dict[int, str]) -> dict:
    release_year = None
    if rd := row.get("release_date"):
        try:
            release_year = int(rd[:4])
        except ValueError:
            release_year = None
    poster_path = row.get("poster_path")
    return {
        "id": row["id"],
        "title": row.get("title") or "",
        "overview": row.get("overview") or "",
        "original_language": row.get("original_language"),
        "release_year": release_year,
        "genres": [genre_map.get(g, str(g)) for g in row.get("genre_ids", [])],
        "vote_average": row.get("vote_average"),
        "vote_count": row.get("vote_count"),
        "popularity": row.get("popularity"),
        "poster_path": poster_path,
        "poster_url": f"{POSTER_BASE}{poster_path}" if poster_path else None,
    }


def fetch_flat(s, genre_map, n_target):
    """Single /discover query, respecting TMDB's 500-page cap."""
    pages = min((n_target + PAGE_SIZE - 1) // PAGE_SIZE, TMDB_PAGE_CAP)
    rows = []
    for page in tqdm(range(1, pages + 1), desc="Flat fetch"):
        results = fetch_page(s, page)
        if results is None:
            tqdm.write(f"TMDB returned 400 on page {page}; stopping at {len(rows)} rows")
            break
        if not results:
            break
        rows.extend(normalize(r, genre_map) for r in results)
    return rows


def fetch_by_year(s, genre_map, year_from, year_to, pages_per_year):
    """Partition by primary_release_year to bypass TMDB's global page cap."""
    rows = []
    years = list(range(year_from, year_to + 1))
    for year in tqdm(years, desc=f"By year ({year_from}–{year_to})"):
        for page in range(1, pages_per_year + 1):
            results = fetch_page(s, page, {"primary_release_year": year})
            if results is None:  # page cap hit
                break
            if not results:  # empty / failed page — try next one, don't bail out of the year
                continue
            rows.extend(normalize(r, genre_map) for r in results)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="Target movie count (default 5000)")
    ap.add_argument("--out", type=Path, default=Path("data/movies.parquet"))
    ap.add_argument("--by-year", action="store_true",
                    help="Partition by primary_release_year to bypass the 500-page global cap")
    ap.add_argument("--year-from", type=int, default=1900)
    ap.add_argument("--year-to", type=int, default=dt.date.today().year)
    ap.add_argument("--pages-per-year", type=int, default=50,
                    help="Max pages per year when --by-year (default 50 ≈ 1000 films/year max)")
    args = ap.parse_args()

    load_dotenv()
    token = os.environ.get("TMDB_BEARER_TOKEN")
    if not token:
        print("error: TMDB_BEARER_TOKEN not set", file=sys.stderr)
        return 1

    s = tmdb_session(token)
    genre_map = get_genre_map(s)

    rows: list[dict] = []
    try:
        if args.by_year:
            rows = fetch_by_year(s, genre_map, args.year_from, args.year_to, args.pages_per_year)
        else:
            rows = fetch_flat(s, genre_map, args.n)
    except KeyboardInterrupt:
        print("\ninterrupted — writing partial results")
    except Exception as e:
        print(f"\nfetch error: {e} — writing partial results")

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset="id")
        .sort_values("vote_count", ascending=False, na_position="last")
        .head(args.n)
        .reset_index(drop=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} movies -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
