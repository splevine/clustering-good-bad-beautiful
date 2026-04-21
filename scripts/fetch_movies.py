"""Fetch top N movies from TMDB by vote count and write data/movies.parquet.

Usage:
    uv run python scripts/fetch_movies.py [--n 5000] [--out data/movies.parquet]

Requires a TMDB v4 Read Access Token in the TMDB_BEARER_TOKEN env var
(or in a .env file in the repo root). Get one free at
https://www.themoviedb.org/settings/api.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
PAGE_SIZE = 20


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


def fetch_page(s: requests.Session, page: int) -> list[dict]:
    params = {
        "sort_by": "vote_count.desc",
        "include_adult": "false",
        "language": "en-US",
        "page": page,
    }
    r = s.get(f"{TMDB_BASE}/discover/movie", params=params, timeout=15)
    if r.status_code == 429:
        retry_after = int(r.headers.get("Retry-After", "2"))
        time.sleep(retry_after)
        return fetch_page(s, page)
    r.raise_for_status()
    return r.json().get("results", [])


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="Number of movies to fetch (default 5000)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/movies.parquet"),
        help="Output parquet path (default data/movies.parquet)",
    )
    args = ap.parse_args()

    load_dotenv()
    token = os.environ.get("TMDB_BEARER_TOKEN")
    if not token:
        print(
            "error: TMDB_BEARER_TOKEN not set. Copy .env.example to .env and fill it in, "
            "or export TMDB_BEARER_TOKEN.",
            file=sys.stderr,
        )
        return 1

    s = tmdb_session(token)
    genre_map = get_genre_map(s)

    pages = (args.n + PAGE_SIZE - 1) // PAGE_SIZE
    rows: list[dict] = []
    for page in tqdm(range(1, pages + 1), desc="Fetching pages"):
        results = fetch_page(s, page)
        if not results:
            break
        rows.extend(normalize(r, genre_map) for r in results)

    df = pd.DataFrame(rows).drop_duplicates(subset="id").head(args.n).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} movies -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
