"""Embed movie overviews into 384-D MiniLM vectors and cache to disk.

Usage:
    uv run python scripts/embed_overviews.py --input data/movies_100k.parquet --out embeddings/overviews_100k.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if args.out.exists():
        print(f"{args.out} already exists — delete to re-embed")
        return 0

    movies = pd.read_parquet(args.input)
    movies = movies[movies["overview"].str.len() > 30].reset_index(drop=True)
    overviews = movies["overview"].tolist()
    print(f"embedding {len(overviews):,} overviews with {args.model}")

    model = SentenceTransformer(args.model)
    X = model.encode(
        overviews,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, X)
    print(f"wrote {args.out} ({X.shape}, {args.out.stat().st_size // (1024 * 1024)} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
