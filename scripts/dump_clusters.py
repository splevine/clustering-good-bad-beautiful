"""Dump cluster metadata (top terms + representative titles) at each hierarchy
layer so that a human (or another LLM outside the API) can produce the labels.
Writes a JSON in the same key format `label_hierarchy.py` caches.

Usage (5K default):
    uv run python scripts/dump_clusters.py
Usage (100K):
    uv run python scripts/dump_clusters.py \\
        --input data/movies_100k.parquet --embeddings embeddings/overviews_100k.npy \\
        --out data/cluster_metadata_100k.json --layers 300 100 30 10 5 --min-cluster-size 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from label_hierarchy import (  # noqa: E402
    build_hierarchy, cluster_repr, fit_bertopic, load_data,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=REPO / "data" / "movies.parquet")
    ap.add_argument("--embeddings", type=Path, default=REPO / "embeddings" / "overviews_minilm.npy")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "cluster_metadata.json")
    ap.add_argument("--layers", nargs="+", type=int, default=[80, 40, 20, 10, 5])
    ap.add_argument("--min-cluster-size", type=int, default=20)
    args = ap.parse_args()

    movies, X = load_data(args.input, args.embeddings)
    overviews = movies["overview"].tolist()
    topics, _, topic_emb, topic_terms = fit_bertopic(overviews, X, min_cluster_size=args.min_cluster_size)
    layers = build_hierarchy(topics, topic_emb, args.layers)

    dump = {}
    for lvl_i, (doc_labels, size) in enumerate(zip(layers, args.layers)):
        cluster_ids = sorted(c for c in set(doc_labels) if c != -1)
        for cid in cluster_ids:
            terms, titles = cluster_repr(movies, doc_labels, cid, topics, topic_terms)
            key = f"L{lvl_i}-{size}-{cid}"
            dump[key] = {"terms": terms, "titles": titles, "size": int((doc_labels == cid).sum())}

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(dump, indent=2))
    print(f"wrote {args.out}  ({len(dump)} clusters)")


if __name__ == "__main__":
    main()
