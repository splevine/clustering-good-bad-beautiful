"""Dump the cluster metadata (top terms + representative titles) at each
hierarchy layer so that a human (or another LLM running outside the API) can
produce the labels. Writes data/cluster_metadata.json in the same key format
label_hierarchy.py uses for its cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO / "scripts"))
from label_hierarchy import (
    LAYER_SIZES,
    build_hierarchy,
    cluster_repr,
    fit_bertopic,
    load_data,
)


def main():
    movies, X = load_data()
    overviews = movies["overview"].tolist()
    topics, _, topic_emb, topic_terms = fit_bertopic(overviews, X)
    layers = build_hierarchy(topics, topic_emb, LAYER_SIZES)

    dump = {}
    for lvl_i, (doc_labels, size) in enumerate(zip(layers, LAYER_SIZES)):
        cluster_ids = sorted(c for c in set(doc_labels) if c != -1)
        for cid in cluster_ids:
            terms, titles = cluster_repr(movies, doc_labels, cid, topics, topic_terms)
            key = f"L{lvl_i}-{size}-{cid}"
            dump[key] = {"terms": terms, "titles": titles, "size": int((doc_labels == cid).sum())}

    out = REPO / "data" / "cluster_metadata.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dump, indent=2))
    print(f"wrote {out}  ({len(dump)} clusters)")


if __name__ == "__main__":
    main()
