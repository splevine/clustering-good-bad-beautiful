"""Render BERTopic's built-in visualization suite as HTML artifacts.

Usage:
    uv run python scripts/bertopic_visuals.py --input data/movies_100k.parquet \\
        --embeddings embeddings/overviews_100k.npy --cache data/cluster_labels_100k.json \\
        --out-prefix topics_100k_ --min-cluster-size 40

Outputs (one HTML per visualization):
    <prefix>scatter.html      — UMAP scatter of all topics (2-D)
    <prefix>hierarchy.html    — dendrogram of topic relationships
    <prefix>heatmap.html      — pairwise topic similarity heatmap
    <prefix>barchart.html     — top c-TF-IDF terms per topic (top N)
    <prefix>over_time.html    — topic prevalence over release years
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
from label_hierarchy import fit_bertopic, load_data  # noqa: E402


DARK = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font_color="#e8e8ea",
    title_font_color="#ffb454",
)


def apply_dark(fig):
    fig.update_layout(**DARK)
    fig.update_xaxes(color="#9aa0a6", gridcolor="#262a36")
    fig.update_yaxes(color="#9aa0a6", gridcolor="#262a36")
    return fig


def load_claude_labels(cache_path: Path, layer_size: int | None = None):
    """Return {topic_id: label} from the cache, using the finest-grained layer
    whose size matches the natural topic count (or the first L0 entry)."""
    if not cache_path.exists():
        return {}
    cache = json.loads(cache_path.read_text())
    labels = {}
    # Prefer entries under "L0-<layer_size>-<cid>"; if layer_size None, any L0
    prefix = f"L0-{layer_size}-" if layer_size is not None else "L0-"
    for key, value in cache.items():
        if key.startswith(prefix):
            # key is "L0-<size>-<cid>"
            cid = int(key.rsplit("-", 1)[-1])
            labels[cid] = value
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=REPO / "data" / "movies.parquet")
    ap.add_argument("--embeddings", type=Path, default=REPO / "embeddings" / "overviews_minilm.npy")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "cluster_labels.json")
    ap.add_argument("--out-prefix", default="topics_")
    ap.add_argument("--min-cluster-size", type=int, default=20)
    ap.add_argument("--top-n-topics", type=int, default=20, help="How many topics to show in charts")
    ap.add_argument("--layer-size-for-labels", type=int, default=None,
                    help="Which L0-<size>-* cache entries to treat as canonical labels. "
                         "Defaults to the number of natural topics after fitting.")
    args = ap.parse_args()

    movies, X = load_data(args.input, args.embeddings)
    overviews = movies["overview"].tolist()

    topics, topic_model, _, _ = fit_bertopic(overviews, X, min_cluster_size=args.min_cluster_size)
    n_real = len(set(topics)) - (1 if -1 in topics else 0)
    layer_size = args.layer_size_for_labels or max(80, n_real)

    claude_labels = load_claude_labels(args.cache, layer_size=layer_size)
    if claude_labels:
        topic_model.set_topic_labels(claude_labels)
        print(f"applied {len(claude_labels)} labels (layer L0-{layer_size})")

    prefix = args.out_prefix
    outputs = []

    # 1. Topic scatter (each topic as a point, sized by frequency)
    print("rendering scatter...")
    fig = topic_model.visualize_topics(custom_labels=bool(claude_labels))
    fig = apply_dark(fig)
    p = REPO / f"{prefix}scatter.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    outputs.append(p)

    # 2. Hierarchy (dendrogram)
    print("rendering hierarchy dendrogram...")
    try:
        hier = topic_model.hierarchical_topics(overviews)
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hier,
                                              top_n_topics=args.top_n_topics,
                                              custom_labels=bool(claude_labels))
        fig = apply_dark(fig)
        p = REPO / f"{prefix}hierarchy.html"
        fig.write_html(str(p), include_plotlyjs="cdn")
        outputs.append(p)
    except Exception as e:
        print(f"  hierarchy failed: {e}")

    # 3. Heatmap (topic similarity)
    print("rendering heatmap...")
    fig = topic_model.visualize_heatmap(top_n_topics=args.top_n_topics,
                                        custom_labels=bool(claude_labels))
    fig = apply_dark(fig)
    p = REPO / f"{prefix}heatmap.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    outputs.append(p)

    # 4. Bar chart (top c-TF-IDF terms per topic)
    print("rendering barchart...")
    fig = topic_model.visualize_barchart(top_n_topics=args.top_n_topics,
                                         custom_labels=bool(claude_labels))
    fig = apply_dark(fig)
    p = REPO / f"{prefix}barchart.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    outputs.append(p)

    # 5. Topics over time
    print("rendering topics_over_time...")
    timestamps = movies["release_year"].fillna(0).astype(int).tolist()
    # BERTopic wants non-zero timestamps; filter to non-zero
    mask = [t > 0 for t in timestamps]
    ot_docs = [d for d, m in zip(overviews, mask) if m]
    ot_ts = [t for t, m in zip(timestamps, mask) if m]
    ot_topics = [t for t, m in zip(topics.tolist(), mask) if m]
    tot = topic_model.topics_over_time(ot_docs, ot_ts, topics=ot_topics, nr_bins=13,
                                       evolution_tuning=True, global_tuning=True)
    fig = topic_model.visualize_topics_over_time(tot,
                                                 top_n_topics=15,
                                                 custom_labels=bool(claude_labels),
                                                 width=1200, height=600)
    fig = apply_dark(fig)
    p = REPO / f"{prefix}over_time.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    outputs.append(p)

    print("\nwrote:")
    for p in outputs:
        print(f"  {p.name} ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
