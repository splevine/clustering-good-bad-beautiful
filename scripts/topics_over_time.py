"""BERTopic topics_over_time: show how thematic clusters emerge and fade
across a century of cinema. Uses `release_year` as the timestamp and the
Claude labels from data/cluster_labels.json when available.

Run:
    uv run python scripts/topics_over_time.py

Outputs:
    topics_over_time.html  — interactive Plotly chart
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from label_hierarchy import fit_bertopic, load_data  # noqa: E402

EMB_DIR = REPO / "embeddings"
DATA_DIR = REPO / "data"
LABELS_CACHE = DATA_DIR / "cluster_labels.json"
OUT_PATH = REPO / "topics_over_time.html"


def main():
    movies, X = load_data()
    movies = movies[movies["release_year"].notna()].reset_index(drop=True)
    X = X[movies.index]  # not strictly needed after reset_index but keeps explicit
    overviews = movies["overview"].tolist()

    topics, topic_model, _, _ = fit_bertopic(overviews, X)

    # Use the finest-layer Claude labels where available so the legend reads well.
    cache = {}
    if LABELS_CACHE.exists():
        cache = json.loads(LABELS_CACHE.read_text())

    # Replace BERTopic's auto-generated labels with Claude's (layer 0, size 80).
    claude_labels = {}
    for key, label in cache.items():
        if key.startswith("L0-80-"):
            cid = int(key.split("-")[-1])
            claude_labels[cid] = label
    if claude_labels:
        topic_model.set_topic_labels(claude_labels)
        print(f"applied {len(claude_labels)} Claude labels")

    # Bin by decade. Movies span 1902-2026 → ~12-13 buckets.
    timestamps = movies["release_year"].astype(int).tolist()

    print("computing topics_over_time...")
    tot = topic_model.topics_over_time(
        overviews,
        timestamps,
        topics=topics.tolist(),
        nr_bins=13,
        evolution_tuning=True,
        global_tuning=True,
    )

    # Pick the top-15 topics by total frequency to avoid a cluttered legend.
    fig = topic_model.visualize_topics_over_time(
        tot,
        top_n_topics=15,
        custom_labels=True,
        title="Emerging and fading themes, 1902 – 2026",
        width=1200,
        height=600,
    )
    # Tighter dark theme so it matches the landing page.
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font_color="#e8e8ea",
        title_font_color="#ffb454",
        xaxis=dict(showgrid=False, color="#9aa0a6"),
        yaxis=dict(gridcolor="#262a36", color="#9aa0a6"),
        legend=dict(font=dict(color="#e8e8ea")),
    )

    fig.write_html(str(OUT_PATH), include_plotlyjs="cdn", full_html=True)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
