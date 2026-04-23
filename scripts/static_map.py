"""Render a beautiful static datamapplot PNG for slides.

Uses datamapplot.create_plot (non-interactive) + the Claude-labeled cluster
assignments from the hierarchy cache. Writes a high-resolution PNG suitable
for slides.

Usage:
    uv run python scripts/static_map.py --input data/movies_100k.parquet \\
        --embeddings embeddings/overviews_100k.npy --cache data/cluster_labels_100k.json \\
        --layers 300 100 30 10 5 --out map_static_100k.png --layer-idx 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from label_hierarchy import (  # noqa: E402
    build_hierarchy, fit_bertopic, generate_labels, load_data,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=REPO / "data" / "movies.parquet")
    ap.add_argument("--embeddings", type=Path, default=REPO / "embeddings" / "overviews_minilm.npy")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "cluster_labels.json")
    ap.add_argument("--layers", nargs="+", type=int, default=[80, 40, 20, 10, 5])
    ap.add_argument("--layer-idx", type=int, default=2,
                    help="Which layer's labels to render (0=finest, len-1=coarsest). Default 2 = mid-grained.")
    ap.add_argument("--out", type=Path, default=REPO / "map_static.png")
    ap.add_argument("--min-cluster-size", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--figsize", nargs=2, type=float, default=[16.0, 12.0])
    args = ap.parse_args()

    import datamapplot
    from umap import UMAP

    movies, X = load_data(args.input, args.embeddings)
    overviews = movies["overview"].tolist()

    topics, _, topic_emb, topic_terms = fit_bertopic(overviews, X, min_cluster_size=args.min_cluster_size)
    layers = build_hierarchy(topics, topic_emb, args.layers)
    layer_labels = generate_labels(movies, topics, topic_terms, layers, args.layers, args.cache)

    print("computing 2D UMAP for the plot...")
    umap_viz = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X)

    chosen = layer_labels[args.layer_idx]
    n_clusters = len(set(chosen)) - (1 if "Unlabelled" in chosen else 0)
    print(f"rendering static plot: layer {args.layer_idx}, {n_clusters} named clusters")

    fig, _ = datamapplot.create_plot(
        umap_viz,
        chosen,
        figsize=tuple(args.figsize),
        darkmode=True,
        label_over_points=True,
        dynamic_label_size=True,
        label_font_size=9,
        use_medoids=True,
        title="Top movies · thematic clusters",
        sub_title=f"BERTopic on sentence embeddings, {n_clusters} themes labelled by Claude",
    )
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
