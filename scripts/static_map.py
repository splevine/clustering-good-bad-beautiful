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

    # Fonts: try Inter (Google), fall back to DejaVu Sans.
    font_family = "Inter"

    fig, _ = datamapplot.create_plot(
        umap_viz,
        chosen,
        figsize=tuple(args.figsize),
        darkmode=True,
        use_medoids=True,
        dynamic_label_size=True,
        dynamic_label_size_scaling_factor=1.1,
        font_family=font_family,
        label_wrap_width=16,
        label_linespacing=1.05,
        label_margin_factor=1.8,
        min_font_size=10,
        max_font_size=32,
        min_font_weight=500,
        max_font_weight=800,
        color_label_arrows=True,
        arrowprops={
            "arrowstyle": "-|>",
            "connectionstyle": "arc3,rad=0.08",
            "linewidth": 0.8,
            "mutation_scale": 8,
            "alpha": 0.55,
        },
        point_size=2,
        alpha=0.55,
        add_glow=True,
        glow_keywords={"kernel": "gaussian", "kernel_bandwidth": 0.35, "approx_patch_size": 64},
        palette_hue_shift=10,
        palette_hue_radius_dependence=1.2,
        noise_color="#2a2d36",
        highlight_labels=[
            "Superheroes", "Space sci-fi", "World War & combat", "Christmas films",
            "Westerns & the frontier", "Heists, spies & capers", "Zombies & plagues",
            "Music biopics & rock",
        ],
        highlight_label_keywords={
            "fontweight": "bold",
            "fontsize": 20,
            "bbox": {
                "boxstyle": "round,pad=0.45",
                "facecolor": "#171a22",
                "edgecolor": "#ffb454",
                "linewidth": 1.2,
                "alpha": 0.9,
            },
        },
        title="100,000 films · thematic clusters",
        sub_title=f"BERTopic on MiniLM overviews · {n_clusters} themes labelled by Claude",
        title_keywords={
            "fontfamily": font_family, "fontsize": 28, "fontweight": "bold",
            "color": "#ffb454",
        },
        sub_title_keywords={
            "fontfamily": font_family, "fontsize": 14, "color": "#9aa0a6",
            "style": "italic",
        },
    )
    fig.patch.set_facecolor("#0f1117")
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="#0f1117", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
