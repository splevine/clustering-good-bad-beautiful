"""Build a 5-layer BERTopic hierarchy and render an interactive datamapplot.

Steps:
1. Load movies + cached overview embeddings.
2. Fit BERTopic to get natural topics from the overviews.
3. Derive hierarchy layers by agglomerative-clustering the topic embeddings
   at the requested cluster counts.
4. Label each cluster at each layer with Claude (falls back to c-TF-IDF if
   ANTHROPIC_API_KEY is missing).
5. Render the datamapplot with search, topic tree, histogram filter on
   release_year, a rich hover tooltip, and click-through to TMDB.

Usage (5K default):
    uv run python scripts/label_hierarchy.py
Usage (100K):
    uv run python scripts/label_hierarchy.py \\
        --input data/movies_100k.parquet --embeddings embeddings/overviews_100k.npy \\
        --out map_100k.html --cache data/cluster_labels_100k.json \\
        --layers 300 100 30 10 5 --title "100,000 movies · thematic hierarchy"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
EMB_DIR = REPO / "embeddings"


def load_data(movies_path: Path, emb_path: Path):
    movies = pd.read_parquet(movies_path)
    movies = movies[movies["overview"].str.len() > 30].reset_index(drop=True)
    if not emb_path.exists():
        sys.exit(f"error: {emb_path} not found. Run scripts/embed_overviews.py first.")
    X = np.load(emb_path)
    assert len(movies) == len(X), (len(movies), X.shape)
    return movies, X


def fit_bertopic(overviews, X, min_cluster_size: int = 20):
    import hdbscan
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    print("fitting BERTopic...")
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=0),
        hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5,
                                      cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2)),
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(overviews, embeddings=X)
    topics = np.asarray(topics)
    topic_emb = np.asarray(topic_model.topic_embeddings_)
    topic_terms = {}
    for tid in sorted(set(topics)):
        if tid == -1:
            topic_terms[-1] = []
            continue
        topic_terms[tid] = [term for term, _ in topic_model.get_topic(tid)[:10]]
    n_real = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"  {n_real} natural topics, {(topics == -1).sum()} noise docs ({(topics == -1).mean():.1%})")
    return topics, topic_model, topic_emb, topic_terms


def build_hierarchy(topics, topic_emb, layer_sizes):
    from sklearn.cluster import AgglomerativeClustering

    real_topic_ids = sorted(t for t in set(topics) if t != -1)
    if topic_emb.shape[0] == len(real_topic_ids) + 1:
        real_emb = topic_emb[1:]
    else:
        real_emb = topic_emb[real_topic_ids]

    layers = []
    for size in layer_sizes:
        n_clusters = min(size, len(real_topic_ids))
        if n_clusters >= len(real_topic_ids):
            coarse_of_topic = np.arange(len(real_topic_ids))
        else:
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            coarse_of_topic = agg.fit_predict(real_emb)
        topic_to_coarse = {tid: int(coarse_of_topic[i]) for i, tid in enumerate(real_topic_ids)}
        per_doc = np.array([topic_to_coarse[t] if t != -1 else -1 for t in topics])
        layers.append(per_doc)
        print(f"  layer (size={size}): {len(set(per_doc)) - (1 if -1 in per_doc else 0)} clusters")
    return layers


def load_label_cache(cache_path: Path):
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def save_label_cache(cache_path: Path, cache):
    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def cluster_repr(movies, doc_labels, cluster_id, topics, topic_terms):
    mask = doc_labels == cluster_id
    sub = movies.loc[mask].sort_values("vote_count", ascending=False)
    titles = sub["title"].head(5).tolist()

    topic_ids_in_cluster = [int(t) for t in np.unique(topics[mask]) if t != -1]
    term_pool = []
    for t in topic_ids_in_cluster[:5]:
        term_pool.extend(topic_terms.get(t, [])[:5])
    seen, terms = set(), []
    for term in term_pool:
        if term not in seen:
            seen.add(term)
            terms.append(term)
        if len(terms) >= 10:
            break
    return terms, titles


def label_with_claude(client, level_idx, level_size, cluster_id, terms, titles):
    prompt = (
        f"You are labeling a movie-cluster in a hierarchical visualization. "
        f"This is hierarchy level {level_idx + 1} (level 1 = finest-grained). "
        f"The level has {level_size} clusters total.\n\n"
        f"Top distinctive words: {', '.join(terms) if terms else '(no terms)'}\n"
        f"5 representative films: {', '.join(titles)}\n\n"
        "Give a concise 2-4 word label capturing the shared theme, tone, or setting. "
        "Avoid generic words like 'films', 'movies', 'cinema'. Prefer specific concepts "
        "(e.g., 'Coming-of-age drama', 'Heist thrillers', 'Cosmic horror'). "
        "Return ONLY the label — no quotes, no explanation."
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=40,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    return text.strip('"\'').strip(". ")[:60] or "Unlabelled"


def generate_labels(movies, topics, topic_terms, layers, layer_sizes, cache_path: Path):
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    cache = load_label_cache(cache_path)

    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        print(f"using Claude for labels (cache: {cache_path})")
    else:
        client = None
        print(f"ANTHROPIC_API_KEY not set — falling back to c-TF-IDF (cache: {cache_path})")

    layer_labels = []
    for lvl_i, (doc_labels, size) in enumerate(zip(layers, layer_sizes)):
        cluster_ids = sorted(c for c in set(doc_labels) if c != -1)
        labels_this = {-1: "Unlabelled"}
        for cid in tqdm(cluster_ids, desc=f"layer {lvl_i + 1}/{len(layer_sizes)} (size {size})"):
            key = f"L{lvl_i}-{size}-{cid}"
            if key in cache:
                labels_this[cid] = cache[key]
                continue
            terms, titles = cluster_repr(movies, doc_labels, cid, topics, topic_terms)
            if client is not None:
                try:
                    label = label_with_claude(client, lvl_i, size, cid, terms, titles)
                except Exception as e:
                    print(f"  API error on {key}: {e}; falling back to keywords for this cluster")
                    label = " · ".join(terms[:3]) or "Unlabelled"
            else:
                label = " · ".join(terms[:3]) or "Unlabelled"
            labels_this[cid] = label
            cache[key] = label
            save_label_cache(cache_path, cache)
        layer_labels.append(np.array([labels_this[c] for c in doc_labels]))
    return layer_labels


def generate_keyword_labels(movies, topics, topic_terms, layers, n_terms=4):
    layer_labels = []
    for doc_labels in layers:
        cluster_ids = sorted(c for c in set(doc_labels) if c != -1)
        labels_this = {-1: "Unlabelled"}
        for cid in cluster_ids:
            terms, _ = cluster_repr(movies, doc_labels, cid, topics, topic_terms)
            labels_this[cid] = " · ".join(terms[:n_terms]) or "Unlabelled"
        layer_labels.append(np.array([labels_this[c] for c in doc_labels]))
    return layer_labels


def build_interactive(movies, X, layer_labels, out_path: Path, title: str, sub_title: str):
    import datamapplot
    from umap import UMAP

    print("computing 2D UMAP for the plot...")
    umap_viz = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X)

    extra = pd.DataFrame({
        "title": movies["title"].tolist(),
        "year": movies["release_year"].fillna(0).astype(int).astype(str).tolist(),
        "genre": movies["genres"].map(lambda gs: gs[0] if len(gs) else "—").tolist(),
        "rating": movies["vote_average"].round(1).astype(str).tolist(),
        "tmdb_id": movies["id"].astype(str).tolist(),
    })
    hover_html = (
        '<div style="font-family: ui-sans-serif, system-ui; padding: 2px 4px;">'
        '<div style="font-weight:600">{title}</div>'
        '<div style="color:#9aa0a6; font-size: 0.9em;">{year} · {genre} · ★ {rating}</div>'
        '</div>'
    )
    print(f"rendering plot with {len(layer_labels)} label layers...")
    plot = datamapplot.create_interactive_plot(
        umap_viz,
        *layer_labels,
        hover_text=movies["title"].tolist(),
        extra_point_data=extra,
        hover_text_html_template=hover_html,
        enable_search=True,
        search_field="title",
        enable_topic_tree=True,
        cluster_layer_colormaps=True,
        histogram_data=movies["release_year"].fillna(0).astype(int),
        histogram_n_bins=30,
        histogram_settings={
            "histogram_title": "Release year",
            "histogram_bin_fill_color": "#ffb454",
            "histogram_bin_selected_fill_color": "#ffb454",
            "histogram_bin_unselected_fill_color": "#3a3f4d",
        },
        darkmode=True,
        on_click="window.open(`https://www.themoviedb.org/movie/{tmdb_id}`)",
        title=title,
        sub_title=sub_title,
    )
    plot.save(str(out_path))
    print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DATA_DIR / "movies.parquet")
    ap.add_argument("--embeddings", type=Path, default=EMB_DIR / "overviews_minilm.npy")
    ap.add_argument("--out", type=Path, default=REPO / "map.html")
    ap.add_argument("--cache", type=Path, default=DATA_DIR / "cluster_labels.json")
    ap.add_argument("--layers", nargs="+", type=int, default=[80, 40, 20, 10, 5])
    ap.add_argument("--min-cluster-size", type=int, default=20)
    ap.add_argument("--title", default="5,000 movies · thematic hierarchy")
    ap.add_argument("--sub-title", default="Hover for details, click to open on TMDB. Labels by Claude.")
    ap.add_argument("--classic", action="store_true",
                    help="Use raw c-TF-IDF keyword labels instead of Claude. Writes to map_classic.html by default.")
    args = ap.parse_args()

    if args.classic and args.out == REPO / "map.html":
        args.out = REPO / "map_classic.html"

    movies, X = load_data(args.input, args.embeddings)
    overviews = movies["overview"].tolist()

    topics, _, topic_emb, topic_terms = fit_bertopic(overviews, X, min_cluster_size=args.min_cluster_size)
    layers = build_hierarchy(topics, topic_emb, args.layers)

    if args.classic:
        layer_labels = generate_keyword_labels(movies, topics, topic_terms, layers)
        print(f"using classic c-TF-IDF keyword labels → {args.out.name}")
    else:
        layer_labels = generate_labels(movies, topics, topic_terms, layers, args.layers, args.cache)

    build_interactive(movies, X, layer_labels, args.out, args.title, args.sub_title)


if __name__ == "__main__":
    main()
