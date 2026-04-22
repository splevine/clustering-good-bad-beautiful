"""Build a 5-layer BERTopic hierarchy and render an interactive datamapplot.

Steps:
1. Load movies + cached overview embeddings (notebooks 01/02 populate these).
2. Fit BERTopic to get ~80 natural topics from the overviews.
3. Derive 5 hierarchy layers by agglomerative-clustering the topic embeddings
   at [80, 40, 20, 10, 5] cluster counts.
4. Label each cluster at each layer with Claude (falls back to c-TF-IDF if
   ANTHROPIC_API_KEY is missing).
5. Render the datamapplot with search, topic tree, histogram filter on
   release_year, a rich hover tooltip, and click-through to TMDB.

Run:
    uv run python scripts/label_hierarchy.py

Outputs:
    map.html                                    — the interactive plot
    data/cluster_labels.json                    — cached Claude labels
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
EMB_DIR = REPO / "embeddings"
DATA_DIR = REPO / "data"
LABELS_CACHE = DATA_DIR / "cluster_labels.json"
OUT_PATH = REPO / "map.html"
LAYER_SIZES = [80, 40, 20, 10, 5]


def load_data():
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    movies = movies[movies["overview"].str.len() > 30].reset_index(drop=True)
    emb_path = EMB_DIR / "overviews_minilm.npy"
    if not emb_path.exists():
        sys.exit(
            "error: overview embeddings not found. Run notebook 01 or 02 first "
            "to populate embeddings/overviews_minilm.npy."
        )
    X = np.load(emb_path)
    assert len(movies) == len(X), (len(movies), X.shape)
    return movies, X


def fit_bertopic(overviews, X):
    """Fit BERTopic and return (topics_array, topic_model, topic_embeddings, topic_terms)."""
    import hdbscan
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    print("fitting BERTopic...")
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=0),
        hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5, cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2)),
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(overviews, embeddings=X)
    topics = np.asarray(topics)
    # topic_embeddings_ has one row per topic including noise (-1 at index 0 in some BERTopic versions)
    topic_emb = np.asarray(topic_model.topic_embeddings_)
    topic_terms = {}
    for tid in sorted(set(topics)):
        if tid == -1:
            topic_terms[-1] = []
            continue
        topic_terms[tid] = [term for term, _ in topic_model.get_topic(tid)[:10]]
    n_real_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"  {n_real_topics} natural topics, {(topics == -1).sum()} noise docs ({(topics == -1).mean():.1%})")
    return topics, topic_model, topic_emb, topic_terms


def build_hierarchy(topics, topic_emb, layer_sizes):
    """Map natural topics to coarser cluster ids at each requested layer size.

    Returns a list of per-doc label arrays (one per layer, fine -> coarse) with
    noise points preserved as -1.
    """
    from sklearn.cluster import AgglomerativeClustering

    real_topic_ids = sorted(t for t in set(topics) if t != -1)
    # BERTopic's topic_embeddings_ row 0 is -1 (noise). Strip it for agg clustering.
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


def load_label_cache():
    if LABELS_CACHE.exists():
        return json.loads(LABELS_CACHE.read_text())
    return {}


def save_label_cache(cache):
    DATA_DIR.mkdir(exist_ok=True)
    LABELS_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def cluster_repr(movies, doc_labels, cluster_id, topics, topic_terms):
    """Return (top_terms, representative_titles) for the given coarse cluster."""
    mask = doc_labels == cluster_id
    sub = movies.loc[mask].sort_values("vote_count", ascending=False)
    titles = sub["title"].head(5).tolist()

    # Aggregate top terms from the original BERTopic topics that landed in this coarse cluster.
    topic_ids_in_cluster = [int(t) for t in np.unique(topics[mask]) if t != -1]
    term_pool = []
    for t in topic_ids_in_cluster[:5]:
        term_pool.extend(topic_terms.get(t, [])[:5])
    # Dedupe preserving order.
    seen = set()
    terms = []
    for term in term_pool:
        if term not in seen:
            seen.add(term)
            terms.append(term)
        if len(terms) >= 10:
            break
    return terms, titles


def label_with_claude(client, level_idx, level_size, cluster_id, terms, titles):
    """Ask Claude for a short, specific cluster label."""
    prompt = (
        f"You are labeling a movie-cluster in a hierarchical visualization. "
        f"This is hierarchy level {level_idx + 1} of 5 "
        f"(level 1 = finest-grained; level 5 = coarsest). The level has {level_size} clusters total.\n\n"
        f"Top distinctive words in this cluster: {', '.join(terms) if terms else '(no terms available)'}\n"
        f"5 representative films (by popularity): {', '.join(titles)}\n\n"
        "Give a concise 2-4 word label that captures the shared theme, tone, or setting. "
        "Avoid generic words like 'films', 'movies', 'cinema'. Prefer specific concepts "
        "(e.g., 'Coming-of-age drama', 'Heist thrillers', 'Cosmic horror', 'Royalty & palace intrigue'). "
        "Return ONLY the label — no quotes, no punctuation at the end, no explanation."
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=40,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    # Strip stray quotes or trailing punctuation.
    return text.strip('"\'' ).strip(". ")[:60] or "Unlabelled"


def generate_labels(movies, topics, topic_terms, layers, layer_sizes):
    """For each layer × cluster, produce a human-readable label. Cached on disk."""
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    cache = load_label_cache()

    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        print("using Claude for labels")
    else:
        client = None
        print("ANTHROPIC_API_KEY not set — falling back to c-TF-IDF labels")

    layer_labels = []
    for lvl_i, (doc_labels, size) in enumerate(zip(layers, layer_sizes)):
        cluster_ids = sorted(c for c in set(doc_labels) if c != -1)
        labels_this_layer = {-1: "Unlabelled"}
        iterator = tqdm(cluster_ids, desc=f"layer {lvl_i + 1}/{len(layer_sizes)} (size {size})")
        for cid in iterator:
            cache_key = f"L{lvl_i}-{size}-{cid}"
            if cache_key in cache:
                labels_this_layer[cid] = cache[cache_key]
                continue
            terms, titles = cluster_repr(movies, doc_labels, cid, topics, topic_terms)
            if client is not None:
                label = label_with_claude(client, lvl_i, size, cid, terms, titles)
            else:
                label = " · ".join(terms[:3]) or "Unlabelled"
            labels_this_layer[cid] = label
            cache[cache_key] = label
            save_label_cache(cache)
        mapped = np.array([labels_this_layer[c] for c in doc_labels])
        layer_labels.append(mapped)
    return layer_labels


def build_interactive(movies, X, layer_labels, args):
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

    # fine layer first, coarsest last (datamapplot convention)
    layer_args = list(layer_labels)
    print(f"rendering plot with {len(layer_args)} label layers...")
    plot = datamapplot.create_interactive_plot(
        umap_viz,
        *layer_args,
        hover_text=movies["title"].tolist(),
        extra_point_data=extra,
        hover_text_html_template=hover_html,
        enable_search=True,
        search_field="title",
        enable_topic_tree=True,
        cluster_layer_colormaps=True,
        histogram_data=movies["release_year"].fillna(0).astype(int),
        histogram_n_bins=30,
        darkmode=True,
        on_click="window.open(`https://www.themoviedb.org/movie/{tmdb_id}`)",
        title="5,000 movies · thematic hierarchy",
        sub_title="5 zoom levels, labels by Claude. Hover for details, click to open on TMDB.",
    )
    plot.save(str(OUT_PATH))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")


def generate_keyword_labels(movies, topics, topic_terms, layers, layer_sizes, n_terms=4):
    """Old-school c-TF-IDF labels: top terms joined with ' · '. No LLM."""
    layer_labels = []
    for lvl_i, doc_labels in enumerate(layers):
        cluster_ids = sorted(c for c in set(doc_labels) if c != -1)
        labels_this_layer = {-1: "Unlabelled"}
        for cid in cluster_ids:
            terms, _ = cluster_repr(movies, doc_labels, cid, topics, topic_terms)
            labels_this_layer[cid] = " · ".join(terms[:n_terms]) or "Unlabelled"
        layer_labels.append(np.array([labels_this_layer[c] for c in doc_labels]))
    return layer_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classic", action="store_true",
                    help="Use raw c-TF-IDF keyword labels instead of Claude. Writes to map_classic.html.")
    ap.add_argument("--filter", default="release_year", help="(reserved for future use)")
    args = ap.parse_args()

    movies, X = load_data()
    overviews = movies["overview"].tolist()

    topics, topic_model, topic_emb, topic_terms = fit_bertopic(overviews, X)
    layers = build_hierarchy(topics, topic_emb, LAYER_SIZES)

    global OUT_PATH
    if args.classic:
        layer_labels = generate_keyword_labels(movies, topics, topic_terms, layers, LAYER_SIZES)
        OUT_PATH = REPO / "map_classic.html"
        print(f"using classic c-TF-IDF keyword labels → {OUT_PATH.name}")
    else:
        layer_labels = generate_labels(movies, topics, topic_terms, layers, LAYER_SIZES)
    build_interactive(movies, X, layer_labels, args)


if __name__ == "__main__":
    main()
