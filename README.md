# Clustering: The Good, The Bad and The Beautiful

Companion repository for the [ODSC AI East 2026](https://odsc.com/east/) talk by [Seth Levine](https://github.com/splevine) — a 30-minute virtual session on when clustering works, when it doesn't, and how to build a modern workflow that turns messy, high-dimensional data into useful and explainable insights.

> Clustering is one of the most widely used and most frequently misapplied techniques in machine learning.

The talk title nods to *The Bad and the Beautiful* (1952), so the running dataset is **the top 5,000 movies** by TMDB vote count — overviews for text clustering, posters for image clustering, and the two compared side by side.

## What's in here

| Notebook | Theme | Run it |
| --- | --- | --- |
| [`notebooks/01_the_good.ipynb`](notebooks/01_the_good.ipynb) | A modern pipeline: UMAP → HDBSCAN → BERTopic on movie overviews | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/splevine/clustering-good-bad-beautiful/blob/main/notebooks/01_the_good.ipynb) |
| [`notebooks/02_the_bad.ipynb`](notebooks/02_the_bad.ipynb) | Failure modes that would have bitten each stage of the pipeline above | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/splevine/clustering-good-bad-beautiful/blob/main/notebooks/02_the_bad.ipynb) |
| [`notebooks/03_the_beautiful.ipynb`](notebooks/03_the_beautiful.ipynb) | CLIP + EVoC on posters, with an interactive datamapplot of thumbnails | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/splevine/clustering-good-bad-beautiful/blob/main/notebooks/03_the_beautiful.ipynb) |

## Tools on display

`huggingface` · `transformers` · `sentence-transformers` · `BERTopic` · `UMAP` · `HDBSCAN` · `EVoC` · `datamapplot`

## Getting the data

The repo does **not** ship posters (copyrighted) or bulky embeddings. Pull them locally:

```bash
# 1. TMDB token — free at https://www.themoviedb.org/settings/api
cp .env.example .env   # then paste your v4 Read Access Token into TMDB_BEARER_TOKEN

# 2. Metadata -> data/movies.parquet  (a few seconds, ~250 API calls)
uv run python scripts/fetch_movies.py

# 3. Posters -> data/posters/  (a few minutes, ~150 MB at w342)
uv run python scripts/fetch_posters.py
```

Both scripts are resumable: re-running skips work already done.

## Running locally

This project uses [`uv`](https://github.com/astral-sh/uv).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/splevine/clustering-good-bad-beautiful.git
cd clustering-good-bad-beautiful
uv sync

# Launch Jupyter
uv run jupyter lab
```

## Companion site

Interactive datamapplots live at **https://splevine.github.io/clustering-good-bad-beautiful/** (GitHub Pages).

## License & attribution

Code: [MIT](LICENSE). Movie metadata and poster images sourced from [TMDB](https://www.themoviedb.org/).

> This application uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise approved by TMDB.
