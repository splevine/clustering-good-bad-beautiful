# Clustering: The Good, The Bad and The Beautiful

Companion repository for the [ODSC AI East 2026](https://odsc.com/east/) talk by [Seth Levine](https://github.com/splevine) — a 30-minute virtual session on when clustering works, when it doesn't, and how to build a modern workflow that turns messy, high-dimensional data into useful and explainable insights.

> Clustering is one of the most widely used and most frequently misapplied techniques in machine learning.

## What's in here

| Notebook | Theme | Run it |
| --- | --- | --- |
| [`notebooks/01_the_bad.ipynb`](notebooks/01_the_bad.ipynb) | Why naive k-means falls apart on real embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/splevine/clustering-good-bad-beautiful/blob/main/notebooks/01_the_bad.ipynb) |
| [`notebooks/02_the_good.ipynb`](notebooks/02_the_good.ipynb) | A modern pipeline: UMAP → HDBSCAN → BERTopic | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/splevine/clustering-good-bad-beautiful/blob/main/notebooks/02_the_good.ipynb) |
| [`notebooks/03_the_beautiful.ipynb`](notebooks/03_the_beautiful.ipynb) | Clustering images with CLIP + EVoC + datamapplot | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/splevine/clustering-good-bad-beautiful/blob/main/notebooks/03_the_beautiful.ipynb) |

## Tools on display

`huggingface` · `transformers` · `sentence-transformers` · `BERTopic` · `UMAP` · `HDBSCAN` · `EVoC` · `datamapplot`

## Running locally

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

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

The interactive datamapplot visualizations live at **https://splevine.github.io/clustering-good-bad-beautiful/** (published via GitHub Pages).

## License

[MIT](LICENSE)
