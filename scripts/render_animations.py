"""Render the UMAP parameter-sweep animations as both MP4 and GIF.

For each modality (text overviews and CLIP poster embeddings) and each
parameter (n_neighbors and min_dist), we produce:

- animation_<modality>[_<param>].mp4   — crisp, small, for web <video>
- animation_<modality>[_<param>].gif   — universal, for slides (Keynote/PowerPoint)

n_neighbors sweeps keep the default filename (no suffix) since they're
the primary "UMAP knob" story. min_dist gets a suffix.

Run:
    uv run python scripts/render_animations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "notebooks"))
import viz  # noqa: E402


EMB_DIR = REPO / "embeddings"
DATA_DIR = REPO / "data"

TOP_GENRES = 8
N_SAMPLE = 2000
SEED = 0


def genre_codes(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    primary = df["genres"].map(lambda gs: gs[0] if len(gs) else "Other")
    top = primary.value_counts().head(TOP_GENRES).index.tolist()
    collapsed = primary.where(primary.isin(top), "Other")
    labels = sorted(collapsed.unique())
    code_map = {label: i for i, label in enumerate(labels)}
    return collapsed.map(code_map).to_numpy(), labels


def stratified_sample(codes: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    per_class = max(1, n // len(np.unique(codes)))
    picks = []
    for c in np.unique(codes):
        idx = np.where(codes == c)[0]
        picks.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    picks = np.array(picks)
    rng.shuffle(picks)
    return picks[:n]


def load_all():
    text_emb = np.load(EMB_DIR / "overviews_minilm.npy")
    clip_emb = np.load(EMB_DIR / "posters_clip_b32.npy")

    all_movies = pd.read_parquet(DATA_DIR / "movies.parquet")

    text_movies = all_movies[all_movies["overview"].str.len() > 30].reset_index(drop=True)
    assert len(text_movies) == len(text_emb), (len(text_movies), len(text_emb))

    poster_dir = DATA_DIR / "posters"
    poster_movies = all_movies.copy()
    poster_movies["poster_file"] = poster_movies["id"].map(lambda i: poster_dir / f"{i}.jpg")
    poster_movies = poster_movies[poster_movies["poster_file"].map(lambda p: p.exists())].reset_index(drop=True)
    assert len(poster_movies) == len(clip_emb), (len(poster_movies), len(clip_emb))

    rng = np.random.default_rng(SEED)
    text_codes, _ = genre_codes(text_movies)
    poster_codes, _ = genre_codes(poster_movies)
    text_idx = stratified_sample(text_codes, N_SAMPLE, rng)
    poster_idx = stratified_sample(poster_codes, N_SAMPLE, rng)

    return {
        "text": (text_emb[text_idx], text_codes[text_idx]),
        "posters": (clip_emb[poster_idx], poster_codes[poster_idx]),
    }


SWEEPS = [
    # (modality, param_name, param_values, fixed_umap_kwargs, suffix)
    ("text",    "n_neighbors", [2, 3, 5, 10, 15, 30, 60],       {"min_dist": 0.1},     ""),
    ("text",    "min_dist",    [0.0, 0.05, 0.1, 0.25, 0.5, 0.8], {"n_neighbors": 15}, "_min_dist"),
    ("posters", "n_neighbors", [2, 3, 5, 10, 15, 30, 60],       {"min_dist": 0.1},     ""),
    ("posters", "min_dist",    [0.0, 0.05, 0.1, 0.25, 0.5, 0.8], {"n_neighbors": 15}, "_min_dist"),
]


def main():
    datasets = load_all()
    for modality, param_name, param_values, fixed, suffix in SWEEPS:
        data, labels = datasets[modality]
        print(f"\n=== {modality} / {param_name} ===")
        anim = viz.create_umap_animation(
            data=data,
            labels=labels,
            param_name=param_name,
            param_values=param_values,
            n_components=3,
            n_static_frames=15,
            n_tween_frames=10,
            rotations_per_cycle=1.0,
            palindrome=True,
            metric="cosine",
            random_state=0,
            **fixed,
        )
        base = REPO / f"animation_{modality}{suffix}"
        mp4 = base.with_suffix(".mp4")
        gif = base.with_suffix(".gif")

        anim.save(str(mp4), writer="ffmpeg", fps=20, dpi=100, bitrate=2400)
        print(f"  wrote {mp4.name} ({mp4.stat().st_size // 1024} KB)")

        anim.save(str(gif), writer="pillow", fps=15)
        print(f"  wrote {gif.name} ({gif.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
