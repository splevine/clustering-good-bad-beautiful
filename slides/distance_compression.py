"""
Distance compression — Manim animation for slides.html scene 10.

Pairwise distance distributions concentrate as dimensionality grows.
At 384-D (the MiniLM embedding space) nearest and farthest neighbors
sit in the same narrow band, so density-based clustering loses contrast.

Render:
    uv run manim -qh slides/distance_compression.py DistanceCompression

Output lands at media/videos/distance_compression/1080p60/DistanceCompression.mp4
"""

from __future__ import annotations

import numpy as np
from manim import (
    Axes,
    Create,
    DOWN,
    FadeIn,
    LEFT,
    RIGHT,
    Scene,
    Text,
    Transform,
    UP,
    VGroup,
    VMobject,
    Write,
    config,
)

config.background_color = "#0f1117"
ACCENT = "#ffb454"
MUTED = "#9aa0a6"
FG = "#e8e8ea"
COOL = "#6fb3d2"


def compute_distributions(dims: list[int], n_points: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    out = {}
    for d in dims:
        X = rng.standard_normal(size=(n_points, d))
        diff = X[:, None, :] - X[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        upper = dists[np.triu_indices(n_points, k=1)]
        normalized = upper / upper.max()
        hist, edges = np.histogram(normalized, bins=40, range=(0, 1), density=True)
        if hist.max() > 0:
            hist = hist / hist.max()
        centers = (edges[:-1] + edges[1:]) / 2

        np.fill_diagonal(dists, np.inf)
        nearest = dists.min(axis=1)
        np.fill_diagonal(dists, -np.inf)
        farthest = dists.max(axis=1)
        ratio = float(np.mean(nearest / farthest))
        out[d] = (centers, hist, ratio)
    return out


class DistanceCompression(Scene):
    def construct(self):
        dims = [2, 5, 10, 20, 50, 100, 200, 384]
        data = compute_distributions(dims)

        title = Text(
            "Distance compression",
            font_size=44,
            color=FG,
            weight="BOLD",
        ).to_edge(UP, buff=0.55)
        subtitle = Text(
            "Pairwise distance distributions as dimensionality grows",
            font_size=20,
            color=MUTED,
        ).next_to(title, DOWN, buff=0.18)

        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.3), run_time=1.0)
        self.wait(0.4)

        axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[0, 1.1, 0.25],
            x_length=9.0,
            y_length=3.6,
            tips=False,
            axis_config={"color": MUTED, "stroke_width": 1.5, "include_numbers": False},
        ).shift(DOWN * 0.4)
        x_label = Text(
            "normalized pairwise distance",
            font_size=18,
            color=MUTED,
        ).next_to(axes, DOWN, buff=0.25)

        self.play(Create(axes), FadeIn(x_label), run_time=0.7)

        def make_curve(d: int, color: str) -> VMobject:
            centers, density, _ = data[d]
            pts = [axes.c2p(x, y) for x, y in zip(centers, density)]
            curve = VMobject(stroke_color=color, stroke_width=4)
            curve.set_points_smoothly(pts)
            return curve

        def make_meta(d: int, color_d: str, color_r: str) -> VGroup:
            d_text = Text(f"d = {d}", font_size=34, color=color_d, weight="BOLD")
            r_text = Text(
                f"ratio = {data[d][2]:.2f}",
                font_size=22,
                color=color_r,
            ).next_to(d_text, DOWN, buff=0.18, aligned_edge=LEFT)
            grp = VGroup(d_text, r_text)
            grp.to_corner(UP + RIGHT, buff=0.6).shift(DOWN * 1.5 + LEFT * 0.2)
            return grp

        curve = make_curve(dims[0], COOL)
        meta = make_meta(dims[0], FG, MUTED)
        self.play(Create(curve), FadeIn(meta), run_time=0.8)
        self.wait(0.4)

        for d in dims[1:]:
            color = ACCENT if d == 384 else COOL
            new_curve = make_curve(d, color)
            new_meta = make_meta(d, color, color if d == 384 else MUTED)
            self.play(
                Transform(curve, new_curve),
                Transform(meta, new_meta),
                run_time=0.7,
            )

        self.wait(0.8)

        punchline = Text(
            "Density-based clustering has nothing to grip.",
            font_size=26,
            color=ACCENT,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(punchline, shift=UP * 0.3), run_time=0.9)
        self.wait(2.2)
