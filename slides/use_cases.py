"""
use_cases.py — "Same data. Two purposes."

Manim animation for slides.html: the exploration vs labeling distinction.
Same dot cloud forks into two paths — top path re-clusters three different
ways (instability as signal); bottom path clusters once and flows into a
classifier (consistency matters).

Render:
    uv run manim -qh slides/use_cases.py UseCasesTwoPaths
"""

from __future__ import annotations

import numpy as np
from manim import (
    Arrow,
    Create,
    DOWN,
    Dot,
    FadeIn,
    LEFT,
    Rectangle,
    RIGHT,
    Scene,
    Text,
    Transform,
    UP,
    VGroup,
    Write,
    config,
)

config.background_color = "#0f1117"
ACCENT = "#ffb454"
MUTED = "#9aa0a6"
FG = "#e8e8ea"
COOL = "#6fb3d2"
WARM = "#d08770"
GREEN = "#a3be8c"
LILAC = "#b48ead"


def dot_cloud(n: int = 30, seed: int = 42, scale: float = 0.5):
    rng = np.random.default_rng(seed)
    centers = [(-0.7, 0.4), (0.6, -0.2), (0.0, 0.6)]
    pts, cls = [], []
    per = n // 3
    for ci, (cx, cy) in enumerate(centers):
        for _ in range(per):
            pts.append((cx + rng.normal(0, 0.32) * scale, cy + rng.normal(0, 0.32) * scale))
            cls.append(ci)
    return pts, cls


def make_dots(positions, cluster_assignments, palette, radius=0.06):
    return VGroup(*[
        Dot(point=[x, y, 0], radius=radius, color=palette[c % len(palette)])
        for (x, y), c in zip(positions, cluster_assignments)
    ])


class UseCasesTwoPaths(Scene):
    def construct(self):
        # Title
        title = Text(
            "Same data. Two purposes.",
            font_size=44, color=FG, weight="BOLD",
        ).to_edge(UP, buff=0.55)
        self.play(Write(title), run_time=0.9)
        self.wait(0.2)

        # Central dataset
        ds_pts, _ = dot_cloud(30, seed=11, scale=0.55)
        dataset = make_dots(ds_pts, [0] * len(ds_pts), [MUTED], radius=0.07)
        dataset.move_to([0, 2.0, 0])
        ds_label = Text("dataset", font_size=18, color=MUTED).next_to(dataset, DOWN, buff=0.18)
        self.play(Create(dataset), FadeIn(ds_label), run_time=0.7)

        # Fork arrows
        left_anchor = [-3.6, 0.55, 0]
        right_anchor = [3.6, 0.55, 0]
        arr_left = Arrow(
            start=dataset.get_bottom() + np.array([-0.25, 0, 0]),
            end=left_anchor,
            color=ACCENT, stroke_width=3, buff=0.25,
            max_tip_length_to_length_ratio=0.1,
        )
        arr_right = Arrow(
            start=dataset.get_bottom() + np.array([0.25, 0, 0]),
            end=right_anchor,
            color=ACCENT, stroke_width=3, buff=0.25,
            max_tip_length_to_length_ratio=0.1,
        )
        self.play(Create(arr_left), Create(arr_right), run_time=0.6)

        # ===== Path A: Exploration =====
        a_label = Text(
            "EXPLORATION",
            font_size=22, color=ACCENT, weight="BOLD",
        ).move_to([-3.6, 0.05, 0])
        self.play(FadeIn(a_label), run_time=0.4)

        a_pts, a_clusters = dot_cloud(27, seed=42)
        palettes_a = [
            [COOL, WARM, GREEN],
            [WARM, GREEN, LILAC],
            [GREEN, ACCENT, COOL],
        ]
        a_dots = make_dots(a_pts, a_clusters, palettes_a[0], radius=0.06)
        a_dots.move_to([-3.6, -1.0, 0])

        a_sub = Text(
            "instability is signal",
            font_size=15, color=MUTED,
        ).move_to([-3.6, -2.05, 0])

        self.play(Create(a_dots), FadeIn(a_sub), run_time=0.7)
        self.wait(0.4)

        # Re-cluster the same dots three times
        rng = np.random.default_rng(7)
        for palette in palettes_a[1:] + [palettes_a[0]]:
            new_assignments = rng.integers(0, 3, size=len(a_pts)).tolist()
            new_dots = make_dots(a_pts, new_assignments, palette, radius=0.06)
            new_dots.move_to([-3.6, -1.0, 0])
            self.play(Transform(a_dots, new_dots), run_time=0.55)

        # ===== Path B: Labeling =====
        b_label = Text(
            "LABELING",
            font_size=22, color=ACCENT, weight="BOLD",
        ).move_to([3.6, 0.05, 0])
        self.play(FadeIn(b_label), run_time=0.4)

        b_pts, b_clusters = dot_cloud(27, seed=42)
        b_dots = make_dots(b_pts, b_clusters, [COOL, WARM, GREEN], radius=0.055)
        b_dots.move_to([1.8, -1.0, 0]).scale(0.85)

        self.play(Create(b_dots), run_time=0.6)

        classifier = Rectangle(width=1.55, height=0.65, color=ACCENT, stroke_width=2.5).move_to([4.0, -1.0, 0])
        clf_text = Text("classifier", font_size=15, color=FG).move_to(classifier.get_center())
        clf_group = VGroup(classifier, clf_text)

        arrow_to_clf = Arrow(
            start=b_dots.get_right(),
            end=classifier.get_left(),
            color=ACCENT, stroke_width=2.5, buff=0.12,
            max_tip_length_to_length_ratio=0.2,
        )
        self.play(Create(arrow_to_clf), run_time=0.4)
        self.play(FadeIn(clf_group), run_time=0.5)

        production = Rectangle(width=1.55, height=0.55, color=GREEN, stroke_width=2.5).move_to([5.6, -1.0, 0])
        prod_text = Text("labels", font_size=14, color=FG).move_to(production.get_center())
        prod_group = VGroup(production, prod_text)

        arrow_to_prod = Arrow(
            start=classifier.get_right(),
            end=production.get_left(),
            color=ACCENT, stroke_width=2.5, buff=0.1,
            max_tip_length_to_length_ratio=0.22,
        )
        self.play(Create(arrow_to_prod), FadeIn(prod_group), run_time=0.7)

        b_sub = Text(
            "consistency matters",
            font_size=15, color=MUTED,
        ).move_to([3.6, -2.05, 0])
        self.play(FadeIn(b_sub), run_time=0.4)
        self.wait(0.6)

        # Punchline
        punchline = Text(
            "Clustering is sketching. Classification is building.",
            font_size=24, color=ACCENT,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(punchline, shift=UP * 0.3), run_time=0.9)
        self.wait(2.4)
