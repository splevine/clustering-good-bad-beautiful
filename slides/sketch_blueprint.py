"""
sketch_blueprint.py — "Sketch → Blueprint."

Manim animation for the closing of the Distinction section. A loose,
hand-drawn-style sketch on the left morphs into a clean, gridded blueprint
on the right. Lands the analogy: clustering = sketch, classification =
blueprint. Drafty exploration vs. precise specification.

Render:
    uv run manim -qh slides/sketch_blueprint.py SketchBlueprint
"""

from __future__ import annotations

import numpy as np
from manim import (
    Create,
    DOWN,
    FadeIn,
    LEFT,
    Line,
    Rectangle,
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


def sketch_blob(center, radius=1.1, n_points=60, jitter=0.18, seed=0):
    """A wobbly closed curve — looks hand-drawn."""
    rng = np.random.default_rng(seed)
    cx, cy = center
    pts = []
    for i in range(n_points):
        theta = 2 * np.pi * i / n_points
        r = radius + rng.normal(0, jitter)
        pts.append([cx + r * np.cos(theta), cy + r * np.sin(theta), 0])
    pts.append(pts[0])  # close the loop
    curve = VMobject(stroke_color=COOL, stroke_width=3)
    curve.set_points_smoothly(pts)
    return curve


def sketch_squiggle(start, end, n_points=24, jitter=0.08, seed=1):
    """A wobbly line from start to end."""
    rng = np.random.default_rng(seed)
    pts = []
    for t in np.linspace(0, 1, n_points):
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        pts.append([x + rng.normal(0, jitter), y + rng.normal(0, jitter), 0])
    line = VMobject(stroke_color=COOL, stroke_width=2.5)
    line.set_points_smoothly(pts)
    return line


def blueprint_grid(center, width=2.4, height=1.8, divisions=4, color=ACCENT):
    """A clean rectangular grid — looks engineered."""
    cx, cy = center
    rect = Rectangle(width=width, height=height, color=color, stroke_width=2.5).move_to([cx, cy, 0])
    lines = VGroup()
    for i in range(1, divisions):
        x = cx - width / 2 + i * width / divisions
        lines.add(Line([x, cy - height / 2, 0], [x, cy + height / 2, 0], color=color, stroke_width=1.2, stroke_opacity=0.55))
    for i in range(1, 3):
        y = cy - height / 2 + i * height / 3
        lines.add(Line([cx - width / 2, y, 0], [cx + width / 2, y, 0], color=color, stroke_width=1.2, stroke_opacity=0.55))
    return VGroup(rect, lines)


class SketchBlueprint(Scene):
    def construct(self):
        # ----- Sketch side (left) -----
        sketch_label = Text(
            "SKETCH",
            font_size=26, color=COOL, weight="BOLD",
        ).move_to([-3.5, 2.6, 0])
        sketch_sub = Text(
            "clustering — explore freely",
            font_size=18, color=MUTED,
        ).next_to(sketch_label, DOWN, buff=0.15)

        self.play(Write(sketch_label), FadeIn(sketch_sub), run_time=0.8)

        # Build a sketchy diagram: blob + a couple of interior squiggles
        blob = sketch_blob(center=(-3.5, 0.0), radius=1.3, jitter=0.2, seed=3)
        inner_a = sketch_blob(center=(-3.9, 0.2), radius=0.45, jitter=0.12, seed=5)
        inner_b = sketch_blob(center=(-3.0, -0.3), radius=0.55, jitter=0.13, seed=7)
        squiggle = sketch_squiggle(start=[-4.5, -0.8, 0], end=[-2.6, 0.7, 0], jitter=0.09, seed=9)

        sketch_group = VGroup(blob, inner_a, inner_b, squiggle)
        self.play(Create(blob), run_time=0.9)
        self.play(Create(inner_a), Create(inner_b), run_time=0.7)
        self.play(Create(squiggle), run_time=0.6)
        self.wait(0.3)

        # ----- Blueprint side (right) -----
        blueprint_label = Text(
            "BLUEPRINT",
            font_size=26, color=ACCENT, weight="BOLD",
        ).move_to([3.5, 2.6, 0])
        blueprint_sub = Text(
            "classification — define precisely",
            font_size=18, color=MUTED,
        ).next_to(blueprint_label, DOWN, buff=0.15)

        self.play(Write(blueprint_label), FadeIn(blueprint_sub), run_time=0.8)

        bp = blueprint_grid(center=(3.5, 0.0), width=2.6, height=2.0, divisions=4, color=ACCENT)
        # Reveal grid stage by stage for craftsmanship feel
        self.play(Create(bp[0]), run_time=0.7)
        self.play(Create(bp[1]), run_time=0.9)

        # Add a few "bolt" dots on grid corners for engineered feel
        corners = [
            (3.5 - 1.3, 0.0 - 1.0), (3.5 + 1.3, 0.0 - 1.0),
            (3.5 - 1.3, 0.0 + 1.0), (3.5 + 1.3, 0.0 + 1.0),
        ]
        bolts = VGroup(*[
            VMobject(stroke_color=ACCENT, fill_color=ACCENT, fill_opacity=1, stroke_width=0).set_points_as_corners([
                [cx - 0.05, cy, 0], [cx, cy + 0.05, 0], [cx + 0.05, cy, 0], [cx, cy - 0.05, 0], [cx - 0.05, cy, 0],
            ])
            for cx, cy in corners
        ])
        self.play(FadeIn(bolts), run_time=0.4)
        self.wait(0.4)

        # Divider arrow between the two sides
        divider = VMobject(stroke_color=MUTED, stroke_width=1, stroke_opacity=0.4)
        divider.set_points_as_corners([[0, -1.6, 0], [0, 2.0, 0]])
        self.play(Create(divider), run_time=0.3)

        # ----- Punchline -----
        punchline = Text(
            "Clustering is sketching. Classification is building.",
            font_size=24, color=ACCENT,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(punchline, shift=UP * 0.3), run_time=0.9)
        self.wait(2.4)
