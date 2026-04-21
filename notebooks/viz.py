"""Animation helpers for the UMAP parameter-sweep visualizations.

Adapted from an earlier MNIST-based notebook. The showpiece is
`create_umap_animation`, which runs UMAP at each value of a swept parameter,
tweens between consecutive embeddings, and returns a matplotlib animation
that morphs smoothly as the parameter changes.
"""

from __future__ import annotations

from typing import Any, Iterator

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from umap import UMAP


def create_3d_scatter(X: np.ndarray, labels: np.ndarray, title: str) -> go.Figure:
    """Rotatable Plotly 3D scatter of a 3-D embedding."""
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode="markers",
                marker=dict(size=2, color=labels, opacity=0.8),
            )
        ]
    )
    fig.update_layout(title=title)
    return fig


def tween(e1: np.ndarray, e2: np.ndarray, n_frames: int = 15) -> Iterator[np.ndarray]:
    """Linearly interpolate `n_frames` steps from embedding `e1` to `e2`."""
    for i in range(n_frames):
        alpha = i / float(n_frames - 1)
        yield (1 - alpha) * e1 + alpha * e2


def generate_umap_embeddings(
    data: np.ndarray,
    param_name: str,
    param_values: list[Any],
    n_components: int = 3,
    **umap_params: Any,
) -> list[np.ndarray]:
    """Run UMAP once per value of `param_name`, returning the list of embeddings."""
    embeddings = []
    for value in tqdm(param_values, desc=f"UMAP sweep: {param_name}"):
        umap_params[param_name] = value
        reducer = UMAP(n_components=n_components, **umap_params)
        embeddings.append(reducer.fit_transform(data))
    return embeddings


def generate_frame_data(
    embeddings: list[np.ndarray], n_static_frames: int = 30, n_tween_frames: int = 15
) -> list[np.ndarray]:
    """Build the full frame sequence: static holds + tweens between embeddings."""
    frames: list[np.ndarray] = []
    for e1, e2 in zip(embeddings[:-1], embeddings[1:]):
        frames.extend([e1] * n_static_frames)
        frames.extend(list(tween(e1, e2, n_tween_frames)))
    frames.extend([embeddings[-1]] * n_static_frames)
    return frames


def create_umap_animation(
    data: np.ndarray,
    labels: np.ndarray,
    param_name: str,
    param_values: list[Any],
    n_components: int = 3,
    n_static_frames: int = 20,
    n_tween_frames: int = 10,
    rotation_speed: float = 3.0,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "tab20",
    **umap_params: Any,
) -> animation.FuncAnimation:
    """Produce a 3-D matplotlib animation sweeping one UMAP parameter.

    The axes rescale dynamically per-frame so the cluster geometry fills the
    viewport as it morphs.
    """
    embeddings = generate_umap_embeddings(data, param_name, param_values, n_components, **umap_params)
    frame_data = generate_frame_data(embeddings, n_static_frames, n_tween_frames)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        frame_data[0][:, 0],
        frame_data[0][:, 1],
        frame_data[0][:, 2],
        s=5,
        c=labels,
        cmap=cmap,
        animated=True,
    )

    title = ax.set_title("", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    def init():
        return (scatter,)

    def animate(i: int):
        frame = frame_data[i]
        x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]

        # Rescale axes so the geometry stays filled as the parameter sweeps.
        pad = 0.1
        for setter, axis in ((ax.set_xlim, x), (ax.set_ylim, y), (ax.set_zlim, z)):
            lo, hi = axis.min(), axis.max()
            span = hi - lo or 1.0
            setter(lo - pad * span, hi + pad * span)

        scatter._offsets3d = (x, y, z)
        ax.view_init(elev=10.0, azim=(i * rotation_speed) % 360)

        param_index = min(i // (n_static_frames + n_tween_frames), len(param_values) - 1)
        title.set_text(f"UMAP · {param_name} = {param_values[param_index]}")
        return (scatter,)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(frame_data),
        interval=50,
        blit=False,
    )
    plt.close(fig)
    return anim
