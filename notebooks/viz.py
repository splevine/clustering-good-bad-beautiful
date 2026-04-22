"""Animation helpers for the UMAP parameter-sweep visualizations.

Adapted from an earlier MNIST-based notebook. The showpiece is
`create_umap_animation`, which runs UMAP at each value of a swept parameter,
tweens between consecutive embeddings, and returns a matplotlib animation
that morphs smoothly as the parameter changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

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
    embeddings: list[np.ndarray],
    param_values: list[Any],
    n_static_frames: int = 30,
    n_tween_frames: int = 15,
    palindrome: bool = False,
) -> tuple[list[np.ndarray], list[Any]]:
    """Build the frame sequence: static holds + tweens between embeddings.

    If `palindrome=True`, the sequence runs forward then back (mid-points
    not duplicated), so the final frame matches the first — making a looped
    `<video>` visually seamless.
    """
    if palindrome and len(embeddings) > 1:
        emb_seq = list(embeddings) + list(reversed(embeddings[:-1]))
        par_seq = list(param_values) + list(reversed(param_values[:-1]))
    else:
        emb_seq = list(embeddings)
        par_seq = list(param_values)

    frames: list[np.ndarray] = []
    frame_params: list[Any] = []
    for i, (e1, e2) in enumerate(zip(emb_seq[:-1], emb_seq[1:])):
        frames.extend([e1] * n_static_frames)
        frame_params.extend([par_seq[i]] * n_static_frames)
        frames.extend(list(tween(e1, e2, n_tween_frames)))
        frame_params.extend([par_seq[i]] * n_tween_frames)
    frames.extend([emb_seq[-1]] * n_static_frames)
    frame_params.extend([par_seq[-1]] * n_static_frames)
    return frames, frame_params


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
    palindrome: bool = False,
    **umap_params: Any,
) -> animation.FuncAnimation:
    """Produce a 3-D matplotlib animation sweeping one UMAP parameter.

    The axes rescale dynamically per-frame so the cluster geometry fills the
    viewport as it morphs. Set `palindrome=True` for a seamless back-and-forth
    loop (start → end → start).
    """
    embeddings = generate_umap_embeddings(data, param_name, param_values, n_components, **umap_params)
    frame_data, frame_params = generate_frame_data(
        embeddings, param_values, n_static_frames, n_tween_frames, palindrome=palindrome
    )

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

        pad = 0.1
        for setter, axis in ((ax.set_xlim, x), (ax.set_ylim, y), (ax.set_zlim, z)):
            lo, hi = axis.min(), axis.max()
            span = hi - lo or 1.0
            setter(lo - pad * span, hi + pad * span)

        scatter._offsets3d = (x, y, z)
        ax.view_init(elev=10.0, azim=(i * rotation_speed) % 360)
        title.set_text(f"UMAP · {param_name} = {frame_params[i]}")
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


_COSMOS_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    html, body { margin: 0; padding: 0; height: 100%; background: __BG__; color: #e8e8ea; font-family: ui-sans-serif, system-ui, sans-serif; overflow: hidden; }
    #info { position: absolute; top: 14px; left: 14px; font-size: 14px; color: #ffb454; pointer-events: none; }
    #status { position: absolute; bottom: 14px; left: 14px; font-size: 12px; color: #9aa0a6; pointer-events: none; }
    #tooltip { position: absolute; padding: 4px 8px; background: rgba(23,26,34,0.92); border: 1px solid #262a36; border-radius: 6px; font-size: 13px; pointer-events: none; display: none; z-index: 10; }
    canvas { display: block; }
  </style>
  <script type="importmap">
  { "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  } }
  </script>
</head>
<body>
  <div id="info">__TITLE__</div>
  <div id="status">loading textures…</div>
  <div id="tooltip"></div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const points = __POINTS_JSON__;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("__BG__");

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 40);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.4;

    // Auto-scale: center and normalize coords
    const xs = points.map(p => p.x), ys = points.map(p => p.y), zs = points.map(p => p.z);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const cz = (Math.min(...zs) + Math.max(...zs)) / 2;
    const span = Math.max(
      Math.max(...xs) - Math.min(...xs),
      Math.max(...ys) - Math.min(...ys),
      Math.max(...zs) - Math.min(...zs),
    );
    const scale = 40 / span;

    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin("anonymous");
    const sprites = [];
    let loaded = 0;
    const status = document.getElementById("status");

    points.forEach((p) => {
      loader.load(
        p.url,
        (texture) => {
          texture.colorSpace = THREE.SRGBColorSpace;
          const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
          const sprite = new THREE.Sprite(material);
          sprite.position.set(
            (p.x - cx) * scale,
            (p.y - cy) * scale,
            (p.z - cz) * scale,
          );
          sprite.scale.set(0.9, 1.35, 1);
          sprite.userData = { title: p.title };
          scene.add(sprite);
          sprites.push(sprite);
          loaded += 1;
          if (loaded % 50 === 0 || loaded === points.length) {
            status.textContent = `loaded ${loaded} / ${points.length}`;
            if (loaded === points.length) setTimeout(() => (status.style.display = "none"), 1500);
          }
        },
        undefined,
        () => { loaded += 1; }  // count errors so the status terminates
      );
    });

    // Hover
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const tooltip = document.getElementById("tooltip");
    window.addEventListener("mousemove", (e) => {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(sprites);
      if (hits.length) {
        tooltip.textContent = hits[0].object.userData.title;
        tooltip.style.left = (e.clientX + 12) + "px";
        tooltip.style.top = (e.clientY + 12) + "px";
        tooltip.style.display = "block";
      } else {
        tooltip.style.display = "none";
      }
    });

    // Pause auto-rotate on user interaction
    controls.addEventListener("start", () => { controls.autoRotate = false; });

    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
"""


def render_poster_cosmos(
    coords_3d: np.ndarray,
    urls: Sequence[str],
    titles: Sequence[str],
    out_path: str | Path,
    title: str = "3D poster constellation",
    background: str = "#0f1117",
) -> Path:
    """Write a self-contained Three.js HTML where each point is a poster sprite.

    Loads images lazily from the given URLs (typically TMDB's CDN). Includes
    orbit/zoom/pan controls, auto-rotate, and hover tooltips with titles.
    """
    assert coords_3d.shape == (len(urls), 3) == (len(titles), 3), "coords/urls/titles lengths must match"
    points = [
        {"x": float(x), "y": float(y), "z": float(z), "url": url, "title": t}
        for (x, y, z), url, t in zip(coords_3d, urls, titles)
    ]
    html = (
        _COSMOS_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__BG__", background)
        .replace("__POINTS_JSON__", json.dumps(points, separators=(",", ":")))
    )
    out = Path(out_path)
    out.write_text(html, encoding="utf-8")
    return out

