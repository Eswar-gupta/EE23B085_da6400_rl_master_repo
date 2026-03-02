import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os
from typing import List, Optional, Tuple


def _get_images_dir() -> str:
    """Returns the absolute path to the Grid_env_images directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Images", "Grid_env_images"
    )


def _load_image(filename: str, cell_size_px: int = 120) -> np.ndarray:
    """Load and resize an image from the Grid_env_images folder."""
    path = os.path.join(_get_images_dir(), filename)
    img = Image.open(path).convert("RGBA")
    img = img.resize((cell_size_px, cell_size_px), Image.LANCZOS)
    return np.array(img)


# Mapping from state label prefix to image file
_LABEL_TO_IMAGE = {
    "Lake":    "pond.jpg",
    "Fire":    "fire.jpg",
    "Smoke":   "smoke.png",
    "Boulder": "boulder.png",
}


def _parse_label(label: str) -> str:
    """Extract the cell-type prefix from a state label like 'Smoke_Water'."""
    return label.split("_")[0]


def generate_grid_image(
    S: List[str],
    path: Optional[List[Tuple[int, int]]] = None,
    points_of_interest: Optional[List[Tuple[int, int]]] = None,
    title: str = "Gridworld",
    figsize: Tuple[float, float] = (4, 4),
    save_path: Optional[str] = None,
    show: bool = True,
    dark_bg: bool = True,
    water_phase: int = 0,
):
    """
    Render the 5×5 Gridworld from a list of state labels.

    Parameters
    ----------
    S : List[str]
        State labels list (e.g. ["Lake_Empty", "Smoke_Water", …]).
    path : list of (x, y) tuples, optional
        A sequence of grid coordinates to draw as a red path with an
        arrow-head at the end.  Validated against the grid boundaries.
    points_of_interest : list of (x, y) tuples, optional
        Grid cells to mark with a location pin icon.
    title : str
        Figure title.
    figsize : tuple
        Matplotlib figure size.
    save_path : str, optional
        If given, the figure is saved to this file path.
    show : bool
        Whether to call plt.show().
    dark_bg : bool
        Use a dark background for the figure.
    water_phase : int (0 or 1)
        Which water-phase slice of the state space to visualise
        (0 = Empty, 1 = Water).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects.
    """
    # --- Derive grid dimensions from model ---
    grid_size = 5  # fixed for this MDP

    # --- Validate path ---
    if path is not None:
        for i, (px, py) in enumerate(path):
            if not (0 <= px < grid_size and 0 <= py < grid_size):
                raise ValueError(
                    f"Path point {i} = ({px}, {py}) is outside the "
                    f"{grid_size}x{grid_size} grid boundaries."
                )

    # --- Validate points of interest ---
    if points_of_interest is not None:
        for i, (px, py) in enumerate(points_of_interest):
            if not (0 <= px < grid_size and 0 <= py < grid_size):
                raise ValueError(
                    f"Point of interest {i} = ({px}, {py}) is outside the "
                    f"{grid_size}x{grid_size} grid boundaries."
                )

    # --- Pre-load images ---
    images = {}
    for label_key, fname in _LABEL_TO_IMAGE.items():
        try:
            images[label_key] = _load_image(fname)
        except FileNotFoundError:
            images[label_key] = None  # will fall back to coloured patch

    # Fallback colours matching the MATLAB reference
    _FALLBACK_COLOURS = {
        "Lake":    (0.3, 0.6, 1.0),
        "Fire":    (1.0, 0.0, 0.0),
        "Smoke":   (0.5, 0.5, 0.5),
        "Boulder": (0.3, 0.3, 0.3),
    }

    # --- Create figure ---
    bg = (0.15, 0.15, 0.15) if dark_bg else "white"
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor("white")

    # --- White grid background ---
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.5, -0.5), grid_size, grid_size,
        boxstyle="square,pad=0", facecolor="white", edgecolor="none", zorder=0
    ))

    # --- Place icons / fallback patches ---
    pad = 0.05  # small padding so image doesn't touch grid lines
    for y in range(grid_size):
        for x in range(grid_size):
            s_idx = x + 5 * y + 25 * water_phase
            label_prefix = _parse_label(S[s_idx])

            if label_prefix == "Free":
                continue  # nothing to draw

            img_arr = images.get(label_prefix)
            if img_arr is not None:
                extent = [x - 0.5 + pad, x + 0.5 - pad,
                          y - 0.5 + pad, y + 0.5 - pad]
                ax.imshow(img_arr, extent=extent, zorder=1, aspect='auto')
            else:
                colour = _FALLBACK_COLOURS.get(label_prefix, (0.8, 0.8, 0.8))
                ax.add_patch(mpatches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor=colour, edgecolor="none", zorder=1
                ))

    # --- Draw grid lines (on top of icons) ---
    for coord in np.arange(-0.5, grid_size + 0.5 + 1e-9, 1):
        ax.plot([coord, coord], [-0.5, grid_size - 0.5], 'k', linewidth=2, zorder=2)
        ax.plot([-0.5, grid_size - 0.5], [coord, coord], 'k', linewidth=2, zorder=2)

    # --- Draw path ---
    if path is not None and len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, 'r-', linewidth=4, zorder=3)

        # Arrow-head on the last segment
        dx = xs[-1] - xs[-2]
        dy = ys[-1] - ys[-2]
        ax.annotate(
            "", xy=(xs[-1], ys[-1]),
            xytext=(xs[-2], ys[-2]),
            arrowprops=dict(arrowstyle="-|>", color="red", lw=4),
            zorder=4,
        )
        # Red dot at the end
        ax.plot(xs[-1], ys[-1], 'ro', markersize=10, zorder=5)

    # --- Draw points of interest ---
    if points_of_interest is not None and len(points_of_interest) > 0:
        # Try to load a location pin image
        poi_img = None
        poi_path = os.path.join(_get_images_dir(), "location.png")
        if os.path.exists(poi_path):
            try:
                poi_img = _load_image("location.png", cell_size_px=100)
            except Exception:
                poi_img = None

        for (px, py) in points_of_interest:
            if poi_img is not None:
                extent = [px - 0.3, px + 0.3, py - 0.1, py + 0.45]
                ax.imshow(poi_img, extent=extent, zorder=6, aspect='auto')
            else:
                # Fallback: matplotlib marker that looks like a location pin
                ax.plot(px, py, marker='v', color='red', markersize=14,
                        markeredgecolor='darkred', markeredgewidth=1.5, zorder=6)

    # --- Axis settings ---
    ax.set_xlim(-0.7, grid_size - 0.3)
    ax.set_ylim(-0.7, grid_size - 0.3)
    ax.set_aspect("equal")
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.tick_params(colors="white" if dark_bg else "black", labelsize=10)
    ax.set_title(title, color="white" if dark_bg else "black", fontsize=14, pad=12)
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    if show:
        plt.show()

    return fig, ax
