import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
from PIL import Image as PILImage
from typing import List, Tuple, Optional


def _fig_to_array(fig: Figure, dpi: int = 150) -> np.ndarray:
    """Rasterize a matplotlib Figure to an RGBA numpy array."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = np.array(PILImage.open(buf).convert('RGBA'))
    buf.close()
    plt.close(fig)
    return img


def merge_figures(
    figures: List[Tuple[Figure, Tuple[int, int]]],
    grid: Tuple[int, int] = (1, 2),
    grand_title: Optional[str] = None,
    grand_title_color: str = 'white',
    figsize: Optional[Tuple[float, float]] = None,
    bg_color: str = '#1a1a2e',
    dpi: int = 150,
    show: bool = True,
    wspace: float = 0.05,
    hspace: float = 0.08,
) -> Tuple[Figure, np.ndarray]:
    """
    Merge multiple matplotlib figures into a single figure by rasterizing
    each one and placing it in a grid of subplots.

    Parameters
    ----------
    figures : list of (fig, (row, col))
        Each element is a tuple of:
          - fig : matplotlib Figure object (will be rasterized and closed)
          - (row, col) : 0-indexed position in the output grid
    grid : (n_rows, n_cols)
        Layout of the output grid, e.g. (1, 2), (2, 2), (2, 3).
    grand_title : str, optional
        A super-title displayed above all subplots.
    grand_title_color : str
        Color of the grand title text.
    figsize : (width, height), optional
        Size of the merged figure. Defaults to (5*n_cols, 5*n_rows).
    bg_color : str
        Background color matching the dark theme ('#1a1a2e').
    dpi : int
        Resolution used when rasterizing each sub-figure.
    show : bool
        Whether to call plt.show() on the merged figure.
    wspace : float
        Horizontal spacing between subplots (fraction of subplot width).
    hspace : float
        Vertical spacing between subplots (fraction of subplot height).

    Returns
    -------
    merged_fig : matplotlib Figure
    axes : np.ndarray of Axes

    Example
    -------
    >>> from core.Plot_generators.value_fuction_visuvalizer import state_value_fuction_visuvalizer
    >>> from core.Plot_generators.Grid_path_generator import generate_grid_image
    >>> from core.Plot_generators.figuer_merger import merge_figures
    >>>
    >>> # Generate individual figures (show=False)
    >>> fig_v, _ = state_value_fuction_visuvalizer(V, S, water=False)
    >>> fig_g, _ = generate_grid_image(S, path=my_path, show=False, water_phase=0)
    >>>
    >>> # Merge side-by-side: fig_v at (0,0), fig_g at (0,1)
    >>> merged, axes = merge_figures(
    ...     figures=[(fig_v, (0, 0)), (fig_g, (0, 1))],
    ...     grid=(1, 2),
    ...     grand_title='Phase 1: Without Water',
    ...     grand_title_color='#FF9800',
    ...     figsize=(12, 5),
    ... )
    >>>
    >>> # 2×2 example
    >>> merged, axes = merge_figures(
    ...     figures=[(fig1, (0,0)), (fig2, (0,1)),
    ...              (fig3, (1,0)), (fig4, (1,1))],
    ...     grid=(2, 2),
    ...     grand_title='All Phases',
    ...     figsize=(12, 10),
    ... )
    """
    n_rows, n_cols = grid

    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)

    # Rasterize all source figures into image arrays
    images = {}
    for fig, pos in figures:
        images[pos] = _fig_to_array(fig, dpi=dpi)

    # Create the merged figure
    merged_fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, facecolor=bg_color,
        squeeze=False,
    )

    if grand_title:
        merged_fig.suptitle(grand_title, fontsize=16, fontweight='bold',
                            color=grand_title_color)

    # Place images or hide empty cells
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_facecolor(bg_color)
            ax.axis('off')

            if (r, c) in images:
                ax.imshow(images[(r, c)])

    plt.subplots_adjust(
        wspace=wspace, hspace=hspace,
        left=0.02, right=0.98,
        top=0.92 if grand_title else 0.98,
        bottom=0.02,
    )

    if show:
        plt.show()

    return merged_fig, axes
