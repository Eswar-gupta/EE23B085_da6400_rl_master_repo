import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, Optional


def state_value_fuction_visuvalizer(
    V: np.ndarray,
    S: List[str],
    water: bool,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 5),
):
    """
    Visualise the state value function V on a 5×5 grid for one water phase.

    Parameters
    ----------
    V : np.ndarray, shape (50,)
        Full value vector over all 50 states.
    S : List[str]
        State labels (length 50).
    water : bool
        True  → show states 25-49 (with water, w=1).
        False → show states 0-24  (without water, w=0).
    title : str, optional
        Custom title. Defaults to 'V*(s) — With Water' / 'V*(s) — Without Water'.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    w = 1 if water else 0

    if title is None:
        title = '$V^*(s)$ — With Water 💧' if water else '$V^*(s)$ — Without Water'

    title_color = '#2196F3' if water else '#FF9800'

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'value', ['#b71c1c', '#e53935', '#ef9a9a',
                  '#e0e0e0',
                  '#a5d6a7', '#43a047', '#1b5e20'])

    vmin, vmax = np.min(V), np.max(V)

    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')

    # Build 5×5 grid (row 0 = top = y=4)
    grid = np.zeros((5, 5))
    for y in range(5):
        for x in range(5):
            grid[4 - y, x] = V[x + 5 * y + 25 * w]

    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_title(title, fontsize=13, fontweight='bold', color=title_color, pad=8)

    # Annotate each cell
    for row in range(5):
        for col in range(5):
            y_coord = 4 - row
            s_idx = col + 5 * y_coord + 25 * w
            val = V[s_idx]
            cell_type = S[s_idx].split('_')[0]

            txt_color = ('white'
                         if abs(val - (vmin + vmax) / 2) > (vmax - vmin) * 0.3
                         else '#1a1a2e')
            """The fhfh    fhfh    fhfh    fhhf    fhfh    fhfh    """
            ax.text(col, row, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, color=txt_color, fontweight='bold')
            ax.text(col, row + 0.38, f'({col},{y_coord}) {cell_type}',
                    ha='center', va='center',
                    fontsize=6, color=txt_color, alpha=0.8)

    # Grid lines
    for edge in [0.5, 1.5, 2.5, 3.5]:
        ax.axhline(edge, color='white', linewidth=0.5, alpha=0.4)
        ax.axvline(edge, color='white', linewidth=0.5, alpha=0.4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#16213e')
    for spine in ax.spines.values():
        spine.set_color(title_color)
        spine.set_linewidth(2)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=9)
    cbar.set_label('$V^*(s)$', color='white', fontsize=12)

    plt.tight_layout()

    return fig, ax