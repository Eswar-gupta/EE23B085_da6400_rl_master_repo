import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple


def transition_probability_visualizer(
    point: Tuple[int, int],
    P: np.ndarray,
    S: List[str],
    water: bool,
    figsize: Tuple[float, float] = (6, 5.5),
):
    """
    Visualise transition probabilities around a point for 4 actions (N, S, E, W).

    Parameters
    ----------
    point : (x, y)
        Grid coordinate of the center state.
    P : np.ndarray, shape (A, S, S')
        Transition probability tensor.
    S : List[str]
        State labels (length 50).
    water : bool
        True → w=1 (with water), False → w=0 (without water).
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, axes : matplotlib Figure and 2×2 Axes array.
    """
    cx, cy = point
    w = 1 if water else 0
    s_idx = cx + 5 * cy + 25 * w

    # 9 neighbor cells: top row = y+1, mid = y, bot = y-1
    neighbors = []
    for dy in [1, 0, -1]:
        for dx in [-1, 0, 1]:
            neighbors.append((cx + dx, cy + dy))

    action_names = ['North ↑', 'South ↓', 'East →', 'West ←']
    action_colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'prob', ['#f7fbff', '#2171b5', '#08306b'])

    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='#1a1a2e')
    fig.suptitle(f'Transition Probabilities from state ({cx},{cy})',
                 fontsize=16, fontweight='bold', color='white')

    for a_idx in range(4):
        ax = axes[a_idx // 2][a_idx % 2]
        grid = np.zeros((3, 3))

        for i, (nx, ny) in enumerate(neighbors):
            if 0 <= nx < 5 and 0 <= ny < 5:
                s_prime = nx + 5 * ny + 25 * w
                grid[i // 3, i % 3] = P[a_idx, s_idx, s_prime]

        ax.imshow(grid, cmap=cmap, vmin=0, vmax=0.8, aspect='equal')
        ax.set_title(f'Action = {action_names[a_idx]}', fontsize=13,
                     fontweight='bold', color=action_colors[a_idx], pad=8)

        for r in range(3):
            for c in range(3):
                val = grid[r, c]
                nx, ny = neighbors[r * 3 + c]

                is_center = (nx == cx and ny == cy)
                bbox = (dict(boxstyle='round,pad=0.2', facecolor='gold', alpha=0.3)
                        if is_center else None)

                txt_color = 'white' if val > 0.35 else '#1a1a2e'
                ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=14, color=txt_color, fontweight='bold', bbox=bbox)
                ax.text(c, r + 0.35, f'({nx},{ny})', ha='center', va='center',
                        fontsize=8, color=txt_color, alpha=0.8)

        for edge in [0.5, 1.5]:
            ax.axhline(edge, color='white', linewidth=0.5, alpha=0.4)
            ax.axvline(edge, color='white', linewidth=0.5, alpha=0.4)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#16213e')
        for spine in ax.spines.values():
            spine.set_color(action_colors[a_idx])
            spine.set_linewidth(2)

    plt.subplots_adjust(hspace=0.4, wspace=0.25,
                        left=0.05, right=0.95, top=0.88, bottom=0.05)

    return fig, axes


def reward_visualizer(
    point: Tuple[int, int],
    S: List[str],
    figsize: Tuple[float, float] = (8, 3.5),
):
    """
    Visualise rewards R(s') around a point for both water phases side by side.

    Reward rules (fixed):
        Boulder → -100,  Fire + water → +100,  Smoke → -11,  else → -1.

    Parameters
    ----------
    point : (x, y)
        Grid coordinate of the center state.
    S : List[str]
        State labels (length 50).
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, axes : matplotlib Figure and 1×2 Axes array.
    """
    cx, cy = point

    neighbors = []
    for dy in [1, 0, -1]:
        for dx in [-1, 0, 1]:
            neighbors.append((cx + dx, cy + dy))

    diag_offsets = {(-1, 1), (1, 1), (-1, -1), (1, -1)}

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'reward', ['#b71c1c', '#e53935', '#ef9a9a',
                   '#e0e0e0',
                   '#a5d6a7', '#43a047', '#1b5e20'])

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='#1a1a2e')
    fig.suptitle(f'Reward R(s\') for neighbors of ({cx},{cy})',
                 fontsize=16, fontweight='bold', color='white')

    phase_names = ['With Water 💧', 'Without Water']
    phase_colors = ['#2196F3', '#FF9800']

    for idx, (w, ax) in enumerate(zip([1, 0], axes)):
        grid = np.full((3, 3), np.nan)

        for i, (nx, ny) in enumerate(neighbors):
            ddx, ddy = nx - cx, ny - cy
            r_row, c_col = i // 3, i % 3

            if (ddx, ddy) in diag_offsets:
                continue

            if not (0 <= nx < 5 and 0 <= ny < 5):
                continue

            s_prime = nx + 5 * ny + 25 * w
            label = S[s_prime]

            if "Boulder" in label:
                grid[r_row, c_col] = -100.0
            elif "Fire" in label and w == 1:
                grid[r_row, c_col] = 100.0
            else:
                val = -1.0
                if "Smoke" in label:
                    val -= 10.0
                grid[r_row, c_col] = val

        masked = np.ma.masked_invalid(grid)
        cmap_copy = cmap.copy()
        cmap_copy.set_bad(color='#0d1117')

        ax.imshow(masked, cmap=cmap_copy, vmin=-100, vmax=100, aspect='equal')
        ax.set_title(phase_names[idx], fontsize=13, fontweight='bold',
                     color=phase_colors[idx], pad=8)

        for r_row in range(3):
            for c_col in range(3):
                nx, ny = neighbors[r_row * 3 + c_col]
                ddx, ddy = nx - cx, ny - cy

                if (ddx, ddy) in diag_offsets:
                    ax.text(c_col, r_row, '—\nN/A', ha='center', va='center',
                            fontsize=9, color='#555', style='italic')
                    continue

                if not (0 <= nx < 5 and 0 <= ny < 5):
                    ax.text(c_col, r_row, '—\nN/A', ha='center', va='center',
                            fontsize=9, color='#555', style='italic')
                    continue

                val = grid[r_row, c_col]
                s_prime = nx + 5 * ny + 25 * w
                cell_type = S[s_prime].split('_')[0]

                is_center = (nx == cx and ny == cy)
                bbox = (dict(boxstyle='round,pad=0.2', facecolor='gold', alpha=0.3)
                        if is_center else None)

                txt_color = 'white' if abs(val) > 50 else '#1a1a2e'
                ax.text(c_col, r_row, f'{val:.0f}', ha='center', va='center',
                        fontsize=16, color=txt_color, fontweight='bold', bbox=bbox)
                ax.text(c_col, r_row + 0.35, f'({nx},{ny}) {cell_type}',
                        ha='center', va='center',
                        fontsize=7, color=txt_color, alpha=0.8)

        for edge in [0.5, 1.5]:
            ax.axhline(edge, color='white', linewidth=0.5, alpha=0.4)
            ax.axvline(edge, color='white', linewidth=0.5, alpha=0.4)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#16213e')
        for spine in ax.spines.values():
            spine.set_color(phase_colors[idx])
            spine.set_linewidth(2)

    plt.subplots_adjust(hspace=0.3, wspace=0.3,
                        left=0.03, right=0.97, top=0.82, bottom=0.05)

    return fig, axes