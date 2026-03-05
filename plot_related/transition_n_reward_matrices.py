import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def visualize_transition_and_reward(x: int, y: int, w: int, P: np.ndarray, R: np.ndarray, S: list, title: str):
    actions = ["North", "South", "East", "West", "Hover"]
    s_idx = x + 5 * y + 25 * w

    neighbors = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = x + dx, y + dy
            if 0 <= nx < 5 and 0 <= ny < 5:
                neighbors.append((nx, ny))

    col_labels = [f"({nx},{ny})" for nx, ny in neighbors]
    n_cols = len(col_labels)

    trans_values = np.zeros((5, n_cols))
    for a in range(5):
        for j, (nx, ny) in enumerate(neighbors):
            trans_values[a, j] = P[a, s_idx, nx + 5*ny + 25*w]
            if w == 0:
                trans_values[a, j] += P[a, s_idx, nx + 5*ny + 25]

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
    os.makedirs(save_dir, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(n_cols * 1.1, 3.5))
    ax1.set_title(f"{title} - Transition Probabilities", fontsize=12, pad=12)
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=max(trans_values.max(), 0.01))
    ax1.set_xlim(0, n_cols)
    ax1.set_ylim(0, 5)
    ax1.set_xticks([i + 0.5 for i in range(n_cols)])
    ax1.set_xticklabels(col_labels, fontsize=9)
    ax1.set_yticks([i + 0.5 for i in range(5)])
    ax1.set_yticklabels(actions[::-1], fontsize=9)
    ax1.tick_params(length=0)
    for a in range(5):
        for j in range(n_cols):
            v = trans_values[4 - a, j]
            color = cmap(norm(v))
            ax1.add_patch(plt.Rectangle((j, a), 1, 1, facecolor=color, edgecolor='#555555'))
            txt_color = 'white' if v > 0.5 * trans_values.max() else 'black'
            ax1.text(j + 0.5, a + 0.5, f"{v:g}", ha='center', va='center', fontsize=9, color=txt_color)
    fig1.tight_layout()
    safe = title.replace(" ", "_")
    fig1.savefig(os.path.join(save_dir, f"{safe}_transition.png"), dpi=150, bbox_inches='tight')

    # reward plus: center + 4 cardinal directions
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    dir_names = ["North", "South", "East", "West"]
    grid_map = {0: (1, 2), 1: (1, 0), 2: (2, 1), 3: (0, 1)}
    center_g = (1, 1)

    cell_info = {}
    cell_info[center_g] = ("Hover", R[s_idx, 4], x, y)
    for i, (dx, dy) in enumerate(dirs):
        nx, ny = x + dx, y + dy
        gx, gy = grid_map[i]
        if 0 <= nx < 5 and 0 <= ny < 5:
            cell_info[(gx, gy)] = (dir_names[i], R[s_idx, i], nx, ny)
        else:
            cell_info[(gx, gy)] = (dir_names[i], None, nx, ny)

    r_vals = [info[1] for info in cell_info.values() if info[1] is not None]
    r_min, r_max = (min(r_vals), max(r_vals)) if r_vals else (0, 1)
    if r_min == r_max:
        r_min -= 1
    r_norm = mcolors.Normalize(vmin=r_min, vmax=r_max)
    r_cmap = plt.cm.RdYlGn

    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
    ax2.set_title(f"{title} - Rewards from ({x},{y})", fontsize=12, pad=12)
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    plus_cells = [center_g, (1, 2), (1, 0), (2, 1), (0, 1)]
    for (gx, gy) in plus_cells:
        if (gx, gy) in cell_info:
            label, r_val, cx, cy = cell_info[(gx, gy)]
            color = r_cmap(r_norm(r_val)) if r_val is not None else '#333333'
            ax2.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, facecolor=color, edgecolor='black', linewidth=1.5))
            if r_val is not None:
                ax2.text(gx, gy + 0.15, f"{r_val:g}", ha='center', va='center', fontsize=12, fontweight='bold')
                s_label = S[cx + 5*cy + 25*w] if 0 <= cx < 5 and 0 <= cy < 5 else "OOB"
                ax2.text(gx, gy - 0.25, f"({cx},{cy}) {s_label}", ha='center', va='center', fontsize=7, color='#333333')
            else:
                ax2.text(gx, gy, f"OOB ({cx},{cy})", ha='center', va='center', fontsize=9, color='gray')

    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, f"{safe}_reward.png"), dpi=150, bbox_inches='tight')

    return fig1, fig2
