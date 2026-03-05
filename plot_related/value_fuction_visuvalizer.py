import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from typing import List

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")

def visualize_value_function(V: np.ndarray, S: List[str], title: str, gamma: float = 0.95, iters: int = 0):
    os.makedirs(SAVE_DIR, exist_ok=True)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'val', ['#b71c1c', '#e53935', '#ef9a9a', '#e0e0e0', '#a5d6a7', '#43a047', '#1b5e20'])
    vmin, vmax = np.min(V), np.max(V)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#1a1a2e')
    subtitles = ["No Water (w=0)", "Has Water (w=1)"]
    colors = ['#FF9800', '#2196F3']

    for idx, w in enumerate([0, 1]):
        ax = axes[idx]

        grid = np.zeros((5, 5))
        for y in range(5):
            for x in range(5):
                grid[4 - y, x] = V[x + 5 * y + 25 * w]

        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        info = f"(γ={gamma}, {iters} iters)" if iters > 0 else ""
        ax.set_title(f"{subtitles[idx]}  {info}", fontsize=11, fontweight='bold', color=colors[idx], pad=8)

        for row in range(5):
            for col in range(5):
                y_coord = 4 - row
                s_idx = col + 5 * y_coord + 25 * w
                val = V[s_idx]
                cell_type = S[s_idx].split('_')[0]

                txt_color = 'white' if abs(val - (vmin + vmax) / 2) > (vmax - vmin) * 0.3 else '#1a1a2e'
                ax.text(col, row, f'{val:.1f}', ha='center', va='center',
                        fontsize=10, color=txt_color, fontweight='bold')
                ax.text(col, row + 0.35, f'({col},{y_coord}) {cell_type}',
                        ha='center', va='center', fontsize=6, color=txt_color, alpha=0.8)

        for edge in [0.5, 1.5, 2.5, 3.5]:
            ax.axhline(edge, color='white', linewidth=0.5, alpha=0.4)
            ax.axvline(edge, color='white', linewidth=0.5, alpha=0.4)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#16213e')
        for spine in ax.spines.values():
            spine.set_color(colors[idx])
            spine.set_linewidth(2)

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.08)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=9)
    cbar.set_label('V*(s)', color='white', fontsize=12)

    fig.suptitle(title, fontsize=14, color='white', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    safe = title.replace(" ", "_")
    fig.savefig(os.path.join(SAVE_DIR, f"{safe}.png"), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.Agent.mdp_model import FiniteMDP
    from core.Agent.agent import Agent

    mdp = FiniteMDP()
    S, A, P, R, gamma = mdp.initialization_MDP_with_DA6400_Question1_manner(0.95)
    agent = Agent(mdp)
    V, iters = agent.value_iteration_on_V()

    visualize_value_function(V, S, "Optimal Value Function", gamma, iters)
    plt.show()
