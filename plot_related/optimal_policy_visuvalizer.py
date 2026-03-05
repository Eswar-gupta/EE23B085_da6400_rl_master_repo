import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "Images", "Grid_env_images")
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")

LABEL_TO_IMG = {"Lake": "pond.jpg", "Fire": "fire.jpg", "Smoke": "smoke.png", "Boulder": "boulder.png"}

def _load_img(fname, size=120):
    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGBA")
    return np.array(img.resize((size, size), Image.LANCZOS))

def visualize_optimal_policy(Q: np.ndarray, S: list, title: str):
    os.makedirs(SAVE_DIR, exist_ok=True)

    images = {}
    for key, fname in LABEL_TO_IMG.items():
        try:
            images[key] = _load_img(fname)
        except FileNotFoundError:
            images[key] = None

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor='#1a1a2e')

    skip = {0: (0, 0), 1: (4, 4)}
    subtitles = {0: "No Water (w=0)", 1: "Has Water (w=1)"}

    for idx, w in enumerate([0, 1]):
        ax = axes[idx]
        ax.set_facecolor('#1a1a2e')

        pad = 0.05
        for y in range(5):
            for x in range(5):
                ax.add_patch(mpatches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='white', edgecolor='none', zorder=0))
                s_idx = x + 5 * y + 25 * w
                cell_type = S[s_idx].split('_')[0]
                if cell_type in images and images[cell_type] is not None:
                    extent = [x - 0.5 + pad, x + 0.5 - pad, y - 0.5 + pad, y + 0.5 - pad]
                    ax.imshow(images[cell_type], extent=extent, zorder=1, aspect='auto')

        for coord in np.arange(-0.5, 5.5, 1):
            ax.plot([coord, coord], [-0.5, 4.5], 'k', linewidth=1.5, zorder=2)
            ax.plot([-0.5, 4.5], [coord, coord], 'k', linewidth=1.5, zorder=2)

        for y in range(5):
            for x in range(5):
                s_idx = x + 5 * y + 25 * w
                cell_type = S[s_idx].split('_')[0]

                if (x, y) == skip[w]:
                    continue

                best_a = np.argmax(Q[s_idx])
                dx, dy = ACTIONS[best_a]

                if dx == 0 and dy == 0:
                    ax.plot(x, y, 'ko', markersize=8, zorder=5)
                else:
                    ax.annotate("", xy=(x + dx * 0.45, y + dy * 0.45), xytext=(x, y),
                                arrowprops=dict(arrowstyle='->', color='black', lw=2.5),
                                zorder=5)

        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(-0.6, 4.6)
        ax.set_aspect('equal')
        ax.set_title(subtitles[w], fontsize=12, color='white', pad=8)
        ax.axis('off')

    fig.suptitle(title, fontsize=14, color='white', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    safe = title.replace(" ", "_")
    fig.savefig(os.path.join(SAVE_DIR, f"{safe}.png"), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig

