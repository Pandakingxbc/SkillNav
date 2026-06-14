"""Five visual variants of the per-category SR bar chart.

Renders the same data with five different styles so the user can
pick. Each style draws inspiration from a popular ML/science paper
convention:

  A. Steel blue (current)          -- clean academic default
  B. Nature paper style            -- muted, thin axes, minimal
  C. Gradient (warm-to-cool by SR) -- color encodes rank
  D. SR + SPL grouped pair         -- two metrics, side by side
  E. Tableau pastel                -- soft, modern, readable

Outputs to docs/per_cat_variants/{A,B,C,D,E}.png for preview.
"""

import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.cm as cm
import numpy as np
import os


TARGETS = ["bed", "chair", "couch", "toilet", "tv", "potted plant"]
SR  = [84.2, 83.1, 77.5, 69.9, 54.8, 54.6]
SPL = [43.2, 39.5, 41.2, 29.8, 24.5, 21.2]
OVERALL_SR = 71.9


OUT_DIR = "/home/yangz/Nav/SkillNav/docs/per_cat_variants"
os.makedirs(OUT_DIR, exist_ok=True)


def annotate_bars(ax, bars, values, fontsize=7, color="#1a1a1a"):
    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 1.2,
            f"{v:.1f}",
            ha="center", va="bottom",
            fontsize=fontsize, color=color,
        )


def style_a_steel_blue():
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=180)
    x = np.arange(len(TARGETS))
    bars = ax.bar(x, SR, width=0.62, color="#4C72B0",
                  edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(OVERALL_SR, ls="--", lw=1.0, color="#D7263D",
               alpha=0.85, zorder=2)
    ax.text(len(TARGETS) - 0.3, OVERALL_SR + 1.5,
            f"Overall {OVERALL_SR:.1f}",
            ha="right", va="bottom", fontsize=7.5,
            color="#D7263D", fontweight="bold")
    annotate_bars(ax, bars, SR)
    _cosmetics(ax, x)
    plt.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "A_steel_blue.png"), dpi=200)
    plt.close(fig)


def style_b_nature():
    """Nature-style: muted gray-blue, thin black outline, no fill grid."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=180)
    x = np.arange(len(TARGETS))
    bars = ax.bar(x, SR, width=0.65, color="#7E9CB3",
                  edgecolor="#222", linewidth=0.7, zorder=3)
    ax.axhline(OVERALL_SR, ls=":", lw=1.0, color="#222",
               alpha=0.9, zorder=2)
    ax.text(len(TARGETS) - 0.3, OVERALL_SR + 1.5,
            f"avg.\\ {OVERALL_SR:.1f}",
            ha="right", va="bottom", fontsize=7,
            color="#222", style="italic")
    annotate_bars(ax, bars, SR)
    _cosmetics(ax, x, grid=False)
    plt.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "B_nature.png"), dpi=200)
    plt.close(fig)


def style_c_gradient():
    """Gradient: color encodes SR rank (warm = high, cool = low)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=180)
    x = np.arange(len(TARGETS))
    # Map SR linearly to a perceptual colormap
    sr_arr = np.array(SR)
    norm = (sr_arr - sr_arr.min()) / (sr_arr.max() - sr_arr.min())
    colors = cm.RdYlBu_r(0.15 + 0.75 * norm)
    bars = ax.bar(x, SR, width=0.62, color=colors,
                  edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(OVERALL_SR, ls="--", lw=1.0, color="#444",
               alpha=0.7, zorder=2)
    ax.text(len(TARGETS) - 0.3, OVERALL_SR + 1.5,
            f"Overall {OVERALL_SR:.1f}",
            ha="right", va="bottom", fontsize=7.5,
            color="#444")
    annotate_bars(ax, bars, SR)
    _cosmetics(ax, x)
    plt.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "C_gradient.png"), dpi=200)
    plt.close(fig)


def style_d_dual():
    """Grouped bar pair: SR (dark blue) + SPL (light orange)."""
    fig, ax = plt.subplots(figsize=(3.6, 2.5), dpi=180)
    x = np.arange(len(TARGETS))
    width = 0.36
    bars_sr = ax.bar(x - width / 2, SR, width=width,
                     color="#2E5F8A", edgecolor="white",
                     linewidth=0.4, label="SR", zorder=3)
    bars_spl = ax.bar(x + width / 2, SPL, width=width,
                      color="#E08E45", edgecolor="white",
                      linewidth=0.4, label="SPL", zorder=3)
    ax.axhline(OVERALL_SR, ls="--", lw=0.9, color="#2E5F8A",
               alpha=0.6, zorder=2)
    annotate_bars(ax, bars_sr, SR, fontsize=6.2)
    annotate_bars(ax, bars_spl, SPL, fontsize=6.2)
    ax.legend(fontsize=7, frameon=False, loc="upper right",
              ncol=2, columnspacing=0.8, handlelength=1.2)
    _cosmetics(ax, x)
    plt.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "D_dual.png"), dpi=200)
    plt.close(fig)


def style_e_pastel():
    """Tableau pastel: soft muted teal + minimal frame."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=180)
    x = np.arange(len(TARGETS))
    bars = ax.bar(x, SR, width=0.6, color="#76B5C5",
                  edgecolor="#3A6E7B", linewidth=0.5, zorder=3)
    ax.axhline(OVERALL_SR, ls="-.", lw=1.1, color="#E66B5C",
               alpha=0.85, zorder=2)
    ax.text(len(TARGETS) - 0.3, OVERALL_SR + 1.5,
            f"Overall {OVERALL_SR:.1f}",
            ha="right", va="bottom", fontsize=7.5,
            color="#C04A3D", fontweight="bold")
    annotate_bars(ax, bars, SR)
    _cosmetics(ax, x)
    plt.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, "E_pastel.png"), dpi=200)
    plt.close(fig)


def _cosmetics(ax, x, grid=True):
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Success Rate (%)", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=7.5)
    if grid:
        ax.grid(axis="y", ls=":", color="#aaaaaa", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")


if __name__ == "__main__":
    style_a_steel_blue()
    style_b_nature()
    style_c_gradient()
    style_d_dual()
    style_e_pastel()
    print(f"5 variants saved under {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
