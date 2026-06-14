"""Per-category bar chart (Style D: dual SR + SPL).

Final paper figure replacing the per-target table. Data source:
docs/paper_data_hm3dv2.md (real run, HM3D val).

Y axis: percent. X axis: object category. Two bars per category
(SR primary, SPL secondary). Dashed horizontal line marks the
overall SR (71.9 %).

Usage:
    python scripts/make_per_category_bar.py
Output:
    paper/figures/per_category_bar.{pdf,png}
"""

import matplotlib.pyplot as plt
import numpy as np
import os


TARGETS = ["bed", "chair", "couch", "toilet", "tv", "plant"]  # "potted plant" → "plant"
SR  = [84.2, 83.1, 77.5, 69.9, 54.8, 54.6]
SPL = [43.2, 39.5, 41.2, 29.8, 24.5, 21.2]
OVERALL_SR = 71.9

C_SR  = "#2E5F8A"
C_SPL = "#E08E45"
C_OVR = "#2E5F8A"


def annotate(ax, bars, values, fontsize=6.2, color="#222"):
    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 1.2,
            f"{v:.1f}",
            ha="center", va="bottom",
            fontsize=fontsize, color=color,
        )


def main():
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=180)
    x = np.arange(len(TARGETS))
    width = 0.36

    bars_sr  = ax.bar(x - width / 2, SR,  width=width,
                      color=C_SR,  edgecolor="white",
                      linewidth=0.5, label="SR",  zorder=3)
    bars_spl = ax.bar(x + width / 2, SPL, width=width,
                      color=C_SPL, edgecolor="white",
                      linewidth=0.5, label="SPL", zorder=3)

    ax.axhline(OVERALL_SR, ls="--", lw=0.9,
               color=C_OVR, alpha=0.55, zorder=2)

    annotate(ax, bars_sr,  SR)
    annotate(ax, bars_spl, SPL)

    ax.legend(fontsize=7, frameon=False, loc="upper right",
              ncol=2, columnspacing=0.8, handlelength=1.2)

    ax.set_xticks(x)
    # Slightly more rotation + a tad more bottom margin so labels
    # never clip into each other (esp. between "tv" and "plant").
    ax.set_xticklabels(TARGETS, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Score (%)", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", ls=":", color="#aaaaaa", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")

    plt.tight_layout(pad=0.4)
    out_pdf = "/home/yangz/Nav/SkillNav/paper/figures/per_category_bar.pdf"
    out_png = "/home/yangz/Nav/SkillNav/paper/figures/per_category_bar.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=240)
    print(f"saved: {out_pdf}\nsaved: {out_png}")


if __name__ == "__main__":
    main()
