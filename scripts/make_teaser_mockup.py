"""SkillNav teaser figure mockup (Figure 1) — blackboard variant.

Layout:
  - Left 55%: simulated top-down floor plan with ApexNav (red) vs
    SkillNav (green) trajectories, Voronoi topology nodes (orange,
    sized by V(v)), anchored frontiers, and an LLM-reply callout.
  - Right 45%: single "blackboard" diagram. Three agents (Memory,
    Safe, Exploration) sit on the periphery of a shared Voronoi
    graph; each writes ONE disjoint channel: Safe writes mu_safe
    (passability gate), Memory writes mu_mem (semantic confidence
    gate), Exploration writes (w_SR, w_IG) fusion weights. The
    graph emits arg max V(v) -> v* -> next waypoint.

All numbers / positions are hand-placed for demonstration.

Usage:
    python scripts/make_teaser_mockup.py
Output:
    docs/teaser_mockup.{pdf,png}
"""

import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch,
                                Circle, Rectangle)
import numpy as np


# ---------- palette ----------
C_APEX = "#D7263D"
C_OURS = "#2E933C"
C_NODE = "#F18F01"
C_AGENT = "#1F77B4"
C_TARGET = "#B22222"
C_WALL = "#444444"
C_FLOOR = "#F3F0E7"
C_CALLOUT = "#FFFCE8"
C_BG = "#FAFAF6"

# agent colors (one per agent, matches their write-channel arrow)
C_MEM = "#6A4C93"        # purple    -> mu_mem  (semantic confidence gate)
C_SAFE = "#1982C4"       # blue      -> mu_safe (passability gate)
C_EXPL = "#2E933C"       # green     -> (w_SR, w_IG) fusion weights

ANCHOR_COLORS = ["#7E57C2", "#26A69A", "#FFB300"]


# =====================================================================
# LEFT: floor plan + trajectories + topology
# =====================================================================

def draw_floorplan(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.add_patch(Rectangle((0, 0), 10, 7, facecolor=C_FLOOR, zorder=0))
    walls = [
        (0, 0, 10, 0.12), (0, 6.88, 10, 0.12),
        (0, 0, 0.12, 7), (9.88, 0, 0.12, 7),
        (5.5, 0, 0.12, 3.0), (5.5, 4.0, 0.12, 3.0),
        (5.5, 3.7, 4.5, 0.12), (7.5, 3.7, 0.12, 1.2),
    ]
    for x, y, w, h in walls:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=C_WALL, zorder=2))

    # room labels — small, off to the side, no overlap with toilet
    ax.text(2.7, 0.5, "living", fontsize=7, color="#999",
            ha="center", style="italic")
    ax.text(6.5, 0.5, "kitchen", fontsize=7, color="#999",
            ha="center", style="italic")
    ax.text(8.7, 5.5, "bath", fontsize=7, color="#999",
            ha="center", style="italic")


def draw_value_map_overlay(ax):
    xs = np.linspace(0, 10, 200)
    ys = np.linspace(0, 7, 140)
    X, Y = np.meshgrid(xs, ys)
    hot1 = np.exp(-((X - 8.6) ** 2 + (Y - 4.5) ** 2) / 0.9)
    hot2 = np.exp(-((X - 5.8) ** 2 + (Y - 3.5) ** 2) / 2.0) * 0.4
    ax.imshow(hot1 + hot2, extent=(0, 10, 0, 7), origin="lower",
              cmap="YlOrRd", alpha=0.30, zorder=1)


def draw_trajectories(ax):
    apex_pts = np.array([
        (1.0, 1.0), (1.5, 2.0), (2.5, 2.8), (3.5, 3.0),
        (5.0, 3.5), (6.2, 3.6), (7.0, 3.2), (7.2, 2.8),
        (7.0, 3.4), (6.4, 3.6), (5.7, 3.6),
        (4.5, 4.5), (4.0, 5.2), (5.0, 5.0), (5.8, 4.5),
    ])
    ours_pts = np.array([
        (1.0, 1.0), (1.8, 2.0), (2.8, 2.8), (4.0, 3.3),
        (5.0, 3.5), (5.8, 3.5), (6.8, 4.0), (7.6, 4.3),
        (8.5, 4.5),
    ])
    ax.plot(apex_pts[:, 0], apex_pts[:, 1], color=C_APEX, linewidth=2.0,
            zorder=4, alpha=0.85)
    ax.plot(ours_pts[:, 0], ours_pts[:, 1], color=C_OURS, linewidth=2.5,
            zorder=5, alpha=0.95)
    ax.scatter(1.0, 1.0, s=120, color=C_AGENT, edgecolor="white",
               linewidth=1.5, zorder=6)
    ax.add_patch(Rectangle((8.3, 4.2), 0.6, 0.6, facecolor="none",
                           edgecolor=C_TARGET, linewidth=2.0, zorder=6))
    ax.text(8.6, 5.0, "toilet", fontsize=8, color=C_TARGET,
            ha="center", fontweight="bold")


def draw_voronoi_and_anchors(ax):
    nodes = [
        (2.5, 3.5, 0.6), (4.5, 3.5, 0.4), (5.8, 3.5, 0.9),
        (7.5, 4.3, 1.2), (6.5, 1.5, 0.3),
    ]
    for x, y, v in nodes:
        radius = 0.14 + 0.25 * v
        ax.add_patch(Circle((x, y), radius, facecolor=C_NODE,
                            edgecolor="white", linewidth=1.5,
                            alpha=0.90, zorder=7))
    edges = [(0, 1), (1, 2), (2, 3), (2, 4)]
    for i, j in edges:
        ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]],
                color=C_NODE, linewidth=1.0, alpha=0.5, zorder=3,
                linestyle="--")

    anchor_groups = [
        (0, [(1.5, 4.2), (2.0, 5.3), (3.5, 4.2)]),
        (2, [(5.4, 4.5), (6.0, 4.5), (5.9, 5.4)]),
        (3, [(7.2, 5.2), (8.2, 5.4), (8.5, 3.5)]),
    ]
    for gid, (node_id, frontiers) in enumerate(anchor_groups):
        color = ANCHOR_COLORS[gid]
        nx, ny, _ = nodes[node_id]
        for fx, fy in frontiers:
            ax.scatter(fx, fy, s=22, color=color, zorder=6,
                       edgecolor="white", linewidth=0.6, marker="D")
            ax.plot([fx, nx], [fy, ny], color=color, linewidth=0.8,
                    alpha=0.45, zorder=3)


def draw_left_callout(ax):
    bbox = FancyBboxPatch(
        (0.3, 0.18), 4.4, 0.95,
        boxstyle="round,pad=0.08", linewidth=0.6,
        facecolor=C_CALLOUT, edgecolor="#888", zorder=10,
    )
    ax.add_patch(bbox)
    ax.text(0.5, 0.82, "Strategic LLM (every 3 s):",
            fontsize=6.5, color="#444", fontweight="bold", zorder=11)
    ax.text(0.5, 0.50,
            '"candidates=3, fp=1 -> phase=2 target-approach,',
            fontsize=6.5, color="#222", family="monospace", zorder=11)
    ax.text(0.5, 0.28,
            ' (w_SR, w_IG) = (0.80, 0.20)"',
            fontsize=6.5, color="#222", family="monospace", zorder=11)


def draw_legend(ax):
    items = [
        (C_APEX, "ApexNav (baseline)"),
        (C_OURS, "SkillNav (ours)"),
        (C_NODE, "Voronoi node (size proportional V(v))"),
    ]
    y = 6.65
    for color, label in items:
        ax.plot([0.3, 0.9], [y, y], color=color, linewidth=2.5)
        ax.text(1.05, y - 0.04, label, fontsize=7,
                va="center", color="#222")
        y -= 0.28
    ax.scatter([0.55], [y], color=ANCHOR_COLORS[0], marker="D", s=22,
               edgecolor="white", linewidth=0.5)
    ax.text(1.05, y - 0.04, "frontier (color = anchor node)",
            fontsize=7, va="center", color="#222")


# =====================================================================
# RIGHT: blackboard interaction diagram
# =====================================================================

def draw_blackboard(ax):
    """Three agents writing disjoint channels into a shared Voronoi graph."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # ----- shared blackboard (central card) -----
    card = FancyBboxPatch(
        (1.6, 3.0), 6.8, 4.2,
        boxstyle="round,pad=0.12", linewidth=1.2,
        facecolor="white", edgecolor="#333",
    )
    ax.add_patch(card)
    ax.text(5.0, 6.95, "Shared Voronoi Blackboard",
            fontsize=9.5, fontweight="bold", ha="center", color="#1a1a1a")

    # mini voronoi graph inside the card
    inner_nodes = [
        (3.0, 4.2), (4.5, 5.0), (5.8, 4.4),
        (6.9, 5.2), (4.2, 6.0), (6.4, 6.1),
    ]
    inner_sizes = [0.20, 0.18, 0.28, 0.40, 0.22, 0.24]
    inner_edges = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 3)]
    for (x, y), s in zip(inner_nodes, inner_sizes):
        ax.add_patch(Circle((x, y), s, facecolor=C_NODE,
                            edgecolor="white", linewidth=1.0, zorder=8))
    for i, j in inner_edges:
        ax.plot([inner_nodes[i][0], inner_nodes[j][0]],
                [inner_nodes[i][1], inner_nodes[j][1]],
                color=C_NODE, linewidth=1.2, alpha=0.55, zorder=6,
                linestyle="-")
    # highlight v* (the chosen biggest node)
    ax.add_patch(Circle(inner_nodes[3], inner_sizes[3] + 0.10,
                        facecolor="none", edgecolor=C_OURS,
                        linewidth=1.8, zorder=9, linestyle="--"))
    ax.text(inner_nodes[3][0] + 0.05, inner_nodes[3][1] - 0.85,
            "v*", fontsize=10, color=C_OURS,
            fontweight="bold", ha="center")

    # formula at the bottom of card
    ax.text(5.0, 3.4,
            r"$V(v) = \mu_{\mathrm{safe}}(v)\,\mu_{\mathrm{mem}}(v)\cdot[A_{\mathrm{IG}}(v) + B(v) + \gamma\,A_{\mathrm{comb}}(v)]$",
            fontsize=8.5, ha="center", color="#1a1a1a")

    # ----- three agents on the periphery -----
    def agent_box(cx, cy, w, h, color, name, role):
        box = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.10", linewidth=1.0,
            facecolor=color + "22", edgecolor=color,
        )
        ax.add_patch(box)
        ax.text(cx, cy + 0.25, name, fontsize=8.5,
                fontweight="bold", color=color, ha="center")
        ax.text(cx, cy - 0.30, role, fontsize=6.8,
                color="#444", ha="center", style="italic")

    # Memory (top-left)
    agent_box(1.6, 8.8, 2.6, 1.1, C_MEM,
              "Memory Agent",
              "verify candidates, suppress FP")
    # Exploration (top-right)
    agent_box(8.4, 8.8, 2.8, 1.1, C_EXPL,
              "Exploration Agent",
              "phase + fusion weights")
    # Safe (bottom-center)
    agent_box(5.0, 1.3, 2.6, 1.1, C_SAFE,
              "Safe Agent",
              "stuck escape, dead-zone")

    # ----- write-channel arrows from agents into the blackboard -----
    def write_arrow(ax_from_xy, ax_to_xy, color, label, label_pos,
                    label_color):
        arrow = FancyArrowPatch(
            ax_from_xy, ax_to_xy,
            arrowstyle="-|>", mutation_scale=18,
            color=color, linewidth=2.2,
            connectionstyle="arc3,rad=0.20",
        )
        ax.add_patch(arrow)
        ax.text(label_pos[0], label_pos[1], label, fontsize=8,
                color=label_color, ha="center", fontweight="bold")

    # Memory writes mu_mem(v) — semantic-confidence gate (decays with FP count)
    write_arrow(
        (1.9, 8.25), (3.0, 6.7),
        C_MEM,
        r"$\mu_{\mathrm{mem}}(v)$  (semantic)",
        (1.6, 7.4), C_MEM,
    )
    # Exploration writes (w_SR, w_IG)
    write_arrow(
        (8.1, 8.25), (7.0, 6.7),
        C_EXPL,
        r"$(w_{\mathrm{SR}}, w_{\mathrm{IG}})$",
        (8.5, 7.4), C_EXPL,
    )
    # Safe writes mu_safe(v) — passability gate (decays on escape failure)
    write_arrow(
        (5.0, 1.95), (5.0, 3.0),
        C_SAFE,
        r"$\mu_{\mathrm{safe}}(v)$  (passability)",
        (6.5, 2.5), C_SAFE,
    )

    # ----- output: arg max -> v* -> next waypoint -----
    out_arrow = FancyArrowPatch(
        (8.4, 4.5), (9.7, 4.5),
        arrowstyle="-|>", mutation_scale=22,
        color="#1a1a1a", linewidth=2.0,
    )
    ax.add_patch(out_arrow)
    ax.text(9.05, 4.85, r"$\arg\max_v V(v)$",
            fontsize=7.5, ha="center", color="#222",
            fontweight="bold")
    ax.text(9.05, 4.15, r"$\to v^\star$",
            fontsize=8.5, ha="center", color=C_OURS,
            fontweight="bold")

    # ----- title / hint at very top -----
    ax.text(5.0, 9.75,
            "Three agents write disjoint channels onto one shared graph",
            fontsize=8.5, ha="center", color="#444", style="italic")


# =====================================================================
# main
# =====================================================================

def main():
    fig = plt.figure(figsize=(12.5, 5.5), dpi=150, facecolor=C_BG)
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[1.25, 1.0],
        wspace=0.04,
        left=0.02, right=0.98, top=0.96, bottom=0.04,
    )

    ax_left = fig.add_subplot(gs[0, 0])
    draw_floorplan(ax_left)
    draw_value_map_overlay(ax_left)
    draw_voronoi_and_anchors(ax_left)
    draw_trajectories(ax_left)
    draw_left_callout(ax_left)
    draw_legend(ax_left)

    ax_right = fig.add_subplot(gs[0, 1])
    draw_blackboard(ax_right)

    out_pdf = "/home/yangz/Nav/SkillNav/docs/teaser_mockup.pdf"
    out_png = "/home/yangz/Nav/SkillNav/docs/teaser_mockup.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    print(f"saved: {out_pdf}\nsaved: {out_png}")


if __name__ == "__main__":
    main()
