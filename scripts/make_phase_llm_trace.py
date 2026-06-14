"""Phase-LLM trace figure (Figure 3 in main.tex, analysis section).

Reads the strategic_agent log produced by the live system and plots:
  - x: time (seconds, relative to a chosen window start)
  - y: SMDP option label (phase 0/1/2) as a step function
  - red dots at the timestamp of every LLM consultation

The figure honestly reflects the current implementation, which fires
the LLM on a fixed 3 s cadence rather than only at option boundaries
(see paper §5 for discussion).

Usage:
    python scripts/make_phase_llm_trace.py [--log <path>] [--window-s 360]
Output:
    paper/figures/phase_llm_trace.{pdf,png}
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches


DEFAULT_LOG_GLOB = "/home/yangz/.ros/log/*/strategic_agent_node-5.log"

LOG_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d+).*?phase=(\d).*?w=\(([0-9.]+),([0-9.]+)\)"
)

# colors per phase
PHASE_COLORS = {
    0: "#5DA9E9",     # broad-explore (blue, IG-heavy)
    1: "#FFB300",     # directed-search (amber)
    2: "#2E933C",     # target-approach (green, SR-heavy)
}
PHASE_NAMES = {0: "broad-explore", 1: "directed-search", 2: "target-approach"}


def find_most_recent_log() -> str:
    candidates = sorted(glob.glob(DEFAULT_LOG_GLOB),
                        key=os.path.getmtime, reverse=True)
    for p in candidates:
        if os.path.getsize(p) > 1024:
            return p
    raise FileNotFoundError(f"no strategic agent log under {DEFAULT_LOG_GLOB}")


def parse_log(path: str) -> list[tuple[dt.datetime, int, float, float]]:
    rows: list[tuple[dt.datetime, int, float, float]] = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            ts = dt.datetime.strptime(m.group(1)[:19], "%Y-%m-%d %H:%M:%S")
            rows.append((ts, int(m.group(2)),
                         float(m.group(3)), float(m.group(4))))
    return rows


def pick_window(rows, window_s: float):
    if not rows:
        raise RuntimeError("no LLM replies parsed from log")
    # Pick a stretch with diverse phases — find earliest 0->2 transition.
    transition_idx = None
    for i in range(1, len(rows)):
        if rows[i][1] != rows[i - 1][1]:
            transition_idx = i
            break
    start_idx = max(0, (transition_idx or 0) - 3)
    t0 = rows[start_idx][0]
    return [(t, p, w_sr, w_ig) for (t, p, w_sr, w_ig) in rows[start_idx:]
            if (t - t0).total_seconds() <= window_s], t0


def plot(rows, t0, out_pdf, out_png):
    times = [(t - t0).total_seconds() for (t, _, _, _) in rows]
    phases = [p for (_, p, _, _) in rows]
    if not times:
        raise RuntimeError("window is empty")

    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=160)
    # Background bands per phase region (build step segments)
    last_p = phases[0]
    seg_start = times[0]
    for i in range(1, len(times) + 1):
        if i == len(times) or phases[i] != last_p:
            seg_end = times[i - 1] if i == len(times) else times[i]
            ax.axvspan(seg_start, seg_end,
                       color=PHASE_COLORS[last_p], alpha=0.12, zorder=0)
            if i < len(times):
                seg_start = times[i]
                last_p = phases[i]

    # step plot of phase
    ax.step(times, phases, where="post", linewidth=1.8, color="#222",
            zorder=3)
    # LLM call dots
    ax.scatter(times, phases, color="#D7263D", s=18, zorder=5,
               edgecolor="white", linewidth=0.5, label="LLM call")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([f"{p}\n{PHASE_NAMES[p]}" for p in (0, 1, 2)],
                       fontsize=7.5)
    ax.set_ylim(-0.4, 2.4)
    ax.set_xlabel("Time within episode (s)", fontsize=8.5)
    ax.set_ylabel("Option", fontsize=8.5)
    ax.grid(True, axis="x", linestyle=":", color="#bbb", alpha=0.6)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    # Annotation: cadence
    n = len(times)
    span = times[-1] - times[0] if n > 1 else 1.0
    cadence = span / max(n - 1, 1)
    ax.text(0.02, 0.95,
            f"n = {n} calls,  mean cadence = {cadence:.2f} s",
            transform=ax.transAxes, fontsize=7.5, color="#333",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffce8",
                      edgecolor="#aaa", linewidth=0.6))

    plt.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    print(f"saved: {out_pdf}\nsaved: {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=None, help="path to strategic_agent log")
    p.add_argument("--window-s", type=float, default=360.0,
                   help="time window to plot (seconds)")
    p.add_argument("--out-dir",
                   default="/home/yangz/Nav/SkillNav/paper/figures")
    args = p.parse_args()

    log_path = args.log or find_most_recent_log()
    print(f"reading: {log_path}")
    rows = parse_log(log_path)
    print(f"parsed:  {len(rows)} LLM replies")
    win_rows, t0 = pick_window(rows, args.window_s)
    print(f"window:  {len(win_rows)} calls, t0={t0}")

    out_pdf = os.path.join(args.out_dir, "phase_llm_trace.pdf")
    out_png = os.path.join(args.out_dir, "phase_llm_trace.png")
    plot(win_rows, t0, out_pdf, out_png)


if __name__ == "__main__":
    main()
