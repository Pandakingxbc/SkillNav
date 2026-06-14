#!/usr/bin/env python3
"""Parse record.ablation_*.txt files into a single metrics table.

Reads each record.ablation_TAG.txt under videos/test_hm3dv2_val/, reconstructs
per-episode SR/SPL/SoftSPL/DTG from the cumulative running averages, and
prints both a console summary and LaTeX-ready row strings for the paper
ablation tables.

Usage:  python3 scripts/extract_ablation_metrics.py [--subset N]
        --subset N : restrict to the first N completed episodes
                     (apples-to-apples across configs with mismatched lengths)
"""
import argparse
import os
import re
import sys
from collections import defaultdict

RECORD_DIR = os.path.join("videos", "test_hm3dv2_val")
TAGS_ORDER = ["A1_base", "A7_no_M", "A4_M_E", "A3_M_S", "A0_full", "A6_no_SMDP"]
TAG_LABEL = {
    "A0_full":   "SkillNav (full)",
    "A7_no_M":   "$-$Memory agent",
    "A4_M_E":    "$-$Safe agent",
    "A3_M_S":    "$-$Exploration agent",
    "A1_base":   "Base (no agents)",
    "A6_no_SMDP":"$-$SMDP (per-tick LLM)",
}


def parse_record(path):
    """Return list of dicts (chronological) with per-ep SR/SPL/SoftSPL/DTG."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        txt = f.read()
    parts = [p for p in re.split(r"Scene ID: ", txt) if p.strip()]
    parts = list(reversed(parts))  # record.txt is reverse-chrono
    eps = []
    for p in parts:
        m_n   = re.search(r"No\.(\d+) task is finished", p)
        m_sr  = re.search(r"Average Success\s+\|\s+([\d\.]+)%", p)
        m_spl = re.search(r"Average SPL\s+\|\s+([\d\.]+)%", p)
        m_sspl= re.search(r"Average Soft SPL\s+\|\s+([\d\.]+)%", p)
        m_dist= re.search(r"Average Distance to Goal\s+\|\s+([\d\.]+)", p)
        m_res = re.search(r"success or not:\s+(.+?)\n", p)
        m_tgt = re.search(r"target to find is\s+(.+?)\n", p)
        if not (m_n and m_sr and m_spl):
            continue
        eps.append({
            "n":       int(m_n.group(1)),
            "sr_avg":  float(m_sr.group(1)),
            "spl_avg": float(m_spl.group(1)),
            "sspl_avg":float(m_sspl.group(1)) if m_sspl else 0.0,
            "dist_avg":float(m_dist.group(1)) if m_dist else 0.0,
            "result":  (m_res.group(1).strip() if m_res else "unknown"),
            "target":  (m_tgt.group(1).strip() if m_tgt else "unknown"),
            "success": 1 if (m_res and m_res.group(1).strip() == "success") else 0,
        })
    # Reconstruct per-ep metrics
    prev_spl = prev_sspl = prev_dist = 0.0
    for e in eps:
        n = e["n"]
        cur_spl  = n * e["spl_avg"]
        cur_sspl = n * e["sspl_avg"]
        cur_dist = n * e["dist_avg"]
        e["per_ep_spl"]  = cur_spl  - prev_spl
        e["per_ep_sspl"] = cur_sspl - prev_sspl
        e["per_ep_dist"] = cur_dist - prev_dist
        prev_spl, prev_sspl, prev_dist = cur_spl, cur_sspl, cur_dist
    return eps


def aggregate(eps):
    if not eps:
        return None
    n = len(eps)
    return {
        "n":      n,
        "sr":     sum(e["success"] for e in eps) / n * 100,
        "spl":    sum(e["per_ep_spl"] for e in eps) / n,
        "sspl":   sum(e["per_ep_sspl"] for e in eps) / n,
        "dist":   sum(e["per_ep_dist"] for e in eps) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0,
                    help="Limit to first N eps per config (0 = no limit)")
    ap.add_argument("--record-dir", default=RECORD_DIR)
    args = ap.parse_args()

    results = {}
    for tag in TAGS_ORDER:
        path = os.path.join(args.record_dir, f"record.ablation_{tag}.txt")
        eps = parse_record(path)
        if args.subset and eps:
            eps = eps[:args.subset]
        results[tag] = (eps, aggregate(eps))

    # Determine intersection length (min across all configs that have data)
    have = [(t, len(e)) for t, (e, _) in results.items() if e]
    if not have:
        print("No ablation data found in", args.record_dir)
        sys.exit(0)
    min_n = min(c for _, c in have)
    print(f"Configs with data: {len(have)}/{len(TAGS_ORDER)}")
    print(f"Per-config ep counts: {dict(have)}")
    print(f"Intersection (first {min_n} eps) used for fair comparison.\n")

    # Console table
    print("=" * 80)
    print(f"{'Tag':<12} {'Label':<28} {'N':>4} {'SR%':>6} {'SPL%':>6} {'SoftSPL%':>9} {'DTG':>6}")
    print("=" * 80)
    for tag in TAGS_ORDER:
        eps, _ = results[tag]
        if not eps:
            print(f"{tag:<12} {TAG_LABEL[tag]:<28} {'--':>4}  (no data)")
            continue
        ix_eps = eps[:min_n]
        agg = aggregate(ix_eps)
        print(f"{tag:<12} {TAG_LABEL[tag]:<28} {agg['n']:>4} "
              f"{agg['sr']:>5.1f} {agg['spl']:>5.1f} {agg['sspl']:>8.1f} {agg['dist']:>6.3f}")

    # LaTeX rows for tab:ablation (5 main rows)
    print("\n" + "=" * 80)
    print("LaTeX rows for tab:ablation:")
    print("=" * 80)
    template = "{label:<32} & {m} & {s} & {e} & {sr:>4.1f} & {spl:>4.1f} & {sspl:>4.1f} \\\\"
    config_axes = {
        "A1_base":   ("--", "--", "--"),
        "A7_no_M":   ("--", "\\checkmark", "\\checkmark"),
        "A4_M_E":    ("\\checkmark", "--", "\\checkmark"),
        "A3_M_S":    ("\\checkmark", "\\checkmark", "--"),
        "A0_full":   ("\\checkmark", "\\checkmark", "\\checkmark"),
    }
    for tag in ["A1_base", "A7_no_M", "A4_M_E", "A3_M_S", "A0_full"]:
        eps, _ = results[tag]
        if not eps:
            print(f"{TAG_LABEL[tag]:<32} & ... & XX.X & XX.X & XX.X \\\\  % {tag} missing")
            continue
        agg = aggregate(eps[:min_n])
        m, s, e = config_axes[tag]
        label = TAG_LABEL[tag]
        if tag == "A0_full":
            label = "\\sysname (full)"
        print(template.format(label=label, m=m, s=s, e=e,
                              sr=agg['sr'], spl=agg['spl'], sspl=agg['sspl']))

    # LaTeX rows for tab:smdp_ablation
    print("\nLaTeX rows for tab:smdp_ablation:")
    for tag in ["A6_no_SMDP", "A0_full"]:
        eps, _ = results[tag]
        if not eps:
            print(f"{TAG_LABEL[tag]:<32} & XX.X & XX.X & XX.X \\\\  % {tag} missing")
            continue
        agg = aggregate(eps[:min_n])
        label = TAG_LABEL[tag] if tag != "A0_full" else "\\sysname (option-boundary)"
        bold = "\\textbf{" + f"{agg['sr']:.1f}" + "}" if tag == "A0_full" else f"{agg['sr']:.1f}"
        bold_spl = "\\textbf{" + f"{agg['spl']:.1f}" + "}" if tag == "A0_full" else f"{agg['spl']:.1f}"
        bold_ssp = "\\textbf{" + f"{agg['sspl']:.1f}" + "}" if tag == "A0_full" else f"{agg['sspl']:.1f}"
        print(f"{label:<32} & {bold} & {bold_spl} & {bold_ssp} \\\\")


if __name__ == "__main__":
    main()
