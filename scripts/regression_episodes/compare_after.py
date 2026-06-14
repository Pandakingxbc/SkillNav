#!/usr/bin/env python3
"""
Compare a freshly-produced record.txt against a regression subset (fp_diff /
stepout_diff) and print before/after stats.

Usage:
    python3 compare_after.py <new_record.txt> <subset.json>

Example:
    python3 scripts/regression_episodes/compare_after.py \
        videos/test_hm3dv2_val/record.txt \
        scripts/regression_episodes/fp_diff.json
"""
import json
import re
import sys
from collections import Counter, defaultdict


def parse_record(path):
    """Return dict (scene, epi) -> outcome."""
    text = open(path).read()
    by_key = {}
    for block in re.split(r'\n\s*\n', text):
        m_scene = re.search(r'Scene ID:\s*(\S+)', block)
        m_epi   = re.search(r'Episode ID:\s*(\d+)', block)
        m_out   = re.search(r'success or not:\s*(.+)', block)
        m_tgt   = re.search(r'target to find is\s+(.+)', block)
        if m_scene and m_epi and m_out:
            scene_short = m_scene.group(1).split('/')[-2]
            key = (scene_short, int(m_epi.group(1)))
            by_key[key] = {
                'outcome': m_out.group(1).strip(),
                'target':  m_tgt.group(1).strip() if m_tgt else '?',
            }
    return by_key


def categorize(o):
    if o == 'success':       return 'success'
    if o == 'false positive': return 'FP'
    if 'stepout' in o:        return 'stepout'
    if 'stucking' in o:       return 'stuck'
    if 'no frontier' in o:    return 'no_frontier'
    return 'other'


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    record_path, subset_path = sys.argv[1], sys.argv[2]
    new = parse_record(record_path)
    subset = json.load(open(subset_path))
    print(f"Loaded {len(new)} episodes from {record_path}")
    print(f"Loaded {len(subset)} target episodes from {subset_path}")
    print()

    by_target = defaultdict(Counter)
    covered = 0
    missing = []
    outcomes = Counter()
    for s in subset:
        key = (s['scene'], int(s['epi']))
        if key not in new:
            missing.append(key); continue
        covered += 1
        c = categorize(new[key]['outcome'])
        outcomes[c] += 1
        by_target[s['target']][c] += 1

    if missing:
        print(f"WARN: {len(missing)} subset episodes missing from new record:")
        for k in missing[:10]:
            print(f"  {k}")
        if len(missing) > 10:
            print(f"  ...({len(missing)-10} more)")
        print()

    print(f"--- Coverage: {covered} / {len(subset)} subset episodes appeared in new record ---\n")

    print(f"{'category':<14} {'count':>6} {'pct':>7}")
    for cat in ['success','FP','stepout','stuck','no_frontier','other']:
        n = outcomes[cat]
        if n or cat in ('success','FP','stepout'):
            print(f"{cat:<14} {n:>6} {100*n/max(1,covered):>6.1f}%")
    print()

    print("--- Per-target breakdown ---")
    print(f"{'target':<14} {'N':>4}   " + " ".join(f"{c:>6}" for c in ['success','FP','stepout','stuck','no_fr','other']))
    for t in sorted(by_target):
        row = by_target[t]
        n = sum(row.values())
        print(f"{t:<14} {n:>4}   " + " ".join(f"{row[c]:>6}" for c in
                                                 ['success','FP','stepout','stuck','no_frontier','other']))


if __name__ == '__main__':
    main()
