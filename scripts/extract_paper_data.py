#!/usr/bin/env python3
"""Extract paper-ready data from a completed Habitat run.

Inputs (all defaults):
  - videos/test_<dataset>_val/record.txt
  - /tmp/skillnav_logs/vlm_calls_default.jsonl     (BLIP2-ITM per-tick)
  - latest ~/.ros/log/<session>/vlm_memory_agent_node-*.log
  - latest ~/.ros/log/<session>/strategic_agent_node-*.log

Outputs:
  - Markdown report to stdout (or --out file)
  - Per-episode CSV at <out>.per_ep.csv

Metrics:
  - Overall SR / SPL / SoftSPL / DTG
  - Per-target SR + failure breakdown
  - LLM (Strategic) calls/ep + token usage + latency
  - VLM (Qwen verifier) calls/ep + latency
  - BLIP2-ITM calls/ep + latency
  - Wall-clock per ep
"""

import argparse
import json
import os
import re
import sys
import glob
import statistics as stats
from collections import defaultdict, Counter
from pathlib import Path


# ----------------- record.txt parsing -----------------

def parse_record(path):
    """Return list of per-episode dicts (oldest first), and the latest cumulative metrics."""
    text = open(path).read()
    eps = []
    for block in re.split(r'\n\s*\n', text):
        m_scene = re.search(r'Scene ID:\s*(\S+)', block)
        m_epi = re.search(r'Episode ID:\s*(\d+)', block)
        m_out = re.search(r'success or not:\s*(.+)', block)
        m_tgt = re.search(r'target to find is\s+(.+)', block)
        m_no = re.search(r'No\.(\d+) task is finished', block)
        m_t = re.search(r'([\d.]+) seconds spend in this task', block)
        # Average metrics shown in the table at TOP of block — these are cumulative
        m_succ = re.search(r'Average Success\s*\|\s*([\d.]+)%', block)
        m_spl  = re.search(r'Average SPL\s*\|\s*([\d.]+)%', block)
        m_soft = re.search(r'Average Soft SPL\s*\|\s*([\d.]+)%', block)
        m_dtg  = re.search(r'Average Distance to Goal\s*\|\s*([\d.]+)', block)
        if not (m_no and m_out and m_tgt):
            continue
        eps.append({
            'no': int(m_no.group(1)),
            'scene': m_scene.group(1).split('/')[-2] if m_scene else None,
            'epi':   int(m_epi.group(1)) if m_epi else None,
            'target': m_tgt.group(1).strip(),
            'outcome': m_out.group(1).strip(),
            'wall_s': float(m_t.group(1)) if m_t else None,
            'cum_succ': float(m_succ.group(1)) if m_succ else None,
            'cum_spl':  float(m_spl.group(1))  if m_spl  else None,
            'cum_soft': float(m_soft.group(1)) if m_soft else None,
            'cum_dtg':  float(m_dtg.group(1))  if m_dtg  else None,
        })
    # latest (top of file = newest) has the final cumulative metrics
    eps.sort(key=lambda e: e['no'])
    return eps


def categorize(outcome):
    if outcome == 'success': return 'success'
    if outcome == 'false positive': return 'false_positive'
    if outcome == 'stepout feasible': return 'stepout_feasible'
    if outcome == '[stepout] false negative': return 'stepout_fn'
    if outcome == 'no frontier': return 'no_frontier'
    if outcome == '[no frontier] false negative': return 'no_frontier_fn'
    if outcome == 'stucking': return 'stucking'
    if outcome == '[stucking] false negative': return 'stucking_fn'
    if outcome == 'infeasible': return 'infeasible'
    return 'other'


# ----------------- VLM (Qwen) verifier log -----------------

def parse_qwen_verifier_log(path):
    """Return list of (timestamp_s, target, decision, latency_s)."""
    if not path or not os.path.exists(path):
        return []
    out = []
    # Each entry looks like:
    # [rosout][INFO] 2026-05-22 21:07:08,333: [VLMMemoryAgent] decision=REJECT conf=0.98 latency=7.08s reason=...
    pat = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*decision=(\w+).*latency=([\d.]+)s',
    )
    target_pat = re.compile(r"target='([^']+)'")
    # Pair latest target with decision line (target comes on previous line typically)
    current_target = None
    for line in open(path, errors='ignore'):
        tm = target_pat.search(line)
        if tm:
            current_target = tm.group(1)
        m = pat.search(line)
        if m:
            out.append({
                'ts_str':  m.group(1),
                'decision': m.group(2),
                'latency_s': float(m.group(3)),
                'target': current_target,
            })
    return out


# ----------------- Strategic LLM log -----------------

def parse_strategic_log(path):
    """Return list of (timestamp_str, phase, w_sr, w_ig, conf, latency_s, prompt_tok, resp_tok)."""
    if not path or not os.path.exists(path):
        return []
    out = []
    # [StrategicAgentLLM] reply phase=0 w=(0.30,0.70) conf=0.50 latency=2.69s tok=(406,131) reason='...'
    pat = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*StrategicAgentLLM\] reply phase=(\d+) '
        r'w=\(([\d.]+),([\d.]+)\) conf=([\d.]+) latency=([\d.]+)s tok=\((\d+),(\d+)\)'
    )
    for line in open(path, errors='ignore'):
        m = pat.search(line)
        if m:
            out.append({
                'ts_str': m.group(1),
                'phase':  int(m.group(2)),
                'w_sr':   float(m.group(3)),
                'w_ig':   float(m.group(4)),
                'conf':   float(m.group(5)),
                'latency_s': float(m.group(6)),
                'prompt_tok': int(m.group(7)),
                'resp_tok':   int(m.group(8)),
            })
    return out


# ----------------- BLIP2-ITM JSONL -----------------

def parse_vlm_jsonl(path):
    """Return list of dicts (timestamp, episode_id, scene, target, latency_ms)."""
    if not path or not os.path.exists(path):
        return []
    out = []
    for line in open(path, errors='ignore'):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(d)
    return out


# ----------------- find latest ROS logs -----------------

def latest_ros_log(pattern):
    """Find most recently-modified ROS log matching pattern."""
    cands = []
    for d in glob.glob(os.path.expanduser('~/.ros/log/*/' + pattern)):
        try:
            cands.append((os.path.getmtime(d), d))
        except OSError:
            pass
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


# ----------------- analysis -----------------

def per_target_breakdown(eps):
    tally = defaultdict(lambda: Counter())
    for e in eps:
        tally[e['target']][categorize(e['outcome'])] += 1
    return tally


def failure_breakdown(eps):
    c = Counter()
    for e in eps:
        c[categorize(e['outcome'])] += 1
    return c


def quantile(arr, q):
    if not arr:
        return None
    s = sorted(arr)
    k = int(round(q * (len(s) - 1)))
    return s[k]


# ----------------- main report -----------------

def render_report(dataset, eps, qwen_calls, strat_calls, blip_calls):
    n = len(eps)
    last = eps[-1] if eps else None  # has cumulative metrics for the WHOLE run via cum_*
    out = []
    out.append(f"# Paper-ready data — **{dataset}** ({n} eps)")
    out.append('')
    out.append('Generated from `videos/test_' + dataset + '_val/record.txt` + ROS / call logs.')
    out.append('')
    out.append('## Overall metrics')
    out.append('')
    if last:
        out.append(f"| Metric | Value |")
        out.append(f"|---|---|")
        out.append(f"| Episodes completed | **{n}** |")
        out.append(f"| Success Rate (SR) | **{last['cum_succ']:.2f}%** |")
        out.append(f"| SPL | **{last['cum_spl']:.2f}%** |")
        out.append(f"| Soft SPL | **{last['cum_soft']:.2f}%** |")
        out.append(f"| Avg Distance to Goal | **{last['cum_dtg']:.4f} m** |")

        # `seconds spend in this task` in record.txt is actually CUMULATIVE
        # across the run, not per-episode. Compute per-ep deltas.
        cum = [e['wall_s'] for e in eps if e['wall_s'] is not None]
        if cum:
            total_s = cum[-1]  # last entry = cumulative total
            per_ep = []
            prev = 0.0
            for v in cum:
                per_ep.append(max(0.0, v - prev))
                prev = v
            out.append(f"| Wall-clock total | {total_s:.0f} s ({total_s/3600:.1f} h) |")
            out.append(f"| Wall-clock / ep (median) | {stats.median(per_ep):.1f} s |")
            out.append(f"| Wall-clock / ep (mean) | {sum(per_ep)/len(per_ep):.1f} s |")
    out.append('')

    # Per-target
    out.append('## Per-target')
    out.append('')
    out.append('| Target | N | Success | FP | Stepout | StepoutFN | NoFront | Stuck | Infeas | Other | SR% |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|')
    tally = per_target_breakdown(eps)
    cats = ['success','false_positive','stepout_feasible','stepout_fn',
            'no_frontier','stucking','infeasible','other']
    no_front_total = lambda c: c['no_frontier'] + c['no_frontier_fn']
    stuck_total    = lambda c: c['stucking'] + c['stucking_fn']
    for t in sorted(tally):
        c = tally[t]; tot = sum(c.values())
        sr_pct = 100 * c['success'] / tot if tot else 0
        out.append(f"| {t} | {tot} | {c['success']} | {c['false_positive']} | {c['stepout_feasible']} | "
                   f"{c['stepout_fn']} | {no_front_total(c)} | {stuck_total(c)} | {c['infeasible']} | "
                   f"{c['other']} | {sr_pct:.1f}% |")
    # Total row
    c = failure_breakdown(eps); tot = sum(c.values())
    sr_pct = 100 * c['success'] / tot if tot else 0
    out.append(f"| **TOTAL** | **{tot}** | **{c['success']}** | {c['false_positive']} | {c['stepout_feasible']} | "
               f"{c['stepout_fn']} | {no_front_total(c)} | {stuck_total(c)} | {c['infeasible']} | "
               f"{c['other']} | **{sr_pct:.1f}%** |")
    out.append('')

    # Failure breakdown
    out.append('## Failure mode breakdown')
    out.append('')
    out.append('| Outcome | Count | % of N |')
    out.append('|---|---|---|')
    for cat, cnt in c.most_common():
        out.append(f"| {cat} | {cnt} | {100*cnt/tot:.1f}% |")
    out.append('')

    # Qwen-VL verifier
    out.append('## VLM verifier (Qwen-VL via DashScope)')
    out.append('')
    if qwen_calls:
        decs = Counter(q['decision'] for q in qwen_calls)
        lats = [q['latency_s'] for q in qwen_calls]
        out.append(f"| Metric | Value |")
        out.append(f"|---|---|")
        out.append(f"| Total verification calls | {len(qwen_calls)} |")
        out.append(f"| Calls / episode (est.) | {len(qwen_calls)/max(1,n):.2f} |")
        out.append(f"| Decision: CONFIRM | {decs['CONFIRM']} ({100*decs['CONFIRM']/len(qwen_calls):.1f}%) |")
        out.append(f"| Decision: REJECT | {decs['REJECT']} ({100*decs['REJECT']/len(qwen_calls):.1f}%) |")
        out.append(f"| Decision: UNCERTAIN | {decs['UNCERTAIN']} ({100*decs['UNCERTAIN']/len(qwen_calls):.1f}%) |")
        out.append(f"| Latency mean | {sum(lats)/len(lats):.2f} s |")
        out.append(f"| Latency p50 | {quantile(lats, 0.50):.2f} s |")
        out.append(f"| Latency p95 | {quantile(lats, 0.95):.2f} s |")
    else:
        out.append('_(no Qwen verifier log found)_')
    out.append('')

    # Strategic LLM (Deepseek)
    out.append('## Strategic LLM (DeepSeek chat)')
    out.append('')
    if strat_calls:
        ptoks = [s['prompt_tok'] for s in strat_calls]
        rtoks = [s['resp_tok'] for s in strat_calls]
        lats  = [s['latency_s'] for s in strat_calls]
        total_toks = sum(ptoks) + sum(rtoks)
        out.append(f"| Metric | Value |")
        out.append(f"|---|---|")
        out.append(f"| Total LLM calls | {len(strat_calls)} |")
        out.append(f"| Calls / episode | {len(strat_calls)/max(1,n):.2f} |")
        out.append(f"| Total prompt tokens | {sum(ptoks):,} |")
        out.append(f"| Total response tokens | {sum(rtoks):,} |")
        out.append(f"| Total tokens (in+out) | **{total_toks:,}** |")
        out.append(f"| Tokens / call (mean) | {total_toks/len(strat_calls):.0f} |")
        out.append(f"| Tokens / episode | {total_toks/max(1,n):.0f} |")
        out.append(f"| Latency mean | {sum(lats)/len(lats):.2f} s |")
        out.append(f"| Latency p50 | {quantile(lats, 0.50):.2f} s |")
        out.append(f"| Latency p95 | {quantile(lats, 0.95):.2f} s |")
    else:
        out.append('_(no strategic LLM log found)_')
    out.append('')

    # BLIP2-ITM
    out.append('## BLIP2-ITM (per-tick value map)')
    out.append('')
    if blip_calls:
        lats = [c['latency_ms'] for c in blip_calls if 'latency_ms' in c]
        out.append(f"| Metric | Value |")
        out.append(f"|---|---|")
        out.append(f"| Total ITM cosine calls | {len(blip_calls):,} |")
        out.append(f"| Calls / episode (est.) | {len(blip_calls)/max(1,n):.1f} |")
        if lats:
            out.append(f"| Latency mean | {sum(lats)/len(lats):.1f} ms |")
            out.append(f"| Latency p50 | {quantile(lats, 0.50):.1f} ms |")
            out.append(f"| Latency p95 | {quantile(lats, 0.95):.1f} ms |")
    else:
        out.append('_(no BLIP2-ITM log found)_')
    out.append('')

    return '\n'.join(out)


def write_per_ep_csv(eps, path):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['no','scene','epi','target','outcome','wall_s',
                    'cum_succ','cum_spl','cum_soft','cum_dtg'])
        for e in eps:
            w.writerow([e['no'], e['scene'], e['epi'], e['target'], e['outcome'],
                        f"{e['wall_s']:.2f}" if e['wall_s'] is not None else '',
                        f"{e['cum_succ']:.2f}" if e['cum_succ'] is not None else '',
                        f"{e['cum_spl']:.2f}"  if e['cum_spl']  is not None else '',
                        f"{e['cum_soft']:.2f}" if e['cum_soft'] is not None else '',
                        f"{e['cum_dtg']:.4f}"  if e['cum_dtg']  is not None else ''])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='hm3dv2',
                    help='dataset suffix used in videos/test_<dataset>_val/')
    ap.add_argument('--record', default=None, help='override record.txt path')
    ap.add_argument('--qwen-log', default=None, help='override Qwen verifier log path')
    ap.add_argument('--strat-log', default=None, help='override Strategic LLM log path')
    ap.add_argument('--blip-jsonl', default='/tmp/skillnav_logs/vlm_calls_default.jsonl')
    ap.add_argument('--out', default=None, help='write report to file (default stdout)')
    args = ap.parse_args()

    rec_path = args.record or f'/home/yangz/Nav/SkillNav/videos/test_{args.dataset}_val/record.txt'
    qwen_path = args.qwen_log or latest_ros_log('vlm_memory_agent_node-*.log')
    strat_path = args.strat_log or latest_ros_log('strategic_agent_node-*.log')

    print(f"[extract] record:  {rec_path}", file=sys.stderr)
    print(f"[extract] qwen:    {qwen_path}", file=sys.stderr)
    print(f"[extract] strat:   {strat_path}", file=sys.stderr)
    print(f"[extract] blip:    {args.blip_jsonl}", file=sys.stderr)

    eps = parse_record(rec_path)
    qwen = parse_qwen_verifier_log(qwen_path)
    strat = parse_strategic_log(strat_path)
    blip = parse_vlm_jsonl(args.blip_jsonl)

    report = render_report(args.dataset, eps, qwen, strat, blip)
    if args.out:
        Path(args.out).write_text(report)
        print(f"[extract] wrote report to {args.out}", file=sys.stderr)
        csv_path = args.out.replace('.md', '.per_ep.csv')
        write_per_ep_csv(eps, csv_path)
        print(f"[extract] wrote per-ep CSV to {csv_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == '__main__':
    main()
