#!/usr/bin/env python3
"""
Compare two test results to identify performance differences.

Usage:
    python compare_test_results.py <test_dir_1> <test_dir_2>

Example:
    python compare_test_results.py videos/test_hm3dv2_val_safe videos/test_hm3dv2_val_mem_safe
"""

import sys
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_record_file(filepath: str) -> Dict:
    """Parse record.txt file and extract metrics."""
    results = {
        'episodes': [],
        'success_count': 0,
        'failure_types': defaultdict(int),
        'total_time': 0,
        'targets': defaultdict(lambda: {'success': 0, 'fail': 0}),
        'avg_success': 0,
        'avg_spl': 0,
        'avg_soft_spl': 0,
        'avg_dist_to_goal': 0,
    }

    current_episode = {}
    first_metrics_found = False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Parse success/failure
        if line.startswith('success or not:'):
            status = line.replace('success or not:', '').strip()
            current_episode['status'] = status

            if status == 'success':
                results['success_count'] += 1
                results['targets'][current_episode.get('target', 'unknown')]['success'] += 1
            else:
                results['failure_types'][status] += 1
                results['targets'][current_episode.get('target', 'unknown')]['fail'] += 1

        # Parse target
        elif line.startswith('target to find is'):
            target = line.replace('target to find is', '').strip()
            current_episode['target'] = target

        # Parse time - only keep the first one (which is the final total)
        elif 'seconds spend' in line and results['total_time'] == 0:
            match = re.search(r'([\d.]+)\s+seconds', line)
            if match:
                results['total_time'] = float(match.group(1))

        # Parse task number
        elif line.startswith('No.') and 'task is finished' in line:
            match = re.search(r'No\.(\d+)', line)
            if match:
                current_episode['task_num'] = int(match.group(1))
                results['episodes'].append(current_episode.copy())
                current_episode = {}

        # Parse metrics table - only use the FIRST occurrence (final results)
        elif 'Average Success' in line and not first_metrics_found:
            match = re.search(r'([\d.]+)%', line)
            if match:
                results['avg_success'] = float(match.group(1))
        elif 'Average SPL' in line and 'Soft' not in line and not first_metrics_found:
            match = re.search(r'([\d.]+)%', line)
            if match:
                results['avg_spl'] = float(match.group(1))
        elif 'Average Soft SPL' in line and not first_metrics_found:
            match = re.search(r'([\d.]+)%', line)
            if match:
                results['avg_soft_spl'] = float(match.group(1))
        elif 'Average Distance to Goal' in line and not first_metrics_found:
            match = re.search(r'([\d.]+)', line)
            if match:
                results['avg_dist_to_goal'] = float(match.group(1))
                first_metrics_found = True  # Mark that we've found the first complete metrics block

        i += 1

    return results


def compare_results(result1: Dict, result2: Dict, name1: str, name2: str):
    """Compare two test results and print analysis."""

    print("=" * 70)
    print(f"COMPARISON: {name1} vs {name2}")
    print("=" * 70)

    # Overall metrics
    print("\n## Overall Metrics")
    print("-" * 50)
    print(f"{'Metric':<25} {'Test 1':>12} {'Test 2':>12} {'Diff':>12}")
    print("-" * 50)

    metrics = [
        ('Success Rate', 'avg_success', '%', True),
        ('SPL', 'avg_spl', '%', True),
        ('Soft SPL', 'avg_soft_spl', '%', True),
        ('Dist to Goal', 'avg_dist_to_goal', 'm', False),
        ('Total Time', 'total_time', 's', False),
    ]

    for name, key, unit, higher_better in metrics:
        v1 = result1.get(key, 0)
        v2 = result2.get(key, 0)
        diff = v2 - v1
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"

        # Color indicator
        if higher_better:
            indicator = "better" if diff > 0 else ("worse" if diff < 0 else "same")
        else:
            indicator = "worse" if diff > 0 else ("better" if diff < 0 else "same")

        print(f"{name:<25} {v1:>10.2f}{unit} {v2:>10.2f}{unit} {diff_str:>10}{unit} ({indicator})")

    # Calculate per-episode time
    n_episodes = len(result1.get('episodes', []))
    if n_episodes > 0:
        time1 = result1.get('total_time', 0) / n_episodes
        time2 = result2.get('total_time', 0) / n_episodes
        diff = time2 - time1
        print(f"{'Time/Episode':<25} {time1:>10.2f}s {time2:>10.2f}s {diff:>+10.2f}s")

    # Failure type comparison
    print("\n## Failure Type Comparison")
    print("-" * 60)
    print(f"{'Failure Type':<30} {'Test 1':>10} {'Test 2':>10} {'Diff':>10}")
    print("-" * 60)

    all_failure_types = set(result1['failure_types'].keys()) | set(result2['failure_types'].keys())

    sorted_failures = []
    for ft in all_failure_types:
        v1 = result1['failure_types'].get(ft, 0)
        v2 = result2['failure_types'].get(ft, 0)
        sorted_failures.append((ft, v1, v2, v2 - v1))

    # Sort by absolute difference
    sorted_failures.sort(key=lambda x: abs(x[3]), reverse=True)

    for ft, v1, v2, diff in sorted_failures:
        diff_str = f"+{diff}" if diff > 0 else f"{diff}"
        indicator = "worse" if diff > 0 else ("better" if diff < 0 else "")
        print(f"{ft:<30} {v1:>10} {v2:>10} {diff_str:>10} {indicator}")

    total_fail_1 = sum(result1['failure_types'].values())
    total_fail_2 = sum(result2['failure_types'].values())
    print("-" * 60)
    print(f"{'TOTAL FAILURES':<30} {total_fail_1:>10} {total_fail_2:>10} {total_fail_2 - total_fail_1:>+10}")

    # Per-target comparison
    print("\n## Per-Target Success Rate")
    print("-" * 70)
    print(f"{'Target':<15} {'Test 1 Success':>15} {'Test 2 Success':>15} {'Diff':>10}")
    print("-" * 70)

    all_targets = set(result1['targets'].keys()) | set(result2['targets'].keys())

    for target in sorted(all_targets):
        t1 = result1['targets'][target]
        t2 = result2['targets'][target]
        total1 = t1['success'] + t1['fail']
        total2 = t2['success'] + t2['fail']

        rate1 = (t1['success'] / total1 * 100) if total1 > 0 else 0
        rate2 = (t2['success'] / total2 * 100) if total2 > 0 else 0

        print(f"{target:<15} {t1['success']:>5}/{total1:<5} ({rate1:>5.1f}%) "
              f"{t2['success']:>5}/{total2:<5} ({rate2:>5.1f}%) {rate2-rate1:>+8.1f}%")

    # Summary
    print("\n## Summary")
    print("-" * 50)

    success_diff = result2.get('avg_success', 0) - result1.get('avg_success', 0)
    time_diff = result2.get('total_time', 0) - result1.get('total_time', 0)
    stepout_diff = result2['failure_types'].get('stepout feasible', 0) - result1['failure_types'].get('stepout feasible', 0)

    print(f"Success Rate Change: {success_diff:+.2f}%")
    print(f"Total Time Change: {time_diff:+.0f}s ({time_diff/60:+.1f}min)")
    print(f"Stepout Feasible Change: {stepout_diff:+d}")

    if stepout_diff > 10 and time_diff > 1000:
        print("\n[!] WARNING: Significant increase in 'stepout feasible' failures")
        print("    This suggests VLM call overhead is causing timeouts.")
        print("    Consider:")
        print("    - Reducing VLM call frequency")
        print("    - Using faster VLM model (qwen3-vl-flash)")
        print("    - Caching VLM results")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    # Find record.txt files
    record1 = os.path.join(dir1, 'record.txt')
    record2 = os.path.join(dir2, 'record.txt')

    if not os.path.exists(record1):
        print(f"Error: {record1} not found")
        sys.exit(1)
    if not os.path.exists(record2):
        print(f"Error: {record2} not found")
        sys.exit(1)

    # Parse and compare
    result1 = parse_record_file(record1)
    result2 = parse_record_file(record2)

    name1 = os.path.basename(dir1)
    name2 = os.path.basename(dir2)

    compare_results(result1, result2, name1, name2)


if __name__ == '__main__':
    main()
