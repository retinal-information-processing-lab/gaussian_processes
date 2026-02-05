#!/usr/bin/env python3
"""
analyze_experiment.py - Summarize and compare experiment results.

Usage:
    # Summary of one experiment
    python analyze_experiment.py --exp baseline

    # Per-config breakdown
    python analyze_experiment.py --exp baseline --details

    # Compare two experiments side by side
    python analyze_experiment.py --compare baseline new_feature

    # List all experiments
    python analyze_experiment.py --list
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_results(results_path):
    """Load results from JSONL file."""
    records = []
    if not results_path.exists():
        return records
    with open(results_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_metadata(metadata_path):
    """Load experiment metadata."""
    if not metadata_path.exists():
        return {}
    with open(metadata_path, 'r') as f:
        return yaml.safe_load(f)


def find_experiment(name, experiments_dir):
    """Find experiment folder by name (fuzzy match with date prefix)."""
    exact = experiments_dir / name
    if exact.exists():
        return exact

    matches = sorted(experiments_dir.glob(f"*_{name}"))
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Ambiguous name '{name}'. Matches:")
        for m in matches:
            print(f"  {m.name}")
        return None

    exploratory_dir = experiments_dir / 'exploratory'
    if exploratory_dir.exists():
        matches = sorted(exploratory_dir.glob(f"*_{name}"))
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous name '{name}'. Matches:")
            for m in matches:
                print(f"  exploratory/{m.name}")
            return None

    print(f"Experiment '{name}' not found.")
    return None


def summarize_experiment(exp_dir, details=False):
    """Print summary statistics for an experiment."""
    results_path = exp_dir / 'results.jsonl'
    metadata_path = exp_dir / 'metadata.yaml'

    records = load_results(results_path)
    metadata = load_metadata(metadata_path)

    if not records:
        print(f"No results found in {results_path}")
        return

    success = [r for r in records if r.get('status') == 'success']
    failed = [r for r in records if r.get('status') != 'success']

    print(f"Experiment: {exp_dir.name}")
    if metadata:
        print(f"  Description: {metadata.get('description', '-')}")
        print(f"  Created:     {metadata.get('created', '-')}")
        print(f"  Git commit:  {metadata.get('git_commit', '-')}")
        print(f"  Type:        {metadata.get('type', '-')}")
    print(f"  Results:     {len(success)} success, {len(failed)} failed")
    print()

    if not success:
        return

    # Summary table
    print(f"{'mode':<15} {'M':>4} {'ntrain':>6} {'seed':>5} {'cell':>4} {'test_r':>7} {'expl_var':>9} {'time_s':>7}")
    print("-" * 70)

    for r in sorted(success, key=lambda x: (x['mode'], x['M'], x.get('n_train', 0), x.get('seed', 0))):
        test_r = f"{r['test_r']:.4f}" if r.get('test_r') is not None else '-'
        expl_var = f"{r['explained_var']:.4f}" if r.get('explained_var') is not None else '-'
        time_s = f"{r['time_total_s']:.1f}" if r.get('time_total_s') is not None else '-'
        print(f"{r['mode']:<15} {r['M']:>4} {r.get('n_train', '-'):>6} {r.get('seed', '-'):>5} "
              f"{r.get('cell', '-'):>4} {test_r:>7} {expl_var:>9} {time_s:>7}")

    if details:
        print(f"\nDetailed parameters:")
        print(f"{'mode':<15} {'M':>4} {'final_A':>8} {'final_l0':>9} {'final_Amp':>10} "
              f"{'final_beta':>10} {'final_rho':>9} {'stopped':>8}")
        print("-" * 80)
        for r in sorted(success, key=lambda x: (x['mode'], x['M'])):
            stopped = 'yes' if r.get('stopped_early') else 'no'
            print(f"{r['mode']:<15} {r['M']:>4} "
                  f"{r.get('final_A', 0):>8.4f} "
                  f"{r.get('final_lambda0', 0):>9.4f} "
                  f"{r.get('final_Amp', 0):>10.4f} "
                  f"{r.get('final_beta', 0):>10.4f} "
                  f"{r.get('final_rho', 0):>9.4f} "
                  f"{stopped:>8}")

    # Mode-level aggregation
    by_mode = defaultdict(list)
    for r in success:
        by_mode[r['mode']].append(r)

    if len(by_mode) > 1:
        print(f"\nPer-mode summary:")
        for mode, recs in sorted(by_mode.items()):
            test_rs = [r['test_r'] for r in recs if r.get('test_r') is not None]
            times = [r['time_total_s'] for r in recs if r.get('time_total_s') is not None]
            if test_rs:
                mean_r = sum(test_rs) / len(test_rs)
                print(f"  {mode}: mean test_r={mean_r:.4f} ({len(recs)} runs, "
                      f"mean time={sum(times)/len(times):.1f}s)")


def compare_experiments(exp_dir1, exp_dir2):
    """Compare two experiments side by side."""
    records1 = load_results(exp_dir1 / 'results.jsonl')
    records2 = load_results(exp_dir2 / 'results.jsonl')

    success1 = {(r['mode'], r['M'], r.get('seed'), r.get('cell')): r
                for r in records1 if r.get('status') == 'success'}
    success2 = {(r['mode'], r['M'], r.get('seed'), r.get('cell')): r
                for r in records2 if r.get('status') == 'success'}

    common_keys = sorted(set(success1.keys()) & set(success2.keys()))

    if not common_keys:
        print("No common configurations to compare.")
        return

    name1 = exp_dir1.name
    name2 = exp_dir2.name

    print(f"Comparing: {name1} vs {name2}")
    print(f"{'config':<25} {'test_r_1':>9} {'test_r_2':>9} {'delta':>8} {'time_1':>7} {'time_2':>7}")
    print("-" * 75)

    for key in common_keys:
        r1, r2 = success1[key], success2[key]
        mode, M, seed, cell = key
        config_str = f"{mode} M={M} s={seed}"

        tr1 = r1.get('test_r')
        tr2 = r2.get('test_r')
        t1 = r1.get('time_total_s')
        t2 = r2.get('time_total_s')

        delta = ''
        if tr1 is not None and tr2 is not None:
            d = tr2 - tr1
            delta = f"{d:+.4f}"

        print(f"{config_str:<25} "
              f"{tr1:.4f if tr1 is not None else '-':>9} "
              f"{tr2:.4f if tr2 is not None else '-':>9} "
              f"{delta:>8} "
              f"{t1:.1f if t1 is not None else '-':>7} "
              f"{t2:.1f if t2 is not None else '-':>7}")


def list_experiments(experiments_dir):
    """List all experiments with descriptions."""
    all_experiments = []

    # Canonical experiments
    for exp_dir in sorted(experiments_dir.iterdir()):
        if exp_dir.is_dir() and exp_dir.name != 'exploratory':
            metadata = load_metadata(exp_dir / 'metadata.yaml')
            results = load_results(exp_dir / 'results.jsonl')
            n_success = len([r for r in results if r.get('status') == 'success'])
            all_experiments.append({
                'name': exp_dir.name,
                'type': metadata.get('type', 'canonical'),
                'description': metadata.get('description', '-'),
                'created': metadata.get('created', '-'),
                'n_results': n_success,
            })

    # Exploratory experiments
    exploratory_dir = experiments_dir / 'exploratory'
    if exploratory_dir.exists():
        for exp_dir in sorted(exploratory_dir.iterdir()):
            if exp_dir.is_dir():
                metadata = load_metadata(exp_dir / 'metadata.yaml')
                results = load_results(exp_dir / 'results.jsonl')
                n_success = len([r for r in results if r.get('status') == 'success'])
                all_experiments.append({
                    'name': f"exploratory/{exp_dir.name}",
                    'type': 'exploratory',
                    'description': metadata.get('description', '-'),
                    'created': metadata.get('created', '-'),
                    'n_results': n_success,
                })

    if not all_experiments:
        print("No experiments found.")
        return

    print(f"{'name':<45} {'type':<12} {'runs':>5} {'created':<20} {'description'}")
    print("-" * 110)
    for exp in all_experiments:
        print(f"{exp['name']:<45} {exp['type']:<12} {exp['n_results']:>5} "
              f"{exp['created']:<20} {exp['description']}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exp', type=str, help='Summarize one experiment')
    group.add_argument('--compare', type=str, nargs=2, metavar=('EXP1', 'EXP2'),
                       help='Compare two experiments')
    group.add_argument('--list', action='store_true', help='List all experiments')

    parser.add_argument('--details', action='store_true',
                        help='Show per-config parameter breakdown')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    experiments_dir = script_dir / 'experiments'

    if args.list:
        list_experiments(experiments_dir)
        return 0

    if args.exp:
        exp_dir = find_experiment(args.exp, experiments_dir)
        if exp_dir is None:
            return 1
        summarize_experiment(exp_dir, details=args.details)
        return 0

    if args.compare:
        exp_dir1 = find_experiment(args.compare[0], experiments_dir)
        exp_dir2 = find_experiment(args.compare[1], experiments_dir)
        if exp_dir1 is None or exp_dir2 is None:
            return 1
        compare_experiments(exp_dir1, exp_dir2)
        return 0


if __name__ == '__main__':
    sys.exit(main())
