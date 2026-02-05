#!/usr/bin/env python3
"""
run_experiment.py - Run a canonical or exploratory experiment.

Canonical: reads frozen config.yaml from an experiment folder created by
create_experiment.py and runs all (mode x M x seed x cell) combinations.

Exploratory (--quick): creates a folder in experiments/exploratory/, freezes
config from configs/quick.yaml with optional CLI overrides, and runs immediately.

Usage:
    # Run a canonical experiment
    python run_experiment.py --exp baseline

    # Quick exploratory run (auto-creates + runs)
    python run_experiment.py --quick test_lr --mode vargp_direct --M 50 --seed 456

    # Resume an interrupted experiment (skips completed combos)
    python run_experiment.py --exp baseline --resume
"""

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def get_git_info(repo_dir):
    """Get current git commit and dirty status."""
    info = {'commit': 'unknown', 'dirty': True}
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True, cwd=repo_dir
        )
        info['commit'] = result.stdout.strip()
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, check=True, cwd=repo_dir
        )
        info['dirty'] = len(result.stdout.strip()) > 0
    except Exception:
        pass
    return info


def find_experiment(name, experiments_dir):
    """Find experiment folder by name (supports fuzzy matching with date prefix)."""
    # Exact match
    exact = experiments_dir / name
    if exact.exists():
        return exact

    # Try with date prefix pattern: YYYY-MM-DD_name
    matches = sorted(experiments_dir.glob(f"*_{name}"))
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Ambiguous experiment name '{name}'. Matches:")
        for m in matches:
            print(f"  {m.name}")
        return None

    # Also check exploratory/
    exploratory_dir = experiments_dir / 'exploratory'
    if exploratory_dir.exists():
        matches = sorted(exploratory_dir.glob(f"*_{name}"))
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous experiment name '{name}'. Matches:")
            for m in matches:
                print(f"  exploratory/{m.name}")
            return None

    print(f"Experiment '{name}' not found in {experiments_dir}")
    return None


def load_completed_combos(results_path):
    """Load already-completed (mode, M, n_train, seed, cell) combos from results.jsonl."""
    completed = set()
    if not results_path.exists():
        return completed
    with open(results_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = (record['mode'], record['M'], record['n_train'],
                       record['seed'], record['cell'])
                completed.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def run_canonical(exp_dir, resume=False):
    """Run a canonical experiment from a frozen config."""
    config_path = exp_dir / 'config.yaml'
    results_path = exp_dir / 'results.jsonl'

    if not config_path.exists():
        print(f"ERROR: No config.yaml in {exp_dir}")
        return 1

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp = config['experiment']
    modes = exp['modes']
    inducing_points = exp['inducing_points']
    n_trains = exp['n_train']
    seeds = exp['seeds']
    cells = exp['cells']

    # Build matrix
    matrix = list(itertools.product(modes, inducing_points, n_trains, seeds, cells))

    # Skip completed if resuming
    completed = set()
    if resume:
        completed = load_completed_combos(results_path)
        if completed:
            print(f"Resuming: {len(completed)} combos already completed")

    remaining = [m for m in matrix if m not in completed]

    print(f"Experiment: {exp_dir.name}")
    print(f"  Total combos:     {len(matrix)}")
    print(f"  Already done:     {len(completed)}")
    print(f"  Remaining:        {len(remaining)}")
    print(f"  Modes:            {modes}")
    print(f"  Inducing points:  {inducing_points}")
    print(f"  n_train:          {n_trains}")
    print(f"  Seeds:            {seeds}")
    print(f"  Cells:            {cells}")
    print(f"  n_iterations:     {exp['n_iterations']}")
    print(f"  Results:          {results_path}")
    print()

    if not remaining:
        print("All combos already completed. Nothing to do.")
        return 0

    # Import here (after printing config) to avoid slow import if just checking
    from run_single_mode import run_single_config, flatten_yaml_config

    passed = 0
    failed = 0

    for i, (mode, M, n_train, seed, cell) in enumerate(remaining):
        desc = f"[{i+1}/{len(remaining)}] mode={mode}, M={M}, ntrain={n_train}, seed={seed}, cell={cell}"
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")

        flat_config = flatten_yaml_config(config, mode, M, n_train, seed, cell)

        try:
            result = run_single_config(flat_config)
            if result is None:
                print(f"  FAILED (returned None)")
                failed += 1
                # Record failure
                fail_record = {
                    'mode': mode, 'M': M, 'n_train': n_train,
                    'seed': seed, 'cell': cell, 'status': 'failed',
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                }
                with open(results_path, 'a') as f:
                    f.write(json.dumps(fail_record) + '\n')
                continue

            # Build JSONL record (matching plan schema)
            record = {
                'mode': result['mode'],
                'M': result['M'],
                'n_train': result['n_train'],
                'seed': result['seed'],
                'cell': result['cell'],
                'status': 'success',
                'test_r': round(result['test_r'], 4) if result['test_r'] is not None else None,
                'explained_var': round(result['explained_var'], 4) if result['explained_var'] is not None else None,
                'final_loss': round(result['final_loss'], 4) if result['final_loss'] is not None else None,
                'time_total_s': round(result['train_time'], 2),
                'time_estep_s': round(result['time_estep_s'], 2) if result['time_estep_s'] is not None else None,
                'time_mstep_s': round(result['time_mstep_s'], 2) if result['time_mstep_s'] is not None else None,
                'final_A': round(result['final_A'], 6),
                'final_lambda0': round(result['final_lambda0'], 6),
                'final_Amp': round(result['final_Amp'], 6),
                'final_beta': round(result['final_beta'], 6),
                'final_rho': round(result['final_rho'], 6),
                'final_sigma_0': round(result['final_sigma_0'], 6),
                'final_eps_0x': round(result['final_eps_0x'], 6),
                'final_eps_0y': round(result['final_eps_0y'], 6),
                'gradient_mode': result.get('gradient_mode'),
                'n_iterations_run': result.get('n_iterations_run'),
                'stopped_early': result.get('stopped_early', False),
                'timestamp': result.get('timestamp', datetime.now().isoformat(timespec='seconds')),
            }

            with open(results_path, 'a') as f:
                f.write(json.dumps(record) + '\n')

            print(f"  test_r={record['test_r']}, time={record['time_total_s']}s")
            passed += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            fail_record = {
                'mode': mode, 'M': M, 'n_train': n_train,
                'seed': seed, 'cell': cell, 'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            with open(results_path, 'a') as f:
                f.write(json.dumps(fail_record) + '\n')

    print(f"\n{'='*60}")
    print(f"Experiment complete: {passed} passed, {failed} failed")
    print(f"Results: {results_path}")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


def run_quick(name, overrides, script_dir):
    """Create and run an exploratory experiment in one step."""
    quick_config_path = script_dir / 'configs' / 'quick.yaml'
    if not quick_config_path.exists():
        print(f"ERROR: Quick config not found: {quick_config_path}")
        return 1

    with open(quick_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides to experiment section
    exp = config['experiment']
    if overrides.get('mode'):
        exp['modes'] = [overrides['mode']]
    if overrides.get('M') is not None:
        exp['inducing_points'] = [overrides['M']]
    if overrides.get('seed') is not None:
        exp['seeds'] = [overrides['seed']]
    if overrides.get('cell') is not None:
        exp['cells'] = [overrides['cell']]
    if overrides.get('ntrain') is not None:
        exp['n_train'] = [overrides['ntrain']]

    # Create exploratory experiment folder
    date_str = datetime.now().strftime('%Y-%m-%d')
    exp_name = f"{date_str}_{name}"
    exp_dir = script_dir / 'experiments' / 'exploratory' / exp_name

    if exp_dir.exists():
        print(f"ERROR: Experiment folder already exists: {exp_dir}")
        return 1

    exp_dir.mkdir(parents=True)

    # Freeze config
    frozen_config = exp_dir / 'config.yaml'
    with open(frozen_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Create metadata
    git_info = get_git_info(script_dir)
    metadata = {
        'name': name,
        'full_name': exp_name,
        'description': f'Quick exploratory run: {name}',
        'created': datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_info['commit'],
        'git_dirty': git_info['dirty'],
        'source_config': 'configs/quick.yaml',
        'type': 'exploratory',
    }
    with open(exp_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    # Print all settings clearly before running
    print(f"Exploratory experiment: {name}")
    print(f"  Mode:       {exp['modes']}")
    print(f"  M:          {exp['inducing_points']}")
    print(f"  n_train:    {exp['n_train']}")
    print(f"  Seeds:      {exp['seeds']}")
    print(f"  Cells:      {exp['cells']}")
    print(f"  Iterations: {exp['n_iterations']}")
    print(f"  dtype:      {config.get('numerical', {}).get('dtype', 'float32')}")
    print(f"  Config frozen to: {frozen_config}")
    print(f"  Running...")
    print()

    return run_canonical(exp_dir, resume=False)


def main():
    parser = argparse.ArgumentParser(
        description='Run canonical or exploratory experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mutually exclusive: --exp or --quick
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exp', type=str,
                       help='Run a canonical experiment by name')
    group.add_argument('--quick', type=str, metavar='NAME',
                       help='Create and run an exploratory experiment')

    # Resume support
    parser.add_argument('--resume', action='store_true',
                        help='Resume an interrupted experiment (skip completed combos)')

    # Quick mode overrides (only used with --quick)
    parser.add_argument('--mode', type=str,
                        choices=['vargp_direct', 'default_gpy', 'vargp_old'],
                        help='Override mode for --quick')
    parser.add_argument('--M', type=int, help='Override M for --quick')
    parser.add_argument('--seed', type=int, help='Override seed for --quick')
    parser.add_argument('--cell', type=int, help='Override cell for --quick')
    parser.add_argument('--ntrain', type=int, help='Override n_train for --quick')

    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.exp:
        experiments_dir = script_dir / 'experiments'
        exp_dir = find_experiment(args.exp, experiments_dir)
        if exp_dir is None:
            return 1
        return run_canonical(exp_dir, resume=args.resume)

    elif args.quick:
        overrides = {
            'mode': args.mode,
            'M': args.M,
            'seed': args.seed,
            'cell': args.cell,
            'ntrain': args.ntrain,
        }
        return run_quick(args.quick, overrides, script_dir)


if __name__ == '__main__':
    sys.exit(main())
