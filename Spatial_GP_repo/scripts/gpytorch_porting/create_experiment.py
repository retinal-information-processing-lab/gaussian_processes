#!/usr/bin/env python3
"""
create_experiment.py - Create a new canonical experiment folder.

Copies the config template to a timestamped experiment folder and records
metadata (git state, description, timestamp).

Usage:
    python create_experiment.py --name baseline --desc "Post-cleanup baseline"
    python create_experiment.py --name baseline --config configs/canonical.yaml
"""

import argparse
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


def create_experiment(name, config_path, description, experiments_dir):
    """Create a new experiment folder with frozen config and metadata."""
    date_str = datetime.now().strftime('%Y-%m-%d')
    exp_name = f"{date_str}_{name}"
    exp_dir = experiments_dir / exp_name

    if exp_dir.exists():
        print(f"ERROR: Experiment folder already exists: {exp_dir}")
        print(f"  Choose a different name or delete the existing folder.")
        return None

    # Create folder
    exp_dir.mkdir(parents=True)

    # Copy config
    frozen_config = exp_dir / 'config.yaml'
    shutil.copy2(config_path, frozen_config)

    # Create metadata
    git_info = get_git_info(Path(__file__).parent)
    metadata = {
        'name': name,
        'full_name': exp_name,
        'description': description,
        'created': datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_info['commit'],
        'git_dirty': git_info['dirty'],
        'source_config': str(config_path),
        'type': 'canonical',
    }

    metadata_path = exp_dir / 'metadata.yaml'
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description='Create a new canonical experiment')
    parser.add_argument('--name', required=True, help='Experiment name (no spaces)')
    parser.add_argument('--desc', required=True, help='Short description of the experiment')
    parser.add_argument('--config', type=str, default='configs/canonical.yaml',
                        help='Config template to use (default: configs/canonical.yaml)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    experiments_dir = script_dir / 'experiments'

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    exp_dir = create_experiment(
        name=args.name,
        config_path=config_path,
        description=args.desc,
        experiments_dir=experiments_dir,
    )

    if exp_dir is None:
        return 1

    print(f"Experiment created: {exp_dir.name}")
    print(f"  Config:   {exp_dir / 'config.yaml'}")
    print(f"  Metadata: {exp_dir / 'metadata.yaml'}")
    print(f"\nNext steps:")
    print(f"  1. (Optional) Edit {exp_dir / 'config.yaml'}")
    print(f"  2. python run_experiment.py --exp {args.name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
