#!/usr/bin/env python3
"""
query_benchmark.py - Query and filter benchmark results from JSONL files.

Reads results/benchmark_results.jsonl and filters by various criteria.

Usage:
    # Show all results
    python query_benchmark.py

    # Filter by mode
    python query_benchmark.py --mode vargp_style

    # Filter by M and seed
    python query_benchmark.py --M 100 --seed 123

    # Filter by multiple criteria
    python query_benchmark.py --mode vargp_style --M 200 --ntrain 2000

    # Compare two seeds side by side
    python query_benchmark.py --compare-seeds 123 456

    # Show only specific fields
    python query_benchmark.py --fields mode M test_r explained_var time_total_s

    # Raw JSON output (for piping to jq)
    python query_benchmark.py --raw

    # Use custom input file
    python query_benchmark.py --input results/custom.jsonl
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    """Load all records from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line: {e}", file=sys.stderr)
    return records


def filter_records(records, filters):
    """Filter records by key-value criteria."""
    result = records
    for key, value in filters.items():
        if value is not None:
            result = [r for r in result if r.get(key) == value]
    return result


def format_value(v, width=10):
    """Format a value for display."""
    if v is None:
        return '-'.center(width)
    elif isinstance(v, float):
        return f"{v:.4f}".rjust(width)
    elif isinstance(v, int):
        return str(v).rjust(width)
    else:
        return str(v)[:width].ljust(width)


def print_table(records, fields):
    """Print records as a formatted table."""
    if not records:
        print("No matching records found.")
        return

    # Calculate column widths
    widths = {}
    for field in fields:
        widths[field] = max(len(field), max(len(format_value(r.get(field)).strip()) for r in records))

    # Header
    header = ' | '.join(field.ljust(widths[field]) for field in fields)
    print(header)
    print('-' * len(header))

    # Rows
    for record in records:
        row = ' | '.join(format_value(record.get(field), widths[field]).ljust(widths[field]) for field in fields)
        print(row)


def compare_seeds(records, seed1, seed2, key_fields=None):
    """Compare results between two seeds side by side."""
    if key_fields is None:
        key_fields = ['mode', 'M', 'ntrain']

    # Group by configuration
    by_config = defaultdict(dict)
    for r in records:
        config_key = tuple(r.get(f) for f in key_fields)
        seed = r.get('seed')
        if seed in (seed1, seed2):
            by_config[config_key][seed] = r

    if not by_config:
        print("No matching records for comparison.")
        return

    # Print comparison table
    metrics = ['test_r', 'explained_var', 'time_total_s']

    # Header
    header_parts = key_fields + [f"{m}_{seed1}" for m in metrics] + [f"{m}_{seed2}" for m in metrics] + ['delta_expl']
    print(' | '.join(f"{h:>12}" for h in header_parts))
    print('-' * (14 * len(header_parts)))

    # Rows
    for config_key in sorted(by_config.keys()):
        seeds_data = by_config[config_key]
        if seed1 not in seeds_data or seed2 not in seeds_data:
            continue

        r1, r2 = seeds_data[seed1], seeds_data[seed2]

        row_parts = [str(v) for v in config_key]

        for m in metrics:
            v = r1.get(m)
            row_parts.append(f"{v:.4f}" if v is not None else '-')

        for m in metrics:
            v = r2.get(m)
            row_parts.append(f"{v:.4f}" if v is not None else '-')

        # Delta for explained_var
        e1, e2 = r1.get('explained_var'), r2.get('explained_var')
        if e1 is not None and e2 is not None:
            delta = e2 - e1
            row_parts.append(f"{delta:+.4f}")
        else:
            row_parts.append('-')

        print(' | '.join(f"{p:>12}" for p in row_parts))


def main():
    parser = argparse.ArgumentParser(
        description='Query benchmark results from JSONL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input file
    parser.add_argument('--input', '-i', type=str,
                        default='results/benchmark_results.jsonl',
                        help='Input JSONL file')

    # Filter options
    parser.add_argument('--mode', type=str, help='Filter by mode')
    parser.add_argument('--M', type=int, help='Filter by M (inducing points)')
    parser.add_argument('--seed', type=int, help='Filter by seed')
    parser.add_argument('--ntrain', type=int, help='Filter by ntrain')
    parser.add_argument('--commit', type=str, help='Filter by commit hash')

    # Output options
    parser.add_argument('--fields', type=str, nargs='+',
                        default=['mode', 'M', 'ntrain', 'seed', 'test_r', 'explained_var', 'time_total_s'],
                        help='Fields to display')
    parser.add_argument('--raw', action='store_true',
                        help='Output raw JSON (one record per line)')
    parser.add_argument('--compare-seeds', type=int, nargs=2, metavar=('SEED1', 'SEED2'),
                        help='Compare results between two seeds')

    # Sorting
    parser.add_argument('--sort-by', type=str, default='timestamp',
                        help='Sort by field (default: timestamp)')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse sort order')

    args = parser.parse_args()

    # Load data
    input_path = Path(__file__).parent / args.input
    records = load_jsonl(input_path)

    if not records:
        print(f"No records found in {input_path}")
        return 1

    print(f"Loaded {len(records)} records from {input_path}\n")

    # Apply filters
    filters = {
        'mode': args.mode,
        'M': args.M,
        'seed': args.seed,
        'ntrain': args.ntrain,
        'commit': args.commit,
    }
    filtered = filter_records(records, filters)

    # Sort
    if args.sort_by and filtered:
        filtered = sorted(
            filtered,
            key=lambda r: (r.get(args.sort_by) is None, r.get(args.sort_by)),
            reverse=args.reverse
        )

    # Output
    if args.compare_seeds:
        compare_seeds(records, args.compare_seeds[0], args.compare_seeds[1])
    elif args.raw:
        for r in filtered:
            print(json.dumps(r))
    else:
        print_table(filtered, args.fields)

    return 0


if __name__ == '__main__':
    sys.exit(main())
