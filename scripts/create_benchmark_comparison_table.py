#!/usr/bin/env python3
#
#  Copyright (c) 2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

import json
import os
import sys
import argparse
import csv
from collections import defaultdict, OrderedDict
import glob
import re

def parse_benchmark_file(filename):
    """Parse a benchmark JSON file and extract median timing results."""
    with open(filename, 'r') as f:
        data = json.load(f)

    benchmarks = {}
    for benchmark in data.get('benchmarks', []):
        name = benchmark.get('name', '')
        if name.endswith('_median'):
            # Extract the base name without _median suffix
            base_name = name[:-7]  # Remove '_median'
            cpu_time = benchmark.get('cpu_time', 0)
            benchmarks[base_name] = cpu_time

    return benchmarks

def extract_target_and_benchmark_from_filename(filename):
    """Extract target name and benchmark name from filename."""
    basename = os.path.basename(filename)
    # Pattern: target-base-benchmark.json or target-peak-benchmark.json
    match = re.match(r'(.+?)-(base|peak)-(.+)\.json$', basename)
    if match:
        target = match.group(1)
        variant = match.group(2)  # 'base' or 'peak'
        benchmark = match.group(3)
        return target, variant, benchmark
    return None, None, None

def find_json_files(directory='.'):
    """Find all benchmark JSON files in the directory."""
    pattern = os.path.join(directory, '*-{base,peak}-*.json')
    files = []
    for suffix in ['base', 'peak']:
        pattern = os.path.join(directory, f'*-{suffix}-*.json')
        files.extend(glob.glob(pattern))
    return files

def calculate_improvement(base_time, peak_time):
    """Calculate performance improvement percentage.

    Returns negative value if peak is faster (improvement),
    positive value if peak is slower (regression).
    """
    if base_time == 0:
        return 0
    return ((peak_time - base_time) / base_time) * 100

def format_percentage(value, threshold=None, use_colors=True, precision=1):
    """Format percentage value for display with optional color highlighting."""
    if value == 0:
        formatted = f"0.{'0' * precision}%"
    else:
        formatted = f"{value:+.{precision}f}%"

    # Add color highlighting if threshold is specified and colors are enabled
    if threshold is not None and use_colors and abs(value) > threshold:
        if value < 0:  # Improvement (faster)
            return f"\033[32m{formatted}\033[0m"  # Green
        else:  # Regression (slower)
            return f"\033[31m{formatted}\033[0m"  # Red

    return formatted

def print_table(comparison_data, targets, tests, output_format='table', threshold=None, precision=1):
    """Print comparison table in specified format."""

    if output_format == 'csv':
        print_csv_table(comparison_data, targets, tests, threshold, precision)
    elif output_format == 'markdown':
        print_markdown_table(comparison_data, targets, tests, threshold, precision)
    else:
        print_ascii_table(comparison_data, targets, tests, threshold, precision)

def print_ascii_table(comparison_data, targets, tests, threshold=None, precision=1):
    """Print ASCII table to stdout."""
    # Calculate column widths
    test_col_width = max(len(test) for test in tests) if tests else 20
    target_col_width = 12

    # Header
    header = f"{'Test':<{test_col_width}}"
    for target in targets:
        header += f" | {target:>{target_col_width}}"
    print(header)

    # Separator
    separator = "-" * test_col_width
    for target in targets:
        separator += " | " + "-" * target_col_width
    print(separator)

    # Data rows
    for test in tests:
        row = f"{test:<{test_col_width}}"
        for target in targets:
            value = comparison_data.get(target, {}).get(test, 0)
            # Get the formatted value with colors
            formatted_value = format_percentage(value, threshold, use_colors=True, precision=precision)
            # Get the plain value for width calculation
            plain_value = format_percentage(value, threshold=None, use_colors=False, precision=precision)

            # Calculate padding needed for proper alignment
            padding = target_col_width - len(plain_value)
            if padding > 0:
                row += f" | {' ' * padding}{formatted_value}"
            else:
                row += f" | {formatted_value}"
        print(row)

def print_csv_table(comparison_data, targets, tests, threshold=None, precision=1):
    """Print CSV table to stdout."""
    writer = csv.writer(sys.stdout)

    # Header
    header = ['Test'] + targets
    writer.writerow(header)

    # Data rows
    for test in tests:
        row = [test]
        for target in targets:
            value = comparison_data.get(target, {}).get(test, 0)
            # CSV format doesn't support colors, so just output raw values
            row.append(f"{value:.{precision}f}")
        writer.writerow(row)

def print_markdown_table(comparison_data, targets, tests, threshold=None, precision=1):
    """Print Markdown table to stdout."""
    # Header
    header = "| Test |"
    for target in targets:
        header += f" {target} |"
    print(header)

    # Separator
    separator = "|------|"
    for target in targets:
        separator += "------|"
    print(separator)

    # Data rows
    for test in tests:
        row = f"| {test} |"
        for target in targets:
            value = comparison_data.get(target, {}).get(test, 0)
            # Markdown doesn't support ANSI colors, so disable coloring
            formatted_value = format_percentage(value, threshold, use_colors=False, precision=precision)
            row += f" {formatted_value} |"
        print(row)

def save_to_file(comparison_data, targets, tests, filename, output_format, threshold=None, precision=1):
    """Save table to file."""
    original_stdout = sys.stdout
    try:
        with open(filename, 'w') as f:
            sys.stdout = f
            # Disable colors when saving to file
            print_table(comparison_data, targets, tests, output_format, threshold=None, precision=precision)
    finally:
        sys.stdout = original_stdout

def sort_targets_by_width(targets):
    """Sort targets by width (x4, x8, x16, x32, x64), keeping other targets alphabetically sorted."""
    width_order = {'x4': 1, 'x8': 2, 'x16': 3, 'x32': 4, 'x64': 5}

    def sort_key(target):
        # Extract width from target name (e.g., avx512skx-x16 -> x16)
        match = re.search(r'x(\d+)$', target)
        if match:
            width_num = int(match.group(1))
            width_str = f"x{width_num}"
            if width_str in width_order:
                # Sort by width order, then by target prefix for stability
                target_prefix = target.rsplit('-x', 1)[0]
                return (0, width_order[width_str], target_prefix)
        return (1, target)  # 1 for non-width targets, sort alphabetically

    return sorted(targets, key=sort_key)

def sort_tests_by_pattern(tests, sort_pattern):
    """Sort tests based on custom regexp pattern order."""
    if not sort_pattern:
        return sorted(tests)

    # Split the sort pattern into individual regexp patterns
    patterns = [p.strip() for p in sort_pattern.split('|')]

    # Compile regexp patterns
    compiled_patterns = []
    for pattern in patterns:
        try:
            compiled_patterns.append(re.compile(pattern))
        except re.error as e:
            print(f"Warning: Invalid regexp pattern '{pattern}': {e}", file=sys.stderr)
            continue

    def sort_key(test):
        # Find the first pattern that matches the test name
        for i, pattern in enumerate(compiled_patterns):
            if pattern.search(test):
                return (i, test)  # Primary sort by pattern order, secondary by test name
        # If no pattern matches, put it at the end
        return (len(compiled_patterns), test)

    return sorted(tests, key=sort_key)

def main():
    parser = argparse.ArgumentParser(
        description='Create benchmark comparison table from JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  create_benchmark_comparison_table.py                              # Create table from all JSON files
  create_benchmark_comparison_table.py --format csv                 # Output as CSV
  create_benchmark_comparison_table.py --format markdown            # Output as Markdown
  create_benchmark_comparison_table.py --output results.csv --format csv  # Save to CSV file
  create_benchmark_comparison_table.py --benchmark 13_rotate        # Only process 13_rotate benchmark
  create_benchmark_comparison_table.py --tests ".*1048576.*"        # Only show tests matching regexp (e.g., size 1048576)
  create_benchmark_comparison_table.py --tests ".*256.*" --format markdown # Show tests with 256 in name in markdown format
  create_benchmark_comparison_table.py --targets ".*skx.*"          # Only show targets matching regexp (e.g., skx targets)
  create_benchmark_comparison_table.py --threshold 5.0              # Highlight results with absolute value > 5.0 percent
  create_benchmark_comparison_table.py --precision 3                # Show 3 decimal places instead of default 1
  create_benchmark_comparison_table.py --sort "int8|int16|int32|int64|float|double" # Custom sort order using regexp patterns
        """
    )

    parser.add_argument('--format', choices=['table', 'csv', 'markdown'],
                       default='table', help='Output format (default: table)')
    parser.add_argument('--output', help='Output filename (default: stdout)')
    parser.add_argument('--benchmark', help='Filter by specific benchmark name')
    parser.add_argument('--tests', help='Filter tests by regexp pattern matching test names (e.g., ".*1048576.*")')
    parser.add_argument('--targets', help='Filter targets by regexp pattern (e.g., ".*skx.*")')
    parser.add_argument('--threshold', type=float, help='Highlight results with absolute value above this threshold (e.g., 5.0 for 5%%)')
    parser.add_argument('--sort', help='Custom sort order for test names using pipe-separated regexp patterns (e.g., "int8|int16|int32|int64|float|double")')
    parser.add_argument('--precision', type=int, default=1, help='Number of decimal places for percentage values (default: 1)')
    parser.add_argument('--directory', default='.', help='Directory to search for JSON files')

    args = parser.parse_args()

    # Find JSON files
    json_files = find_json_files(args.directory)
    if not json_files:
        print("Error: No benchmark JSON files found matching pattern '*-{base,peak}-*.json'", file=sys.stderr)
        sys.exit(1)

    # Group files by target and benchmark
    file_groups = defaultdict(lambda: defaultdict(dict))

    # Compile target regexp if provided
    target_pattern = None
    if args.targets:
        try:
            target_pattern = re.compile(args.targets)
        except re.error as e:
            print(f"Error: Invalid regexp pattern '{args.targets}': {e}", file=sys.stderr)
            sys.exit(1)

    for filename in json_files:
        target, variant, benchmark = extract_target_and_benchmark_from_filename(filename)
        if target and variant and benchmark:
            # Filter by benchmark if specified
            if args.benchmark and benchmark != args.benchmark:
                continue
            # Filter by target pattern if specified
            if target_pattern and not target_pattern.search(target):
                continue
            file_groups[target][benchmark][variant] = filename

    if not file_groups:
        print("Error: No valid target pairs found", file=sys.stderr)
        sys.exit(1)

    # Process each target-benchmark combination
    comparison_data = defaultdict(dict)  # comparison_data[target][test] = improvement_percentage
    all_tests = set()
    all_targets = set()

    for target, benchmarks in file_groups.items():
        for benchmark, variants in benchmarks.items():
            if 'base' not in variants or 'peak' not in variants:
                print(f"Warning: Missing base or peak variant for {target}-{benchmark}", file=sys.stderr)
                continue

            try:
                base_data = parse_benchmark_file(variants['base'])
                peak_data = parse_benchmark_file(variants['peak'])

                # Compare each test
                all_tests_in_benchmark = set(base_data.keys()) | set(peak_data.keys())
                for test in all_tests_in_benchmark:
                    # Filter by test name pattern if specified
                    if args.tests:
                        try:
                            test_pattern = re.compile(args.tests)
                            if not test_pattern.search(test):
                                continue
                        except re.error as e:
                            print(f"Error: Invalid regexp pattern '{args.tests}': {e}", file=sys.stderr)
                            sys.exit(1)

                    base_time = base_data.get(test, 0)
                    peak_time = peak_data.get(test, 0)

                    if base_time > 0 and peak_time > 0:
                        improvement = calculate_improvement(base_time, peak_time)
                        comparison_data[target][test] = improvement
                        all_tests.add(test)
                        all_targets.add(target)

            except Exception as e:
                print(f"Error processing {target}-{benchmark}: {e}", file=sys.stderr)
                continue

    if not all_tests:
        print("Error: No valid benchmark data found", file=sys.stderr)
        sys.exit(1)

    # Sort targets by width and tests for consistent output
    targets = sort_targets_by_width(all_targets)
    tests = sort_tests_by_pattern(all_tests, args.sort)

    # Print summary
    print(f"# Benchmark Comparison Results", file=sys.stderr)
    print(f"# Found {len(targets)} targets and {len(tests)} tests", file=sys.stderr)
    print(f"# Negative values indicate peak is faster than base (improvement)", file=sys.stderr)
    print(f"# Positive values indicate peak is slower than base (regression)", file=sys.stderr)
    print("", file=sys.stderr)

    # Output results
    if args.output:
        save_to_file(comparison_data, targets, tests, args.output, args.format, args.threshold, args.precision)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print_table(comparison_data, targets, tests, args.format, args.threshold, args.precision)

if __name__ == '__main__':
    main()
