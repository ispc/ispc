#!/usr/bin/env python3
# Copyright (c) 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Merges Google Benchmark JSON results of separate runs of the same benchmark executable.

Runs must be made with --benchmark_repetitions and --benchmark_report_aggregates_only. Repetitions
in one process agree closely, but timings can shift from one process to another, so for every case
the median of the process with the minimum time (cpu_time or real_time) is kept.
"""

import argparse
import json
import sys


def merge(paths, key):
    doc = None
    best = {}
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        for b in doc["benchmarks"]:
            if b.get("aggregate_name") == "median":
                best[b["name"]] = min(best.get(b["name"], b), b, key=lambda x: x[key])
    doc["benchmarks"] = list(best.values())
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", choices=["cpu_time", "real_time"], default="cpu_time", help="time to minimize")
    parser.add_argument("output", help="merged JSON file to write")
    parser.add_argument("runs", nargs="+", help="JSON files of the separate runs")
    args = parser.parse_args()
    doc = merge(args.runs, args.key)
    if not doc["benchmarks"]:
        sys.exit(f"No medians in {', '.join(args.runs)}: run with --benchmark_repetitions=2 or more")
    with open(args.output, "w") as f:
        json.dump(doc, f, indent=2)


if __name__ == "__main__":
    main()
