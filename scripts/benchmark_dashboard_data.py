#!/usr/bin/env python3
# Copyright (c) 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Selects the benchmark cases of one target (one page) shown on the performance dashboard.

Input files are merged results (see merge_benchmark_runs.py) named <benchmark>.<target>.json. The
cases are written in Google Benchmark JSON format, as github-action-benchmark reads it, to two
files: <target>.gated.json, where a regression fails the weekly job, and <target>.not-gated.json.
Case names get the benchmark name as a prefix: 07_loop_unroll_varying/unroll_varying_for_float_1/4096.

With --filter, prints the --benchmark_filter of the cases shown on the dashboard instead, from the
results of a dry run, so that only these cases are measured.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

# Every benchmark function is shown with one size: the closest one to this. Such a working set fits
# in L1/L2, so the time mostly depends on the generated code.
DASHBOARD_SIZE = 4096

# Benchmarks that mostly measure memory traffic: their largest size is shown as well.
MEMORY_BENCHMARKS = {"01_aossoa", "02_soaaos", "test02_array"}

# Benchmarks that are too noisy to fail the weekly job.
NOT_GATED = {"05_packed_load_store", "09_shuffle", "17_pdep_pext"}
# Same for avx2-* targets: variable shift/rotate go through the stack, whose alignment changes the timing.
NOT_GATED_AVX2 = {"13_rotate", "14_shift"}


def parse_case(run_name):
    """Splits 'func/4096/0' into 'func' and the size 4096 (None if there is no numeric argument)."""
    func, _, args = run_name.partition("/")
    size = args.split("/")[0]
    return func, int(size) if size.isdigit() else None


def select(bench, run_names):
    """Returns the run names of one benchmark that are shown on the dashboard."""
    run_names = [n for n in run_names if not n.startswith("warm_up/")]
    sizes = defaultdict(set)
    for name in run_names:
        func, size = parse_case(name)
        if size is not None:
            sizes[func].add(size)
    shown = set()
    for func, func_sizes in sizes.items():
        shown.add((func, min(func_sizes, key=lambda s: abs(math.log(max(s, 1) / DASHBOARD_SIZE)))))
        if bench in MEMORY_BENCHMARKS:
            shown.add((func, max(func_sizes)))
    return [n for n in run_names if parse_case(n)[1] is None or parse_case(n) in shown]


def benchmark_filter(bench, run_names):
    """Returns a --benchmark_filter (POSIX extended regex) of the shown cases and warm_up."""
    names = [re.sub(r"([][.(){}*+?|^$\\])", r"\\\1", n) for n in select(bench, run_names)]
    return "^(" + "|".join(["warm_up/.*"] + names) + ")$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="ISPC target of the results, or dispatch for the auto-dispatch binaries")
    parser.add_argument("--out-dir", help="directory for the output files")
    parser.add_argument("--filter", action="store_true", help="print the filter of one dry run result")
    parser.add_argument("results", nargs="+", help="merged results <benchmark>.<target>.json")
    args = parser.parse_args()

    if args.filter:
        if len(args.results) != 1:
            parser.error("--filter needs one dry run result <benchmark>.<target>.*.json")
        with open(args.results[0]) as f:
            doc = json.load(f)
        bench = os.path.basename(args.results[0]).split(".")[0]
        print(benchmark_filter(bench, dict.fromkeys(b["run_name"] for b in doc["benchmarks"])))
        return
    if not args.target or not args.out_dir:
        parser.error("--target and --out-dir are required")

    gated, not_gated = [], []
    context = None
    for path in sorted(args.results):
        bench = os.path.basename(path)
        suffix = "." + args.target + ".json"
        if not bench.endswith(suffix):
            sys.exit(f"{path}: expected a file name ending with {suffix}")
        bench = bench[: -len(suffix)]
        with open(path) as f:
            doc = json.load(f)
        context = context or doc["context"]
        shown = set(select(bench, [b["run_name"] for b in doc["benchmarks"]]))
        skip_gate = bench in NOT_GATED or (args.target.startswith("avx2-") and bench in NOT_GATED_AVX2)
        for b in doc["benchmarks"]:
            if b["run_name"] in shown:
                b = dict(b, name=bench + "/" + b["run_name"])
                (not_gated if skip_gate else gated).append(b)

    if not gated:
        sys.exit(f"{args.target}: no gated cases")
    os.makedirs(args.out_dir, exist_ok=True)
    for name, cases in (("gated", gated), ("not-gated", not_gated)):
        with open(os.path.join(args.out_dir, f"{args.target}.{name}.json"), "w") as f:
            json.dump({"context": context, "benchmarks": cases}, f, indent=2)
    print(f"{args.target}: {len(gated)} gated and {len(not_gated)} not gated cases")


if __name__ == "__main__":
    main()
