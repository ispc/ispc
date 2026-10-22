#!/bin/bash
# Copyright (c) 2025-2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

show_usage() {
    cat <<EOF
Usage: $0 [BASE_BUILD] [PEAK_BUILD] [TEST_NAME] [FILTER]

Run and compare ISPC benchmark performance between two builds.

ARGUMENTS:
  BASE_BUILD      Directory name of base build (default: base)
  PEAK_BUILD      Directory name of peak build (default: peak)
  TEST_NAME       Name of the benchmark executable, i.e. <test>.<target>
                  (default: 08_masked_load_store.avx512icl-x32)
  FILTER          Optional benchmark filter pattern (default: none)

EXAMPLES:
  $0                                          # Run default comparison
  $0 base peak 01_aossoa.avx2-i32x8          # Compare specific builds and test
  $0 base peak 08_masked_load_store.avx2-i32x8 ".*load" # Run with benchmark filter

ENVIRONMENT:
  RUNS            Number of separate processes per build (default: 3)

The script will:
1. Run the benchmark test RUNS times on each build, alternating BASE_BUILD and
   PEAK_BUILD, with 3 repetitions per process
2. Keep the minimum of the per-process medians of every case. Repetitions in one
   process agree closely, but timings can shift from one process to another
3. Compare results using Google Benchmark's compare.py tool
4. Show median performance differences

Output files:
- {BASE_BUILD}-{TEST_NAME}.json, {PEAK_BUILD}-{TEST_NAME}.json (minimum over processes)
- {BASE_BUILD}-{TEST_NAME}.run<N>.json, {PEAK_BUILD}-{TEST_NAME}.run<N>.json

Requirements:
- Both build directories must exist with compiled benchmarks
- Google Benchmark compare.py tool must be available
- Tests run with CPU affinity (taskset) and ASLR disabled (setarch -R)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi

BASE=${1:-base}
PEAK=${2:-peak}
TEST=${3:-08_masked_load_store.avx512icl-x32}
FILTER=${4:-}
RUNS=${RUNS:-3}

SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)

COMPARE_PY="$SCRIPTS_DIR/../benchmarks/vendor/google/benchmark/tools/compare.py"

# Build filter argument if provided
FILTER_ARG=""
if [ -n "$FILTER" ]; then
    FILTER_ARG="--benchmark_filter=$FILTER"
fi

if [[ ! $RUNS =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNS must be a positive integer, got '$RUNS'" >&2
    exit 1
fi

# Results of an earlier invocation must not be compared if a run fails.
rm -f $BASE-$TEST.json $BASE-$TEST.run*.json $PEAK-$TEST.json $PEAK-$TEST.run*.json

for RUN in $(seq 1 "$RUNS"); do
    for BUILD in $BASE $PEAK; do
        setarch -R x86_64 taskset --cpu-list 1 ./$BUILD/bin/$TEST --benchmark_repetitions=3 --benchmark_out=$BUILD-$TEST.run$RUN.json --benchmark_out_format=json --benchmark_report_aggregates_only=true $FILTER_ARG || exit 1
    done
done

# Keep the median of the process with the minimum time for every case.
for BUILD in $BASE $PEAK; do
    python3 - $BUILD-$TEST.json $(seq -f "$BUILD-$TEST.run%g.json" 1 "$RUNS") <<'PY' || exit 1
import json
import sys

out, runs = sys.argv[1], sys.argv[2:]
best = {}
for path in runs:
    with open(path) as f:
        doc = json.load(f)
    for b in doc["benchmarks"]:
        if b.get("aggregate_name") != "median":
            continue
        if b["name"] not in best or b["cpu_time"] < best[b["name"]]["cpu_time"]:
            best[b["name"]] = b
doc["benchmarks"] = list(best.values())
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
PY
done

set -o pipefail
python3 $COMPARE_PY benchmarks ./$BASE-$TEST.json ./$PEAK-$TEST.json | grep median
