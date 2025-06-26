#!/bin/bash
# Copyright (c) 2025, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

show_usage() {
    cat <<EOF
Usage: $0 [BASE_BUILD] [PEAK_BUILD] [TEST_NAME] [FILTER]

Run and compare ISPC benchmark performance between two builds.

ARGUMENTS:
  BASE_BUILD      Directory name of base build (default: avx512icl-x32)
  PEAK_BUILD      Directory name of peak build (default: avx512icl-x32-new)
  TEST_NAME       Name of the benchmark test to run (default: 08_masked_load_store)
  FILTER          Optional benchmark filter pattern (default: none)

EXAMPLES:
  $0                                          # Run default comparison
  $0 avx2-base avx2-peak 01_simple           # Compare specific builds and test
  $0 base peak 08_masked_load_store ".*load" # Run with benchmark filter

The script will:
1. Run the benchmark test on BASE_BUILD with 3 repetitions
2. Run the benchmark test on PEAK_BUILD with 3 repetitions
3. Compare results using Google Benchmark's compare.py tool
4. Show median performance differences

Output files:
- {BASE_BUILD}-{TEST_NAME}.json
- {PEAK_BUILD}-{TEST_NAME}.json

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

BASE=${1:-avx512icl-x32}
PEAK=${2:-avx512icl-x32-new}
TEST=${3:-08_masked_load_store}
FILTER=${4:-}

SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)

COMPARE_PY="$SCRIPTS_DIR/../benchmarks/vendor/google/benchmark/tools/compare.py"

# Build filter argument if provided
FILTER_ARG=""
if [ -n "$FILTER" ]; then
    FILTER_ARG="--benchmark_filter=$FILTER"
fi

setarch -R x86_64 taskset --cpu-list 1 ./$BASE/bin/$TEST --benchmark_repetitions=3 --benchmark_out=$BASE-$TEST.json --benchmark_out_format=json --benchmark_report_aggregates_only=true $FILTER_ARG
setarch -R x86_64 taskset --cpu-list 1 ./$PEAK/bin/$TEST --benchmark_repetitions=3 --benchmark_out=$PEAK-$TEST.json --benchmark_out_format=json --benchmark_report_aggregates_only=true $FILTER_ARG

python3 $COMPARE_PY benchmarks ./$BASE-$TEST.json ./$PEAK-$TEST.json | grep median
