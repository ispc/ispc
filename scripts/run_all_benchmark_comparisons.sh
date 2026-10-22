#!/bin/bash
# Copyright (c) 2025-2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

show_usage() {
    cat <<EOF
Usage: $0 [BENCHMARK_NAME] [FILTER]

Run benchmark comparisons for all targets built in base/ and peak/ directories.

ARGUMENTS:
  BENCHMARK_NAME  Name of benchmark test to run (default: 13_rotate)
  FILTER          Optional benchmark filter pattern (default: none)

EXAMPLES:
  $0                               # Run 13_rotate for all targets
  $0 08_masked_load_store          # Run specific benchmark for all targets
  $0 01_simple ".*vector"          # Run with benchmark filter

DIRECTORY STRUCTURE:
The script expects base/ and peak/ build directories created by
build_benchmarks.sh. Every benchmark is built once per target:
  base/bin/<benchmark>.<target>
  peak/bin/<benchmark>.<target>

BEHAVIOR:
1. Finds all base/bin/<benchmark>.<target> executables
2. Verifies the same executable exists in peak/bin
3. Runs performance comparison using run_and_compare_benchmarks.sh
4. Provides colored output and summary statistics

OUTPUT:
- Green: Successful comparisons
- Red: Errors and failed comparisons
- Yellow: Warnings and skipped targets
- Blue: Status information

The script will exit with code 1 if any comparisons fail.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi

BENCHMARK=${1:-13_rotate}
FILTER=${2:-}
SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)
COMPARE_SCRIPT="$SCRIPTS_DIR/run_and_compare_benchmarks.sh"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Running benchmark comparisons for all target pairs ===${NC}"
echo -e "${BLUE}Benchmark: ${BENCHMARK}${NC}"
if [ -n "$FILTER" ]; then
    echo -e "${BLUE}Filter: ${FILTER}${NC}"
fi
echo

BASE_DIR=base
PEAK_DIR=peak

# Find all targets the benchmark was built for
shopt -s nullglob
EXECUTABLES=("$BASE_DIR/bin/$BENCHMARK".*)

if [ ${#EXECUTABLES[@]} -eq 0 ]; then
    echo -e "${RED}Error: No '${BENCHMARK}.<target>' executables found in $BASE_DIR/bin/${NC}"
    exit 1
fi

# Track results
TOTAL_COMPARISONS=0
SUCCESSFUL_COMPARISONS=0
FAILED_COMPARISONS=0

for EXECUTABLE in "${EXECUTABLES[@]}"; do
    EXECUTABLE=${EXECUTABLE##*/}
    TARGET=${EXECUTABLE#${BENCHMARK}.}

    echo -e "${YELLOW}=== Processing target: ${TARGET} ===${NC}"

    if [ ! -f "$PEAK_DIR/bin/$EXECUTABLE" ]; then
        echo -e "${RED}Warning: Benchmark '$EXECUTABLE' not found in $PEAK_DIR/bin/${NC}"
        echo -e "${YELLOW}Skipping ${TARGET}${NC}"
        echo
        continue
    fi

    echo -e "${GREEN}Running comparison: $BASE_DIR vs $PEAK_DIR${NC}"

    # Run the comparison
    TOTAL_COMPARISONS=$((TOTAL_COMPARISONS + 1))

    if "$COMPARE_SCRIPT" "$BASE_DIR" "$PEAK_DIR" "$EXECUTABLE" ${FILTER:+"$FILTER"}; then
        SUCCESSFUL_COMPARISONS=$((SUCCESSFUL_COMPARISONS + 1))
        echo -e "${GREEN}✓ Comparison successful for ${TARGET}${NC}"
    else
        FAILED_COMPARISONS=$((FAILED_COMPARISONS + 1))
        echo -e "${RED}✗ Comparison failed for ${TARGET}${NC}"
    fi

    echo -e "${BLUE}------------------------------------------------${NC}"
    echo
done

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${BLUE}Total targets processed: ${TOTAL_COMPARISONS}${NC}"
echo -e "${GREEN}Successful comparisons: ${SUCCESSFUL_COMPARISONS}${NC}"
if [ $FAILED_COMPARISONS -gt 0 ]; then
    echo -e "${RED}Failed comparisons: ${FAILED_COMPARISONS}${NC}"
fi

if [ $TOTAL_COMPARISONS -eq 0 ]; then
    echo -e "${YELLOW}No valid target pairs found.${NC}"
    echo -e "${YELLOW}Expected executables: base/bin/<benchmark>.<target> and peak/bin/<benchmark>.<target>${NC}"
    exit 1
elif [ $FAILED_COMPARISONS -gt 0 ]; then
    exit 1
else
    echo -e "${GREEN}All comparisons completed successfully!${NC}"
fi
