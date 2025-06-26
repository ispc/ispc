#!/bin/bash
# Copyright (c) 2025, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

show_usage() {
    cat <<EOF
Usage: $0 [BENCHMARK_NAME] [FILTER]

Run benchmark comparisons for all target pairs in the current directory.

ARGUMENTS:
  BENCHMARK_NAME  Name of benchmark test to run (default: 13_rotate)
  FILTER          Optional benchmark filter pattern (default: none)

EXAMPLES:
  $0                               # Run 13_rotate for all target pairs
  $0 08_masked_load_store          # Run specific benchmark for all targets
  $0 01_simple ".*vector"          # Run with benchmark filter

DIRECTORY STRUCTURE:
The script expects paired directories in the format:
  <target>-base/    - Base version build
  <target>-peak/    - Peak version build

Example directory structure:
  avx512skx-x16-base/
  avx512skx-x16-peak/
  avx512icl-x32-base/
  avx512icl-x32-peak/

BEHAVIOR:
1. Scans current directory for directories ending with '-base'
2. For each base directory, looks for corresponding '-peak' directory
3. Verifies benchmark executable exists in both directories
4. Runs performance comparison using run_and_compare_benchmarks.sh
5. Provides colored output and summary statistics

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

# Find all base directories
BASE_DIRS=$(ls -d */ 2>/dev/null | grep '\-base/$' | sed 's|/$||' | sort)

if [ -z "$BASE_DIRS" ]; then
    echo -e "${RED}Error: No base directories found (directories ending with '-base')${NC}"
    echo -e "${YELLOW}Available directories:${NC}"
    ls -d */ 2>/dev/null | sed 's|/$||'
    exit 1
fi

# Track results
TOTAL_COMPARISONS=0
SUCCESSFUL_COMPARISONS=0
FAILED_COMPARISONS=0

for BASE_DIR in $BASE_DIRS; do
    # Extract the target prefix (everything before '-base')
    TARGET_PREFIX=${BASE_DIR%-base}
    PEAK_DIR="${TARGET_PREFIX}-peak"

    echo -e "${YELLOW}=== Processing target: ${TARGET_PREFIX} ===${NC}"

    # Check if corresponding peak directory exists
    if [ ! -d "$PEAK_DIR" ]; then
        echo -e "${RED}Warning: Peak directory '$PEAK_DIR' not found for base '$BASE_DIR'${NC}"
        echo -e "${YELLOW}Skipping ${TARGET_PREFIX}${NC}"
        echo
        continue
    fi

    # Check if benchmark executable exists in both directories
    if [ ! -f "$BASE_DIR/bin/$BENCHMARK" ]; then
        echo -e "${RED}Warning: Benchmark '$BENCHMARK' not found in $BASE_DIR/bin/${NC}"
        echo -e "${YELLOW}Skipping ${TARGET_PREFIX}${NC}"
        echo
        continue
    fi

    if [ ! -f "$PEAK_DIR/bin/$BENCHMARK" ]; then
        echo -e "${RED}Warning: Benchmark '$BENCHMARK' not found in $PEAK_DIR/bin/${NC}"
        echo -e "${YELLOW}Skipping ${TARGET_PREFIX}${NC}"
        echo
        continue
    fi

    echo -e "${GREEN}Running comparison: $BASE_DIR vs $PEAK_DIR${NC}"

    # Run the comparison
    TOTAL_COMPARISONS=$((TOTAL_COMPARISONS + 1))

    if [ -n "$FILTER" ]; then
        if "$COMPARE_SCRIPT" "$BASE_DIR" "$PEAK_DIR" "$BENCHMARK" "$FILTER"; then
            SUCCESSFUL_COMPARISONS=$((SUCCESSFUL_COMPARISONS + 1))
            echo -e "${GREEN}✓ Comparison successful for ${TARGET_PREFIX}${NC}"
        else
            FAILED_COMPARISONS=$((FAILED_COMPARISONS + 1))
            echo -e "${RED}✗ Comparison failed for ${TARGET_PREFIX}${NC}"
        fi
    else
        if "$COMPARE_SCRIPT" "$BASE_DIR" "$PEAK_DIR" "$BENCHMARK"; then
            SUCCESSFUL_COMPARISONS=$((SUCCESSFUL_COMPARISONS + 1))
            echo -e "${GREEN}✓ Comparison successful for ${TARGET_PREFIX}${NC}"
        else
            FAILED_COMPARISONS=$((FAILED_COMPARISONS + 1))
            echo -e "${RED}✗ Comparison failed for ${TARGET_PREFIX}${NC}"
        fi
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
    echo -e "${YELLOW}Expected directory structure: <target>-base and <target>-peak${NC}"
    exit 1
elif [ $FAILED_COMPARISONS -gt 0 ]; then
    exit 1
else
    echo -e "${GREEN}All comparisons completed successfully!${NC}"
fi
