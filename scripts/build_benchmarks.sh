#!/bin/bash
# Copyright (c) 2025, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

DEFAULT_TARGETS="avx512skx-x4 avx512skx-x8 avx512skx-x16 avx512skx-x32 avx512skx-x64 avx512icl-x4 avx512icl-x8 avx512icl-x16 avx512icl-x32 avx512icl-x64"

show_usage() {
    cat <<EOF
Usage: $0 [BASE_SHA] [PEAK_SHA] [BUILD_TARGETS]

Build ISPC benchmarks for performance comparison between two commits.

ARGUMENTS:
  BASE_SHA        Base commit SHA for comparison (default: HEAD^)
  PEAK_SHA        Peak commit SHA for comparison (default: HEAD)
  BUILD_TARGETS   Space-separated list of ISPC targets (default: all AVX-512 targets)

DEFAULT TARGETS:
  $DEFAULT_TARGETS

EXAMPLES:
  $0                                  # Compare HEAD^ vs HEAD with default targets
  $0 abc123 def456                    # Compare specific commits with default targets
  $0 HEAD~5 HEAD "avx2-i32x8 avx512skx-x16"  # Compare commits with custom targets

The script will:
1. Check out PEAK_SHA and build benchmarks for each target
2. Check out BASE_SHA and build benchmarks for each target
3. Create separate build directories for each target/version combination

Build directories will be named: {target}-peak and {target}-base
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi

BASE_SHA="${1:-HEAD^}"
PEAK_SHA="${2:-HEAD}"
BUILD_TARGETS="${3:-$DEFAULT_TARGETS}"

echo "Building benchmarks for ISPC with base SHA: $BASE_SHA and peak SHA: $PEAK_SHA"
echo "Targets: $BUILD_TARGETS"

# Convert space-separated targets to array
IFS=' ' read -ra TARGET_ARRAY <<< "$BUILD_TARGETS"

# Build peak version for each target
echo "Checking out peak SHA: $PEAK_SHA"
git checkout "$PEAK_SHA"

for target in "${TARGET_ARRAY[@]}"; do
    echo "Building peak version for target: $target"
    BUILD_DIR="${target}-peak"
    LOG_FILE="${target}-peak.build.log"

    rm -rf "$BUILD_DIR"

    cmake ../ -G Ninja \
        -DISPC_SLIM_BINARY=ON \
        -DISPC_INCLUDE_EXAMPLES=OFF \
        -DISPC_INCLUDE_RT=OFF \
        -DISPC_INCLUDE_UTILS=OFF \
        -DISPC_INCLUDE_TESTS=OFF \
        -DISPC_INCLUDE_BENCHMARKS=ON \
        -DARM_ENABLED=OFF \
        -B "$BUILD_DIR" \
        -DBENCHMARKS_ISPC_FLAGS="-O3 -woff" \
        -DBENCHMARKS_ISPC_TARGETS="$target"

    echo ""
    echo "Build phase:"
    echo "============"

    cmake --build "$BUILD_DIR"
done

# Build base version for each target
echo "Checking out base SHA: $BASE_SHA"
git checkout "$BASE_SHA"

for target in "${TARGET_ARRAY[@]}"; do
    echo "Building base version for target: $target"
    BUILD_DIR="${target}-base"

    rm -rf "$BUILD_DIR"

    cmake ../ -G Ninja \
        -DISPC_SLIM_BINARY=ON \
        -DISPC_INCLUDE_EXAMPLES=OFF \
        -DISPC_INCLUDE_RT=OFF \
        -DISPC_INCLUDE_UTILS=OFF \
        -DISPC_INCLUDE_TESTS=OFF \
        -DISPC_INCLUDE_BENCHMARKS=ON \
        -DARM_ENABLED=OFF \
        -B "$BUILD_DIR" \
        -DBENCHMARKS_ISPC_FLAGS="-O3 -woff" \
        -DBENCHMARKS_ISPC_TARGETS="$target"

    echo ""
    echo "Build phase:"
    echo "============"

    cmake --build "$BUILD_DIR"
done

echo "Build completed for all targets: $BUILD_TARGETS"
