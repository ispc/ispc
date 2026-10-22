#!/bin/bash
# Copyright (c) 2025-2026, Intel Corporation
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
1. Check out PEAK_SHA and build benchmarks for all targets in "peak" directory
2. Check out BASE_SHA and build benchmarks for all targets in "base" directory

Every benchmark is built once per target, e.g. peak/bin/01_aossoa.avx512skx-x16.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi

BASE_SHA="${1:-HEAD^}"
PEAK_SHA="${2:-HEAD}"
BUILD_TARGETS="${3:-$DEFAULT_TARGETS}"

# Resolve before the first checkout, which changes what HEAD^ means.
BASE_SHA=$(git rev-parse --verify "$BASE_SHA^{commit}")
PEAK_SHA=$(git rev-parse --verify "$PEAK_SHA^{commit}")

# Both commits must build one executable per target, otherwise their binaries can't be compared.
for SHA in "$BASE_SHA" "$PEAK_SHA"; do
    if ! git grep -q BENCHMARKS_ISPC_TARGET_LIST "$SHA" -- :/benchmarks/cmake/AddBenchmark.cmake; then
        echo "Error: $SHA predates per-target benchmark executables, it can't be compared with this script."
        exit 1
    fi
done

echo "Building benchmarks for ISPC with base SHA: $BASE_SHA and peak SHA: $PEAK_SHA"
echo "Targets: $BUILD_TARGETS"

# CMake expects a comma separated list of targets
CMAKE_TARGETS="${BUILD_TARGETS// /,}"

build() {
    local BUILD_DIR="$1"

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
        -DBENCHMARKS_ISPC_TARGETS="$CMAKE_TARGETS"

    echo ""
    echo "Build phase:"
    echo "============"

    cmake --build "$BUILD_DIR"
}

echo "Checking out peak SHA: $PEAK_SHA"
git checkout "$PEAK_SHA"
build peak

echo "Checking out base SHA: $BASE_SHA"
git checkout "$BASE_SHA"
build base

echo "Build completed for all targets: $BUILD_TARGETS"
