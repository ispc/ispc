#!/bin/bash -e
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Builds ISPC with the benchmarks and runs them for the performance dashboard (benchmarks-weekly.yml).
# Runs in the docker/ubuntu/22.04/benchmarks container:
#   run-benchmarks.sh SOURCE_DIR BUILD_DIR RESULTS_DIR
#
# Environment:
#   TARGETS        Comma separated ISPC targets. The CPU must support all of them.
#   RUNS           Number of separate processes per benchmark (default: 3).
#   REPETITIONS    Repetitions in every process (default: 5).
#   MIN_TIME       Minimum time of every repetition (default: 0.03s).
#   BENCHMARK_CPU  CPU to run the benchmarks on (default: 1).
#
# Only the cases shown on the dashboard are run (see scripts/benchmark_dashboard_data.py). The
# auto-dispatch binary 16_dispatcher_overhead.dispatch is shown on its own page, "dispatch".
# RESULTS_DIR gets <benchmark>.<target>.json (the minimum of the per-process medians, see
# scripts/merge_benchmark_runs.py), dashboard/<page>.{gated,not-gated}.json, targets.json (the
# dashboard pages), raw results and logs of every process.

shopt -s nullglob

SRC=$1
BUILD=$2
RESULTS=$3
TARGETS=${TARGETS:-avx2-i32x8}
RUNS=${RUNS:-3}
REPETITIONS=${REPETITIONS:-5}
MIN_TIME=${MIN_TIME:-0.03s}
BENCHMARK_CPU=${BENCHMARK_CPU:-1}

# Medians are only reported with 2 or more repetitions.
if [[ ! $RUNS =~ ^[1-9][0-9]*$ || ! $REPETITIONS =~ ^[1-9][0-9]*$ || $REPETITIONS -lt 2 ]]; then
    echo "RUNS must be a positive integer and REPETITIONS at least 2, got '$RUNS' and '$REPETITIONS'" >&2
    exit 1
fi

TARGET_LIST=()
for TARGET in ${TARGETS//,/ }; do
    if [[ " ${TARGET_LIST[*]} " != *" $TARGET "* ]]; then
        TARGET_LIST+=("$TARGET")
    fi
done

mkdir -p "$RESULTS/raw" "$RESULTS/logs"
lscpu > "$RESULTS/lscpu.txt"

# The C++ code is compiled with clang of the LLVM package, so reference timings only change with LLVM.
cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DISPC_SLIM_BINARY=ON \
    -DISPC_INCLUDE_BENCHMARKS=ON \
    -DISPC_INCLUDE_EXAMPLES=OFF \
    -DISPC_INCLUDE_RT=OFF \
    -DISPC_INCLUDE_TESTS=OFF \
    -DISPC_INCLUDE_UTILS=OFF \
    -DBENCHMARKS_ISPC_TARGETS="$(IFS=,; echo "${TARGET_LIST[*]}")"
# A dispatch binary of an earlier configuration in a reused build directory would be run as well.
rm -f "$BUILD"/bin/*.dispatch
cmake --build "$BUILD"
"$BUILD/bin/ispc" --version > "$RESULTS/versions.txt"
clang++ --version | head -1 >> "$RESULTS/versions.txt"

# 04_intrinsics (2520 cases per target) is not run: it is too big for the dashboard.
EXES=()
for TARGET in "${TARGET_LIST[@]}"; do
    COUNT=${#EXES[@]}
    for EXE in "$BUILD"/bin/*."$TARGET"; do
        case $(basename "$EXE") in
            *_intrinsics.*) ;;
            *) EXES+=("$(basename "$EXE")") ;;
        esac
    done
    if [ ${#EXES[@]} -eq "$COUNT" ]; then
        echo "No benchmarks were built for $TARGET" >&2
        exit 1
    fi
done
PAGES=("${TARGET_LIST[@]}")
DISPATCH=("$BUILD"/bin/*.dispatch)
if [ ${#DISPATCH[@]} -ne 0 ]; then
    EXES+=("${DISPATCH[@]##*/}")
    PAGES+=(dispatch)
fi

# The cases of every executable that are shown on the dashboard, from a dry run. It also fails early
# for targets the CPU can't run.
declare -A FILTER
for EXE in "${EXES[@]}"; do
    if ! "$BUILD/bin/$EXE" --benchmark_dry_run --benchmark_out="$RESULTS/raw/$EXE.dry-run.json" \
        --benchmark_out_format=json > "$RESULTS/logs/$EXE.dry-run.txt" 2>&1; then
        echo "$EXE failed, see logs/$EXE.dry-run.txt" >&2
        exit 1
    fi
    FILTER[$EXE]=$(python3 "$SRC/scripts/benchmark_dashboard_data.py" --filter "$RESULTS/raw/$EXE.dry-run.json")
done

# The processes of a case are far apart in time, so a short disturbance only affects one of them.
# The environment is empty and paths are the same every night, so the stack placement is the same.
FAILED=()
for RUN in $(seq 1 "$RUNS"); do
    for EXE in "${EXES[@]}"; do
        echo "Run $RUN of $RUNS: $EXE"
        if ! env -i PATH=/usr/bin:/bin setarch -R taskset --cpu-list "$BENCHMARK_CPU" "$BUILD/bin/$EXE" \
            --benchmark_repetitions="$REPETITIONS" \
            --benchmark_min_time="$MIN_TIME" \
            --benchmark_filter="${FILTER[$EXE]}" \
            --benchmark_report_aggregates_only=true \
            --benchmark_out="$RESULTS/raw/$EXE.run$RUN.json" \
            --benchmark_out_format=json > "$RESULTS/logs/$EXE.run$RUN.txt" 2>&1; then
            echo "$EXE failed, see logs/$EXE.run$RUN.txt"
            FAILED+=("$EXE.run$RUN")
        fi
    done
done
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "Failed runs: ${FAILED[*]}" >&2
    exit 1
fi

for EXE in "${EXES[@]}"; do
    python3 "$SRC/scripts/merge_benchmark_runs.py" --key real_time "$RESULTS/$EXE.json" "$RESULTS"/raw/"$EXE".run*.json
done
for PAGE in "${PAGES[@]}"; do
    python3 "$SRC/scripts/benchmark_dashboard_data.py" --target "$PAGE" --out-dir "$RESULTS/dashboard" \
        "$RESULTS"/*."$PAGE".json
done
# The dispatch charts are named after the targets of the dispatcher, so results of another target
# list are not compared with them.
python3 - "$RESULTS/dashboard" "${PAGES[@]}" > "$RESULTS/targets.json" <<'EOF'
import json, sys
pages = []
for page in sys.argv[2:]:
    name = page
    if page == "dispatch":
        with open(sys.argv[1] + "/dispatch.gated.json") as f:
            name += " (" + json.load(f)["context"]["ispc_targets"] + ")"
    pages.append({"target": page, "name": name})
print(json.dumps(pages))
EOF
