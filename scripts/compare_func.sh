#!/bin/bash
# Copyright (c) 2025, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Script to compare function assembly output between two ISPC compilers
# 
# Usage: ./compare_func.sh [--side-by-side|-s] <function_name> <target> <filename> [extra_flags] [mcpu]
#
# Examples:
#   Basic comparison with unmangled function name:
#     ./compare_func.sh min_v avx2-i64x4 min_half.ispc
#
#   Using mangled function name (when function has overloads):
#     ./compare_func.sh min_v___vyhvyh avx2-i64x4 min_half.ispc
#
#   With optimization flags:
#     ./compare_func.sh min_v avx2-i32x8 t.ispc "-O2 --cpu=skx"
#
#   Side-by-side comparison with performance analysis:
#     ./compare_func.sh -s min_v avx2-i32x8 t.ispc "-O2 --cpu=skx" skylake
#
#   Comparing different targets:
#     ./compare_func.sh sin_func sse4-i32x4 math.ispc
#     ./compare_func.sh sin_func avx2-i32x8 math.ispc  
#     ./compare_func.sh sin_func avx512skx-i32x16 math.ispc
#
#   Environment variable usage (override default compiler paths):
#     NEW_ISPC=/path/to/new/ispc OLD_ISPC=/path/to/old/ispc ./compare_func.sh func_name target file.ispc
#
#   Multiple optimization levels comparison:
#     ./compare_func.sh compute_func avx2-i32x8 compute.ispc "-O0"
#     ./compare_func.sh compute_func avx2-i32x8 compute.ispc "-O1" 
#     ./compare_func.sh compute_func avx2-i32x8 compute.ispc "-O2"

# Parse side-by-side flag
SIDE_BY_SIDE=false
if [[ "$1" == "--side-by-side" || "$1" == "-s" ]]; then
    SIDE_BY_SIDE=true
    shift
fi

if [ $# -lt 3 ] || [ $# -gt 5 ]; then
    echo "Usage: $0 [--side-by-side|-s] <function_name> <target> <filename> [extra_flags] [mcpu]"
    echo "Example: $0 min_v avx2-i64x4 min_half.ispc (unmangled name)"
    echo "         $0 min_v___vyhvyh avx2-i64x4 min_half.ispc (mangled name)"
    echo "         $0 min_v avx2-i32x8 t.ispc \"-O2 --cpu=skx\" (with extra flags)"
    echo "         $0 -s min_v avx2-i32x8 t.ispc \"-O2 --cpu=skx\" skylake (with mcpu for llvm-mca)"
    echo "Options:"
    echo "  --side-by-side, -s    Show assembly and MCA output side by side"
    exit 1
fi

FUNCTION_NAME="$1"
TARGET="$2"
FILENAME="$3"
EXTRA_FLAGS="$4"
MCPU="$5"

# Check if file exists
if [ ! -f "$FILENAME" ]; then
    echo "Error: File '$FILENAME' not found"
    exit 1
fi

# Paths to compilers. Can be overridden by environment variables.
NEW_ISPC=${NEW_ISPC:-./bin/ispc}
OLD_ISPC=${OLD_ISPC:-../../ispc-v1.27.0-linux/bin/ispc}

# Check if compilers exist
if [ ! -f "$NEW_ISPC" ]; then
    echo "Error: New ISPC compiler not found at $NEW_ISPC"
    exit 1
fi

if [ ! -f "$OLD_ISPC" ]; then
    echo "Error: Old ISPC compiler not found at $OLD_ISPC"
    exit 1
fi

# Common flags
FLAGS="--target=$TARGET --emit-asm --x86-asm-syntax=intel -o - $EXTRA_FLAGS"

# Function to find mangled function name from unmangled name
find_mangled_name() {
    local input="$1"
    local unmangled_name="$2"

    # Look for function starting with unmangled name followed by ___
    echo "$input" | grep -E "^${unmangled_name}___[^:]*:" | head -1 | sed 's/:.*$//'
}

# Function to extract function assembly from output
extract_function() {
    local input="$1"
    local func_name="$2"

    # If function name doesn't contain ___, try to find mangled version
    if [[ "$func_name" != *"___"* ]]; then
        local mangled_name=$(find_mangled_name "$input" "$func_name")
        if [ -n "$mangled_name" ]; then
            func_name="$mangled_name"
            echo "# Found mangled name: $func_name" >&2
        fi
    fi

    # Extract only the instructions between function label and ret instruction
    echo "$input" | awk -v fname="$func_name" '
        $0 ~ "^" fname ":" {
            found=1
            next
        }
        found && /^[[:space:]]*ret[[:space:]]*$/ {
            print
            exit
        }
        found && !/^[[:space:]]*#/ && !/^[[:space:]]*$/ && !/^\./ {
            print
        }
    '
}

echo "=== Comparing function '$FUNCTION_NAME' for target '$TARGET' ==="
echo

# Get assembly from new compiler
NEW_ASM=$(eval "$NEW_ISPC $FLAGS $FILENAME" 2>&1)
if [ $? -ne 0 ]; then
    echo "Error running new ISPC compiler:"
    echo "$NEW_ASM"
    exit 1
fi

NEW_FUNC=$(extract_function "$NEW_ASM" "$FUNCTION_NAME" 2>/dev/null)
if [ -z "$NEW_FUNC" ]; then
    echo "Function '$FUNCTION_NAME' not found in new compiler output"
    echo "Available functions:"
    echo "$NEW_ASM" | grep -E '^\w+:.*# @' | sed 's/:.*# @.*//' | sort
    exit 1
fi

# Get assembly from old compiler
OLD_ASM=$(eval "$OLD_ISPC $FLAGS $FILENAME" 2>&1)
if [ $? -ne 0 ]; then
    echo "Error running old ISPC compiler:"
    echo "$OLD_ASM"
    exit 1
fi

OLD_FUNC=$(extract_function "$OLD_ASM" "$FUNCTION_NAME" 2>/dev/null)
if [ -z "$OLD_FUNC" ]; then
    echo "Function '$FUNCTION_NAME' not found in old compiler output"
    echo "Available functions:"
    echo "$OLD_ASM" | grep -E '^\w+:.*# @' | sed 's/:.*# @.*//' | sort
    exit 1
fi

# Show assembly output
if [ "$SIDE_BY_SIDE" = true ]; then
    echo "=== ASSEMBLY COMPARISON ==="
    NEW_VERSION="$($NEW_ISPC --version 2>&1 | head -1)"
    OLD_VERSION="$($OLD_ISPC --version 2>&1 | head -1)"
    printf "%-40s | %s\n" "OLD ISPC ($OLD_VERSION)" "NEW ISPC ($NEW_VERSION)"
    printf "%-40s-+-%s\n" "$(printf '%*s' 40 '' | tr ' ' '-')" "$(printf '%*s' 40 '' | tr ' ' '-')"

    diff --side-by-side --width=120 --left-column <(echo "$OLD_FUNC") <(echo "$NEW_FUNC") || true
else
    echo "=== NEW ISPC ($($NEW_ISPC --version 2>&1 | head -1)) ==="
    echo "$NEW_FUNC"
    echo

    echo "=== OLD ISPC ($($OLD_ISPC --version 2>&1 | head -1)) ==="
    echo "$OLD_FUNC"
    echo
fi

# Show detailed diff (only if not side-by-side)
if [ "$SIDE_BY_SIDE" = false ]; then
    echo "=== DETAILED DIFF (- OLD, + NEW) ==="
    if diff -u <(echo "$OLD_FUNC") <(echo "$NEW_FUNC") >/dev/null; then
        echo "No differences found"
    else
        diff -u <(echo "$OLD_FUNC") <(echo "$NEW_FUNC") | tail -n +3
    fi
fi

# Run llvm-mca analysis if MCPU is provided
EXTRA_MCA_OPTS=""
# EXTRA_MCA_OPTS="-timeline -dispatch-stats -scheduler-stats -register-file-stats"
if [ -n "$MCPU" ]; then
    echo
    echo "=== LLVM-MCA ANALYSIS ==="

    # Check if llvm-mca is available
    if ! command -v llvm-mca &> /dev/null; then
        echo "llvm-mca not found, skipping performance analysis"
    else
        if [ "$SIDE_BY_SIDE" = true ]; then
            echo "=== MCA ANALYSIS COMPARISON ==="
            printf "%-60s | %s\n" "OLD ISPC MCA Analysis" "NEW ISPC MCA Analysis"
            printf "%-60s-+-%s\n" "$(printf '%*s' 60 '' | tr ' ' '-')" "$(printf '%*s' 60 '' | tr ' ' '-')"

            # Add appropriate assembly syntax based on target
            if [[ "$TARGET" == *"neon"* ]]; then
                OLD_MCA=$(echo "$OLD_FUNC" | llvm-mca --mtriple=aarch64-linux-gnu --mcpu="$MCPU" $EXTRA_MCA_OPTS 2>/dev/null || echo "llvm-mca analysis failed for old version")
                NEW_MCA=$(echo "$NEW_FUNC" | llvm-mca --mtriple=aarch64-linux-gnu --mcpu="$MCPU" $EXTRA_MCA_OPTS 2>/dev/null || echo "llvm-mca analysis failed for new version")
            else
                OLD_MCA=$((echo ".intel_syntax noprefix"; echo "$OLD_FUNC") | llvm-mca --mcpu="$MCPU" $EXTRA_MCA_OPTS 2>/dev/null || echo "llvm-mca analysis failed for old version")
                NEW_MCA=$((echo ".intel_syntax noprefix"; echo "$NEW_FUNC") | llvm-mca --mcpu="$MCPU" $EXTRA_MCA_OPTS 2>/dev/null || echo "llvm-mca analysis failed for new version")
            fi

            diff --side-by-side --width=140 --left-column <(echo "$OLD_MCA") <(echo "$NEW_MCA") || true
        else
            echo
            echo "--- NEW ISPC Performance Analysis ---"
            if [[ "$TARGET" == *"neon"* ]]; then
                echo "$NEW_FUNC" | llvm-mca --mtriple=aarch64-linux-gnu --mcpu="$MCPU" $EXTRA_MCA_OPTS 2>/dev/null || echo "llvm-mca analysis failed for new version"
            else
                (echo ".intel_syntax noprefix"; echo "$NEW_FUNC") | llvm-mca --mcpu="$MCPU" $EXTRA_MCA_OPTS 2>/dev/null || echo "llvm-mca analysis failed for new version"
            fi

            echo
            echo "--- OLD ISPC Performance Analysis ---"
            if [[ "$TARGET" == *"neon"* ]]; then
                echo "$OLD_FUNC" | llvm-mca --mtriple=aarch64-linux-gnu --mcpu="$MCPU" $EXTRA_MCA_OPTS 2>/dev/null || echo "llvm-mca analysis failed for old version"
            else
                (echo ".intel_syntax noprefix"; echo "$OLD_FUNC") | llvm-mca --mcpu="$MCPU" $EXTRA_MCA_OPTS 2>/dev/null || echo "llvm-mca analysis failed for old version"
            fi
        fi
    fi
fi

# Exit with a non-zero return code if the assembly is different.
diff -q <(echo "$OLD_FUNC") <(echo "$NEW_FUNC") >/dev/null
exit $?
