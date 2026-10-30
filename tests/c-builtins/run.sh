#!/bin/sh
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Links an object compiled from c_builtins.ispc with main.c for the target OS,
# runs it and checks the output of print() and num_cores() (#3887). It runs on
# the target OS, so it is POSIX sh.
# Usage: run.sh <object> <C compiler> [<C compiler options>...]
# RUN_WRAPPER, if set, runs the test, e.g. "qemu-aarch64 -L /usr/aarch64-linux-gnu".

set -e
obj=$1
shift
dir=$(cd "$(dirname "$0")" && pwd)
exe=$(basename "$obj" .o).run

"$@" "$dir/main.c" "$obj" -o "$exe"
status=0
$RUN_WRAPPER "./$exe" > "$exe.out" || status=$?
cat "$exe.out"
if [ "$status" -ne 0 ]; then
    echo "FAIL: $obj: num_cores() is not the number of online processors"
    exit 1
fi
if ! grep -Fqx 'uniform: 42 1.500000 2.250000' "$exe.out" || ! grep -Fqx 'varying: [0,1,2,3]' "$exe.out"; then
    echo "FAIL: $obj: unexpected print() output"
    exit 1
fi
echo "PASS: $obj"
