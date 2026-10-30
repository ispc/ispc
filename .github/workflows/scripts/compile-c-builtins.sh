#!/bin/bash -e
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Compiles tests/c-builtins/c_builtins.ispc for other target OSes, so that
# tests/c-builtins/run.sh can run it there (#3887).
# Usage: compile-c-builtins.sh <ispc> <output directory> <os>/<arch>...
# The objects are named <os>-<arch>.o, e.g. freebsd-x86-64.o. Their targets are
# 4-wide, see run.sh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$SCRIPT_DIR/../../../tests/c-builtins/c_builtins.ispc"
ISPC=$1
OUT=$2
shift 2

mkdir -p "$OUT"
for os_arch in "$@"; do
  os=${os_arch%/*}
  arch=${os_arch#*/}
  case "$arch" in
    x86-64|x86) target=sse2-i32x4 ;;
    aarch64|arm) target=neon-i32x4 ;;
    *) echo "Unknown architecture: $arch"; exit 1 ;;
  esac
  echo "Compiling for $os/$arch ($target)"
  "$ISPC" "$SOURCE" --target-os="$os" --arch="$arch" --target="$target" --pic -o "$OUT/$os-$arch.o"
done
