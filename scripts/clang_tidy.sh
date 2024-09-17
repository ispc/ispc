#!/bin/bash
#  Copyright (c) 2024, Intel Corporation
#  SPDX-License-Identifier: BSD-3-Clause

set -e

BUILD_FOLDER=${1:-"build"}

if [ ! -d "$BUILD_FOLDER" ]; then
  echo "Build folder '$BUILD_FOLDER' does not exist."
  exit 1
fi

if [ ! -f "$BUILD_FOLDER/compile_commands.json" ]; then
  echo "compile_commands.json not found in '$BUILD_FOLDER'."
  exit 1
fi

CLANG_TIDY=${2:-"clang-tidy-18"}
if ! command -v $CLANG_TIDY &> /dev/null; then
  echo "$CLANG_TIDY not found. Please install clang-tidy-18."
  exit 1
else
  CLANG_TIDY_PATH=$(which $CLANG_TIDY)
  echo "Using clang-tidy: $CLANG_TIDY_PATH"
  "$CLANG_TIDY_PATH" --version
fi

FILES=$(ls                                  \
    src/*.{cpp,h}                           \
    src/opt/*.{cpp,h}                       \
    builtins/*{cpp,hpp,c}                   \
    common/*.h                              \
)

# Run clang-tidy with the compilation database from the given build folder in parallel
run-clang-tidy -clang-tidy-binary "$CLANG_TIDY_PATH" -p "$BUILD_FOLDER" $FILES
