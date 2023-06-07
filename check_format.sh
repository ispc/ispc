#!/bin/bash
# ##################################################
#  Copyright (c) 2019-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause
EXIT_CODE=0
echo "\
############################################################################
Checking formatting of modified files. It is expected that the files were
formatted with clang-format 12.0.0. It is also expected that clang-format
version 12.0.0 is used for the check. Otherwise the result can ne unexpected.
############################################################################"

CLANG_FORMAT="clang-format"
[[ ! -z $1 ]] && CLANG_FORMAT=$1
which "$CLANG_FORMAT" || { echo "No $CLANG_FORMAT found in PATH" && exit 1; }
REQUIRED_VERSION="12.0.0"
VERSION_STRING="clang-format version $REQUIRED_VERSION.*"
CURRENT_VERSION="$($CLANG_FORMAT --version)"
if ! [[ $CURRENT_VERSION =~ $VERSION_STRING ]] ; then
    echo WARNING: clang-format version $REQUIRED_VERSION is required but $CURRENT_VERSION is used.
    echo The results can be unexpected.
fi

# Check all source files.
# For benchmarks folder do not check 03_complex, as these tests come from real projects with their formatting.
FILES=$(ls                                  \
    src/*.{cpp,h}                           \
    src/opt/*.{cpp,h}                       \
    *.cpp                                   \
    builtins/*{cpp,hpp,c}                   \
    benchmarks/{01,02}*/*{cpp,ispc}         \
    common/*.h                              \
    stdlib.ispc                             \
    ispcrt/*.{h,hpp,cpp}                    \
    ispcrt/detail/*.h                       \
    ispcrt/detail/cpu/*.{h,cpp}             \
    ispcrt/detail/gpu/*.{h,cpp}             \
    ispcrt/tests/level_zero_mock/*.{h,cpp}  \
    ispcrt/tests/mock_tests/*.cpp           \
)
for FILE in $FILES; do
    diff -uN --label original/$FILE --label formatted/$FILE <(cat $FILE) <($CLANG_FORMAT $FILE)
    if [ $? -ne 0 ]; then
        echo "[!] INCORRECT FORMATTING! $FILE" >&2
            EXIT_CODE=1
        fi
    done
if [ $EXIT_CODE -eq 0 ]; then 
    echo "No formatting issues found"
fi
exit $EXIT_CODE
