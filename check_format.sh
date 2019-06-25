#!/bin/bash
# ##################################################
#  Copyright (c) 2019, Intel Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
EXIT_CODE=0
echo "\
############################################################################
Checking formatting of modified files. It is expected that the files were
formatted with clang-format 8.0.0. It is also expected that clang-format
version 8.0.0 is used for the check. Otherwise the result can ne unexpected.
############################################################################"

CLANG_FORMAT="clang-format"
REQUIRED_VERSION="8.0.0"
VERSION_STRING="$CLANG_FORMAT version $REQUIRED_VERSION .*"
CURRENT_VERSION="$(clang-format --version)"
if ! [[ $CURRENT_VERSION =~ $VERSION_STRING ]] ; then
    echo WARNING: $CLANG_FORMAT version $REQUIRED_VERSION is required but $CURRENT_VERSION is used.
    echo The results can be unexpected.
fi

LAST_REVISION=`git rev-parse HEAD`
# Get the list of changed files, ignore deleted
CHANGED_FILES=`git diff-tree --no-commit-id --name-only -r $LAST_REVISION --diff-filter=RCMA | grep -E "\.(cpp|h|c)$"`
for FILE in $CHANGED_FILES; do
    $CLANG_FORMAT $FILE | cmp  $FILE >/dev/null
    if [ $? -ne 0 ]; then
        echo "[!] INCORRECT FORMATTING! $FILE" >&2
            EXIT_CODE=1
        fi
    done
if [ $EXIT_CODE -eq 0 ]; then 
    echo "No formatting issues found"
fi
exit $EXIT_CODE
