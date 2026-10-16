#!/bin/bash
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Checks the codeql/fixes branch that the finalizer pushed in claude.yml. It
# runs outside that job, from the selected main commit, so nothing Claude ran
# can change this check. No code from the branch runs here.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/codeql-common.sh"

fail() { note "codeql/fixes must not be merged: $*"; echo "::error::codeql/fixes: $*"; exit 1; }

# ls-remote matches ref name suffixes, so keep only the exact ref.
refs=$(git ls-remote origin refs/heads/codeql/fixes)
new=$(jq -Rr 'split("\t") | select(.[1] == "refs/heads/codeql/fixes") | .[0]' <<< "$refs")
if [ -z "$new" ] || [ "$new" = "$BRANCH_SHA" ]; then
  note "codeql/fixes was not pushed in this run."
  exit 0
fi
git fetch -q --no-tags origin refs/heads/codeql/fixes

# The new head must keep the selected main snapshot and every pending fix.
git merge-base --is-ancestor "$BASE_SHA" "$new" || fail "$new does not contain main $BASE_SHA"
if [ -n "$BRANCH_SHA" ]; then
  git merge-base --is-ancestor "$BRANCH_SHA" "$new" || fail "$new does not contain codeql/fixes $BRANCH_SHA"
fi

# Without rename detection, a renamed file lists both its old and new path.
protected=$(git diff --no-renames --name-only -z "$BASE_SHA" "$new" | protected_paths)
[ -z "$protected" ] || fail "the fixes touch protected paths: $protected"

note "codeql/fixes at $new passed the checks."
