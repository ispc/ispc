#!/bin/bash
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Pushes the combined CodeQL fixes from fixes/fixes.bundle to codeql/fixes.
# The bundle was built by Claude in a job without write access, so check it
# here before this job's write token touches it. No code from it runs here.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/codeql-common.sh"

fail() { note "Not pushed: $*"; exit 1; }

if [ -n "$BRANCH_SHA" ]; then lease="$BRANCH_SHA"; else lease=""; fi

git bundle verify -q fixes/fixes.bundle > /dev/null || fail "invalid bundle"
git fetch -q fixes/fixes.bundle +refs/heads/codeql/fixes:refs/codeql/combined || fail "no codeql/fixes in the bundle"
new=$(git rev-parse refs/codeql/combined)

# The new head must keep the selected main snapshot and every pending fix.
git merge-base --is-ancestor "$BASE_SHA" "$new" || fail "$new does not contain main $BASE_SHA"
if [ -n "$BRANCH_SHA" ]; then
  git merge-base --is-ancestor "$BRANCH_SHA" "$new" || fail "$new does not contain codeql/fixes $BRANCH_SHA"
  if [ "$new" = "$BRANCH_SHA" ]; then note "No new fixes."; exit 0; fi
fi

# Without rename detection, a renamed file lists both its old and new path.
protected=$(git diff --no-renames --name-only -z "$BASE_SHA" "$new" | protected_paths)
[ -z "$protected" ] || fail "the fixes touch protected paths: $protected"

# Don't change a branch that a PR is open for or that was already merged.
pulls=$(gh api "repos/$GH_REPO/pulls?state=all&head=${GH_REPO%%/*}:codeql/fixes&per_page=100" --paginate |
  jq -s --arg repo "$GH_REPO" 'add // [] | map(select(.head.repo.full_name? == $repo))')
if jq -e 'any(.[]; .state == "open")' <<< "$pulls" > /dev/null; then
  fail "a PR from codeql/fixes is open"
fi
if [ -n "$BRANCH_SHA" ] && jq -e --arg sha "$BRANCH_SHA" 'any(.[]; .merged_at != null and .head.sha == $sha)' <<< "$pulls" > /dev/null; then
  fail "codeql/fixes $BRANCH_SHA was already merged; delete the branch"
fi

# The lease protects a branch that changed since the alerts were selected; an
# empty lease requires the branch to be absent.
git push --force-with-lease="refs/heads/codeql/fixes:$lease" origin "$new:refs/heads/codeql/fixes" ||
  fail "the push was rejected"

echo "pushed=true" >> "$GITHUB_OUTPUT"
body=$(jq -sRr @uri fixes/pr-body.md)
note "Pushed codeql/fixes at $new. If no draft PR follows, open it: $GITHUB_SERVER_URL/$GH_REPO/compare/main...codeql/fixes?expand=1&body=$body"
