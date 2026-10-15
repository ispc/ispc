#!/bin/bash
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Selects the open CodeQL alerts on main that the CodeQL Alert Fixer investigates.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/codeql-common.sh"

if [ -n "$REQUESTED_ALERT" ] && [[ ! "$REQUESTED_ALERT" =~ ^[1-9][0-9]{0,8}$ ]]; then
  echo "::error::alert_number must be a positive integer."
  exit 1
fi

owner=${GH_REPO%%/*}
pulls=$(gh api "repos/$GH_REPO/pulls?state=all&head=$owner:codeql/fixes&per_page=100" --paginate |
  jq -sc --arg repo "$GH_REPO" 'add // [] | map(select(.head.repo.full_name == $repo))')
if jq -e 'any(.[]; .state == "open")' <<< "$pulls" > /dev/null; then
  note "An open PR from codeql/fixes exists; no alerts will be investigated."
  echo "matrix=[]" >> "$GITHUB_OUTPUT"
  exit 0
fi

refs=$(gh api "repos/$GH_REPO/git/matching-refs/heads/codeql/fixes")
branch_sha=$(jq -r '[.[] | select(.ref == "refs/heads/codeql/fixes") | .object.sha][0] // ""' <<< "$refs")
if [ -n "$branch_sha" ] && jq -e --arg sha "$branch_sha" 'any(.[]; .merged_at != null and .head.sha == $sha)' <<< "$pulls" > /dev/null; then
  note "The merged codeql/fixes branch still exists; delete it before another batch."
  echo "matrix=[]" >> "$GITHUB_OUTPUT"
  exit 0
fi

base_sha=$(gh api "repos/$GH_REPO/commits/main" --jq .sha)
alerts=$(gh api "repos/$GH_REPO/code-scanning/alerts?state=open&tool_name=CodeQL&ref=refs/heads/main&per_page=100" --paginate |
  jq -sc 'add // [] | sort_by(.number)')

# Commit trailers record pending fixes without another state store.
covered='[]'
if [ -n "$branch_sha" ]; then
  covered=$(gh api "repos/$GH_REPO/compare/$base_sha...$branch_sha?per_page=100" --paginate |
    jq -sc '[.[].commits[]?.commit.message | split("\n\n")[-1] | split("\n")[] |
      capture("^CodeQL-Alert: (?<n>[1-9][0-9]{0,8})$")? | .n | tonumber] | unique')
fi

# Open issues hold alerts for a maintainer's decision.
issues=$(open_bot_issues)

selected='[]'
while IFS= read -r alert; do
  n=$(jq .number <<< "$alert")
  if [ -n "$REQUESTED_ALERT" ] && [ "$REQUESTED_ALERT" != "$n" ]; then continue; fi
  if jq -e --argjson n "$n" 'index($n) != null' <<< "$covered" > /dev/null; then
    note "Alert #$n: already has a pending fix on codeql/fixes; skipped."
    continue
  fi

  # Manual selection overrides an issue, but never an open PR or pending fix.
  if [ -z "$REQUESTED_ALERT" ]; then
    issue=$(jq -c --argjson n "$n" 'map(select(.alerts | index($n) != null)) | max_by(.number)' <<< "$issues")
    if [ "$issue" != null ]; then
      note "Alert #$n: waits in open issue #$(jq .number <<< "$issue"); skipped. Dispatch with alert_number=$n to retry."
      continue
    fi
  fi

  selected=$(jq -c --argjson a "$alert" '. + [{number: $a.number, analyzed_sha: $a.most_recent_instance.commit_sha,
    rule: $a.rule.id, path: $a.most_recent_instance.location.path, line: $a.most_recent_instance.location.start_line}]' <<< "$selected")
done < <(jq -c '.[]' <<< "$alerts")

if [ -n "$REQUESTED_ALERT" ] && ! jq -e --argjson n "$REQUESTED_ALERT" 'any(.[]; .number == $n)' <<< "$alerts" > /dev/null; then
  note "Alert #$REQUESTED_ALERT is not an open CodeQL alert on main."
fi
if [ "$(jq length <<< "$selected")" -gt 256 ]; then
  echo "::error::Selected batch exceeds GitHub's 256-job matrix limit; no partial batch."
  exit 1
fi
note "Selected: $(jq -r 'map("#\(.number)") | join(", ") | if . == "" then "none" else . end' <<< "$selected")."
echo "matrix=$selected" >> "$GITHUB_OUTPUT"
echo "base_sha=$base_sha" >> "$GITHUB_OUTPUT"
echo "branch_sha=$branch_sha" >> "$GITHUB_OUTPUT"
echo "covered=$covered" >> "$GITHUB_OUTPUT"
