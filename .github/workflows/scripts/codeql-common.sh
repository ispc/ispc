#!/bin/bash
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Shared code for the CodeQL alert fixer scripts.

# Prints to the log and to the job summary.
note() { printf '%s\n' "$*"; printf -- '- %s\n' "$*" >> "$GITHUB_STEP_SUMMARY"; }

# Prints the open issues of github-actions[bot] with the alert numbers each one holds.
open_bot_issues() {
  gh api "repos/$GH_REPO/issues?state=open&creator=github-actions%5Bbot%5D&per_page=100" --paginate |
    jq -sc '
      def alert_numbers:
        ((.body // "") | split("\n")[0] // "") as $first |
        if ($first | startswith("<!-- codeql-unsure-alerts: ")) then
          ($first | (capture("^<!-- codeql-unsure-alerts: (?<ids>\\[.*\\]) -->$")? // error("invalid CodeQL issue marker")) | .ids | fromjson) |
          if type == "array" and all(.[]; type == "number" and . > 0 and . == floor)
          then unique else error("invalid alert IDs in issue") end
        else [(.title | capture("^\\[codeql-alert #(?<n>[0-9]+)\\]")? | .n | tonumber)] end;
      add // [] | map(select(.user.login == "github-actions[bot]" and (has("pull_request") | not)) |
        {number, state, alerts: alert_numbers})'
}

# Paths a CodeQL fix must not touch: CI, Claude config, and the build and test
# scripts and the test runner that CI and check-all run. Changes there need a
# human-authored PR.
PROTECTED_PATHS_RE='^(\.github/|\.claude/|scripts/|cmake/|utils/|(.*/)?CMakeLists\.txt$|(.*/)?lit(\.site)?\.cfg(\.py)?$|.*\.(py|sh|ps1|cmake)$|\.gitmodules$|\.gitattributes$|\.mcp\.json$|CLAUDE(\.local)?\.md$)'

# Prints the files that match PROTECTED_PATHS_RE, read as NUL-separated paths
# from stdin. NUL separation avoids Git's quoting of unusual file names.
protected_paths() {
  { grep -zE "$PROTECTED_PATHS_RE" || true; } | tr '\0' ' '
}

# Prints the protected files that patch file $1 touches. --numstat shows only
# the new name of a renamed file, so renames and copies count as protected.
protected_patch_paths() {
  if grep -qE '^(rename|copy) from ' "$1"; then echo "(renamed or copied files)"; return; fi
  git apply --numstat -z "$1" | sed -z 's/^[-0-9]*\t[-0-9]*\t//' | protected_paths
}
