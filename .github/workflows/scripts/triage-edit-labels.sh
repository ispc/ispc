#!/usr/bin/env bash
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# Adds labels to the issue that triggered the workflow. Adding labels is the
# only thing this script can do, and the issue number is bound to the
# triggering event rather than taken from an argument, so a prompt injection
# cannot redirect it to another issue or change other issue fields.
#
# Usage: .github/workflows/scripts/triage-edit-labels.sh --add-label Codegen --add-label Performance
#
set -euo pipefail

export GH_REPO="${GH_REPO:-${GITHUB_REPOSITORY:?GH_REPO or GITHUB_REPOSITORY must be set}}"

# Read from event payload so the issue number is bound to the triggering event
ISSUE=$(jq -r '.issue.number // empty' "${GITHUB_EVENT_PATH:?GITHUB_EVENT_PATH not set}")
if ! [[ "$ISSUE" =~ ^[0-9]+$ ]]; then
  echo "Error: no issue number in event payload" >&2
  exit 1
fi

ADD_LABELS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --add-label)
      [[ $# -ge 2 ]] || { echo "Error: --add-label needs a value" >&2; exit 1; }
      ADD_LABELS+=("$2")
      shift 2
      ;;
    *)
      echo "Error: unknown argument (only --add-label is accepted)" >&2
      exit 1
      ;;
  esac
done

if [[ ${#ADD_LABELS[@]} -eq 0 ]]; then
  echo "Error: no labels given" >&2
  exit 1
fi

# Fetch valid labels from the repo
VALID_LABELS=$(gh label list --limit 500 --json name --jq '.[].name')

# Process/maintainer labels this script must never touch, regardless of what the
# model asks for. Enforced here so the guarantee holds even if the prompt is
# subverted. "duplicate" is set by triage-mark-duplicate.sh, "Good First Issue"
# is a maintainer decision, and dependencies, github_actions and llvm-trunk are
# automation-managed.
PROTECTED_LABELS=("Good First Issue" "duplicate" "dependencies" "github_actions" "llvm-trunk")

is_protected() {
  local candidate="$1" protected
  for protected in "${PROTECTED_LABELS[@]}"; do
    [[ "$candidate" == "$protected" ]] && return 0
  done
  return 1
}

# Keep only labels that exist in the repo and are not protected.
GH_ARGS=("issue" "edit" "$ISSUE")
ADDED=()
for label in "${ADD_LABELS[@]}"; do
  if grep -qxF "$label" <<<"$VALID_LABELS" && ! is_protected "$label"; then
    GH_ARGS+=("--add-label" "$label")
    ADDED+=("$label")
  else
    echo "Skipped: $label"
  fi
done

if [[ ${#ADDED[@]} -eq 0 ]]; then
  exit 0
fi

# `gh issue edit --add-label` is additive, so existing labels are preserved.
gh "${GH_ARGS[@]}"
echo "Added: ${ADDED[*]}"
