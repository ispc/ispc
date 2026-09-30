#!/usr/bin/env bash
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# Edits labels on a GitHub issue. Labels are the only thing this script can
# change, and the issue number is bound to the triggering event rather than
# taken from an argument, so a prompt injection cannot redirect it to another
# issue or mutate other issue fields.
#
# Usage: .github/scripts/edit-issue-labels.sh --add-label bug --add-label performance --remove-label untriaged
#
set -euo pipefail

# Read from event payload so the issue number is bound to the triggering event
ISSUE=$(jq -r '.issue.number // empty' "${GITHUB_EVENT_PATH:?GITHUB_EVENT_PATH not set}")
if ! [[ "$ISSUE" =~ ^[0-9]+$ ]]; then
  echo "Error: no issue number in event payload" >&2
  exit 1
fi

ADD_LABELS=()
REMOVE_LABELS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --add-label)
      ADD_LABELS+=("$2")
      shift 2
      ;;
    --remove-label)
      REMOVE_LABELS+=("$2")
      shift 2
      ;;
    *)
      echo "Error: unknown argument (only --add-label and --remove-label are accepted)" >&2
      exit 1
      ;;
  esac
done

if [[ ${#ADD_LABELS[@]} -eq 0 && ${#REMOVE_LABELS[@]} -eq 0 ]]; then
  exit 1
fi

# Fetch valid labels from the repo
VALID_LABELS=$(gh label list --limit 500 --json name --jq '.[].name')

# Process/maintainer labels this script must never touch, regardless of what the
# model asks for. Enforced here so the guarantee holds even if the prompt is
# subverted. Duplicate detection and "Good First Issue" are maintainer/other-
# workflow decisions; dependencies/github_actions are automation-managed.
PROTECTED_LABELS=("Good First Issue" "duplicate" "dependencies" "github_actions")

is_protected() {
  local candidate="$1" protected
  for protected in "${PROTECTED_LABELS[@]}"; do
    [[ "$candidate" == "$protected" ]] && return 0
  done
  return 1
}

# Keep only labels that exist in the repo and are not protected.
FILTERED_ADD=()
for label in "${ADD_LABELS[@]}"; do
  if grep -qxF "$label" <<<"$VALID_LABELS" && ! is_protected "$label"; then
    FILTERED_ADD+=("$label")
  fi
done

FILTERED_REMOVE=()
for label in "${REMOVE_LABELS[@]}"; do
  if grep -qxF "$label" <<<"$VALID_LABELS" && ! is_protected "$label"; then
    FILTERED_REMOVE+=("$label")
  fi
done

if [[ ${#FILTERED_ADD[@]} -eq 0 && ${#FILTERED_REMOVE[@]} -eq 0 ]]; then
  exit 0
fi

# Build gh command arguments. `gh issue edit --add-label` is additive, so
# existing labels on the issue are preserved.
GH_ARGS=("issue" "edit" "$ISSUE")

for label in "${FILTERED_ADD[@]}"; do
  GH_ARGS+=("--add-label" "$label")
done

for label in "${FILTERED_REMOVE[@]}"; do
  GH_ARGS+=("--remove-label" "$label")
done

gh "${GH_ARGS[@]}"

if [[ ${#FILTERED_ADD[@]} -gt 0 ]]; then
  echo "Added: ${FILTERED_ADD[*]}"
fi
if [[ ${#FILTERED_REMOVE[@]} -gt 0 ]]; then
  echo "Removed: ${FILTERED_REMOVE[*]}"
fi
