#!/usr/bin/env bash
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# Sets the type of the issue that triggered the workflow, but only if it has no
# type yet, so a type set by a maintainer is never overwritten. The issue number
# is read from the event payload and only Bug and Enhancement are accepted.
#
# Usage: .github/workflows/scripts/triage-set-issue-type.sh Bug|Enhancement
#
set -euo pipefail

export GH_REPO="${GH_REPO:-${GITHUB_REPOSITORY:?GH_REPO or GITHUB_REPOSITORY must be set}}"

if [[ $# -ne 1 ]] || ! [[ "$1" == "Bug" || "$1" == "Enhancement" ]]; then
  echo "Error: give exactly one type, Bug or Enhancement" >&2
  exit 1
fi
TYPE="$1"

# Read from event payload so the issue number is bound to the triggering event.
ISSUE=$(jq -r '.issue.number // .inputs.issue_number // empty' "${GITHUB_EVENT_PATH:?GITHUB_EVENT_PATH not set}")
if ! [[ "$ISSUE" =~ ^[0-9]+$ ]]; then
  echo "Error: no issue number in event payload" >&2
  exit 1
fi

CURRENT=$(gh api "repos/$GH_REPO/issues/$ISSUE" --jq '.type.name // empty')
if [[ -n "$CURRENT" ]]; then
  echo "Issue #$ISSUE already has type $CURRENT; not changed"
  exit 0
fi

gh api -X PATCH "repos/$GH_REPO/issues/$ISSUE" -f type="$TYPE" --silent
echo "Set the type of #$ISSUE to $TYPE"
