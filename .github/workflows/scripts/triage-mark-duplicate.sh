#!/usr/bin/env bash
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# Marks the issue that triggered the workflow as a possible duplicate of an
# open issue: applies the "duplicate" label and posts one fixed comment. The
# issue number is read from the event payload and the comment text is fixed,
# so the model only chooses whether to run this and which open issue it
# duplicates.
#
# Usage: .github/workflows/scripts/triage-mark-duplicate.sh <ORIGINAL_ISSUE_NUMBER>
#
set -euo pipefail

export GH_REPO="${GH_REPO:-${GITHUB_REPOSITORY:?GH_REPO or GITHUB_REPOSITORY must be set}}"

DUP_LABEL="duplicate"

if [[ $# -ne 1 ]] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
  echo "Error: give exactly one numeric issue number, the open issue this one duplicates" >&2
  exit 1
fi
ORIGINAL="$1"

# Read from event payload so the issue number is bound to the triggering event.
ISSUE=$(jq -r '.issue.number // empty' "${GITHUB_EVENT_PATH:?GITHUB_EVENT_PATH not set}")
if ! [[ "$ISSUE" =~ ^[0-9]+$ ]]; then
  echo "Error: no issue number in event payload" >&2
  exit 1
fi

if [[ "$ORIGINAL" == "$ISSUE" ]]; then
  echo "Error: an issue can't duplicate itself" >&2
  exit 1
fi

# The original must be an open issue (not a PR) in this repository.
if [[ "$(gh issue view "$ORIGINAL" --json state --jq .state)" != "OPEN" ]]; then
  echo "Error: #$ORIGINAL is not an open issue" >&2
  exit 1
fi

# Additive: any existing labels on the issue are preserved.
gh issue edit "$ISSUE" --add-label "$DUP_LABEL"
gh issue comment "$ISSUE" --body "This issue may be a duplicate of #$ORIGINAL. A maintainer will confirm; until then, please follow #$ORIGINAL for updates."
echo "Marked #$ISSUE as a possible duplicate of #$ORIGINAL"
