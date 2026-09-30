#!/usr/bin/env bash
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# Applies the "duplicate" label to the issue that triggered the workflow. The
# label is fixed and the issue number is read from the event payload, so the
# only decision left to the model is whether to run this script at all - it
# cannot choose a different label or target a different issue.
#
set -euo pipefail

DUP_LABEL="duplicate"

if [[ $# -ne 0 ]]; then
  echo "Error: mark-duplicate.sh takes no arguments" >&2
  exit 1
fi

# Read from event payload so the issue number is bound to the triggering event.
ISSUE=$(jq -r '.issue.number // empty' "${GITHUB_EVENT_PATH:?GITHUB_EVENT_PATH not set}")
if ! [[ "$ISSUE" =~ ^[0-9]+$ ]]; then
  echo "Error: no issue number in event payload" >&2
  exit 1
fi

# Only apply the label if it actually exists in the repository.
if ! gh label list --limit 500 --json name --jq '.[].name' | grep -qxF "$DUP_LABEL"; then
  echo "Error: the '$DUP_LABEL' label does not exist in this repository" >&2
  exit 1
fi

# Additive: any existing labels on the issue are preserved.
gh issue edit "$ISSUE" --add-label "$DUP_LABEL"
echo "Applied '$DUP_LABEL' label to issue #$ISSUE"
