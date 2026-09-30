#!/usr/bin/env bash
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

# Wrapper around gh CLI that only allows the specific read-only subcommands and
# flags the issue-triage command needs. All commands are scoped to the current
# repository via GH_REPO or GITHUB_REPOSITORY so an injected query cannot reach
# across repositories.
#
# Usage:
#   .github/scripts/gh.sh issue view 123
#   .github/scripts/gh.sh issue view 123 --comments
#   .github/scripts/gh.sh search issues "search query" --limit 10
#   .github/scripts/gh.sh label list --limit 100

export GH_HOST=github.com

REPO="${GH_REPO:-${GITHUB_REPOSITORY:-}}"
if [[ -z "$REPO" || "$REPO" == */*/* || "$REPO" != */* ]]; then
  echo "Error: GH_REPO or GITHUB_REPOSITORY must be set to owner/repo format (e.g., GITHUB_REPOSITORY=ispc/ispc)" >&2
  exit 1
fi
export GH_REPO="$REPO"

SUB1="${1:-}"
SUB2="${2:-}"
CMD="$SUB1 $SUB2"
case "$CMD" in
  "issue view"|"search issues"|"label list")
    ;;
  *)
    echo "Error: only 'issue view', 'search issues', 'label list' are allowed (e.g., .github/scripts/gh.sh issue view 123)" >&2
    exit 1
    ;;
esac

shift 2

# Separate the allowed flags (--comments, --limit) from positional arguments.
POSITIONAL=()
FLAGS=()
skip_next=false
for arg in "$@"; do
  if [[ "$skip_next" == true ]]; then
    FLAGS+=("$arg")
    skip_next=false
  elif [[ "$arg" == -* ]]; then
    case "${arg%%=*}" in
      --comments)
        FLAGS+=("$arg")
        ;;
      --limit)
        FLAGS+=("$arg")
        # --limit takes a value; skip the next arg unless --limit=N was used.
        [[ "$arg" != *=* ]] && skip_next=true
        ;;
      *)
        echo "Error: only --comments and --limit flags are allowed (e.g., .github/scripts/gh.sh search issues \"bug report\" --limit 10)" >&2
        exit 1
        ;;
    esac
  else
    POSITIONAL+=("$arg")
  fi
done

if [[ "$CMD" == "search issues" ]]; then
  if [[ ${#POSITIONAL[@]} -ne 1 ]]; then
    echo "Error: search issues requires exactly one quoted query (e.g., .github/scripts/gh.sh search issues \"bug report\" --limit 10)" >&2
    exit 1
  fi
  QUERY="${POSITIONAL[0]}"
  QUERY_LOWER="${QUERY,,}"
  if [[ "$QUERY_LOWER" == *"repo:"* || "$QUERY_LOWER" == *"org:"* || "$QUERY_LOWER" == *"user:"* ]]; then
    echo "Error: search query must not contain repo:, org:, or user: qualifiers (e.g., .github/scripts/gh.sh search issues \"bug report\" --limit 10)" >&2
    exit 1
  fi
  gh "$SUB1" "$SUB2" "$QUERY" --repo "$REPO" "${FLAGS[@]}"
elif [[ "$CMD" == "issue view" ]]; then
  if [[ ${#POSITIONAL[@]} -ne 1 ]] || ! [[ "${POSITIONAL[0]}" =~ ^[0-9]+$ ]]; then
    echo "Error: issue view requires exactly one numeric issue number (e.g., .github/scripts/gh.sh issue view 123)" >&2
    exit 1
  fi
  gh "$SUB1" "$SUB2" "${POSITIONAL[0]}" "${FLAGS[@]}"
else
  if [[ ${#POSITIONAL[@]} -ne 0 ]]; then
    echo "Error: label list does not accept positional arguments (e.g., .github/scripts/gh.sh label list --limit 100)" >&2
    exit 1
  fi
  gh "$SUB1" "$SUB2" "${FLAGS[@]}"
fi
