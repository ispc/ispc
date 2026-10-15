#!/bin/bash
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Validates each investigation result of the CodeQL Alert Fixer and collects them in batch/.

set -euo pipefail
mkdir -p batch/patches
# Keep original responses and patches, including unusable reports, for inspection.
if [ -d results ]; then cp -R results batch/reports; fi
: > batch/results.jsonl
while IFS= read -r alert; do
  n=$(jq .number <<< "$alert")
  # download-artifact extracts a single matching artifact directly into results/.
  dir="results/codeql-alert-$n-$GITHUB_RUN_ID-$BASE_SHA/$n"
  if [ ! -d "$dir" ]; then dir="results/$n"; fi
  file="$dir/result.json"
  reason="missing result"
  if [ -f "$file" ]; then reason="invalid or incomplete result (see reports/${dir#results/}/result.json)"; fi
  # Require one explicit completed decision. Optional investigation
  # details are data for the integrator, never commands for this job.
  if [ -f "$file" ] && jq -se --argjson alert "$alert" '
      length == 1 and (.[0] |
        type == "object" and
        ((has("alert_number") | not) or .alert_number == $alert.number) and
        ((has("analyzed_sha") | not) or .analyzed_sha == $alert.analyzed_sha) and
        .execution_status == "complete" and
        (.verdict | IN("bug", "false-positive", "unsure", "already-resolved")) and
        .error == null and
        ((has("fix_available") | not) or .fix_available == (.verdict == "bug")) and
        (if .verdict == "false-positive" or .verdict == "unsure"
         then (.evidence | type == "string" and test("\\S")) else true end) and
        (if .verdict == "unsure"
         then (.unresolved_fact | type == "string" and test("\\S"))
         else .unresolved_fact == null end))' "$file" > /dev/null; then
    result=$(jq -c --argjson alert "$alert" '. + {alert_number: $alert.number,
      analyzed_sha: $alert.analyzed_sha, alert: $alert}' "$file")
  else
    result=$(jq -nc --argjson alert "$alert" --arg reason "$reason" '{alert_number: $alert.number,
      alert: $alert, verdict: null, execution_status: "incomplete", error: $reason}')
  fi
  if jq -e '.execution_status == "complete" and .verdict == "bug"' <<< "$result" > /dev/null; then
    if [ -s "$dir/fix.patch" ]; then
      cp "$dir/fix.patch" "batch/patches/$n.patch"
    else
      result=$(jq -c '.execution_status = "incomplete" | .error = "bug verdict without a patch"' <<< "$result")
    fi
  fi
  echo "$result" >> batch/results.jsonl
  jq -r '"- Alert #\(.alert_number): \(.verdict // "no verdict"), \(.execution_status)." +
    (if .error then " " + .error else "" end)' <<< "$result" >> "$GITHUB_STEP_SUMMARY"
done < <(jq -c '.[]' <<< "$MATRIX")
jq -s . batch/results.jsonl > batch/results.json
rm batch/results.jsonl
echo "has_fixes=$(jq 'any(.[]; .execution_status == "complete" and .verdict == "bug")' batch/results.json)" >> "$GITHUB_OUTPUT"
