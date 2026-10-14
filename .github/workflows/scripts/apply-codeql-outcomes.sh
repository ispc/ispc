#!/bin/bash
#
# Copyright 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Dismisses the false positives in batch/results.json and opens one issue for new Unsure alerts.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/codeql-common.sh"
issues=$(open_bot_issues)

# Dismisses only the analysis the worker examined, and only while the
# alert's file is unchanged on main. Otherwise a later run checks again.
dismiss() {
  local r=$1 n alert files changed comment state
  n=$(jq .alert_number <<< "$r")
  alert=$(gh api "repos/$GH_REPO/code-scanning/alerts/$n") || return 1
  if [ "$(jq -r .state <<< "$alert")" = dismissed ]; then
    note "Alert #$n: already dismissed."
    return 0
  fi
  if ! jq -e --argjson a "$(jq -c .alert <<< "$r")" '.state == "open" and .rule.id == $a.rule and
      .most_recent_instance.ref == "refs/heads/main" and .most_recent_instance.commit_sha == $a.analyzed_sha and
      .most_recent_instance.location.path == $a.path and .most_recent_instance.location.start_line == $a.line' \
      <<< "$alert" > /dev/null; then
    note "Alert #$n: changed since it was analyzed; not dismissed, a later run checks it again."
    return 0
  fi
  files=$(gh api "repos/$GH_REPO/compare/$(jq -r .alert.analyzed_sha <<< "$r")...main" --jq '[.files[]?.filename]') || return 1
  # The compare API lists at most 300 files. Any doubt means no dismissal.
  changed=$(jq --arg path "$(jq -r .alert.path <<< "$r")" 'length >= 300 or index($path) != null' <<< "$files") || return 1
  if [ "$changed" != false ]; then
    note "Alert #$n: its file changed on main since the analyzed commit; not dismissed, a later run checks it again."
    return 0
  fi
  comment=$(jq -r --arg run "$RUN_URL" '("Automated false positive (" + $run + "): " + .evidence)[:280]' <<< "$r")
  jq -n --arg comment "$comment" '{state: "dismissed", dismissed_reason: "false positive", dismissed_comment: $comment}' |
    gh api -X PATCH "repos/$GH_REPO/code-scanning/alerts/$n" --input - > /dev/null || return 1
  # A successful write is not proof; read the state back.
  state=$(gh api "repos/$GH_REPO/code-scanning/alerts/$n" --jq .state) || return 1
  if [ "$state" != dismissed ]; then
    note "Alert #$n: dismissal not confirmed (state: $state)."
    return 1
  fi
  note "Alert #$n: dismissed as a false positive."
}

# One issue for new Unsure alerts; existing issues keep their state and contents.
record_unsure_batch() {
  local reported count number title
  reported=$(jq -c '[.[].alerts[]] | unique' <<< "$issues") || return 1
  jq --argjson reported "$reported" '
    map(select(.execution_status == "complete" and .verdict == "unsure") |
      select(.alert_number as $n | ($reported | index($n)) == null)) |
    unique_by(.alert_number) | sort_by(.alert_number)' batch/results.json > batch/unsure.json || return 1
  count=$(jq length batch/unsure.json) || return 1
  if [ "$count" -eq 0 ]; then return 0; fi

  # Put the machine-readable index first, outside all agent-provided text.
  # Keep the issue concise; full reports remain in the workflow artifacts.
  jq -r --arg root "$GITHUB_SERVER_URL/$GH_REPO/security/code-scanning" --arg run "$RUN_URL" '
    def brief($limit):
      ($limit | floor | if . < 0 then 0 else . end) as $n |
      if length > $n then .[:$n] + "…" else . end;
    def block: "```text\n" + gsub("```"; "` ` `") + "\n```";
    . as $reports |
    ("<!-- codeql-unsure-alerts: " + (map(.alert_number) | tojson) + " -->\n\n" +
     "These alerts need a maintainer’s decision. Automatic runs skip alerts recorded in this issue, while it remains open. Closed issues do not suppress an open alert.\n\n" +
     "Run: " + $run + "\n\nFull reports are available in the workflow artifacts.\n\n") as $intro |
    (((60000 - ($intro | length)) / length) | floor) as $entry_budget |
    $intro + (map(
      ("### [CodeQL alert #\(.alert_number)](\($root)/\(.alert_number))\n\n") as $heading |
      ([0, $entry_budget - ($heading | length) - 140] | max) as $budget |
      $heading +
      (("\(.alert.rule) at \(.alert.path):\(.alert.line); commit \(.analyzed_sha)" |
        brief($budget / 6)) | block) + "\n\n" +
      "Open question:\n\n" + ((.unresolved_fact | brief($budget / 3)) | block) + "\n\n" +
      "Evidence:\n\n" + ((.evidence | brief($budget / 2)) | block)
    ) | join("\n\n"))' batch/unsure.json > batch/unsure-body.md || return 1

  gh api "repos/$GH_REPO/labels/codeql-unsure" > /dev/null 2>&1 ||
    gh api -X POST "repos/$GH_REPO/labels" -f name=codeql-unsure -f color=fbca04 \
      -f description="The CodeQL alert fixer needs a maintainer's answer" > /dev/null || return 1
  title="[CodeQL] Alerts requiring investigation"
  number=$(jq -n --arg title "$title" --rawfile body batch/unsure-body.md \
    '{title: $title, body: $body, labels: ["codeql-unsure"]}' |
    gh api -X POST "repos/$GH_REPO/issues" --input - --jq .number) || return 1
  note "Opened issue $GITHUB_SERVER_URL/$GH_REPO/issues/$number for Unsure alerts $(jq -r 'map("#\(.alert_number)") | join(", ")' batch/unsure.json)."
}

failed=0
while IFS= read -r r; do
  n=$(jq .alert_number <<< "$r")
  if [ "$(jq -r .execution_status <<< "$r")" != complete ]; then continue; fi
  case "$(jq -r .verdict <<< "$r")" in
    false-positive)
      dismiss "$r" || { note "Alert #$n: dismissal failed; see the job log."; failed=1; } ;;
    already-resolved)
      if state=$(gh api "repos/$GH_REPO/code-scanning/alerts/$n" --jq .state); then
        case "$state" in
          fixed|dismissed) note "Alert #$n: already $state." ;;
          *) note "Alert #$n: reported resolved, but the alert is $state; treated as incomplete, a later run retries." ;;
        esac
      else
        failed=1
      fi ;;
  esac
done < <(jq -c '.[]' batch/results.json)
record_unsure_batch || { note "Recording Unsure alerts failed; see the job log."; failed=1; }
exit "$failed"
