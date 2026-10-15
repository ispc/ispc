---
description: Combine and test the validated CodeQL fixes of one run on the codeql/fixes branch
---

# CodeQL Fixes Finalizer

Combine the validated fixes of one CodeQL Alert Fixer run on the codeql/fixes branch and test them.

## Arguments

$ARGUMENTS

Parse arguments:
- `--run-id <RUN_ID>` - CodeQL Alert Fixer run with the validated batch
- `--base-sha <BASE_SHA>` - selected main snapshot
- `--branch-sha <BRANCH_SHA>` - selected codeql/fixes SHA, or `none` if the branch was absent
- `--covered <COVERED>` - alert IDs with pending fixes on codeql/fixes, as a JSON array
- `--batch-dir <BATCH_DIR>` - directory with results.json and patches/
- `--output-dir <OUTPUT_DIR>` - directory for pr-body.md

## Instructions

Use the Agent tool to spawn the `codeql-fixes-finalizer` agent with this prompt:

```
Combine the validated CodeQL fixes on the codeql/fixes branch.
Follow your instructions.

Arguments: $ARGUMENTS
```

Give the agent's final report as your final message.
