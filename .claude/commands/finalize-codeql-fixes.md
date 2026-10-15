---
description: Push the validated CodeQL fixes of one run to the codeql/fixes branch
---

# CodeQL Fixes Finalizer

Combine the validated fixes of one CodeQL Alert Fixer run on the codeql/fixes branch and push it.

## Arguments

$ARGUMENTS

Parse arguments:
- `--run-id <RUN_ID>` - CodeQL Alert Fixer run with the validated batch
- `--base-sha <BASE_SHA>` - selected main snapshot
- `--branch-sha <BRANCH_SHA>` - selected codeql/fixes SHA, or `none` if the branch was absent
- `--covered <COVERED>` - alert IDs with pending fixes on codeql/fixes, as a JSON array

## Instructions

Use the Task tool to spawn the `codeql-fixes-finalizer` agent with this prompt:

```
Push the validated CodeQL fixes to the codeql/fixes branch.
Follow your instructions.

Arguments: $ARGUMENTS
```

Give the agent's final report as your final message.
