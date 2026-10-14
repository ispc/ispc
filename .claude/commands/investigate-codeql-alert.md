---
description: Investigate one CodeQL alert and export the verdict and fix
---

# CodeQL Alert Investigation

Investigate one CodeQL alert for the CodeQL Alert Fixer workflow, which runs this once per alert.

## Arguments

$ARGUMENTS

Parse arguments:
- `--alert <ALERT>` - CodeQL alert number
- `--analyzed-sha <ANALYZED_SHA>` - commit the alert was reported for
- `--base-sha <BASE_SHA>` - main commit of the checkout
- `--input-dir <INPUT_DIR>` - directory with alert.json
- `--result-dir <RESULT_DIR>` - directory for result.json and fix.patch

## Instructions

Use the Task tool to spawn the `codeql-alert-investigator` agent with this prompt:

```
Investigate the CodeQL alert and write the result files.
Follow your instructions.

Arguments: $ARGUMENTS
```
