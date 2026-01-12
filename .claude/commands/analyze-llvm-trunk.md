---
description: Analyze LLVM trunk nightly build failures and propose fixes
allowed-tools: Bash(gh *), Bash(git *), Bash(cmake *), Bash(build/bin/ispc *), Read, Edit, Grep, Glob, Write, Task, WebFetch
---

# LLVM Trunk Failure Analyzer

Analyze the failed "Nightly Linux tests / LLVM trunk" workflow run and propose fixes.

## Arguments

$ARGUMENTS

Parse arguments:
- `--run-id <ID>` - Workflow run ID to analyze (optional, auto-detects latest failed run)
- `--current-sha <SHA>` - Current LLVM commit SHA
- `--previous-sha <SHA>` - Previous successful LLVM commit SHA

## Instructions

Use the Task tool to spawn the `llvm-trunk-analyzer` agent with this prompt:

```
Analyze the LLVM trunk nightly build failure.

Arguments: $ARGUMENTS

Follow your workflow to:
1. Fetch workflow logs (use --run-id if provided, otherwise find latest failed run)
2. Check recent ISPC commits (last 24 hours) for potential regressions in https://github.com/ispc/ispc
3. Categorize the failure (ISPC regression, LLVM API change, lit test failure, LLVM regression)
4. Analyze root cause
5. Propose a fix with code changes or draft an issue for LLVM bugs
```
