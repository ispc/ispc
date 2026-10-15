---
name: codex
description: Run OpenAI Codex CLI (via AWS Bedrock) non-interactively to get a second opinion, review a diff, or delegate a task. Use when the user asks to "run codex", "ask codex", or get a Codex review/answer.
---

# Running Codex

Codex is installed at `/usr/bin/codex` and configured in `~/.codex/config.toml`
to use the `amazon-bedrock-runtime` provider (region `us-east-2`). Always run it
non-interactively with `codex exec`; the interactive TUI cannot be driven from
Claude Code.

## 1. Precheck

```bash
echo "token: ${AWS_BEARER_TOKEN_BEDROCK:+set}"; echo "proxy: ${https_proxy:+set}"
```

Both must print `set`. Never print or log the token value. If either is
missing, stop and tell the user: the token comes from the Bedrock Token
Generator, and Bedrock is only reachable through `proxy-us.intel.com:911`.

## 2. Run

```bash
OUT=$(mktemp /tmp/codex-out.XXXXXX)
timeout 600 codex exec \
  -C /localdisk2/aneshlya/ispc/applications.compilers.ispc.core \
  -s read-only \
  -o "$OUT" \
  "<prompt>" </dev/null > /tmp/codex-log.txt 2>&1
echo "exit=$?"; cat "$OUT"
```

- `-o` writes only Codex's final message to `$OUT`; the full transcript goes to
  `/tmp/codex-log.txt`. Read the log only when the run fails.
- `</dev/null` is required, otherwise `codex exec` waits for stdin.
- Use `-s read-only` by default. Use `-s workspace-write` only when the user
  explicitly wants Codex to edit files, and review its changes with
  `git diff` afterwards.
- Runs can take several minutes: use a Bash timeout of 600000 ms, or
  `run_in_background` for long tasks.
- Codex does not see this conversation. Put everything it needs in the prompt:
  goal, relevant file paths, branch, and constraints. For a diff review, tell it
  to run `git diff` (or `git diff main...HEAD`) itself rather than pasting
  the diff.

## 3. Report

Relay Codex's answer to the user, clearly attributed to Codex. Do not treat its
claims as verified: check any finding against the code before acting on it, and
say which ones you confirmed.

## Troubleshooting

- `Model metadata for ... not found`: harmless warning, ignore it.
- `failed to load AWS credentials: the credential provider was not enabled`:
  `AWS_BEARER_TOKEN_BEDROCK` is missing or expired in the process making the
  request. For `codex exec`, fix the current environment. For the user's
  interactive `codex`, the background app-server daemon was started without the
  token. Check it without printing the value:
  ```bash
  PID=$(python3 -I -c 'import json,os;print(json.load(open(os.path.expanduser("~/.codex/app-server-daemon/daemon.pid")))["pid"])')
  tr '\0' '\n' < /proc/$PID/environ | grep -c '^AWS_BEARER_TOKEN_BEDROCK='
  ```
  If it prints `0`, kill that exact PID (never `pkill -f`, which matches the
  calling shell). Codex starts a new daemon on the next launch.
- `401` from `api.openai.com`: the config is not applied. Top-level keys
  (`model`, `model_provider`, ...) must come before the first `[table]` header in
  `~/.codex/config.toml`.
- Connection errors or hangs: the proxy env vars are missing.
