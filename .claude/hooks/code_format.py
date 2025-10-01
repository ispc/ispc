#!/usr/bin/env python3
"""
Script to format files edited by Claude:
- C/C++/ISPC files: use clang-format (version 20.1 required)
- Other files: remove trailing spaces
"""

import datetime
import json
import os
import re
# Subprocess is used with default shell which is False, it's safe and doesn't allow shell injection
# so we can ignore the Bandit warning
import subprocess #nosec
import sys
from glob import glob
from pathlib import Path

REQUIRED_CLANG_FORMAT_VERSION = "20.1"
CLANG_FORMAT_GLOBS = [
    "src/*.cpp", "src/*.h",
    "src/opt/*.cpp", "src/opt/*.h",
    "builtins/*.cpp", "builtins/*.hpp", "builtins/*.c",
    "benchmarks/01*/*.cpp", "benchmarks/01*/*.ispc",
    "benchmarks/02*/*.cpp", "benchmarks/02*/*.ispc",
    "common/*.h",
    "stdlib/stdlib.ispc",
    "stdlib/include/*.isph",
    "ispcrt/*.h", "ispcrt/*.hpp", "ispcrt/*.cpp",
    "ispcrt/detail/*.h",
    "ispcrt/detail/cpu/*.h", "ispcrt/detail/cpu/*.cpp",
    "ispcrt/detail/gpu/*.h", "ispcrt/detail/gpu/*.cpp",
    "ispcrt/tests/level_zero_mock/*.h", "ispcrt/tests/level_zero_mock/*.cpp",
    "ispcrt/tests/mock_tests/*.cpp",
]
LOG_PATH = Path(os.environ.get("CLAUDE_HOOK_LOG", "/tmp/claude_hook_debug.log"))


def log_debug(message, data=None):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"[{timestamp}] {message}\n")
            if data is not None:
                f.write(f"Data: {json.dumps(data, indent=2, ensure_ascii=False)}\n")
            f.write("---\n")
    except Exception:
        pass


def ensure_clang_format_version(required=REQUIRED_CLANG_FORMAT_VERSION):
    try:
        result = subprocess.run(
            ["clang-format", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        version_output = (result.stdout or result.stderr or "").strip()
        m = re.search(r"clang-format\s+version\s+(\d+\.\d+(?:\.\d+)?)", version_output)
        found = m.group(1) if m else version_output
        ok = str(found).startswith(required)
        if not ok:
            log_debug("Wrong clang-format version", {"found": found, "required": required})
            return False
        log_debug("Clang-format version check passed", {"version": found})
        return True
    except FileNotFoundError:
        log_debug("clang-format not found in PATH")
        return False
    except Exception as e:
        log_debug("Error checking clang-format", {"error": str(e)})
        return False


def enumerate_clang_format_targets():
    files = []
    for pattern in CLANG_FORMAT_GLOBS:
        files.extend(Path(p).resolve() for p in glob(pattern))
    return set(files)


def read_target_path_from_claude_json_stdin():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return None
        tool_args = json.loads(raw)
        ti = tool_args.get("tool_input", {}) if isinstance(tool_args, dict) else {}
        path_str = ti.get("file_path") or ti.get("path") or ""
        if not path_str:
            return None
        p = Path(path_str).resolve()
        return p if p.is_file() else None
    except Exception as e:
        log_debug("Error parsing stdin JSON for target path", {"error": str(e)})
        return None


def apply_clang_format(filepath):
    try:
        subprocess.run(["clang-format", "-i", str(filepath)], check=True, capture_output=True, text=True)
        log_debug("clang-format completed successfully", {"file": str(filepath)})
        return True
    except Exception as e:
        log_debug("Error running clang-format", {"file": str(filepath), "error": str(e)})
        return False


def strip_trailing_whitespace(filepath):
    try:
        with filepath.open("r", encoding="utf-8", newline="") as f:
            lines = f.readlines()

        processed = []
        for line in lines:
            if line.endswith("\n"):
                processed.append(line[:-1].rstrip(" \t") + "\n")
            else:
                processed.append(line.rstrip(" \t"))

        with filepath.open("w", encoding="utf-8", newline="") as f:
            f.writelines(processed)

        log_debug("Text processing completed", {"file": str(filepath)})
        return True
    except Exception as e:
        log_debug("Error processing text file", {"file": str(filepath), "error": str(e)})
        return False


def main():
    has_required_clang_format = ensure_clang_format_version()
    clang_targets = enumerate_clang_format_targets()
    target_path = read_target_path_from_claude_json_stdin()

    if not target_path:
        print("No valid file path found in input")
        sys.exit(0)

    if target_path in clang_targets and has_required_clang_format:
        apply_clang_format(target_path)
    else:
        strip_trailing_whitespace(target_path)

    print("Done!")
    sys.exit(0)


if __name__ == "__main__":
    main()
