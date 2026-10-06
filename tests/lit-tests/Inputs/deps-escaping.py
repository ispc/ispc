# Copyright (c) 2026, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
import shutil
import subprocess  # nosec B404: Run the compiler under test without a shell.
import sys


ispc = shutil.which(sys.argv[1])
assert ispc, "ISPC executable not found"
ispc = str(Path(ispc).resolve())
directory = Path(sys.argv[2]).resolve()
directory.mkdir(parents=True, exist_ok=True)
common = [ispc, f"--target={sys.argv[3]}", "--nostdlib", "--nowrap"]


def run(*args):
    return subprocess.run(common + list(args), cwd=directory, check=True,
                          capture_output=True, text=True).stdout


def check(actual, expected):
    assert actual == expected, f"expected {expected!r}, got {actual!r}"


(directory / "source.ispc").write_text("uniform int value;\n")

# -MT is raw, while -MQ and implicit targets quote Make syntax. Repeated
# target options retain ISPC's existing last-option-wins behavior.
check(run("source.ispc", "-M", "-MT", "$(OBJDIR)/one.o two.o"),
      "$(OBJDIR)/one.o two.o: source.ispc\n")
check(run("source.ispc", "-M", "-MQ", "$(OBJDIR)/one.o two.o"),
      "$$(OBJDIR)/one.o\\ two.o: source.ispc\n")
check(run("source.ispc", "-M", "-MQ", "unused", "-MT", "$(OBJDIR)/out.o"),
      "$(OBJDIR)/out.o: source.ispc\n")
check(run("source.ispc", "-M", "-MT", "unused", "-MQ", "$(OBJDIR)/out.o"),
      "$$(OBJDIR)/out.o: source.ispc\n")

# These are target strings only, so they can be tested on Windows too.
targets = [
    ("space name.o", "space\\ name.o"),
    ("tab\tname.o", "tab\\\tname.o"),
    (r"back\slash.o", r"back\slash.o"),
    (r"two\\slashes.o", r"two\\slashes.o"),
    (r"back\ space.o", r"back\\\ space.o"),
    (r"two\\ space.o", r"two\\\\\ space.o"),
    ("back\\\tname.o", "back\\\\\\\tname.o"),
    ("hash#dollar$.o", "hash\\#dollar$$.o"),
    ("C:/some path/out.o", "C\\:/some\\ path/out.o"),
    (r"C:\some path\out.o", r"C\:\some\ path\out.o"),
]
for target, quoted in targets:
    check(run("source.ispc", "-M", "-MQ", target), f"{quoted}: source.ispc\n")
    check(run("source.ispc", "-M", "-MT", target), f"{target}: source.ispc\n")

# Exercise both the source path and preprocessor-registered include paths.
# The latter arrive C-escaped and must be unescaped before Make quoting.
paths = [("space #$.ispc", "space\\ \\#$$.ispc")]
if os.name != "nt":
    paths += [
        ("tab\tname.ispc", "tab\\\tname.ispc"),
        (r"back\slash.ispc", r"back\slash.ispc"),
        (r"back\ space.ispc", r"back\\\ space.ispc"),
        ("colon:name.ispc", "colon\\:name.ispc"),
    ]
for filename, quoted in paths:
    (directory / filename).write_text("uniform int value;\n")
    check(run(filename, "-M"), f"{quoted[:-5]}.o: {quoted}\n")
    check(run(filename, "-M", "-MT", "out.o"), f"out.o: {quoted}\n")
    run(filename, "-M", "-MF", "deps.d", "-MQ", "out file.o")
    check((directory / "deps.d").read_text(), f"out\\ file.o: {quoted}\n")
    # Include names are header-name tokens, not C string literals.
    (directory / "include.ispc").write_text(f'#include "{filename}"\n')
    included = run("include.ispc", "-M", "-MT", "out.o")
    # Clang prefixes an include found beside the source with the native ./.
    check(included.replace("\n .\\", "\n ./"),
          f"out.o: include.ispc \\\n ./{quoted}\n")

# A named output also determines the implicit target. Flat dependency output
# must keep its original (non-Make) format.
run("source.ispc", "-M", "-MF", "deps.d", "-o", "out #$.o")
check((directory / "deps.d").read_text(), "out\\ \\#$$.o: source.ispc\n")
run("space #$.ispc", "-MMM", "flat.d")
check((directory / "flat.d").read_text(), "space #$.ispc\n")

missing = subprocess.run(common + ["source.ispc", "-MQ"], cwd=directory,
                         capture_output=True, text=True)
assert missing.returncode != 0
assert "No target name specified after -MQ option." in missing.stderr
