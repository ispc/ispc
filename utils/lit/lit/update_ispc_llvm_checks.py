#!/usr/bin/env python3

from sys import stderr
from traceback import print_exc
import argparse
import os
import re
import sys

from UpdateTestChecks import common


def update_test(ti):
    prefix_list = []

    # ----------------------------------------------------
    # 1. Parse RUN lines — only ISPC is supported
    # ----------------------------------------------------
    for line in ti.run_lines:
        if "|" not in line:
            continue

        # REQUIRE: must contain --emit-llvm-text
        if "--emit-llvm-text" not in line:
            common.error(f"RUN line missing required --emit-llvm-text:\n{line}")
            sys.exit(1)

        cropped = line
        # Handle conditional RUN lines like: %if %{ ... %}
        if "%if" in line:
            m = re.search(r"%{\s*(.*?)\s*%}", line)
            if m:
                cropped = m.group(1)

        commands = [cmd.strip() for cmd in cropped.split("|")]
        if len(commands) < 2:
            continue

        tool_cmd = commands[-2]
        filecheck_cmd = commands[-1]

        # Skip non-FileCheck RUN lines defensively
        if not filecheck_cmd.startswith("FileCheck "):
            continue

        common.verify_filecheck_prefixes(filecheck_cmd)

        # Extract raw arguments after `%{ispc}` or `ispc`
        m = re.match(r"^(?:%{ispc}|ispc)(.*)$", tool_cmd)
        if not m:
            # Non-ISPC tool, ignore in this simplified script
            continue

        tool_args = (
            m.group(1)
            .replace("< %s", "")
            .replace("%s", "")
            .strip()
        )

        prefixes = common.get_check_prefixes(filecheck_cmd)

        # preprocess_cmd is always None for ISPC in this simplified script
        prefix_list.append((prefixes, tool_args, None))

    # If there are no valid RUN lines, nothing to do.
    if not prefix_list:
        return

    # ----------------------------------------------------
    # 2. Run ISPC and gather IR from each RUN line
    # ----------------------------------------------------
    ginfo = common.make_ir_generalizer(ti.args.version, False)

    builder = common.FunctionTestBuilder(
        run_list=prefix_list,
        flags=ti.args,
        scrubber_args=[],
        path=ti.path,
        ginfo=ginfo,
    )

    tbaa_per_prefix = {}

    for prefixes, tool_args, _ in prefix_list:
        # ISPC invocation: always "ispc <args> file"
        ispc_args = (tool_args + " " + ti.path).strip()

        raw = common.invoke_tool(
            "ispc",
            ispc_args,
            ti.path,
            preprocess_cmd=None,
            verbose=ti.args.verbose,
        )

        builder.process_run_line(
            common.OPT_FUNCTION_RE,
            common.scrub_body,
            raw,
            prefixes,
        )
        builder.processed_prefixes(prefixes)

        tbaa_per_prefix[tuple(prefixes)] = common.get_tbaa_records(
            ti.args.version, raw
        )

    # ----------------------------------------------------
    # 3. Build checks normally (appended at the end initially)
    # ----------------------------------------------------
    prefix_set = {p for prefs, _, _ in prefix_list for p in prefs}
    original = common.collect_original_check_lines(ti, prefix_set)

    func_dict = builder.finish_and_get_func_dict()

    output_lines = []
    common.dump_input_lines(output_lines, ti, prefix_set, "//")

    common.add_checks_at_end(
        output_lines,
        prefix_list,
        builder.func_order(),
        "//",
        lambda lines, prefixes, func: common.add_ir_checks(
            lines,
            "//",
            prefixes,
            func_dict,
            func,
            ti.args.preserve_names,
            ti.args.function_signature,
            ginfo,
            {},                # no global-var checks in this simplified script
            tbaa_per_prefix,    # keep TBAA metadata available if needed
            is_filtered=builder.is_filtered(),
            original_check_lines=original.get(func, {}),
            check_inst_comments=ti.args.check_inst_comments,
        ),
    )

    # ----------------------------------------------------
    # 4. MOVE CHECK BLOCK ABOVE FIRST FUNCTION
    # ----------------------------------------------------
    check_re = re.compile(
        r"^[ \t]*(?:;|//)\s*CHECK_[A-Za-z0-9_-]+(?:-[A-Z0-9_-]+)?\s*:"
    )

    first_check = None
    for i, line in enumerate(output_lines):
        if check_re.search(line):
            first_check = i
            break

    if first_check is not None:
        checks = output_lines[first_check:]
        body = output_lines[:first_check]

        func_re = re.compile(
            r"""^\s*
                (?:export\s+)? (?:inline\s+)? (?:task\s+)? (?:unmasked\s+)?
                (?:uniform|varying)?\s*
                [A-Za-z_]\w*\s+[A-Za-z_]\w*\s*\(
            """,
            re.VERBOSE,
        )

        insert = None
        for i, line in enumerate(body):
            if func_re.match(line):
                insert = i
                break
        if insert is None:
            insert = len(body)

        merged = body[:insert] + checks + body[insert:]
    else:
        merged = output_lines[:]

    # ----------------------------------------------------
    # 5. Convert ";" → "//"
    # ----------------------------------------------------
    converted = []
    for line in merged:
        stripped = line.lstrip()
        if stripped.startswith(";"):
            leading_ws = line[: len(line) - len(stripped)]
            rest = stripped.lstrip(";")
            if rest.startswith(" "):
                rest = rest[1:]
            line = f"{leading_ws}// {rest}"
        converted.append(line)

    # ----------------------------------------------------
    # 6. Remove duplicate NOTE lines
    # ----------------------------------------------------
    final = []
    seen = set()
    for line in converted:
        stripped = line.strip()
        if stripped.startswith("// NOTE:"):
            if stripped in seen:
                continue
            seen.add(stripped)
        final.append(line)

    with open(ti.path, "wb") as f:
        f.writelines([l.encode("utf-8") + b"\n" for l in final])


def main():
    parser = argparse.ArgumentParser()

    # Options that make sense to tweak for ISPC tests:
    parser.add_argument(
        "--function-signature",
        action="store_true",
        help="Keep full function signature in CHECK-LABEL lines.",
    )
    parser.add_argument(
        "--preserve-names",
        action="store_true",
        help="Do not scrub IR value names; use them directly in checks.",
    )
    parser.add_argument(
        "--check-inst-comments",
        action="store_true",
        help="Emit FileCheck lines that also check instruction comments.",
    )

    # Internal options expected by `common`, but not useful to change for ISPC:
    parser.set_defaults(
        check_attributes=False,  # do not check attributes for ISPC tests
    )

    parser.add_argument("tests", nargs="+")
    args = common.parse_commandline_args(parser)

    script = os.path.basename(__file__)
    rc = 0

    for ti in common.itertests(args.tests, parser, script_name=script):
        try:
            update_test(ti)
        except Exception:
            stderr.write(f"Error updating test {ti.path}\n")
            print_exc()
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
