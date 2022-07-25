#!/usr/bin/python3
import os
import glob
import shutil
import subprocess
import sys
from pathlib import Path


def replace_in_file(file_name, old_value, new_value):
    with open(file_name, 'r') as in_file:
        file_lines = []
        for line in in_file.readlines():
            file_lines.append(line.replace(old_value, new_value))
    with open(file_name, 'w') as out_file:
        for line in file_lines:
            out_file.write(line)

def main():
    print(str(sys.argv))

    arch = sys.argv[1]
    target = sys.argv[2]
    o_level = sys.argv[3]
    dst_dir_name = sys.argv[4]

    generate_debug = False
    if len(sys.argv) > 5 and sys.argv[5] == "debug":
        generate_debug = True

    ci_project_dir = os.getenv("GITHUB_WORKSPACE")

    dst_base_dir = os.path.join(os.getcwd(), dst_dir_name)
    Path(dst_base_dir).mkdir(parents=True, exist_ok=True)

    if target == "gen9-x8":
        test_postfix = ".simd8.spv"
    elif target == "gen9-x16":
        test_postfix = ".simd16.spv"
    else:
        print("Unsupported target: " + target)
        sys.exit(1)

    test_path_pattern = os.path.join(os.getcwd(), "tests") + "/*.ispc"
    print(test_path_pattern)

    tests_count = 0
    skip_count = 0
    skip_on_tgllp_count = 0
    skip_on_dg2_count = 0
    failing_tests = []

    run_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "run.py")

    run_tests_extra_params = ""
    if generate_debug:
        run_tests_extra_params = "--ispc-flags -g"
        replace_in_file(run_file_path, "DEBUG = False", "DEBUG = True")

    for test in sorted(glob.glob(test_path_pattern)):

        base_test_name = os.path.basename(test)

        # Check if test is skipped for target architecture
        with open(test) as f:
            file_content = f.read()
            if "skip on arch=%s"%(arch) in file_content:
                print("SKIP: Test %s is skipped due to metainfo in source code for arch %s" % (base_test_name, arch))
                skip_count += 1
                continue

        if os.name == 'nt':
            cmd = 'python run_tests.py --l0loader=%s\level-zero %s' % (ci_project_dir, run_tests_extra_params)
        else:
            cmd = './run_tests.py %s' % (run_tests_extra_params)

        tests_count += 1

        # For debug we want to use only 250 test cases for each simd width
        if tests_count > 250 and generate_debug:
            break

        cmd += " -a %s -t %s -o %s --save-bin %s --ispc_output=spv --test_time 20" % (arch, target, o_level, "tests/"+base_test_name)
        print("execute: %s"%(cmd))
        proc = subprocess.Popen(cmd, shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=None, universal_newlines=True)
        out = proc.communicate()
        exit_code = proc.poll()
        if exit_code != 0:
            failing_tests.append(base_test_name)
            print("XXX Compile error for test %s:\n%s" % (base_test_name, out[0]))
            continue

        found_binary = False
        # Find generated binary
        for path in Path(os.getcwd()).glob("tmp*/"+base_test_name+"*"):
            src_dir = os.path.dirname(path)
            test_name = base_test_name + test_postfix

            dst_dir = os.path.join(dst_base_dir, test_name)
            shutil.move(src_dir, dst_dir)

            # add skip files if needed
            with open(os.path.join("tests", base_test_name)) as f:
                file_content = f.read()
                if "skip on cpu=tgllp" in file_content:
                    open(os.path.join(dst_dir, "tgllp.skip"), 'w+').close()
                    open(os.path.join(dst_dir, "dg1.skip"), 'w+').close()
                    skip_on_tgllp_count += 1
                if "skip on cpu=dg2" in file_content:
                    open(os.path.join(dst_dir, "dg2.skip"), 'w+').close()
                    skip_on_dg2_count += 1

            # remove unecessary files
            static_obj_file = os.path.join(dst_dir, "test_static_l0.obj")
            if os.path.exists(static_obj_file):
                os.remove(static_obj_file)
            found_binary = True
            # for loop should have only one iteration
            break

        if not found_binary:
            failing_tests.append(base_test_name)
            print("XXX Binary not found for test %s:" % (base_test_name))

    # Copy run.py to all directories with test binaries
    for test_dir in glob.glob(dst_base_dir + "/*.ispc.*"):
        shutil.copy(run_file_path, test_dir)

    failing_tests_count = len(failing_tests)
    passing_tests_count = tests_count - failing_tests_count

    print("==== Generating tests skipped:  %d ====" % (skip_count))
    print("==== Generating tests passing / total:  %d / %d ====" % (passing_tests_count, tests_count))
    print("==== Including tests skipped on TGLLP: %d, and DG2: %d ====" % (skip_on_tgllp_count, skip_on_dg2_count))
    for test in failing_tests:
        print("FAILED: %s" % (test))


if __name__ == '__main__':
    main()
