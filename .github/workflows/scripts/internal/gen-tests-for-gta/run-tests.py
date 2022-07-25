#!/usr/bin/python3
import os
import glob
import subprocess
import sys

print(str(sys.argv))

dst_dir_name = sys.argv[1]
dst_base_dir = os.path.join(os.getcwd(), dst_dir_name)

tests_count = 0
failing_tests = []
# Run all tests
for test_dir in glob.glob(dst_base_dir + "/*.ispc.*"):
    if os.name == 'nt':
        cmd = 'python run.py'
    else:
        cmd = './run.py'
    tests_count += 1


    print("Executing: " + cmd + " in dir: " + test_dir)
    proc = subprocess.Popen(cmd, shell=True, cwd=test_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=None, universal_newlines=True)
    out = proc.communicate()
    exit_code = proc.poll()
    if exit_code != 0:
        print("XXX Test " + test_dir + " failed with exit code: " + str(exit_code))
        failing_tests.append(test_dir)
    print(out[0])

failing_tests_count = len(failing_tests)
passing_tests_count = tests_count - failing_tests_count

print("==== Tests passing / total:  %d / %d ====" % (passing_tests_count, tests_count))
for test in failing_tests:
    print("FAILED: %s" % (test))
