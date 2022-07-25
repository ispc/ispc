#!/usr/bin/python3

DEBUG = False

import os
import stat
import subprocess
import sys
import subprocess
import signal

# Representation of output which goes to console.
# It consist of stdout & stderr.
class Output(object):
    stdout: str
    stderr: str
    def __init__(self, stdout, stderr):
        self.stdout = stdout
        self.stderr = stderr

    # By default we can get string from instance of this class
    # which will return merged stdout with stderr.
    def __str__(self):
        return self.stdout + self.stderr

class TestCommandTool:
    def run(self, cmd, timeout, cwd = os.getcwd(), test_env = os.environ.copy(), print_output = True):
        exit_code = 0
        output = ""

        proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=None, universal_newlines=True, env=test_env)
        try:
            out = proc.communicate(timeout=timeout)
            exit_code = proc.poll()
            output = Output(out[0], out[1])
            if print_output:
                print(output, flush=True)

        except subprocess.TimeoutExpired:
            print("Timeout Expired, going to kill process", flush=True)
            # force to kill process
            self._kill(proc)
            # Timeout exit code
            exit_code = 124

        return (exit_code, output)

    def _kill(self, proc_to_kill):
        pid = proc_to_kill.pid
        print("Killing process " + str(pid), flush=True)
        # Windows
        if os.name == 'nt':
            proc = subprocess.Popen(['taskkill', '/F', '/T', '/PID', str(pid)], shell=True)
            proc.wait()
        # Linux
        else:
            proc = subprocess.Popen('pkill -TERM -P '+ str(pid), shell=True)
            proc.wait()

# checks whether print ouput is correct
# (whether test and reference outputs are same)
# NOTE: output contains both test and reference lines
def check_print_output(output):
    # if message about spill size is appeared, remove it from output
    spill_line_idx = output.find("Spill")
    if spill_line_idx != -1:
        output = output[0:spill_line_idx]

    lines = output.splitlines()
    if len(lines) == 0 or len(lines) % 2:
        return False
    else:
        return lines[0:len(lines)//2] == lines[len(lines)//2:len(lines)]

def run(run_debug = False):
    test_env = os.environ.copy()
    if run_debug:
        test_env["ISPCRT_IGC_OPTIONS"] = "+ -g"

    # find binary
    for element in os.scandir('.'):
        if element.is_file() and '.ispc.' in element.name:
             binName = element.name
             break

    if os.name == 'nt':
        binTest = binName
    else:
        binTest = "./" + binName

    print("Executing " + binTest)
    st = os.stat(binTest)
    os.chmod(binTest, st.st_mode | stat.S_IEXEC)

    exit_code, output = TestCommandTool().run(binTest, None, os.getcwd(), test_env)
    if exit_code != 0:
        print("Test execution failed (%d):\n%s" % (exit_code, output))
        exit(exit_code)

    # early return if not print test
    if not binName.startswith("print"):
        exit(0)

    output_equality = check_print_output(output.stdout)
    if not output_equality:
        print("Print outputs check failed\n")
        exit(1)

if __name__ == '__main__':
    run(DEBUG)
