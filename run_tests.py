#!/usr/bin/python

# test-running driver for ispc

from optparse import OptionParser
import multiprocessing
from ctypes import c_int
import os
import sys
import glob
import re
import signal
import random
import string
import mutex
import subprocess
import shlex
import platform

is_windows = (platform.system() == 'Windows' or
              'CYGWIN_NT' in platform.system())

parser = OptionParser()
parser.add_option("-r", "--random-shuffle", dest="random", help="Randomly order tests",
                  default=False, action="store_true")
parser.add_option("-g", "--generics-include", dest="include_file", help="Filename for header implementing functions for generics",
                  default=None)
parser.add_option('-t', '--target', dest='target',
                  help='Set compilation target (sse2, sse2-x2, sse4, sse4-x2, avx, avx-x2, generic-4, generic-8, generic-16)',
                  default="sse4")
parser.add_option('-a', '--arch', dest='arch',
                  help='Set architecture (x86, x86-64)',
                  default="x86-64")
parser.add_option("-c", "--compiler", dest="compiler_exe", help="Compiler binary to use to run tests",
                  default=None)
parser.add_option('-o', '--no-opt', dest='no_opt', help='Disable optimization',
                  default=False, action="store_true")
parser.add_option('-v', '--verbose', dest='verbose', help='Enable verbose output',
                  default=False, action="store_true")
if not is_windows:
    parser.add_option('--valgrind', dest='valgrind', help='Run tests with valgrind',
                      default=False, action="store_true")

(options, args) = parser.parse_args()

if not is_windows and options.valgrind:
    valgrind_cmd = "valgrind "
else:
    valgrind_cmd = ""

is_generic_target = options.target.find("generic-") != -1
if is_generic_target and options.include_file == None:
    if options.target == "generic-4":
        print "No generics #include specified; using examples/intrinsics/sse4.h"
        options.include_file = "examples/intrinsics/sse4.h"
    elif options.target == "generic-8":
        print "No generics #include specified and no default available for \"generic-8\" target.";
        sys.exit(1)
    elif options.target == "generic-16":
        print "No generics #include specified; using examples/intrinsics/generic-16.h"
        options.include_file = "examples/intrinsics/generic-16.h"

if options.compiler_exe == None:
    if is_windows:
        options.compiler_exe = "cl"
    else:
        options.compiler_exe = "g++"

# if no specific test files are specified, run all of the tests in tests/
# and failing_tests/
if len(args) == 0:
    files = glob.glob("tests/*ispc") + glob.glob("failing_tests/*ispc") + \
        glob.glob("tests_errors/*ispc")
else:
    files = args

# randomly shuffle the tests if asked to do so
if (options.random):
    random.seed()
    random.shuffle(files)

# counter
total_tests = 0

# We'd like to use the Lock class from the multiprocessing package to
# serialize accesses to finished_tests_counter.  Unfortunately, the version of
# python that ships with OSX 10.5 has this bug:
# http://bugs.python.org/issue5261.  Therefore, we use the (deprecated but
# still available) mutex class.
#finished_tests_counter_lock = multiprocessing.Lock()
if not is_windows:
    finished_tests_mutex = mutex.mutex()
    finished_tests_counter = multiprocessing.Value(c_int)

# utility routine to print an update on the number of tests that have been
# finished.  Should be called with the mutex (or lock) held..
def update_progress(fn):
    finished_tests_counter.value = finished_tests_counter.value + 1
    progress_str = " Done %d / %d [%s]" % (finished_tests_counter.value, total_tests, fn)
    # spaces to clear out detrius from previous printing...
    for x in range(30):
        progress_str += ' '
    progress_str += '\r'
    sys.stdout.write(progress_str)
    sys.stdout.flush()
    finished_tests_mutex.unlock()

def run_command(cmd):
    if options.verbose:
        print "Running: %s" % cmd
    sp = subprocess.Popen(shlex.split(cmd), stdin=None,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    out = sp.communicate()
    output = ""
    output += out[0]
    output += out[1]
    return (sp.returncode, output)

# run the commands in cmd_list
def run_cmds(compile_cmds, run_cmd, filename, expect_failure):
    for cmd in compile_cmds:
        (return_code, output) = run_command(cmd)
        compile_failed = (return_code != 0)
        if compile_failed:
            print "Compilation of test %s failed            " % filename
            if output != "":
                print "%s" % output
            return (1, 0)

    (return_code, output) = run_command(run_cmd)
    run_failed = (return_code != 0)

    surprise = ((expect_failure and not run_failed) or
                (not expect_failure and run_failed))
    if surprise == True:
        print "Test %s %s (return code %d)            " % \
            (filename, "unexpectedly passed" if expect_failure else "failed",
             return_code)
    if output != "":
        print "%s" % output
    if surprise == True:
        return (0, 1)
    else:
        return (0, 0)


def run_test(filename):
    # is this a test to make sure an error is issued?
    want_error = (filename.find("tests_errors") != -1)
    if want_error == True:
        ispc_cmd = "./ispc --werror --nowrap %s --arch=%s --target=%s" % \
            (filename, options.arch, options.target)
        (return_code, output) = run_command(ispc_cmd)
        got_error = (return_code != 0)

        # figure out the error message we're expecting
        file = open(filename, 'r')
        firstline = file.readline()
        firstline = string.replace(firstline, "//", "")
        firstline = string.lstrip(firstline)
        firstline = string.rstrip(firstline)
        file.close()

        if (output.find(firstline) == -1):
            print "OUT %s" % filename
            print "Didnt see expected error message %s from test %s.\nActual output:\n%s" % \
                (firstline, filename, output)
            return (1, 0)
        elif got_error == False:
            print "Unexpectedly no errors issued from test %s" % filename
            return (1, 0)
        else:
            return (0, 0)
    else:
        # do we expect this test to fail?
        should_fail = (filename.find("failing_") != -1)

        # We need to figure out the signature of the test
        # function that this test has.
        sig2def = { "f_v(" : 0, "f_f(" : 1, "f_fu(" : 2, "f_fi(" : 3, 
                    "f_du(" : 4, "f_duf(" : 5, "f_di(" : 6 }
        file = open(filename, 'r')
        match = -1
        for line in file:
            # look for lines with 'export'...
            if line.find("export") == -1:
                continue
            # one of them should have a function with one of the
            # declarations in sig2def
            for pattern, ident in sig2def.items():
                if line.find(pattern) != -1:
                    match = ident
                    break
        file.close()
        if match == -1:
            print "Fatal error: unable to find function signature " + \
                  "in test %s" % filename
            return (1, 0)
        else:
            is_generic_target = options.target.find("generic-") != -1
            if is_generic_target:
                obj_name = "%s.cpp" % filename

            global is_windows
            if is_windows:
                if not is_generic_target:
                    obj_name = "%s.obj" % filename
                exe_name = "%s.exe" % filename

                cc_cmd = "%s /I. /Zi /nologo /DTEST_SIG=%d test_static.cpp %s /Fe%s" % \
                         (options.compiler_exe, match, obj_name, exe_name)
                if should_fail:
                    cc_cmd += " /DEXPECT_FAILURE"
            else:
                if not is_generic_target:
                    obj_name = "%s.o" % filename
                exe_name = "%s.run" % filename

                if options.arch == 'x86':
                    gcc_arch = '-m32'
                else:
                    gcc_arch = '-m64'
                cc_cmd = "%s -msse4.2 -I. %s test_static.cpp -DTEST_SIG=%d %s -o %s" % \
                         (options.compiler_exe, gcc_arch, match, obj_name, exe_name)
                if platform.system() == 'Darwin':
                    cc_cmd += ' -Wl,-no_pie'
                if should_fail:
                    cc_cmd += " -DEXPECT_FAILURE"

            ispc_cmd = "./ispc --woff %s -o %s --arch=%s --target=%s" % \
                       (filename, obj_name, options.arch, options.target)
            if options.no_opt:
                ispc_cmd += " -O0" 
            if is_generic_target:
                ispc_cmd += " --emit-c++ --c++-include-file=%s" % options.include_file
    
        # compile the ispc code, make the executable, and run it...
        global valgrind_cmd
        (compile_error, run_error) = run_cmds([ispc_cmd, cc_cmd], 
                                              valgrind_cmd + " " + exe_name, \
                                              filename, should_fail)
        # clean up after running the test
        try:
            if not run_error:
                os.unlink(exe_name)
                if is_windows:
                    os.unlink(filename + ".pdb")
                    os.unlink(filename + ".ilk")
            os.unlink(obj_name)
        except:
            None

        return (compile_error, run_error)

# pull tests to run from the given queue and run them.  Multiple copies of
# this function will be running in parallel across all of the CPU cores of
# the system.
def run_tasks_from_queue(queue, queue_ret):
    compile_error_files = [ ]
    run_error_files = [ ]
    while True:
        filename = queue.get()
        if (filename == 'STOP'):
            queue_ret.put((compile_error_files, run_error_files))
            sys.exit(0)

        (compile_error, run_error) = run_test(filename)
        if compile_error != 0:
            compile_error_files += [ filename ]
        if run_error != 0:
            run_error_files += [ filename ]

        # If not for http://bugs.python.org/issue5261 on OSX, we'd like to do this:
        #with finished_tests_counter_lock:
            #update_progress(filename)
        # but instead we do this...
        finished_tests_mutex.lock(update_progress, filename)

task_threads = []

def sigint(signum, frame):
    for t in task_threads:
        t.terminate()
    sys.exit(1)

if __name__ == '__main__':
    total_tests = len(files)

    compile_error_files = [ ]
    run_error_files = [ ]
    if is_windows:
        # cl.exe gets itself all confused if we have multiple instances of
        # it running concurrently and operating on the same .cpp file
        # (test_static.cpp), even if we are generating a differently-named
        # exe in the end.  So run serially. :-(
        nthreads = 1
        num_done = 0
        print "Running %d tests." % (total_tests)
        for fn in files:
            (compile_error, run_error) = run_test(fn)
            if compile_error != 0:
                compile_error_files += fn
            if run_error != 0:
                run_error_files += fn

            num_done += 1
            progress_str = " Done %d / %d [%s]" % (num_done, total_tests, fn)
            # spaces to clear out detrius from previous printing...
            for x in range(30):
                progress_str += ' '
            progress_str += '\r'
            sys.stdout.write(progress_str)
            sys.stdout.flush()
    else:
        nthreads = multiprocessing.cpu_count()
        print "Found %d CPUs. Running %d tests." % (nthreads, total_tests)

        # put each of the test filenames into a queue
        q = multiprocessing.Queue()
        for fn in files:
            q.put(fn)
        for x in range(nthreads):
            q.put('STOP')
        qret = multiprocessing.Queue()

        # need to catch sigint so that we can terminate all of the tasks if
        # we're interrupted
        signal.signal(signal.SIGINT, sigint)

        # launch jobs to run tests
        for x in range(nthreads):
            t = multiprocessing.Process(target=run_tasks_from_queue, args=(q,qret))
            task_threads.append(t)
            t.start()

        # wait for them to all finish and then return the number that failed
        # (i.e. return 0 if all is ok)
        for t in task_threads:
            t.join()
        print

        while not qret.empty():
            (c, r) = qret.get()
            compile_error_files += c
            run_error_files += r

    if len(compile_error_files) > 0:
        compile_error_files.sort()
        print "%d / %d tests FAILED compilation:" % (len(compile_error_files), total_tests)
        for f in compile_error_files:
            print "\t%s" % f
    if len(run_error_files) > 0:
        run_error_files.sort()
        print "%d / %d tests FAILED execution:" % (len(run_error_files), total_tests)
        for f in run_error_files:
            print "\t%s" % f

    sys.exit(len(compile_error_files) + len(run_error_files))
