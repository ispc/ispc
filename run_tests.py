#!/usr/bin/env python3
#
#  Copyright (c) 2013-2020, Intel Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Supported operating systems
from enum import Enum, unique
@unique
class OS(Enum):
    Unknown = 0
    Windows = 1
    Linux = 2
    Mac = 3
    FreeBSD = 4

@unique
class Status(Enum):
    Success = 0
    Compfail = 1
    Runfail = 2
    Skip = 3

StatusStr = {Status.Success: "PASSED",
             Status.Compfail: "FAILED compilation",
             Status.Runfail: "FAILED execution",
             Status.Skip: "SKIPPED",
            }

# The description of host testing system
class Host(object):
    def set_os(self, system):
        if system == 'Windows' or 'CYGWIN_NT' in system:
            self.os = OS.Windows
        elif system == 'Darwin':
            self.os = OS.Mac
        elif system == 'Linux':
            self.os = OS.Linux
        elif system == 'FreeBSD':
            self.os = OS.FreeBSD
        else:
            self.os = OS.Unknown

    # set ispc exe using ISPC_HOME or PATH environment variables
    def set_ispc_exe(self):
        ispc_exe = ""
        ispc_ext = ""
        if self.is_windows():
            ispc_ext = ".exe"
        if "ISPC_HOME" in os.environ:
            if os.path.exists(os.environ["ISPC_HOME"] + os.sep + "ispc" + ispc_ext):
                ispc_exe = os.environ["ISPC_HOME"] + os.sep + "ispc" + ispc_ext
        PATH_dir = os.environ["PATH"].split(os.pathsep)
        for counter in PATH_dir:
            if ispc_exe == "":
                if os.path.exists(counter + os.sep + "ispc" + ispc_ext):
                    ispc_exe = counter + os.sep + "ispc" + ispc_ext
        # checks the required ispc compiler otherwise prints an error message
        if ispc_exe == "":
            error("ISPC compiler not found.\nAdded path to ispc compiler to your PATH variable or ISPC_HOME variable\n", 1)
        # use relative path
        self.ispc_exe = os.path.relpath(ispc_exe, os.getcwd())

    def __init__(self, system):
        self.set_os(system)
        self.set_ispc_exe()

    def is_windows(self):
        return self.os == OS.Windows

    def is_linux(self):
        return self.os == OS.Linux

    def is_mac(self):
        return self.os == OS.Mac

    def is_freebsd(self):
        return self.os == OS.FreeBSD

    def set_ispc_cmd(self, ispc_flags):
        self.ispc_cmd = self.ispc_exe + " " + ispc_flags

# The description of testing target configuration
class TargetConfig(object):
    def __init__(self, arch, target, include_file):
        self.arch = arch
        self.target = target
        self.generic = target.find("generic-") != -1 and target != "generic-1" and target != "generic-x1"
        self.include_file = include_file
        self.set_target()

    def is_generic(self):
        return self.generic

    # set arch/target (and include_file for generic targets)
    def set_target(self):
        if self.target == 'neon':
            self.arch = 'aarch64'

        if self.is_generic() and self.include_file == None:
            if self.target == "generic-4" or self.target == "generic-x4":
                error("No generics #include specified; using examples/intrinsics/sse4.h\n", 2)
                self.include_file = "examples/intrinsics/sse4.h"
                self.target = "generic-4"
            elif self.target == "generic-8" or self.target == "generic-x8":
                error("No generics #include specified and no default available for \"generic-8\" target.\n", 1)
                self.target = "generic-8"
            elif self.target == "generic-16" or self.target == "generic-x16":
                error("No generics #include specified; using examples/intrinsics/generic-16.h\n", 2)
                self.include_file = "examples/intrinsics/generic-16.h"
                self.target = "generic-16"
            elif self.target == "generic-32" or self.target == "generic-x32":
                error("No generics #include specified; using examples/intrinsics/generic-32.h\n", 2)
                self.include_file = "examples/intrinsics/generic-32.h"
                self.target = "generic-32"
            elif self.target == "generic-64" or self.target == "generic-x64":
                error("No generics #include specified; using examples/intrinsics/generic-64.h\n", 2)
                self.include_file = "examples/intrinsics/generic-64.h"
                self.target = "generic-64"

# test-running driver for ispc
# utility routine to print an update on the number of tests that have been
# finished.  Should be called with the lock held..
def update_progress(fn, total_tests_arg, counter, max_test_length_arg):
    counter.value += 1
    if options.non_interactive == False:
        progress_str = " Done %d / %d [%s]" % (counter.value, total_tests_arg, fn)
        # spaces to clear out detrius from previous printing...
        spaces_needed = max_test_length_arg - len(fn)
        for x in range(spaces_needed):
            progress_str += ' '
        progress_str += '\r'
        sys.stdout.write(progress_str)
        sys.stdout.flush()

# 240 is enough even for longest test under sde.
def run_command(cmd, timeout=600):
    if options.verbose:
        print_debug("Running: %s\n" % cmd, s, run_tests_log)

    # Here's a bit tricky part. To pass a command for execution we should
    # break down the line in to arguments. shlex class is designed exactly
    # for this purpose, but by default it interprets escape sequences.
    # On Windows backslaches are all over the place and they are treates as
    # ESC-sequences, so we have to set manually to not interpret them.
    lexer = shlex.shlex(cmd, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ''
    arg_list = list(lexer)

    # prepare for OSError exceptions raised in the child process (re-raised in the parent)
    try:
        proc = subprocess.Popen(arg_list, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        print_debug("ERROR: The child (%s) raised an exception: %s\n" % (arg_list, sys.exc_info()[1]), s, run_tests_log)
        raise

    is_timeout = False
    # read data from stdout and stderr
    try:
        out = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out = proc.communicate()
        is_timeout = True
    except:
        print_debug("ERROR: The child (%s) raised an exception: %s\n" % (arg_list, sys.exc_info()[1]), s, run_tests_log)
        raise

    output = ""
    output += out[0].decode("utf-8")
    output += out[1].decode("utf-8")
    return (proc.returncode, output, is_timeout)

# run the commands in cmd_list
def run_cmds(compile_cmds, run_cmd, filename, expect_failure):
    for cmd in compile_cmds:
        (return_code, output, timeout) = run_command(cmd, 10)
        compile_failed = (return_code != 0)
        if compile_failed:
            print_debug("Compilation of test %s failed %s           \n" % (filename, "due to TIMEOUT" if timeout else ""), s, run_tests_log)
            if output != "":
                print_debug("%s" % output, s, run_tests_log)
            return Status.Compfail
    if not options.save_bin:
        (return_code, output, timeout) = run_command(run_cmd)
        run_failed = (return_code != 0) or timeout
    else:
        run_failed = 0
    surprise = ((expect_failure and not run_failed) or
                (not expect_failure and run_failed))
    if surprise == True:
        print_debug("Test %s %s (return code %d)            \n" % \
            (filename, "unexpectedly passed" if expect_failure else "failed",
             return_code), s, run_tests_log)
    if output:
        print_debug("%s\n" % output, s, run_tests_log)
    if surprise == True:
        return Status.Runfail
    else:
        return Status.Success


def add_prefix(path, host):
    if host.is_windows():
    # On Windows we run tests in tmp dir, so the root is one level up.
        input_prefix = "..\\"
    else:
        input_prefix = ""
    path = input_prefix + path
    path = os.path.abspath(path)
    return path

# FIXME: needs documentation
def check_test(filename, host, target):
    prev_arch = False
    prev_os = False
    done_arch = True
    done_os = True
    done = True
    if host.is_windows():
        oss = "windows"
    elif host.is_linux():
        oss = "linux"
    elif host.is_mac():
        oss = "mac"
    elif host.is_freebsd():
        oss = "freebsd"
    else:
        oss = "unknown"

    with open(add_prefix(filename, host)) as f:
        b = f.read()
    for run in re.finditer('// *rule: run on .*', b):
        arch = re.match('.* arch=.*', run.group())
        if arch != None:
            if re.search(' arch='+target.arch+'$', arch.group()) != None:
                prev_arch = True
            if re.search(' arch='+target.arch+' ', arch.group()) != None:
                prev_arch = True
            done_arch = prev_arch
        OS = re.match('.* OS=.*', run.group())
        if OS != None:
            if re.search(' OS='+oss, OS.group()) != None:
                prev_os = True
            done_os = prev_os
    done = done_arch and done_os
    for skip in re.finditer('// *rule: skip on .*', b):
        if re.search(' arch=' + target.arch + '$', skip.group())!=None:
            done = False
        if re.search(' arch=' + target.arch + ' ', skip.group())!=None:
            done = False
        if re.search(' OS=' + oss, skip.group())!=None:
            done = False
    return done


def run_test(testname, host, target):
    # testname is a path to the test from the root of ispc dir
    # filename is a path to the test from the current dir
    # ispc_exe_rel is a relative path to ispc
    filename = add_prefix(testname, host)
    ispc_exe_rel = add_prefix(host.ispc_cmd, host)

    # is this a test to make sure an error is issued?
    want_error = (filename.find("tests_errors") != -1)
    if want_error == True:
        ispc_cmd = ispc_exe_rel + " --werror --nowrap %s --arch=%s --target=%s" % \
            (filename, target.arch, target.target)
        (return_code, output, timeout) = run_command(ispc_cmd, 10)
        got_error = (return_code != 0) or timeout

        # figure out the error message we're expecting
        file = open(filename, 'r')
        firstline = file.readline()
        firstline = firstline.replace("//", "")
        firstline = firstline.lstrip()
        firstline = firstline.rstrip()
        file.close()

        if re.search(firstline, output) == None:
            print_debug("Didn't see expected error message %s from test %s.\nActual output:\n%s\n" % \
                (firstline, testname, output), s, run_tests_log)
            return Status.Compfail
        elif got_error == False:
            print_debug("Unexpectedly no errors issued from test %s\n" % testname, s, run_tests_log)
            return Status.Compfail
        else:
            return Status.Success
    else:
        # do we expect this test to fail?
        should_fail = (testname.find("failing_") != -1)

        # We need to figure out the signature of the test
        # function that this test has.
        sig2def = { "f_v(" : 0, "f_f(" : 1, "f_fu(" : 2, "f_fi(" : 3,
                    "f_du(" : 4, "f_duf(" : 5, "f_di(" : 6, "f_sz" : 7 }
        file = open(filename, 'r')
        match = -1
        for line in file:
            # look for lines with 'export'...
            if line.find("export") == -1:
                continue
            # one of them should have a function with one of the
            # declarations in sig2def
            for pattern, ident in list(sig2def.items()):
                if line.find(pattern) != -1:
                    match = ident
                    break
        file.close()
        if match == -1:
            error("unable to find function signature in test %s\n" % testname, 0)
            return Status.Compfail
        else:
            if host.is_windows():
                if target.is_generic():
                    obj_name = "%s.cpp" % os.path.basename(filename)
                else:
                    obj_name = "%s.obj" % os.path.basename(filename)
                exe_name = "%s.exe" % os.path.basename(filename)

                cc_cmd = "%s /I. /Zi /nologo /DTEST_SIG=%d %s %s /Fe%s" % \
                         (options.compiler_exe, match, add_prefix("test_static.cpp", host), obj_name, exe_name)
                if should_fail:
                    cc_cmd += " /DEXPECT_FAILURE"
            else:
                if target.is_generic():
                    obj_name = "%s.cpp" % testname
                else:
                    obj_name = "%s.o" % testname
                exe_name = "%s.run" % testname

                if target.arch == 'arm':
                     gcc_arch = '--with-fpu=hardfp -marm -mfpu=neon -mfloat-abi=hard'
                elif target.arch == 'x86':
                    gcc_arch = '-m32'
                elif target.arch == 'aarch64':
                    gcc_arch = '-march=armv8-a -target aarch64-linux-gnueabi --static'
                else:
                    gcc_arch = '-m64'

                gcc_isa=""
                if target.target == 'generic-4':
                    gcc_isa = '-msse4.2'
                if target.target == 'generic-8':
                    gcc_isa = '-mavx'

                cc_cmd = "%s -O2 -I. %s %s test_static.cpp -DTEST_SIG=%d %s -o %s" % \
                    (options.compiler_exe, gcc_arch, gcc_isa, match, obj_name, exe_name)

                if platform.system() == 'Darwin':
                    cc_cmd += ' -Wl,-no_pie'
                if should_fail:
                    cc_cmd += " -DEXPECT_FAILURE"

            ispc_cmd = ispc_exe_rel + " --woff %s -o %s --arch=%s --target=%s" % \
                        (filename, obj_name, target.arch, target.target)

            if options.no_opt:
                ispc_cmd += " -O0"
            if target.is_generic():
                ispc_cmd += " --emit-c++ --c++-include-file=%s" % add_prefix(target.include_file, host)

        # compile the ispc code, make the executable, and run it...
        ispc_cmd += " -h " + filename + ".h"
        cc_cmd += " -DTEST_HEADER=<" + filename + ".h>"
        status = run_cmds([ispc_cmd, cc_cmd], options.wrapexe + " " + exe_name,
                          testname, should_fail)

        # clean up after running the test
        try:
            os.unlink(filename + ".h")
            if not options.save_bin:
                if status != Status.Runfail:
                    os.unlink(exe_name)
                    if host.is_windows():
                        basename = os.path.basename(filename)
                        os.unlink("%s.pdb" % basename)
                        os.unlink("%s.ilk" % basename)
                os.unlink(obj_name)
        except:
            None

        return status

# pull tests to run from the given queue and run them.  Multiple copies of
# this function will be running in parallel across all of the CPU cores of
# the system.
def run_tasks_from_queue(queue, queue_ret, total_tests_arg, max_test_length_arg, counter, mutex, glob_var):
    # This is needed on windows because windows doesn't copy globals from parent process while multiprocessing
    host = glob_var[0]
    global options
    options = glob_var[1]
    global s
    s = glob_var[2]
    target = glob_var[3]
    global run_tests_log
    run_tests_log = glob_var[4]

    if host.is_windows():
        tmpdir = "tmp%d" % os.getpid()
        while os.access(tmpdir, os.F_OK):
            tmpdir = "%sx" % tmpdir
        os.mkdir(tmpdir)
        os.chdir(tmpdir)
    else:
        olddir = ""

    for filename in iter(queue.get, 'STOP'):
        status = Status.Skip
        if check_test(filename, host, target):
            try:
                status = run_test(filename, host, target)
            except:
                # This is in case the child has unexpectedly died or some other exception happened
                # Count it as runfail and continue with next test.
                print_debug("ERROR: run_test function raised an exception: %s\n" % (sys.exc_info()[1]), s, run_tests_log)
                status = Status.Runfail

        queue_ret.put((filename, status))
        with mutex:
            update_progress(filename, total_tests_arg, counter, max_test_length_arg)

        # Task done for the test.
        queue.task_done()

    if host.is_windows():
        try:
            os.remove("test_static.obj")
            # vc*.pdb trick is in anticipaton of new versions of VS.
            vcpdb = glob.glob("vc*.pdb")[0]
            os.remove(vcpdb)
            os.chdir("..")
            # This will fail if there were failing tests or
            # Windows is in bad mood.
            os.rmdir(tmpdir)
        except:
            None

    # Task done for terminating `STOP`.
    queue.task_done()


def sigint(signum, frame):
    for t in task_threads:
        t.terminate()
    sys.exit(1)


def file_check(results, host, target):
    global exit_code
    exit_code = 0
    compfails = [fname for fname, status in results if status == Status.Compfail]
    runfails = [fname for fname, status in results if status == Status.Runfail]
    errors = len(compfails) + len(runfails)
    new_compfails = []
    new_runfails = []
    new_passes_compfails = []
    new_passes_runfails = []
# Open file fail_db.txt
    f = open(test_states, 'r')
    f_lines = f.readlines()
    f.close()
# Detect OS
    if platform.system() == 'Windows' or 'CYGWIN_NT' in platform.system():
        OS = "Windows"
    else:
        if platform.system() == 'Darwin':
            OS = "Mac"
        else:
            OS = "Linux"
# Detect opt_set
    if options.no_opt == True:
        opt = "-O0"
    else:
        opt = "-O2"
# Detect LLVM version
    temp1 = common.take_lines(host.ispc_exe + " --version", "first")
    temp2 = re.search('LLVM [0-9]*\.[0-9]*', temp1)
    if temp2 != None:
        llvm_version = temp2.group()
    else:
        llvm_version = "unknown LLVM"
# Detect compiler version
    if OS != "Windows":
        temp1 = common.take_lines(options.compiler_exe + " --version", "first")
        temp2 = re.search("[0-9]*\.[0-9]*\.[0-9]", temp1)
        if temp2 == None:
            temp3 = re.search("[0-9]*\.[0-9]*", temp1)
        else:
            temp3 = re.search("[0-9]*\.[0-9]*", temp2.group())
        compiler_version = options.compiler_exe + temp3.group()
    else:
        compiler_version = "cl"
    possible_compilers=set()
    for x in f_lines:
        if x.startswith("."):
            possible_compilers.add(x.split(' ')[-3])
    #if not compiler_version in possible_compilers:
    #    error("\n**********\nWe don't have history of fails for compiler " +
    #            compiler_version +
    #            "\nAll fails will be new!!!\n**********", 2)
    new_line = " "+target.arch.rjust(6)+" "+target.target.rjust(14)+" "+OS.rjust(7)+" "+llvm_version+" "+compiler_version.rjust(10)+" "+opt+" *\n"
    new_compfails = compfails[:]
    new_runfails = runfails[:]
    new_f_lines = f_lines[:]
    for j in range(0, len(f_lines)):
        if (((" "+target.arch+" ") in f_lines[j]) and
           ((" "+target.target+" ") in f_lines[j]) and
           ((" "+OS+" ") in f_lines[j]) and
           ((" "+llvm_version+" ") in f_lines[j]) and
           ((" "+compiler_version+" ") in f_lines[j]) and
           ((" "+opt+" ") in f_lines[j])):
            if (" compfail " in f_lines[j]):
                f = 0
                for i in range(0, len(compfails)):
                    if compfails[i] in f_lines[j]:
                        new_compfails.remove(compfails[i])
                    else:
                        f = f + 1
                if f == len(compfails):
                    temp3 = f_lines[j].split(" ")
                    new_passes_compfails.append(temp3[0])
                    if options.update == "FP":
                        new_f_lines.remove(f_lines[j])
            if (" runfail " in f_lines[j]):
                f = 0
                for i in range(0, len(runfails)):
                    if runfails[i] in f_lines[j]:
                        new_runfails.remove(runfails[i])
                    else:
                        f = f + 1
                if f == len(runfails):
                    temp3 = f_lines[j].split(" ")
                    new_passes_runfails.append(temp3[0])
                    if options.update == "FP":
                        new_f_lines.remove(f_lines[j])
    if len(new_runfails) != 0:
        new_runfails.sort()
        print_debug("NEW RUNFAILS:\n", s, run_tests_log)
        exit_code = 1
        for i in range (0,len(new_runfails)):
            new_f_lines.append(new_runfails[i] + " runfail " + new_line)
            print_debug("\t" + new_runfails[i] + "\n", s, run_tests_log)
    if len(new_compfails) != 0:
        new_compfails.sort()
        print_debug("NEW COMPFAILS:\n", s, run_tests_log)
        exit_code = 1
        for i in range (0,len(new_compfails)):
            new_f_lines.append(new_compfails[i] + " compfail " + new_line)
            print_debug("\t" + new_compfails[i] + "\n", s, run_tests_log)
    if len(new_runfails) == 0 and len(new_compfails) == 0:
        print_debug("No new fails\n", s, run_tests_log)
    if len(new_passes_runfails) != 0:
        new_passes_runfails.sort()
        print_debug("NEW PASSES after RUNFAILS:\n", s, run_tests_log)
        for i in range (0,len(new_passes_runfails)):
            print_debug("\t" + new_passes_runfails[i] + "\n", s, run_tests_log)
    if len(new_passes_compfails) != 0:
        new_passes_compfails.sort()
        print_debug("NEW PASSES after COMPFAILS:\n", s, run_tests_log)
        for i in range (0,len(new_passes_compfails)):
            print_debug("\t" + new_passes_compfails[i] + "\n", s, run_tests_log)

    if options.update != "":
        output = open(test_states, 'w')
        output.writelines(new_f_lines)
        output.close()
    return [new_runfails, new_compfails, new_passes_runfails, new_passes_compfails, new_line, errors]

# TODO: This function is out of date, it needs update and test coverage.
def verify():
    # Open file fail_db.txt
    f = open(test_states, 'r')
    f_lines = f.readlines()
    f.close()
    check = [["g++", "clang++", "cl"],["-O0", "-O2"],["x86","x86-64"],
             ["Linux","Windows","Mac"],["LLVM 3.2","LLVM 3.3","LLVM 3.4","LLVM 3.5","LLVM 3.6","LLVM trunk"],
             ["sse2-i32x4", "sse2-i32x8",
              "sse4-i32x4", "sse4-i32x8", "sse4-i16x8", "sse4-i8x16",
              "avx1-i32x4", "avx1-i32x8", "avx1-i32x16", "avx1-i64x4",
              "avx2-i32x4", "avx2-i32x8", "avx2-i32x16", "avx2-i64x4",
              "generic-1", "generic-4", "generic-8",
              "generic-16", "generic-32", "generic-64",
              "avx512knl-i32x16", "avx512skx-i32x16", "avx512skx-i32x8"]]
    for i in range (0,len(f_lines)):
        if f_lines[i][0] == "%":
            continue
        for j in range(0,len(check)):
            temp = 0
            for t in range(0,len(check[j])):
                if " " + check[j][t] + " " in f_lines[i]:
                    temp = temp + 1
            if temp != 1:
                print_debug("error in line " + str(i) + "\n", False, run_tests_log)
                break

# populate ex_state test table and run info with testing results
def populate_ex_state(options, target, total_tests, test_result):
    # Detect opt_set
    if options.no_opt == True:
        opt = "-O0"
    else:
        opt = "-O2"

    try:
        common.ex_state.add_to_rinf_testall(total_tests)
        for fname, status in test_result:
            # one-hot encoding
            succ = status == Status.Success
            runf = status == Status.Runfail
            comp = status == Status.Compfail
            skip = status == Status.Skip
            # We do not add skipped tests to test table as we do not know the test result
            if status != Status.Skip:
                common.ex_state.add_to_tt(fname, target.arch, opt, target.target, runf, comp)
            common.ex_state.add_to_rinf(target.arch, opt, target.target, succ, runf, comp, skip)

    except:
        print_debug("Exception in ex_state. Skipping...\n", s, run_tests_log)

# set compiler exe depending on the OS
def set_compiler_exe(host, options):
    if options.compiler_exe == None:
        if host.is_windows():
            options.compiler_exe = "cl.exe"
        else:
            options.compiler_exe = "clang++"
    # checks the required compiler otherwise prints an error message
    check_compiler_exists(options.compiler_exe)

# returns the list of test files
def get_test_files(host, args):
    if len(args) == 0:
        ispc_root = "."
        files = glob.glob(ispc_root + os.sep + "tests" + os.sep + "*ispc") + \
            glob.glob(ispc_root + os.sep + "tests_errors" + os.sep + "*ispc")
    else:
        if host.is_windows():
            argfiles = [ ]
            for f in args:
                # we have to glob ourselves if this is being run under a DOS
                # shell, as it passes wildcard as is.
                argfiles += glob.glob(f)
        else:
            argfiles = args

        files = [ ]
        for f in argfiles:
            if os.path.splitext(f.lower())[1] != ".ispc":
                error("Ignoring file %s, which doesn't have an .ispc extension.\n" % f, 2)
            else:
                files += [ f ]
    return files

# checks the required compiler in PATH otherwise prints an error message
def check_compiler_exists(compiler_exe):
    for path in os.environ["PATH"].split(os.pathsep):
        if os.path.exists(path + os.sep + compiler_exe):
            return
    error("missing the required compiler: %s \n" % compiler_exe, 1)

def print_result(status, results, s, run_tests_log):
    title = StatusStr[status]
    file_list = [fname for fname, fstatus in results if status == fstatus]
    total_tests = len(results)
    print_debug("%d / %d tests %s\n" % (len(file_list), total_tests, title), s, run_tests_log)
    if status == Status.Success:
        return
    for f in sorted(file_list):
        print_debug("\t%s\n" % f, s, run_tests_log)

def run_tests(options1, args, print_version):
    global options
    options = options1
    global s
    s = options.silent

    # prepare run_tests_log and fail_db file
    global run_tests_log
    if options.in_file:
        run_tests_log = os.getcwd() + os.sep + options.in_file
        if print_version == 1:
            common.remove_if_exists(run_tests_log)
    else:
        run_tests_log = ""
    if options.verify:
        verify()
        return 0

    # disable fancy error/warning printing with ANSI colors, so grepping for error
    # messages doesn't get confused
    os.environ["TERM"] = "dumb"

    host = Host(platform.system())
    host.set_ispc_cmd(options.ispc_flags)

    print_debug("Testing ispc: " + host.ispc_exe + "\n", s, run_tests_log)

    target = TargetConfig(options.arch, options.target, options.include_file)

    set_compiler_exe(host, options)

    # print compilers versions
    if print_version > 0:
        common.print_version(host.ispc_exe, "", options.compiler_exe, False, run_tests_log, host.is_windows())

    # if no specific test files are specified, run all of the tests in tests/
    # and tests_errors/
    files = get_test_files(host, args)

    # max_test_length is used to issue exact number of whitespace characters when
    # updating status. Otherwise update causes new lines standard 80 char terminal
    # on both Linux and Windows.
    max_test_length = max([len(f) for f in files])

    # randomly shuffle the tests if asked to do so
    if (options.random):
        random.seed()
        random.shuffle(files)

    # counter
    total_tests = len(files)

    results = []

    nthreads = min([multiprocessing.cpu_count(), options.num_jobs, len(files)])
    print_debug("Running %d jobs in parallel. Running %d tests.\n" % (nthreads, total_tests), s, run_tests_log)

    # put each of the test filenames into a queue
    test_queue = multiprocessing.JoinableQueue()
    for fn in files:
        test_queue.put(fn)
    for x in range(nthreads):
        test_queue.put('STOP')

    # qret is a queue for returned data
    qret = multiprocessing.Queue()

    # need to catch sigint so that we can terminate all of the tasks if
    # we're interrupted
    signal.signal(signal.SIGINT, sigint)

    finished_tests_counter = multiprocessing.Value('i') # 'i' is typecode of ctypes.c_int
    # lock to protect counter increment and stdout printing
    lock = multiprocessing.Lock()

    start_time = time.time()
    # launch jobs to run tests
    glob_var = [host, options, s, target, run_tests_log]
    # task_threads has to be global as it is used in sigint handler
    global task_threads
    task_threads = [0] * nthreads
    for x in range(nthreads):
        task_threads[x] = multiprocessing.Process(target=run_tasks_from_queue, args=(test_queue, qret, total_tests,
                max_test_length, finished_tests_counter, lock, glob_var))
        task_threads[x].start()

    # wait for them all to finish and rid the queue of STOPs
    # join() here just waits for synchronization
    test_queue.join()

    if options.non_interactive == False:
        print_debug("\n", s, run_tests_log)

    temp_time = (time.time() - start_time)
    elapsed_time = time.strftime('%Hh%Mm%Ssec.', time.gmtime(temp_time))

    while not qret.empty():
        results.append(qret.get())

    # populate ex_state test table and run info with testing results
    populate_ex_state(options, target, total_tests, results)

    if options.non_interactive:
        print_debug(" Done %d / %d\n" % (finished_tests_counter.value, total_tests), s, run_tests_log)
    for status in Status:
        print_result(status, results, s, run_tests_log)
    fails = [status != Status.Compfail and status != Status.Runfail for _, status in results]
    if sum(fails) == 0:
        print_debug("No fails\n", s, run_tests_log)

    if len(args) == 0:
        R = file_check(results, host, target)
    else:
        error("don't check new fails for incomplete suite of tests", 2)
        R = 0

    if options.time:
        print_debug("Elapsed time: " + elapsed_time + "\n", s, run_tests_log)

    return [R, elapsed_time]


from optparse import OptionParser
import multiprocessing
import os
import sys
import glob
import re
import signal
import random
import threading
import subprocess
import shlex
import platform
import tempfile
import os.path
import time
# our functions
import common
print_debug = common.print_debug
error = common.error
exit_code = 0
test_states = "fail_db.txt"

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-r", "--random-shuffle", dest="random", help="Randomly order tests",
                  default=False, action="store_true")
    parser.add_option("-g", "--generics-include", dest="include_file", help="Filename for header implementing functions for generics",
                  default=None)
    parser.add_option("-f", "--ispc-flags", dest="ispc_flags", help="Additional flags for ispc (-g, -O1, ...)",
                  default="")
    parser.add_option('-t', '--target', dest='target',
                  help=('Set compilation target. For example: sse4-i32x4, avx2-i32x8, avx512skx-i32x16, etc.'), default="sse4-i32x4")
    parser.add_option('-a', '--arch', dest='arch',
                  help='Set architecture (arm, aarch64, x86, x86-64)',default="x86-64")
    parser.add_option("-c", "--compiler", dest="compiler_exe", help="C/C++ compiler binary to use to run tests",
                  default=None)
    parser.add_option('-o', '--no-opt', dest='no_opt', help='Disable optimization',
                  default=False, action="store_true")
    parser.add_option('-j', '--jobs', dest='num_jobs', help='Maximum number of jobs to run in parallel',
                  default="1024", type="int")
    parser.add_option('-v', '--verbose', dest='verbose', help='Enable verbose output',
                  default=False, action="store_true")
    parser.add_option('--wrap-exe', dest='wrapexe',
                  help='Executable to wrap test runs with (e.g. "valgrind" or "sde -knl -- ")',
                  default="")
    parser.add_option('--time', dest='time', help='Enable time output',
                  default=False, action="store_true")
    parser.add_option('--non-interactive', dest='non_interactive', help='Disable interactive status updates',
                  default=False, action="store_true")
    parser.add_option('-u', "--update-errors", dest='update', help='Update file with fails (F of FP)', default="")
    parser.add_option('-s', "--silent", dest='silent', help='enable silent mode without any output', default=False,
                  action = "store_true")
    parser.add_option("--file", dest='in_file', help='file to save run_tests output', default="")
    parser.add_option("--verify", dest='verify', help='verify the file fail_db.txt', default=False, action="store_true")
    parser.add_option("--save-bin", dest='save_bin', help='compile and create bin, but don\'t execute it',
                  default=False, action="store_true")
    (options, args) = parser.parse_args()

    L = run_tests(options, args, 1)
    exit(exit_code)
