#!/usr/bin/python
#
#  Copyright (c) 2013, Intel Corporation
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

def run_command(cmd):
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

    sp = subprocess.Popen(arg_list, stdin=None,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    out = sp.communicate()
    output = ""
    output += out[0].decode("utf-8")
    output += out[1].decode("utf-8")

    return (sp.returncode, output)

# run the commands in cmd_list
def run_cmds(compile_cmds, run_cmd, filename, expect_failure):
    for cmd in compile_cmds:
        (return_code, output) = run_command(cmd)
        compile_failed = (return_code != 0)
        if compile_failed:
            print_debug("Compilation of test %s failed            \n" % filename, s, run_tests_log)
            if output != "":
                print_debug("%s" % output.encode("utf-8"), s, run_tests_log)
            return (1, 0)

    (return_code, output) = run_command(run_cmd)
    run_failed = (return_code != 0)

    surprise = ((expect_failure and not run_failed) or
                (not expect_failure and run_failed))
    if surprise == True:
        print_debug("Test %s %s (return code %d)            \n" % \
            (filename, "unexpectedly passed" if expect_failure else "failed",
             return_code), s, run_tests_log)
    if output != "":
        print_debug("%s\n" % output.encode("utf-8"), s, run_tests_log)
    if surprise == True:
        return (0, 1)
    else:
        return (0, 0)


def add_prefix(path):
    global is_windows
    if is_windows:
    # On Windows we run tests in tmp dir, so the root is one level up.
        input_prefix = "..\\"
    else:
        input_prefix = ""
    path = input_prefix + path
    path = os.path.abspath(path)
    return path


def check_test(filename):
    prev_arch = False
    prev_os = False
    done_arch = True
    done_os = True
    done = True
    global is_windows
    if is_windows:
        oss = "windows"
    else:
        oss = "linux"
    b = buffer(file(add_prefix(filename)).read());
    for run in re.finditer('// *rule: run on .*', b):
        arch = re.match('.* arch=.*', run.group())
        if arch != None:
            if re.search(' arch='+options.arch+'$', arch.group()) != None:
                prev_arch = True
            if re.search(' arch='+options.arch+' ', arch.group()) != None:
                prev_arch = True
            done_arch = prev_arch
        OS = re.match('.* OS=.*', run.group())
        if OS != None:
            if re.search(' OS='+oss, OS.group()) != None:
                prev_os = True
            done_os = prev_os
    done = done_arch and done_os
    for skip in re.finditer('// *rule: skip on .*', b):
        if re.search(' arch=' + options.arch + '$', skip.group())!=None:
            done = False
        if re.search(' arch=' + options.arch + ' ', skip.group())!=None:
            done = False
        if re.search(' OS=' + oss, skip.group())!=None:
            done = False
    return done


def run_test(testname):
    # testname is a path to the test from the root of ispc dir
    # filename is a path to the test from the current dir
    # ispc_exe_rel is a relative path to ispc
    filename = add_prefix(testname)
    ispc_exe_rel = add_prefix(ispc_exe)

    # is this a test to make sure an error is issued?
    want_error = (filename.find("tests_errors") != -1)
    if want_error == True:
        ispc_cmd = ispc_exe_rel + " --werror --nowrap %s --arch=%s --target=%s" % \
            (filename, options.arch, options.target)
        (return_code, output) = run_command(ispc_cmd)
        got_error = (return_code != 0)

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
            return (1, 0)
        elif got_error == False:
            print_debug("Unexpectedly no errors issued from test %s\n" % testname, s, run_tests_log)
            return (1, 0)
        else:
            return (0, 0)
    else:
        # do we expect this test to fail?
        should_fail = (testname.find("failing_") != -1)

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
            for pattern, ident in list(sig2def.items()):
                if line.find(pattern) != -1:
                    match = ident
                    break
        file.close()
        if match == -1:
            error("unable to find function signature in test %s\n" % testname, 0)
            return (1, 0)
        else:
            global is_generic_target
            if is_windows:
                if is_generic_target:
                    obj_name = "%s.cpp" % os.path.basename(filename)
                else:
                    obj_name = "%s.obj" % os.path.basename(filename)
                exe_name = "%s.exe" % os.path.basename(filename)

                cc_cmd = "%s /I. /I../winstuff /Zi /nologo /DTEST_SIG=%d %s %s /Fe%s" % \
                         (options.compiler_exe, match, add_prefix("test_static.cpp"), obj_name, exe_name)
                if should_fail:
                    cc_cmd += " /DEXPECT_FAILURE"
            else:
                if is_generic_target:
                    obj_name = "%s.cpp" % testname
                else:
                    obj_name = "%s.o" % testname
                exe_name = "%s.run" % testname

                if options.arch == 'arm':
                     gcc_arch = '--with-fpu=hardfp -marm -mfpu=neon -mfloat-abi=hard'
                else:
                    if options.arch == 'x86':
                        gcc_arch = '-m32'
                    else:
                        gcc_arch = '-m64'

                gcc_isa=""
                if options.target == 'generic-4':
                    gcc_isa = '-msse4.2'
                if (options.target == 'generic-8'):
                  if (options.include_file.find("knc-i1x8.h")!=-1 or options.include_file.find("knc-i1x8unsafe_fast.h")!=-1):
                    gcc_isa = '-mmic'
                  else:
                    gcc_isa = '-mavx'
                if (options.target == 'generic-16' or options.target == 'generic-32' or options.target == 'generic-64') \
                        and (options.include_file.find("knc-i1x16.h")!=-1 or options.include_file.find("knc.h")!=-1 or options.include_file.find("knc2x.h")!=-1):
                    gcc_isa = '-mmic'

                cc_cmd = "%s -O2 -I. %s %s test_static.cpp -DTEST_SIG=%d %s -o %s" % \
                         (options.compiler_exe, gcc_arch, gcc_isa, match, obj_name, exe_name)
                if platform.system() == 'Darwin':
                    cc_cmd += ' -Wl,-no_pie'
                if should_fail:
                    cc_cmd += " -DEXPECT_FAILURE"

            ispc_cmd = ispc_exe_rel + " --woff %s -o %s --arch=%s --target=%s" % \
                       (filename, obj_name, options.arch, options.target)
            if options.no_opt:
                ispc_cmd += " -O0" 
            if is_generic_target:
                ispc_cmd += " --emit-c++ --c++-include-file=%s" % add_prefix(options.include_file)

        # compile the ispc code, make the executable, and run it...
        (compile_error, run_error) = run_cmds([ispc_cmd, cc_cmd], 
                                              options.wrapexe + " " + exe_name, \
                                              testname, should_fail)

        # clean up after running the test
        try:
            if not run_error:
                os.unlink(exe_name)
                if is_windows:
                    basename = os.path.basename(filename)
                    os.unlink("%s.pdb" % basename)
                    os.unlink("%s.ilk" % basename)
            os.unlink(obj_name)
        except:
            None

        return (compile_error, run_error)

# pull tests to run from the given queue and run them.  Multiple copies of
# this function will be running in parallel across all of the CPU cores of
# the system.
def run_tasks_from_queue(queue, queue_ret, queue_skip, total_tests_arg, max_test_length_arg, counter, mutex, glob_var):
    # This is needed on windows because windows doen't copy globals from parent process whili multiprocessing
    global is_windows
    is_windows = glob_var[0]
    global options
    options = glob_var[1]
    global s
    s = glob_var[2]
    global ispc_exe
    ispc_exe = glob_var[3]
    global is_generic_target
    is_generic_target = glob_var[4]
    global run_tests_log
    run_tests_log = glob_var[5]    

    if is_windows:
        tmpdir = "tmp%d" % os.getpid()
        os.mkdir(tmpdir)
        os.chdir(tmpdir)
    else:
        olddir = ""
        
    compile_error_files = [ ]
    run_error_files = [ ]
    skip_files = [ ]
    while True:
        filename = queue.get()
        if (filename == 'STOP'):
            queue_ret.put((compile_error_files, run_error_files, skip_files))
            if is_windows:
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
                
            sys.exit(0)

        if check_test(filename):
            (compile_error, run_error) = run_test(filename)
            if compile_error != 0:
                compile_error_files += [ filename ]
            if run_error != 0:
                run_error_files += [ filename ]

            with mutex:
                update_progress(filename, total_tests_arg, counter, max_test_length_arg)
        else:
            skip_files += [ filename ]


def sigint(signum, frame):
    for t in task_threads:
        t.terminate()
    sys.exit(1)


def file_check(compfails, runfails):
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
    temp1 = common.take_lines(ispc_exe + " --version", "first")
    llvm_version = temp1[-10:-2]
# Detect compiler version
    if is_windows == False:
        temp1 = common.take_lines(options.compiler_exe + " --version", "first")
        temp2 = re.search("[0-9]*\.[0-9]*\.[0-9]", temp1)
        if temp2 == None:
            temp3 = re.search("[0-9]*\.[0-9]*", temp1)
        else:
            temp3 = re.search("[0-9]*\.[0-9]*", temp2.group())
        compiler_version = options.compiler_exe + temp3.group()
    else:
        compiler_version = "cl" 
    new_line = " "+options.arch.rjust(6)+" "+options.target.rjust(14)+" "+OS.rjust(7)+" "+llvm_version+" "+compiler_version.rjust(10)+" "+opt+" *\n"

    new_compfails = compfails[:]
    new_runfails = runfails[:]
    new_f_lines = f_lines[:]
    for j in range(0, len(f_lines)):
        if (((" "+options.arch+" ") in f_lines[j]) and
           ((" "+options.target+" ") in f_lines[j]) and
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
        print_debug("NEW RUNFAILS:\n", s, run_tests_log)
        for i in range (0,len(new_runfails)):
            new_f_lines.append(new_runfails[i] + " runfail " + new_line)
            print_debug("\t" + new_runfails[i] + "\n", s, run_tests_log)
    if len(new_compfails) != 0:
        print_debug("NEW COMPFAILS:\n", s, run_tests_log)
        for i in range (0,len(new_compfails)):
            new_f_lines.append(new_compfails[i] + " compfail " + new_line)
            print_debug("\t" + new_compfails[i] + "\n", s, run_tests_log)
    if len(new_runfails) == 0 and len(new_compfails) == 0:
        print_debug("No new fails\n", s, run_tests_log)
    if len(new_passes_runfails) != 0:
        print_debug("NEW PASSES after RUNFAILS:\n", s, run_tests_log)
        for i in range (0,len(new_passes_runfails)):
            print_debug("\t" + new_passes_runfails[i] + "\n", s, run_tests_log)
    if len(new_passes_compfails) != 0:
        print_debug("NEW PASSES after COMPFAILS:\n", s, run_tests_log)
        for i in range (0,len(new_passes_compfails)):
            print_debug("\t" + new_passes_compfails[i] + "\n", s, run_tests_log)
    
    if options.update != "":
        output = open(test_states, 'w')
        output.writelines(new_f_lines)
        output.close()
    return [new_runfails, new_compfails, new_passes_runfails, new_passes_compfails, new_line, errors]

def verify():
    # Open file fail_db.txt
    f = open(test_states, 'r')
    f_lines = f.readlines()
    f.close()
    check = [["g++", "clang++", "cl"],["-O0", "-O2"],["x86","x86-64"],
             ["Linux","Windows","Mac"],["LLVM 3.1","LLVM 3.2","LLVM 3.3","LLVM head"],
             ["sse2-i32x4", "sse2-i32x8", "sse4-i32x4", "sse4-i32x8", "sse4-i16x8",
              "sse4-i8x16", "avx1-i32x8", "avx1-i32x16", "avx1-i64x4", "avx1.1-i32x8", "avx1.1-i32x16",
              "avx2-i32x8", "avx2-i32x16", "generic-1", "generic-4", "generic-8",
              "generic-16", "generic-32", "generic-64"]]
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


def run_tests(options1, args, print_version):
    global options
    options = options1
    global s
    s = options.silent
    
    # prepare run_tests_log and fail_db files
    global run_tests_log
    if options.in_file:
        run_tests_log = os.getcwd() + os.sep + options.in_file
        if print_version == 1:
            common.remove_if_exists(run_tests_log)
    else:
        run_tests_log = ""
    global test_states
    test_states = "fail_db.txt"
    if options.verify:
        verify()
        return 0

    # disable fancy error/warning printing with ANSI colors, so grepping for error
    # messages doesn't get confused
    os.environ["TERM"] = "dumb"
 
    # This script is affected by http://bugs.python.org/issue5261 on OSX 10.5 Leopard
    # git history has a workaround for that issue.
    global is_windows 
    is_windows = (platform.system() == 'Windows' or
                'CYGWIN_NT' in platform.system())
 
    if options.target == 'neon':
        options.arch = 'arm'
 
    # use relative path to not depend on host directory, which may possibly
    # have white spaces and unicode characters.
    global ispc_exe
    ispc_exe = ""
    if not is_windows:
        if os.environ.get("ISPC_HOME") != None:
            if os.path.exists(os.environ["ISPC_HOME"] + os.sep + "ispc"):
                ispc_exe = os.environ["ISPC_HOME"] + os.sep + "ispc"
            else:
                PATH_dir = string.split(os.getenv("PATH"), os.pathsep)
                for counter in PATH_dir:
                    if os.path.exists(counter + os.sep + "ispc"):
                        ispc_exe = counter + os.sep + "ispc"
    else:
        if os.path.exists(".\\Release\\ispc.exe"):
            ispc_exe = ".\\Release\\ispc.exe"
        else:
            error("You don't have ispc.exe compiler in .\\Release.\n", 1)
    # checks the required ispc compiler otherwise prints an error message
    if ispc_exe == "":
        error("ISPC compiler not found.\nAdded path to ispc compiler to your PATH variable or ISPC_HOME variable\n", 1)
    print_debug("Testing ispc: " + ispc_exe + "\n", s, run_tests_log)
    ispc_exe += " " + options.ispc_flags

    global is_generic_target 
    is_generic_target = (options.target.find("generic-") != -1 and
                     options.target != "generic-1" and options.target != "generic-x1")
    if is_generic_target and options.include_file == None:
        if options.target == "generic-4" or options.target == "generic-x4":
            error("No generics #include specified; using examples/intrinsics/sse4.h\n", 2)
            options.include_file = "examples/intrinsics/sse4.h"
            options.target = "generic-4"
        elif options.target == "generic-8" or options.target == "generic-x8":
            error("No generics #include specified and no default available for \"generic-8\" target.\n", 1)
            options.target = "generic-8"
        elif options.target == "generic-16" or options.target == "generic-x16":
            error("No generics #include specified; using examples/intrinsics/generic-16.h\n", 2)
            options.include_file = "examples/intrinsics/generic-16.h"
            options.target = "generic-16"
        elif options.target == "generic-32" or options.target == "generic-x32":
            error("No generics #include specified; using examples/intrinsics/generic-32.h\n", 2)
            options.include_file = "examples/intrinsics/generic-32.h"
            options.target = "generic-32"
        elif options.target == "generic-64" or options.target == "generic-x64":
            error("No generics #include specified; using examples/intrinsics/generic-64.h\n", 2)
            options.include_file = "examples/intrinsics/generic-64.h"
            options.target = "generic-64"
 
    if options.compiler_exe == None:
        if is_windows:
            options.compiler_exe = "cl.exe"
        else:
            options.compiler_exe = "clang++"
 
    # checks the required compiler otherwise prints an error message
    PATH_dir = string.split(os.getenv("PATH"), os.pathsep) 
    compiler_exists = False
 
    for counter in PATH_dir:
        if os.path.exists(counter + os.sep + options.compiler_exe):
            compiler_exists = True
            break
 
    if not compiler_exists:
        error("missing the required compiler: %s \n" % options.compiler_exe, 1)

    # print compilers versions
    if print_version > 0:
        common.print_version(ispc_exe, "", options.compiler_exe, False, run_tests_log, is_windows)
 
    ispc_root = "."
    
    # if no specific test files are specified, run all of the tests in tests/,
    # failing_tests/, and tests_errors/
    if len(args) == 0:
        files = glob.glob(ispc_root + os.sep + "tests" + os.sep + "*ispc") + \
            glob.glob(ispc_root + os.sep + "tests_errors" + os.sep + "*ispc")
    else:
        if is_windows:
            argfiles = [ ]
            for f in args:
                # we have to glob ourselves if this is being run under a DOS
                # shell, as it passes wildcard as is.
                argfiles += glob.glob(f)
        else:
            argfiles = args
        
        files = [ ]
        for f in argfiles:
            if os.path.splitext(string.lower(f))[1] != ".ispc":
                error("Ignoring file %s, which doesn't have an .ispc extension.\n" % f, 2)
            else:
                files += [ f ]
 
    # max_test_length is used to issue exact number of whitespace characters when
    # updating status. Otherwise update causes new lines standard 80 char terminal
    # on both Linux and Windows.
    max_test_length = 0
    for f in files:
        max_test_length = max(max_test_length, len(f))
 
    # randomly shuffle the tests if asked to do so
    if (options.random):
        random.seed()
        random.shuffle(files)
 
    # counter
    total_tests = len(files)

    compile_error_files = [ ]
    run_error_files = [ ]
    skip_files = [ ]

    nthreads = min(multiprocessing.cpu_count(), options.num_jobs)
    nthreads = min(nthreads, len(files))
    print_debug("Running %d jobs in parallel. Running %d tests.\n" % (nthreads, total_tests), s, run_tests_log)

    # put each of the test filenames into a queue
    q = multiprocessing.Queue()
    for fn in files:
        q.put(fn)
    for x in range(nthreads):
        q.put('STOP')
    qret = multiprocessing.Queue()
    qskip = multiprocessing.Queue()

    # need to catch sigint so that we can terminate all of the tasks if
    # we're interrupted
    signal.signal(signal.SIGINT, sigint)

    finished_tests_counter = multiprocessing.Value(c_int)
    finished_tests_counter_lock = multiprocessing.Lock()

    start_time = time.time()
    # launch jobs to run tests
    glob_var = [is_windows, options, s, ispc_exe, is_generic_target, run_tests_log]
    global task_threads
    task_threads = [0] * nthreads
    for x in range(nthreads):
        task_threads[x] = multiprocessing.Process(target=run_tasks_from_queue, args=(q, qret, qskip, total_tests,
            max_test_length, finished_tests_counter, finished_tests_counter_lock, glob_var))
        task_threads[x].start()
    # wait for them to all finish and then return the number that failed
    # (i.e. return 0 if all is ok)
    for t in task_threads:
        t.join()
    if options.non_interactive == False:
        print_debug("\n", s, run_tests_log)

    elapsed_time = time.time() - start_time

    while not qret.empty():
        (c, r, skip) = qret.get()
        compile_error_files += c
        run_error_files += r
        skip_files += skip

    if options.non_interactive:
        print_debug(" Done %d / %d\n" % (finished_tests_counter.value, total_tests), s, run_tests_log)
    if len(skip_files) > 0:
        skip_files.sort()
        print_debug("%d / %d tests SKIPPED:\n" % (len(skip_files), total_tests), s, run_tests_log)
        for f in skip_files:
            print_debug("\t%s\n" % f, s, run_tests_log)
    if len(compile_error_files) > 0:
        compile_error_files.sort()
        print_debug("%d / %d tests FAILED compilation:\n" % (len(compile_error_files), total_tests), s, run_tests_log)
        for f in compile_error_files:
            print_debug("\t%s\n" % f, s, run_tests_log)
    if len(run_error_files) > 0:
        run_error_files.sort()
        print_debug("%d / %d tests FAILED execution:\n" % (len(run_error_files), total_tests), s, run_tests_log)
        for f in run_error_files:
            print_debug("\t%s\n" % f, s, run_tests_log)
    if len(compile_error_files) == 0 and len(run_error_files) == 0:
        print_debug("No fails\n", s, run_tests_log)

    R = file_check(compile_error_files, run_error_files)

    if options.time:
        print_debug("Elapsed time: %d s\n" % elapsed_time, s, run_tests_log)

    return R


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

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-r", "--random-shuffle", dest="random", help="Randomly order tests",
                  default=False, action="store_true")
    parser.add_option("-g", "--generics-include", dest="include_file", help="Filename for header implementing functions for generics",
                  default=None)
    parser.add_option("-f", "--ispc-flags", dest="ispc_flags", help="Additional flags for ispc (-g, -O1, ...)",
                  default="")
    parser.add_option('-t', '--target', dest='target',
                  help='Set compilation target (sse2-i32x4, sse2-i32x8, sse4-i32x4, sse4-i32x8, sse4-i16x8, sse4-i8x16, avx1-i32x8, avx1-i32x16, avx1.1-i32x8, avx1.1-i32x16, avx2-i32x8, avx2-i32x16, generic-x1, generic-x4, generic-x8, generic-x16, generic-x32, generic-x64)',
                                    default="sse4")
    parser.add_option('-a', '--arch', dest='arch',
                  help='Set architecture (arm, x86, x86-64)',
                                    default="x86-64")
    parser.add_option("-c", "--compiler", dest="compiler_exe", help="C/C++ compiler binary to use to run tests",
                  default=None)
    parser.add_option('-o', '--no-opt', dest='no_opt', help='Disable optimization',
                  default=False, action="store_true")
    parser.add_option('-j', '--jobs', dest='num_jobs', help='Maximum number of jobs to run in parallel',
                  default="1024", type="int")
    parser.add_option('-v', '--verbose', dest='verbose', help='Enable verbose output',
                  default=False, action="store_true")
    parser.add_option('--wrap-exe', dest='wrapexe',
                  help='Executable to wrap test runs with (e.g. "valgrind")',
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
    (options, args) = parser.parse_args()
    L = run_tests(options, args, 1)
    exit(0)
