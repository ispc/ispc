#!/usr/bin/env python3
#
#  Copyright (c) 2013-2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

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
            error("ISPC compiler not found.\nAdd path to ispc compiler to your PATH or ISPC_HOME env variable\n", 1)
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
    def __init__(self, arch, target, cpu):
        if arch == "x86_64" or arch == "x86-64":
            self.arch = "x86-64"
        else:
            self.arch = arch
        self.target = target
        self.xe = target.find("gen9") != -1 or target.find("xe") != -1
        self.set_cpu(cpu)
        self.set_target()

    def is_xe(self):
        return self.xe

    def set_cpu(self, cpu):
        if cpu is not None:
            self.cpu = cpu
            # Alias all of acm-* devices to dg2.
            if cpu.startswith("acm-"):
                self.cpu = "dg2"
            # Alias all of mtl-* devices to mtl.
            if cpu.startswith("mtl-"):
                self.cpu = "mtl"
        else:
            self.cpu = "unspec"

    # set arch/target
    def set_target(self):
        if self.target == 'neon':
            self.arch = 'aarch64'

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

def canonicalize_filename(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]

    # Replace invalid Python identifier characters with underscores
    # Valid Python identifiers can only contain letters, digits, and underscores
    # and cannot start with a digit
    canonicalized = re.sub(r'[^a-zA-Z0-9_]', '_', basename)

    if canonicalized and canonicalized[0].isdigit():
        canonicalized = '_' + canonicalized

    if not canonicalized:
        canonicalized = '_'

    return canonicalized

def call_test_function(module_name, test_sig, func_sig, width, verbose=False):
    """
    Call the test function in a subprocess for better isolation.

    Args:
        module_name: Name of the compiled module
        test_sig: Test signature identifier
        func_sig: Function signature string
        width: Vector width for the test
        verbose: Enable verbose output

    Returns:
        Status enum value indicating test result
    """

    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_runner_path = os.path.join(script_dir, 'nanobind_runner.py')

    # Prepare command arguments using the same Python interpreter
    cmd = [
        sys.executable,
        test_runner_path,
        module_name,
        str(test_sig),
        func_sig,
        str(width)
    ]

    if verbose:
        cmd.append('true')

    try:
        # Run the test in a subprocess with a timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()
        )

        # Print any output from the subprocess if verbose
        if verbose and result.stdout:
            print_debug(f"Test output: {result.stdout}", s, run_tests_log)
        if verbose and result.stderr:
            print_debug(f"Test stderr: {result.stderr}", s, run_tests_log)

        # Convert exit code back to Status enum
        if result.returncode == 0:
            return Status.Success
        else:
            return Status.Runfail

    except Exception as e:
        print_debug(f"Unexpected error running test {module_name}: {e}", s, run_tests_log)
        return Status.Runfail

def build_ispc_extension(module_name, ispc_object, nb_wrapper, header, test_sig, width):
    """
    Build the ISPC extension module using setuptools and nanobind.

    Usually, nanobind is built via CMake-based rules provided by the nanobind
    project. To avoid the extra hassle of doing that, we use a different
    approach. There is an nb_combined.cpp file in nanobind that contains
    basically everything needed in one place. It also contains the general
    rules to build it."
    """
    import os
    import sys
    import sysconfig
    import tempfile
    import shutil
    import nanobind
    from setuptools import setup, Extension
    from setuptools.dist import Distribution
    from setuptools.command.build_ext import build_ext

    # Get nanobind specific paths
    nanobind_dir = os.path.dirname(nanobind.__file__)
    nanobind_src_dir = os.path.join(nanobind_dir, "src")
    robin_map_include = os.path.join(nanobind_dir, "ext/robin_map/include")
    nanobind_sources = [
        os.path.join(nanobind_src_dir, "nb_combined.cpp"),
    ]

    # Create temporary directory for build files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define an extension module
        ext_modules = [
            Extension(
                module_name,
                [nb_wrapper, 'tests/test_static.cpp'] + nanobind_sources,
                include_dirs=[
                    nanobind.include_dir(),
                    robin_map_include,
                    'tests',
                ],
                # Link against the ISPC object file that is built by ISPC
                extra_objects=[ispc_object],
                language='c++',
                extra_compile_args=[
                    # Compiler flags required for nb_combined.cpp
                    "-std=c++17",
                    "-fvisibility=hidden",
                    "-O3",
                    "-fno-strict-aliasing",
                    "-ffunction-sections",
                    "-fdata-sections",
                    "-fPIC",
                    # Flags are required to build test_static.cpp
                    f"-DTEST_SIG={test_sig}",
                    f"-DTEST_WIDTH={width}",
                    f"-DTEST_HEADER=\"{header}\"",
                    "-w",
                ],
                # Macros required for nanobind
                define_macros=[
                    ("NDEBUG", None),
                    ("NB_COMPACT_ASSERTIONS", None),
                ],
            ),
        ]

        # Create distribution and build command directly
        dist = Distribution({
            'name': module_name,
            'ext_modules': ext_modules,
            'zip_safe': False,
        })

        # Create and configure build_ext command
        build_ext_cmd = build_ext(dist)
        build_ext_cmd.inplace = True
        build_ext_cmd.build_temp = temp_dir
        build_ext_cmd.finalize_options()

        if not options.verbose:
            # Suppress output
            import contextlib
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    build_ext_cmd.run()
        else:
            # Actually run the build commands
            build_ext_cmd.run()

        # Get the platform-specific extension suffix to figure out the soname
        # of the built extension
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        if ext_suffix is None:
            ext_suffix = sysconfig.get_config_var('SO')
            if ext_suffix is None:
                ext_suffix = '.so'

        so_filename = f"{module_name}{ext_suffix}"

        # Verify the built extension exists
        if not os.path.exists(so_filename):
            raise RuntimeError(f"Extension {so_filename} was not created successfully")

        return so_filename

def run_command(cmd, timeout=600, cwd="."):
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
        proc = subprocess.Popen(arg_list, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
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
    output = Output(out[0].decode("utf-8"), out[1].decode("utf-8"))
    return (proc.returncode, output, is_timeout)

# checks whether print ouput is correct
# (whether test and reference outputs are same)
# NOTE: output contains both test and reference lines
def check_print_output(output):
    lines = output.splitlines()
    if len(lines) == 0 or len(lines) % 2:
        return False
    else:
        return lines[0:len(lines)//2] == lines[len(lines)//2:len(lines)]

# run the commands in cmd_list
def run_cmds(compile_cmds, run_cmd, filename, expect_failure, sig, exe_wd="."):
    for cmd in compile_cmds:
        (return_code, output, timeout) = run_command(cmd, options.test_time)
        compile_failed = (return_code != 0)
        if compile_failed:
            print_debug("Compilation of test %s failed %s           \n" % (filename, "due to TIMEOUT" if timeout else ""), s, run_tests_log)
            if output != "":
                print_debug("%s" % output, s, run_tests_log)
            return Status.Compfail

    if not options.save_bin:
        (return_code, output, timeout) = run_command(run_cmd, options.test_time, cwd=exe_wd)
        if sig < 32:
            run_failed = (return_code != 0) or timeout
        else:
            # check only stdout
            output_equality = check_print_output(output.stdout)
            if not output_equality:
                print_debug("Print outputs check failed\n", s, run_tests_log)
            run_failed = (return_code != 0) or not output_equality or timeout

    else:
        run_failed = 0

    surprise = ((expect_failure and not run_failed) or
                (not expect_failure and run_failed))
    if surprise == True:
        print_debug("Test %s %s (return code %d)            \n" % \
            (filename, "unexpectedly passed" if expect_failure else "failed",
             return_code), s, run_tests_log)
    if str(output):
        print_debug("%s\n" % output, s, run_tests_log)
    if surprise == True:
        return Status.Runfail
    else:
        return Status.Success


def add_prefix(path, host, target):
    # In JIT mode, we don't use temp directories, so no prefix needed
    if hasattr(options, 'jit_mode') and options.jit_mode:
        input_prefix = ""
    elif host.is_windows():
    # On Windows we run tests in tmp dir, so the root is one level up.
        input_prefix = "..\\"
    else:
        # For Xe target we run tests in tmp dir since output file has
        # the same name for all tests, so the root is one level up
        if target.is_xe():
            input_prefix = "../"
        else:
            input_prefix = ""
    path = input_prefix + path
    path = os.path.abspath(path)
    return path

# Return True if test should be skipped,
# return False otherwise.
#
# Default rules are to run always:
# //rule: run on <key>=*
#
# Rules can be overrriden by putting
# comments to test file:
# //rule: run on <key>=<value>
# //rule: skip on <key>=<value>
#
# Currently supported keys are:
# [arch, OS, cpu, target, jit].
#
# * (asterisk) represent any value.
# Proper regexps can also be used.
#
# Rules order is important,
# rule may override all previous rules.
#
# Examples:
#
# 1. Run only on arch xe64:
# // rule: skip on arch=*
# // rule: run on arch=xe64
#
# 2. Run only on Linux OS:
# // rule: skip on OS=*
# // rule: run on OS=linux
#
# 3. Skip only generic target for windows:
# // rule: skip on target=generic.*
# // rule: run on OS=!windows
#
def check_if_skip_test(filename, host, target):
    # by default we're not skipping test
    skip = False

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

    target_cpu = target.cpu

    # Determine JIT mode status
    jit_mode = "true" if options.jit_mode else "false"

    rule_values = {"arch": target.arch, "OS": oss, "cpu": target_cpu, "target": target.target, "jit": jit_mode}

    test_file_path = add_prefix(filename, host, target)
    with open(test_file_path) as test_file:
        # scan test file line by line
        while True:
            test_line = test_file.readline()
            # EOF
            if not test_line:
                break;
            rule = re.search('// *rule: (run|skip) on (arch|OS|cpu|target|jit)=(.*)', test_line)
            # no match for this line -> look at next line
            if rule == None:
                if "rule:" in test_line:
                    print_debug("%s: Warning: Unrecognized rule: %s\n" % (filename, test_line), s, run_tests_log)
                continue

            rule_action = rule.group(1)
            rule_key = rule.group(2)
            rule_value = rule.group(3)

            # support negation of rules
            negate = False
            if rule_value.startswith("!"):
                rule_value = rule_value[1:]
                negate = True

            # extend rule_value to a proper regexp
            if rule_value == "*":
                rule_value = ".*"

            # rule_value can be a regexp that can match the actual value
            # check here if it matches
            val = rule_values[rule_key]
            match = re.fullmatch(rule_value, val)

            # Apply negation logic correctly
            if negate:
                match = not match

            if match:
                if rule_action == "run":
                    skip = False
                elif rule_action == "skip":
                    skip = True

    return skip


def run_test(testname, host, target, jit_lib_path=None):
    # testname is a path to the test from the root of ispc dir
    # filename is a path to the test from the current dir
    # ispc_exe_rel is a relative path to ispc
    filename = add_prefix(testname, host, target)

    # Get global path to tests. Note, that test is in tests/func-tests/ directory.
    test_dir = os.path.dirname(os.path.dirname(filename))

    # Debug check is now supported only for xe
    if options.debug_check and target.is_xe():
        ispc_exe_rel = add_prefix(host.ispc_cmd + " -g", host, target)
    else:
        ispc_exe_rel = add_prefix(host.ispc_cmd, host, target)

    # is this a test to make sure an error is issued?
    want_error = (filename.find("tests_errors") != -1)
    if want_error == True:
        ispc_cmd = ispc_exe_rel + " --werror --nowrap %s --arch=%s --target=%s" % \
            (filename, target.arch, target.target)
        (return_code, output, timeout) = run_command(ispc_cmd, options.test_time)
        got_error = (return_code != 0) or timeout

        # figure out the error message we're expecting
        file = open(filename, 'r')
        firstline = file.readline()
        firstline = firstline.replace("//", "")
        firstline = firstline.lstrip()
        firstline = firstline.rstrip()
        file.close()

        if re.search(firstline, output.__str__()) == None:
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
                    "f_du(" : 4, "f_duf(" : 5, "f_di(" : 6, "f_sz" : 7,
                    "f_t(" : 8, "print_uf(" : 32, "print_f(" : 33,
                    "print_fuf(" : 34, "print_no(" : 35 }
        def2sig = { k: v for v, k in sig2def.items() }
        file = open(filename, 'r')
        match = -1
        for line in file:
            # look for lines with 'export'...
            if line.find("task") == -1 and line.find("export") == -1:
                continue
            # one of them should have a function with one of the
            # declarations in sig2def
            for pattern, ident in list(sig2def.items()):
                if line.find(pattern) != -1:
                    match = ident
                    break
        file.close()

        # Figure out target width
        width = -1
        target_match = re.match('.*-(i[0-9]*)?x([0-9]*)', options.target)
        # If target does not contain width in a standard way:
        if target_match == None:
            error("Unable to detect the target width for target %s\nOnly canonical form of the target names is supported, deprecated forms are not supported" % options.target, 0)
            return Status.Compfail
        width = int(target_match.group(2))

        if match == -1:
            error("Unable to find function signature in test %s\n" % testname, 0)
            return Status.Compfail
        else:
            xe_target = options.target
            if host.is_windows():
                if target.is_xe():
                    obj_name = "test_xe.bin" if options.ispc_output == "ze" else "test_xe.spv"
                else:
                    obj_name = "%s.obj" % os.path.basename(filename)

                if target.arch == "wasm32" or target.arch == "wasm64":
                    exe_name = "%s.js" % os.path.realpath(filename)
                else:
                    exe_name = "%s.exe" % os.path.basename(filename)

                test_file = "tests\\test_static_jit.cpp" if options.jit_mode else "tests\\test_static.cpp"
                if options.jit_mode:
                    # For JIT mode, we need to link against the ISPC library and don't use object files
                    include_path = "src\\include"
                    # Use unique object file name to avoid conflicts in parallel execution
                    obj_file = "%s.obj" % os.path.basename(filename)
                    cc_cmd = "%s /Itests /I%s /nologo /DTEST_SIG=%d /DTEST_WIDTH=%d %s ispc.lib /Fo%s /Fe%s /link /LIBPATH:%s" % \
                             (options.compiler_exe, include_path, match, width, add_prefix(test_file, host, target), obj_file, exe_name, jit_lib_path)
                else:
                    cc_cmd = "%s /Itests /Zi /nologo /DTEST_SIG=%d /DTEST_WIDTH=%d %s %s /Fe%s" % \
                             (options.compiler_exe, match, width, add_prefix(test_file, host, target), obj_name, exe_name)
                # Increase the stack size for Windows up to 8MB because some
                # tests for -O0/x86 can generate quite large stack frames.
                cc_cmd += " /F8388608"
                if target.is_xe():
                    l0_test_file = "tests\\test_static_l0.cpp"
                    cc_cmd = "%s /Itests /I%s\\include /nologo /DTEST_SIG=%d /DTEST_WIDTH=%d %s %s /Fe%s ze_loader.lib /link /LIBPATH:%s\\lib" % \
                         (options.compiler_exe, options.l0loader, match, width, " /DTEST_ZEBIN" if options.ispc_output == "ze" else " /DTEST_SPV", \
                         add_prefix(l0_test_file, host, target), exe_name, options.l0loader)
                if options.calling_conv == "vectorcall":
                    cc_cmd += " /DVECTORCALL_CONV"
                if should_fail:
                    cc_cmd += " /DEXPECT_FAILURE"
            else:
                if target.is_xe():
                    obj_name = "test_xe.bin" if options.ispc_output == "ze" else "test_xe.spv"
                else:
                    obj_name = "%s.o" % testname

                if target.arch == "wasm32" or target.arch == "wasm64":
                    exe_name = "%s.js" % os.path.realpath(testname)
                else:
                    exe_name = "%s.run" % testname

                if target.arch == 'arm':
                    gcc_arch = '--with-fpu=hardfp -marm -mfpu=neon -mfloat-abi=hard'
                elif target.arch == 'x86' or target.arch == "wasm32":
                    gcc_arch = '-m32'
                elif target.arch == 'aarch64':
                    gcc_arch = '-march=armv8-a'
                elif target.arch == 'riscv64':
                    gcc_arch = '-march=rv64gcv'
                    if options.wrapexe and options.wrapexe.startswith('qemu'):
                        gcc_arch += ' -static'
                elif target.arch == 'wasm64':
                    gcc_arch = '-sMEMORY64'
                else:
                    gcc_arch = '-m64'

                test_file = "tests/test_static_jit.cpp" if options.jit_mode else "tests/test_static.cpp"
                if options.jit_mode:
                    # For JIT mode, we need to link against the ISPC library and don't use object files
                    cc_cmd = "%s -O2 -I tests/ -I src/include %s %s -DTEST_SIG=%d -DTEST_WIDTH=%d -lispc -L%s -Wl,-rpath,%s -o %s" % \
                        (options.compiler_exe, gcc_arch, test_file, match, width, jit_lib_path, jit_lib_path, exe_name)
                else:
                    cc_cmd = "%s -O2 -I tests/ %s %s -DTEST_SIG=%d -DTEST_WIDTH=%d %s -o %s" % \
                        (options.compiler_exe, gcc_arch, test_file, match, width, obj_name, exe_name)

                # Produce position independent code for both c++ and ispc compilations.
                # The motivation for this is that Clang 15 changed default
                # from "-mrelocation-model static" to "-mrelocation-model pic", so
                # we enable PIC compilation to have it consistently regardless compiler version.
                cc_cmd += ' -fPIE'
                if should_fail:
                    cc_cmd += " -DEXPECT_FAILURE"

                if target.is_xe():
                    exe_name = "%s.run" % os.path.basename(testname)
                    l0_test_file = "tests/test_static_l0.cpp"
                    cc_cmd = "%s -O0 -I tests -I %s/include -lze_loader -L %s/lib \
                            %s %s -DTEST_SIG=%d -DTEST_WIDTH=%d -o %s" % \
                            (options.compiler_exe, options.l0loader, options.l0loader, gcc_arch, add_prefix(l0_test_file, host, target),
                             match, width, exe_name)
                    exe_name = "./" + exe_name
                    cc_cmd += " -DTEST_ZEBIN" if options.ispc_output == "ze" else " -DTEST_SPV"
            if options.jit_mode:
                # In JIT mode, we don't need to pre-compile ISPC files as they're compiled at runtime
                # We just need to ensure the test file is available
                ispc_cmd = "echo JIT mode: ISPC file will be compiled at runtime"
            else:
                ispc_cmd = ispc_exe_rel + " -I %s --pic --woff %s -o %s --arch=%s --target=%s -DTEST_SIG=%d" % \
                            (test_dir, filename, obj_name, options.arch, xe_target if target.is_xe() else options.target, match)

            if target.is_xe():
                ispc_cmd += " --emit-zebin" if options.ispc_output == "ze" else " --emit-spirv"
                ispc_cmd += " -DISPC_GPU"
            if options.device != None:
                ispc_cmd += " --device="+ options.device

            if options.opt == 'O0':
                ispc_cmd += " -O0"
            elif options.opt == 'O1':
                ispc_cmd += " -O1"
            elif options.opt == 'O2':
                ispc_cmd += " -O2"

            if options.calling_conv == "vectorcall" and host.is_windows():
                ispc_cmd += " --vectorcall"

            # we enabled float16 tests which requires this flag on windows
            if host.is_windows():
                ispc_cmd += " --include-float16-conversions"

        exe_wd = "."
        if target.arch == "wasm32" or target.arch == "wasm64":
            cc_cmd += " -D__WASM__"
            options.wrapexe = os.environ["EMSDK_NODE"]
            if target.arch == "wasm64":
                options.wrapexe += " --experimental-wasm-memory64"
            exe_wd = os.path.realpath("./tests")
        # compile the ispc code, make the executable, and run it...
        if not options.jit_mode:
            header = f"{filename}.h"
            ispc_cmd += " -h " + header
            cc_cmd += f" -DTEST_HEADER=\"<{header}>\""

        if options.nanobind:
            module_name = canonicalize_filename(filename)
            nb_wrapper = module_name + ".cpp"
            ispc_cmd += f" --nanobind-wrapper={nb_wrapper}"

            # Run ISPC to generate obj_name and nb_wrapper
            return_code, output, timeout = run_command(ispc_cmd, options.test_time)
            if return_code != 0:
                print_debug("Compilation of test %s failed %s           \n" % (filename, "due to TIMEOUT" if timeout else ""), s, run_tests_log)
                if output != "":
                    print_debug("%s" % output, s, run_tests_log)
                return Status.Compfail

            # Compile shared library extension to be used as Python module
            ext_soname = build_ispc_extension(module_name, obj_name, nb_wrapper, header, match, width)

            # Import the freshly built module and call the test function from it
            status = call_test_function(module_name, match, def2sig[match], width, options.verbose)
        else:
            if options.jit_mode:
                # In JIT mode, pass the ISPC file and target to the test executable
                run_cmd = options.wrapexe + " " + exe_name + " " + filename + " " + (xe_target if target.is_xe() else options.target)
                status = run_cmds([ispc_cmd, cc_cmd], run_cmd, testname, should_fail, match, exe_wd=exe_wd)
            else:
                status = run_cmds([ispc_cmd, cc_cmd], options.wrapexe + " " + exe_name,
                                  testname, should_fail, match, exe_wd=exe_wd)

        # clean up after running the test
        try:
            if not options.jit_mode:
                os.unlink(header)
            if options.nanobind:
                os.unlink(nb_wrapper)
                if not options.save_bin:
                    os.unlink(ext_soname)
            if not options.save_bin:
                if status != Status.Runfail:
                    os.unlink(exe_name)
                    if host.is_windows():
                        basename = os.path.basename(filename)
                        os.unlink("%s.pdb" % basename)
                        os.unlink("%s.ilk" % basename)
                if not options.jit_mode:
                    os.unlink(obj_name)
                os.unlink(filename + ".wasm")
                os.unlink(filename + ".js")
                os.unlink(filename + ".html")
        except:
            None

        # Clean up JIT mode object file on Windows (outside try/except to avoid silent failures)
        if options.jit_mode and host.is_windows() and not options.save_bin:
            try:
                obj_file = "%s.obj" % os.path.basename(filename)
                if os.path.exists(obj_file):
                    os.unlink(obj_file)
            except Exception as e:
                print(f"Warning: Failed to clean up JIT object file {obj_file}: {e}")

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
    jit_lib_path = glob_var[5]

    if (host.is_windows() or target.is_xe()) and not options.jit_mode:
        tmpdir = "tmp%d" % os.getpid()
        while os.access(tmpdir, os.F_OK):
            tmpdir = "%sx" % tmpdir
        os.mkdir(tmpdir)
        os.chdir(tmpdir)
    else:
        olddir = ""

    for filename in iter(queue.get, 'STOP'):
        status = Status.Skip
        if not check_if_skip_test(filename, host, target):
            try:
                status = run_test(filename, host, target, jit_lib_path)
            except:
                # This is in case the child has unexpectedly died or some other exception happened
                # Count it as runfail and continue with next test.
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, file=sys.stderr)
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
    else:
        if target.is_xe():
            try:
                os.chdir("..")
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
# Open fail db file
    f = open(options.fail_db, 'r')
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
    opt = options.opt
# Detect testing output
    ispc_output = options.ispc_output
# Detect LLVM version
    temp1 = common.take_lines(host.ispc_exe + " --version", "first")
    temp2 = re.search(r'LLVM [0-9]*\.[0-9]*', temp1)
    if temp2 != None:
        llvm_version = temp2.group()
    else:
        llvm_version = "unknown LLVM"
# Detect compiler version
    if OS != "Windows":
        temp1 = common.take_lines(options.compiler_exe + " --version", "first")
        temp2 = re.search(r"[0-9]*\.[0-9]*\.[0-9]", temp1)
        if temp2 == None:
            temp3 = re.search(r"[0-9]*\.[0-9]*", temp1)
        else:
            temp3 = re.search(r"[0-9]*\.[0-9]*", temp2.group())
        compiler_version = options.compiler_exe + temp3.group()
    else:
        compiler_version = "cl"
    cpu = target.cpu
    possible_compilers=set()
    for x in f_lines:
        if x.startswith("."):
            possible_compilers.add(x.split(' ')[-3])
    #if not compiler_version in possible_compilers:
    #    error("\n**********\nWe don't have history of fails for compiler " +
    #            compiler_version +
    #            "\nAll fails will be new!!!\n**********", 2)
    new_line = " "+target.arch.rjust(6)+" "+target.target.rjust(14)+" "+cpu+" "+OS.rjust(7)+" "+llvm_version+" "+compiler_version.rjust(10)+" "+opt+ " " + ispc_output + " *\n"
    new_compfails = compfails[:]
    new_runfails = runfails[:]
    new_f_lines = f_lines[:]
    for j in range(0, len(f_lines)):
        if (((" "+target.arch+" ") in f_lines[j]) and
           ((" "+target.target+" ") in f_lines[j]) and
           ((" "+cpu+" ") in f_lines[j]) and
           ((" "+OS+" ") in f_lines[j]) and
           ((" "+llvm_version+" ") in f_lines[j]) and
           ((" "+compiler_version+" ") in f_lines[j]) and
           ((" "+opt+" ") in f_lines[j]) and
           ((" "+ispc_output+" ") in f_lines[j])):
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
        output = open(options.fail_db, 'w')
        output.writelines(new_f_lines)
        output.close()
    return [new_runfails, new_compfails, new_passes_runfails, new_passes_compfails, new_line, errors]

# TODO: This function is out of date, it needs update and test coverage.
def verify():
    # Open fail db file
    f = open(options.fail_db, 'r')
    f_lines = f.readlines()
    f.close()
    check = [["g++", "clang++", "cl"],["-O0", "-O2"],["x86","x86-64"],
             ["Linux","Windows","Mac"],["LLVM 3.2","LLVM 3.3","LLVM 3.4","LLVM 3.5","LLVM 3.6","LLVM trunk"],
             ["sse2-i32x4", "sse2-i32x8",
              "sse4-i32x4", "sse4-i32x8", "sse4-i16x8", "sse4-i8x16",
              "sse4.1-i32x4", "sse4.1-i32x8", "sse4.1-i16x8", "sse4.1-i8x16",
              "avx1-i32x4", "avx1-i32x8", "avx1-i32x16", "avx1-i64x4",
              "avx2-i32x4", "avx2-i32x8", "avx2-i32x16", "avx2-i64x4",
              "avx512skx-x16", "avx512skx-x8", "avx512skx-x4", "avx512skx-x64", "avx512skx-x32"]]
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
    opt = options.opt

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
        if options.arch == "wasm32" or options.arch == "wasm64":
          options.compiler_exe = "emcc"
        elif host.is_windows():
            options.compiler_exe = "cl.exe"
        else:
            options.compiler_exe = "clang++"
    # checks the required compiler otherwise prints an error message
    check_compiler_exists(options.compiler_exe)

# set ISPC output format
def set_ispc_output(target, options):
    if options.ispc_output == None:
        if target.is_xe():
            options.ispc_output = "spv"
        else:
            options.ispc_output = "obj"
    else:
        if not target.is_xe() and not options.ispc_output=="obj" or target.is_xe() and options.ispc_output=="obj":
            error("unsupported test output \"%s\" is specified for target: %s \n" % (options.ispc_output, target.target), 1)

# returns the list of test files
def get_test_files(host, args):
    if len(args) == 0:
        ispc_root = "."
        test_files = f"{ispc_root}{os.sep}tests{os.sep}func-tests{os.sep}*ispc"
        files = glob.glob(test_files)
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

# determine the ISPC library path for JIT mode
def get_jit_library_path(host, options):
    if options.jit_lib_path:
        lib_path = options.jit_lib_path
    else:
        # Auto-discover library path
        ispc_dir = os.path.dirname(os.path.abspath(host.ispc_exe))

        if host.is_windows():
            # On Windows: ispc.exe is in build/bin/Release, ispc.lib is in build/Release
            # So from build/bin/Release, go up to build, then check Release/Debug
            build_dir = os.path.dirname(os.path.dirname(ispc_dir))  # build/bin/Release -> build
            release_lib_path = os.path.join(build_dir, "Release")
            debug_lib_path = os.path.join(build_dir, "Debug")
            if os.path.exists(release_lib_path):
                lib_path = release_lib_path
            elif os.path.exists(debug_lib_path):
                lib_path = debug_lib_path
            else:
                # Fallback to build/lib
                lib_path = os.path.join(build_dir, "lib")
        else:
            # For all other platforms: <path_to_ispc>/../lib
            lib_path = os.path.join(os.path.dirname(ispc_dir), "lib")

    # Validate library path exists
    if not os.path.exists(lib_path):
        error("ISPC library directory not found: %s\n" % lib_path, 1)

    # Check for library file existence
    if host.is_windows():
        # Check for import library (for linking)
        lib_file = os.path.join(lib_path, "ispc.lib")
        if not os.path.exists(lib_file):
            error("ISPC import library not found: %s\n" % lib_file, 1)

        # Also check that DLL exists in bin directory (for runtime)
        ispc_dir = os.path.dirname(os.path.abspath(host.ispc_exe))
        dll_file = os.path.join(ispc_dir, "ispc.dll")
        if not os.path.exists(dll_file):
            error("ISPC runtime library not found: %s\n" % dll_file, 1)
    else:
        # On Unix systems, check for both static and shared libraries
        macos_lib = os.path.join(lib_path, "libispc.dylib")
        linux_lib = os.path.join(lib_path, "libispc.so")
        if not os.path.exists(macos_lib) and not os.path.exists(linux_lib):
            error("ISPC library file not found. Checked:\n  %s\n  %s\n" % (macos_lib, linux_lib), 1)

    return lib_path

def print_result(status, results, s, run_tests_log, csv):
    title = StatusStr[status]
    file_list = [fname for fname, fstatus in results if status == fstatus]
    total_tests = len(results)
    print_debug("%d / %d tests %s\n" % (len(file_list), total_tests, title), s, run_tests_log)
    if status == Status.Success:
        return
    for f in sorted(file_list):
        print_debug("\t%s\n" % f, s, run_tests_log)
        print_debug("%s;%s\n" % (f, title), csv, csv) # dump result to csv if filename is non-empty

def run_tests(options1, args, print_version):
    global exit_code
    global options
    options = options1
    global s
    s = options.silent
    # prepare run_tests_log and fail_db file
    if len(args) != 0 and not os.path.exists(options.fail_db):
        print("Fail database file not found!")
        exit_code = 1
        return 0

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

    target = TargetConfig(options.arch, options.target, options.device)

    if target.target.startswith("xe2lpg") and not 'ISPCRT_IGC_OPTIONS' in os.environ:
       os.environ['ISPCRT_IGC_OPTIONS'] = '+ -ftranslate-legacy-memory-intrinsics'

    if options.debug_check and (not target.is_xe() or not host.is_linux()):
        print("--debug_check is supported only for xe target and only on Linux OS")
        exit_code = 1
        return 0

    if options.jit_mode and options.nanobind:
        print("--jit mode is not compatible with --nanobind mode")
        exit_code = 1
        return 0

    if options.jit_mode and target.is_xe():
        print("--jit mode is not yet supported for xe targets")
        exit_code = 1
        return 0

    print_debug("Testing ISPC compiler: " + host.ispc_exe + "\n", s, run_tests_log)
    print_debug("Testing ISPC target: %s\n" % options.target, s, run_tests_log)
    print_debug("Testing ISPC arch: %s\n" % options.arch, s, run_tests_log)
    print_debug("ISPCRT_IGC_OPTIONS: %s\n" % os.environ.get('ISPCRT_IGC_OPTIONS', None), s, run_tests_log)

    set_compiler_exe(host, options)
    set_ispc_output(target, options)

    # Get JIT library path if JIT mode is enabled
    if options.jit_mode:
        jit_lib_path = get_jit_library_path(host, options)
        print_debug("JIT library path: %s\n" % jit_lib_path, s, run_tests_log)

    # print compilers versions
    if print_version > 0:
        common.print_version(host.ispc_exe, "", options.compiler_exe, False, run_tests_log, host.is_windows())

    # if no specific test files are specified, run all of the tests in tests/func-tests
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
    jit_lib_path_for_workers = jit_lib_path if options.jit_mode else None
    glob_var = [host, options, s, target, run_tests_log, jit_lib_path_for_workers]
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

    run_succeed_files = [fname for fname, fstatus in results if fstatus == Status.Success]
    skip_files = [fname for fname, fstatus in results if fstatus == Status.Skip]

    total_tests_executed = total_tests-len(skip_files)

    if options.non_interactive:
        print_debug(" Done %d / %d\n" % (finished_tests_counter.value, total_tests), s, run_tests_log)

    print_debug("\nExecuted %d / %d (%d skipped)\n\n" % (total_tests_executed, total_tests, len(skip_files)), s, run_tests_log)

    # Pass rate
    if (total_tests_executed) > 0:
        pass_rate = (len(run_succeed_files)/total_tests_executed)*100
    else:
        pass_rate = -1
    print_debug("PASSRATE (%d/%d) = %d%% \n\n" % (len(run_succeed_files), total_tests_executed, pass_rate), s, run_tests_log)

    for status in Status:
        print_result(status, results, s, run_tests_log, options.csv)
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
# Subprocess is used with default shell which is False, it's safe and doesn't allow shell injection
# so we can ignore the Bandit warning
import subprocess #nosec
import shlex
import platform
import tempfile
import os.path
import time
# our functions
import common
import traceback
print_debug = common.print_debug
error = common.error
exit_code = 0

# Use different default targets on different architectures.
default_target = "sse4-i32x4"
default_arch = "x86-64"
if platform.machine() == "arm":
    default_target = "neon-i32x4"
    default_arch = "arm"
elif platform.machine() == "aarch64":
    default_target = "neon-i32x4"
    default_arch = "aarch64"
elif platform.machine() == "arm64":
    default_target = "neon-i32x4"
    default_arch = "aarch64"
elif "86" in platform.machine() or platform.machine() == "AMD64":
    # Some variant of x86: x86_64, i386, i486, i586, i686
    # Windows reports platform as AMD64
    pass
else:
    print_debug("WARNING: host machine was not recognized - " + str(platform.machine()), False, "")

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-r", "--random-shuffle", dest="random", help="Randomly order tests",
                  default=False, action="store_true")
    parser.add_option("-f", "--ispc-flags", dest="ispc_flags", help="Additional flags for ispc (-g, -O1, ...)",
                  default="")
    parser.add_option('-t', '--target', dest='target',
                  help=('Set compilation target. For example: sse4-i32x4, avx2-i32x8, avx512skx-x16, etc.'), default=default_target)
    parser.add_option('-a', '--arch', dest='arch',
                  help='Set architecture (arm, aarch64, x86, x86-64, xe64)', default=default_arch)
    parser.add_option("-c", "--compiler", dest="compiler_exe", help="C/C++ compiler binary to use to run tests",
                  default=None)
    parser.add_option('-o', '--opt', dest='opt', choices=['', 'O0', 'O1', 'O2'], help='Set optimization level passed to the compiler (O0, O1, O2).',
                  default='O2')
    parser.add_option('-j', '--jobs', dest='num_jobs', help='Maximum number of jobs to run in parallel',
                  default="1024", type="int")
    parser.add_option('-v', '--verbose', dest='verbose', help='Enable verbose output',
                  default=False, action="store_true")
    parser.add_option('--wrap-exe', dest='wrapexe',
                  help='Executable to wrap test runs with (e.g. "valgrind" or "sde -spr -- ")',
                  default="")
    parser.add_option('--time', dest='time', help='Enable time output',
                  default=False, action="store_true")
    parser.add_option('--non-interactive', dest='non_interactive', help='Disable interactive status updates',
                  default=False, action="store_true")
    parser.add_option('-u', "--update-errors", dest='update', help='Update file with fails (F of FP)', default="")
    parser.add_option('-s', "--silent", dest='silent', help='enable silent mode without any output', default=False,
                  action = "store_true")
    parser.add_option("--file", dest='in_file', help='file to save run_tests output', default="")
    parser.add_option("--l0loader", dest='l0loader', help='Path to L0 loader', default="")
    parser.add_option("--device", dest='device', help='Specify target ISPC device. For example: core2, skx, cortex-a35, skl, tgllp, acm-g11, etc.', default=None)
    parser.add_option("--ispc_output", dest='ispc_output', choices=['obj', 'spv', 'ze'], help='Specify ISPC output', default=None)
    parser.add_option("--fail_db", dest='fail_db', help='File to use as a fail database', default='tests/fail_db.txt', type=str)
    parser.add_option("--debug_check", dest='debug_check', help='Run tests in debug mode with validating debug info', default=False, action="store_true")
    parser.add_option("--verify", dest='verify', help='verify the fail database file', default=False, action="store_true")
    parser.add_option("--save-bin", dest='save_bin', help='compile and create bin, but don\'t execute it',
                  default=False, action="store_true")
    parser.add_option('--csv', dest="csv", help="file to save testing results", default="")
    parser.add_option('--test_time', dest="test_time", help="time needed for each test", default=600, type="int", action="store")
    parser.add_option('--calling_conv', dest="calling_conv", help="Specify the calling convention to use", default=None, type="str", action="store")
    parser.add_option("--nanobind", dest='nanobind', help='Enable nanobind compilation mode', default=False, action="store_true")
    parser.add_option("--jit", dest='jit_mode', help='Enable JIT compilation mode', default=False, action="store_true")
    parser.add_option("--jit-lib-path", dest='jit_lib_path', help='Path to ISPC library directory for JIT mode', default=None)

    (options, args) = parser.parse_args()
    L = run_tests(options, args, 1)
    exit(exit_code)
