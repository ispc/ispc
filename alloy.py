#!/usr/bin/env python3
#
#  Copyright (c) 2013-2021, Intel Corporation
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

# // Author: Filippov Ilia

from collections import OrderedDict
from enum import Enum, auto
import re
import traceback

class SelfbuildType(Enum):
    # not a selfbuild
    SINGLE = auto()
    # complete selfbuild
    SELF = auto()
    # first phase of selfbuild only
    SELF_PHASE1 = auto()
    # second phase of selfbuild only
    SELF_PHASE2 = auto()

def alloy_error(line, error_type = 1):
    global return_status
    if error_type == 1:
        return_status = 1
    common.error(line, error_type)

def tail_and_save(file_in, file_out, tail = 100):
    with open(file_in, 'r') as f_in:
        lines = f_in.readlines()[-tail:]

    with open(file_out, 'w') as f_out:
        f_out.writelines(lines)


def setting_paths(llvm, ispc, sde):
    if llvm != "":
        os.environ["LLVM_HOME"]=llvm
    if ispc != "":
        os.environ["ISPC_HOME"]=ispc
    if sde != "":
        os.environ["SDE_HOME"]=sde

def get_sde():
    sde_exe = ""
    PATH_dir = os.environ["PATH"].split(os.pathsep)
    if current_OS == "Windows":
        sde_n = "sde.exe"
    else:
        sde_n = "sde"
    for counter in PATH_dir:
        if os.path.exists(counter + os.sep + sde_n) and sde_exe == "":
            sde_exe = counter + os.sep + sde_n
    if os.environ.get("SDE_HOME") != None:
        if os.path.exists(os.environ.get("SDE_HOME") + os.sep + sde_n):
            sde_exe = os.environ.get("SDE_HOME") + os.sep + sde_n
    return sde_exe

def check_LLVM(which_LLVM):
    answer = []
    if which_LLVM[0] == " ":
        return answer
    p = os.environ["LLVM_HOME"]
    for i in range(0,len(which_LLVM)):
        if not os.path.exists(p + os.sep + "bin-" + which_LLVM[i] + os.sep + "bin"):
            answer.append(which_LLVM[i])
    return answer

def try_do_LLVM(text, command, from_validation, verbose=False):
    print_debug("Command line: "+command+"\n", True, alloy_build)
    if from_validation == True:
        text = text + "\n"
    print_debug("Trying to " + text, from_validation, alloy_build)

    with subprocess.Popen(command, shell=True,universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
        for line in proc.stdout:
            print_debug(line, not verbose, alloy_build)
    proc.wait()
    exit_status = proc.returncode
    if exit_status != 0:
        print_debug("ERROR.\n", from_validation, alloy_build)
        alloy_error("can't " + text, 1)
    print_debug("DONE.\n", from_validation, alloy_build)

def checkout_LLVM(component, version_LLVM, target_dir, from_validation, verbose):
    # Identify the component
    GIT_REPO_BASE="https://github.com/llvm/llvm-project.git"

    # Identify the version
    # An example of using branch (instead of final tag) is the following (for 9.0):
    # git: "origin/release/9.x"
    if  version_LLVM == "trunk":
        GIT_TAG="main"
    elif  version_LLVM == "13_0":
        GIT_TAG="llvmorg-13.0.0"
    elif  version_LLVM == "12_0":
        GIT_TAG="llvmorg-12.0.1"
    elif  version_LLVM == "11_1":
        GIT_TAG="llvmorg-11.1.0"
    elif  version_LLVM == "11_0":
        GIT_TAG="llvmorg-11.0.1"
    elif  version_LLVM == "10_0":
        GIT_TAG="llvmorg-10.0.1"
    elif  version_LLVM == "9_0":
        GIT_TAG="llvmorg-9.0.1"
    elif  version_LLVM == "8_0":
        GIT_TAG="llvmorg-8.0.1"
    elif  version_LLVM == "7_1":
        GIT_TAG="llvmorg-7.1.0"
    elif  version_LLVM == "7_0":
        GIT_TAG="llvmorg-7.0.1"
    elif  version_LLVM == "6_0":
        GIT_TAG="llvmorg-6.0.1"
    else:
        alloy_error("Unsupported llvm version: " + version_LLVM, 1)

    try_do_LLVM("clone "+component+" from "+GIT_REPO_BASE+" to "+target_dir+" ",
                "git clone "+GIT_REPO_BASE+" "+target_dir,
                from_validation, verbose)
    if GIT_TAG != "main":
        os.chdir(target_dir)
        try_do_LLVM("switch to "+GIT_TAG+" tag ",
                    "git checkout -b "+GIT_TAG+" "+GIT_TAG, from_validation, verbose)
        os.chdir("..")

# ISPC uses LLVM dumps for debug output, so build correctly it requires these functions to be
# present in LLVM libraries. In LLVM 5.0 they are not there by default and require explicit enabling.
# In later version this functionality is triggered by enabling assertions.
def get_llvm_enable_dump_switch(version_LLVM):
    return " -DLLVM_ENABLE_DUMP=ON "

def get_llvm_disable_assertions_switch(llvm_disable_assertions):
    if llvm_disable_assertions == True:
        return "  -DLLVM_ENABLE_ASSERTIONS=OFF"
    else:
        return "  -DLLVM_ENABLE_ASSERTIONS=ON"

def build_LLVM(version_LLVM, folder, debug, selfbuild, extra, from_validation, force, make, gcc_toolchain_path, llvm_disable_assertions, verbose):
    print_debug("Building LLVM. Version: " + version_LLVM + ".\n", from_validation, alloy_build)
    # Here we understand what and where do we want to build
    current_path = os.getcwd()
    llvm_home = os.environ["LLVM_HOME"]

    make_sure_dir_exists(llvm_home)

    FOLDER_NAME=version_LLVM
    version_LLVM = re.sub('\.', '_', version_LLVM)

    os.chdir(llvm_home)
    if folder == "":
        folder = FOLDER_NAME
    if debug == True:
        folder = folder + "dbg"
    LLVM_SRC="llvm-" + folder
    LLVM_BUILD="build-" + folder
    LLVM_BIN="bin-" + folder
    if os.path.exists(LLVM_BIN + os.sep + "bin") and not force:
        alloy_error("you have folder " + LLVM_BIN + ".\nIf you want to rebuild use --force", 1)
    LLVM_BUILD_selfbuild = LLVM_BUILD + "_temp"
    LLVM_BIN_selfbuild = LLVM_BIN + "_temp"

    # Selfbuild phase2 assumes that directories are already create, for all other cases, create them.
    if selfbuild is SelfbuildType.SINGLE or selfbuild is SelfbuildType.SELF or selfbuild is SelfbuildType.SELF_PHASE1:
        common.remove_if_exists(LLVM_SRC)
        common.remove_if_exists(LLVM_BUILD)
        common.remove_if_exists(LLVM_BIN)
    if selfbuild is SelfbuildType.SELF or selfbuild is SelfbuildType.SELF_PHASE1:
        common.remove_if_exists(LLVM_BUILD_selfbuild)
        common.remove_if_exists(LLVM_BIN_selfbuild)
    print_debug("Using folders: " + LLVM_SRC + " " + LLVM_BUILD + " " + LLVM_BIN + " in " +
        llvm_home + "\n", from_validation, alloy_build)

    # Starting from MacOS 10.9 Maverics, C and C++ library headers are part of the SDK, not the OS itself.
    # System root must be specified during the compiler build, so the compiler knows the default location to search for headers.
    # C headers are located at system root location, while C++ headers are part of the toolchain.
    # I.e. specifying system root solved C header problem. For C++ headers we enable libc++ build as part of clang build (our own toolchain).
    # Note that on Sierra there's an issue with using C headers from High Sierra SDK, which instantiates as compile error:
    #     error: 'utimensat' is only available on macOS 10.13 or newer
    # This is due to using SDK targeting OS, which is newer than current one.
    mac_system_root = ""
    if current_OS == "MacOS" \
        and int(current_OS_version.split(".")[0]) >= 13:
        search_path = os.environ["PATH"].split(os.pathsep)
        found_xcrun = False
        for path in search_path:
            if os.path.exists(os.path.join(path, "xcrun")):
                found_xcrun = True
        if found_xcrun:
            mac_system_root = "`xcrun --show-sdk-path`"
        else:
            alloy_error("Can't find XCode (xcrun tool) - it's required on MacOS 10.9 and newer", 1)

    # prepare configuration parameters
    llvm_enable_projects = " -DLLVM_ENABLE_PROJECTS=\"clang"
    if current_OS == "MacOS" and int(current_OS_version.split(".")[0]) >= 13:
        # Starting with MacOS 10.9 Maverics, the system doesn't contain headers for standard C++ library and
        # the default library is libc++, bit libstdc++. The headers are part of XCode now. But we are checking out
        # headers as part of LLVM source tree, so they will be installed in clang location and clang will be able
        # to find them. Though they may not match to the library installed in the system, but seems that this should
        # not happen.
        # Note, that we can also build a libc++ library, but it must be on system default location or should be passed
        # to the linker explicitly (either through command line or environment variables). So we are not doing it
        # currently to make the build process easier.

        # We either need to explicitly opt-out from using libcxxabi from this repo, or build and use it,
        # otherwise a build error will occure (attempt to use just built libcxxabi, which was not built).
        # An option to build seems to be a better one.
        llvm_enable_projects +=";libcxx;libcxxabi"
    if current_OS == "Linux":
        # OpenMP is needed for Xe enabled builds.
        # Starting from Ubuntu 20.04 libomp-dev package doesn't install omp.h to default location.
        llvm_enable_projects +=";openmp"
    if extra == True:
        llvm_enable_projects +=";compiler-rt;clang-tools-extra"
    llvm_enable_projects += "\""

    if selfbuild is SelfbuildType.SINGLE or selfbuild is SelfbuildType.SELF or selfbuild is SelfbuildType.SELF_PHASE1:
        # clone llvm repo
        checkout_LLVM("llvm", version_LLVM, LLVM_SRC, from_validation, verbose)

        # patch llvm
        os.chdir(LLVM_SRC)
        patches = glob.glob(os.environ["ISPC_HOME"] + os.sep + "llvm_patches" + os.sep + "*.*")
        for patch in patches:
            if version_LLVM in os.path.basename(patch):
                try_do_LLVM("patch LLVM with patch " + patch + " ", "git apply " + patch, from_validation, verbose)
        os.chdir("../")

    # configuring llvm and build for first phase of selfbuild
    cmakelists_path = LLVM_SRC + "/llvm"
    if selfbuild is SelfbuildType.SELF or selfbuild is SelfbuildType.SELF_PHASE1:
        print_debug("Making selfbuild and use folders " + LLVM_BUILD_selfbuild + " and " +
            LLVM_BIN_selfbuild + "\n", from_validation, alloy_build)
        os.makedirs(LLVM_BUILD_selfbuild)
        os.makedirs(LLVM_BIN_selfbuild)
        os.chdir(LLVM_BUILD_selfbuild)
        try_do_LLVM("configure release version for selfbuild ",
                "cmake -G " + "\"" + generator + "\"" + " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" +
                "  -DCMAKE_INSTALL_PREFIX=" + llvm_home + "/" + LLVM_BIN_selfbuild +
                "  -DCMAKE_BUILD_TYPE=Release" +
                llvm_enable_projects +
                get_llvm_enable_dump_switch(version_LLVM) +
                get_llvm_disable_assertions_switch(llvm_disable_assertions) +
                "  -DLLVM_INSTALL_UTILS=ON" +
                (("  -DGCC_INSTALL_PREFIX=" + gcc_toolchain_path) if gcc_toolchain_path != "" else "") +
                (("  -DCMAKE_C_COMPILER=" + gcc_toolchain_path+"/bin/gcc") if gcc_toolchain_path != "" else "") +
                (("  -DCMAKE_CXX_COMPILER=" + gcc_toolchain_path+"/bin/g++") if gcc_toolchain_path != "" else "") +
                (("  -DDEFAULT_SYSROOT=" + mac_system_root) if mac_system_root != "" else "") +
                "  -DLLVM_TARGETS_TO_BUILD=AArch64\;ARM\;X86" +
                "  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly" +
                " ../" + cmakelists_path,
                from_validation, verbose)
        try_do_LLVM("build release version for selfbuild ", make, from_validation, verbose)
        try_do_LLVM("install release version for selfbuild ", "make install", from_validation, verbose)
        os.chdir("../")

    # set compiler to use if this is selfbuild
    selfbuild_compiler = ""
    if selfbuild is SelfbuildType.SELF or selfbuild is SelfbuildType.SELF_PHASE2:
        selfbuild_compiler = ("  -DCMAKE_C_COMPILER=" +llvm_home+ "/" + LLVM_BIN_selfbuild + "/bin/clang " +
                              "  -DCMAKE_CXX_COMPILER="+llvm_home+ "/" + LLVM_BIN_selfbuild + "/bin/clang++ ")
        print_debug("Use compiler for selfbuild: " + selfbuild_compiler + "\n", from_validation, alloy_build)


    # configure and build for regular build or second phase of selfbuild
    if selfbuild is SelfbuildType.SINGLE or selfbuild is SelfbuildType.SELF or selfbuild is SelfbuildType.SELF_PHASE2:
        os.makedirs(LLVM_BUILD)
        os.makedirs(LLVM_BIN)
        os.chdir(LLVM_BUILD)
        build_type = "Release" if debug == False else "Debug"
        if current_OS != "Windows":
            try_do_LLVM("configure " + build_type + " version ",
                    "cmake -G " + "\"" + generator + "\"" + " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" +
                    selfbuild_compiler +
                    "  -DCMAKE_INSTALL_PREFIX=" + llvm_home + "/" + LLVM_BIN +
                    "  -DCMAKE_BUILD_TYPE=" + build_type +
                    llvm_enable_projects +
                    get_llvm_enable_dump_switch(version_LLVM) +
                    get_llvm_disable_assertions_switch(llvm_disable_assertions) +
                    "  -DLLVM_INSTALL_UTILS=ON" +
                    (("  -DGCC_INSTALL_PREFIX=" + gcc_toolchain_path) if gcc_toolchain_path != "" else "") +
                    (("  -DCMAKE_C_COMPILER=" + gcc_toolchain_path+"/bin/gcc") if gcc_toolchain_path != "" and selfbuild_compiler == "" else "") +
                    (("  -DCMAKE_CXX_COMPILER=" + gcc_toolchain_path+"/bin/g++") if gcc_toolchain_path != "" and selfbuild_compiler == "" else "") +
                    (("  -DDEFAULT_SYSROOT=" + mac_system_root) if mac_system_root != "" else "") +
                    "  -DLLVM_TARGETS_TO_BUILD=AArch64\;ARM\;X86" +
                    "  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly" +
                    " ../" + cmakelists_path,
                    from_validation, verbose)
        else:
            try_do_LLVM("configure " + build_type + " version ",
                    'cmake -Thost=x64 -G ' + '\"' + generator + '\"' + ' -DCMAKE_INSTALL_PREFIX="..\\'+ LLVM_BIN + '" ' +
                    '  -DCMAKE_BUILD_TYPE=' + build_type +
                    llvm_enable_projects +
                    get_llvm_enable_dump_switch(version_LLVM) +
                    get_llvm_disable_assertions_switch(llvm_disable_assertions) +
                    '  -DLLVM_INSTALL_UTILS=ON' +
                    '  -DLLVM_TARGETS_TO_BUILD=AArch64\;ARM\;X86' +
                    '  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly' +
                    '  -DLLVM_LIT_TOOLS_DIR="C:\\gnuwin32\\bin" ..\\' + cmakelists_path,
                    from_validation, verbose)

        # building llvm
        if current_OS != "Windows":
            try_do_LLVM("build LLVM ", make, from_validation, verbose)
            try_do_LLVM("install LLVM ", "make install", from_validation, verbose)
        else:
            try_do_LLVM("build LLVM and then install LLVM ", "msbuild INSTALL.vcxproj /V:m /p:Platform=x64 /p:Configuration=" + build_type + " /t:rebuild", from_validation, verbose)
        os.chdir(current_path)


def unsupported_llvm_targets(LLVM_VERSION):
    prohibited_list = {"6.0":["avx512skx-i32x8", "avx512skx-i32x4", "avx512skx-i8x64", "avx512skx-i16x32"],
                       "7.0":["avx512skx-i32x8", "avx512skx-i32x4", "avx512skx-i8x64", "avx512skx-i16x32"],
                       "8.0":["avx512skx-i8x64", "avx512skx-i16x32"],
                       "9.0":["avx512skx-i8x64", "avx512skx-i16x32"]
                       }
    if LLVM_VERSION in prohibited_list:
        return prohibited_list[LLVM_VERSION]
    return []


# Split targets into categories: native, sde.
# native - native targets run natively on current hardware.
# sde - native target, which need to be emulated on current hardware.
def check_targets():
    result = []
    result_sde = []
    # check what native targets do we have
    if current_OS != "Windows":
        if options.ispc_build_compiler == "clang":
            cisa_compiler = "clang"
        elif options.ispc_build_compiler == "gcc":
            cisa_compiler = "g++"

        try_do_LLVM("build check_ISA", cisa_compiler + " check_isa.cpp -o check_isa.exe", True)
    else:
        try_do_LLVM("build check_ISA", "cl check_isa.cpp", True)

    # Dictionary mapping hardware architecture to its targets.
    # The value in the dictionary is:
    # [
    #   list of targets corresponding to this architecture,
    #   list of other architecture executable on this hardware,
    #   flag for sde to emulate this platform,
    #   flag is this is supported on current platform
    # ]
    target_dict = OrderedDict([
      ("SSE2",   [["sse2-i32x4",  "sse2-i32x8"],
                 ["SSE2"], "-p4", False]),
      ("SSE4",   [["sse4-i32x4",  "sse4-i32x8",   "sse4-i16x8", "sse4-i8x16"],
                 ["SSE2", "SSE4"], "-wsm", False]),
      ("AVX",    [["avx1-i32x4",  "avx1-i32x8",  "avx1-i32x16",  "avx1-i64x4"],
                 ["SSE2", "SSE4", "AVX"], "-snb", False]),
      ("AVX2",   [["avx2-i32x4", "avx2-i32x8",  "avx2-i32x16",  "avx2-i64x4", "avx2-i8x32", "avx2-i16x16"],
                 ["SSE2", "SSE4", "AVX", "AVX2"], "-hsw", False]),
      ("KNL",    [["avx512knl-i32x16"],
                 ["SSE2", "SSE4", "AVX", "AVX2", "KNL"], "-knl", False]),
      ("SKX",    [["avx512skx-i32x16", "avx512skx-i32x8", "avx512skx-i32x4", "avx512skx-i8x64", "avx512skx-i16x32"],
                 ["SSE2", "SSE4", "AVX", "AVX2", "SKX"], "-skx", False])
    ])

    hw_arch = take_lines("check_isa.exe", "first").split()[1]

    if not (hw_arch in target_dict):
        alloy_error("Architecture " + hw_arch + " was not recognized", 1)

    # Mark all compatible architecutres in the dictionary.
    for compatible_arch in target_dict[hw_arch][1]:
        target_dict[compatible_arch][3] = True

    # Now initialize result and result_sde.
    for key in target_dict:
        item = target_dict[key]
        targets = item[0]
        if item[3]:
            # Supported natively
            result = result + targets
        else:
            # Supported through SDE
            for target in targets:
                result_sde = result_sde + [[item[2], target]]

    # now check what targets we have with the help of SDE
    sde_exists = get_sde()
    if sde_exists == "":
        alloy_error("you haven't got sde neither in SDE_HOME nor in your PATH.\n" +
            "To test all platforms please set SDE_HOME to path containing SDE.\n" +
            "Please refer to http://www.intel.com/software/sde for SDE download information.", 2)

    return [result, result_sde]

def build_ispc(version_LLVM, make):
    current_path = os.getcwd()
    ispc_home = os.environ["ISPC_HOME"]
    os.chdir(ispc_home)

    make_ispc = "make " + options.ispc_build_compiler + " -j" + options.speed
    ISPC_BUILD="build-" + version_LLVM
    ISPC_BIN="bin-" + version_LLVM
    if not os.path.exists(ISPC_BUILD):
        os.makedirs(ISPC_BUILD)
    if not os.path.exists(ISPC_BUILD):
        os.makedirs(ISPC_BIN)
    os.chdir(ISPC_BUILD)

    if current_OS != "Windows":
        p_temp = os.getenv("PATH")
        os.environ["PATH"] = os.environ["LLVM_HOME"] + "/bin-" + version_LLVM + "/bin:" + os.environ["PATH"]

        folder = os.environ["LLVM_HOME"]  + os.sep + "llvm-"
        if options.folder == "":
            folder += version_LLVM
        if options.debug == True:
            folder +=  "dbg"

        try_do_LLVM("configure ispc build", 'cmake -DCMAKE_INSTALL_PREFIX="..\\'+ ISPC_BIN + '" ' +
                    '  -DCMAKE_BUILD_TYPE=Release' +
                        ispc_home, True)
        try_do_LLVM("build ISPC with LLVM version " + version_LLVM + " ", make_ispc, True)
        try_do_LLVM("install ISPC ", "make install", True)
        copyfile(os.path.join(ispc_home, ISPC_BIN, "bin", "ispc"), os.path.join(ispc_home, + "ispc"))
        os.environ["PATH"] = p_temp
    else:
        try_do_LLVM("configure ispc build", 'cmake -Thost=x64 -G ' + '\"' + generator + '\"' + ' -DCMAKE_INSTALL_PREFIX="..\\'+ ISPC_BIN + '" ' +
                    '  -DCMAKE_BUILD_TYPE=Release ' +
                        ispc_home, True)
        try_do_LLVM("clean ISPC for building", "msbuild ispc.vcxproj /t:clean", True)
        try_do_LLVM("build ISPC with LLVM version " + version_LLVM + " ", "msbuild ispc.vcxproj /V:m /p:Platform=x64 /p:Configuration=Release /t:rebuild", True)
        try_do_LLVM("install ISPC  ", "msbuild INSTALL.vcxproj /p:Platform=x64 /p:Configuration=Release", True)
        copyfile(os.path.join(ispc_home, ISPC_BIN, "bin", "ispc.exe"), os.path.join(ispc_home, + "ispc.exe"))

    os.chdir(current_path)

def execute_stability(stability, R, print_version):
    global return_status
    try:
        stability1 = copy.deepcopy(stability)

        b_temp = run_tests.run_tests(stability1, [], print_version)
        temp = b_temp[0]
        time = b_temp[1]
        for j in range(0,4):
            R[j][0] = R[j][0] + temp[j] # new_runfails, new_compfails, new_passes_runfails, new_passes_compfails
            for i in range(0,len(temp[j])):
                R[j][1].append(temp[4])
        number_of_fails = temp[5]
        number_of_new_fails = len(temp[0]) + len(temp[1])
        number_of_passes = len(temp[2]) + len(temp[3])
        if number_of_fails == 0:
            str_fails = ". No fails"
        else:
            str_fails = ". Fails: " + str(number_of_fails)
        if number_of_new_fails == 0:
            str_new_fails = ", No new fails"
        else:
            str_new_fails = ", New fails: " + str(number_of_new_fails)
        if number_of_passes == 0:
            str_new_passes = "."
        else:
            str_new_passes = ", " + str(number_of_passes) + " new passes."
        if stability.time:
            str_time = " " + time + "\n"
        else:
            str_time = "\n"
        print_debug(temp[4][1:-3] + stability1.ispc_flags + str_fails + str_new_fails + str_new_passes + str_time, False, stability_log)
    except Exception as e:
        print_debug("Exception: " + str(e), False, stability_log)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stderr)
        print_debug("ERROR: Exception in execute_stability: %s\n" % (sys.exc_info()[1]), False, stability_log)
        return_status = 1


'''
R       = [[new_runfails,        [new_line, new_line...]],
           [new_compfails,       [new_line, new_line...]],
           [new_passes_runfails, [new_line, new_line...]],
           [new_passes_runfails, [new_line, new_line...]]]
'''
def output_test_results(R):
    ttt = ["NEW RUNFAILS: ", "NEW COMPFAILS: ", "NEW PASSES RUNFAILS: ", "NEW PASSES COMPFAILS: "]
    for j in range(0, 4):
        if len(R[j][0]) == 0:
            print_debug("NO " + ttt[j][:-2] + "\n", False, stability_log)
        else:
            print_debug(ttt[j] + str(len(R[j][0])) + "\n", False, stability_log)
            to_print = {}
            for (fail_name, opt_str) in zip(R[j][0], R[j][1]):
                if fail_name not in to_print:
                    to_print[fail_name] = []
                to_print[fail_name].append(opt_str)

            # sort
            for key in to_print.keys():
                to_print[key] = sorted(to_print[key])

            # print out
            for fail_name in sorted(to_print.keys()):
                print_debug("\t" + fail_name + "\n", True, stability_log)
                for opt_str in to_print[fail_name]:
                    print_debug("\t\t\t" + opt_str, True, stability_log)

def concatenate_test_results(R1, R2):
    R = [[[],[]],[[],[]],[[],[]],[[],[]]]
    for j in range(0, 4):
        R[j][0] = R1[j][0] + R2[j][0]
        R[j][1] = R1[j][1] + R2[j][1]
    return R

def validation_run(only, only_targets, reference_branch, number, update, speed_number, make, perf_llvm, time):
    os.chdir(os.environ["ISPC_HOME"])
    if current_OS != "Windows":
        os.environ["PATH"] = os.environ["ISPC_HOME"] + ":" + os.environ["PATH"]
    print_debug("Command: " + ' '.join(sys.argv) + "\n", False, "")
    print_debug("Folder: " + os.environ["ISPC_HOME"] + "\n", False, "")
    date = datetime.datetime.now()
    print_debug("Date: " + date.strftime('%H:%M %d/%m/%Y') + "\n", False, "")
    newest_LLVM="13.0"
# *** *** ***
# Stability validation run
# *** *** ***
    if ((("stability" in only) == True) or ("performance" in only) == False):
        print_debug("\n\nStability validation run\n\n", False, "")
        stability = common.EmptyClass()
# stability constant options
        stability.save_bin = False
        stability.random = False
        stability.ispc_flags = options.ispc_flags
        stability.compiler_exe = options.compiler_exe
        stability.num_jobs = speed_number
        stability.verbose = False
        stability.time = time
        # 1200 is more than default value in run_tests.py (600).
        # There's a single test, which requires longer time on AVX2 capable server (Github Action):
        # tests/idiv.ispc running for avx512-i8x64 for x86 under SDE.
        # For any other tests it should be more than enough.
        stability.test_time = 1200
        stability.csv = ""
        stability.non_interactive = True
        stability.update = update
        stability.include_file = None
        stability.silent = True
        stability.in_file = "." + os.sep + f_date + os.sep + "run_tests_log.log"
        stability.verify = False
        stability.fail_db = "fail_db.txt"
        stability.device = None
        stability.ispc_output = None
        stability.debug_check = False
# stability varying options
        stability.target = ""
        stability.arch = ""
        stability.opt = ""
        stability.wrapexe = ""
# prepare parameters of run
        [targets_t, sde_targets_t] = check_targets()
        rebuild = True
        opts = []
        archs = []
        LLVM = []
        targets = []
        sde_targets = []
        dbg_begin = 0
        dbg_total = 1

# parsing option only, update parameters of run
        if "-O2" in only:
            opts.append("O2")
        if "-O1" in only:
            opts.append("O1")
        if "-O0" in only:
            opts.append("O0")
        if "debug" in only:
            if not ("nodebug" in only):
                dbg_begin = 1
            dbg_total = 2
        if "x86" in only and not ("x86-64" in only):
            archs.append("x86")
        if "x86-64" in only:
            archs.append("x86-64")
        if "native" in only:
            sde_targets_t = []
        for i in ["6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "trunk"]:
            if i in only:
                LLVM.append(i)
        if "current" in only:
            LLVM = [" "]
            rebuild = False
        else:
            common.check_tools(1)

        if only_targets != "":
            only_targets += " "
            only_targets_t = only_targets.split(" ")

            for i in only_targets_t:
                if i == "":
                    continue
                err = True
                for j in range(0,len(targets_t)):
                    if i in targets_t[j]:
                        targets.append(targets_t[j])
                        err = False
                for j in range(0,len(sde_targets_t)):
                    if i in sde_targets_t[j][1]:
                        sde_targets.append(sde_targets_t[j])
                        err = False
                if err == True:
                    alloy_error("You haven't sde for target " + i, 1)
        else:
            targets = targets_t
            sde_targets = sde_targets_t

        if "build" in only:
            targets = []
            sde_targets = []
            only = only + " stability "
# finish parameters of run, prepare LLVM
        if len(opts) == 0:
            opts = ["O2"]
        if len(archs) == 0:
            archs = ["x86", "x86-64"]
        if len(LLVM) == 0:
            LLVM = [newest_LLVM, "trunk"]
        need_LLVM = check_LLVM(LLVM)
        for i in range(0,len(need_LLVM)):
            build_LLVM(need_LLVM[i], "", "", "", False, False, False, True, False, make, options.gcc_toolchain_path, False, True, False)
# begin validation run for stabitily
        common.remove_if_exists(stability.in_file)
        R = [[[],[]],[[],[]],[[],[]],[[],[]]]
        print_debug("\n" + common.get_host_name() + "\n", False, stability_log)
        print_debug("\n_________________________STABILITY REPORT_________________________\n", False, stability_log)
        ispc_flags_tmp = stability.ispc_flags
        for i in range(0,len(LLVM)):
            R_tmp = [[[],[]],[[],[]],[[],[]],[[],[]]]
            print_version = 2
            if rebuild:
                build_ispc(LLVM[i], make)
            for j in range(0,len(targets)):
                stability.target = targets[j]
                # the target might be not supported by the chosen llvm version
                if (stability.target in unsupported_llvm_targets(LLVM[i])):
                    print_debug("Warning: target " + stability.target + " is not supported in LLVM " + LLVM[i] + "\n", False, stability_log)
                    continue

                # now set archs for targets
                arch = archs
                for i1 in range(0,len(arch)):
                    for i2 in range(0,len(opts)):
                        for i3 in range(dbg_begin,dbg_total):
                            stability.arch = arch[i1]
                            stability.opt = opts[i2]
                            stability.ispc_flags = ispc_flags_tmp
                            if (i3 != 0):
                                stability.ispc_flags += " -g"
                            execute_stability(stability, R_tmp, print_version)
                            print_version = 0
            for j in range(0,len(sde_targets)):
                stability.target = sde_targets[j][1]
                # the target might be not supported by the chosen llvm version
                if (stability.target in unsupported_llvm_targets(LLVM[i])):
                    print_debug("Warning: target " + stability.target + " is not supported in LLVM " + LLVM[i] + "\n", False, stability_log)
                    continue

                stability.wrapexe = get_sde() + " " + sde_targets[j][0] + " -- "
                arch = archs
                for i1 in range(0,len(arch)):
                    for i2 in range(0,len(opts)):
                        for i3 in range(dbg_begin,dbg_total):
                            stability.arch = arch[i1]
                            stability.opt = opts[i2]
                            stability.ispc_flags = ispc_flags_tmp
                            if (i3 != 0):
                                stability.ispc_flags += " -g"
                            execute_stability(stability, R_tmp, print_version)
                            print_version = 0
            # Output testing results separate for each tested LLVM version
            R = concatenate_test_results(R, R_tmp)
            output_test_results(R_tmp)
            print_debug("\n", False, stability_log)

        print_debug("\n----------------------------------------\nTOTAL:\n", False, stability_log)
        output_test_results(R)
        print_debug("__________________Watch stability.log for details_________________\n", False, stability_log)

# *** *** ***
# Performance validation run
# *** *** ***
    if ((("performance" in only) == True) or ("stability" in only) == False):
        print_debug("\n\nPerformance validation run\n\n", False, "")
        common.check_tools(1)
        performance = common.EmptyClass()
# performance constant options
        performance.number = number
        performance.config = "." + os.sep + "perf.ini"
        performance.path = "." + os.sep
        performance.silent = True
        performance.output = ""
        performance.compiler = ""
        performance.ref = "ispc_ref"
        if current_OS == "Windows":
            performance.ref = "ispc_ref.exe"
        performance.perf_target = ""
        performance.in_file = "." + os.sep + f_date + os.sep + "performance.log"
# prepare newest LLVM
        need_LLVM = check_LLVM([newest_LLVM])
        if len(need_LLVM) != 0:
            build_LLVM(need_LLVM[0], "", "", "", False, False, False, True, False, make, options.gcc_toolchain_path, True, False)
        if perf_llvm == False:
            # prepare reference point. build both test and reference compilers
            try_do_LLVM("apply git", "git branch", True)
            temp4 = take_lines("git branch", "all")
            for line in temp4:
                if "*" in line:
                    current_branch = line[2:-1]
            stashing = True
            sys.stdout.write("Please, don't interrupt script here! You can have not sync git status after interruption!\n")
            if "No local changes" in take_lines("git stash", "first"):
                stashing = False
            #try_do_LLVM("stash current branch ", "git stash", True)
            try_do_LLVM("checkout reference branch " + reference_branch + " ", "git checkout " + reference_branch, True)
            sys.stdout.write(".\n")
            build_ispc(newest_LLVM, make)
            sys.stdout.write(".\n")
            if current_OS != "Windows":
                os.rename("ispc", "ispc_ref")
            else:
                common.remove_if_exists("ispc_ref.exe")
                os.rename("ispc.exe", "ispc_ref.exe")
            try_do_LLVM("checkout test branch " + current_branch + " ", "git checkout " + current_branch, True)
            if stashing:
                try_do_LLVM("return current branch ", "git stash pop", True)
            sys.stdout.write("You can interrupt script now.\n")
            build_ispc(newest_LLVM, make)
        else:
            # build compiler with two different LLVM versions
            if len(check_LLVM([reference_branch])) != 0:
                alloy_error("you haven't got llvm called " + reference_branch, 1)
            build_ispc(newest_LLVM, make)
            os.rename("ispc", "ispc_ref")
            build_ispc(reference_branch, make)
        # begin validation run for performance. output is inserted into perf()
        perf.perf(performance, [])
    # dumping gathered info to the file
    common.ex_state.dump(alloy_folder + "test_table.dump", common.ex_state.tt)


def Main():
    global current_OS
    global current_OS_version
    global return_status
    current_OS_version = platform.release()
    if (platform.system() == 'Windows' or 'CYGWIN_NT' in platform.system()) == True:
        current_OS = "Windows"
    else:
        if (platform.system() == 'Darwin'):
            current_OS = "MacOS"
        else:
            current_OS = "Linux"
    if (options.build_llvm == False and options.validation_run == False):
        parser.print_help()
        exit(1)

    # set appropriate makefile target
    # gcc and g++ options are equal and added for ease of use
    if options.ispc_build_compiler != "clang" and \
       options.ispc_build_compiler != "gcc":
        alloy_error("unknow option for --ispc-build-compiler: " + options.ispc_build_compiler, 1)
        parser.print_help()
        exit(1)

    # check and normalize selfbuild switches
    selfbuild = SelfbuildType.SINGLE
    if (options.selfbuild and (options.selfbuild_phase1 or options.selfbuild_phase2)) or (options.selfbuild_phase1 and options.selfbuild_phase2):
        alloy_error("Only one of --selfbuild* switches can be used at the same time", 1)
    if options.selfbuild:
        selfbuild = SelfbuildType.SELF
    if options.selfbuild_phase1:
        selfbuild = SelfbuildType.SELF_PHASE1
    if options.selfbuild_phase2:
        selfbuild = SelfbuildType.SELF_PHASE2

    setting_paths(options.llvm_home, options.ispc_home, options.sde_home)
    if os.environ.get("LLVM_HOME") == None:
        alloy_error("you have no LLVM_HOME", 1)
    if os.environ.get("ISPC_HOME") == None:
        alloy_error("you have no ISPC_HOME", 1)
    if options.only != "":
        test_only_r = " 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 trunk current build stability performance x86 x86-64 x86_64 -O0 -O1 -O2 native debug nodebug "
        test_only = options.only.split(" ")
        for iterator in test_only:
            if not (" " + iterator + " " in test_only_r):
                alloy_error("unknown option for only: " + iterator, 1)
    if current_OS == "Windows" and selfbuild is not SelfbuildType.SINGLE:
        alloy_error("Selfbuild is not supported on Windows", 1)
    global f_date
    f_date = "logs"
    common.remove_if_exists(f_date)
    os.makedirs(f_date)
    global alloy_folder
    alloy_folder = os.getcwd() + os.sep + f_date + os.sep
    global alloy_build
    alloy_build = alloy_folder + "alloy_build.log"
    global stability_log
    stability_log = alloy_folder + "stability.log"
    current_path = os.getcwd()
    make = "make -j" + options.speed
    if os.environ["ISPC_HOME"] != os.getcwd():
        alloy_error("your ISPC_HOME and your current path are different! (" + os.environ["ISPC_HOME"] + " is not equal to " + os.getcwd() +
        ")\n", 2)
    if options.perf_llvm == True:
        if options.branch == "main":
            options.branch = "trunk"
    global generator
    if options.generator:
        generator = options.generator
    else:
        if current_OS == "Windows":
            generator = "Visual Studio 16"
        else:
            generator = "Unix Makefiles"
    try:
        start_time = time.time()
        if options.build_llvm:
            build_LLVM(options.version, options.folder,
                    options.debug, selfbuild, options.extra, False, options.force, make, options.gcc_toolchain_path, options.llvm_disable_assertions, options.verbose)
        if options.validation_run:
            validation_run(options.only, options.only_targets, options.branch,
                    options.number_for_performance, options.update, int(options.speed),
                    make, options.perf_llvm, options.time)
        elapsed_time = time.time() - start_time
        if options.time:
            print_debug("Elapsed time: " + time.strftime('%Hh%Mm%Ssec.', time.gmtime(elapsed_time)) + "\n", False, "")
    except Exception as e:
        print_debug("Exception: " + str(e) + "\n", False, stability_log)
        return_status = 1

    # Finish execution: time reporting and copy log
    try:
        os.chdir(current_path)
        date_name = "alloy_results_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if os.path.exists(date_name):
            alloy_error("It's forbidden to run alloy two times in a second, logs are in ./logs", 1)
        os.rename(f_date, date_name)
        print_debug("Logs are in " + date_name + "\n", False, "")
    except Exception as e:
        # Do not return non-zero exit code here, as it's not a critical error and testing might be considered successful.
        print_debug("Exception: " + str(e), False, stability_log)

    if current_OS == "Windows":
        # Windows hangs from time to time on exit, so returning without cleanup.
        sys.stdout.flush()
        os._exit(return_status)
    exit(return_status)

###Main###
from optparse import OptionParser
from optparse import OptionGroup
import sys
import os
import errno
import operator
import time
import glob
import platform
import smtplib
import datetime
import copy
import multiprocessing
import subprocess
import re
from shutil import copyfile
# our drivers
import run_tests
import perf
import common
take_lines = common.take_lines
print_debug = common.print_debug
make_sure_dir_exists = common.make_sure_dir_exists
return_status = 0
if __name__ == '__main__':
    # parsing options
    class MyParser(OptionParser):
        def format_epilog(self, formatter):
            return self.epilog
    examples =  ("Examples:\n" +
    "Download and build LLVM trunk\n\talloy.py -b\n" +
    "Download and build LLVM 13.0. Rewrite LLVM folders\n\talloy.py -b --version=13.0 --force\n" +
    "Validation run with LLVM trunk; x86, x86-64; -O2;\nall supported targets; performance\n\talloy.py -r\n" +
    "Validation run with all avx targets and sse4-i8x16 without performance\n\talloy.py -r --only=stability --only-targets='avx sse4-i8x16'\n" +
    "Validation run with avx2-i32x8, all sse4 and sse2 targets\nand all targets with i32x16\n\talloy.py -r --only-targets='avx2-i32x8 sse4 i32x16 sse2'\n" +
    "Stability validation run with LLVM 7.0, 8.0; -O0; x86,\nupdate fail_db.txt with passes and fails\n\talloy.py -r --only='7.0 -O0 stability 8.0 x86' --update-errors=FP\n" +
    "Try to build compiler with all LLVM\n\talloy.py -r --only=build\n" +
    "Performance validation run with 10 runs of each test and comparing to branch 'old'\n\talloy.py -r --only=performance --compare-with=old --number=10\n" +
    "Validation run. Update fail_db.txt with new fails\n\talloy.py -r --update-errors=F\n" +
    "Test KNL target (requires sde)\n\talloy.py -r --only='stability' --only-targets='avx512knl-i32x16'\n")

    num_threads="%s" % multiprocessing.cpu_count()
    parser = MyParser(usage="Usage: alloy.py -r/-b [options]", epilog=examples)
    parser.add_option('-b', '--build-llvm', dest='build_llvm',
        help='ask to build LLVM', default=False, action="store_true")
    parser.add_option('-r', '--run', dest='validation_run',
        help='ask for validation run', default=False, action="store_true")
    parser.add_option('-j', dest='speed',
        help='set -j for make', default=num_threads)
    parser.add_option('--ispc-build-compiler', dest='ispc_build_compiler',
        help='set compiler to build ispc binary (clang/gcc)', default="clang")
    # options for activity "build LLVM"
    llvm_group = OptionGroup(parser, "Options for building LLVM",
                    "These options must be used with -b option.")
    llvm_group.add_option('--version', dest='version',
        help='version of llvm to build: 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 trunk. Default: trunk', default="trunk")
    llvm_group.add_option('--with-gcc-toolchain', dest='gcc_toolchain_path',
         help='GCC install dir to use when building clang. It is important to set when ' +
         'you have alternative gcc installation. Note that otherwise gcc from standard ' +
         'location will be used, not from your PATH', default="")
    llvm_group.add_option('--debug', dest='debug',
        help='debug build of LLVM', default=False, action="store_true")
    llvm_group.add_option('--folder', dest='folder',
        help='folder to build LLVM in', default="")
    llvm_group.add_option('--selfbuild', dest='selfbuild',
        help='make selfbuild of LLVM and clang', default=False, action="store_true")
    llvm_group.add_option('--selfbuild-phase1', dest='selfbuild_phase1',
        help='make selfbuild of LLVM and clang, first phase only', default=False, action="store_true")
    llvm_group.add_option('--selfbuild-phase2', dest='selfbuild_phase2',
        help='make selfbuild of LLVM and clang, second phase only', default=False, action="store_true")
    llvm_group.add_option('--llvm-disable-assertions', dest='llvm_disable_assertions',
        help='build LLVM with assertions disabled', default=False, action="store_true")
    llvm_group.add_option('--force', dest='force',
        help='rebuild LLVM', default=False, action='store_true')
    llvm_group.add_option('--extra', dest='extra',
        help='load extra clang tools', default=False, action='store_true')
    llvm_group.add_option('--verbose', dest='verbose',
        help='verbose output during the build', default=False, action='store_true')
    parser.add_option_group(llvm_group)
    # options for activity "validation run"
    run_group = OptionGroup(parser, "Options for validation run",
                    "These options must be used with -r option.")
    run_group.add_option('--compare-with', dest='branch',
        help='set performance reference point. Default: main', default="main")
    run_group.add_option('--compiler', dest='compiler_exe',
        help='C/C++ compiler binary to use to run tests.', default=None)
    run_group.add_option('--ispc-flags', dest='ispc_flags',
        help='extra ispc flags.', default="")
    run_group.add_option('--number', dest='number_for_performance',
        help='number of performance runs for each test. Default: 5', default=5)
    run_group.add_option('--update-errors', dest='update',
        help='rewrite fail_db.txt file according to received results (F or FP)', default="")
    run_group.add_option('--only-targets', dest='only_targets',
        help='set list of targets to test. Possible values - all subnames of targets', default="")
    run_group.add_option('--time', dest='time',
        help='display time of testing', default=False, action='store_true')
    run_group.add_option('--only', dest='only',
        help='set types of tests. Possible values:\n' +
            '-O0, -O1, -O2, x86, x86-64, stability (test only stability), performance (test only performance),\n' +
            'build (only build with different LLVM), 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, trunk, native (do not use SDE),\n' +
            'current (do not rebuild ISPC), debug (only with debug info), nodebug (only without debug info, default).',
            default="")
    run_group.add_option('--perf_LLVM', dest='perf_llvm',
        help='compare LLVM 8.0 with "--compare-with", default trunk', default=False, action='store_true')
    run_group.add_option('--generator', dest='generator',
        help='specify cmake generator', default="")
    parser.add_option_group(run_group)
    # options for activity "setup PATHS"
    setup_group = OptionGroup(parser, "Options for setup",
                    "These options must be use with -r or -b to setup environment variables")
    setup_group.add_option('--llvm_home', dest='llvm_home',help='path to LLVM',default="")
    setup_group.add_option('--ispc_home', dest='ispc_home',help='path to ISPC',default="")
    setup_group.add_option('--sde_home', dest='sde_home',help='path to SDE',default="")
    parser.add_option_group(setup_group)
    (options, args) = parser.parse_args()
    Main()
