#!/usr/bin/python
#
#  Copyright (c) 2013-2019, Intel Corporation
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
import re

def tail_and_save(file_in, file_out, tail = 100):    
    with open(file_in, 'r') as f_in:
        lines = f_in.readlines()[-tail:]
        
    with open(file_out, 'w') as f_out:
        f_out.writelines(lines)


def attach_mail_file(msg, filename, name, tail = -1):
    if os.path.exists(filename):
        if tail > 0:
            tail_and_save(filename, filename + '.tail', tail)
            fp = open(filename + '.tail', "rb")
        else:
            fp = open(filename, "rb")
        
        to_attach = MIMEBase("application", "octet-stream")
        to_attach.set_payload(fp.read())
        encode_base64(to_attach)
        to_attach.add_header("Content-Disposition", "attachment", filename=name)
        fp.close()
        msg.attach(to_attach)


def setting_paths(llvm, ispc, sde):
    if llvm != "":
        os.environ["LLVM_HOME"]=llvm
    if ispc != "":
        os.environ["ISPC_HOME"]=ispc
    if sde != "":
        os.environ["SDE_HOME"]=sde

def get_sde():
    sde_exe = ""
    PATH_dir = string.split(os.getenv("PATH"), os.pathsep)
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

def try_do_LLVM(text, command, from_validation):
    print_debug("Command line: "+command+"\n", True, alloy_build)
    if from_validation == True:
        text = text + "\n"
    print_debug("Trying to " + text, from_validation, alloy_build)
    postfix = ""
    if current_OS == "Windows":
        postfix = " 1>> " + alloy_build + " 2>&1"
    else:
        postfix = " >> " + alloy_build + " 2>> " + alloy_build
    if os.system(command + postfix) != 0:
        print_debug("ERROR.\n", from_validation, alloy_build)
        if options.notify != "":
            msg = MIMEMultipart()
            attach_mail_file(msg, stability_log, "stability.log")
            send_mail("ERROR: Non-zero exit status while executing " + command + ". Examine build log for more information.", msg)
        error("can't " + text, 1)
    print_debug("DONE.\n", from_validation, alloy_build)

def checkout_LLVM(component, use_git, version_LLVM, revision, target_dir, from_validation):
    # Identify the component
    GIT_REPO_BASE="http://llvm.org/git/"
    #GIT_REPO_BASE="https://github.com/llvm-mirror/"
    if component == "llvm":
        SVN_REPO="https://llvm.org/svn/llvm-project/llvm/"
        GIT_REPO=GIT_REPO_BASE+"llvm.git"
    elif component == "clang":
        SVN_REPO="https://llvm.org/svn/llvm-project/cfe/"
        GIT_REPO=GIT_REPO_BASE+"clang.git"
    elif component == "libcxx":
        SVN_REPO="https://llvm.org/svn/llvm-project/libcxx/"
        GIT_REPO=GIT_REPO_BASE+"libcxx.git"
    elif component == "clang-tools-extra":
        SVN_REPO="https://llvm.org/svn/llvm-project/clang-tools-extra/"
        GIT_REPO=GIT_REPO_BASE+"clang-tools-extra.git"
    elif component == "compiler-rt":
        SVN_REPO="https://llvm.org/svn/llvm-project/compiler-rt/"
        GIT_REPO=GIT_REPO_BASE+"compiler-rt.git"
    else:
        error("Trying to checkout unidentified component: " + component, 1)

    # Identify the version
    if  version_LLVM == "trunk":
        SVN_PATH="trunk"
        GIT_BRANCH="master"
    elif  version_LLVM == "8_0":
        SVN_PATH="tags/RELEASE_800/final"
        GIT_BRANCH="release_80"
    elif  version_LLVM == "7_0":
        SVN_PATH="tags/RELEASE_701/final"
        GIT_BRANCH="release_70"
    elif  version_LLVM == "6_0":
        SVN_PATH="tags/RELEASE_601/final"
        GIT_BRANCH="release_60"
    elif  version_LLVM == "5_0":
        SVN_PATH="tags/RELEASE_502/final"
        GIT_BRANCH="release_50"
    elif  version_LLVM == "4_0":
        SVN_PATH="tags/RELEASE_401/final"
        GIT_BRANCH="release_40"
    elif  version_LLVM == "3_9":
        SVN_PATH="tags/RELEASE_390/final"
        GIT_BRANCH="release_39"
    elif  version_LLVM == "3_8":
        SVN_PATH="tags/RELEASE_381/final"
        GIT_BRANCH="release_38"
    elif  version_LLVM == "3_7":
        SVN_PATH="tags/RELEASE_370/final"
        GIT_BRANCH="release_37"
    elif  version_LLVM == "3_6":
        SVN_PATH="tags/RELEASE_362/final"
        GIT_BRANCH="release_36"
    elif  version_LLVM == "3_5":
        SVN_PATH="tags/RELEASE_351/final"
        GIT_BRANCH="release_35"
    elif  version_LLVM == "3_4":
        SVN_PATH="tags/RELEASE_34/dot2-final"
        GIT_BRANCH="release_34"
    elif  version_LLVM == "3_3":
        SVN_PATH="tags/RELEASE_33/final"
        GIT_BRANCH="release_33"
    elif  version_LLVM == "3_2":
        SVN_PATH="tags/RELEASE_32/final"
        GIT_BRANCH="release_32"
    else:
        error("Unsupported llvm version: " + version_LLVM, 1)

    if use_git:
        try_do_LLVM("clone "+component+" from "+GIT_REPO+" to "+target_dir+" ",
                    "git clone "+GIT_REPO+" "+target_dir,
                    from_validation)
        if GIT_BRANCH != "master":
            os.chdir(target_dir)
            try_do_LLVM("switch to "+GIT_BRANCH+" branch ",
                        "git checkout -b "+GIT_BRANCH+" remotes/origin/"+GIT_BRANCH, from_validation)
            os.chdir("..")
    else:
        try_do_LLVM("load "+component+" from "+SVN_REPO+SVN_PATH+" ",
                    "svn co --non-interactive "+revision+" "+SVN_REPO+SVN_PATH+" "+target_dir,
                    from_validation)

# ISPC uses LLVM dumps for debug output, so build correctly it requires these functions to be
# present in LLVM libraries. In LLVM 5.0 they are not there by default and require explicit enabling.
# In later version this functionality is triggered by enabling assertions.
def get_llvm_enable_dump_switch(version_LLVM):
    if version_LLVM in ["3_2", "3_3", "3_4", "3_5", "3_6", "3_7", "3_8", "3_9", "4_0"]:
        return ""
    elif version_LLVM == "5_0":
        return " -DCMAKE_C_FLAGS=-DLLVM_ENABLE_DUMP -DCMAKE_CXX_FLAGS=-DLLVM_ENABLE_DUMP "
    else:
        return " -DLLVM_ENABLE_DUMP=ON "

def build_LLVM(version_LLVM, revision, folder, tarball, debug, selfbuild, extra, from_validation, force, make, gcc_toolchain_path, use_git):
    print_debug("Building LLVM. Version: " + version_LLVM + ". ", from_validation, alloy_build)
    if revision != "":
        print_debug("Revision: " + revision + ".\n", from_validation, alloy_build)
    else:
        print_debug("\n", from_validation, alloy_build)
    # Here we understand what and where do we want to build
    current_path = os.getcwd()
    llvm_home = os.environ["LLVM_HOME"]
    
    make_sure_dir_exists(llvm_home)

    FOLDER_NAME=version_LLVM
    version_LLVM = re.sub('\.', '_', version_LLVM)
    
    os.chdir(llvm_home)
    if revision != "":
        FOLDER_NAME = FOLDER_NAME + "_" + revision
        revision = "-" + revision
    if folder == "":
        folder = FOLDER_NAME
    if debug == True:
        folder = folder + "dbg"
    LLVM_SRC="llvm-" + folder
    LLVM_BUILD="build-" + folder
    LLVM_BIN="bin-" + folder
    if os.path.exists(LLVM_BIN + os.sep + "bin") and not force:
        error("you have folder " + LLVM_BIN + ".\nIf you want to rebuild use --force", 1)
    LLVM_BUILD_selfbuild = LLVM_BUILD + "_temp"
    LLVM_BIN_selfbuild = LLVM_BIN + "_temp"
    common.remove_if_exists(LLVM_SRC)
    common.remove_if_exists(LLVM_BUILD)
    common.remove_if_exists(LLVM_BIN)

    # Starting with MacOS 10.9 Maverics, we depend on Xcode being installed, as it contains C and C++ library headers.
    # sysroot trick below helps finding C headers. For C++ we just check out libc++ sources.
    # Starting macOS 10.12 Sierra we rely on "Command Line Tools for Xcode" to be installed. They are required anyway,
    # and bring C headers to the system. You can install it by running "xcode-select --install".
    # Note that on Sierra there's an issue with using C headers from High Sierra SDK, which instantiates as compile error:
    #     error: 'utimensat' is only available on macOS 10.13 or newer
    # This is due to using SDK targeting OS, which is newer than current one.
    mac_system_root = ""
    if current_OS == "MacOS" \
        and int(current_OS_version.split(".")[0]) >= 13 \
        and int(current_OS_version.split(".")[0]) < 16:
        search_path = string.split(os.environ["PATH"], os.pathsep)
        found_xcrun = False
        for path in search_path:
            if os.path.exists(os.path.join(path, "xcrun")):
                found_xcrun = True
        if found_xcrun:
            mac_system_root = "`xcrun --show-sdk-path`"
        else:
            error("Can't find XCode (xcrun tool) - it's required on MacOS 10.9 and newer", 1)

    if selfbuild:
        common.remove_if_exists(LLVM_BUILD_selfbuild)
        common.remove_if_exists(LLVM_BIN_selfbuild)
    print_debug("Using folders: " + LLVM_SRC + " " + LLVM_BUILD + " " + LLVM_BIN + " in " + 
        llvm_home + "\n", from_validation, alloy_build)
    # load llvm
    if tarball == "":
        checkout_LLVM("llvm", options.use_git, version_LLVM, revision, LLVM_SRC, from_validation)
        os.chdir(LLVM_SRC + "/tools")
        checkout_LLVM("clang", options.use_git, version_LLVM, revision, "clang", from_validation)
        os.chdir("..")
        if current_OS == "MacOS" and int(current_OS_version.split(".")[0]) >= 13:
            # Starting with MacOS 10.9 Maverics, the system doesn't contain headers for standard C++ library and
            # the default library is libc++, bit libstdc++. The headers are part of XCode now. But we are checking out
            # headers as part of LLVM source tree, so they will be installed in clang location and clang will be able
            # to find them. Though they may not match to the library installed in the system, but seems that this should
            # not happen.
            # Note, that we can also build a libc++ library, but it must be on system default location or should be passed
            # to the linker explicitly (either through command line or environment variables). So we are not doing it
            # currently to make the build process easier.
            os.chdir("projects")
            checkout_LLVM("libcxx", options.use_git, version_LLVM, revision, "libcxx", from_validation)
            os.chdir("..")
        if extra == True:
            os.chdir("tools/clang/tools")
            checkout_LLVM("clang-tools-extra", options.use_git, version_LLVM, revision, "extra", from_validation)
            os.chdir("../../../projects")
            checkout_LLVM("compiler-rt", options.use_git, version_LLVM, revision, "compiler-rt", from_validation)
            os.chdir("..")
    else:
        tar = tarball.split(" ")
        os.makedirs(LLVM_SRC) 
        os.chdir(LLVM_SRC) 
        try_do_LLVM("untar LLVM from " + tar[0] + " ",
                    "tar -xvzf " + tar[0] + " --strip-components 1", from_validation)
        os.chdir("./tools") 
        os.makedirs("clang") 
        os.chdir("./clang") 
        try_do_LLVM("untar clang from " + tar[1] + " ",
                    "tar -xvzf " + tar[1] + " --strip-components 1", from_validation)
        os.chdir("../../")
    # paching llvm
    patches = glob.glob(os.environ["ISPC_HOME"] + os.sep + "llvm_patches" + os.sep + "*.*")
    for patch in patches:
        if version_LLVM in os.path.basename(patch):
            if current_OS != "Windows":
                try_do_LLVM("patch LLVM with patch " + patch + " ", "patch -p0 < " + patch, from_validation)
            else:
                try_do_LLVM("patch LLVM with patch " + patch + " ", "C:\\gnuwin32\\bin\\patch.exe -p0 < " + patch, from_validation)
    os.chdir("../")
    # configuring llvm, build first part of selfbuild
    os.makedirs(LLVM_BUILD)
    os.makedirs(LLVM_BIN)
    selfbuild_compiler = ""
    LLVM_configure_capable = ["3_2", "3_3", "3_4", "3_5", "3_6", "3_7"]
    if selfbuild:
        print_debug("Making selfbuild and use folders " + LLVM_BUILD_selfbuild + " and " +
            LLVM_BIN_selfbuild + "\n", from_validation, alloy_build)
        os.makedirs(LLVM_BUILD_selfbuild)
        os.makedirs(LLVM_BIN_selfbuild)
        os.chdir(LLVM_BUILD_selfbuild)
        if  version_LLVM not in LLVM_configure_capable:
            try_do_LLVM("configure release version for selfbuild ",
                    "cmake -G " + "\"" + generator + "\"" + " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" +
                    "  -DCMAKE_INSTALL_PREFIX=" + llvm_home + "/" + LLVM_BIN_selfbuild +
                    "  -DCMAKE_BUILD_TYPE=Release" +
                    get_llvm_enable_dump_switch(version_LLVM) +
                    "  -DLLVM_ENABLE_ASSERTIONS=ON" +
                    "  -DLLVM_INSTALL_UTILS=ON" +
                    (("  -DGCC_INSTALL_PREFIX=" + gcc_toolchain_path) if gcc_toolchain_path != "" else "") +
                    (("  -DCMAKE_C_COMPILER=" + gcc_toolchain_path+"/bin/gcc") if gcc_toolchain_path != "" else "") +
                    (("  -DCMAKE_CXX_COMPILER=" + gcc_toolchain_path+"/bin/g++") if gcc_toolchain_path != "" else "") +
                    (("  -DDEFAULT_SYSROOT=" + mac_system_root) if mac_system_root != "" else "") +
                    "  -DLLVM_TARGETS_TO_BUILD=NVPTX\;X86" +
                    " ../" + LLVM_SRC,
                    from_validation)
            selfbuild_compiler = ("  -DCMAKE_C_COMPILER=" +llvm_home+ "/" + LLVM_BIN_selfbuild + "/bin/clang " +
                                  "  -DCMAKE_CXX_COMPILER="+llvm_home+ "/" + LLVM_BIN_selfbuild + "/bin/clang++ ")
        else:
            try_do_LLVM("configure release version for selfbuild ",
                        "../" + LLVM_SRC + "/configure --prefix=" + llvm_home + "/" +
                        LLVM_BIN_selfbuild + " --enable-optimized" +
                        " --enable-targets=x86,x86_64,nvptx" +
                        ((" --with-gcc-toolchain=" + gcc_toolchain_path) if gcc_toolchain_path != "" else "") +
                        ((" --with-default-sysroot=" + mac_system_root) if mac_system_root != "" else ""),
                        from_validation)
            selfbuild_compiler = ("CC=" +llvm_home+ "/" + LLVM_BIN_selfbuild + "/bin/clang " +
                                  "CXX="+llvm_home+ "/" + LLVM_BIN_selfbuild + "/bin/clang++ ")
        try_do_LLVM("build release version for selfbuild ",
                    make, from_validation)
        try_do_LLVM("install release version for selfbuild ",
                    "make install",
                    from_validation)
        os.chdir("../")

        print_debug("Now we have compiler for selfbuild: " + selfbuild_compiler + "\n", from_validation, alloy_build)
    os.chdir(LLVM_BUILD)
    if debug == False:
        if current_OS != "Windows":
            if  version_LLVM not in LLVM_configure_capable:
                try_do_LLVM("configure release version ",
                        "cmake -G " + "\"" + generator + "\"" + " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" +
                        selfbuild_compiler +
                        "  -DCMAKE_INSTALL_PREFIX=" + llvm_home + "/" + LLVM_BIN +
                        "  -DCMAKE_BUILD_TYPE=Release" +
                        get_llvm_enable_dump_switch(version_LLVM) +
                        "  -DLLVM_ENABLE_ASSERTIONS=ON" +
                        "  -DLLVM_INSTALL_UTILS=ON" +
                        (("  -DGCC_INSTALL_PREFIX=" + gcc_toolchain_path) if gcc_toolchain_path != "" else "") +
                        (("  -DCMAKE_C_COMPILER=" + gcc_toolchain_path+"/bin/gcc") if gcc_toolchain_path != "" and selfbuild_compiler == "" else "") +
                        (("  -DCMAKE_CXX_COMPILER=" + gcc_toolchain_path+"/bin/g++") if gcc_toolchain_path != "" and selfbuild_compiler == "" else "") +
                        (("  -DDEFAULT_SYSROOT=" + mac_system_root) if mac_system_root != "" else "") +
                        "  -DLLVM_TARGETS_TO_BUILD=NVPTX\;X86" +
                        " ../" + LLVM_SRC,
                        from_validation)
            else:
                try_do_LLVM("configure release version ",
                        selfbuild_compiler + "../" + LLVM_SRC + "/configure --prefix=" + llvm_home + "/" +
                        LLVM_BIN + " --enable-optimized" +
                        " --enable-targets=x86,x86_64,nvptx" +
                        ((" --with-gcc-toolchain=" + gcc_toolchain_path) if gcc_toolchain_path != "" else "") +
                        ((" --with-default-sysroot=" + mac_system_root) if mac_system_root != "" else ""),
                        from_validation)
        else:
            try_do_LLVM("configure release version ",
                    'cmake -G ' + '\"' + generator + '\"' + ' -DCMAKE_INSTALL_PREFIX="..\\'+ LLVM_BIN + '" ' +
                    '  -DCMAKE_BUILD_TYPE=Release' +
                    get_llvm_enable_dump_switch(version_LLVM) +
                    '  -DLLVM_ENABLE_ASSERTIONS=ON' +
                    '  -DLLVM_INSTALL_UTILS=ON' +
                    '  -DLLVM_TARGETS_TO_BUILD=X86' +
                    '  -DLLVM_LIT_TOOLS_DIR="C:\\gnuwin32\\bin" ..\\' + LLVM_SRC,
                    from_validation)
    else:
        if  version_LLVM not in LLVM_configure_capable:
            try_do_LLVM("configure debug version ",
                    "cmake -G " + "\"" + generator + "\"" + " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" +
                    selfbuild_compiler +
                    "  -DCMAKE_INSTALL_PREFIX=" + llvm_home + "/" + LLVM_BIN +
                    "  -DCMAKE_BUILD_TYPE=Debug" +
                    get_llvm_enable_dump_switch(version_LLVM) +
                    "  -DLLVM_ENABLE_ASSERTIONS=ON" +
                    "  -DLLVM_INSTALL_UTILS=ON" +
                    (("  -DGCC_INSTALL_PREFIX=" + gcc_toolchain_path) if gcc_toolchain_path != "" else "") +
                    (("  -DCMAKE_C_COMPILER=" + gcc_toolchain_path+"/bin/gcc") if gcc_toolchain_path != "" and selfbuild_compiler == "" else "") +
                    (("  -DCMAKE_CXX_COMPILER=" + gcc_toolchain_path+"/bin/g++") if gcc_toolchain_path != "" and selfbuild_compiler == "" else "") +
                    (("  -DDEFAULT_SYSROOT=" + mac_system_root) if mac_system_root != "" else "") +
                    "  -DLLVM_TARGETS_TO_BUILD=NVPTX\;X86" +
                    " ../" + LLVM_SRC,
                    from_validation)
        else:
            try_do_LLVM("configure debug version ",
                        selfbuild_compiler + "../" + LLVM_SRC + "/configure --prefix=" + llvm_home + "/" + LLVM_BIN +
                        " --enable-debug-runtime --enable-debug-symbols --enable-keep-symbols" +
                        " --enable-targets=x86,x86_64,nvptx" +
                        ((" --with-gcc-toolchain=" + gcc_toolchain_path) if gcc_toolchain_path != "" else "") +
                        ((" --with-default-sysroot=" + mac_system_root) if mac_system_root != "" else ""),
                        from_validation)
    # building llvm
    if current_OS != "Windows":
        try_do_LLVM("build LLVM ", make, from_validation)
        try_do_LLVM("install LLVM ", "make install", from_validation)
    else:
        try_do_LLVM("build LLVM and than install LLVM ", "msbuild INSTALL.vcxproj /V:m /p:Platform=Win32 /p:Configuration=Release /t:rebuild", from_validation)
    os.chdir(current_path) 


def unsupported_llvm_targets(LLVM_VERSION):
    prohibited_list = {"3.2":["avx512knl-i32x16", "avx512skx-i32x16", "avx512skx-i32x8"],
                       "3.3":["avx512knl-i32x16", "avx512skx-i32x16", "avx512skx-i32x8"],
                       "3.4":["avx512knl-i32x16", "avx512skx-i32x16", "avx512skx-i32x8"],
                       "3.5":["avx512knl-i32x16", "avx512skx-i32x16", "avx512skx-i32x8"],
                       "3.6":["avx512knl-i32x16", "avx512skx-i32x16", "avx512skx-i32x8"],
                       "3.7":["avx512skx-i32x16", "avx512skx-i32x8"],
                       "3.8":["avx512skx-i32x8"],
                       "3.9":["avx512skx-i32x8"],
                       "4.0":["avx512skx-i32x8"],
                       "5.0":["avx512skx-i32x8"],
                       "6.0":["avx512skx-i32x8"],
                       "7.0":["avx512skx-i32x8"]}
    if LLVM_VERSION in prohibited_list:
        return prohibited_list[LLVM_VERSION]
    return []


# Split targets into categories: native, generic, knc, sde.
# native - native targets run natively on current hardware.
# generic - hardware agnostic generic target.
# knc - knc target. This one is special, as it requires additional steps to run.
# sde - native target, which need to be emulated on current hardware.
def check_targets():
    result = []
    result_generic = []
    result_knc = []
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
      ("AVX1.1", [["avx1.1-i32x8","avx1.1-i32x16","avx1.1-i64x4"],
                 ["SSE2", "SSE4", "AVX", "AVX1.1"], "-ivb", False]),
      ("AVX2",   [["avx2-i32x8",  "avx2-i32x16",  "avx2-i64x4"],
                 ["SSE2", "SSE4", "AVX", "AVX1.1", "AVX2"], "-hsw", False]),
      ("KNL",    [["avx512knl-i32x16"],
                 ["SSE2", "SSE4", "AVX", "AVX1.1", "AVX2", "KNL"], "-knl", False]),
      ("SKX",    [["avx512skx-i32x16", "avx512skx-i32x8"],
                 ["SSE2", "SSE4", "AVX", "AVX1.1", "AVX2", "SKX"], "-skx", False])
    ])

    hw_arch = take_lines("check_isa.exe", "first").split()[1]

    if not (hw_arch in target_dict):
        error("Architecture " + hw_arch + " was not recognized", 1)

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

    # generate targets for KNC
    if  current_OS == "Linux":
        result_knc = ["knc-generic"]

    if current_OS != "Windows":
        result_generic = ["generic-4", "generic-16", "generic-8", "generic-1", "generic-32", "generic-64"]

    # now check what targets we have with the help of SDE
    sde_exists = get_sde()
    if sde_exists == "":
        error("you haven't got sde neither in SDE_HOME nor in your PATH.\n" + 
            "To test all platforms please set SDE_HOME to path containing SDE.\n" +
            "Please refer to http://www.intel.com/software/sde for SDE download information.", 2)

    return [result, result_generic, result_sde, result_knc]

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

        llvm_rev = ""
        # determine LLVM revision
        p = subprocess.Popen("svn info " + folder, shell=True, \
               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (info_llvm, err) = p.communicate()
        info_llvm = re.split('\n', info_llvm)
        for i in info_llvm:
            if len(i) > 0 and i.startswith("Last Changed Rev: "):
                llvm_rev = str(i[len("Last Changed Rev: "):])

        if llvm_rev != "":
            common.ex_state.switch_revision(llvm_rev)
            print_debug("\nBuilding ISPC with LLVM %s (%s):\n" \
                                              % (version_LLVM, llvm_rev), False, stability_log)
        else:
            print_debug("Unable to retrieve LLVM revision\n", False, stability_log)
            raise

        try_do_LLVM("recognize LLVM revision", "svn info " + folder, True)
        try_do_LLVM("configure ispc build", 'cmake -DCMAKE_INSTALL_PREFIX="..\\'+ ISPC_BIN + '" ' +
                    '  -DCMAKE_BUILD_TYPE=Release' +
                        ispc_home, True)
        try_do_LLVM("build ISPC with LLVM version " + version_LLVM + " ", make_ispc, True)
        try_do_LLVM("install ISPC ", "make install", True)
        copyfile(os.path.join(ispc_home, ISPC_BIN, "bin", "ispc"), os.path.join(ispc_home, + "ispc"))
        os.environ["PATH"] = p_temp
    else:
        try_do_LLVM("configure ispc build", 'cmake -G ' + '\"' + generator + '\"' + ' -DCMAKE_INSTALL_PREFIX="..\\'+ ISPC_BIN + '" ' +
                    '  -DCMAKE_BUILD_TYPE=Release ' +
                        ispc_home, True)
        try_do_LLVM("clean ISPC for building", "msbuild ispc.vcxproj /t:clean", True)
        try_do_LLVM("build ISPC with LLVM version " + version_LLVM + " ", "msbuild ispc.vcxproj /V:m /p:Platform=Win32 /p:Configuration=Release /t:rebuild", True)
        try_do_LLVM("install ISPC  ", "msbuild INSTALL.vcxproj /p:Platform=Win32 /p:Configuration=Release", True)
        copyfile(os.path.join(ispc_home, ISPC_BIN, "bin", "ispc.exe"), os.path.join(ispc_home, + "ispc.exe"))

    os.chdir(current_path)

def execute_stability(stability, R, print_version):
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

def validation_run(only, only_targets, reference_branch, number, notify, update, speed_number, make, perf_llvm, time):
    os.chdir(os.environ["ISPC_HOME"])
    if current_OS != "Windows":
        os.environ["PATH"] = os.environ["ISPC_HOME"] + ":" + os.environ["PATH"]
    if options.notify != "":
        common.remove_if_exists(os.environ["ISPC_HOME"] + os.sep + "notify_log.log")
        msg = MIMEMultipart()
    print_debug("Command: " + ' '.join(sys.argv) + "\n", False, "")
    print_debug("Folder: " + os.environ["ISPC_HOME"] + "\n", False, "")
    date = datetime.datetime.now()
    print_debug("Date: " + date.strftime('%H:%M %d/%m/%Y') + "\n", False, "")
    newest_LLVM="3.6"
    msg_additional_info = ""
# *** *** ***
# Stability validation run
# *** *** ***
    if ((("stability" in only) == True) or ("performance" in only) == False):
        print_debug("\n\nStability validation run\n\n", False, "")
        stability = common.EmptyClass()
# stability constant options
        stability.save_bin = False
        stability.random = False
        stability.ispc_flags = ""
        stability.compiler_exe = None
        stability.num_jobs = speed_number
        stability.verbose = False
        stability.time = time
        stability.non_interactive = True
        stability.update = update
        stability.include_file = None
        stability.silent = True
        stability.in_file = "." + os.sep + f_date + os.sep + "run_tests_log.log"
        stability.verify = False
# stability varying options
        stability.target = ""
        stability.arch = ""
        stability.no_opt = False
        stability.wrapexe = ""
# prepare parameters of run
        [targets_t, targets_generic_t, sde_targets_t, targets_knc_t] = check_targets()
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
            opts.append(False)
        if "-O0" in only:
            opts.append(True)
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
        for i in ["3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "4.0", "5.0", "6.0", "7.0", "8.0", "trunk"]:
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
            if "generic" in only_targets_t:
                only_targets_t.append("generic-4")
                only_targets_t.append("generic-16")
            while "generic" in only_targets_t:
                only_targets_t.remove("generic")
            
            for i in only_targets_t:
                if i == "":
                    continue
                err = True
                for j in range(0,len(targets_t)):
                    if i in targets_t[j]:
                        targets.append(targets_t[j])
                        err = False
                for j in range(0,len(targets_generic_t)):
                    if i in targets_generic_t[j]:
                        targets.append(targets_generic_t[j])
                        err = False
                for j in range(0,len(sde_targets_t)):
                    if i in sde_targets_t[j][1]:
                        sde_targets.append(sde_targets_t[j])
                        err = False
                for j in range(0,len(targets_knc_t)):
                    if i in targets_knc_t[j]:
                        targets.append(targets_knc_t[j])
                        err = False
                if err == True:
                    error("You haven't sde for target " + i, 1)
        else:
            targets = targets_t + targets_generic_t[:-4]
            sde_targets = sde_targets_t

        if "build" in only:
            targets = []
            sde_targets = []
            only = only + " stability "
# finish parameters of run, prepare LLVM
        if len(opts) == 0:
            opts = [False]
        if len(archs) == 0:
            archs = ["x86", "x86-64"]
        if len(LLVM) == 0:
            LLVM = [newest_LLVM, "trunk"]
        gen_archs = ["x86-64"]
        need_LLVM = check_LLVM(LLVM)
        for i in range(0,len(need_LLVM)):
            build_LLVM(need_LLVM[i], "", "", "", False, False, False, True, False, make, options.gcc_toolchain_path, False)
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

                # *always* specify default values for global variables on each loop iteration
                stability.wrapexe = ""
                stability.compiler_exe = None
                # choosing right compiler for a given target
                # sometimes clang++ is not avaluable. if --ispc-build-compiler = gcc we will pass in g++ compiler
                if options.ispc_build_compiler == "gcc":
                    stability.compiler_exe = "g++"
                # but 'knc/knl' generic target is supported only by icpc, so set explicitly
                if ("knc-generic" in stability.target):
                    stability.compiler_exe = "icpc"
                # now set archs for targets
                if ("generic" in stability.target):
                    arch = gen_archs
                else:
                    arch = archs
                for i1 in range(0,len(arch)):
                    for i2 in range(0,len(opts)):
                        for i3 in range(dbg_begin,dbg_total):
                            stability.arch = arch[i1]
                            stability.no_opt = opts[i2]
                            stability.ispc_flags = ispc_flags_tmp
                            if (i3 != 0):
                                stability.ispc_flags += " -g"
                            try:
                                execute_stability(stability, R_tmp, print_version)
                            except:
                                print_debug("ERROR: Exception in execute_stability - maybe some test subprocess terminated before it should have\n", False, stability_log)
                            print_version = 0
            for j in range(0,len(sde_targets)):
                stability.target = sde_targets[j][1]
                # the target might be not supported by the chosen llvm version
                if (stability.target in unsupported_llvm_targets(LLVM[i])):
                    print_debug("Warning: target " + stability.target + " is not supported in LLVM " + LLVM[i] + "\n", False, stability_log)
                    continue
                # *always* specify default values for global variables on each loop iteration
                stability.wrapexe = ""
                stability.compiler_exe = None
                # choosing right compiler for a given target
                # sometimes clang++ is not avaluable. if --ispc-build-compiler = gcc we will pass in g++ compiler
                if options.ispc_build_compiler == "gcc":
                    stability.compiler_exe = "g++"
                if ("knc-generic" in stability.target):
                    stability.compiler_exe = "icpc"
                stability.wrapexe = get_sde() + " " + sde_targets[j][0] + " -- "
                if ("generic" in stability.target):
                    arch = gen_archs
                else:
                    arch = archs
                for i1 in range(0,len(arch)):
                    for i2 in range(0,len(opts)):
                        for i3 in range(dbg_begin,dbg_total):
                            stability.arch = arch[i1]
                            stability.no_opt = opts[i2]
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
        if options.notify != "":
            # e-mail header for performance test:
            msg_additional_info += "New runfails(%d) New compfails(%d) New passes runfails(%d) New passes compfails(%d)" \
                                                                      % (len(R[0][0]), len(R[1][0]), len(R[2][0]), len(R[3][0]))
            attach_mail_file(msg, stability.in_file, "run_tests_log.log", 100)
            attach_mail_file(msg, stability_log, "stability.log")

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
            build_LLVM(need_LLVM[0], "", "", "", False, False, False, True, False, make, options.gcc_toolchain_path, options.use_git)
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
                error("you haven't got llvm called " + reference_branch, 1)
            build_ispc(newest_LLVM, make)
            os.rename("ispc", "ispc_ref")
            build_ispc(reference_branch, make)
        # begin validation run for performance. output is inserted into perf()
        perf.perf(performance, [])
        if options.notify != "":
            attach_mail_file(msg, performance.in_file, "performance.log")
            attach_mail_file(msg, "." + os.sep + "logs" + os.sep + "perf_build.log", "perf_build.log")
    # dumping gathered info to the file
    common.ex_state.dump(alloy_folder + "test_table.dump", common.ex_state.tt)

    # sending e-mail with results
    if options.notify != "":
        send_mail(msg_additional_info, msg)

def send_mail(body_header, msg):
    try:
        fp = open(os.environ["ISPC_HOME"] + os.sep + "notify_log.log", 'rb')
        f_lines = fp.readlines()
        fp.close()
    except:
        body_header += "\nUnable to open notify_log.log: " + str(sys.exc_info()) + "\n"
        print_debug("Unable to open notify_log.log: " + str(sys.exc_info()) + "\n", False, stability_log)
    
    body = "Hostname: " + common.get_host_name() + "\n\n"

    if  not sys.exc_info()[0] == None:
        body += "ERROR: Exception(last) - " + str(sys.exc_info()) + '\n'

    body += body_header + '\n'
    for i in range(0, len(f_lines)):
        body += f_lines[i][:-1]
        body += '   \n'

    attach_mail_file(msg, alloy_build, "alloy_build.log", 100) # build.log is always being sent
    smtp_server = os.environ["SMTP_ISPC"]
    msg['Subject'] = options.notify_subject
    msg['From'] = "ISPC_test_system"
    msg['To'] = options.notify
    text = MIMEText(body, "", "KOI-8")
    msg.attach(text)
    s = smtplib.SMTP(smtp_server)
    s.sendmail(options.notify, options.notify.split(" "), msg.as_string())
    s.quit()

def Main():
    global current_OS
    global current_OS_version
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
        exit(0)

    # set appropriate makefile target
    # gcc and g++ options are equal and added for ease of use 
    if options.ispc_build_compiler != "clang" and \
       options.ispc_build_compiler != "gcc":   
        error("unknow option for --ispc-build-compiler: " + options.ispc_build_compiler, 1)
        parser.print_help()
        exit(0)

    if options.notify != "":
        # in case 'notify' option is used but build (in '-b' for example) failed we do not want to have trash in our message body
        # NOTE! 'notify.log' must also be cleaned up at the beginning of every message sending function, i.e. in 'validation_run()'
        common.remove_if_exists(os.environ["ISPC_HOME"] + os.sep + "notify_log.log")

    setting_paths(options.llvm_home, options.ispc_home, options.sde_home)
    if os.environ.get("LLVM_HOME") == None:
        error("you have no LLVM_HOME", 1)
    if os.environ.get("ISPC_HOME") == None:
        error("you have no ISPC_HOME", 1)
    if options.notify != "":
        if os.environ.get("SMTP_ISPC") == None:
            error("you have no SMTP_ISPC in your environment for option notify", 1)
    if options.only != "":
        test_only_r = " 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 5.0 6.0 7.0 8.0 trunk current build stability performance x86 x86-64 x86_64 -O0 -O2 native debug nodebug "
        test_only = options.only.split(" ")
        for iterator in test_only:
            if not (" " + iterator + " " in test_only_r):
                error("unknown option for only: " + iterator, 1)
    if current_OS == "Windows":
        if options.debug == True or options.selfbuild == True or options.tarball != "":
            error("Debug, selfbuild and tarball options are unsupported on windows", 1)
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
        error("you ISPC_HOME and your current path are different! (" + os.environ["ISPC_HOME"] + " is not equal to " + os.getcwd() +
        ")\n", 2)
    if options.perf_llvm == True:
        if options.branch == "master":
            options.branch = "trunk"
    if options.use_git and options.revision != "":
        error("--revision is not supported with --git", 1)
    global generator
    if options.generator:
        generator = options.generator
    else:
        if current_OS == "Windows":
            generator = "Visual Studio 14"
        else:
            generator = "Unix Makefiles"
    try:
        start_time = time.time()
        if options.build_llvm:
            build_LLVM(options.version, options.revision, options.folder, options.tarball,
                    options.debug, options.selfbuild, options.extra, False, options.force, make, options.gcc_toolchain_path, options.use_git)
        if options.validation_run:
            validation_run(options.only, options.only_targets, options.branch,
                    options.number_for_performance, options.notify, options.update, int(options.speed),
                    make, options.perf_llvm, options.time)
        elapsed_time = time.time() - start_time
        if options.time:
            print_debug("Elapsed time: " + time.strftime('%Hh%Mm%Ssec.', time.gmtime(elapsed_time)) + "\n", False, "")
    finally:
        os.chdir(current_path)
        date_name = "alloy_results_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if os.path.exists(date_name):
            error("It's forbidden to run alloy two times in a second, logs are in ./logs", 1)
        os.rename(f_date, date_name)
        print_debug("Logs are in " + date_name + "\n", False, "")
        exit(0)

###Main###
from optparse import OptionParser
from optparse import OptionGroup
import sys
import os
import errno
import operator
import time
import glob
import string
import platform
import smtplib
import datetime
import copy
import multiprocessing
import subprocess
import re
from shutil import copyfile
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.mime.text import MIMEText
from email.Encoders import encode_base64
# our drivers
import run_tests
import perf
import common
error = common.error
take_lines = common.take_lines
print_debug = common.print_debug
make_sure_dir_exists = common.make_sure_dir_exists
if __name__ == '__main__':
    # parsing options
    class MyParser(OptionParser):
        def format_epilog(self, formatter):
            return self.epilog
    examples =  ("Examples:\n" +
    "Load and build LLVM from trunk\n\talloy.py -b\n" +
    "Load and build LLVM 3.3. Rewrite LLVM folders\n\talloy.py -b --version=3.3 --force\n" +
    "Untar files llvm.tgz clang.tgz, build LLVM from them in folder bin-from_tar\n\talloy.py -b --tarball='llvm.tgz clang.tgz' --folder=from_tar\n" +
    "Load LLVM from trunk, revision r172870. Build it. Do selfbuild\n\talloy.py -b --revision=r172870 --selfbuild\n" +
    "Validation run with LLVM 3.3, trunk; x86, x86-64; -O2;\nall supported targets; performance\n\talloy.py -r\n" + 
    "Validation run with all avx targets and sse4-i8x16 without performance\n\talloy.py -r --only=stability --only-targets='avx sse4-i8x16'\n" +
    "Validation run with avx2-i32x8, all sse4 and sse2 targets\nand all targets with i32x16\n\talloy.py -r --only-targets='avx2-i32x8 sse4 i32x16 sse2'\n" +
    "Stability validation run with LLVM 3.2, 3.3; -O0; x86,\nupdate fail_db.txt with passes and fails\n\talloy.py -r --only='3.2 -O0 stability 3.3 x86' --update-errors=FP\n" +
    "Try to build compiler with all LLVM\n\talloy.py -r --only=build\n" +
    "Performance validation run with 10 runs of each test and comparing to branch 'old'\n\talloy.py -r --only=performance --compare-with=old --number=10\n" +
    "Validation run. Update fail_db.txt with new fails, send results to my@my.com\n\talloy.py -r --update-errors=F --notify='my@my.com'\n" +
    "Test KNC target (not tested when tested all supported targets, so should be set explicitly via --only-targets)\n\talloy.py -r --only='stability' --only-targets='knc-generic'\n" +
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
        help='version of llvm to build: 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 5.0 6.0 7.0 8.0 trunk. Default: trunk', default="trunk")
    llvm_group.add_option('--with-gcc-toolchain', dest='gcc_toolchain_path',
         help='GCC install dir to use when building clang. It is important to set when ' +
         'you have alternative gcc installation. Note that otherwise gcc from standard ' +
         'location will be used, not from your PATH', default="")
    llvm_group.add_option('--revision', dest='revision',
        help='revision of llvm to build in format r172870 (not supported with --git)', default="")
    llvm_group.add_option('--debug', dest='debug',
        help='debug build of LLVM?', default=False, action="store_true")
    llvm_group.add_option('--folder', dest='folder',
        help='folder to build LLVM in', default="")
    llvm_group.add_option('--tarball', dest='tarball',
        help='"llvm_tarball clang_tarball"', default="")
    llvm_group.add_option('--selfbuild', dest='selfbuild',
        help='make selfbuild of LLVM and clang', default=False, action="store_true")
    llvm_group.add_option('--force', dest='force',
        help='rebuild LLVM', default=False, action='store_true')
    llvm_group.add_option('--extra', dest='extra',
        help='load extra clang tools', default=False, action='store_true')
    llvm_group.add_option('--git', dest='use_git',
        help='use git llvm repository instead of svn', default=False, action='store_true')
    parser.add_option_group(llvm_group)
    # options for activity "validation run"
    run_group = OptionGroup(parser, "Options for validation run",
                    "These options must be used with -r option.")
    run_group.add_option('--compare-with', dest='branch',
        help='set performance reference point. Dafault: master', default="master")
    run_group.add_option('--number', dest='number_for_performance',
        help='number of performance runs for each test. Default: 5', default=5)
    run_group.add_option('--notify', dest='notify',
        help='email to sent results to', default="")
    run_group.add_option('--notify-subject', dest='notify_subject',
        help='set the subject of the notification email, the default is ISPC test system results', default="ISPC test system results")
    run_group.add_option('--update-errors', dest='update',
        help='rewrite fail_db.txt file according to received results (F or FP)', default="")
    run_group.add_option('--only-targets', dest='only_targets',
        help='set list of targets to test. Possible values - all subnames of targets, plus "knc-generic" for "generic" ' +
             'version of knc support', default="")
    run_group.add_option('--time', dest='time',
        help='display time of testing', default=False, action='store_true')
    run_group.add_option('--only', dest='only',
        help='set types of tests. Possible values:\n' + 
            '-O0, -O2, x86, x86-64, stability (test only stability), performance (test only performance),\n' +
            'build (only build with different LLVM), 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 5.0, 6.0, 7.0, 8.0, trunk, native (do not use SDE),\n' +
            'current (do not rebuild ISPC), debug (only with debug info), nodebug (only without debug info, default).',
            default="")
    run_group.add_option('--perf_LLVM', dest='perf_llvm',
        help='compare LLVM 3.6 with "--compare-with", default trunk', default=False, action='store_true')
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
