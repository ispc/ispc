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

# // Author: Filippov Ilia

def attach_mail_file(msg, filename, name):
    if os.path.exists(filename):
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
    if from_validation == True:
        text = text + "\n"
    print_debug("Trying to " + text, from_validation, alloy_build)
    if os.system(command + " >> " + alloy_build + " 2>> " + alloy_build) != 0:
        print_debug("ERROR.\n", from_validation, alloy_build)
        error("can't " + text, 1)
    print_debug("DONE.\n", from_validation, alloy_build)

def build_LLVM(version_LLVM, revision, folder, tarball, debug, selfbuild, from_validation, force):
    print_debug("Building LLVM. Version: " + version_LLVM + ". ", from_validation, alloy_build)
    if revision != "":
        print_debug("Revision: " + revision + ".\n", from_validation, alloy_build)
    else:
        print_debug("\n", from_validation, alloy_build)
    # Here we understand what and where do we want to build
    current_path = os.getcwd()
    llvm_home = os.environ["LLVM_HOME"]
    os.chdir(llvm_home)
    FOLDER_NAME=version_LLVM
    if  version_LLVM == "trunk":
        SVN_PATH="trunk"
    if  version_LLVM == "3.3":
        SVN_PATH="tags/RELEASE_33/final"
        version_LLVM = "3_3"
    if  version_LLVM == "3.2":
        SVN_PATH="tags/RELEASE_32/final"
        version_LLVM = "3_2"
    if  version_LLVM == "3.1":
        SVN_PATH="tags/RELEASE_31/final"
        version_LLVM = "3_1"
    if revision != "":
        FOLDER_NAME = FOLDER_NAME + "_" + revision
        revision = "-" + revision
    if folder == "":
        folder = FOLDER_NAME
    LLVM_SRC="llvm-" + folder
    LLVM_BUILD="build-" + folder
    LLVM_BIN="bin-" + folder
    if os.path.exists(LLVM_BIN) and not force:
        print_debug("You have folder " + LLVM_BIN + ". If you want to rebuild use --force\n", False, "")
        exit(0)
    LLVM_BUILD_selfbuild = LLVM_BUILD + "_temp"
    LLVM_BIN_selfbuild = LLVM_BIN + "_temp"
    common.remove_if_exists(LLVM_SRC)
    common.remove_if_exists(LLVM_BUILD)
    common.remove_if_exists(LLVM_BIN)
    if selfbuild:
        common.remove_if_exists(LLVM_BUILD_selfbuild)
        common.remove_if_exists(LLVM_BIN_selfbuild)
    MAKE = "gmake"
    print_debug("Using folders: " + LLVM_SRC + " " + LLVM_BUILD + " " + LLVM_BIN + " in " + 
        llvm_home + "\n", from_validation, alloy_build)
    # load llvm
    if tarball == "":
        try_do_LLVM("load LLVM from http://llvm.org/svn/llvm-project/llvm/" + SVN_PATH + " ",
                    "svn co " + revision + " http://llvm.org/svn/llvm-project/llvm/" + SVN_PATH + " " + LLVM_SRC,
                    from_validation)
        os.chdir(LLVM_SRC + "/tools")
        try_do_LLVM("load clang from http://llvm.org/svn/llvm-project/cfe/" + SVN_PATH + " ",
                    "svn co " + revision + " http://llvm.org/svn/llvm-project/cfe/" + SVN_PATH + " clang",
                    from_validation)
        os.chdir("../")
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
    patches = glob.glob(os.environ["ISPC_HOME"] + "/llvm_patches/*.*")
    for patch in patches:
        if version_LLVM in os.path.basename(patch):
            try_do_LLVM("patch LLVM with patch" + patch + " ", "patch -p0 < " + patch, from_validation)
    os.chdir("../")
    # configuring llvm, build first part of selfbuild
    os.makedirs(LLVM_BUILD)
    os.makedirs(LLVM_BIN)
    selfbuild_compiler = ""
    if selfbuild:
        print_debug("Making selfbuild and use folders " + LLVM_BUILD_selfbuild + " and " +
            LLVM_BIN_selfbuild + "\n", from_validation, alloy_build)
        os.makedirs(LLVM_BUILD_selfbuild)
        os.makedirs(LLVM_BIN_selfbuild)
        os.chdir(LLVM_BUILD_selfbuild)
        try_do_LLVM("configure release version for selfbuild ",
                    "../" + LLVM_SRC + "/configure --prefix=" + llvm_home + "/" +
                    LLVM_BIN_selfbuild + " --enable-optimized",
                    from_validation)
        try_do_LLVM("build release version for selfbuild ",
                    MAKE + " -j32", from_validation)
        try_do_LLVM("install release version for selfbuild ",
                    MAKE + " install",
                    from_validation)
        os.chdir("../")
        selfbuild_compiler = " CC="+llvm_home+ "/" + LLVM_BIN_selfbuild + "/bin/clang"
        print_debug("Now we have compiler for selfbuild: " + selfbuild_compiler + "\n", from_validation, alloy_build)
    os.chdir(LLVM_BUILD)
    if debug == False:
        try_do_LLVM("configure release version ",
                    "../" + LLVM_SRC + "/configure --prefix=" + llvm_home + "/" +
                    LLVM_BIN + " --enable-optimized" + selfbuild_compiler,
                    from_validation)
    else:
        try_do_LLVM("configure debug version ",
                    "../" + LLVM_SRC + "/configure --prefix=" + llvm_home + "/" + LLVM_BIN +
                    " --enable-debug-runtime --enable-debug-symbols --enable-keep-symbols" + selfbuild_compiler,
                    from_validation)
    # building llvm
    try_do_LLVM("build LLVM ", MAKE + " -j32", from_validation)
    try_do_LLVM("install LLVM ", MAKE + " install", from_validation)
    os.chdir(current_path) 

def check_targets():
    answer = []
    answer_sde = []
    SSE2 = False;
    SSE4 = False;
    AVX = False;
    AVX11 = False;
    AVX2 = False;
    cpu = open("/proc/cpuinfo")
    f_lines = cpu.readlines()
    cpu.close()
    # check what native targets do we have
    for i in range(0,len(f_lines)):
        if SSE2 == False and "sse2" in f_lines[i]:
            SSE2 = True;
            answer = answer + ["sse2-i32x4", "sse2-i32x8"]
        if SSE4 == False and "sse4_1" in f_lines[i]:
            SSE4 = True;
            answer = answer + ["sse4-i32x4", "sse4-i32x8", "sse4-i16x8", "sse4-i8x16"]
        if AVX == False and "avx" in f_lines[i]:
            AVX = True;
            answer = answer + ["avx1-i32x8", "avx1-i32x16"]
        if AVX11 == False and "rdrand" in f_lines[i]:
            AVX11 = True;
            answer = answer + ["avx1.1-i32x8", "avx1.1-i32x16"]
        if AVX2 == False and "avx2" in f_lines[i]:
            AVX2 = True;
            answer = answer + ["avx2-i32x8", "avx2-i32x16"]
    answer = answer + ["generic-4", "generic-16", "generic-8", "generic-1", "generic-32", "generic-64"]
    # now check what targets we have with the help of SDE
    sde_exists = ""
    PATH_dir = string.split(os.getenv("PATH"), os.pathsep)
    for counter in PATH_dir:
        if os.path.exists(counter + os.sep + "sde") and sde_exists == "":
            sde_exists = counter + os.sep + "sde"
    if os.environ.get("SDE_HOME") != None:
        if os.path.exists(os.environ.get("SDE_HOME") + os.sep + "sde"):
            sde_exists = os.environ.get("SDE_HOME") + os.sep + "sde"
    if sde_exists == "":
        error("you haven't got sde neither in SDE_HOME nor in your PATH.\n" + 
            "To test all platforms please set SDE_HOME to path containing SDE.\n" +
            "Please refer to http://www.intel.com/software/sde for SDE download information.", 2)
        return [answer, answer_sde]
    # here we have SDE
    os.system(sde_exists + " -help > " + temp_alloy_file)
    cpu = open(temp_alloy_file)
    f_lines = cpu.readlines()
    cpu.close()
    for i in range(0,len(f_lines)):
        if SSE4 == False and "wsm" in f_lines[i]:
            answer_sde = answer_sde + [["-wsm", "sse4-i32x4"], ["-wsm", "sse4-i32x8"], ["-wsm", "sse4-i16x8"], ["-wsm", "sse4-i8x16"]]
        if AVX == False and "snb" in f_lines[i]:
            answer_sde = answer_sde + [["-snb", "avx1-i32x8"], ["-snb", "avx1-i32x16"]]
        if AVX11 == False and "ivb" in f_lines[i]:
            answer_sde = answer_sde + [["-ivb", "avx1.1-i32x8"], ["ivb", "avx1.1-i32x16"]]
        if AVX2 == False and "hsw" in f_lines[i]:
            answer_sde = answer_sde + [["-hsw", "avx2-i32x8"], ["-hsw", "avx2-i32x16"]]
    return [answer, answer_sde]

def build_ispc(version_LLVM):
    current_path = os.getcwd()
    os.chdir(os.environ["ISPC_HOME"])
    p_temp = os.getenv("PATH")
    os.environ["PATH"] = os.environ["LLVM_HOME"] + "/bin-" + version_LLVM + "/bin:" + os.environ["PATH"]
    os.system("make clean >> " + alloy_build)
    try_do_LLVM("build ISPC with LLVM version " + version_LLVM + " ", "make -j32", True)
    os.environ["PATH"] = p_temp
    os.chdir(current_path)

def execute_stability(stability, R, print_version):
    stability1 = copy.deepcopy(stability)
    temp = run_tests.run_tests(stability1, [], print_version)
    for j in range(0,4):
        R[j][0] = R[j][0] + temp[j]
        for i in range(0,len(temp[j])):
            R[j][1].append(temp[4])
    number_of_fails = temp[5]
    number_of_new_fails = len(temp[0]) + len(temp[1])
    if number_of_fails == 0:
        str_fails = ". No fails"
    else:
        str_fails = ". Fails: " + str(number_of_fails)
    if number_of_new_fails == 0:
        str_new_fails = ", No new fails.\n"
    else:
        str_new_fails = ", New fails: " + str(number_of_new_fails) + ".\n"
    print_debug(temp[4][1:-3] + str_fails + str_new_fails, False, stability_log)

def run_special_tests():
   i = 5 

def validation_run(only, only_targets, reference_branch, notify, update):
    current_path = os.getcwd()
    os.chdir(os.environ["ISPC_HOME"])
    os.environ["PATH"] = os.environ["ISPC_HOME"] + ":" + os.environ["PATH"]
    if options.notify != "":
        if os.environ.get("SMTP_ISPC") == None:
            error("you have no SMTP_ISPC in your environment for option notify", 1)
        common.remove_if_exists(os.environ["ISPC_HOME"] + os.sep + "all_answer.txt")
        smtp_server = os.environ["SMTP_ISPC"]
        msg = MIMEMultipart()
        msg['Subject'] = 'ISPC test system results'
        msg['From'] = 'ISPC_test_system'
        msg['To'] = options.notify
    print_debug("Command: " + ' '.join(sys.argv) + "\n", False, "")
    print_debug("Folder: " + os.environ["ISPC_HOME"] + "\n", False, "")
    date = datetime.datetime.now()
    print_debug("Date: " + date.strftime('%H:%M %d/%m/%Y') + "\n", False, "")
    class options_for_drivers:
        pass
# *** *** ***
# Stability validation run
# *** *** ***
    if ((("stability" in only) == True) or ("performance" in only) == False):
        print_debug("\n\nStability validation run\n\n", False, "")
        stability = options_for_drivers()
# stability constant options
        stability.random = False
        stability.ispc_flags = ""
        stability.compiler_exe = None
        stability.num_jobs = 1024
        stability.verbose = False
        stability.time = False
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
        common.check_tools(1)
        [targets_t, sde_targets_t] = check_targets()
        rebuild = True
        opts = []
        archs = []
        LLVM = []
        targets = []
        sde_targets = []
# parsing option only, update parameters of run
        if "-O2" in only:
            opts.append(False)
        if "-O0" in only:
            opts.append(True)
        if "x86" in only and not ("x86-64" in only):
            archs.append("x86")
        if "x86-64" in only:
            archs.append("x86-64")
        if "native" in only:
            sde_targets_t = []
        for i in ["3.1", "3.2", "3.3", "trunk"]:
            if i in only:
                LLVM.append(i)
        if "current" in only:
            LLVM = [" "]
            rebuild = False
        if only_targets != "":
            only_targets_t = only_targets.split(" ")
            for i in only_targets_t:
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
                    error("You haven't sde for target " + i, 1)
        else:
            targets = targets_t[:-4]
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
            LLVM = ["3.3", "trunk"]
        gen_archs = ["x86-64"]
        need_LLVM = check_LLVM(LLVM)
        for i in range(0,len(need_LLVM)):
            build_LLVM(need_LLVM[i], "", "", "", False, False, True, False)
# begin validation run for stabitily
        common.remove_if_exists(stability.in_file)
        R = [[[],[]],[[],[]],[[],[]],[[],[]]]
        print_debug("\n_________________________STABILITY REPORT_________________________\n", False, stability_log)
        for i in range(0,len(LLVM)):
            print_version = 2
            if rebuild:
                build_ispc(LLVM[i])
            for j in range(0,len(targets)):
                stability.target = targets[j]
                stability.wrapexe = ""
                if "generic" in targets[j]:
                    arch = gen_archs
                else:
                    arch = archs
                for i1 in range(0,len(arch)):
                    for i2 in range(0,len(opts)):
                        stability.arch = arch[i1]
                        stability.no_opt = opts[i2]
                        execute_stability(stability, R, print_version)
                        print_version = 0
            for j in range(0,len(sde_targets)):
                stability.target = sde_targets[j][1]
                stability.wrapexe = os.environ["SDE_HOME"] + "/sde " + sde_targets[j][0] + " -- "
                for i1 in range(0,len(archs)):
                    for i2 in range(0,len(opts)):
                        stability.arch = archs[i1]
                        stability.no_opt = opts[i2]
                        execute_stability(stability, R, print_version)
                        print_version = 0
# run special tests like embree
# 
        run_special_tests()
        ttt = ["NEW RUNFAILS: ", "NEW COMPFAILS: ", "NEW PASSES RUNFAILS: ", "NEW PASSES COMPFAILS: "]
        for j in range(0,4):
            if len(R[j][0]) == 0:
                print_debug("NO " + ttt[j][:-2] + "\n", False, stability_log)
            else:
                print_debug(ttt[j] + str(len(R[j][0])) + "\n", False, stability_log)
                temp5 = [[],[]]
                for i in range(0,len(R[j][0])):
                    er = True
                    for k in range(0,len(temp5[0])):
                        if R[j][0][i] == temp5[0][k]:
                            temp5[1][k].append(R[j][1][i])
                            er = False
                    if er == True:
                        temp5[0].append(R[j][0][i])
                        temp5[1].append([R[j][1][i]])
                for i in range(0,len(temp5[0])):
                    print_debug("\t" + temp5[0][i] + "\n", True, stability_log)
                    for k in range(0,len(temp5[1][i])):
                        print_debug("\t\t\t" + temp5[1][i][k], True, stability_log)
        print_debug("__________________Watch stability.log for details_________________\n", False, stability_log)
        if options.notify != "":
            attach_mail_file(msg, stability.in_file, "run_tests_log.log")
            attach_mail_file(msg, stability_log, "stability.log")

# *** *** ***
# Performance validation run
# *** *** ***
    if ((("performance" in only) == True) or ("stability" in only) == False):
        print_debug("\n\nPerformance validation run\n\n", False, "")
        performance = options_for_drivers()
# performance constant options
        performance.number = 5
        performance.config = "./perf.ini"
        performance.path = "./"
        performance.silent = True
        performance.output = ""
        performance.compiler = ""
        performance.ref = "ispc_ref"
        performance.in_file = "." + os.sep + f_date + os.sep + "performance.log"
# prepare LLVM 3.3 as newest LLVM
        need_LLVM = check_LLVM(["3.3"])
        if len(need_LLVM) != 0:
            build_LLVM(need_LLVM[i], "", "", "", False, False, True, False)
# prepare reference point. build both test and reference compilers
        os.system("git branch > " + temp_alloy_file)
        br = open(temp_alloy_file)
        temp4 = br.readlines()
        br.close()
        for line in temp4:
            if "*" in line:
                current_branch = line[2:-1]
        stashing = True
        sys.stdout.write("Please, don't interrupt script here! You can have not sync git status after interruption!\n")
        if "No local changes" in detect_version("git stash"):
            stashing = False
        #try_do_LLVM("stash current branch ", "git stash", True)
        try_do_LLVM("checkout reference branch " + reference_branch + " ", "git checkout " + reference_branch, True)
        sys.stdout.write(".\n")
        build_ispc("3.3")
        sys.stdout.write(".\n")
        os.rename("ispc", "ispc_ref")
        try_do_LLVM("checkout test branch " + current_branch + " ", "git checkout " + current_branch, True)
        if stashing:
            try_do_LLVM("return current branch ", "git stash pop", True)
        sys.stdout.write("You can interrupt script now.\n")
        build_ispc("3.3")
# begin validation run for performance. output is inserted into perf()
        perf.perf(performance, [])
        if options.notify != "":
            attach_mail_file(msg, performance.in_file, "performance.log")
            attach_mail_file(msg, "." + os.sep + "logs" + os.sep + "perf_build.log", "perf_build.log")

    print_debug("Logs are in alloy_results_[date]", False, "")

# sending e-mail with results
    if options.notify != "":
        fp = open(os.environ["ISPC_HOME"] + os.sep + "all_answer.txt", 'rb')
        f_lines = fp.readlines()
        fp.close()
        line = ""
        for i in range(0,len(f_lines)):
            line = line + f_lines[i][:-1]
            line = line + '   \n'
        text = MIMEText(line, "", "KOI-8")
        msg.attach(text)
        attach_mail_file(msg, alloy_build, "alloy_build.log")
        s = smtplib.SMTP(smtp_server)
        s.sendmail('ISPC_test_system', options.notify, msg.as_string())
        s.quit()
# exit of validation routine
    common.remove_if_exists(temp_alloy_file)
    os.chdir(current_path)

def Main():
    if (platform.system() == 'Windows' or 'CYGWIN_NT' in platform.system()) == True:
        error("Windows isn't supported now", 1)
    if (options.build_llvm == False and
       options.validation_run == False and
       options.llvm_home == "" and
       options.ispc_home == "" and
       options.sde_home == ""):
        parser.print_help()
        exit(0)
    global f_date
    f_date = "logs"
    common.remove_if_exists(f_date)
    os.makedirs(f_date)
    global temp_alloy_file
    temp_alloy_file = os.getcwd() + os.sep + f_date + os.sep + "temp_detect_version"
    global alloy_build
    alloy_build = os.getcwd() + os.sep + f_date + os.sep + "alloy_build.log"
    common.remove_if_exists(alloy_build) 
    global stability_log
    stability_log = os.getcwd() + os.sep + f_date + os.sep + "stability.log"
    common.remove_if_exists(stability_log)
    setting_paths(options.llvm_home, options.ispc_home, options.sde_home)
    if os.environ.get("LLVM_HOME") == None:
        error("you have no LLVM_HOME", 1)
    if os.environ.get("ISPC_HOME") == None:
        error("you have no ISPC_HOME", 1)
    if options.build_llvm:
        build_LLVM(options.version, options.revision, options.folder, options.tarball,
                    options.debug, options.selfbuild, False, options.force)
    if options.validation_run:
        validation_run(options.only, options.only_targets, options.branch, options.notify, options.update)
    os.rename(f_date, "alloy_results_" + datetime.datetime.now().strftime('%H_%M_%d_%m_%Y'))

###Main###
from optparse import OptionParser
import sys
import os
import operator
import time
import glob
import string
import platform
import smtplib
import datetime
import copy
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.mime.text import MIMEText
from email.Encoders import encode_base64
# our drivers
import run_tests
import perf
import common
error = common.error
detect_version = common.detect_version
print_debug = common.print_debug
# parsing options
parser = OptionParser()
# options for activity "build LLVM"
parser.add_option('-b', '--build-llvm', dest='build_llvm',
    help='ask to build LLVM', default=False, action="store_true")
parser.add_option('--version', dest='version',
    help='version of llvm to build: 3.1 3.2 3.3 trunk', default="trunk")
parser.add_option('--revision', dest='revision',
    help='revision of llvm to build in format r172870', default="")
parser.add_option('--debug', dest='debug',
    help='debug build of LLVM?', default=False, action="store_true")
parser.add_option('--folder', dest='folder',
    help='folder to build LLVM in', default="")
parser.add_option('--tarball', dest='tarball',
    help='"llvm_tarball clang_tarball"', default="")
parser.add_option('--selfbuild', dest='selfbuild',
    help='make selfbuild of LLVM and clang', default=False, action="store_true")
parser.add_option('--force', dest='force',
    help='rebuild LLVM', default=False, action='store_true')
# options for activity "setup PATHS"
parser.add_option('--llvm_home', dest='llvm_home',help='path to LLVM',default="")
parser.add_option('--ispc_home', dest='ispc_home',help='path to ISPC',default="")
parser.add_option('--sde_home', dest='sde_home',help='path to SDE',default="")
# options for activity "validation run"
parser.add_option('-r', '--run', dest='validation_run',
    help='ask for validation run', default=False, action="store_true")
parser.add_option('--compare-with', dest='branch',
    help='set performance reference point', default="master")
parser.add_option('--only-targets', dest='only_targets',
    help='set list of targets to test. Possible values - all subnames of targets.\n' +
        'Example: --only-targets="avx2-i32x8 sse4 i32x16 sse2"', default="")
parser.add_option('--notify', dest='notify',
    help='sent results to email', default="")
parser.add_option('--only', dest='only',
    help='set types of tests. Possible values:\n' + 
        '-O0, -O2, x86, x86-64, stability (test only stability), performance (test only performance)\n' +
        'build (only build with different LLVM), 3.1, 3.2, 3.3, trunk, native (do not use SDE), current (do not rebuild ISPC).\n' +
        'Example: --only="3.2 -O0 stability 3.3"', default="")
parser.add_option('--update-errors', dest='update',
    help='rewrite fail_db.txt file according to received results (F or FP)', default="")
(options, args) = parser.parse_args()
Main()
