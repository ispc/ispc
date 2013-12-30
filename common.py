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
import sys
import os
import shutil

def write_to_file(filename, line):
    f = open(filename, 'a')
    f.writelines(line)
    f.close()

#remove file if it exists
def remove_if_exists(filename):
    if os.path.exists(filename):
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        else:
            os.remove(filename)

# detect version which is printed after command
def take_lines(command, which):
    os.system(command + " > " + "temp_detect_version")
    version = open("temp_detect_version")
    if which == "first":
        answer = version.readline()
    if which == "all":
        answer = version.readlines()
    version.close()
    remove_if_exists("temp_detect_version")
    return answer

# print versions of compilers
def print_version(ispc_test, ispc_ref, ref_compiler, s, perf_log, is_windows):
    print_debug("\nUsing test compiler: " + take_lines(ispc_test + " --version", "first"), s, perf_log)
    if ispc_ref != "":
        print_debug("Using ref compiler:  " + take_lines(ispc_ref + " --version", "first"), s, perf_log)
    if is_windows == False:
        temp1 = take_lines(ref_compiler + " --version", "first")
    else:
        os.system(ref_compiler + " 2>&1" + " 2> temp_detect_version > temp_detect_version1" )
        version = open("temp_detect_version")
        temp1 = version.readline()
        version.close()
        remove_if_exists("temp_detect_version")
        remove_if_exists("temp_detect_version1")
    print_debug("Using C/C++ compiler: " + temp1 + "\n", s, perf_log)

# print everything from scripts instead errors
def print_debug(line, silent, filename):
    if silent == False:
        sys.stdout.write(line)
        sys.stdout.flush()
        if os.environ.get("ISPC_HOME") != None:
            if os.path.exists(os.environ.get("ISPC_HOME")):
                write_to_file(os.environ["ISPC_HOME"] + os.sep + "notify_log.log", line)
    if filename != "":
        write_to_file(filename, line)

# print errors from scripts
# type 1 for error in environment
# type 2 for warning
# type 3 for error of compiler or test which isn't the goal of script 
def error(line, error_type):
    line = line + "\n"
    if error_type == 1:
        sys.stderr.write("Fatal error: " + line)
        sys.exit(1)
    if error_type == 2:
        sys.stderr.write("Warning: " + line)
    if error_type == 0:
        print_debug("FIND ERROR: " + line, False, "")

def check_tools(m):
    input_tools=[[[1,4],"m4 --version", "bad m4 version"],
                 [[2,4],"bison --version", "bad bison version"],
                 [[2,5], "flex --version", "bad flex version"]]
    ret = 1 
    for t in range(0,len(input_tools)):
        t1 = ((take_lines(input_tools[t][1], "first"))[:-1].split(" "))
        for i in range(0,len(t1)):
            t11 = t1[i].split(".")
            f = True
            for j in range(0,len(t11)):
                if not t11[j].isdigit():
                    f = False
            if f == True:
                for j in range(0,len(t11)):
                    if j < len(input_tools[t][0]):
                        if int(t11[j])<input_tools[t][0][j]:
                            error(input_tools[t][2], m)
                            ret = 0
                            break
                        if int(t11[j])>input_tools[t][0][j]:
                            break
    return ret
