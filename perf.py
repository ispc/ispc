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

# // Author: Filippov Ilia

def print_file(line):
    if options.output != "":
        output = open(options.output, 'w')
        output.writelines(line)
        output.close()

def execute_test(commands):
    r = 0
    common.remove_if_exists(perf_temp+"_test")
    common.remove_if_exists(perf_temp+"_ref")
    for k in range(int(options.number)):
        r = r + os.system(commands[0])
        if options.ref:
            r = r + os.system(commands[1])
    return r

#gathers all tests results and made an item test from answer structure
def run_test(commands, c1, c2, test, test_ref, b_serial):
    if execute_test(commands) != 0:
        error("Execution fails of test %s\n" % test[0], 0)
        global exit_code
        exit_code = 1
        return
    print_debug("TEST COMPILER:\n", s, perf_log)
    analyse_test(c1, c2, test, b_serial, perf_temp+"_test")
    if options.ref:
        print_debug("REFERENCE COMPILER:\n", s, perf_log)
        analyse_test(c1, c2, test_ref, b_serial, perf_temp+"_ref")


def analyse_test(c1, c2, test, b_serial, perf_temp_n):
    tasks = [] #list of results with tasks, it will be test[2]
    ispc = [] #list of results without tasks, it will be test[1]
    absolute_tasks = []  #list of absolute results with tasks, it will be test[4]
    absolute_ispc = [] #list of absolute results without tasks, ut will be test[3]
    serial = [] #list serial times, it will be test[5]
    j = 1
    for line in open(perf_temp_n): # we take test output
        if "speedup" in line: # we are interested only in lines with speedup
            if j == c1: # we are interested only in lines with c1 numbers
                line = line.expandtabs(0)
                line = line.replace("("," ")
                line = line.split(",")
                for i in range(len(line)):
                    subline = line[i].split(" ")
                    number = float(subline[1][:-1])
                    if "speedup from ISPC + tasks" in line[i]:
                        tasks.append(number)
                    else:
                        ispc.append(number)
                c1 = c1 + c2
            j+=1
        if "million cycles" in line:
            if j == c1:
                if line[0] == '@':
                    print_debug(line, True, perf_log)
                else:
                    line = line.replace("]","[")
                    line = line.split("[")
                    number = float(line[3])
                    if "tasks" in line[1]:
                        absolute_tasks.append(number)
                    else:
                        if "ispc" in line[1]:
                            absolute_ispc.append(number)
                    if "serial" in line[1]:
                        serial.append(number)

    if len(ispc) != 0:
        if len(tasks) != 0:
            print_debug("ISPC speedup / ISPC + tasks speedup / ISPC time / ISPC + tasks time / serial time\n", s, perf_log)
            for i in range(0,len(serial)):
                print_debug("%10s   /\t%10s\t    /%9s  /    %10s\t    /%10s\n" %
                    (ispc[i], tasks[i], absolute_ispc[i], absolute_tasks[i], serial[i]), s, perf_log)
        else:
            print_debug("ISPC speedup / ISPC time / serial time\n", s, perf_log)
            for i in range(0,len(serial)):
                print_debug("%10s   /%9s  /%10s\n" % (ispc[i], absolute_ispc[i], serial[i]), s, perf_log)
    else:
        if len(tasks) != 0:
            print_debug("ISPC + tasks speedup / ISPC + tasks time / serial time\n", s, perf_log)
            for i in range(0,len(serial)):
                print_debug("%10s\t     /    %10s\t /%10s\n" % (tasks[i], absolute_tasks[i], serial[i]), s, perf_log)

    test[1] = test[1] + ispc
    test[2] = test[2] + tasks
    test[3] = test[3] + absolute_ispc
    test[4] = test[4] + absolute_tasks
    if b_serial == True:
        #if we concatenate outputs we should use only the first serial answer.
        test[5] = test[5] + serial

def cpu_get():
    p = open("/proc/stat", 'r')
    cpu = p.readline()
    p.close()
    cpu = cpu.split(" ")
    cpu_usage = (int(cpu[2]) + int(cpu[3]) + int(cpu[4]))
    cpu_all = cpu_usage + int(cpu[5])
    return [cpu_usage, cpu_all]

#returns cpu_usage
def cpu_check():
    if is_windows == False:
        if is_mac == False:
            cpu1 = cpu_get()
            time.sleep(1)
            cpu2 = cpu_get()
            cpu_percent = (float(cpu1[0] - cpu2[0])/float(cpu1[1] - cpu2[1]))*100
        else:
            os.system("sysctl -n vm.loadavg > cpu_temp")
            c = open("cpu_temp", 'r')
            c_line = c.readline()
            c.close
            os.remove("cpu_temp")
            R = c_line.split(' ')
            cpu_percent = float(R[1]) * 3
    else:
        os.system("wmic cpu get loadpercentage /value > cpu_temp")
        c = open("cpu_temp", 'r')
        c_lines = c.readlines()
        c.close()
        os.remove("cpu_temp")
        t = "0"
        for i in c_lines[2]:
            if i.isdigit():
                t = t + i
        cpu_percent = int(t)
    return cpu_percent

#returns geomean of list
def geomean(par):
    temp = 1
    l = len(par)
    for i in range(l):
        temp = temp * par[i]
    if l != 0:
        temp = temp ** (1.0/l)
    else:
        temp = 0
    return round(temp, 2)

#takes an answer struct and print it.
#answer struct: list answer contains lists test
#test[0] - name of test
#test[1] - list of results without tasks
#test[2] - list of results with tasks
#test[3] - list of absolute results without tasks
#test[4] - list of absolute results with tasks
#test[5] - list of absolute time without ISPC (serial)
#test[1..4] may be empty
def print_answer(answer, target_number):
    filelist = []
    print_debug("--------------------------------------------------------------------------\n", s, perf_log)
    print_debug("test name:\t    ISPC speedup: ISPC + tasks speedup: | " +
        "    ISPC time:    ISPC + tasks time:  serial:\n", s, perf_log)
    if target_number > 1:
        if options.output == "":
            options.output = "targets.csv"
        filelist.append("test name,ISPC speedup" + "," * target_number + "ISPC + tasks speedup\n")
        filelist.append("," + options.perf_target + "," + options.perf_target + "\n")
    else:
        filelist.append("test name,ISPC speedup,diff," +
            "ISPC + tasks speedup,diff,ISPC time,diff,ISPC + tasks time,diff,serial,diff\n")
    max_t = [0,0,0,0,0]
    diff_t = [0,0,0,0,0]
    geomean_t = []
    list_of_max = []
    for i1 in range(target_number):
        geomean_t.append([0,0,0,0,0])
        list_of_max.append([[],[],[],[],[]])
    list_of_compare = [[],[],[],[],[],[]]
    target_k = 0
    temp_str_1 = ""
    temp_str_2 = ""
    for i in range(len(answer)):
        list_of_compare[0].append(answer[i][0])
        for t in range(1,6):
            if len(answer[i][t]) == 0:
                max_t[t-1] = "n/a"
                diff_t[t-1] = "n/a"
                list_of_compare[t].append(0);
            else:
                if t < 3:
                    mm = max(answer[i][t])
                else:
                    mm = min(answer[i][t])
                list_of_compare[t].append(mm)
                max_t[t-1] = '%.2f' % mm
                list_of_max[i % target_number][t-1].append(mm)
                diff_t[t-1] = '%.2f' % (max(answer[i][t]) - min(answer[i][t]))
        print_debug("%s:\n" % answer[i][0], s, perf_log)
        print_debug("\t\tmax:\t%5s\t\t%10s\t|min:%10s\t%10s\t%10s\n" %
            (max_t[0], max_t[1], max_t[2], max_t[3], max_t[4]), s, perf_log)
        print_debug("\t\tdiff:\t%5s\t\t%10s\t|%14s\t%10s\t%10s\n" %
            (diff_t[0], diff_t[1], diff_t[2], diff_t[3], diff_t[4]), s, perf_log)
        for t in range(0,5):
            if max_t[t] == "n/a":
                max_t[t] = ""
            if diff_t[t] == "n/a":
                diff_t[t] = ""
        if target_number > 1:
            if target_k == 0:
                temp_str_1 = answer[i][0] + ","
                temp_str_2 = ""
            temp_str_1 += max_t[0] + ","
            temp_str_2 += max_t[1] + ","
            target_k = target_k + 1
            if target_k == target_number:
                filelist.append(temp_str_1 + temp_str_2[:-1] + "\n")
                target_k = 0
        else:
            filelist.append(answer[i][0] + "," +
                        max_t[0] + "," + diff_t[0] + "," +  max_t[1] + "," + diff_t[1] + "," +
                        max_t[2] + "," + diff_t[2] + "," +  max_t[3] + "," + diff_t[3] + "," +
                        max_t[4] + "," + diff_t[4] + "\n")
    for i in range(0,5):
        for i1 in range(target_number):
            geomean_t[i1][i] = geomean(list_of_max[i1][i])
    print_debug("---------------------------------------------------------------------------------\n", s, perf_log)
    print_debug("Geomean:\t\t%5s\t\t%10s\t|%14s\t%10s\t%10s\n" %
        (geomean_t[0][0], geomean_t[0][1], geomean_t[0][2], geomean_t[0][3], geomean_t[0][4]), s, perf_log)
    if target_number > 1:
        temp_str_1 = "Geomean,"
        temp_str_2 = ""
        for i in range(target_number):
            temp_str_1 += str(geomean_t[i][0]) + ","
            temp_str_2 += str(geomean_t[i][1]) + ","
        filelist.append(temp_str_1 + temp_str_2[:-1] + "\n")
    else:
        filelist.append("Geomean," + str(geomean_t[0][0]) + ",," + str(geomean_t[0][1])
            + ",," + str(geomean_t[0][2]) + ",," + str(geomean_t[0][3]) + ",," + str(geomean_t[0][4]) + "\n")
    print_file(filelist)
    return list_of_compare


def compare(A, B):
    print_debug("\n\n_____________________PERFORMANCE REPORT____________________________\n", False, "")
    print_debug("test name:                 ISPC time: ISPC time ref: %:\n", False, "")
    for i in range(0,len(A[0])):
        if B[3][i] == 0:
            p1 = 0
        else:
            p1 = 100 - 100 * A[3][i]/B[3][i]
        print_debug("%21s:  %10.2f %10.2f %10.2f" % (A[0][i], A[3][i], B[3][i], abs(p1)), False, "")
        if p1 < -1:
            print_debug(" <+", False, "")
        if p1 > 1:
            print_debug(" <-", False, "")
        print_debug("\n", False, "")
    print_debug("\n", False, "")

    print_debug("test name:                 TASKS time: TASKS time ref: %:\n", False, "")
    for i in range(0,len(A[0])):
        if B[4][i] == 0:
            p2 = 0
        else:
            p2 = 100 - 100 * A[4][i]/B[4][i]
        print_debug("%21s:  %10.2f %10.2f %10.2f" % (A[0][i], A[4][i], B[4][i], abs(p2)), False, "")
        if p2 < -1:
            print_debug(" <+", False, "")
        if p2 > 1:
            print_debug(" <-", False, "")
        print_debug("\n", False, "")
    if "performance.log" in options.in_file:
        print_debug("\n\n_________________Watch performance.log for details________________\n", False, "")
    else:
        print_debug("\n\n__________________________________________________________________\n", False, "")



def perf(options1, args):
    global options
    options = options1
    global s
    s = options.silent

    # save current OS
    global is_windows
    is_windows = (platform.system() == 'Windows' or
              'CYGWIN_NT' in platform.system())
    global is_mac
    is_mac = (platform.system() == 'Darwin')

    # save current path
    pwd = os.getcwd()
    pwd = pwd + os.sep
    pwd1 = pwd
    if is_windows:
        pwd1 = "..\\..\\"

    if options.perf_target != "":
        test_only_r = " sse2-i32x4 sse2-i32x8 \
                        sse4-i32x4 sse4-i32x8 sse4-i16x8 sse4-i8x16 \
                        avx1-i32x4 avx1-i32x8 avx1-i32x16 avx1-i64x4 \
                        avx2-i32x4 avx2-i32x8 avx2-i32x16 avx2-i64x4 \
                        avx512knl-x16 \
                        avx512skx-x16 avx512skx-x8 avx512skx-x4 avx512skx-x64 avx512skx-x32"
        test_only = options.perf_target.split(",")
        for iterator in test_only:
            if not (" " + iterator + " " in test_only_r):
                error("unknow option for target: " + iterator, 1)

    # check if cpu usage is low now
    cpu_percent = cpu_check()
    if cpu_percent > 20:
        error("CPU Usage is very high.\nClose other applications.\n", 2)

    # prepare build.log, perf_temp and perf.log files
    global perf_log
    if options.in_file:
        perf_log = pwd + options.in_file
        common.remove_if_exists(perf_log)
    else:
        perf_log = ""
    global build_log
    build_log = pwd + os.sep + "logs" + os.sep + "perf_build.log"
    common.remove_if_exists(build_log)
    if os.path.exists(pwd + os.sep + "logs") == False:
        os.makedirs(pwd + os.sep + "logs")
    global perf_temp
    perf_temp = pwd + "perf_temp"


    global ispc_test
    global ispc_ref
    global ref_compiler
    global refc_compiler
    # check that required compilers exist
    PATH_dir = os.environ["PATH"].split(os.pathsep)
    ispc_test_exists = False
    ispc_ref_exists = False
    ref_compiler_exists = False
    if is_windows == False:
        ispc_test = "ispc"
        ref_compiler = "clang++"
        refc_compiler = "clang"
        if options.compiler != "":
            if options.compiler == "clang" or options.compiler == "clang++":
                ref_compiler = "clang++"
                refc_compiler = "clang"
            if options.compiler == "icc" or options.compiler == "icpc":
                ref_compiler = "icpc"
                refc_compiler = "icc"
            if options.compiler == "gcc" or options.compiler == "g++":
                ref_compiler = "g++"
                refc_compiler = "gcc"
    else:
        ispc_test = "ispc.exe"
        ref_compiler = "cl.exe"
    ispc_ref = options.ref
    if options.ref != "":
        options.ref = True
    if os.environ.get("ISPC_HOME") != None:
        if os.path.exists(os.environ["ISPC_HOME"] + os.sep + ispc_test):
            ispc_test_exists = True
            ispc_test = os.environ["ISPC_HOME"] + os.sep + ispc_test
    for counter in PATH_dir:
        if ispc_test_exists == False:
            if os.path.exists(counter + os.sep + ispc_test):
                ispc_test_exists = True
                ispc_test = counter + os.sep + ispc_test
        if os.path.exists(counter + os.sep + ref_compiler):
            ref_compiler_exists = True
        if ispc_ref != "":
            if os.path.exists(counter + os.sep + ispc_ref):
                ispc_ref_exists = True
                ispc_ref = counter + os.sep + ispc_ref
    if not ispc_test_exists:
        error("ISPC compiler not found.\nAdded path to ispc compiler to your PATH variable or ISPC_HOME variable\n", 1)
    if not ref_compiler_exists:
        error("C/C++ compiler %s not found.\nAdded path to %s compiler to your PATH variable.\n" % (ref_compiler, ref_compiler), 1)
    if options.ref:
        if not ispc_ref_exists:
            error("ISPC reference compiler not found.\nAdded path to ispc reference compiler to your PATH variable.\n", 1)

    # checks that config file exists
    path_config = os.path.normpath(options.config)
    if os.path.exists(path_config) == False:
        error("config file not found: %s.\nSet path to your config file in --config.\n" % options.config, 1)
        sys.exit()

    # read lines from config file except comments
    f = open(path_config, 'r')
    f_lines = f.readlines()
    f.close()
    lines =[]
    for i in range(len(f_lines)):
        if f_lines[i][0] != "%":
            lines.append(f_lines[i])
    length = len(lines)
    # end of preparations

    print_debug("Okey go go go!\n\n", s, perf_log)
    # report command line
    if __name__ == "__main__":
        print_debug("Command line: %s\n" % " ".join(map(str, sys.argv)), s, perf_log)
    # report used ispc
    print_debug("Testing ispc: " + ispc_test + "\n", s, perf_log)

    #print compilers versions
    common.print_version(ispc_test, ispc_ref, ref_compiler, False, perf_log, is_windows)

    # begin
    i = 0
    answer = []
    answer_ref = []
    # loop for all tests
    perf_targets = [""]
    target_number = 1
    target_str_temp = ""
    if options.perf_target != "":
        perf_targets = options.perf_target.split(',')
        target_str_temp = " -DISPC_IA_TARGETS="
        target_number = len(perf_targets)
    # Generate build targets for tests
    if options.generator:
        generator = options.generator
    else:
        if is_windows == True:
            generator = "Visual Studio 16"
        else:
            generator = "Unix Makefiles"
    examples_folder_ref = "examples_ref"
    examples_folder_test = "examples_test"
    install_prefix = "install"
    cmake_command = "cmake -G " + "\"" + generator + "\"" + " -DCMAKE_INSTALL_PREFIX=" + install_prefix + " " + pwd + "examples" + os.sep + "cpu"
    if is_windows == False:
        cmake_command += " -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang"
    for target_i in range(target_number):
        cur_target = perf_targets[target_i]
        target_str = target_str_temp + cur_target
        if options.ref:
            build_folder = examples_folder_ref + os.sep + cur_target
            if os.path.exists(build_folder):
                shutil.rmtree(build_folder)
            os.makedirs(build_folder)
            cmake_command_ref = "cd " + build_folder + " && " + cmake_command + \
                                " -DISPC_EXECUTABLE=" + ispc_ref + target_str + " >> " + build_log
            if os.system(cmake_command_ref) != 0:
                error("Cmake command failed with reference compiler %s\n" % ispc_ref, 1)
            # Build and install tests for reference compiler
            if is_windows == False:
                bu_command_ref = "cd " + build_folder + " && make install >> "+ build_log+" 2>> "+ build_log
            else:
                bu_command_ref = "msbuild " + build_folder + os.sep + "INSTALL.vcxproj /V:m /p:Configuration=Release /t:rebuild >> " + build_log
            if os.system(bu_command_ref) != 0:
                error("Build failed with reference compiler %s\n" % ispc_ref, 1)
        build_folder = examples_folder_test + os.sep + cur_target
        if os.path.exists(build_folder):
            shutil.rmtree(build_folder)
        os.makedirs(build_folder)
        cmake_command_test = "cd " + build_folder + " && " + cmake_command + \
                             " -DISPC_EXECUTABLE=" + ispc_test + target_str + " >> " + build_log
        if os.system(cmake_command_test) != 0:
            error("Cmake command failed with test compiler %s\n" % ispc_test, 1)
        # Build and install tests for test compiler
        if is_windows == False:
            bu_command_test = "cd " + build_folder + " && make install >> "+ build_log+" 2>> "+ build_log
        else:
            bu_command_test = "msbuild " + build_folder + os.sep + "INSTALL.vcxproj /V:m /p:Configuration=Release /t:rebuild >> " + build_log
        if os.system(bu_command_test) != 0:
            error("Build failed with test compiler %s\n" % ispc_test, 1)
    # Run tests
    while i < length-2:
        # we read name of test
        print_debug("%s" % lines[i], s, perf_log)
        # read location of test
        folder = lines[i+1]
        folder = folder[:-1]
        example = folder
        # read parameters of test
        command = lines[i+2]
        command = command[:-1]
        temp = 0
        # execute test for each target
        for target_i in range(target_number):
            test = [lines[i][:-1],[],[],[],[],[]]
            test_ref = [lines[i][:-1],[],[],[],[],[]]
            cur_target = perf_targets[target_i]
            folder = os.path.normpath(options.path + os.sep + examples_folder_test + os.sep + cur_target + \
                                      os.sep + install_prefix + os.sep + "examples" + os.sep + example)
            folder_ref = os.path.normpath(options.path + os.sep + examples_folder_ref + os.sep + cur_target + \
                                          os.sep + install_prefix + os.sep + "examples" + os.sep + example)
            # check that test exists
            if os.path.exists(folder) == False:
                error("Can't find test %s. Your path is: \"%s\".\nChange current location to ISPC_HOME or set path to ISPC_HOME in --path.\n" %
                 (lines[i][:-1], folder), 1)
            if is_windows == False:
                ex_command_ref = "cd "+ folder_ref + " && ./" + example + " " + command + " >> " + perf_temp + "_ref"
                ex_command = "cd "+ folder + " && ./" + example + " " + command + " >> " + perf_temp + "_test"
            else:
                ex_command_ref = "cd "+ folder_ref + " && " + example + ".exe " + command + " >> " + perf_temp + "_ref"
                ex_command = "cd "+ folder + " && " + example + ".exe " + command + " >> " + perf_temp + "_test"
            commands = [ex_command, ex_command_ref]
            # parsing config parameters
            next_line = lines[i+3]
            if next_line[0] == "!": # we should take only one part of test output
                R = next_line.split(' ')
                c1 = int(R[1]) #c1 is a number of string which we want to use in test output
                c2 = int(R[2]) #c2 is total number of strings in test output
                temp = 1
            else:
                c1 = 1
                c2 = 1
            next_line = lines[i+3]
            if next_line[0] == "^":
                temp = 1
            if next_line[0] == "^" and target_number == 1:  #we should concatenate result of this test with previous one
                run_test(commands, c1, c2, answer[len(answer)-1], answer_ref[len(answer)-1], False)
            else: #we run this test and append it's result to answer structure
                run_test(commands, c1, c2, test, test_ref, True)
                answer.append(test)
                answer_ref.append(test_ref)
        i = i + temp
        # preparing next loop iteration
        i+=4

    # delete temp file
    common.remove_if_exists(perf_temp+"_test")
    common.remove_if_exists(perf_temp+"_ref")

    #print collected answer
    if target_number > 1:
        s = True
    print_debug("\n\nTEST COMPILER:\n", s, perf_log)
    A = print_answer(answer, target_number)
    if options.ref != "":
        print_debug("\n\nREFERENCE COMPILER:\n", s, perf_log)
        B = print_answer(answer_ref, target_number)
        # print perf report
        compare(A,B)



###Main###
from optparse import OptionParser
import sys
import os
import operator
import time
import glob
import platform
import shutil
# our functions
import common
print_debug = common.print_debug
error = common.error
exit_code = 0

if __name__ == "__main__":
    # parsing options
    parser = OptionParser()
    parser.add_option('-n', '--number', dest='number',
        help='number of repeats', default="3")
    parser.add_option('-c', '--config', dest='config',
        help='config file of tests', default="./perf.ini")
    parser.add_option('-p', '--path', dest='path',
        help='path to ispc root', default=".")
    parser.add_option('-s', '--silent', dest='silent',
        help='silent mode, only table output', default=False, action="store_true")
    parser.add_option('-o', '--output', dest='output',
        help='output file for script reading', default="")
    parser.add_option('--compiler', dest='compiler',
        help='C/C++ compiler', default="")
    parser.add_option('-r', '--ref', dest='ref',
        help='set reference compiler for compare', default="")
    parser.add_option('-f', '--file', dest='in_file',
        help='file to save perf output', default="")
    parser.add_option('-t', '--target', dest='perf_target',
        help='set ispc target for building benchmarks (both test and ref)', default="")
    parser.add_option('-g', '--generator', dest='generator',
        help='cmake generator')
    (options, args) = parser.parse_args()
    perf(options, args)
    exit(exit_code)
