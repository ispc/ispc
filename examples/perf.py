#!/usr/bin/python
# // Author: Filippov Ilia

from optparse import OptionParser
import sys
import os
import operator
import time
import glob
import string
import platform

def print_debug(line):
    if options.silent == False:
        sys.stdout.write(line)

def print_file(line):
    if options.output != "":
        output = open(options.output, 'w')
        output.writelines(line)
        output.close()

def build_test():
    global build_log
    global is_windows
    if is_windows == False:
        os.system("make clean >> "+build_log)
        return os.system("make CXX="+ref_compiler+" CC="+refc_compiler+" >> "+build_log+" 2>> "+build_log)
    else:
        os.system("msbuild /t:clean >> " + build_log)
        return os.system("msbuild /V:m /p:Platform=x64 /p:Configuration=Release /p:TargetDir=.\ /t:rebuild >> " + build_log)

def execute_test(command):
    global perf_temp
    r = 0
    if os.path.exists(perf_temp):
        os.remove(perf_temp)
    for k in range(int(options.number)):
        r = r + os.system(command)
    return r

#gathers all tests results and made an item test from answer structure
def run_test(command, c1, c2, test, b_serial):
    global perf_temp
    if build_test() != 0:
        sys.stdout.write("ERROR: Compilation fails\n")
        return
    if execute_test(command) != 0:
        sys.stdout.write("ERROR: Execution fails\n")
        return
    tasks = [] #list of results with tasks, it will be test[2]
    ispc = [] #list of results without tasks, it will be test[1]
    absolute_tasks = []  #list of absolute results with tasks, it will be test[4]
    absolute_ispc = [] #list of absolute results without tasks, ut will be test[3]
    serial = [] #list serial times, it will be test[5]
    j = 1
    for line in open(perf_temp): # we take test output
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
            print_debug("ISPC speedup / ISPC + tasks speedup / ISPC time / ISPC + tasks time / serial time\n")
            for i in range(0,len(serial)):
                print_debug("%10s   /\t%10s\t    /%9s  /    %10s\t    /%10s\n" %
                    (ispc[i], tasks[i], absolute_ispc[i], absolute_tasks[i], serial[i]))
        else:
            print_debug("ISPC speedup / ISPC time / serial time\n")
            for i in range(0,len(serial)):
                print_debug("%10s   /%9s  /%10s\n" % (ispc[i], absolute_ispc[i], serial[i]))
    else:
        if len(tasks) != 0:
            print_debug("ISPC + tasks speedup / ISPC + tasks time / serial time\n")
            for i in range(0,len(serial)):
                print_debug("%10s\t     /    %10s\t /%10s\n" % (tasks[i], absolute_tasks[i], serial[i]))

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
    temp = temp ** (1.0/l)
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
def print_answer(answer):
    filelist = []
    print_debug("--------------------------------------------------------------------------\n")
    print_debug("test name:\t    ISPC speedup: ISPC + tasks speedup: | " + 
        "ISPC time:    ISPC + tasks time:  serial:\n")
    filelist.append("test name,ISPC speedup,diff," +
        "ISPC + tasks speedup,diff,ISPC time,diff,ISPC + tasks time,diff,serial,diff\n")
    max_t = [0,0,0,0,0]
    diff_t = [0,0,0,0,0]
    geomean_t = [0,0,0,0,0]
    list_of_max = [[],[],[],[],[]]
    for i in range(len(answer)):
        for t in range(1,6):
            if len(answer[i][t]) == 0:
                max_t[t-1] = "n/a"
                diff_t[t-1] = "n/a"
            else:
                if t < 3:
                    mm = max(answer[i][t])
                else:
                    mm = min(answer[i][t])
                max_t[t-1] = '%.2f' % mm
                list_of_max[t-1].append(mm)
                diff_t[t-1] = '%.2f' % (max(answer[i][t]) - min(answer[i][t]))
        print_debug("%s:\n" % answer[i][0])
        print_debug("\t\tmax:\t%5s\t\t%10s\t|%10s\t%10s\t%10s\n" %
            (max_t[0], max_t[1], max_t[2], max_t[3], max_t[4]))
        print_debug("\t\tdiff:\t%5s\t\t%10s\t|%10s\t%10s\t%10s\n" %
            (diff_t[0], diff_t[1], diff_t[2], diff_t[3], diff_t[4]))
        for t in range(0,5):
            if max_t[t] == "n/a":
                max_t[t] = ""
            if diff_t[t] == "n/a":
                diff_t[t] = ""
        filelist.append(answer[i][0] + "," +
                        max_t[0] + "," + diff_t[0] + "," +  max_t[1] + "," + diff_t[1] + "," +
                        max_t[2] + "," + diff_t[2] + "," +  max_t[3] + "," + diff_t[3] + "," +
                        max_t[4] + "," + diff_t[4] + "\n")
    for i in range(0,5):
        geomean_t[i] = geomean(list_of_max[i])
    print_debug("---------------------------------------------------------------------------------\n")
    print_debug("Geomean:\t\t%5s\t\t%10s\t|%10s\t%10s\t%10s\n" %
        (geomean_t[0], geomean_t[1], geomean_t[2], geomean_t[3], geomean_t[4]))
    filelist.append("Geomean," + str(geomean_t[0]) + ",," + str(geomean_t[1])
        + ",," + str(geomean_t[2]) + ",," + str(geomean_t[3]) + ",," + str(geomean_t[4]) + "\n")
    print_file(filelist)


###Main###
# parsing options
parser = OptionParser()
parser.add_option('-n', '--number', dest='number',
    help='number of repeats', default="3")
parser.add_option('-c', '--config', dest='config',
    help='config file of tests', default="./perf.ini")
parser.add_option('-p', '--path', dest='path',
    help='path to examples directory', default="./")
parser.add_option('-s', '--silent', dest='silent',
    help='silent mode, only table output', default=False, action="store_true")
parser.add_option('-o', '--output', dest='output',
    help='output file for script reading', default="")
parser.add_option('--compiler', dest='compiler',
    help='reference compiler', default="")
(options, args) = parser.parse_args()

global is_windows
is_windows = (platform.system() == 'Windows' or
              'CYGWIN_NT' in platform.system())
global is_mac
is_mac = (platform.system() == 'Darwin')

# save corrent path
pwd = os.getcwd()
pwd = pwd + os.sep
if is_windows:
    pwd = "..\\"

# check if cpu usage is low now
cpu_percent = cpu_check()
if cpu_percent > 20:
    sys.stdout.write("Warning: CPU Usage is very high.\n")
    sys.stdout.write("Close other applications.\n")

# check that required compilers exist
PATH_dir = string.split(os.getenv("PATH"), os.pathsep)
compiler_exists = False
ref_compiler_exists = False
if is_windows == False:
    compiler = "ispc"
    ref_compiler = "g++"
    refc_compiler = "gcc"
    if options.compiler != "":
        if options.compiler == "clang" or options.compiler == "clang++":
            ref_compiler = "clang++"
            refc_compiler = "clang"
        if options.compiler == "icc" or options.compiler == "icpc":
            ref_compiler = "icpc"
            refc_compiler = "icc"
else:
    compiler = "ispc.exe"
    ref_compiler = "cl.exe"
for counter in PATH_dir:
    if os.path.exists(counter + os.sep + compiler):
        compiler_exists = True
    if os.path.exists(counter + os.sep + ref_compiler):
        ref_compiler_exists = True
if not compiler_exists:
    sys.stderr.write("Fatal error: ISPC compiler not found.\n")
    sys.stderr.write("Added path to ispc compiler to your PATH variable.\n")
    sys.exit()
if not ref_compiler_exists:
    sys.stderr.write("Fatal error: reference compiler %s not found.\n" % ref_compiler)
    sys.stderr.write("Added path to %s compiler to your PATH variable.\n" % ref_compiler)
    sys.exit()

# checks that config file exists
path_config = os.path.normpath(options.config)
if os.path.exists(path_config) == False:
    sys.stderr.write("Fatal error: config file not found: %s.\n" % options.config) 
    sys.stderr.write("Set path to your config file in --config.\n")
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

# prepare build.log and perf_temp files
global build_log
build_log = pwd + "build.log"
if is_windows == False:
    if os.path.exists(build_log):
        os.remove(build_log)
else:
    if os.path.exists("build.log"):
        os.remove("build.log")
global perf_temp
perf_temp = pwd + "perf_temp"

i = 0
answer = []
print_debug("Okey go go go!\n\n")
os.system(compiler + " --version >" + build_log)
version = open(build_log)
print_debug("Using test compiler: " + version.readline())
version.close()

if is_windows == False:
    os.system(ref_compiler + " --version >" + build_log)
else:
    os.system(ref_compiler + " 2>" + build_log + " 1>&2")

version = open(build_log)
print_debug("Using reference compiler: " + version.readline())
version.close()


# loop for all tests
while i < length-2:
    # we read name of test
    print_debug("%s" % lines[i])
    test = [lines[i][:-1],[],[],[],[],[]]
    # read location of test
    folder = lines[i+1]
    folder = folder[:-1]
    folder = os.path.normpath(options.path + os.sep + folder)
    # check that test exists
    if os.path.exists(folder) == False:
        sys.stdout.write("Fatal error: Can't find test %s. Your path is: \"%s\".\n" % (lines[i][:-1], options.path))
        sys.stdout.write("Change current location to /examples or set path to /examples in --path.\n")
        exit(0)
    os.chdir(folder)
    # read parameters of test
    command = lines[i+2]
    command = command[:-1]
    if is_windows == False:
        command = "./"+command + " >> " + perf_temp
    else:
        command = "x64\\Release\\"+command + " >> " + perf_temp
    # parsing config parameters
    next_line = lines[i+3]
    if next_line[0] == "!": # we should take only one part of test output
        R = next_line.split(' ')
        c1 = int(R[1]) #c1 is a number of string which we want to use in test output
        c2 = int(R[2]) #c2 is total number of strings in test output
        i = i+1
    else:
        c1 = 1
        c2 = 1
    next_line = lines[i+3]
    if next_line[0] == "^":  #we should concatenate result of this test with previous one
        run_test(command, c1, c2, answer[len(answer)-1], False)
        i = i+1
    else: #we run this test and append it's result to answer structure
        run_test(command, c1, c2, test, True)
        answer.append(test)
    # preparing next loop iteration
    os.chdir(pwd)
    i+=4

# delete temp file
if os.path.exists(perf_temp):
    os.remove(perf_temp)
#print collected answer
print_answer(answer)
