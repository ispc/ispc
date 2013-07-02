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

def build_test():
    global build_log
    global is_windows
    if is_windows == False:
        os.system("make clean >> "+build_log)
        return os.system("make >> "+build_log+" 2>> "+build_log)
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
def run_test(command, c1, c2, test):
    global perf_temp
    if build_test() != 0:
        sys.stdout.write("ERROR: Compilation fails\n")
        return
    if execute_test(command) != 0:
        sys.stdout.write("ERROR: Execution fails\n")
        return
    tasks = [] #list of results with tasks, it will be test[2]
    ispc = [] #list of results without tasks, it will be test[1]
    j = 1
    for line in open(perf_temp): # we take test output
        if "speedup" in line: # we are interested only in lines with speedup
            if j == c1: # we are interested only in lines with c1 numbers
                sys.stdout.write(line)
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
    test[1] = test[1] + ispc
    test[2] = test[2] + tasks


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
        cpu1 = cpu_get()
        time.sleep(1)
        cpu2 = cpu_get()
        cpu_percent = (float(cpu1[0] - cpu2[0])/float(cpu1[1] - cpu2[1]))*100
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
#test[1] or test[2] may be empty
def print_answer(answer):
    sys.stdout.write("Name of test:\t\tISPC:\tISPC + tasks:\n")
    max_t = [0,0]
    diff_t = [0,0]
    geomean_t = [0,0]
    list_of_max = [[],[]]
    for i in range(len(answer)):
        for t in range(1,3):
            if len(answer[i][t]) == 0:
                max_t[t-1] = "n/a"
                diff_t[t-1] = "n/a"
            else:
                list_of_max[t-1].append(max(answer[i][t]))
                max_t[t-1] = str(max(answer[i][t]))
                diff_t[t-1] = str(max(answer[i][t]) - min(answer[i][t]))
        sys.stdout.write("%s:\n" % answer[i][0])
        sys.stdout.write("\t\tmax:\t%s\t%s\n" % (max_t[0], max_t[1]))
        sys.stdout.write("\t\tdiff:\t%s\t%s\n" % (diff_t[0], diff_t[1]))

    geomean_t[0] = geomean(list_of_max[0])
    geomean_t[1] = geomean(list_of_max[1])
    sys.stdout.write("---------------------------------------------\n")
    sys.stdout.write("Geomean:\t\t%s\t%s\n" % (geomean_t[0], geomean_t[1]))

###Main###
# parsing options
parser = OptionParser()
parser.add_option('-n', '--number', dest='number',
    help='number of repeats', default="3")
parser.add_option('-c', '--config', dest='config',
    help='config file of tests', default="./perf.ini")
parser.add_option('-p', '--path', dest='path',
    help='path to examples directory', default="./")
(options, args) = parser.parse_args()

global is_windows
is_windows = (platform.system() == 'Windows' or
              'CYGWIN_NT' in platform.system())

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
sys.stdout.write("Okey go go go!\n\n")
# loop for all tests
while i < length-2:
    # we read name of test
    sys.stdout.write("%s" % lines[i])
    test = [lines[i][:-1],[],[]]
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
        run_test(command, c1, c2, answer[len(answer)-1])
        i = i+1
    else: #we run this test and append it's result to answer structure
        run_test(command, c1, c2, test)
        answer.append(test)
    # preparing next loop iteration
    os.chdir(pwd)
    i+=4

# delete temp file
if os.path.exists(perf_temp):
    os.remove(perf_temp)
#print collected answer
print_answer(answer)
