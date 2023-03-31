#!/usr/bin/env python3
#
#  Copyright (c) 2013-2023, Intel Corporation
# 
#  SPDX-License-Identifier: BSD-3-Clause
 
# // Author: Filippov Ilia

import common
import sys
import os
import re
from pkg_resources import parse_version
print_debug = common.print_debug
error = common.error
take_lines = common.take_lines

exists = [False, False, False, False, False, False, False, False, False]
names = ["m4", "bison", "flex", "sde", "ispc", "clang", "gcc", "icc", "cmake"]

PATH_dir = os.environ["PATH"].split(os.pathsep)
for counter in PATH_dir:
    for i in range(0,len(exists)):
        if os.path.exists(counter + os.sep + names[i]):
            exists[i] = True

print_debug("=== in PATH: ===\n", False, "")
print_debug("Tools:\n", False, "")
for i in range(0,3):
    if exists[i]:
        print_debug(take_lines(names[i] + " --version", "first"), False, "")
    else:
        error("you don't have " + names[i], 0)
if exists[0] and exists[1] and exists[2]:
    if common.check_tools(2):
        print_debug("Tools' versions are ok\n", False, "")
print_debug("\nSDE:\n", False, "")
if exists[3]:
    print_debug(take_lines(names[3] + " --version", "first"), False, "")
else:
    error("you don't have " + names[3], 2)
print_debug("\nISPC:\n", False, "")
if exists[4]:
    print_debug(take_lines(names[4] + " --version", "first"), False, "")
else:
    error("you don't have " + names[4], 2)
print_debug("\nC/C++ compilers:\n", False, "")
for i in range(5,8):
    if exists[i]:
        print_debug(take_lines(names[i] + " --version", "first"), False, "")
    else:
        error("you don't have " + names[i], 2)
print_debug("\nCMake:\n", False, "")
if exists[8]:
    first_line = take_lines(names[8] + " --version", "first")
    matched_version = re.search(r"\d+\.\d+\.\d+", first_line)
    if matched_version is None:
        error("Unable to parse cmake version")
    else:
        cmake_version = matched_version.group(0)
        if (parse_version(cmake_version) >= parse_version("3.13.0")):
            print_debug(first_line, False, "")
        else:
            error("CMake version is older than needed. Please install version 3.8 or newer", 2)
else:
    error("you don't have " + names[8], 2)

print_debug("\n=== in ISPC specific environment variables: ===\n", False, "")
if os.environ.get("LLVM_HOME") == None:
    error("you have no LLVM_HOME", 2)
else:
    print_debug("Your LLVM_HOME:" + os.environ.get("LLVM_HOME") + "\n", False, "")
if os.environ.get("ISPC_HOME") == None:
    error("you have no ISPC_HOME", 2)
else:
    print_debug("Your ISPC_HOME:" + os.environ.get("ISPC_HOME") + "\n", False, "")
    if os.path.exists(os.environ.get("ISPC_HOME") + os.sep + "ispc"):
        print_debug("You have ISPC in your ISPC_HOME: " +
        take_lines(os.environ.get("ISPC_HOME") + os.sep + "ispc" + " --version", "first"), False, "")
    else:
        error("you don't have ISPC in your ISPC_HOME", 2)
if os.environ.get("SDE_HOME") == None:
    error("You have no SDE_HOME", 2)
else:
    print_debug("Your SDE_HOME:" + os.environ.get("SDE_HOME") + "\n", False, "")
    if os.path.exists(os.environ.get("SDE_HOME") + os.sep + "sde"):
        print_debug("You have sde in your SDE_HOME: " +
        take_lines(os.environ.get("SDE_HOME") + os.sep + "sde" + " --version", "first"), False, "")
    else:
        error("you don't have any SDE in your ISPC_HOME", 2)
