#!/bin/bash
# ##################################################
#  Copyright (c) 2020-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#   Author: Arina Neshlyaeva

set -o errexit
scriptName="${BASH_SOURCE[0]}"
scriptPath="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Flags which can be overridden by user input
# Default values are below
ispc_home=${scriptPath}/..
llvm_dir=${ispc_home}/llvm
llvm_versiion=13.0
ispc_build=${ispc_home}/build
ispc_install=${ispc_home}/install
speed=4

# Print usage
usage() {
  echo -n "${scriptName} [OPTION]...

This is a script to build ISPC from scratch, including building patched version of LLVM.
NOTE THAT RELEASE VERSION OF ISPC MUST BE BUILT WITH PATCHED VERSION OF LLVM.

 Options:
  -v, --version       LLVM version to build with. ${llvm_version} is currently recommended.
  -h, --ispc_home     Path to ISPC HOME directory.
  -l, --llvm_dir      Path to use for LLVM build. ${llvm_dir} is default.
  -b, --ispc_build    Path to ISPC build directory. ${ispc_build} is default.
  -i, --ispc_install  Path to ISPC install directory. ${ispc_install} is default.
  -j, --speed         Number of threads for build. ${speed} is default.
  --help              Display this help and exit.
"
}

# Read the options and set the flags
while [[ $1 = -?* ]]; do
  case $1 in
    --help) usage >&2; exit 0 ;;
    -v|--version) shift; llvm_version=${1} ;;
    -l|--llvm_dir) shift; llvm_dir=${1} ;;
    -h|--ispc_home) shift; ispc_home=${1} ;;
    -b|--ispc_build) shift; ispc_build=${1} ;;
    -i|--ispc_install) shift; ispc_install=${1} ;;
    -j) shift; speed=${1};;
    *) die "invalid option: '$1'." ;;
  esac
  shift
done

# Check environment
declare -a required=("python3" "cmake" "bison" "flex" "m4" "make")
for i in "${required[@]}"
do
   command -v $i >/dev/null 2>&1 || { echo >&2 "$i is required for the build but not found. Aborting."; exit 1; }
done

# Create directory to LLVM sources and binaries if it does not exist
mkdir -p ${llvm_dir}

# Set necessary environment variables for alloy script
export LLVM_HOME=${llvm_dir}
export ISPC_HOME=${ispc_home}
export LLVM_VERSION=${llvm_version}

# Run alloy.py to checkout, patch and build LLVM
python3 ${ISPC_HOME}/alloy.py -b --version=${llvm_version} --selfbuild -j ${speed} && \
    rm -rf ${LLVM_HOME}/build-${LLVM_VERSION} ${LLVM_HOME}/llvm-${LLVM_VERSION} ${LLVM_HOME}/bin-${LLVM_VERSION}_temp ${LLVM_HOME}/build-${LLVM_VERSION}_temp
exitCode=$?; if [[ ${exitCode} != 0 ]]; then exit ${exitCode}; fi
export PATH=${LLVM_HOME}/bin-${LLVM_VERSION}/bin:$PATH

# Configure ISPC build
mkdir -p ${ispc_build} && cd ${ispc_build}
cmake ../ -DCMAKE_INSTALL_PREFIX=${ispc_install} -DISPC_INCLUDE_EXAMPLES=OFF -DISPC_INCLUDE_UTILS=OFF
exitCode=$?; if [[ ${exitCode} != 0 ]]; then exit ${exitCode}; fi

# Build ISPC
make ispc -j${speed} install

