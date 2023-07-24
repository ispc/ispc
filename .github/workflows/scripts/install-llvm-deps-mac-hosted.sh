#!/bin/bash -e
ls -ald /Applications/Xcode*
xcrun --show-sdk-path
mkdir -p llvm
rm -rf llvm/*
echo "LLVM_HOME=${GITHUB_WORKSPACE}/llvm" >> $GITHUB_ENV
echo "ISPC_HOME=${GITHUB_WORKSPACE}" >> $GITHUB_ENV

