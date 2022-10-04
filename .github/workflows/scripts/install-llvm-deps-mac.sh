#!/bin/bash -e
ls -ald /Applications/Xcode*
xcrun --show-sdk-path
# There are several Xcode versions installed on GHA runnner.
# /Applications/Xcode.app is a symlink pointing to the one that needs to be used.
# But the one, which is currently "selected" doesn't use symlink.
# We need canonical location to make resulting clang build working on other machines.
sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer
xcrun --show-sdk-path
mkdir llvm
echo "LLVM_HOME=${GITHUB_WORKSPACE}/llvm" >> $GITHUB_ENV
echo "ISPC_HOME=${GITHUB_WORKSPACE}" >> $GITHUB_ENV

