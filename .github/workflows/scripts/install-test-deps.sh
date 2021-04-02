#!/bin/bash -e
echo "APT::Acquire::Retries \"3\";" | sudo tee -a /etc/apt/apt.conf.d/80-retries
sudo apt-get update && sudo apt-get install libc6-dev-i386 g++-multilib lib32stdc++6
find /usr -name cdefs.h || echo "Find errors were ignored"
wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://software.intel.com/content/dam/develop/external/us/en/documents/downloads/"$SDE_TAR_NAME"-lin.tar.bz2
tar xf "$SDE_TAR_NAME"-lin.tar.bz2
tar xf ispc-trunk-linux.tar.gz

#GA requires to set env putting value to $GITHUB_ENV & $GITHUB_PATH
echo "SDE_HOME=$GITHUB_WORKSPACE/$SDE_TAR_NAME-lin" >> $GITHUB_ENV
echo "$GITHUB_WORKSPACE/ispc-trunk-linux/bin" >> $GITHUB_PATH
echo "ISPC_HOME=$GITHUB_WORKSPACE" >> $GITHUB_ENV
echo "LLVM_HOME=$GITHUB_WORKSPACE" >> $GITHUB_ENV
