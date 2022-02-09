#!/bin/bash -e
echo "APT::Acquire::Retries \"3\";" | sudo tee -a /etc/apt/apt.conf.d/80-retries

# SVML requires installing Intel Compiler

wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"

# if apt-get fails, retry several time.
for i in {1..5}
do
  sudo apt-get update 2>&1 | tee log${i}.txt
  sudo apt-get install libc6-dev-i386 g++-multilib lib32stdc++6 2>&1 | tee -a log${i}.txt
  sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic-2021.2.0 2>&1 | tee -a log${i}.txt
  if [[ ! `grep "^Err: " log${i}.txt` && ! `grep "^E: " log${i}.txt` ]]; then
    echo "APT packages installation was successful"
    break
  else
    if [[ ${i} -eq 5 ]]; then
      echo "APT had unrecoverable errors, exiting"
      exit 1
    else
      sleep_time=$((${i} * 10))
      echo "APT packages installation failed, sleeping ${sleep_time} seconds"
      sleep ${sleep_time}
      sudo rm -rf /var/lib/apt/lists/*
    fi
  fi
done

find /usr -name cdefs.h || echo "Find errors were ignored"
wget -U "$USER_AGENT" --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://downloadmirror.intel.com/"$SDE_MIRROR_ID"/"$SDE_TAR_NAME"-lin.tar.xz
tar xf "$SDE_TAR_NAME"-lin.tar.bz2
tar xf ispc-trunk-linux.tar.gz

#GA requires to set env putting value to $GITHUB_ENV & $GITHUB_PATH
echo "SDE_HOME=$GITHUB_WORKSPACE/$SDE_TAR_NAME-lin" >> $GITHUB_ENV
echo "$GITHUB_WORKSPACE/ispc-trunk-linux/bin" >> $GITHUB_PATH
echo "ISPC_HOME=$GITHUB_WORKSPACE" >> $GITHUB_ENV
echo "LLVM_HOME=$GITHUB_WORKSPACE" >> $GITHUB_ENV
