#!/bin/bash -e
echo "APT::Acquire::Retries \"3\";" | sudo tee -a /etc/apt/apt.conf.d/80-retries

# if apt-get fails, retry several time.
for i in {1..5}
do
  sudo apt-get update | tee log${i}.txt
  sudo apt-get install libc6-dev-i386 g++-multilib lib32stdc++6 | tee -a log${i}.txt
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
# Remark about user agent: it might or might now work with default user agent, but
# from time to time the settings are changed and browser-like user agent is required to make it work.
wget -U "$USER_AGENT" --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://software.intel.com/content/dam/develop/external/us/en/documents/downloads/"$SDE_TAR_NAME"-lin.tar.bz2
tar xf "$SDE_TAR_NAME"-lin.tar.bz2
tar xf ispc-trunk-linux.tar.gz

#GA requires to set env putting value to $GITHUB_ENV & $GITHUB_PATH
echo "SDE_HOME=$GITHUB_WORKSPACE/$SDE_TAR_NAME-lin" >> $GITHUB_ENV
echo "$GITHUB_WORKSPACE/ispc-trunk-linux/bin" >> $GITHUB_PATH
echo "ISPC_HOME=$GITHUB_WORKSPACE" >> $GITHUB_ENV
echo "LLVM_HOME=$GITHUB_WORKSPACE" >> $GITHUB_ENV
