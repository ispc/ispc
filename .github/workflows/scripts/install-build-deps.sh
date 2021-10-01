#!/bin/bash -e
echo "APT::Acquire::Retries \"3\";" | sudo tee -a /etc/apt/apt.conf.d/80-retries

# if apt-get fails, retry several time.
for i in {1..5}
do
  sudo apt-get update | tee log${i}.txt
  sudo apt-get install bison flex libc6-dev-i386 g++-multilib lib32stdc++6 ncurses-dev | tee -a log${i}.txt
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

[ -n "$LLVM_REPO" ] && wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 $LLVM_REPO/releases/download/llvm-$LLVM_VERSION-ispc-dev/$LLVM_TAR
tar xf $LLVM_TAR
echo "${GITHUB_WORKSPACE}/bin-$LLVM_VERSION/bin" >> $GITHUB_PATH
