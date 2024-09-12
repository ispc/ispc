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
echo -U "$USER_AGENT" --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://downloadmirror.intel.com/"$SDE_MIRROR_ID"/"$SDE_TAR_NAME"-lin.tar.xz
wget -q -U "$USER_AGENT" --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://downloadmirror.intel.com/"$SDE_MIRROR_ID"/"$SDE_TAR_NAME"-lin.tar.xz
tar xf "$SDE_TAR_NAME"-lin.tar.xz
tar xf ispc-trunk-linux.tar.gz

if [ -v INSTALL_COMPUTE_RUNTIME ]; then
    echo "install Compute Runtime"
    wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy unified' > /tmp/intel.gpu.focal.list
    sudo mv /tmp/intel.gpu.focal.list /etc/apt/sources.list.d/
    sudo apt-get -y update
    sudo apt-get --no-install-recommends install -y intel-opencl-icd \
        intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
        libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
        libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
        mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all
fi

#GA requires to set env putting value to $GITHUB_ENV & $GITHUB_PATH
echo "SDE_HOME=$GITHUB_WORKSPACE/$SDE_TAR_NAME-lin" >> $GITHUB_ENV
echo "$GITHUB_WORKSPACE/ispc-trunk-linux/bin" >> $GITHUB_PATH
echo "ISPC_HOME=$GITHUB_WORKSPACE" >> $GITHUB_ENV
echo "LLVM_HOME=$GITHUB_WORKSPACE" >> $GITHUB_ENV
