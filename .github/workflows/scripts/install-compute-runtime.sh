#!/bin/bash -e

wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17384.11/intel-igc-core_1.0.17384.11_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17384.11/intel-igc-opencl_1.0.17384.11_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.31.30508.7/intel-level-zero-gpu-dbgsym_1.3.30508.7_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/24.31.30508.7/intel-level-zero-gpu_1.3.30508.7_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.31.30508.7/intel-opencl-icd-dbgsym_24.31.30508.7_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/24.31.30508.7/intel-opencl-icd_24.31.30508.7_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.31.30508.7/libigdgmm12_22.4.1_amd64.deb

sudo apt install ocl-icd-libopencl1
sudo dpkg -i *.deb
