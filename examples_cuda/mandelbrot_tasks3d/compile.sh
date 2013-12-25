#!/bin/sh
ptxas -arch=sm_35 -c -o kernel.gpu.o kernel_cu.ptx       
fatbinary -arch=sm_35 -create kernel.fatbin -elf kernel.gpu.o 
nvcc -arch=sm_35 -Xptxas="-v" -dc  kernel_driver.cu   -lcudadevrt
nvcc -arch=sm_35 -Xptxas="-v" -dlink -o mandel_nvcc.o kernel.fatbin kernel_driver.o  -rdc=true -lcudadevrt

