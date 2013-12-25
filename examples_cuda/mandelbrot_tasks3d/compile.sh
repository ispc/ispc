#!/bin/sh
ptxas -arch=sm_35 -c -o kernel.gpu.o kernel_cu.ptx       
fatbinary -arch=sm_35 -create kernel.fatbin -elf kernel.gpu.o 
nvcc -arch=sm_35 -Xptxas="-v" -dlink -o mandel_cu.o kernel.fatbin kernel_driver.cu  -rdc=true -lcudadevrt

