
EXAMPLE=mandelbrot
CPP_SRC=mandelbrot.cpp mandelbrot_serial.cpp
ISPC_SRC=mandelbrot.ispc
ISPC_TARGETS=sse2,sse4-x2,avx-x2

mandelbrot_1wide: objs/mandelbrot.o objs/mandelbrot_serial.o objs/tasksys.o 
	ispc -O2 --arch=x86-64  --target=generic-1 mandelbrot.ispc -o objs/mandelbrot_ispc_generic.o -h objs/mandelbrot_ispc.h
	g++ -o $@ $^ objs/mandelbrot_ispc_generic.o -lm -lpthread -lstdc++

include ../common.mk
