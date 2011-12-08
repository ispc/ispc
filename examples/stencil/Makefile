
ARCH = $(shell uname)

TASK_CXX=../tasksys.cpp
TASK_LIB=-lpthread
TASK_OBJ=$(addprefix objs/, $(subst ../,, $(TASK_CXX:.cpp=.o)))

CXX=g++
CXXFLAGS=-Iobjs/ -O3 -Wall -m64
ISPC=ispc
ISPCFLAGS=-O2 --target=sse2,sse4-x2,avx-x2 --arch=x86-64

OBJS=objs/stencil.o objs/stencil_serial.o $(TASK_OBJ) objs/stencil_ispc.o \
	objs/stencil_ispc_sse2.o objs/stencil_ispc_sse4.o \
	objs/stencil_ispc_avx.o

default: stencil

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ stencil

stencil: dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm $(TASK_LIB)

objs/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/%.o: ../%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/stencil.o: objs/stencil_ispc.h 

objs/%_ispc.h objs/%_ispc.o objs/%_ispc_sse2.o objs/%_ispc_sse4.o objs/%_ispc_avx.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
