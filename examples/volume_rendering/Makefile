
ARCH = $(shell uname)

TASK_CXX=../tasksys.cpp
TASK_LIB=-lpthread
TASK_OBJ=$(addprefix objs/, $(subst ../,, $(TASK_CXX:.cpp=.o)))

CXX=g++
CXXFLAGS=-Iobjs/ -O3 -Wall -m64
ISPC=ispc
ISPCFLAGS=-O2 --target=sse2,sse4-x2 --arch=x86-64

OBJS=objs/volume.o objs/volume_serial.o $(TASK_OBJ) objs/volume_ispc.o \
	objs/volume_ispc_sse2.o objs/volume_ispc_sse4.o

default: volume

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ volume

volume: dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm $(TASK_LIB)

objs/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/%.o: ../%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/volume.o: objs/volume_ispc.h 

objs/%_ispc.h objs/%_ispc.o objs/%_ispc_sse2.o objs/%_ispc_sse4.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
