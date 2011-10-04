
ARCH = $(shell uname)

TASK_CXX=../tasksys.cpp
TASK_LIB=-lpthread
TASK_OBJ=$(addprefix objs/, $(subst ../,, $(TASK_CXX:.cpp=.o)))

CXX=g++
CXXFLAGS=-Iobjs/ -O3 -Wall -m64
ISPC=ispc
ISPCFLAGS=-O2 --target=sse2,sse4,avx --arch=x86-64

ISPC_OBJS=objs/ao_ispc.o objs/ao_ispc_sse2.o objs/ao_ispc_sse4.o \
	objs/ao_ispc_avx.o
OBJS=objs/ao.o objs/ao_serial.o $(ISPC_OBJS) $(TASK_OBJ)

default: ao

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ ao

ao: dirs $(OBJS) $(TASK_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm $(TASK_LIB)

objs/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/%.o: ../%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/ao.o: objs/ao_ispc.h 

objs/%_ispc.h objs/%_ispc.o objs/%_ispc_sse2.o objs/%_ispc_sse4.o objs/%_ispc_avx.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
