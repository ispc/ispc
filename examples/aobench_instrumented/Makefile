
CXX=clang++ -m64
CXXFLAGS=-Iobjs/ -g3 -Wall
ISPC=ispc
ISPCFLAGS=-O2 --instrument --arch=x86-64 --target=sse2

default: ao

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ ao

ao: objs/ao.o objs/instrument.o objs/ao_instrumented_ispc.o ../tasksys.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm -lpthread

objs/%.o: %.cpp dirs
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/ao.o: objs/ao_instrumented_ispc.h

objs/%_ispc.h objs/%_ispc.o: %.ispc dirs
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
