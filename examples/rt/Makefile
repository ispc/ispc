
CXX=g++
CXXFLAGS=-Iobjs/ -O3 -Wall
ISPC=ispc
ISPCFLAGS=-O2 --target=sse4x2

default: rt

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ rt

rt: dirs objs/rt.o objs/rt_serial.o objs/rt_ispc.o
	$(CXX) $(CXXFLAGS) -o $@ objs/rt.o objs/rt_ispc.o objs/rt_serial.o -lm

objs/%.o: %.cpp objs/rt_ispc.h
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/%_ispc.h objs/%_ispc.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
