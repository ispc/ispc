
TASK_CXX=../tasksys.cpp
TASK_LIB=-lpthread
TASK_OBJ=tasksys.o

CXX=g++
CXXFLAGS=-Iobjs/ -O2 -m64
LIBS=-lm $(TASK_LIB) -lstdc++
ISPC=ispc -O2 --arch=x86-64 $(ISPC_FLAGS)
ISPC_OBJS=$(addprefix objs/, $(ISPC_SRC:.ispc=)_ispc.o $(ISPC_SRC:.ispc=)_ispc_sse2.o \
	$(ISPC_SRC:.ispc=)_ispc_sse4.o $(ISPC_SRC:.ispc=)_ispc_avx.o)
ISPC_HEADER=objs/$(ISPC_SRC:.ispc=_ispc.h)
CPP_OBJS=$(addprefix objs/, $(CPP_SRC:.cpp=.o) $(TASK_OBJ))

default: $(EXAMPLE)

all: $(EXAMPLE) $(EXAMPLE)-sse4 $(EXAMPLE)-generic16 $(EXAMPLE)-scalar

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

objs/%.cpp objs/%.o objs/%.h: dirs

clean:
	/bin/rm -rf objs *~ $(EXAMPLE) $(EXAMPLE)-sse4 $(EXAMPLE)-generic16

$(EXAMPLE): $(CPP_OBJS) $(ISPC_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

objs/%.o: %.cpp dirs $(ISPC_HEADER)
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/%.o: ../%.cpp dirs
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/$(EXAMPLE).o: objs/$(EXAMPLE)_ispc.h

objs/%_ispc.h objs/%_ispc.o objs/%_ispc_sse2.o objs/%_ispc_sse4.o objs/%_ispc_avx.o: %.ispc
	$(ISPC) --target=$(ISPC_TARGETS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h

objs/$(ISPC_SRC:.ispc=)_sse4.cpp: $(ISPC_SRC)
	$(ISPC) $< -o $@ --target=generic-4 --emit-c++ --c++-include-file=sse4.h

objs/$(ISPC_SRC:.ispc=)_sse4.o: objs/$(ISPC_SRC:.ispc=)_sse4.cpp
	$(CXX) -I../intrinsics -msse4.2 $< $(CXXFLAGS) -c -o $@

$(EXAMPLE)-sse4: $(CPP_OBJS) objs/$(ISPC_SRC:.ispc=)_sse4.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

objs/$(ISPC_SRC:.ispc=)_generic16.cpp: $(ISPC_SRC)
	$(ISPC) $< -o $@ --target=generic-16 --emit-c++ --c++-include-file=generic-16.h

objs/$(ISPC_SRC:.ispc=)_generic16.o: objs/$(ISPC_SRC:.ispc=)_generic16.cpp
	$(CXX) -I../intrinsics $< $(CXXFLAGS) -c -o $@

$(EXAMPLE)-generic16: $(CPP_OBJS) objs/$(ISPC_SRC:.ispc=)_generic16.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

objs/$(ISPC_SRC:.ispc=)_scalar.o: $(ISPC_SRC)
	$(ISPC) $< -o $@ --target=generic-1

$(EXAMPLE)-scalar: $(CPP_OBJS) objs/$(ISPC_SRC:.ispc=)_scalar.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)
