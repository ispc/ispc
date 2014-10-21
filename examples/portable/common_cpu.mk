
TASK_CXX=../omp_tasksys.cpp ../../util/ispc_malloc.cpp
TASK_LIB=-lpthread
TASK_OBJ=objs/omp_tasksys.o objs/ispc_malloc.o

CXX=clang++
CXX=icc -openmp
CXXFLAGS+=-Iobjs/ -O2 -I../../ -I../../util 
CXXFLAGS+=-DISPC_USE_OMP
CC=clang
CC=icc -openmp
CCFLAGS+=-Iobjs/ -O2 -I../../ -I../../util
CCFLAGS+=-DISPC_USE_OMP

LIBS=-lm $(TASK_LIB) -lstdc++
ISPC=ispc
ISPC_FLAGS+=-O2
ISPC_HEADER=objs/$(ISPC_SRC:.ispc=_ispc.h)

ARCH:=$(shell uname -m | sed -e s/x86_64/x86/ -e s/i686/x86/ -e s/arm.*/arm/ -e s/sa110/arm/)

ifeq ($(ARCH),x86)
  ISPC_OBJS=$(addprefix objs/, $(ISPC_SRC:.ispc=)_ispc.o)
  COMMA=,
  ifneq (,$(findstring $(COMMA),$(ISPC_IA_TARGETS)))
    #$(info multi-target detected: $(ISPC_IA_TARGETS))
    ifneq (,$(findstring sse2,$(ISPC_IA_TARGETS)))
      ISPC_OBJS+=$(addprefix objs/, $(ISPC_SRC:.ispc=)_ispc_sse2.o)
    endif
    ifneq (,$(findstring sse4,$(ISPC_IA_TARGETS)))
      ISPC_OBJS+=$(addprefix objs/, $(ISPC_SRC:.ispc=)_ispc_sse4.o)
    endif
    ifneq (,$(findstring avx1-,$(ISPC_IA_TARGETS)))
      ISPC_OBJS+=$(addprefix objs/, $(ISPC_SRC:.ispc=)_ispc_avx.o)
    endif
    ifneq (,$(findstring avx1.1,$(ISPC_IA_TARGETS)))
      ISPC_OBJS+=$(addprefix objs/, $(ISPC_SRC:.ispc=)_ispc_avx11.o)
    endif
    ifneq (,$(findstring avx2,$(ISPC_IA_TARGETS)))
      ISPC_OBJS+=$(addprefix objs/, $(ISPC_SRC:.ispc=)_ispc_avx2.o)
    endif
  endif
  ISPC_TARGETS=$(ISPC_IA_TARGETS)
  ARCH_BIT:=$(shell getconf LONG_BIT)
  ifeq ($(ARCH_BIT),32)
    ISPC_FLAGS += --arch=x86
    CXXFLAGS += -m32
    CCFLAGS += -m32
  else
    ISPC_FLAGS += --arch=x86-64
    CXXFLAGS += -m64
    CCFLAGS += -m64
  endif
else ifeq ($(ARCH),arm)
  ISPC_OBJS=$(addprefix objs/, $(ISPC_SRC:.ispc=_ispc.o))
  ISPC_TARGETS=$(ISPC_ARM_TARGETS)
else
  $(error Unknown architecture $(ARCH) from uname -m)
endif

CPP_OBJS=$(addprefix objs/, $(CPP_SRC:.cpp=.o))
CC_OBJS=$(addprefix objs/, $(CC_SRC:.c=.o))
OBJS=$(CPP_OBJS) $(CC_OBJS) $(TASK_OBJ) $(ISPC_OBJS)

default: $(EXAMPLE)

all: $(EXAMPLE) $(EXAMPLE)-sse4 $(EXAMPLE)-generic16 $(EXAMPLE)-scalar

.PHONY: dirs clean

dirs:
	/bin/mkdir -p objs/

objs/%.cpp objs/%.o objs/%.h: dirs

clean:
	/bin/rm -rf objs *~ $(EXAMPLE) $(EXAMPLE)-sse4 $(EXAMPLE)-generic16 ref test

$(EXAMPLE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

objs/%.o: %.cpp dirs $(ISPC_HEADER)
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/%.o: %.c dirs $(ISPC_HEADER)
	$(CC) $< $(CCFLAGS) -c -o $@

objs/%.o: ../%.cpp dirs
	$(CXX) $< $(CXXFLAGS) -c -o $@
objs/%.o: ../../%.cpp dirs
	$(CXX) $< $(CXXFLAGS) -c -o $@
objs/%.o: ../../util/%.cpp dirs
	$(CXX) $< $(CXXFLAGS) -c -o $@

objs/$(EXAMPLE).o: objs/$(EXAMPLE)_ispc.h dirs

objs/%_ispc.h objs/%_ispc.o objs/%_ispc_sse2.o objs/%_ispc_sse4.o objs/%_ispc_avx.o objs/%_ispc_avx11.o objs/%_ispc_avx2.o: %.ispc dirs
	$(ISPC) $(ISPC_FLAGS) --target=$(ISPC_TARGETS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h

objs/$(ISPC_SRC:.ispc=)_sse4.cpp: $(ISPC_SRC)
	$(ISPC) $(ISPC_FLAGS) $< -o $@ --target=generic-4 --emit-c++ --c++-include-file=sse4.h

objs/$(ISPC_SRC:.ispc=)_sse4.o: objs/$(ISPC_SRC:.ispc=)_sse4.cpp
	$(CXX) -I../../intrinsics -msse4.2 $< $(CXXFLAGS) -c -o $@

$(EXAMPLE)-sse4: $(CPP_OBJS) objs/$(ISPC_SRC:.ispc=)_sse4.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

objs/$(ISPC_SRC:.ispc=)_generic16.cpp: $(ISPC_SRC)
	$(ISPC) $(ISPC_FLAGS) $< -o $@ --target=generic-16 --emit-c++ --c++-include-file=generic-16.h

objs/$(ISPC_SRC:.ispc=)_generic16.o: objs/$(ISPC_SRC:.ispc=)_generic16.cpp
	$(CXX) -I../../intrinsics $< $(CXXFLAGS) -c -o $@

$(EXAMPLE)-generic16: $(CPP_OBJS) objs/$(ISPC_SRC:.ispc=)_generic16.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

objs/$(ISPC_SRC:.ispc=)_scalar.o: $(ISPC_SRC)
	$(ISPC) $(ISPC_FLAGS) $< -o $@ --target=generic-1

$(EXAMPLE)-scalar: $(CPP_OBJS) objs/$(ISPC_SRC:.ispc=)_scalar.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)
