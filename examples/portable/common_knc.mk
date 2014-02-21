TASK_CXX=../omp_tasksys.cpp ../../util/ispc_malloc.cpp
TASK_OBJ=objs_knc/omp_tasksys.o objs_knc/ispc_malloc.o
TASK_LIB=-openmp

CXX=icc -openmp -mmic
CXXFLAGS+=-Iobjs_knc/ -O2 -I../../ -I../../util  -I./
CXXFLAGS+=  -DISPC_USE_OMP
CC=icc -openmp -mmic
CCFLAGS+= -Iobjs_knc/ -O2 -I../../ -I../../util -I./
CCFLAGS+=-DISPC_USE_OMP

LD=icc -mmic -openmp

LIBS=-lm $(TASK_LIB) -lstdc++
ISPC=ispc
ISPC_FLAGS+=-O2
ISPC_FLAGS+= --target=$(ISPC_TARGET) --c++-include-file=$(ISPC_INTRINSICS)

ISPC_HEADERS=$(ISPC_SRC:%.ispc=objs_knc/%_ispc.h)
ISPC_OBJ=$(ISPC_SRC:%.ispc=objs_knc/%_ispc.o)
CXX_OBJ=$(CXX_SRC:%.cpp=objs_knc/%.o)
CXX_OBJ+=$(TASK_OBJ)

PROG=$(EXAMPLE)_knc

all: dirs $(PROG)

dirs:
	/bin/mkdir -p objs_knc/

objs_knc/%.cpp objs_knc/%.o objs_knc/%.h: dirs

clean: 
	/bin/rm -rf $(PROG) objs_knc

$(PROG): $(ISPC_OBJ) $(CXX_OBJ) 
	$(LD) -o $@ $^ $(LDFLAGS)

objs_knc/%.o: %.cpp
	$(CXX) $(CXXFLAGS)  -o $@ -c $<

objs_knc/%.o: ../%.cpp
	$(CXX) $(CXXFLAGS)  -o $@ -c $<
objs_knc/%.o: ../../%.cpp
	$(CXX) $(CXXFLAGS)  -o $@ -c $<
objs_knc/%.o: ../../util/%.cpp
	$(CXX) $(CXXFLAGS)  -o $@ -c $<

objs_knc/%_ispc.o: %.ispc
	$(ISPC) $(ISPC_FLAGS) --emit-c++ -o objs_knc/$*_ispc_zmm.cpp -h objs_knc/$*_ispc.h $< 
	$(CXX) $(CXXFLAGS) -o $@ objs_knc/$*_ispc_zmm.cpp  -c

