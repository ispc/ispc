#
# ispc Makefile
#

ARCH_OS = $(shell uname)
ifeq ($(ARCH_OS), Darwin)
	ARCH_OS2 = "OSX"
else
	ARCH_OS2 = $(shell uname -o)
endif
ARCH_TYPE = $(shell arch)

ifeq ($(shell llvm-config --version), 3.1svn)
  LLVM_LIBS=-lLLVMAsmParser -lLLVMInstrumentation -lLLVMLinker			\
	-lLLVMArchive -lLLVMBitReader -lLLVMDebugInfo -lLLVMJIT -lLLVMipo	\
	-lLLVMBitWriter -lLLVMTableGen -lLLVMCBackendInfo			\
	-lLLVMX86Disassembler -lLLVMX86CodeGen -lLLVMSelectionDAG		\
	-lLLVMAsmPrinter -lLLVMX86AsmParser -lLLVMX86Desc -lLLVMX86Info		\
	-lLLVMX86AsmPrinter -lLLVMX86Utils -lLLVMMCDisassembler	-lLLVMMCParser	\
	-lLLVMCodeGen -lLLVMScalarOpts	-lLLVMInstCombine -lLLVMTransformUtils	\
	-lLLVMipa -lLLVMAnalysis -lLLVMMCJIT -lLLVMRuntimeDyld			\
	-lLLVMExecutionEngine -lLLVMTarget -lLLVMMC -lLLVMObject -lLLVMCore 	\
	-lLLVMSupport
else
  LLVM_LIBS=$(shell llvm-config --libs)
endif

CLANG=clang
CLANG_LIBS = -lclangFrontend -lclangDriver \
             -lclangSerialization -lclangParse -lclangSema \
             -lclangAnalysis -lclangAST -lclangLex -lclangBasic

ISPC_LIBS=$(shell llvm-config --ldflags) $(CLANG_LIBS) $(LLVM_LIBS) \
	-lpthread

ifeq ($(ARCH_OS),Linux)
	ISPC_LIBS += -ldl
endif

ifeq ($(ARCH_OS2),Msys)
	ISPC_LIBS += -lshlwapi -limagehlp -lpsapi
endif

LLVM_CXXFLAGS=$(shell llvm-config --cppflags)
LLVM_VERSION=LLVM_$(shell llvm-config --version | sed s/\\./_/)
LLVM_VERSION_DEF=-D$(LLVM_VERSION)

BUILD_DATE=$(shell date +%Y%m%d)
BUILD_VERSION=$(shell git log --abbrev-commit --abbrev=16 | head -1)

CXX=g++
CPP=cpp
OPT=-g3
CXXFLAGS=$(OPT) $(LLVM_CXXFLAGS) -I. -Iobjs/ -Wall $(LLVM_VERSION_DEF) \
	-DBUILD_DATE="\"$(BUILD_DATE)\"" -DBUILD_VERSION="\"$(BUILD_VERSION)\""

LDFLAGS=
ifeq ($(ARCH_OS),Linux)
  # try to link everything statically under Linux (including libstdc++) so
  # that the binaries we generate will be portable across distributions...
    LDFLAGS=-static
endif

LEX=flex
YACC=bison -d -v -t

###########################################################################

CXX_SRC=ast.cpp builtins.cpp cbackend.cpp ctx.cpp decl.cpp expr.cpp func.cpp \
	ispc.cpp llvmutil.cpp main.cpp module.cpp opt.cpp stmt.cpp sym.cpp \
	type.cpp util.cpp
HEADERS=ast.h builtins.h ctx.h decl.h expr.h func.h ispc.h llvmutil.h module.h \
	opt.h stmt.h sym.h type.h util.h
TARGETS=avx1 avx1-x2 avx2 avx2-x2 sse2 sse2-x2 sse4 sse4-x2 generic-4 generic-8 \
	generic-16 generic-1
BUILTINS_SRC=$(addprefix builtins/target-, $(addsuffix .ll, $(TARGETS))) \
	builtins/dispatch.ll
BUILTINS_OBJS=$(addprefix builtins-, $(notdir $(BUILTINS_SRC:.ll=.o))) \
	builtins-c-32.cpp builtins-c-64.cpp 
BISON_SRC=parse.yy
FLEX_SRC=lex.ll

OBJS=$(addprefix objs/, $(CXX_SRC:.cpp=.o) $(BUILTINS_OBJS) \
	stdlib_generic_ispc.o stdlib_x86_ispc.o \
	$(BISON_SRC:.yy=.o) $(FLEX_SRC:.ll=.o))

default: ispc

.PHONY: dirs clean depend doxygen print_llvm_src
.PRECIOUS: objs/builtins-%.cpp

depend: $(CXX_SRC) $(HEADERS)
	@echo Updating dependencies
	@gcc -MM $(CXXFLAGS) $(CXX_SRC) | sed 's_^\([a-z]\)_objs/\1_g' > depend

-include depend

dirs:
	@echo Creating objs/ directory
	@/bin/mkdir -p objs

print_llvm_src:
	@echo Using LLVM `llvm-config --version` from `llvm-config --libdir`

clean:
	/bin/rm -rf objs ispc

doxygen:
	/bin/rm -rf docs/doxygen
	doxygen doxygen.cfg

ispc: print_llvm_src dirs $(OBJS)
	@echo Creating ispc executable
	@$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(ISPC_LIBS)

objs/%.o: %.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/cbackend.o: cbackend.cpp
	@echo Compiling $<
	@$(CXX) -fno-rtti -fno-exceptions $(CXXFLAGS) -o $@ -c $<

objs/%.o: objs/%.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/parse.cc: parse.yy
	@echo Running bison on $<
	@$(YACC) -o $@ $<

objs/parse.o: objs/parse.cc $(HEADERS)
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/lex.cpp: lex.ll 
	@echo Running flex on $<
	@$(LEX) -o $@ $<

objs/lex.o: objs/lex.cpp $(HEADERS) objs/parse.cc
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/builtins-%.cpp: builtins/%.ll builtins/util.m4 $(wildcard builtins/*common.ll)
	@echo Creating C++ source from builtins definition file $<
	@m4 -Ibuiltins/ -DLLVM_VERSION=$(LLVM_VERSION) $< | python bitcode2cpp.py $< > $@

objs/builtins-c-32.cpp: builtins/builtins.c
	@echo Creating C++ source from builtins definition file $<
	@$(CLANG) -m32 -emit-llvm -c $< -o - | llvm-dis - | python bitcode2cpp.py c-32 > $@

objs/builtins-c-64.cpp: builtins/builtins.c
	@echo Creating C++ source from builtins definition file $<
	@$(CLANG) -m64 -emit-llvm -c $< -o - | llvm-dis - | python bitcode2cpp.py c-64 > $@

objs/stdlib_generic_ispc.cpp: stdlib.ispc
	@echo Creating C++ source from $< for generic
	@$(CLANG) -E -x c -DISPC_TARGET_GENERIC=1 -DISPC=1 -DPI=3.1415926536 $< -o - | \
		python stdlib2cpp.py generic > $@

objs/stdlib_x86_ispc.cpp: stdlib.ispc
	@echo Creating C++ source from $< for x86
	@$(CLANG) -E -x c -DISPC=1 -DPI=3.1415926536 $< -o - | \
		python stdlib2cpp.py x86 > $@
