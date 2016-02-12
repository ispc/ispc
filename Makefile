#
#  Copyright (c) 2010-2016, Intel Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# ispc Makefile
#

define newline


endef

define WARNING_BODY
 ============================== !!! WARNING !!! =============================== \n
Location of LLVM files in your PATH is different than path in LLVM_HOME \n
variable (or LLVM_HOME is not set). The most likely this means that you are \n
using default LLVM installation on your system, which is very bad sign. \n
Note, that ISPC uses LLVM optimizer and is highly dependent on it. We recommend \n
using *patched* version of LLVM 3.4 or 3.5. Patches are availible in \n
llvm_patches folder. You can build LLVM manually, or run our scripts, which \n
will do all the work for you. Do the following: \n
1. Create a folder, where LLVM will reside and set LLVM_HOME variable to its \n
  path. \n
2. Set ISPC_HOME variable to your ISPC location (probably current folder).
3. Run alloy.py tool to checkout and build LLVM: \n
  alloy.py -b --version=3.5 \n
4. Add $$LLVM_HOME/bin-3.5/bin path to your PATH. \n
==============================================================================
endef

# If you have your own special version of llvm and/or clang, change
# these variables to match.
LLVM_CONFIG=$(shell which llvm-config)
CLANG_INCLUDE=$(shell $(LLVM_CONFIG) --includedir)

RIGHT_LLVM = $(WARNING_BODY)
ifdef LLVM_HOME
	ifeq ($(findstring $(LLVM_HOME), $(LLVM_CONFIG)), $(LLVM_HOME))
		RIGHT_LLVM = LLVM from $$LLVM_HOME is used.
	endif
endif

# Enable ARM by request
# To enable: make ARM_ENABLED=1
ARM_ENABLED=0

# Disable NVPTX by request
# To enable: make NVPTX_ENABLED=1
NVPTX_ENABLED=0

# Add llvm bin to the path so any scripts run will go to the right llvm-config
LLVM_BIN= $(shell $(LLVM_CONFIG) --bindir)
export PATH:=$(LLVM_BIN):$(PATH)

ARCH_OS = $(shell uname)
ifeq ($(ARCH_OS), Darwin)
	ARCH_OS2 = "OSX"
else
	ARCH_OS2 = $(shell uname -o)
endif
ARCH_TYPE = $(shell arch)

DNDEBUG_FLAG=$(shell $(LLVM_CONFIG) --cxxflags | grep -o "\-DNDEBUG")
LLVM_CXXFLAGS=$(shell $(LLVM_CONFIG) --cppflags) $(DNDEBUG_FLAG)
LLVM_VERSION=LLVM_$(shell $(LLVM_CONFIG) --version | sed -e 's/svn//' -e 's/\./_/' -e 's/\..*//')
LLVM_VERSION_DEF=-D$(LLVM_VERSION)

LLVM_COMPONENTS = engine ipo bitreader bitwriter instrumentation linker 
# Component "option" was introduced in 3.3 and starting with 3.4 it is required for the link step.
# We check if it's available before adding it (to not break 3.2 and earlier).
ifeq ($(shell $(LLVM_CONFIG) --components |grep -c option), 1)
    LLVM_COMPONENTS+=option
endif
ifneq ($(ARM_ENABLED), 0)
    LLVM_COMPONENTS+=arm
endif
ifneq ($(NVPTX_ENABLED), 0)
    LLVM_COMPONENTS+=nvptx
endif	
LLVM_LIBS=$(shell $(LLVM_CONFIG) --libs $(LLVM_COMPONENTS))

CLANG=clang
CLANG_LIBS = -lclangFrontend -lclangDriver \
             -lclangSerialization -lclangParse -lclangSema \
             -lclangAnalysis -lclangAST -lclangBasic \
             -lclangEdit -lclangLex

ISPC_LIBS=$(shell $(LLVM_CONFIG) --ldflags) $(CLANG_LIBS) $(LLVM_LIBS) \
	-lpthread

ifeq ($(LLVM_VERSION),LLVM_3_4)
    ISPC_LIBS += -lcurses
endif

# There is no logical OR in GNU make. 
# This 'ifneq' acts like if( !($(LLVM_VERSION) == LLVM_3_2 || $(LLVM_VERSION) == LLVM_3_3 || $(LLVM_VERSION) == LLVM_3_4))
ifeq (,$(filter $(LLVM_VERSION), LLVM_3_2 LLVM_3_3 LLVM_3_4))
    ISPC_LIBS += -lcurses -lz
    # This is here because llvm-config fails to report dependency on tinfo library in some case.
    # This is described in LLVM bug 16902.
    ifeq ($(ARCH_OS),Linux)
        ifneq ($(shell ldconfig -p |grep -c tinfo), 0)
            ISPC_LIBS += -ltinfo
	endif
    endif
endif

ifeq ($(ARCH_OS),Linux)
	ISPC_LIBS += -ldl
endif

ifeq ($(ARCH_OS2),Msys)
	ISPC_LIBS += -lshlwapi -limagehlp -lpsapi
endif

# Define build time stamp and revision.
# For revision we use GIT or SVN info.
BUILD_DATE=$(shell date +%Y%m%d)
GIT_REVISION:=$(shell git log --abbrev-commit --abbrev=16 2>/dev/null | head -1)
ifeq (${GIT_REVISION},)
    SVN_REVISION:=$(shell svn log -l 1 2>/dev/null | grep -o \^r[[:digit:]]\* )
    ifeq (${SVN_REVISION},)
        # Failed to get revision info
        BUILD_VERSION:="no_version_info"
    else
        # SVN revision info
        BUILD_VERSION:=$(SVN_REVISION)
    endif
else
    # GIT revision info
    BUILD_VERSION:=$(GIT_REVISION)
endif

CXX=clang++
OPT=-O2
CXXFLAGS=$(OPT) $(LLVM_CXXFLAGS) -I. -Iobjs/ -I$(CLANG_INCLUDE)  \
	$(LLVM_VERSION_DEF) \
	-Wall \
	-DBUILD_DATE="\"$(BUILD_DATE)\"" -DBUILD_VERSION="\"$(BUILD_VERSION)\"" \
	-Wno-sign-compare -Wno-unused-function -Werror

# if( !($(LLVM_VERSION) == LLVM_3_2 || $(LLVM_VERSION) == LLVM_3_3 || $(LLVM_VERSION) == LLVM_3_4))
ifeq (,$(filter $(LLVM_VERSION), LLVM_3_2 LLVM_3_3 LLVM_3_4))
	CXXFLAGS+=-std=c++11 -Wno-c99-extensions -Wno-deprecated-register -fno-rtti
endif
ifneq ($(ARM_ENABLED), 0)
    CXXFLAGS+=-DISPC_ARM_ENABLED
endif
ifneq ($(NVPTX_ENABLED), 0)
    CXXFLAGS+=-DISPC_NVPTX_ENABLED
endif

LDFLAGS=
ifeq ($(ARCH_OS),Linux)
  # try to link everything statically under Linux (including libstdc++) so
  # that the binaries we generate will be portable across distributions...
#    LDFLAGS=-static
  # Linking everything statically isn't easy (too many things are required),
  # but linking libstdc++ and libgcc is necessary when building with relatively
  # new gcc, when going to distribute to old systems.
#    LDFLAGS=-static-libgcc -static-libstdc++
endif

LEX=flex
YACC=bison -d -v -t

###########################################################################

CXX_SRC=ast.cpp builtins.cpp cbackend.cpp ctx.cpp decl.cpp expr.cpp func.cpp \
	ispc.cpp llvmutil.cpp main.cpp module.cpp opt.cpp stmt.cpp sym.cpp \
	type.cpp util.cpp
HEADERS=ast.h builtins.h ctx.h decl.h expr.h func.h ispc.h llvmutil.h module.h \
	opt.h stmt.h sym.h type.h util.h
TARGETS=avx2-i64x4 avx11-i64x4 avx1-i64x4 avx1 avx1-x2 avx11 avx11-x2 avx2 avx2-x2 \
	sse2 sse2-x2 sse4-8 sse4-16 sse4 sse4-x2 \
	generic-4 generic-8 generic-16 generic-32 generic-64 generic-1 knl skx
ifneq ($(ARM_ENABLED), 0)
    TARGETS+=neon-32 neon-16 neon-8
endif
ifneq ($(NVPTX_ENABLED), 0)
    TARGETS+=nvptx
endif
# These files need to be compiled in two versions - 32 and 64 bits.
BUILTINS_SRC_TARGET=$(addprefix builtins/target-, $(addsuffix .ll, $(TARGETS)))
# These are files to be compiled in single version.
BUILTINS_SRC_COMMON=builtins/dispatch.ll
BUILTINS_OBJS_32=$(addprefix builtins-, $(notdir $(BUILTINS_SRC_TARGET:.ll=-32bit.o)))
BUILTINS_OBJS_64=$(addprefix builtins-, $(notdir $(BUILTINS_SRC_TARGET:.ll=-64bit.o)))
BUILTINS_OBJS=$(addprefix builtins-, $(notdir $(BUILTINS_SRC_COMMON:.ll=.o))) \
	$(BUILTINS_OBJS_32) $(BUILTINS_OBJS_64) \
	builtins-c-32.cpp builtins-c-64.cpp
BISON_SRC=parse.yy
FLEX_SRC=lex.ll

OBJS=$(addprefix objs/, $(CXX_SRC:.cpp=.o) $(BUILTINS_OBJS) \
       stdlib_mask1_ispc.o stdlib_mask8_ispc.o stdlib_mask16_ispc.o stdlib_mask32_ispc.o stdlib_mask64_ispc.o \
	$(BISON_SRC:.yy=.o) $(FLEX_SRC:.ll=.o))

default: ispc

.PHONY: dirs clean depend doxygen print_llvm_src llvm_check
.PRECIOUS: objs/builtins-%.cpp

depend: llvm_check $(CXX_SRC) $(HEADERS)
	@echo Updating dependencies
	@$(CXX) -MM $(CXXFLAGS) $(CXX_SRC) | sed 's_^\([a-z]\)_objs/\1_g' > depend

-include depend

dirs:
	@echo Creating objs/ directory
	@/bin/mkdir -p objs

llvm_check:
	@llvm-config --version > /dev/null || \
	(echo; \
	 echo "******************************************"; \
	 echo "ERROR: llvm-config not found in your PATH";  \
	 echo "******************************************"; \
	 echo; exit 1)
	@echo -e '$(subst $(newline), ,$(RIGHT_LLVM))'

print_llvm_src: llvm_check
	@echo Using LLVM `llvm-config --version` from `llvm-config --libdir`
	@echo Using compiler to build: `$(CXX) --version | head -1`

clean:
	/bin/rm -rf objs ispc

doxygen:
	/bin/rm -rf docs/doxygen
	doxygen doxygen.cfg

ispc: print_llvm_src dirs $(OBJS)
	@echo Creating ispc executable
	@$(CXX) $(OPT) $(LDFLAGS) -o $@ $(OBJS) $(ISPC_LIBS)

# Use clang as a default compiler, instead of gcc
# This is default now.
clang: ispc
clang: CXX=clang++

# Use gcc as a default compiler
gcc: ispc
gcc: CXX=g++

# Build ispc with address sanitizer instrumentation using clang compiler
# Note that this is not portable build
asan: clang
asan: OPT+=-fsanitize=address

# Do debug build, i.e. -O0 -g
debug: ispc
debug: OPT=-O0 -g

objs/%.o: %.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/cbackend.o: cbackend.cpp
	@echo Compiling $<
	@$(CXX) -fno-rtti -fno-exceptions $(CXXFLAGS) -o $@ -c $<

objs/opt.o: opt.cpp
	@echo Compiling $<
	@$(CXX) -fno-rtti $(CXXFLAGS) -o $@ -c $<

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

objs/builtins-dispatch.cpp: builtins/dispatch.ll builtins/util.m4 builtins/util-nvptx.m4 builtins/svml.m4 $(wildcard builtins/*common.ll)
	@echo Creating C++ source from builtins definition file $<
	@m4 -Ibuiltins/ -DLLVM_VERSION=$(LLVM_VERSION) -DBUILD_OS=UNIX $< | python bitcode2cpp.py $< > $@

objs/builtins-%-32bit.cpp: builtins/%.ll builtins/util.m4 builtins/util-nvptx.m4 builtins/svml.m4 $(wildcard builtins/*common.ll)
	@echo Creating C++ source from builtins definition file $< \(32 bit version\)
	@m4 -Ibuiltins/ -DLLVM_VERSION=$(LLVM_VERSION) -DBUILD_OS=UNIX -DRUNTIME=32 $< | python bitcode2cpp.py $< 32bit > $@

objs/builtins-%-64bit.cpp: builtins/%.ll builtins/util.m4 builtins/util-nvptx.m4 builtins/svml.m4 $(wildcard builtins/*common.ll)
	@echo Creating C++ source from builtins definition file $< \(64 bit version\)
	@m4 -Ibuiltins/ -DLLVM_VERSION=$(LLVM_VERSION) -DBUILD_OS=UNIX -DRUNTIME=64 $< | python bitcode2cpp.py $< 64bit > $@

objs/builtins-c-32.cpp: builtins/builtins.c
	@echo Creating C++ source from builtins definition file $<
	@$(CLANG) -m32 -emit-llvm -c $< -o - | llvm-dis - | python bitcode2cpp.py c 32 > $@

objs/builtins-c-64.cpp: builtins/builtins.c
	@echo Creating C++ source from builtins definition file $<
	@$(CLANG) -m64 -emit-llvm -c $< -o - | llvm-dis - | python bitcode2cpp.py c 64 > $@

objs/stdlib_mask1_ispc.cpp: stdlib.ispc
	@echo Creating C++ source from $< for mask1
	@$(CLANG) -E -x c -DISPC_MASK_BITS=1 -DISPC=1 -DPI=3.14159265358979 $< -o - | \
		python stdlib2cpp.py mask1 > $@

objs/stdlib_mask8_ispc.cpp: stdlib.ispc
	@echo Creating C++ source from $< for mask8
	@$(CLANG) -E -x c -DISPC_MASK_BITS=8 -DISPC=1 -DPI=3.14159265358979 $< -o - | \
		python stdlib2cpp.py mask8 > $@

objs/stdlib_mask16_ispc.cpp: stdlib.ispc
	@echo Creating C++ source from $< for mask16
	@$(CLANG) -E -x c -DISPC_MASK_BITS=16 -DISPC=1 -DPI=3.14159265358979 $< -o - | \
		python stdlib2cpp.py mask16 > $@

objs/stdlib_mask32_ispc.cpp: stdlib.ispc
	@echo Creating C++ source from $< for mask32
	@$(CLANG) -E -x c -DISPC_MASK_BITS=32 -DISPC=1 -DPI=3.14159265358979 $< -o - | \
		python stdlib2cpp.py mask32 > $@

objs/stdlib_mask64_ispc.cpp: stdlib.ispc
	@echo Creating C++ source from $< for mask64
	@$(CLANG) -E -x c -DISPC_MASK_BITS=64 -DISPC=1 -DPI=3.14159265358979 $< -o - | \
		python stdlib2cpp.py mask64 > $@
