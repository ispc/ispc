/*
  Copyright (c) 2010-2015, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "module.h"
#include "options.h"
#include <llvm-c/Target.h>

#if defined(_WIN32) || defined(_WIN64)
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#ifdef ISPC_IS_WINDOWS
#include <windows.h>
#endif // ISPC_IS_WINDOWS

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#ifdef ISPC_IS_LINUX
#include <malloc.h>
#endif

extern "C" {
    void ISPCLaunch(void **handlePtr, void *f, void *d, int,int,int);
    void ISPCSync(void *handle);
    void *ISPCAlloc(void **handlePtr, int64_t size, int32_t alignment);
}

typedef int (*WIDTH)(void);
typedef void (*F_V)(float*);
typedef void (*F_F)(float*, float*);
typedef void (*F_FU)(float*, float*, float);
typedef void (*F_FI)(float*, float*, int*);
typedef void (*F_DU)(float*, double*, double);
typedef void (*F_DUF)(float*, double*, float);
typedef void (*F_DI)(float*, double*, int*);
typedef void (*RESULT)(float*);

void ISPCLaunch(void **handle, void *f, void *d, int count0, int count1, int count2) {
    *handle = (void *)0xdeadbeef;
    typedef void (*TaskFuncType)(void *, int, int, int, int, int, int, int, int, int, int);
    TaskFuncType func = (TaskFuncType)f;
    int count = count0*count1*count2, idx = 0;
    for (int k = 0; k < count2; ++k)
      for (int j = 0; j < count1; ++j)
        for (int i = 0; i < count0; ++i)
        func(d, 0, 1, idx++, count, i,j,k,count0,count1,count2);
}

void ISPCSync(void *) {
}


void *ISPCAlloc(void **handle, int64_t size, int32_t alignment) {
    *handle = (void *)0xdeadbeef;
    // and now, we leak...
#ifdef ISPC_IS_WINDOWS
    return _aligned_malloc(size, alignment);
#endif
#ifdef ISPC_IS_LINUX
    return memalign(alignment, size);
#endif
#ifdef ISPC_IS_APPLE
    void *mem = malloc(size + (alignment-1) + sizeof(void*));
    char *amem = ((char*)mem) + sizeof(void*);
    amem = amem + uint32_t(alignment - (reinterpret_cast<uint64_t>(amem) &
                                        (alignment - 1)));
    ((void**)amem)[-1] = mem;
    return amem;
#endif
}


#if defined(_WIN32) || defined(_WIN64)
#define ALIGN __declspec(align(64))
#else
#define ALIGN __attribute__((aligned(64)))
#endif

// Maximum file size: 20 KB
#define MAXFILESIZE 20480

int main(int argc, char *argv[]) {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();
    LLVMInitializeX86TargetMC();

    char* filename = argv[argc-1];
    char src[MAXFILESIZE];
    memset(src, 0, sizeof(src));
    FILE* srcfile = fopen(filename, "r");
    size_t s = fread(src, sizeof(char), MAXFILESIZE, srcfile);
    (void)s;

    int sig = -1;
    if (strstr(src, "f_v(")) {
        sig = 0;
    }
    else if (strstr(src, "f_f(")) {
        sig = 1;
    }
    else if (strstr(src, "f_fu(")) {
        sig = 2;
    }
    else if (strstr(src, "f_fi(")) {
        sig = 3;
    }
    else if (strstr(src, "f_du(")) {
        sig = 4;
    }
    else if (strstr(src, "f_duf(")) {
        sig = 5;
    }
    else if (strstr(src, "f_di(")) {
        sig = 6;
    }
    else if (strstr(src, "f_sz(")) {
        sig = 7;
    }
    
    // Ignore f_sz tests for now.
    if (sig == 7) {
        return 0;
    }

    OptionParseResult opr = ParseOptions(argc - 1,argv);
    int ret = (opr & OPTION_PARSE_RESULT_ERROR) ? 1 : 0;
    if (ret == 1)
        return 1;

    int ec = Module::CompileAndJIT(src);
    if (ec != 0) {
        return 1;
    }

    F_V f_v = (F_V)Module::GetFunctionAddress("f_v");
    F_F f_f = (F_F)Module::GetFunctionAddress("f_f");
    F_FU f_fu = (F_FU)Module::GetFunctionAddress("f_fu");
    F_FI f_fi = (F_FI)Module::GetFunctionAddress("f_fi");
    F_DU f_du = (F_DU)Module::GetFunctionAddress("f_du");
    F_DUF f_duf = (F_DUF)Module::GetFunctionAddress("f_duf");
    F_DI f_di = (F_DI)Module::GetFunctionAddress("f_di");
    
    WIDTH width = (WIDTH)Module::GetFunctionAddress("width");
    RESULT result = (RESULT)Module::GetFunctionAddress("result");

    int w = width();
    assert(w <= 64);

    ALIGN float returned_result[64];
    ALIGN float vfloat[64];
    ALIGN double vdouble[64];
    ALIGN int vint[64];
    ALIGN int vint2[64];

    for (int i = 0; i < 64; ++i) {
        returned_result[i] = -1e20;
        vfloat[i] = i+1;
        vdouble[i] = i+1;
        vint[i] = 2*(i+1);
        vint2[i] = i+5;
    }

    float b = 5.;

    switch (sig) {
    case 0:
        f_v(returned_result);
        break;
    case 1:
        f_f(returned_result, vfloat);
        break;
    case 2:
        f_fu(returned_result, vfloat, b);
        break;
    case 3:
        f_fi(returned_result, vfloat, vint);
        break;
    case 4:
        f_du(returned_result, vdouble, 5.);
        break;
    case 5:
        f_duf(returned_result, vdouble, 5.f);
        break;
    case 6:
        f_di(returned_result, vdouble, vint2);
        break;
    default:
        return 1;
    }

    float expected_result[64];
    memset(expected_result, 0, 64*sizeof(float));
    result(expected_result);

    int errors = 0;
    for (int i = 0; i < w; ++i) {
        if (returned_result[i] != expected_result[i]) {
#ifdef EXPECT_FAILURE
            // bingo, failed
            return 1;
#else
            printf("%s: value %d disagrees: returned %f [%a], expected %f [%a]\n",
                   argv[0], i, returned_result[i], returned_result[i],
                   expected_result[i], expected_result[i]);
            ++errors;
#endif // EXPECT_FAILURE
        }
    }

#ifdef EXPECT_FAILURE
    // Don't expect to get here
    return 0;
#else
    return errors > 0;
#endif
}
