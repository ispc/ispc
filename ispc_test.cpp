/*
  Copyright (c) 2010-2011, Intel Corporation
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

#define _CRT_SECURE_NO_WARNINGS

#ifdef ISPC_IS_WINDOWS
#define NOMINMAX
#include <windows.h>
#endif
#include <stdio.h>
#include <stdint.h>

#ifdef ISPC_HAVE_SVML
#include <xmmintrin.h>
extern "C" {
    extern __m128 __svml_sinf4(__m128);
    extern __m128 __svml_cosf4(__m128);
    extern __m128 __svml_sincosf4(__m128 *,__m128);
    extern __m128 __svml_tanf4(__m128);
    extern __m128 __svml_atanf4(__m128);
    extern __m128 __svml_atan2f4(__m128, __m128);
    extern __m128 __svml_expf4(__m128);
    extern __m128 __svml_logf4(__m128);
    extern __m128 __svml_powf4(__m128, __m128);
}
#endif

#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Instructions.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/Target/TargetSelect.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetData.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/PassManager.h>
#include <llvm/Support/CFG.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Assembly/PrintModulePass.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/Support/MemoryBuffer.h>
#ifndef LLVM_2_8
#include <llvm/Support/system_error.h>
#endif

extern "C" { 
    void ISPCLaunch(void *, void *);
    void ISPCSync();
    void *ISPCMalloc(int64_t size, int32_t alignment);
    void ISPCFree(void *ptr);
}

void ISPCLaunch(void *func, void *data) {
    typedef void (*TaskFuncType)(void *, int, int);
    TaskFuncType tft = (TaskFuncType)(func);
    tft(data, 0, 1);
}


void ISPCSync() {
}


#ifdef ISPC_IS_WINDOWS
void *ISPCMalloc(int64_t size, int32_t alignment) {
    return _aligned_malloc(size, alignment);
}


void ISPCFree(void *ptr) {
    _aligned_free(ptr);
}
#endif

static void usage(int ret) {
    fprintf(stderr, "usage: ispc_test\n");
    fprintf(stderr, "\t[-h/--help]\tprint help\n");
    fprintf(stderr, "\t<files>\n");
    exit(ret);
}

static void svml_missing() {
    fprintf(stderr, "Program called unavailable SVML function!\n");
    exit(1);
}

// On Windows, sin() is an overloaded function, so we need an unambiguous
// function we can take the address of when wiring up the external references
// below.

double Sin(double x) { return sin(x); }
double Cos(double x) { return cos(x); }
double Tan(double x) { return tan(x); }
double Atan(double x) { return atan(x); }
double Atan2(double y, double x) { return atan2(y, x); }
double Pow(double a, double b) { return pow(a, b); }
double Exp(double x) { return exp(x); }
double Log(double x) { return log(x); }

static bool lRunTest(const char *fn) {
    llvm::LLVMContext *ctx = new llvm::LLVMContext;

#ifdef LLVM_2_8
    std::string err;
    llvm::MemoryBuffer *buf = llvm::MemoryBuffer::getFileOrSTDIN(fn, &err);
    if (!buf) {
        fprintf(stderr, "Unable to open file \"%s\": %s\n", fn, err.c_str());
        delete ctx;
        return false;
    }
    std::string bcErr;
    llvm::Module *module = llvm::ParseBitcodeFile(buf, *ctx, &bcErr);
#else
    llvm::OwningPtr<llvm::MemoryBuffer> buf;
    llvm::error_code err = llvm::MemoryBuffer::getFileOrSTDIN(fn, buf);
    if (err) {
        fprintf(stderr, "Unable to open file \"%s\": %s\n", fn, err.message().c_str());
        delete ctx;
        return false;
    }
    std::string bcErr;
    llvm::Module *module = llvm::ParseBitcodeFile(buf.get(), *ctx, &bcErr);
#endif

    if (!module) {
        fprintf(stderr, "Bitcode reader failed for \"%s\": %s\n", fn, bcErr.c_str());
        delete ctx;
        return false;
    }

    std::string eeError;
    llvm::ExecutionEngine *ee = llvm::ExecutionEngine::createJIT(module, &eeError);
    if (!ee) {
        fprintf(stderr, "Unable to create ExecutionEngine: %s\n", eeError.c_str());
        return false;
    }

    llvm::Function *func;
#define DO_FUNC(FUNC ,FUNCNAME)                           \
    if ((func = module->getFunction(FUNCNAME)) != NULL)   \
        ee->addGlobalMapping(func, (void *)FUNC)
    DO_FUNC(ISPCLaunch, "ISPCLaunch");
    DO_FUNC(ISPCSync, "ISPCSync");
#ifdef ISPC_IS_WINDOWS
    DO_FUNC(ISPCMalloc, "ISPCMalloc");
    DO_FUNC(ISPCFree, "ISPCFree");
#endif // ISPC_IS_WINDOWS
    DO_FUNC(putchar, "putchar");
    DO_FUNC(printf, "printf");
    DO_FUNC(fflush, "fflush");
    DO_FUNC(sinf, "sinf");
    DO_FUNC(cosf, "cosf");
    DO_FUNC(tanf, "tanf");
    DO_FUNC(atanf, "atanf");
    DO_FUNC(atan2f, "atan2f");
    DO_FUNC(powf, "powf");
    DO_FUNC(expf, "expf");
    DO_FUNC(logf, "logf");
    DO_FUNC(Sin, "sin");
    DO_FUNC(Cos, "cos");
    DO_FUNC(Tan, "tan");
    DO_FUNC(Atan, "atan");
    DO_FUNC(Atan2, "atan2");
    DO_FUNC(Pow, "pow");
    DO_FUNC(Exp, "exp");
    DO_FUNC(Log, "log");
    DO_FUNC(memset, "memset");
#ifdef ISPC_IS_APPLE
    DO_FUNC(memset_pattern4, "memset_pattern4");
    DO_FUNC(memset_pattern8, "memset_pattern8");
    DO_FUNC(memset_pattern16, "memset_pattern16");
#endif

#ifdef ISPC_HAVE_SVML
#define DO_SVML(FUNC ,FUNCNAME)                           \
    if ((func = module->getFunction(FUNCNAME)) != NULL)   \
        ee->addGlobalMapping(func, (void *)FUNC)
#else
#define DO_SVML(FUNC, FUNCNAME)                                         \
    if ((func = module->getFunction(FUNCNAME)) != NULL)                 \
        ee->addGlobalMapping(func, (void *)svml_missing)
#endif

    DO_SVML(__svml_sinf4, "__svml_sinf4");
    DO_SVML(__svml_cosf4, "__svml_cosf4");
    DO_SVML(__svml_sincosf4, "__svml_sincosf4");
    DO_SVML(__svml_tanf4, "__svml_tanf4");
    DO_SVML(__svml_atanf4, "__svml_atanf4");
    DO_SVML(__svml_atan2f4, "__svml_atan2f4");
    DO_SVML(__svml_expf4, "__svml_expf4");
    DO_SVML(__svml_logf4, "__svml_logf4");
    DO_SVML(__svml_powf4, "__svml_powf4");

    // figure out the vector width in the compiled code
    func = module->getFunction("width");
    if (!func) {
        fprintf(stderr, "No width() function found!\n");
        return false;
    }
    int width;
    {
        typedef int (*PFN)();
        PFN pfn = reinterpret_cast<PFN>(ee->getPointerToFunction(func));
        width = pfn();
        assert(width == 4 || width == 8 || width == 12 || width == 16);
    }

    // find the value that returns the desired result
    func = module->getFunction("result");
    bool foundResult = (func != NULL);
    float result[16];
    for (int i = 0; i < 16; ++i)
        result[i] = 0;
    bool ok = true;
    if (foundResult) {
        typedef void (*PFN)(float *);
        PFN pfn = reinterpret_cast<PFN>(ee->getPointerToFunction(func));
        pfn(result);
    }
    else
        fprintf(stderr, "Warning: no result() function found.\n");

    // try to find a function to run
    float returned[16];
    for (int i = 0; i < 16; ++i)
        returned[i] = 0;
    float vfloat[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    double vdouble[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    int vint[16] = { 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32 };
    int vint2[16] = { 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};

    if ((func = module->getFunction("f_v")) != NULL) {
        typedef void (*PFN)(float *);
        PFN pfn = reinterpret_cast<PFN>(ee->getPointerToFunction(func));
        pfn(returned);
    }
    else if ((func = module->getFunction("f_f")) != NULL) {
        typedef void (*PFN)(float *, float *);
        PFN pfn = reinterpret_cast<PFN>(ee->getPointerToFunction(func));
        llvm::verifyFunction(*func);
        pfn(returned, vfloat);
    }
    else if ((func = module->getFunction("f_fu")) != NULL) {
        typedef void (*PFN)(float *, float *, float fu);
        PFN pfn = reinterpret_cast<PFN>(ee->getPointerToFunction(func));
        llvm::verifyFunction(*func);
        pfn(returned, vfloat, 5.);
    }
    else if ((func = module->getFunction("f_fi")) != NULL) {
        typedef void (*PFN)(float *, float *, int *);
        PFN pfn = reinterpret_cast<PFN>(ee->getPointerToFunction(func));
        pfn(returned, vfloat, vint);
    }
    else if ((func = module->getFunction("f_du")) != NULL) {
        typedef void (*PFN)(float *, double *, double);
        PFN pfn = reinterpret_cast<PFN>(ee->getPointerToFunction(func));
        pfn(returned, vdouble, 5.);
    }
    else if ((func = module->getFunction("f_duf")) != NULL) {
        typedef void (*PFN)(float *, double *, float);
        PFN pfn = reinterpret_cast<PFN>(ee->getPointerToFunction(func));
        pfn(returned, vdouble, 5.f);
    }
    else if ((func = module->getFunction("f_di")) != NULL) {
        typedef void (*PFN)(float *, double *, int *);
        PFN pfn = reinterpret_cast<PFN>(ee->getPointerToFunction(func));
        pfn(returned, vdouble, vint2);
    }
    else {
        fprintf(stderr, "Unable to find runnable function in file \"%s\"\n", fn);
        ok = false;
    }

    // see if we got the right result
    if (ok) {
        if (foundResult) {
            for (int i = 0; i < width; ++i)
                if (returned[i] != result[i]) {
                    ok = false;
                    fprintf(stderr, "Test \"%s\" RETURNED %d: %g / %a EXPECTED %g / %a\n",
                            fn, i, returned[i], returned[i], result[i], result[i]);
                }
        }
        else {
            for (int i = 0; i < width; ++i)
                fprintf(stderr, "Test \"%s\" returned %d: %g / %a\n",
                        fn, i, returned[i], returned[i]);
        }
    }

    delete ee;
    delete ctx;

    return ok && foundResult;
}

int main(int argc, char *argv[]) {
    llvm::InitializeNativeTarget();

    std::vector<const char *> files;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
            usage(0);
        else
            files.push_back(argv[i]);
    }

    int passes = 0, fails = 0;
    for (unsigned int i = 0; i < files.size(); ++i) {
        if (lRunTest(files[i])) ++passes;
        else ++fails;
    }

    if (fails > 0)
        fprintf(stderr, "%d/%d tests passed\n", passes, passes+fails);
    return fails > 0;
}
