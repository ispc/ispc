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

/** @file main.cpp
    @brief main() entrypoint implementation for ispc
*/

#include "ispc.h"
#include "module.h"
#include <stdio.h>
#include <stdlib.h>
#include <llvm/Support/PrettyStackTrace.h>
#ifdef LLVM_2_8
#include <llvm/System/Signals.h>
#else
#include <llvm/Support/Signals.h>
#endif

#ifdef ISPC_IS_WINDOWS
#define strcasecmp stricmp
#define BUILD_DATE __DATE__
#define BUILD_VERSION ""
#endif // ISPC_IS_WINDOWS

static void usage(int ret) {
    printf("This is the Intel(r) SPMD Program Compiler (ispc), build %s (%s)\n\n", BUILD_DATE, BUILD_VERSION);
    printf("usage: ispc\n");
    printf("    [--arch={x86,x86-64}]\t\tSelect target architecture\n");
    printf("    [--cpu=<cpu>]\t\t\tSelect target CPU type\n");
    printf("         (atom, barcelona, core2, corei7, corei7-avx, istanbul, nocona,\n");
    printf("          penryn, westmere)\n");
#ifndef ISPC_IS_WINDOWS
    printf("    [-D<foo>]\t\t\t\t#define value when running preprocessor\n");
#endif
    printf("    [--debug]\t\t\t\tPrint information useful for debugging ispc\n");
    printf("    [--emit-asm]\t\t\tGenerate assembly language file as output\n");
    printf("    [--emit-llvm]\t\t\tEmit LLVM bitode file as output\n");
    printf("    [--emit-obj]\t\t\tGenerate object file file as output\n");
    printf("    [--fast-math]\t\t\tPerform non-IEEE-compliant optimizations of numeric expressions\n");
    printf("    [-g]\t\t\t\tGenerate debugging information\n");
    printf("    [--help]\t\t\t\tPrint help\n");
    printf("    [-h] <name>\t\t\t\tOutput filename for header\n");
    printf("    [--instrument]\t\t\tEmit instrumentation to gather performance data\n");
    printf("    [--math-lib=<option>]\t\tSelect math library\n");
    printf("        default\t\t\t\tUse ispc's built-in math functions\n");
    printf("        fast\t\t\t\tUse high-performance but lower-accuracy math functions\n");
    printf("        svml\t\t\t\tUse the Intel SVML math libraries\n");
    printf("        system\t\t\t\tUse the system's math library (*may be quite slow*)\n");
    printf("    [--nostdlib]\t\t\tDon't make the ispc standard library available\n");
#ifndef ISPC_IS_WINDOWS
    printf("    [--nocpp]\t\t\t\tDon't run the C preprocessor\n");
#endif
    printf("    [-o/--outfile] <name>\t\tOutput filename for bitcode (may be \"-\" for standard output)\n");
    printf("    [-O0/-O1]\t\t\t\tSet optimization level\n");
    printf("    [--opt=<option>]\t\t\tSet optimization option\n");
    printf("        disable-blended-masked-stores\t\tScalarize masked stores on SSE (vs. using vblendps)\n");
    printf("        disable-coherent-control-flow\t\tDisable coherent control flow optimizations\n");
    printf("        disable-uniform-control-flow\t\tDisable uniform control flow optimizations\n");
    printf("        disable-gather-scatter-optimizations\tDisable improvements to gather/scatter\n");
    printf("        disable-blending-removal\t\tDisable eliminating blend at same scope\n");
    printf("        disable-gather-scatter-flattening\tDisable flattening when all lanes are on\n");
    printf("        disable-uniform-memory-optimizations\tDisable uniform-based coherent memory access\n");
    printf("        disable-masked-store-optimizations\tDisable lowering to regular stores when possible\n");
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
    printf("    [--target={sse2,sse4,sse4x2,avx,avx-x2}] Select target ISA (SSE4 is default unless compiling for atom; then SSE2 is.)\n");
#else
    printf("    [--target={sse2,sse4,sse4x2}] Select target ISA (SSE4 is default unless compiling for atom; then SSE2 is.)\n");
#endif // LLVM 3.0
    printf("    [--version]\t\t\t\tPrint ispc version\n");
    printf("    [--woff]\t\t\t\tDisable warnings\n");
    printf("    [--wno-perf]\t\t\tDon't issue warnings related to performance-related issues\n");
    printf("    <file to compile or \"-\" for stdin>\n");
    exit(ret);
}

/** Given a target name string, set initialize the global g->target
    structure appropriately. 
*/
static void lDoTarget(const char *target) {
    if (!strcasecmp(target, "sse2")) {
        g->target.isa = Target::SSE2;
        g->target.nativeVectorWidth = 4;
        g->target.vectorWidth = 4;
    }
    else if (!strcasecmp(target, "sse4")) {
        g->target.isa = Target::SSE4;
        g->target.nativeVectorWidth = 4;
        g->target.vectorWidth = 4;
    }
    else if (!strcasecmp(target, "sse4x2")) {
        g->target.isa = Target::SSE4;
        g->target.nativeVectorWidth = 4;
        g->target.vectorWidth = 8;
    }
#if defined(LLVM_3_0) || defined(LLVM_3_0svn)
    else if (!strcasecmp(target, "avx")) {
        g->target.isa = Target::AVX;
        g->target.nativeVectorWidth = 8;
        g->target.vectorWidth = 8;
    }
    else if (!strcasecmp(target, "avx-x2")) {
        g->target.isa = Target::AVX;
        g->target.nativeVectorWidth = 8;
        g->target.vectorWidth = 16;
    }
#endif // LLVM 3.0
    else
        usage(1);
}


/** We take arguments from both the command line as well as from the
    ISPC_ARGS environment variable.  This function returns a new set of
    arguments representing the ones from those two sources merged together.
 */ 
static void lGetAllArgs(int Argc, char *Argv[], int &argc, char *argv[128]) {
    // Copy over the command line arguments (passed in)
    for (int i = 0; i < Argc; ++i)
        argv[i] = Argv[i];
    argc = Argc;

    // See if we have any set via the environment variable
    const char *env = getenv("ISPC_ARGS");
    if (!env)
        return;
    while (true) {
        // Look for the next space in the string, which delimits the end of
        // the current argument
        const char *end = strchr(env, ' ');
        if (end == NULL)
            end = env + strlen(env);
        int len = end - env;

        // Copy the argument into a newly allocated memory (so we can
        // NUL-terminate it).
        char *ptr = new char[len+1];
        strncpy(ptr, env, len);
        ptr[len] = '\0';

        // Add it to the args array and get out of here 
        argv[argc++] = ptr;
        if (*end == '\0')
            break;

        // Advance the starting pointer of the string to the next non-space
        // character
        env = end+1;
        while (*env == ' ')
            ++env;

        // Hit the end of the string; get out of here
        if (*env == '\0')
            break;
    }
}


int main(int Argc, char *Argv[]) {
    int argc;
    char *argv[128];
    lGetAllArgs(Argc, Argv, argc, argv);

    // Use LLVM's little utility function to print out nice stack traces if
    // we crash
    llvm::sys::PrintStackTraceOnErrorSignal();
    llvm::PrettyStackTraceProgram X(argc, argv);

    char *file = NULL;
    const char *headerFileName = NULL;
    const char *outFileName = NULL;

    // Initiailize globals early so that we can set various option values
    // as we're parsing below
    g = new Globals;

    bool debugSet = false, optSet = false, targetSet = false;
    Module::OutputType ot = Module::Object;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help"))
            usage(0);
#ifndef ISPC_IS_WINDOWS
        else if (!strncmp(argv[i], "-D", 2)) {
            g->cppArgs.push_back(argv[i]);
        }
#endif // !ISPC_IS_WINDOWS
        else if (!strncmp(argv[i], "--arch=", 7)) {
            g->target.arch = argv[i] + 7;
            if (g->target.arch == "x86")
                g->target.is32bit = true;
            else if (g->target.arch == "x86-64")
                g->target.is32bit = false;
        }
        else if (!strncmp(argv[i], "--cpu=", 6))
            g->target.cpu = argv[i] + 6;
        else if (!strcmp(argv[i], "--fast-math"))
            g->opt.fastMath = true;
        else if (!strcmp(argv[i], "--debug"))
            g->debugPrint = true;
        else if (!strcmp(argv[i], "--instrument"))
            g->emitInstrumentation = true;
        else if (!strcmp(argv[i], "-g")) {
            g->generateDebuggingSymbols = true;
            debugSet = true;
        }
        else if (!strcmp(argv[i], "--emit-asm"))
            ot = Module::Asm;
        else if (!strcmp(argv[i], "--emit-llvm"))
            ot = Module::Bitcode;
        else if (!strcmp(argv[i], "--emit-obj"))
            ot = Module::Object;
        else if (!strcmp(argv[i], "--target")) {
            if (++i == argc) usage(1);
            lDoTarget(argv[i]);
            targetSet = true;
        }
        else if (!strncmp(argv[i], "--target=", 9)) {
            const char *target = argv[i] + 9;
            lDoTarget(target);
        }
        else if (!strncmp(argv[i], "--math-lib=", 11)) {
            const char *lib = argv[i] + 11;
            if (!strcmp(lib, "default"))
                g->mathLib = Globals::Math_ISPC;
            else if (!strcmp(lib, "fast"))
                g->mathLib = Globals::Math_ISPCFast;
            else if (!strcmp(lib, "svml"))
                g->mathLib = Globals::Math_SVML;
            else if (!strcmp(lib, "system"))
                g->mathLib = Globals::Math_System;
            else
                usage(1);
        }
        else if (!strncmp(argv[i], "--opt=", 6)) {
            const char *opt = argv[i] + 6;
            if (!strcmp(opt, "disable-blended-masked-stores"))
                g->opt.disableBlendedMaskedStores = true;
            else if (!strcmp(opt, "disable-coherent-control-flow"))
                g->opt.disableCoherentControlFlow = true;
            else if (!strcmp(opt, "disable-uniform-control-flow"))
                g->opt.disableUniformControlFlow = true;
            else if (!strcmp(opt, "disable-gather-scatter-optimizations"))
                g->opt.disableGatherScatterOptimizations = true;
            else if (!strcmp(opt, "disable-blending-removal"))
                g->opt.disableMaskedStoreToStore = true;
            else if (!strcmp(opt, "disable-gather-scatter-flattening"))
                g->opt.disableGatherScatterFlattening = true;
            else if (!strcmp(opt, "disable-uniform-memory-optimizations"))
                g->opt.disableUniformMemoryOptimizations = true;
            else if (!strcmp(opt, "disable-masked-store-optimizations"))
                g->opt.disableMaskedStoreOptimizations = true;
            else 
                usage(1);
        }
        else if (!strcmp(argv[i], "--woff") || !strcmp(argv[i], "-woff")) {
            g->disableWarnings = true;
            g->emitPerfWarnings = false;
        }
        else if (!strcmp(argv[i], "--wno-perf") || !strcmp(argv[i], "-wno-perf"))
            g->emitPerfWarnings = false;
        else if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--outfile")) {
            if (++i == argc) usage(1);
            outFileName = argv[i];
        }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--header-outfile")) {
            if (++i == argc) usage(1);
            headerFileName = argv[i];
        }
        else if (!strcmp(argv[i], "-O0")) {
            g->opt.level = 0;
            optSet = true;
        }
        else if (!strcmp(argv[i], "-O") ||  !strcmp(argv[i], "-O1") || 
                 !strcmp(argv[i], "-O2") || !strcmp(argv[i], "-O3")) {
            g->opt.level = 1;
            optSet = true;
        }
        else if (!strcmp(argv[i], "-"))
            ;
        else if (!strcmp(argv[i], "--nostdlib"))
            g->includeStdlib = false;
        else if (!strcmp(argv[i], "--nocpp"))
            g->runCPP = false;
        else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--version")) {
            printf("Intel(r) SPMD Program Compiler (ispc) build %s (%s)\n", 
                   BUILD_DATE, BUILD_VERSION);
            return 0;
        }
        else if (argv[i][0] == '-')
            usage(1);
        else {
            if (file != NULL)
                usage(1);
            else
                file = argv[i];
        }
    }

    // If the user specified -g, then the default optimization level is 0.
    // If -g wasn't specified, the default optimization level is 1 (full
    // optimization).
    if (debugSet && !optSet)
        g->opt.level = 0;

    // Make SSE2 the default target on atom unless the target has been set
    // explicitly.
    if (!targetSet && (g->target.cpu == "atom"))
        lDoTarget("sse2");

    m = new Module(file);
    if (m->CompileFile() == 0) {
        if (outFileName != NULL)
            if (!m->WriteOutput(ot, outFileName))
                return 1;
        if (headerFileName != NULL)
            if (!m->WriteOutput(Module::Header, headerFileName))
                return 1;
    }
    int errorCount = m->errorCount;
    delete m;

    return errorCount > 0;
}
