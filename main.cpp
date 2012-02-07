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
#include "util.h"
#include "type.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef ISPC_IS_WINDOWS
  #include <time.h>
#endif // ISPC_IS_WINDOWS
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
  #include <llvm/Support/TargetRegistry.h>
  #include <llvm/Support/TargetSelect.h>
#else
  #include <llvm/Target/TargetRegistry.h>
  #include <llvm/Target/TargetSelect.h>
  #include <llvm/Target/SubtargetFeature.h>
#endif

#ifdef ISPC_IS_WINDOWS
#define strcasecmp stricmp
#ifndef BUILD_DATE
#define BUILD_DATE __DATE__
#endif
#define BUILD_VERSION ""
#endif // ISPC_IS_WINDOWS

static void
lPrintVersion() {
    printf("Intel(r) SPMD Program Compiler (ispc), build %s (%s, LLVM %s)\n", 
           BUILD_DATE, BUILD_VERSION,
#ifdef LLVM_2_9
           "2.9"
#elif defined(LLVM_3_0) || defined(LLVM_3_0svn)
           "3.0"
#elif defined(LLVM_3_1) || defined(LLVM_3_1svn)
           "3.1"
#else
#error "Unhandled LLVM version"
#endif 
           );
}


static void
usage(int ret) {
    lPrintVersion();
    printf("\nusage: ispc\n");
    printf("    [--addressing={32,64}]\t\tSelect 32- or 64-bit addressing. (Note that 32-bit\n");
    printf("                          \t\taddressing calculations are done by default, even\n");
    printf("                          \t\ton 64-bit target architectures.)\n");
    printf("    [--arch={%s}]\t\tSelect target architecture\n", 
           Target::SupportedTargetArchs());
    printf("    [--c++-include-file=<name>]\t\tSpecify name of file to emit in #include statement in generated C++ code.\n");
    printf("    [--cpu=<cpu>]\t\t\tSelect target CPU type\n");
    printf("         <cpu>={%s}\n", Target::SupportedTargetCPUs());
    printf("    [-D<foo>]\t\t\t\t#define given value when running preprocessor\n");
    printf("    [--emit-asm]\t\t\tGenerate assembly language file as output\n");
#ifndef LLVM_2_9
    printf("    [--emit-c++]\t\t\tEmit a C++ source file as output\n");
#endif // !LLVM_2_9
    printf("    [--emit-llvm]\t\t\tEmit LLVM bitode file as output\n");
    printf("    [--emit-obj]\t\t\tGenerate object file file as output (default)\n");
    printf("    [-g]\t\t\t\tGenerate debugging information\n");
    printf("    [--help]\t\t\t\tPrint help\n");
    printf("    [--help-dev]\t\t\tPrint help for developer options\n");
    printf("    [-h <name>/--header-outfile=<name>]\tOutput filename for header\n");
    printf("    [-I <path>]\t\t\t\tAdd <path> to #include file search path\n");
    printf("    [--instrument]\t\t\tEmit instrumentation to gather performance data\n");
    printf("    [--math-lib=<option>]\t\tSelect math library\n");
    printf("        default\t\t\t\tUse ispc's built-in math functions\n");
    printf("        fast\t\t\t\tUse high-performance but lower-accuracy math functions\n");
    printf("        svml\t\t\t\tUse the Intel(r) SVML math libraries\n");
    printf("        system\t\t\t\tUse the system's math library (*may be quite slow*)\n");
    printf("    [--nostdlib]\t\t\tDon't make the ispc standard library available\n");
    printf("    [--nocpp]\t\t\t\tDon't run the C preprocessor\n");
    printf("    [-o <name>/--outfile=<name>]\tOutput filename (may be \"-\" for standard output)\n");
    printf("    [-O0/-O1]\t\t\t\tSet optimization level (-O1 is default)\n");
    printf("    [--opt=<option>]\t\t\tSet optimization option\n");
    printf("        disable-assertions\t\tRemove assertion statements from final code.\n");
    printf("        disable-loop-unroll\t\tDisable loop unrolling.\n");
    printf("        fast-masked-vload\t\tFaster masked vector loads on SSE (may go past end of array)\n");
    printf("        fast-math\t\t\tPerform non-IEEE-compliant optimizations of numeric expressions\n");
#ifndef ISPC_IS_WINDOWS
    printf("    [--pic]\t\t\t\tGenerate position-independent code\n");
#endif // !ISPC_IS_WINDOWS
    printf("    [--quiet]\t\t\t\tSuppress all output\n");
    printf("    [--target=<isa>]\t\t\tSelect target ISA. <isa>={%s}\n", Target::SupportedTargetISAs());
    printf("    [--version]\t\t\t\tPrint ispc version\n");
    printf("    [--werror]\t\t\t\tTreat warnings as errors\n");
    printf("    [--woff]\t\t\t\tDisable warnings\n");
    printf("    [--wno-perf]\t\t\tDon't issue warnings related to performance-related issues\n");
    printf("    <file to compile or \"-\" for stdin>\n");
    exit(ret);
}


static void
devUsage(int ret) {
    lPrintVersion();
    printf("\nusage (developer options): ispc\n");
    printf("    [--debug]\t\t\t\tPrint information useful for debugging ispc\n");
    printf("    [--fuzz-test]\t\t\tRandomly perturb program input to test error conditions\n");
    printf("    [--fuzz-seed=<value>]\t\tSeed value for RNG for fuzz testing\n");
    printf("    [--opt=<option>]\t\t\tSet optimization option\n");
    printf("        disable-all-on-optimizations\n");
    printf("        disable-blended-masked-stores\t\tScalarize masked stores on SSE (vs. using vblendps)\n");
    printf("        disable-blending-removal\t\tDisable eliminating blend at same scope\n");
    printf("        disable-coherent-control-flow\t\tDisable coherent control flow optimizations\n");
    printf("        disable-gather-scatter-flattening\tDisable flattening when all lanes are on\n");
    printf("        disable-gather-scatter-optimizations\tDisable improvements to gather/scatter\n");
    printf("        disable-handle-pseudo-memory-ops\n");
    printf("        disable-uniform-control-flow\t\tDisable uniform control flow optimizations\n");
    printf("        disable-uniform-memory-optimizations\tDisable uniform-based coherent memory access\n");
    printf("    [--yydebug]\t\t\tPrint debugging information during parsing\n");
    exit(ret);
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

#if 0
    // Use LLVM's little utility function to print out nice stack traces if
    // we crash
    llvm::sys::PrintStackTraceOnErrorSignal();
    llvm::PrettyStackTraceProgram X(argc, argv);
#endif

    // initialize available LLVM targets
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    LLVMInitializeX86TargetMC();
#endif

    AtomicType::Init();

    char *file = NULL;
    const char *headerFileName = NULL;
    const char *outFileName = NULL;
    const char *includeFileName = NULL;

    // Initiailize globals early so that we can set various option values
    // as we're parsing below
    g = new Globals;

    bool debugSet = false, optSet = false;
    Module::OutputType ot = Module::Object;
    bool generatePIC = false;
    const char *arch = NULL, *cpu = NULL, *target = NULL;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help"))
            usage(0);
        if (!strcmp(argv[i], "--help-dev"))
            devUsage(0);
        else if (!strncmp(argv[i], "-D", 2))
            g->cppArgs.push_back(argv[i]);
        else if (!strncmp(argv[i], "--addressing=", 13)) {
            if (atoi(argv[i] + 13) == 64)
                g->opt.force32BitAddressing = false;
            else if (atoi(argv[i] + 13) == 32)
                g->opt.force32BitAddressing = true;
            else {
                fprintf(stderr, "Addressing width \"%s\" invalid--only 32 and "
                        "64 are allowed.\n", argv[i]+13);
                usage(1);
            }
        }
        else if (!strncmp(argv[i], "--arch=", 7))
            arch = argv[i] + 7;
        else if (!strncmp(argv[i], "--cpu=", 6))
            cpu = argv[i] + 6;
        else if (!strcmp(argv[i], "--fast-math")) {
            fprintf(stderr, "--fast-math option has been renamed to --opt=fast-math!\n");
            usage(1);
        }
        else if (!strcmp(argv[i], "--fast-masked-vload")) {
            fprintf(stderr, "--fast-masked-vload option has been renamed to "
                    "--opt=fast-masked-vload!\n");
            usage(1);
        }
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
#ifndef LLVM_2_9
        else if (!strcmp(argv[i], "--emit-c++"))
            ot = Module::CXX;
#endif // !LLVM_2_9
        else if (!strcmp(argv[i], "--emit-llvm"))
            ot = Module::Bitcode;
        else if (!strcmp(argv[i], "--emit-obj"))
            ot = Module::Object;
        else if (!strcmp(argv[i], "-I")) {
            if (++i == argc) {
                fprintf(stderr, "No path specified after -I option.\n");
                usage(1);
            }
            g->includePath.push_back(argv[i]);
        }
        else if (!strncmp(argv[i], "-I", 2))
            g->includePath.push_back(argv[i]+2);
        else if (!strcmp(argv[i], "--fuzz-test"))
            g->enableFuzzTest = true;
        else if (!strncmp(argv[i], "--fuzz-seed=", 12))
            g->fuzzTestSeed = atoi(argv[i] + 12);
        else if (!strcmp(argv[i], "--target")) {
            // FIXME: should remove this way of specifying the target...
            if (++i == argc) {
                fprintf(stderr, "No target specified after --target option.\n");
                usage(1);
            }
            target = argv[i];
        }
        else if (!strncmp(argv[i], "--target=", 9))
            target = argv[i] + 9;
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
            else {
                fprintf(stderr, "Unknown --math-lib= option \"%s\".\n", lib);
                usage(1);
            }
        }
        else if (!strncmp(argv[i], "--opt=", 6)) {
            const char *opt = argv[i] + 6;
            if (!strcmp(opt, "fast-math"))
                g->opt.fastMath = true;
            else if (!strcmp(opt, "fast-masked-vload"))
                g->opt.fastMaskedVload = true;
            else if (!strcmp(opt, "disable-assertions"))
                g->opt.disableAsserts = true;
            else if (!strcmp(opt, "disable-loop-unroll"))
                g->opt.unrollLoops = false;

            // These are only used for performance tests of specific
            // optimizations
            else if (!strcmp(opt, "disable-all-on-optimizations"))
                g->opt.disableMaskAllOnOptimizations = true;
            else if (!strcmp(opt, "disable-handle-pseudo-memory-ops"))
                g->opt.disableHandlePseudoMemoryOps = true;
            else if (!strcmp(opt, "disable-blended-masked-stores"))
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
            else {
                fprintf(stderr, "Unknown --opt= option \"%s\".\n", opt);
                usage(1);
            }
        }
        else if (!strcmp(argv[i], "--woff") || !strcmp(argv[i], "-woff")) {
            g->disableWarnings = true;
            g->emitPerfWarnings = false;
        }
        else if (!strcmp(argv[i], "--werror"))
            g->warningsAsErrors = true;
        else if (!strcmp(argv[i], "--nowrap"))
            g->disableLineWrap = true;
        else if (!strcmp(argv[i], "--wno-perf") || !strcmp(argv[i], "-wno-perf"))
            g->emitPerfWarnings = false;
        else if (!strcmp(argv[i], "-o")) {
            if (++i == argc) {
                fprintf(stderr, "No output file specified after -o option.\n");
                usage(1);
            }
            outFileName = argv[i];
        }
        else if (!strcmp(argv[i], "--outfile="))
            outFileName = argv[i] + strlen("--outfile=");
        else if (!strcmp(argv[i], "-h")) {
            if (++i == argc) {
                fprintf(stderr, "No header file name specified after -h option.\n");
                usage(1);
            }
            headerFileName = argv[i];
        }
        else if (!strncmp(argv[i], "--header-outfile=", 17)) {
            headerFileName = argv[i] + strlen("--header-outfile=");
        }
        else if (!strncmp(argv[i], "--c++-include-file=", 19)) {
            includeFileName = argv[i] + strlen("--c++-include-file=");
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
#ifndef ISPC_IS_WINDOWS
        else if (!strcmp(argv[i], "--pic"))
            generatePIC = true;
#endif // !ISPC_IS_WINDOWS
        else if (!strcmp(argv[i], "--quiet"))
            g->quiet = true;
        else if (!strcmp(argv[i], "--yydebug")) {
            extern int yydebug;
            yydebug = 1;
        }
        else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--version")) {
            lPrintVersion();
            return 0;
        }
        else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option \"%s\".\n", argv[i]);
            usage(1);
        }
        else {
            if (file != NULL) {
                fprintf(stderr, "Multiple input files specified on command "
                        "line: \"%s\" and \"%s\".\n", file, argv[i]);
                usage(1);
            }
            else
                file = argv[i];
        }
    }

    // If the user specified -g, then the default optimization level is 0.
    // If -g wasn't specified, the default optimization level is 1 (full
    // optimization).
    if (debugSet && !optSet)
        g->opt.level = 0;

    if (g->enableFuzzTest) {
        if (g->fuzzTestSeed == -1) {
#ifdef ISPC_IS_WINDOWS
            int seed = (unsigned)time(NULL);
#else
            int seed = getpid();
#endif
            g->fuzzTestSeed = seed;
            Warning(SourcePos(), "Using seed %d for fuzz testing", 
                    g->fuzzTestSeed);
        }
#ifdef ISPC_IS_WINDOWS
        srand(g->fuzzTestSeed);
#else
        srand48(g->fuzzTestSeed);
#endif
    }

    if (outFileName == NULL && headerFileName == NULL)
        Warning(SourcePos(), "No output file or header file name specified. "
                "Program will be compiled and warnings/errors will "
                "be issued, but no output will be generated.");

    return Module::CompileAndOutput(file, arch, cpu, target, generatePIC,
                                    ot, outFileName, headerFileName, 
                                    includeFileName);
}
