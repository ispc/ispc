/*
  Copyright (c) 2010-2019, Intel Corporation
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
#include "type.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef ISPC_HOST_IS_WINDOWS
#include <time.h>
#else
#include <unistd.h>
#endif // ISPC_HOST_IS_WINDOWS
#include <llvm/Support/Debug.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>

#ifdef ISPC_HOST_IS_WINDOWS
#define strcasecmp stricmp
#ifndef BUILD_DATE
#define BUILD_DATE __DATE__
#endif
#define BUILD_VERSION ""
#if _MSC_VER >= 1900
#define ISPC_VS_VERSION "Visual Studio 2015 and later"
#else
#define ISPC_VS_VERSION "Visual Studio 2013 and earlier"
#endif
#endif // ISPC_HOST_IS_WINDOWS

#define MAX_NUM_ARGS (512)

static void lPrintVersion() {
#ifdef ISPC_HOST_IS_WINDOWS
    printf("Intel(r) SPMD Program Compiler (ispc), %s (build date %s, LLVM %s)\n"
           "Supported Visual Studio versions: %s.\n",
           ISPC_VERSION, BUILD_DATE, ISPC_LLVM_VERSION_STRING, ISPC_VS_VERSION);
#else
    printf("Intel(r) SPMD Program Compiler (ispc), %s (build %s @ %s, LLVM %s)\n", ISPC_VERSION, BUILD_VERSION,
           BUILD_DATE, ISPC_LLVM_VERSION_STRING);
#endif

// The recommended way to build ISPC assumes custom LLVM build with a set of patches.
// If the default LLVM distribution is used, then the resuling ISPC binary may contain
// known and already fixed stability and performance problems.
#ifdef ISPC_NO_DUMPS
    printf("This version is likely linked against non-recommended LLVM binaries.\n"
           "For best stability and performance please use official binary distribution from "
           "http://ispc.github.io/downloads.html");
#endif
}

static void usage(int ret) {
    lPrintVersion();
    printf("\nusage: ispc\n");
    printf("    [--addressing={32,64}]\t\tSelect 32- or 64-bit addressing. (Note that 32-bit\n");
    printf("                          \t\taddressing calculations are done by default, even\n");
    printf("                          \t\ton 64-bit target architectures.)\n");
    printf("    [--arch={%s}]\t\tSelect target architecture\n", Target::SupportedArchs());
    printf("    [--c++-include-file=<name>]\t\tSpecify name of file to emit in #include statement in generated C++ "
           "code.\n");
#ifndef ISPC_HOST_IS_WINDOWS
    printf("    [--colored-output]\t\tAlways use terminal colors in error/warning messages\n");
#endif
    printf("    ");
    char cpuHelp[2048];
    snprintf(cpuHelp, sizeof(cpuHelp), "[--cpu=<cpu>]\t\t\tSelect target CPU type\n<cpu>={%s}\n",
             Target::SupportedCPUs().c_str());
    PrintWithWordBreaks(cpuHelp, 16, TerminalWidth(), stdout);
    printf("    [-D<foo>]\t\t\t\t#define given value when running preprocessor\n");
    printf("    [--dev-stub <filename>]\t\tEmit device-side offload stub functions to file\n");
    printf("    [--dllexport]\t\t\tMake non-static functions DLL exported.  Windows target only\n");
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5
    printf("    [--dwarf-version={2,3,4}]\t\tGenerate source-level debug information with given DWARF version "
           "(triggers -g).  Ignored for Windows target\n");
#endif
    printf("    [--emit-asm]\t\t\tGenerate assembly language file as output\n");
    printf("    [--x86-asm-syntax=<option>]\t\tSelect style of code if generating assembly\n");
    printf("        intel\t\t\t\tEmit Intel-style assembly\n");
    printf("        att\t\t\t\tEmit AT&T-style assembly\n");
    printf("    [--emit-c++]\t\t\tEmit a C++ source file as output\n");
    printf("    [--emit-llvm]\t\t\tEmit LLVM bitode file as output\n");
    printf("    [--emit-llvm-text]\t\t\tEmit LLVM bitode file as output in textual form\n");
    printf("    [--emit-obj]\t\t\tGenerate object file file as output (default)\n");
    printf("    [--force-alignment=<value>]\t\tForce alignment in memory allocations routine to be <value>\n");
    printf("    [-g]\t\t\t\tGenerate source-level debug information\n");
    printf("    [--help]\t\t\t\tPrint help\n");
    printf("    [--help-dev]\t\t\tPrint help for developer options\n");
    printf("    [--host-stub <filename>]\t\tEmit host-side offload stub functions to file\n");
    printf("    [-h <name>/--header-outfile=<name>]\tOutput filename for header\n");
    printf("    [-I <path>]\t\t\t\tAdd <path> to #include file search path\n");
    printf("    [--instrument]\t\t\tEmit instrumentation to gather performance data\n");
    printf("    [--math-lib=<option>]\t\tSelect math library\n");
    printf("        default\t\t\t\tUse ispc's built-in math functions\n");
    printf("        fast\t\t\t\tUse high-performance but lower-accuracy math functions\n");
    printf("        svml\t\t\t\tUse the Intel(r) SVML math libraries\n");
    printf("        system\t\t\t\tUse the system's math library (*may be quite slow*)\n");
    printf("    [-MMM <filename>]\t\t\tWrite #include dependencies to given file.\n");
    printf("    [-M]\t\t\t\tOutput a rule suitable for `make' describing the dependencies of the main source file to "
           "stdout.\n");
    printf("    [-MF <filename>]\t\t\tWhen used with `-M', specifies a file to write the dependencies to.\n");
    printf("    [-MT <filename>]\t\t\tWhen used with `-M', changes the target of the rule emitted by dependency "
           "generation.\n");
    printf("    [--no-omit-frame-pointer]\t\tDisable frame pointer omission. It may be useful for profiling\n");
    printf("    [--nostdlib]\t\t\tDon't make the ispc standard library available\n");
    printf("    [--no-pragma-once]\t\t\tDon't use #pragma once in created headers\n");
    printf("    [--nocpp]\t\t\t\tDon't run the C preprocessor\n");
    printf("    [-o <name>/--outfile=<name>]\tOutput filename (may be \"-\" for standard output)\n");
    printf("    [-O0/-O(1/2/3)]\t\t\tSet optimization level. Default behavior is to optimize for speed.\n");
    printf("        -O0\t\t\t\tOptimizations disabled.\n");
    printf("        -O1\t\t\t\tOptimization for size.\n");
    printf("        -O2/O3\t\t\t\tOptimization for speed.\n");
    printf("    [--opt=<option>]\t\t\tSet optimization option\n");
    printf("        disable-assertions\t\tRemove assertion statements from final code.\n");
    printf("        disable-fma\t\t\tDisable 'fused multiply-add' instructions (on targets that support them)\n");
    printf("        disable-loop-unroll\t\tDisable loop unrolling.\n");
    printf("        fast-masked-vload\t\tFaster masked vector loads on SSE (may go past end of array)\n");
    printf("        fast-math\t\t\tPerform non-IEEE-compliant optimizations of numeric expressions\n");
    printf("        force-aligned-memory\t\tAlways issue \"aligned\" vector load and store instructions\n");
    printf("    [--pic]\t\t\t\tGenerate position-independent code.  Ignored for Windows target\n");
    printf("    [--quiet]\t\t\t\tSuppress all output\n");
    printf("    ");
    char targetHelp[2048];
    snprintf(targetHelp, sizeof(targetHelp),
             "[--target=<t>]\t\t\tSelect target ISA and width.\n"
             "<t>={%s}",
             Target::SupportedTargets());
    PrintWithWordBreaks(targetHelp, 24, TerminalWidth(), stdout);
    printf("    ");
    snprintf(targetHelp, sizeof(targetHelp), "[--target-os=<os>]\t\t\tSelect target OS.  <os>={%s}",
             Target::SupportedOSes());
    PrintWithWordBreaks(targetHelp, 24, TerminalWidth(), stdout);
    printf("    [--version]\t\t\t\tPrint ispc version\n");
    printf("    [--werror]\t\t\t\tTreat warnings as errors\n");
    printf("    [--woff]\t\t\t\tDisable warnings\n");
    printf("    [--wno-perf]\t\t\tDon't issue warnings related to performance-related issues\n");
    printf("    [@<filename>]\t\t\tRead additional arguments from the given file\n");
    printf("    <file to compile or \"-\" for stdin>\n");
    exit(ret);
}

static void devUsage(int ret) {
    lPrintVersion();
    printf("\nusage (developer options): ispc\n");
    printf("    [--debug]\t\t\t\tPrint information useful for debugging ispc\n");
    printf("    [--debug-llvm]\t\t\tEnable LLVM debugging information (dumps to stderr)\n");
    printf("    [--print-target]\t\t\tPrint target's information\n");
    printf("    [--fuzz-test]\t\t\tRandomly perturb program input to test error conditions\n");
    printf("    [--fuzz-seed=<value>]\t\tSeed value for RNG for fuzz testing\n");
    printf("    [--opt=<option>]\t\t\tSet optimization option\n");
    printf("        disable-all-on-optimizations\t\tDisable optimizations that take advantage of \"all on\" mask\n");
    printf("        disable-blended-masked-stores\t\tScalarize masked stores on SSE (vs. using vblendps)\n");
    printf("        disable-blending-removal\t\tDisable eliminating blend at same scope\n");
    printf("        disable-coalescing\t\t\tDisable gather coalescing\n");
    printf("        disable-coherent-control-flow\t\tDisable coherent control flow optimizations\n");
    printf("        disable-gather-scatter-flattening\tDisable flattening when all lanes are on\n");
    printf("        disable-gather-scatter-optimizations\tDisable improvements to gather/scatter\n");
    printf("        disable-handle-pseudo-memory-ops\tLeave __pseudo_* calls for gather/scatter/etc. in final IR\n");
    printf("        disable-uniform-control-flow\t\tDisable uniform control flow optimizations\n");
    printf("        disable-uniform-memory-optimizations\tDisable uniform-based coherent memory access\n");
    printf("    [--yydebug]\t\t\t\tPrint debugging information during parsing\n");
#ifndef ISPC_NO_DUMPS
    printf("    [--debug-phase=<value>]\t\tSet optimization phases to dump. "
           "--debug-phase=first,210:220,300,305,310:last\n");
    printf("    [--dump-file]\t\t\tDump module IR to file(s) in current directory\n");
#endif
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_4 || ISPC_LLVM_VERSION == ISPC_LLVM_3_5 // 3.4, 3.5
    printf("    [--debug-ir=<value>]\t\tSet optimization phase to generate debugIR after it\n");
#endif
    printf("    [--off-phase=<value>]\t\tSwitch off optimization phases. --off-phase=first,210:220,300,305,310:last\n");
    exit(ret);
}

/** Define an abstract base-class that implements the parsing of an character source and
 *  the breaking of it into the individual arguments
 */
class ArgFactory {
  private:
    char *AllocateString(std::string string) {
        int len = string.length();
        char *ptr = new char[len + 1];
        strncpy(ptr, string.c_str(), len);
        ptr[len] = '\0';
        return ptr;
    }

    /** Method provided by the derived classes to retrieve the next character from the stream.
     */
    virtual char GetNextChar() = 0;

  public:
    ArgFactory() {}

    char *GetNextArg() {
        std::string arg;
        char c = GetNextChar();

        // First consume any white-space before the argument
        while (isspace(c))
            c = GetNextChar();

        if (c == '\0')
            // Reached the end so no more arguments
            return NULL;

        // c now has the first character of the next argument, so collect the rest
        while (c != '\0' && !isspace(c)) {
            arg += c;
            c = GetNextChar();
        }

        return AllocateString(arg);
    }
};

/** Define a class to break the contents of an open file into the individual arguments */
class FileArgFactory : public ArgFactory {
  private:
    FILE *InputFile;

    virtual char GetNextChar() {
        int c = fgetc(InputFile);
        if (c == EOF) {
            return '\0';
        } else {
            return c;
        }
    }

  public:
    FileArgFactory(FILE *file) : InputFile(file) {}
};

/** Define a class to break a NUL-terminated string into the individual arguments */
class StringArgFactory : public ArgFactory {
  private:
    const char *InputString;

    virtual char GetNextChar() {
        char c = *InputString;

        if (c != '\0')
            ++InputString;

        return c;
    }

  public:
    StringArgFactory(const char *string) : InputString(string) {}
};

// Forward reference
static void lAddSingleArg(char *arg, int &argc, char *argv[MAX_NUM_ARGS]);

/** Add all args from a given factory to the argc/argv passed as parameters, which could
 *  include recursing into another ArgFactory.
 */
static void lAddArgsFromFactory(ArgFactory &Args, int &argc, char *argv[MAX_NUM_ARGS]) {
    while (true) {
        char *NextArg = Args.GetNextArg();
        if (NextArg == NULL)
            break;
        lAddSingleArg(NextArg, argc, argv);
    }
}

/** Parse an open file for arguments and add them to the argc/argv passed as parameters */
static void lAddArgsFromFile(FILE *file, int &argc, char *argv[MAX_NUM_ARGS]) {
    FileArgFactory args(file);
    lAddArgsFromFactory(args, argc, argv);
}

/** Parse a string for arguments and add them to the argc/argv passed as parameters */
static void lAddArgsFromString(const char *string, int &argc, char *argv[MAX_NUM_ARGS]) {
    StringArgFactory args(string);
    lAddArgsFromFactory(args, argc, argv);
}

/** Add a single argument to the argc/argv passed as parameters. If the argument is of the
 *  form @<filename> and <filename> exists and is readable, the arguments in the file will be
 *  inserted into argc/argv in place of the original argument.
 */
static void lAddSingleArg(char *arg, int &argc, char *argv[MAX_NUM_ARGS]) {
    if (arg[0] == '@') {
        char *filename = &arg[1];
        FILE *file = fopen(filename, "r");
        if (file != NULL) {
            lAddArgsFromFile(file, argc, argv);
            fclose(file);
            arg = NULL;
        }
    }
    if (arg != NULL) {
        if (argc >= MAX_NUM_ARGS) {
            fprintf(stderr, "More than %d arguments have been specified - aborting\n", MAX_NUM_ARGS);
            exit(EXIT_FAILURE);
        }
        // printf("Arg %d: %s\n", argc, arg);
        argv[argc++] = arg;
    }
}

/** We take arguments from both the command line as well as from the
 *  ISPC_ARGS environment variable - and each of these can include a file containing
 *  additional arguments using @<filename>. This function returns a new set of
 *  arguments representing the ones from all these sources merged together.
 */
static void lGetAllArgs(int Argc, char *Argv[], int &argc, char *argv[MAX_NUM_ARGS]) {
    argc = 0;

    // Copy over the command line arguments (passed in)
    for (int i = 0; i < Argc; ++i)
        lAddSingleArg(Argv[i], argc, argv);

    // See if we have any set via the environment variable
    const char *env = getenv("ISPC_ARGS");
    if (env)
        lAddArgsFromString(env, argc, argv);
}

static void lSignal(void *) { FATAL("Unhandled signal sent to process; terminating."); }

static int ParsingPhaseName(char *stage) {
    if (strncmp(stage, "first", 5) == 0) {
        return 0;
    } else if (strncmp(stage, "last", 4) == 0) {
        return LAST_OPT_NUMBER;
    } else {
        int t = atoi(stage);
        if (t < 0 || t > LAST_OPT_NUMBER) {
            fprintf(stderr, "Phases must be from 0 to %d. %s is incorrect.\n", LAST_OPT_NUMBER, stage);
            exit(0);
        } else {
            return t;
        }
    }
}

static std::set<int> ParsingPhases(char *stages) {
    std::set<int> phases;
    auto len = strnlen(stages, 100);
    Assert(len && len < 100 && "phases string is too long!");
    int begin = ParsingPhaseName(stages);
    int end = begin;

    for (unsigned i = 0; i < strlen(stages); i++) {
        if ((stages[i] == ',') || (i == strlen(stages) - 1)) {
            for (int j = begin; j < end + 1; j++) {
                phases.insert(j);
            }
            begin = ParsingPhaseName(stages + i + 1);
            end = begin;
        } else if (stages[i] == ':') {
            end = ParsingPhaseName(stages + i + 1);
        }
    }
    return phases;
}

static void lParseInclude(const char *path) {
#ifdef ISPC_HOST_IS_WINDOWS
    char delim = ';';
#else
    char delim = ':';
#endif
    size_t pos = 0, pos_end;
    std::string str_path(path);
    do {
        pos_end = str_path.find(delim, pos);
        size_t len = (pos_end == std::string::npos) ?
                                                    // Not found, copy till end of the string.
                         std::string::npos
                                                    :
                                                    // Copy [pos, pos_end).
                         (pos_end - pos);
        std::string s = str_path.substr(pos, len);
        g->includePath.push_back(s);
        pos = pos_end + 1;
    } while (pos_end != std::string::npos);
}

int main(int Argc, char *Argv[]) {
    int argc;
    char *argv[MAX_NUM_ARGS];
    lGetAllArgs(Argc, Argv, argc, argv);

    llvm::sys::AddSignalHandler(lSignal, NULL);

    // initialize available LLVM targets
    // TO-DO : Revisit after experimenting on arm and aarch64 hardware.
#ifndef __arm__
    // FIXME: LLVM build on ARM doesn't build the x86 targets by default.
    // It's not clear that anyone's going to want to generate x86 from an
    // ARM host, though...
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();
    LLVMInitializeX86TargetMC();
#endif // !__ARM__

#ifdef ISPC_ARM_ENABLED
    // Generating ARM and AARCH64 from x86 is more likely to be useful, though.
    LLVMInitializeARMTargetInfo();
    LLVMInitializeARMTarget();
    LLVMInitializeARMAsmPrinter();
    LLVMInitializeARMAsmParser();
    LLVMInitializeARMDisassembler();
    LLVMInitializeARMTargetMC();

    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64AsmPrinter();
    LLVMInitializeAArch64AsmParser();
    LLVMInitializeAArch64Disassembler();
    LLVMInitializeAArch64TargetMC();
#endif

#ifdef ISPC_NVPTX_ENABLED
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXAsmPrinter();
    LLVMInitializeNVPTXTargetMC();
#endif /* ISPC_NVPTX_ENABLED */

    char *file = NULL;
    const char *headerFileName = NULL;
    const char *outFileName = NULL;
    const char *includeFileName = NULL;
    const char *depsFileName = NULL;
    const char *depsTargetName = NULL;
    const char *hostStubFileName = NULL;
    const char *devStubFileName = NULL;
    // Initiailize globals early so that we can set various option values
    // as we're parsing below
    g = new Globals;

    Module::OutputType ot = Module::Object;
    Module::OutputFlags flags = Module::NoFlags;
    const char *arch = NULL, *cpu = NULL, *target = NULL, *intelAsmSyntax = NULL;

    // Default settings for PS4
    if (g->target_os == TargetOS::OS_PS4) {
        flags |= Module::GeneratePIC;
        cpu = "btver2";
    }
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help"))
            usage(0);
        if (!strcmp(argv[i], "--help-dev"))
            devUsage(0);
        else if (!strncmp(argv[i], "-D", 2))
            g->cppArgs.push_back(argv[i]);
        else if (!strncmp(argv[i], "--addressing=", 13)) {
            if (atoi(argv[i] + 13) == 64)
                // FIXME: this doesn't make sense on 32 bit platform.
                g->opt.force32BitAddressing = false;
            else if (atoi(argv[i] + 13) == 32)
                g->opt.force32BitAddressing = true;
            else {
                fprintf(stderr,
                        "Addressing width \"%s\" invalid--only 32 and "
                        "64 are allowed.\n",
                        argv[i] + 13);
                usage(1);
            }
        } else if (!strncmp(argv[i], "--arch=", 7)) {
            // Do not allow to set arch for PS4 target, it is pre-defined.
            if (g->target_os != TargetOS::OS_PS4) {
                arch = argv[i] + 7;
                // Define arch alias
                // LLVM TargetRegistry uses "x86-64", while triple uses "x86_64".
                // We support both as input and internally keep it as "x86-64".
                if (std::string(arch) == "x86_64") {
                    arch = "x86-64";
                }
            }
        } else if (!strncmp(argv[i], "--x86-asm-syntax=", 17)) {
            intelAsmSyntax = argv[i] + 17;
            if (!((std::string(intelAsmSyntax) == "intel") || (std::string(intelAsmSyntax) == "att"))) {
                intelAsmSyntax = NULL;
                fprintf(stderr,
                        "Invalid value for --x86-asm-syntax: \"%s\" -- "
                        "only intel and att are allowed.\n",
                        argv[i] + 17);
            }
        } else if (!strncmp(argv[i], "--cpu=", 6)) {
            // Do not allow to set cpu for PS4 target, it is pre-defined.
            if (g->target_os != TargetOS::OS_PS4) {
                cpu = argv[i] + 6;
            }
        } else if (!strcmp(argv[i], "--fast-math")) {
            fprintf(stderr, "--fast-math option has been renamed to --opt=fast-math!\n");
            usage(1);
        } else if (!strcmp(argv[i], "--fast-masked-vload")) {
            fprintf(stderr, "--fast-masked-vload option has been renamed to "
                            "--opt=fast-masked-vload!\n");
            usage(1);
        } else if (!strcmp(argv[i], "--debug"))
            g->debugPrint = true;
        else if (!strcmp(argv[i], "--debug-llvm"))
            llvm::DebugFlag = true;
        else if (!strcmp(argv[i], "--dllexport"))
            g->dllExport = true;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5
        else if (!strncmp(argv[i], "--dwarf-version=", 16)) {
            int val = atoi(argv[i] + 16);
            if (2 <= val && val <= 4) {
                g->generateDebuggingSymbols = true;
                g->generateDWARFVersion = val;
            } else {
                fprintf(stderr,
                        "Invalid value for DWARF version: \"%s\" -- "
                        "only 2, 3 and 4 are allowed.\n",
                        argv[i] + 16);
                usage(1);
            }
        }
#endif
        else if (!strcmp(argv[i], "--print-target"))
            g->printTarget = true;
        else if (!strcmp(argv[i], "--no-omit-frame-pointer"))
            g->NoOmitFramePointer = true;
        else if (!strcmp(argv[i], "--instrument"))
            g->emitInstrumentation = true;
        else if (!strcmp(argv[i], "--no-pragma-once"))
            g->noPragmaOnce = true;
        else if (!strcmp(argv[i], "-g")) {
            g->generateDebuggingSymbols = true;
        } else if (!strcmp(argv[i], "--emit-asm"))
            ot = Module::Asm;
        else if (!strcmp(argv[i], "--emit-c++"))
            ot = Module::CXX;
        else if (!strcmp(argv[i], "--emit-llvm"))
            ot = Module::Bitcode;
        else if (!strcmp(argv[i], "--emit-llvm-text"))
            ot = Module::BitcodeText;
        else if (!strcmp(argv[i], "--emit-obj"))
            ot = Module::Object;
        else if (!strcmp(argv[i], "-I")) {
            if (++i == argc) {
                fprintf(stderr, "No path specified after -I option.\n");
                usage(1);
            }
            lParseInclude(argv[i]);
        } else if (!strncmp(argv[i], "-I", 2))
            lParseInclude(argv[i] + 2);
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
        } else if (!strncmp(argv[i], "--target=", 9)) {
            target = argv[i] + 9;
        } else if (!strncmp(argv[i], "--target-os=", 12)) {
            g->target_os = StringToOS(argv[i] + 12);
            if (g->target_os == OS_ERROR) {
                fprintf(stderr, "Unsupported value for --target-os, supported values are: %s\n",
                        Target::SupportedOSes());
                usage(1);
            }
        } else if (!strncmp(argv[i], "--math-lib=", 11)) {
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
        } else if (!strncmp(argv[i], "--opt=", 6)) {
            const char *opt = argv[i] + 6;
            if (!strcmp(opt, "fast-math"))
                g->opt.fastMath = true;
            else if (!strcmp(opt, "fast-masked-vload"))
                g->opt.fastMaskedVload = true;
            else if (!strcmp(opt, "disable-assertions"))
                g->opt.disableAsserts = true;
            else if (!strcmp(opt, "disable-loop-unroll"))
                g->opt.unrollLoops = false;
            else if (!strcmp(opt, "disable-fma"))
                g->opt.disableFMA = true;
            else if (!strcmp(opt, "force-aligned-memory"))
                g->opt.forceAlignedMemory = true;

            // These are only used for performance tests of specific
            // optimizations
            else if (!strcmp(opt, "disable-all-on-optimizations"))
                g->opt.disableMaskAllOnOptimizations = true;
            else if (!strcmp(opt, "disable-coalescing"))
                g->opt.disableCoalescing = true;
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
        } else if (!strncmp(argv[i], "--force-alignment=", 18)) {
            g->forceAlignment = atoi(argv[i] + 18);
        } else if (!strcmp(argv[i], "--woff") || !strcmp(argv[i], "-woff")) {
            g->disableWarnings = true;
            g->emitPerfWarnings = false;
        } else if (!strcmp(argv[i], "--werror"))
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
        } else if (!strncmp(argv[i], "--outfile=", 10))
            outFileName = argv[i] + strlen("--outfile=");
        else if (!strcmp(argv[i], "-h")) {
            if (++i == argc) {
                fprintf(stderr, "No header file name specified after -h option.\n");
                usage(1);
            }
            headerFileName = argv[i];
        } else if (!strncmp(argv[i], "--header-outfile=", 17)) {
            headerFileName = argv[i] + strlen("--header-outfile=");
        } else if (!strncmp(argv[i], "--c++-include-file=", 19)) {
            includeFileName = argv[i] + strlen("--c++-include-file=");
        } else if (!strcmp(argv[i], "-O0")) {
            g->opt.level = 0;
            g->codegenOptLevel = Globals::CodegenOptLevel::None;
        } else if (!strcmp(argv[i], "-O") || !strcmp(argv[i], "-O1") || !strcmp(argv[i], "-O2") ||
                   !strcmp(argv[i], "-O3")) {
            g->opt.level = 1;
            g->codegenOptLevel = Globals::CodegenOptLevel::Aggressive;
            if (!strcmp(argv[i], "-O1"))
                g->opt.disableCoherentControlFlow = true;
        } else if (!strcmp(argv[i], "-"))
            ;
        else if (!strcmp(argv[i], "--nostdlib"))
            g->includeStdlib = false;
        else if (!strcmp(argv[i], "--nocpp"))
            g->runCPP = false;
        else if (!strcmp(argv[i], "--pic"))
            flags |= Module::GeneratePIC;
#ifndef ISPC_IS_HOST_WINDOWS
        else if (!strcmp(argv[i], "--colored-output"))
            g->forceColoredOutput = true;
#endif // !ISPC_IS_HOST_WINDOWS
        else if (!strcmp(argv[i], "--quiet"))
            g->quiet = true;
        else if (!strcmp(argv[i], "--yydebug")) {
            extern int yydebug;
            yydebug = 1;
        } else if (!strcmp(argv[i], "-MMM")) {
            if (++i == argc) {
                fprintf(stderr, "No output file name specified after -MMM option.\n");
                usage(1);
            }
            depsFileName = argv[i];
            flags |= Module::GenerateFlatDeps;
        } else if (!strcmp(argv[i], "-M")) {
            flags |= Module::GenerateMakeRuleForDeps | Module::OutputDepsToStdout;
        } else if (!strcmp(argv[i], "-MF")) {
            depsFileName = nullptr;
            if (++i == argc) {
                fprintf(stderr, "No output file name specified after -MF option.\n");
                usage(1);
            }
            depsFileName = argv[i];
        } else if (!strcmp(argv[i], "-MT")) {
            depsTargetName = nullptr;
            if (++i == argc) {
                fprintf(stderr, "No target name specified after -MT option.\n");
                usage(1);
            }
            depsTargetName = argv[i];
        } else if (!strcmp(argv[i], "--dev-stub")) {
            if (++i == argc) {
                fprintf(stderr, "No output file name specified after --dev-stub option.\n");
                usage(1);
            }
            devStubFileName = argv[i];
        } else if (!strcmp(argv[i], "--host-stub")) {
            if (++i == argc) {
                fprintf(stderr, "No output file name specified after --host-stub option.\n");
                usage(1);
            }
            hostStubFileName = argv[i];
        }
#ifndef ISPC_NO_DUMPS
        else if (strncmp(argv[i], "--debug-phase=", 14) == 0) {
            fprintf(stderr, "WARNING: Adding debug phases may change the way PassManager"
                            "handles the phases and it may possibly make some bugs go"
                            "away or introduce the new ones.\n");
            g->debug_stages = ParsingPhases(argv[i] + strlen("--debug-phase="));
        } else if (strncmp(argv[i], "--dump-file", 11) == 0)
            g->dumpFile = true;
#endif

#if ISPC_LLVM_VERSION == ISPC_LLVM_3_4 || ISPC_LLVM_VERSION == ISPC_LLVM_3_5 // 3.4, 3.5
        else if (strncmp(argv[i], "--debug-ir=", 11) == 0) {
            g->debugIR = ParsingPhaseName(argv[i] + strlen("--debug-ir="));
        }
#endif
        else if (strncmp(argv[i], "--off-phase=", 12) == 0) {
            g->off_stages = ParsingPhases(argv[i] + strlen("--off-phase="));
        } else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--version")) {
            lPrintVersion();
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option \"%s\".\n", argv[i]);
            usage(1);
        } else {
            if (file != NULL) {
                fprintf(stderr,
                        "Multiple input files specified on command "
                        "line: \"%s\" and \"%s\".\n",
                        file, argv[i]);
                usage(1);
            } else
                file = argv[i];
        }
    }

    if (g->enableFuzzTest) {
        if (g->fuzzTestSeed == -1) {
#ifdef ISPC_HOST_IS_WINDOWS
            int seed = (unsigned)time(NULL);
#else
            int seed = getpid();
#endif
            g->fuzzTestSeed = seed;
            Warning(SourcePos(), "Using seed %d for fuzz testing", g->fuzzTestSeed);
        }
#ifdef ISPC_HOST_IS_WINDOWS
        srand(g->fuzzTestSeed);
#else
        srand48(g->fuzzTestSeed);
#endif
    }

    if (depsFileName != NULL)
        flags &= ~Module::OutputDepsToStdout;

    if (depsFileName != NULL && 0 == (flags & (Module::GenerateFlatDeps | Module::GenerateMakeRuleForDeps))) {
        Warning(SourcePos(), "Dependency file name specified with -MF, but no "
                             "mode specified; did you forget to specify -M or -MMM? "
                             "No dependency output will be generated.");
        depsFileName = NULL;
    }

    if ((Module::GenerateFlatDeps | Module::GenerateMakeRuleForDeps) ==
        (flags & (Module::GenerateFlatDeps | Module::GenerateMakeRuleForDeps))) {
        Warning(SourcePos(), "Both -M and -MMM specified on the command line. "
                             "-MMM takes precedence.");
        flags &= Module::GenerateMakeRuleForDeps;
    }

    if (outFileName == NULL && headerFileName == NULL &&
        (depsFileName == NULL && 0 == (flags & Module::OutputDepsToStdout)) && hostStubFileName == NULL &&
        devStubFileName == NULL)
        Warning(SourcePos(), "No output file or header file name specified. "
                             "Program will be compiled and warnings/errors will "
                             "be issued, but no output will be generated.");

    if (g->target_os == OS_WINDOWS && (flags & Module::GeneratePIC) != 0) {
        Warning(SourcePos(), "--pic switch for Windows target will be ignored.");
    }

    if (g->target_os != OS_WINDOWS && g->dllExport) {
        Warning(SourcePos(), "--dllexport switch will be ignored, as the target OS is not Windows.");
    }

    if ((ot == Module::Asm) && (intelAsmSyntax != NULL)) {
        std::vector<const char *> Args(3);
        Args[0] = "ispc (LLVM option parsing)";
        Args[2] = nullptr;
        if (std::string(intelAsmSyntax) == "intel")
            Args[1] = "--x86-asm-syntax=intel";
        else
            Args[1] = "--x86-asm-syntax=att";
        llvm::cl::ParseCommandLineOptions(2, Args.data());
    }

    return Module::CompileAndOutput(file, arch, cpu, target, flags, ot, outFileName, headerFileName, includeFileName,
                                    depsFileName, depsTargetName, hostStubFileName, devStubFileName);
}
